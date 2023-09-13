from utils import set_seed, collate_fn, AverageMeter, accuracy
from model import Dreconstruction
import torch
from tqdm import tqdm
import numpy as np
import sys


def train_rec(args, model, train_dataloader, dev_dataloader):
    # num_steps_pre = 0
    # num_steps_rec = 0
    loss_avg = AverageMeter()
    eval_loss_avg = AverageMeter()

    loss_avg.reset()
    eval_loss_avg.reset()

    rector = Dreconstruction(model.config.hidden_size, args.rec_num, args.rec_drop)
    rector.cuda()
    optimizer_mlp = torch.optim.Adam(rector.parameters(), lr=1e-4)
    # schedulor_rec = torch.optim.lr_scheduler.ExponentialLR(optimizer_mlp, 0.7)
    rector.rec = False
    best_eval = float('inf')
    patient = 0
    model.eval()

    for epoch in range(int(1000)):

        rector.train()
        rector.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            # batch = {key: value.cuda() for key, value in batch.items()}
            # labels = batch['labels']
            outputs = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            pooled = outputs[0].mean(dim=1)
            loss = rector(pooled) * 10000
            loss.backward()
            loss_avg.update(loss.item(), n=len(batch['labels']))
            optimizer_mlp.step()
            # scheduler.step()
            rector.zero_grad()
            # eval
            # schedulor_rec.step()
        print("\n")
        print("_" * 15)
        print("mse loss of epoch", epoch + 1, loss_avg.avg)

        for step, batch in enumerate(dev_dataloader):
            rector.eval()
            # batch = {key: value.to(args.device) for key, value in batch.items()}
            batch = {key: value.cuda() for key, value in batch.items()}
            # labels = batch['labels']
            outputs = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            pooled = outputs[0].mean(dim=1)
            val_loss = rector(pooled) * 10000
            eval_loss_avg.update(val_loss.item(), n=len(batch['labels']))
        # print("\n")
        print("val rec loss:", eval_loss_avg.avg)
        sys.stdout.flush()
        if eval_loss_avg.avg < best_eval:
            best_eval = eval_loss_avg.avg
            patient = 0
        else:
            patient += 1

        loss_avg.reset()
        eval_loss_avg.reset()
        if patient > 3:
            break

    return rector
