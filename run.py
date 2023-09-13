import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import set_seed, collate_fn, AverageMeter, accuracy
from datasets import load_metric
from model import BertForSequenceClassification
import warnings
from data import load
from train_rec import train_rec
import sys

warnings.filterwarnings("ignore")


def train(args, model, train_dataset, dev_dataset, test_dataset, ood_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True,
                                  drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.val_batch_size, collate_fn=collate_fn, shuffle=True,
                                drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size*5, collate_fn=collate_fn, drop_last=False)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=total_steps)
    #
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=total_steps)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=0,
                                                                   # Default value in run_glue.py
                                                                   num_training_steps=total_steps, num_cycles=6)

    ##stage I:  trian the reconstructor
    if args.train_rec:
        rector = train_rec(args, model, train_dataloader, dev_dataloader)
        model.insampler = rector
        print("insampler done")

    ## Stage II: train the detector
    best_eval = -float('inf')
    patient = 0
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    id_acc_avg = AverageMeter()

    num_steps = 0
    final_res = {}

    # eval_loss_avg.reset()
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        model.train()
        loss_avg.reset()
        acc_avg.reset()
        id_acc_avg.reset()
        epoch += 1
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            # if args.train_rec:
            #     batch["insampler"] = rector
            batch["epoch"] = epoch
            batch["mode"] = "train"
            outputs = model(**batch)
            loss, logits = outputs[0], outputs[1]
            labels = outputs[2]
            acc = accuracy(logits, labels)
            id_acc = accuracy(logits[:args.batch_size], labels[:args.batch_size])
            loss['loss'].backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            # print(labels)
            loss_avg.update(loss['loss'].item(), len(batch['labels']))
            acc_avg.update(acc.item(), len(batch['labels']))
            id_acc_avg.update(id_acc.item(), args.batch_size)

        print('\n')
        print("-" * 15)
        print("train loss of epoch:", epoch, loss_avg.avg)
        print("train acc:", acc_avg.avg, "id acc:", id_acc_avg.avg)
        # print("train id acc:", id_acc_avg.avg)
        dev_acc, dev_f1 = evaluate(args, model, dev_dataloader, tag="dev")
        # print("dev result:", dev_res)
        print("dev result acc and f1 scroe:", dev_acc, dev_f1)
        print("-" * 15)
        sys.stdout.flush()
        # print(results)

        loss_avg.reset()
        # eval_loss_avg.reset()
        if dev_f1 > best_eval:
            print("performing testing")
            test_acc, f1_score, id_f1 = evaluate(args, model, test_dataloader, tag="test")
            print("test result acc and f1 scroe:", test_acc, f1_score)

            # ood_res = detect_ood()
            best_eval = dev_f1
            # final_res = dict(ood_res, **{"test_acc": results['test_accuracy'], 'eval_acc': best_eval})
            final_res = dict(
                {"test_acc": test_acc, 'eval_f1': best_eval, "f1": f1_score, "eval_acc": dev_acc, "in_f1": id_f1})
            patient = 0
        else:
            patient += 1

        if (patient > 20 and epoch > 30) or epoch == args.num_train_epochs:
            save_results(args, final_res)
            break
    # save_results(args, final_res)


from sklearn.metrics import f1_score


def evaluate(args, model, eval_dataset, id_dataset=None, tag="dev"):
    model.eval()

    if tag == "dev":
        acc_avg = AverageMeter()
        labels_all = []
        preds_all = []
        with torch.no_grad():
            for step, batch in enumerate(eval_dataset):
                batch = {key: value.to(args.device) for key, value in batch.items()}
                batch["mode"] = tag
                outputs = model(**batch)
                logits = outputs[1]
                labels = outputs[2]
                acc = accuracy(logits, labels)
                acc_avg.update(acc.item(), len(labels))
                _, pred = logits.topk(1, 1, True, True)
                labels_all.append(labels.cpu().numpy())
                preds_all.append(pred.cpu().numpy())
            f1_scores = f1_score(np.concatenate(labels_all, axis=0), np.concatenate(preds_all, axis=0), average=None)
            return acc_avg.avg, np.mean(f1_scores)

    if tag == "test":
        acc_avg = AverageMeter()
        # ood_acc_avg = AverageMeter()
        labels_all = []
        preds_all = []
        id_labels_all = []
        id_preds_all = []
        with torch.no_grad():
            for step, batch in enumerate(eval_dataset):
                batch = {key: value.to(args.device) for key, value in batch.items()}
                batch["mode"] = tag
                outputs = model(**batch)
                logits = outputs[1]
                labels = batch["labels"]
                acc = accuracy(logits, labels)
                acc_avg.update(acc.item(), len(batch['labels']))

                _, pred = logits.topk(1, 1, True, True)
                labels_all.append(labels.cpu().numpy())
                preds_all.append(pred.cpu().numpy())
            f1_scores = f1_score(np.concatenate(labels_all, axis=0), np.concatenate(preds_all, axis=0), average=None)

            return acc_avg.avg, np.mean(f1_scores), np.mean(f1_scores[:-1])


import os
import pandas as pd


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.task_name, args.seed, args.rec_drop, args.train_rec, args.shot, args.convex, args.known_ratio]
    names = ['dataset', 'seed', 'rec_drop', "is_rec", 'shot', "convex", "knonw_ratio"]
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'results_rec.csv'
    results_path = os.path.join(args.save_results_path, file_name)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = df1.append(new, ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print('test_results')
    print(data_diagram)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="roberta-large;bert-base-uncased")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--task_name", default="clinc150", type=str)
    parser.add_argument("--data_dir", default="path_to_your_dataset", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=200.0, type=float)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--project_name", type=str, default="ood-fewshot")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--shot", type=float, default=0.05)
    parser.add_argument("--train_rec", action='store_true', help="mix reconstruction id samples")
    parser.add_argument("--convex", action='store_true', help="K+1 head")
    parser.add_argument("--freeze", action='store_true', help="freeze the model")
    parser.add_argument("--rec_num", type=int, default=10)
    parser.add_argument("--rec_drop", type=float, default=0.3)
    parser.add_argument("--known_ratio", type=float, default=0.75)
    parser.add_argument("--save_results_path", type=str, default='/data/liming/few-shot-nlp',
                        help="the path to save results")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset, dev_dataset, test_dataset, ood_dataset, num_labels = load(args.task_name, tokenizer, shot=args.shot,
                                                                             max_seq_length=args.max_seq_length,
                                                                             is_id=True,
                                                                             dir=args.data_dir,
                                                                             known_cls_ratio=args.known_ratio)

    if args.model_name_or_path.startswith('bert'):
        config = BertConfig.from_pretrained(args.model_name_or_path)
        # config.gradient_checkpointing = True
        config.freeze = args.freeze
        config.train_rec = args.train_rec
        config.rec_num = args.rec_num
        config.convex = args.convex
        config.num_labels = num_labels

        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)

    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k', 'clinc150']
    print("data size", len(train_dataset), len(dev_dataset), len(test_dataset), num_labels)
    train(args, model, train_dataset, dev_dataset, test_dataset, ood_dataset)


if __name__ == "__main__":
    main()
