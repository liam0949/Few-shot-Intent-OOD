import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, RobertaModel, BertModel
from sklearn.covariance import EmpiricalCovariance


class ConvexSampler(nn.Module):
    def __init__(self, oos_label_id):
        super(ConvexSampler, self).__init__()
        # self.num_convex = round(args.n_oos/5)
        self.num_convex = 50
        self.num_convex_val = 50
        self.oos_label_id = oos_label_id

    def forward(self, z, label_ids, mode=None):
        convex_list = []
        # print(z)
        # print(label_ids)
        if mode == 'train':
            if torch.unique(label_ids).size(0) > 3 and self.num_convex!=0:
                while len(convex_list) < self.num_convex:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    # print(cdt)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(self.num_convex, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.oos_label_id] * self.num_convex).cuda()), dim=0)
        elif mode == 'dev':
            if torch.unique(label_ids).size(0) > 3 and self.num_convex_val!=0:
                val_num = self.num_convex_val
                while len(convex_list) < val_num:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    # print(cdt)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(val_num, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.oos_label_id] * val_num).cuda()), dim=0)
        # print(z)
        # print(label_ids)
        return z, label_ids


class Dreconstruction(nn.Module):
    def __init__(self, in_dim, num_rec, dropout=0.3):
        super(Dreconstruction, self).__init__()
        feat_dim = in_dim * 2
        self.recons = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, in_dim)
        )
        self.num_rec = num_rec
        self.rec = True

    def forward(self, emb, label_ids=None):
        if self.rec:
            emb_enlarged = emb.repeat(self.num_rec, 1)
            recset = self.recons(emb_enlarged)

            recset = torch.cat((emb, recset), dim=0)
            label_ids = label_ids.repeat(self.num_rec + 1)

            indices = torch.randperm(recset.size()[0])
            recset = recset[indices]
            label_ids = label_ids[indices]
            # print("rec done")
            return recset, label_ids
        recs = self.recons(emb)
        return F.mse_loss(recs, emb)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        # self.rec_drop = config.rec_drop
        self.rec_num = config.rec_num
        self.train_rec = config.train_rec
        self.insampler = None
        self.convex = config.convex
        self.sampler = ConvexSampler(config.num_labels)
        # if config.freeze:
        #     for name, param in self.bert.named_parameters():
        #         param.requires_grad = False
        #         if "encoder.layer.11" in name or "pooler" in name:
        #             # print("set last layer trainable")
        #             param.requires_grad = True

        if self.convex:
            self.classifier = nn.Sequential(
                # nn.Linear(in_dim, in_dim),
                # nn.ReLU()
                # nn.Linear(in_dim, in_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                # nn.Linear(feat_dim, n_way)
                nn.Linear(config.hidden_size, self.num_labels + 1)
            )
        else:
            self.classifier = nn.Sequential(
                # nn.Linear(in_dim, in_dim),
                # nn.ReLU()
                # nn.Linear(in_dim, in_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                # nn.Linear(feat_dim, n_way)
                nn.Linear(config.hidden_size, self.num_labels)
            )
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            epoch=1,
            mode=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        # pooled_output = pooled = outputs[1]
        pooled_output = pooled = outputs[0].mean(dim=1)
        if mode == "train" and self.train_rec and labels is not None:
            self.insampler.rec = True
            pooled_output, labels = self.insampler(pooled_output, label_ids=labels)
            self.insampler.rec = False
        if self.convex and labels is not None:
            pooled_output, labels = self.sampler(pooled_output, labels, mode=mode)
            # pooled_output = self.dropout(pooled_output)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = {}
        if labels is not None and self.training:
            loss_fct = CrossEntropyLoss()
            if self.convex:
                loss_all = loss_fct(logits.view(-1, self.num_labels + 1), labels.view(-1))
            else:
                loss_all = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss["loss"] = loss_all
            # loss = loss + self.config.alpha * cos_loss
            # loss = loss

        # return ((loss,) + output)
        return loss, logits, labels, pooled
