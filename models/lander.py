#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dgl
import dgl.function as fn
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss
from .graphconv import GraphConv


class LANDER(nn.Module):
    def __init__(
        self,
        feature_dim,
        nhid,
        num_conv=4,
        dropout=0,
        use_GAT=True,
        K=1,
        balance=False,
        use_cluster_feat=True,
        use_focal_loss=True,
        **kwargs
    ):
        super(LANDER, self).__init__()
        nhid_half = int(nhid / 2)
        classifier_dim = 8
        self.use_cluster_feat = use_cluster_feat
        self.use_focal_loss = use_focal_loss

        if self.use_cluster_feat:
            self.feature_dim = feature_dim * 2
        else:
            self.feature_dim = feature_dim

        input_dim = (feature_dim, nhid, nhid, nhid_half)
        output_dim = (nhid, nhid, nhid_half, nhid_half)
        self.conv = nn.ModuleList()
        self.conv.append(GraphConv(self.feature_dim, nhid, dropout, use_GAT, K))
        for i in range(1, num_conv):
            self.conv.append(
                GraphConv(input_dim[i], output_dim[i], dropout, use_GAT, K)
            )

        self.src_mlp = nn.Linear(output_dim[num_conv - 1], classifier_dim - 2)
        self.dst_mlp = nn.Linear(output_dim[num_conv - 1], classifier_dim - 2)

        self.classifier_conn = nn.Sequential(
            nn.PReLU(2 * classifier_dim),
            nn.Linear(2 * classifier_dim, classifier_dim),
            nn.PReLU(classifier_dim),
            nn.Linear(classifier_dim, 1)
        )

        if self.use_focal_loss:
            self.loss_conn = FocalLoss(2)
        else:
            self.loss_conn = nn.BCEWithLogitsLoss()
        self.loss_den = nn.MSELoss()

        self.balance = balance

    def pred_conn(self, edges):
        src_feat = self.src_mlp(edges.src["conv_features"])
        dst_feat = self.dst_mlp(edges.dst["conv_features"])
        # coo_dist = torch.norm(torch.cat([edges.src['xws'] - edges.dst['xws'],
        #                                  edges.src['yws'] - edges.dst['yws']],
        #                                  dim=-1), dim=-1, keepdim=True)
        # feat_cat = torch.cat((src_feat,
        #                       dst_feat), dim=1)
        feat_cat = torch.cat((src_feat,
                            edges.src['xws'],
                            edges.src['yws'],
                            dst_feat,
                            edges.dst['xws'],
                            edges.dst['yws']), dim=1)

        pred_conn = self.classifier_conn(feat_cat)
        return {"pred_conn": pred_conn}

    def pred_den_msg(self, edges):
        prob = edges.data["prob_conn"][:, 0]
        res = edges.data["raw_affine"] * (prob - (1. - prob))
        return {"pred_den_msg": res}
    
    # def sup_con_loss_msg(self, msgs):
    #     logits = msgs["pred_conn"]
    #     mask = msgs["labels"].view(-1, 1)

    #     logits_max, _ = torch.max(logits).detach()
    #     logits = logits - logits_max
    #     exp_logits = torch.exp(logits)
    #     log_prob = logits - torch.log(exp_logits.sum(0, keep_dim=True))
    #     mean_log_prob = 
    #     return {"sup_con_loss_msg": res}

    def forward(self, bipartite):
        if self.use_cluster_feat:
            neighbor_x = torch.cat(
                [
                    bipartite.ndata["features"],
                    bipartite.ndata["cluster_features"],
                ],
                axis=1,
            )
        else:
            neighbor_x = bipartite.ndata["features"]

        for i in range(len(self.conv)):
            neighbor_x = self.conv[i](bipartite, neighbor_x)

        bipartite.ndata["conv_features"] = neighbor_x

        bipartite.apply_edges(self.pred_conn)
        bipartite.edata["prob_conn"] = 1. / (1. + torch.exp(-bipartite.edata["pred_conn"]))

        rev_bipartite = dgl.reverse(bipartite, copy_edata=True)
        rev_bipartite.update_all(
            self.pred_den_msg, fn.mean("pred_den_msg", "pred_den")
        )
        bipartite = dgl.reverse(rev_bipartite, copy_edata=True)
        return bipartite

    def compute_loss(self, bipartite):
        pred_den = bipartite.srcdata["pred_den"]
        loss_den = self.loss_den(pred_den, bipartite.srcdata["density"])

        labels_conn = bipartite.edata["labels_conn"]
        mask_conn = bipartite.edata["mask_conn"]

        if self.balance:
            labels_conn = bipartite.edata["labels_conn"]
            neg_check = torch.logical_and(
                bipartite.edata["labels_conn"] == 0, mask_conn
            )
            num_neg = torch.sum(neg_check).item()
            neg_indices = torch.where(neg_check)[0]
            pos_check = torch.logical_and(
                bipartite.edata["labels_conn"] == 1, mask_conn
            )
            num_pos = torch.sum(pos_check).item()
            pos_indices = torch.where(pos_check)[0]
            if num_pos > num_neg:
                mask_conn[
                    pos_indices[
                        np.random.choice(
                            num_pos, num_pos - num_neg, replace=False
                        )
                    ]
                ] = 0
            elif num_pos < num_neg:
                mask_conn[
                    neg_indices[
                        np.random.choice(
                            num_neg, num_neg - num_pos, replace=False
                        )
                    ]
                ] = 0

        # In subgraph training, it may happen that all edges are masked in a batch
        if mask_conn.sum() > 0:
            loss_conn = self.loss_conn(
                bipartite.edata["pred_conn"][mask_conn], labels_conn[mask_conn].view(-1, 1).float()
            )
            loss = loss_den + loss_conn
            loss_den_val = loss_den.item()
            loss_conn_val = loss_conn.item()
        else:
            loss = loss_den
            loss_den_val = loss_den.item()
            loss_conn_val = 0

        return loss, loss_den_val, loss_conn_val
    '''
    def compute_loss_new(self, bipartite):
        labels_conn = bipartite.edata["labels_conn"]
        mask_conn = bipartite.edata["mask_conn"]

        # In subgraph training, it may happen that all edges are masked in a batch
        print(f'\n\n')
        src, dst = bipartite.edges()
        src_nodes = torch.unique(src)
        loss_contrastive = 0.
        for node_id in src_nodes:
            idcs = src == node_id
            logits = bipartite.edata['pred_conn'][idcs]
            labels = bipartite.edata['labels_conn'][idcs]
            print(f"edata: {logits}")
            print(f"labels: {labels}")

            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            exp_logits = torch.exp(logits)
            
            print(f'\n')
        if mask_conn.sum() > 0:
            loss_conn = self.loss_conn(
                bipartite.edata["prob_conn"][mask_conn], labels_conn[mask_conn].view(-1, 1).float()
            )
            loss = loss_contrastive + loss_conn
            loss_contrastive_val = loss_contrastive.item()
            loss_conn_val = loss_conn.item()
        else:
            loss = loss_contrastive
            loss_contrastive_val = loss_contrastive.item()
            loss_conn_val = 0

        return loss, loss_contrastive_val, loss_conn_val
    '''