import os

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .UV_Encoders import UV_Encoder
from .UV_Aggregators import UV_Aggregator
from .Social_Encoders import Social_Encoder
from .Social_Aggregators import Social_Aggregator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GraphRecModel(nn.Module, ABC):

    def __init__(self,
                 num_users,
                 num_items,
                 num_ratings,
                 learning_rate,
                 embed_dim,
                 final_prop,
                 loss_function,
                 use_cuda,
                 history_u_lists,
                 history_ur_lists,
                 history_v_lists,
                 history_vr_lists,
                 social_adj_lists,
                 random_seed,
                 name="GraphRec",
                 **kwargs):
        super(GraphRecModel, self).__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.final_prop = final_prop

        self.u2e = nn.Embedding(num_users, embed_dim).to(self.device)
        self.v2e = nn.Embedding(num_items, embed_dim).to(self.device)
        self.r2e = nn.Embedding(num_ratings, embed_dim).to(self.device)

        # user feature
        # features: item * rating
        self.agg_u_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, embed_dim, cuda=self.device,
                                            uv=True)
        self.enc_u_history = UV_Encoder(self.u2e, embed_dim, history_u_lists, history_ur_lists,
                                         self.agg_u_history, cuda=self.device,uv=True)
        # neighobrs
        self.agg_u_social = Social_Aggregator(lambda nodes: self.enc_u_history(nodes).t(), self.u2e, embed_dim,
                                               cuda=self.device)
        self.enc_u = Social_Encoder(lambda nodes: self.enc_u_history(nodes).t(), embed_dim,
                                     social_adj_lists, self.agg_u_social,
                                     base_model=self.enc_u_history, cuda=self.device)

        # item feature: user * rating
        self.agg_v_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, embed_dim, cuda=self.device,uv=False)
        self.enc_v_history = UV_Encoder(self.v2e, embed_dim, history_v_lists, history_vr_lists,
                                         self.agg_v_history, cuda=self.device, uv=False)

        self.embed_dim = self.enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)

        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        if loss_function == 'mse':
            self.criterion = nn.MSELoss()
        if loss_function == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def propagate_user_embeddings(self, nodes_u):
        embeds_u = self.enc_u(nodes_u)
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        return x_u

    def propagate_item_embeddings(self, nodes_v):
        embeds_v = self.enc_v_history(nodes_v)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        return x_v

    def propagate_embeddings(self, nodes_u, nodes_v):
        x_u = self.propagate_user_embeddings(nodes_u)
        x_v = self.propagate_item_embeddings(nodes_v)
        return x_u, x_v

    def cat_and_propagate_user_item_embeddings(self, x_u, x_v):
        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def forward(self, nodes_u, nodes_v):
        x_u, x_v = self.propagate_embeddings(nodes_u, nodes_v)
        if self.final_prop is True:
            return self.cat_and_propagate_user_item_embeddings(x_u, x_v)
        else:
            scores = torch.diagonal(torch.matmul(x_u, x_v.t()))
            return scores

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)
    
    def get_top_k(self, preds, train_mask, device, k=100):
        if self.final_prop == True:
            return torch.topk(torch.where(torch.tensor(train_mask).to(device), torch.tensor(preds, dtype=torch.float32).to(device),
                                          torch.tensor(-np.inf, dtype=torch.float32).to(device)), k=k, sorted=True)
        if self.final_prop == False:
            return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                          torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)