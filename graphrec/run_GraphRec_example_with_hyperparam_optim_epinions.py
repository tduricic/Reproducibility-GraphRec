import os

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from config_epinions import (
    MAX_NUM_EPOCHS, GRACE_PERIOD, EPOCHS, CPU, GPU, NUM_SAMPLES, DATA_ROOT_DIR, NUM_WORKERS, TEST_BATCH_SIZE,
    REDUCTION_FACTOR, MAX_FAILURES, LOCAL_DIR, DATASET_NAME
)

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(config, checkpoint_dir=None, data_dir=None):
    trainset, validset, _ = load_data(data_dir)

    graphrec, device = create_model(config, data_dir)

    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=config["lr"], alpha=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        graphrec.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=int(TEST_BATCH_SIZE), shuffle=True, num_workers=NUM_WORKERS)


    for epoch in range(1, EPOCHS + 1):
        graphrec.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_nodes_u, batch_nodes_v, labels_list = data
            batch_nodes_u, batch_nodes_v, labels_list = batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = graphrec.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
            # TODO check why is this different
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                running_loss = 0.0

        validation_loss, validation_rmse, validation_mae = test(graphrec, device, val_loader)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((graphrec.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=validation_loss, rmse=validation_rmse, mae=validation_mae)

    print("Finished Training!")


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            counter += 1
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
            loss = model.loss(test_u, test_v, tmp_target)
            running_loss += loss.item()
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    loss = running_loss / counter
    rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return loss, rmse, mae


def create_model(config, data_dir):
    path_data = data_dir + '/' + DATASET_NAME + ".pkl"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, \
    social_adj_lists, item_adj_lists, ratings_list = pickle.load(data_file)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    u2e = nn.Embedding(num_users, config["embed_dim"]).to(device)
    v2e = nn.Embedding(num_items, config["embed_dim"]).to(device)
    r2e = nn.Embedding(num_ratings, config["embed_dim"]).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, config["embed_dim"], cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, config["embed_dim"], history_u_lists, history_ur_lists, agg_u_history, cuda=device,
                               uv=True)
    # neighbors
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, config["embed_dim"], cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), config["embed_dim"], social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, config["embed_dim"], cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, config["embed_dim"], history_v_lists, history_vr_lists, agg_v_history, cuda=device,
                               uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)

    return graphrec, device

# def load_data(data_dir):
#    path_data = data_dir + '/' + DATASET_NAME + ".pkl"
#    data_file = open(path_data, 'rb')
#    _, _, _, _, train_u, train_v, train_r, test_u, test_v, test_r, _, _ = pickle.load(data_file)
#    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
#                                              torch.FloatTensor(train_r))
#    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
#                                             torch.FloatTensor(test_r))
#    return trainset, testset


def load_data(data_dir):
    path_data = data_dir + '/' + DATASET_NAME + ".pkl"
    data_file = open(path_data, 'rb')
    _, _, _, _, traindata, validdata, testdata, _, _, _ = pickle.load(data_file)

    traindata = np.array(traindata)
    validdata = np.array(validdata)
    testdata = np.array(testdata)

    train_u = traindata[:, 0]
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]

    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]

    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))

    return trainset, validset, testset


def main():
    _, _, testset = load_data(DATA_ROOT_DIR)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True)

    config = {
        "batch_size": tune.choice([32, 64, 128, 512]),
        "embed_dim": tune.choice([8, 16, 32, 64, 128, 256]),
        "lr": tune.choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    }

    scheduler = ASHAScheduler(
        metric="rmse",
        mode="min",
        max_t=MAX_NUM_EPOCHS,
        grace_period=GRACE_PERIOD,
        reduction_factor=REDUCTION_FACTOR)

    reporter = CLIReporter(
        metric_columns=["loss", "rmse", "mae", "training_iteration"])

    result = tune.run(
        partial(train, data_dir=DATA_ROOT_DIR),
        resources_per_trial={"cpu": CPU, "gpu": GPU},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter,
        raise_on_failed_trial=False,
        max_failures=MAX_FAILURES,
        local_dir=LOCAL_DIR)

    best_trial = result.get_best_trial("rmse", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation rmse: {}".format(best_trial.last_result["rmse"]))
    print("Best trial final validation mae: {}".format(best_trial.last_result["mae"]))

    best_trained_model, device = create_model(config, DATA_ROOT_DIR)
    best_checkpoint_dir = best_trial.checkpoint.value

    print(best_checkpoint_dir)

    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_rmse, test_mae = test(best_trained_model, device, test_loader)
    print("Best trial test set rmse and mae: {0}, {1}".format(test_rmse, test_mae))


if __name__ == "__main__":
    print(os.getcwd())
    main()
