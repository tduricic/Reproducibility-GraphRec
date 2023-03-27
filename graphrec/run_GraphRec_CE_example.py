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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
from BPRData import BPRData
from scipy.sparse import csr_matrix

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

    def __init__(self, enc_u, enc_v_history, r2e, final_prop=True):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.final_prop = final_prop

        print(self.final_prop)

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
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCEWithLogitsLoss()

    def get_and_propagate_user_embeddings(self, nodes_u):
        embeds_u = self.enc_u(nodes_u)
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        return x_u

    def get_and_propagate_item_embeddings(self, nodes_v):
        embeds_v = self.enc_v_history(nodes_v)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        return x_v

    def forward(self, nodes_u, nodes_v):
        x_u = self.get_and_propagate_user_embeddings(nodes_u)
        x_v = self.get_and_propagate_item_embeddings(nodes_v)

        if self.final_prop is True:
            x_uv = torch.cat((x_u, x_v), 1)
            x = F.relu(self.bn3(self.w_uv1(x_uv)))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.bn4(self.w_uv2(x)))
            x = F.dropout(x, training=self.training)
            scores = self.w_uv3(x)
            return scores.squeeze()
        else:
            scores = torch.diagonal(torch.matmul(x_u, x_v.t()))
            return scores

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_valid_rmse, best_valid_mae, best_test_rmse, best_test_mae):
    model.train()
    print("start negative sample...")
    train_loader.dataset.ng_sample()
    print("finish negative sample...")
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        u, pos, neg = data
        u = u.long().to(device)
        pos = pos.long().to(device)
        neg = neg.long().to(device)

        batch_nodes_u = torch.hstack((u, u))
        batch_nodes_v = torch.hstack((pos, neg))
        labels_list = torch.hstack((torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])))

        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best validation rmse/mae: %.6f / %.6f, the best test rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_valid_rmse, best_valid_mae, best_test_rmse, best_test_mae))
            running_loss = 0.0
    return 0

def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def create_csr_matrix_from_user_item_ratings(u, v, r, num_users, num_items):
    return csr_matrix((r, (u, v)), shape=(num_users, num_items))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='Load from checkpoint or not')
    parser.add_argument('--dataset', type=str, default='toy_dataset')
    parser.add_argument('--percentage', type=str, default='_sixty')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--final_prop', type=str, default="yes")
    parser.add_argument('--num_ng', type=int, default=1)
    args = parser.parse_args()

    print(os.getcwd())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    path_data = './data/' + args.dataset + '/' + args.dataset + '.pkl'
    # path_data = './data/' + args.dataset + '/' + args.dataset + args.percentage + '.pkl'
    data_file = open(path_data, 'rb')

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, \
    social_adj_lists, item_adj_lists, ratings_list = pickle.load(data_file)

    train_data = np.array(traindata)
    valid_data = np.array(validdata)
    test_data = np.array(testdata)

    train_u = train_data[:, 0]
    train_v = train_data[:, 1]
    train_r = train_data[:, 2]

    valid_u = valid_data[:, 0]
    valid_v = valid_data[:, 1]
    valid_r = valid_data[:, 2]

    test_u = test_data[:, 0]
    test_v = test_data[:, 1]
    test_r = test_data[:, 2]
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """
    num_users = max(set(train_u.tolist() + valid_u.tolist() + test_u.tolist())) + 1
    num_items = max(set(train_v.tolist() + valid_v.tolist() + test_v.tolist())) + 1
    trainMat = create_csr_matrix_from_user_item_ratings(train_u, train_v, train_r, num_users, num_items)

    trainset = BPRData(train_data, num_items, trainMat, args.num_ng, True)
    validset = BPRData(valid_data, num_items, trainMat, 0, False)
    testset = BPRData(test_data, num_items, trainMat, 0, False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    # num_users = history_u_lists.__len__()
    # num_items = history_v_lists.__len__()

    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    if args.final_prop == "yes":
        final_prop = True
    if args.final_prop == "no":
        final_prop = False

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e, final_prop).to(device)
    print(graphrec.final_prop)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    # load from checkpoint
    if args.load_from_checkpoint == True:
        checkpoint = torch.load('models/' + args.dataset + '.pt')
        graphrec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_valid_rmse = 9999.0
    best_valid_mae = 9999.0
    best_test_rmse = 9999.0
    best_test_mae= 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_valid_rmse, best_valid_mae, best_test_rmse, best_test_mae)
        valid_rmse, valid_mae = test(graphrec, device, valid_loader)

        # early stopping (no validation set in toy dataset)
        if best_valid_rmse > valid_rmse:
            best_valid_rmse = valid_rmse
            best_valid_mae = valid_mae
            endure_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': graphrec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.dataset + args.percentage + '.pt')
        else:
            endure_count += 1
        print("RMSE on valid set: %.4f, MAE:%.4f " % (valid_rmse, valid_mae))
        test_rmse, test_mae = test(graphrec, device, test_loader)
        if best_test_rmse > test_rmse:
            best_test_rmse = test_rmse
            best_test_mae = test_mae
        print("RMSE on test set: %.4f, MAE:%.4f " % (test_rmse, test_mae))

        if endure_count > 5:
            break
    print('finished')


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Duration in seconds: " + str(end - start))