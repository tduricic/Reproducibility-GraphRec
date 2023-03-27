import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
from scipy.sparse import csr_matrix
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
import evaluate
import gc
torch.autograd.set_detect_anomaly(True)

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

    def __init__(self,
                 num_users,
                 num_items,
                 num_ratings,
                 embed_dim,
                 history_u_lists,
                 history_ur_lists,
                 history_v_lists,
                 history_vr_lists,
                 social_adj_lists,
                 device):
        super(GraphRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self.embed_dim = embed_dim

        self.u2e = nn.Embedding(num_users, self.embed_dim).to(device)
        self.v2e = nn.Embedding(num_items, self.embed_dim).to(device)
        self.r2e = nn.Embedding(num_ratings, self.embed_dim).to(device)

        # user feature
        # features: item * rating
        self.agg_u_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, self.embed_dim, cuda=device, uv=True)
        self.enc_u_history = UV_Encoder(self.u2e, self.embed_dim, history_u_lists, history_ur_lists, self.agg_u_history, cuda=device, uv=True)
        # neighobrs
        self.agg_u_social = Social_Aggregator(lambda nodes: self.enc_u_history(nodes).t(), self.u2e, self.embed_dim, cuda=device)
        self.enc_u = Social_Encoder(lambda nodes: self.enc_u_history(nodes).t(), self.embed_dim, social_adj_lists, self.agg_u_social, base_model=self.enc_u_history, cuda=device)

        # item feature: user * rating
        self.agg_v_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, self.embed_dim, cuda=device, uv=False)
        self.enc_v_history = UV_Encoder(self.v2e, self.embed_dim, history_v_lists, history_vr_lists, self.agg_v_history, cuda=device, uv=False)

        self.enc_u = self.enc_u
        self.enc_v_history = self.enc_v_history
        self.embed_dim = self.enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.r2e = self.r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

    def get_user_embeddings(self, nodes_u):
        embeds_u = self.enc_u(nodes_u)
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        return x_u

    def get_item_embeddings(self, nodes_v):
        embeds_v = self.enc_v_history(nodes_v)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        return x_v

    def forward(self, nodes_u, nodes_pos, nodes_neg):
        x_u = self.get_user_embeddings(nodes_u)
        x_i =  self.get_item_embeddings(nodes_pos)
        x_j = self.get_item_embeddings(nodes_neg)
        x_ui = torch.mul(x_u, x_i).sum(dim=1)
        x_uj = torch.mul(x_u, x_j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = 0.0001 * (x_u.norm(dim=1).pow(2).sum() + x_i.norm(dim=1).pow(2).sum() + x_j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization


def train(model, device, train_loader, optimizer, epoch, reg,  best_valid_HR, best_valid_NDCG, best_test_HR, best_test_NDCG):
    model.train()
    print("start negative sample...")
    train_loader.dataset.ng_sample()
    print("finish negative sample...")
    epoch_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        u, pos, neg = data
        u = u.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()
        loss = model.forward(u, pos, neg)
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
        running_loss += loss.item()

        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best validation HR/NDCG: %.6f / %.6f, The best test HR/NDCG: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_valid_HR, best_valid_NDCG, best_test_HR, best_test_NDCG))
            running_loss = 0.0

    return epoch_loss

    # for i, data in enumerate(train_loader, 0):
    #     batch_nodes_u, batch_nodes_v, labels_list = data
    #     optimizer.zero_grad()
    #     loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     running_loss += loss.item()
    #     if i % 100 == 0:
    #         print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
    #             epoch, i, running_loss / 100, best_rmse, best_mae))
    #         running_loss = 0.0
    # return 0


def test(model, device, test_loader, top_k):
    model.eval()
    HR, NDCG = [], []

    for user, item_i in test_loader:
        user = user.to(device)
        item_i = item_i.to(device)

        userEmbed = model.get_user_embeddings(user)
        testItemEmbed = model.get_item_embeddings(item_i)
        pred_i = torch.sum(torch.mul(userEmbed, testItemEmbed), dim=1)

        batch = int(user.cpu().numpy().size / 101)
        assert user.cpu().numpy().size % 101 == 0
        for i in range(batch):
            batch_scores = pred_i[i * 101: (i + 1) * 101].view(-1)
            _, indices = torch.topk(batch_scores, top_k)
            tmp_item_i = item_i[i * 101: (i + 1) * 101]
            recommends = torch.take(tmp_item_i, indices).cpu().numpy().tolist()
            gt_item = tmp_item_i[0].item()
            HR.append(evaluate.hit(gt_item, recommends))
            NDCG.append(evaluate.ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)


def create_csr_matrix_from_user_item_ratings(u, v, r, num_users, num_items):
    return csr_matrix((r, (u, v)), shape=(num_users, num_items))

def create_csr_matrix_from_social_adj_lists(social_adj_lists, num_users):
    row_ids = []
    col_ids = []
    values = []
    for source_user_id in social_adj_lists:
        for target_user_id in social_adj_lists[source_user_id]:
            row_ids.append(source_user_id)
            col_ids.append(target_user_id)
            values.append(1)
    trustMat = csr_matrix((values, (row_ids, col_ids)), shape=(num_users, num_users))
    return ((trustMat + trustMat.T) != 0) * 1

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


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
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--num_ng', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--reg', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    embed_dim = args.embed_dim

    path_data = './data/' + args.dataset + '/' + args.dataset + '.pkl'
    # path_data = './data/' + args.dataset + '/' + args.dataset + args.percentage + '.pkl'
    data_file = open(path_data, 'rb')

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, \
    social_adj_lists, item_adj_lists, ratings_list = pickle.load(data_file)
    # num_users = history_u_lists.__len__()
    # num_items = history_v_lists.__len__()
    num_users = max(list(history_u_lists.keys())) + 1
    num_items = max(list(history_v_lists.keys())) + 1
    num_ratings = ratings_list.__len__()

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

    trainMat = create_csr_matrix_from_user_item_ratings(train_u, train_v, train_r, num_users, num_items)
    trustMat = create_csr_matrix_from_social_adj_lists(social_adj_lists, num_users)

    num_users = trainMat.shape[0]
    num_items = trainMat.shape[1]

    train_data = np.hstack((np.array(train_u.reshape(-1, 1)), np.array(train_v.reshape(-1, 1)))).tolist()
    valid_data = np.hstack((np.array(valid_u.reshape(-1, 1)), np.array(valid_v.reshape(-1, 1)))).tolist()
    test_data = np.hstack((np.array(test_u.reshape(-1, 1)), np.array(test_v.reshape(-1, 1)))).tolist()
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

    trainset = BPRData(train_data, num_items, trainMat, args.num_ng, True)
    validset = BPRData(valid_data, num_items, trainMat, 0, False)
    testset = BPRData(test_data, num_items, trainMat, 0, False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size/2, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # model
    graphrec = GraphRec(num_users,
                 num_items,
                 num_ratings,
                 embed_dim,
                 history_u_lists,
                 history_ur_lists,
                 history_v_lists,
                 history_vr_lists,
                 social_adj_lists,
                 device).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    gc.collect()

    # load from checkpoint
    if args.load_from_checkpoint == True:
        checkpoint = torch.load('models/' + args.dataset + '.pt')
        graphrec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_valid_HR = 9999.0
    best_valid_NDCG = 9999.0
    best_test_HR = 9999.0
    best_test_NDCG = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        train(graphrec, device, train_loader, optimizer, epoch, args.reg, best_valid_HR, best_valid_NDCG, best_test_HR, best_test_NDCG)
        HR, NDCG = test(graphrec, device, valid_loader, args.top_k)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_valid_HR > HR:
            best_valid_HR = HR
            best_valid_NDCG = NDCG
            endure_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': graphrec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.dataset + '_BPR.pt')
        else:
            endure_count += 1
        print("HR on valid set: %.4f, NDCG:%.4f " % (HR, NDCG))
        HR, NDCG = test(graphrec, device, test_loader, args.top_k)
        if best_test_HR > HR:
            best_test_HR = HR
            best_test_NDCG = NDCG
        print('HR on test set: %.4f, NDCG:%.4f ' % (HR, NDCG))

        if endure_count > 5:
            break
    print('finished')


if __name__ == "__main__":
    main()