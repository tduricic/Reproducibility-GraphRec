import operator

import numpy as np

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.folder import build_model_folder
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger

from .GraphRecModel import GraphRecModel
from .BPRData import BPRData


import torch.utils.data
import torch
import random
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm

np.random.seed(0)

class GraphRec(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Graph neural networks for social recommendation

        GraphRec presented by Fan et al. in `paper <https://dl.acm.org/doi/pdf/10.1145/3308558.3313488>`

        Args:
            meta:
                verbose: Field to enable verbose logs
                save_recs: Field to enable recommendation lists storage
                validation_metric: Metric used for model validation
            epochs: Number of epochs
            batch_size: Batch sizer
            lr: Learning rate
            factors: Number of latent factors

        To include the recommendation model, add it to the config file adopting the following pattern:

        .. code:: yaml

        models:
            external.GraphRec:
            meta:
                verbose: True
                save_recs: False
                validation_metric: nDCG@10
            epochs: 10
            batch_size: 128
            lr: 0.001
            factors: 64
        """
        self._params_list = [
            ("_epochs", "epochs", "epochs", 20, int, None),
            ("_batch_size", "batch_size", "batch_size", 128, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_final_prop", "final_prop", "final_prop", True, bool, None),
            ("_loss_function", "loss_function", "loss_function", "mse", str, None),
            ("_num_ng", "num_ng", "num_ng", 1, int, None),
            ("_gpu_id", "gpu_id", "gpu_id", 0, int, None),
            ("_seed", "seed", "seed", 42, int, None)
        ]
        self.autoset_params()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
        self._use_cuda = False
        if torch.cuda.is_available():
            self._use_cuda = True
        self._device = torch.device("cuda" if self._use_cuda else "cpu")

        if self._config.dataset == "toy_dataset":
            social_connections_filepath = "./data/" + config.dataset + "/processed/filtered_social_connections.tsv"
        else:
            social_connections_filepath = "./data/" + config.dataset + "/final/social_connections_comm.tsv"

        self._social_adj_lists = self.create_social_adj_lists(social_connections_filepath)
        # TODO what is still missing is to make a different implementation based on if there is or there isn't a validation set
        if hasattr(data, "val_dict"):
            self._history_u_lists, self._history_ur_lists, self._history_v_lists, self._history_vr_lists, \
            self._train_u, self._train_v, self._train_r, self._val_u, self._val_v, self._val_r, \
            self._test_u, self._test_v, self._test_r, self._ratings_list = \
                self.preprocess_data_validation(data.train_dict, data.val_dict, data.test_dict)
            self._num_users = max(set(self._train_u + self._val_u + self._test_u)) + 1
            self._num_items = max(set(self._train_v + self._val_v + self._test_v)) + 1
        else:
            self._history_u_lists, self._history_ur_lists, self._history_v_lists, self._history_vr_lists, \
            self._train_u, self._train_v, self._train_r, self._test_u, self._test_v, self._test_r, self._ratings_list = \
                self.preprocess_data_test(data.train_dict, data.test_dict)
            self._num_users = max(set(self._train_u + self._test_u)) + 1
            self._num_items = max(set(self._train_v + self._test_v)) + 1


        self._embed_dim = params.factors

        # self._valset = torch.utils.data.TensorDataset(torch.LongTensor(self._val_u),
        #                                                torch.LongTensor(self._val_v),
        #                                                torch.FloatTensor(self._val_r))
        # self._testset = torch.utils.data.TensorDataset(torch.LongTensor(self._test_u), torch.LongTensor(self._test_v),
        #                                         torch.FloatTensor(self._test_r))

        # self._val_loader = torch.utils.data.DataLoader(self._valset, batch_size=params.batch_size, shuffle=True)
        # self._test_loader = torch.utils.data.DataLoader(self._testset, batch_size=params.batch_size, shuffle=True)
        # self._num_users = self._history_u_lists.__len__()
        # self._num_items = self._history_v_lists.__len__()
        # self._num_users = max(list(self._history_u_lists.keys()))+1
        # self._num_items = max(list(self._history_v_lists.keys()))+1
        # self._num_users = len(set(self._train_u + self._val_u + self._test_u))
        # self._num_items = len(set(self._train_v + self._val_v + self._test_v))


        #self._unique_users = set(self._train_u + self._val_u + self._test_u)

        self._num_ratings = self._ratings_list.__len__()

        self._model = GraphRecModel(num_users=self._num_users,
                               num_items=self._num_items,
                               num_ratings=self._num_ratings,
                               learning_rate=params.lr,
                               embed_dim=self._embed_dim,
                               final_prop=self._final_prop,
                               loss_function=self._loss_function,
                               use_cuda=self._use_cuda,
                               history_u_lists=self._history_u_lists,
                               history_ur_lists=self._history_ur_lists,
                               history_v_lists=self._history_v_lists,
                               history_vr_lists=self._history_vr_lists,
                               social_adj_lists=self._social_adj_lists,
                               random_seed=self._seed).to(self._device)
        #self._model = GraphRec(self._enc_u, self._enc_v_history, self._r2e, random_seed=self._seed).to(self._device)

    @property
    def name(self):
        return "GraphRec" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()
        self._model.train()

        if self._loss_function == "mse":
            train_set = torch.utils.data.TensorDataset(torch.LongTensor(self._train_u), torch.LongTensor(self._train_v), torch.FloatTensor(self._train_r))
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self._batch_size, shuffle=True)
            for it in self.iterate(self._epochs):
                print("Epoch: " + str(it))
                epoch_loss = 0.0
                for i, (batch_nodes_u, batch_nodes_v, labels_list) in enumerate(train_loader, 0):
                    self._model.optimizer.zero_grad()
                    loss = self._model.loss(batch_nodes_u.to(self._device), batch_nodes_v.to(self._device), labels_list.to(self._device))
                    loss.backward(retain_graph=True)
                    self._model.optimizer.step()
                    epoch_loss += loss.item()

                self.evaluate(it, epoch_loss / (it + 1))
        elif self._loss_function == "bce":
            train_mat = csr_matrix((self._train_r, (self._train_u, self._train_v)), shape=(self._num_users, self._num_items))
            train_data = np.hstack((np.array(self._train_u).reshape(-1, 1), np.array(self._train_v).reshape(-1, 1))).tolist()
            train_set = BPRData(train_data, self._num_items, train_mat, self._num_ng, True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(self._batch_size / 2), shuffle=True, num_workers=0)
            print("start negative sample...")
            train_loader.dataset.ng_sample()
            print("finish negative sample...")
            for it in self.iterate(self._epochs):
                print("Epoch: " + str(it))
                epoch_loss = 0.0
                for i, data in enumerate(train_loader, 0):
                    u, pos, neg = data
                    u = u.long().to(self._device)
                    pos = pos.long().to(self._device)
                    neg = neg.long().to(self._device)

                    batch_nodes_u = torch.hstack((u, u))
                    batch_nodes_v = torch.hstack((pos, neg))
                    labels_list = torch.hstack((torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])))

                    self._model.optimizer.zero_grad()
                    loss = self._model.loss(batch_nodes_u.to(self._device), batch_nodes_v.to(self._device),
                                            labels_list.to(self._device))
                    loss.backward(retain_graph=True)
                    self._model.optimizer.step()
                    epoch_loss += loss.item()
                self.evaluate(it, epoch_loss / (it + 1))

    def get_final_prop_recommendations(self, k: int = 100):
        predictions_top_k_val = {}
        predictions_top_k_test = {}
        with torch.no_grad():
            # Small hack, only test items are the candidate items
            # v = torch.tensor(list(set(self._test_v))).to(self._device)
            x_u, x_v = self._model.propagate_embeddings(torch.LongTensor(list(range(self._num_users))), torch.LongTensor(list(range(self._num_items))))

            #for index, offset in enumerate(range(0, len(self._data.users), self._batch_size)):
            #    offset_stop = min(offset + self._batch_size, len(self._data.users))
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = None
                for user_index in range(offset, offset_stop):
                    if user_index not in self._data.users:
                        if predictions is None:
                            predictions = np.full(self._num_items, -np.inf)[self._data.items]
                        else:
                            predictions = np.vstack((predictions, np.full(self._num_items, -np.inf)[self._data.items]))
                    else:
                        user_id = self._data.users[user_index]
                        x_uid = x_u[user_id].repeat(self._num_items, 1).to(self._device)

                        # TODO can we replace numpy with pytorch?
                        u_predictions = self._model.cat_and_propagate_user_item_embeddings(x_uid, x_v).data.cpu().numpy()
                        if predictions is None:
                            predictions = np.array(u_predictions)[self._data.items]
                        else:
                            predictions = np.vstack((predictions, u_predictions[self._data.items]))
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        self._model.train()
        return predictions_top_k_val, predictions_top_k_test

    def get_matmul_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        x_u, x_v = self._model.propagate_embeddings(torch.LongTensor(list(range(self._num_users))), torch.LongTensor(list(range(self._num_items))))
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = torch.matmul(x_u[offset: offset_stop], x_v.t())[:, self._data.items]
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_recommendations(self, k: int = 100):
        self._model.eval()
        if self._final_prop:
            predictions_top_k_val, predictions_top_k_test = self.get_final_prop_recommendations(k)
        elif not self._final_prop:
            predictions_top_k_val, predictions_top_k_test = self.get_matmul_recommendations(k)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], self._device, k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath (
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False

    def preprocess_data_validation(self, train_dict, val_dict, test_dict):
        history_u_lists = self.create_history_u_lists(train_dict)
        history_ur_lists = self.create_history_ur_lists(train_dict)
        history_v_lists = self.create_history_v_lists(train_dict)
        history_vr_lists = self.create_history_vr_lists(train_dict)
        train_u, train_v, train_r = self.create_uvr(train_dict)
        val_u, val_v, val_r = self.create_uvr(val_dict)
        test_u, test_v, test_r  = self.create_uvr(test_dict)
        ratings_list = self.create_ratings_list(train_r + val_r + test_r)

        return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
        train_u, train_v, train_r, val_u, val_v, val_r, test_u, test_v, test_r, ratings_list

    def preprocess_data_test(self, train_dict, test_dict):
        history_u_lists = self.create_history_u_lists(train_dict)
        history_ur_lists = self.create_history_ur_lists(train_dict)
        history_v_lists = self.create_history_v_lists(train_dict)
        history_vr_lists = self.create_history_vr_lists(train_dict)
        train_u, train_v, train_r = self.create_uvr(train_dict)
        test_u, test_v, test_r  = self.create_uvr(test_dict)
        ratings_list = self.create_ratings_list(train_r + test_r)

        return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
        train_u, train_v, train_r, test_u, test_v, test_r, ratings_list

    def create_history_u_lists(self, user_item_ratings_dict):
        history_u_lists = {}
        for user_id in user_item_ratings_dict:
            history_u_lists[user_id] = list(user_item_ratings_dict[user_id].keys())
        return history_u_lists

    def create_history_ur_lists(self, user_item_ratings_dict):
        history_ur_lists = {}
        for user_id in user_item_ratings_dict:
            history_ur_lists[user_id] = list(user_item_ratings_dict[user_id].values())
        return history_ur_lists

    def create_history_v_lists(self, user_item_ratings_dict):
        history_v_lists = {}
        for user_id in user_item_ratings_dict:
            for item_id in user_item_ratings_dict[user_id]:
                if item_id not in history_v_lists:
                    history_v_lists[item_id] = []
                history_v_lists[item_id].append(user_id)
        return history_v_lists

    def create_history_vr_lists(self, user_item_ratings_dict):
        history_vr_lists = {}
        for user_id in user_item_ratings_dict:
            for item_id in user_item_ratings_dict[user_id]:
                if item_id not in history_vr_lists:
                    history_vr_lists[item_id] = []
                history_vr_lists[item_id].append(user_item_ratings_dict[user_id][item_id])
        return history_vr_lists

    def create_uvr(self, user_item_ratings_dict):
        uvr_list = []
        u_list = []
        v_list = []
        r_list = []
        for user_id in user_item_ratings_dict:
            for item_id in user_item_ratings_dict[user_id]:
                uvr_list.append((user_id, item_id, user_item_ratings_dict[user_id][item_id]))
        random.shuffle(uvr_list)
        for (u, v, r) in uvr_list:
            u_list.append(u)
            v_list.append(v)
            r_list.append(r)
        return u_list, v_list, r_list

    def create_social_adj_lists(self, social_connections_filepath):
        social_adj_lists = {}
        with open(social_connections_filepath) as f:
            for line in f:
                tokens = line.split('\t')
                user_1 = int(tokens[0])
                user_2 = int(tokens[1])

                if user_1 not in social_adj_lists:
                    social_adj_lists[user_1] = set()
                if user_2 not in social_adj_lists:
                    social_adj_lists[user_2] = set()

                social_adj_lists[user_1].add(user_2)
                social_adj_lists[user_2].add(user_1)

        return social_adj_lists


    def create_ratings_list(self, ratings_list):
        return sorted(list(set(ratings_list)))