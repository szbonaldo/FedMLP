import logging
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
import seaborn as sns

from utils.feature_visual import tnse_Visual


class LocalUpdate(object):
    def __init__(self, args, client_id, dataset, idxs, class_pos_idx, class_neg_idx, active_class_list=None):
        self.args = args
        self.client_id = client_id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset, idxs, client_id, args, class_neg_idx, active_class_list)
        self.class_num_list = self.local_dataset.get_num_of_each_class(args)
        self.loss_w = [len(self.local_dataset) / i for i in self.class_num_list]
        self.loss_w_unknown = [1] * len(self.class_num_list)
        self.loss_w_unknown[client_id] = len(self.local_dataset) / self.class_num_list[client_id]
        self.loss_balanced = [1] * len(self.class_num_list)
        logging.info(self.loss_w)
        logging.info(
            f"---> Client{client_id}, each class num: {self.class_num_list}, total num: {len(self.local_dataset)}")
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr
        self.class_pos_idx = class_pos_idx
        self.class_neg_idx = class_neg_idx

    def train(self, rnd, net):
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")

        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        epoch_false_negetive_loss_avg = []
        epoch_true_negetive_loss_avg = []
        for i in range(self.args.n_classes - self.args.annotation_num):
            epoch_false_negetive_loss_avg.append(0)
            epoch_true_negetive_loss_avg.append(0)
        print(self.loss_w)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(), reduction='none')  # include sigmoid

        for epoch in range(self.args.local_ep):
            epoch_false_negetive_loss = []
            epoch_true_negetive_loss = []
            loss_false_negetive = []
            loss_true_negetive = []
            for i in range(self.args.n_classes - self.args.annotation_num):
                loss_false_negetive.append([])
                loss_true_negetive.append([])

            print('local_epoch:', epoch)
            batch_loss = []
            active_class_list_client = []
            for samples, item, active_class_list in self.ldr_train:
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                for i in range(self.args.annotation_num):
                    active_class_list_client.append(active_class_list[i][0].item())

                negetive_class_list_client = []
                _, logits = net(images)

                loss = bce_criterion(logits, labels)    # tensor(32, 5)

                class_miss_loss = []
                for i in range(self.args.n_classes):
                    if i not in active_class_list_client:
                        negetive_class_list_client.append(i)
                        class_miss_loss.append(loss[:, i].clone().detach().cpu())
                # print(self.class_pos_idx)
                # print(self.class_neg_idx)   # 不是从小到大

                for i in range(len(item)):
                    for j in range(len(negetive_class_list_client)):
                        if item[i].item() in self.class_neg_idx[negetive_class_list_client[j]]:
                            loss_false_negetive[j].append(class_miss_loss[j][i].item())

                        elif item[i].item() in self.class_pos_idx[negetive_class_list_client[j]]:
                            pass
                        else:
                            loss_true_negetive[j].append(class_miss_loss[j][i].item())

                loss = loss.sum()/(self.args.batch_size * self.args.n_classes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())

                self.iter_num += 1
            # if rnd % 5 == 0:
            #     for j in range(len(negetive_class_list_client)):
            #         plt.title(f'round:{rnd},epoch:{epoch},client:{self.client_id},miss class:{negetive_class_list_client[j]} loss distribution')
            #         sns.kdeplot(loss_false_negetive[j], label='FN')
            #         sns.kdeplot(loss_true_negetive[j], label='TN')
            #         plt.legend()
            #         plt.savefig(f'loss_fig/round:{rnd},epoch:{epoch},client:{self.client_id},miss class:{negetive_class_list_client[j]}_loss_distribution.png')
            #         print('ok')

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            for j in range(len(negetive_class_list_client)):
                epoch_false_negetive_loss.append(np.array(loss_false_negetive[j]).mean())
                epoch_true_negetive_loss.append(np.array(loss_true_negetive[j]).mean())


            epoch_false_negetive_loss_avg = np.add(epoch_false_negetive_loss_avg, np.array(epoch_false_negetive_loss))
            epoch_true_negetive_loss_avg = np.add(epoch_true_negetive_loss_avg, np.array(epoch_true_negetive_loss))

        net.cpu()
        self.optimizer.zero_grad()
        return net.state_dict(), np.array(epoch_loss).mean(), epoch_false_negetive_loss_avg/self.args.local_ep, epoch_true_negetive_loss_avg/self.args.local_ep, negetive_class_list_client

    def test(self, rnd, net, model_name):
        net.eval()
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                             reduction='none')  # include sigmoid
        epoch_false_negetive_loss = []
        epoch_true_negetive_loss = []
        loss_false_negetive = []
        loss_true_negetive = []
        for i in range(self.args.n_classes - self.args.annotation_num):
            loss_false_negetive.append([])
            loss_true_negetive.append([])

        active_class_list_client = []
        feature = torch.tensor([]).cuda()
        flags = torch.zeros([self.args.n_classes - self.args.annotation_num, len(self.ldr_train.dataset)]).cuda()  # 0:FN, 1:TN
        flag_class_one = torch.zeros([1, len(self.ldr_train.dataset)]).cuda()
        # print(flags.shape)
        # input()
        count = 0
        with torch.no_grad():
            for samples, item, active_class_list in self.ldr_train:
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                for i in range(len(item)):
                    flag_class_one[0, count*len(item)+i] = int(labels[i, 1] == 0)
                for i in range(self.args.annotation_num):
                    active_class_list_client.append(active_class_list[i][0].item())
                negetive_class_list_client = []
                feature_batch, logits = net(images)
                feature = torch.cat((feature, feature_batch), dim=0)

                loss = bce_criterion(logits, labels)  # tensor(32, 5)
                class_miss_loss = []
                for i in range(self.args.n_classes):
                    if i not in active_class_list_client:
                        negetive_class_list_client.append(i)
                        class_miss_loss.append(loss[:, i].clone().detach().cpu())

                for i in range(len(item)):
                    for j in range(len(negetive_class_list_client)):
                        if item[i].item() in self.class_neg_idx[negetive_class_list_client[j]]:
                            loss_false_negetive[j].append(class_miss_loss[j][i].item())
                            flags[j, i+count*len(item)] = 0

                        elif item[i].item() in self.class_pos_idx[negetive_class_list_client[j]]:
                            pass
                        else:
                            loss_true_negetive[j].append(class_miss_loss[j][i].item())
                            flags[j, i+count*len(item)] = 1
                count += 1

        # for j in range(len(negetive_class_list_client)):
        #     plt.title(
        #         f'model:{model_name},round:{rnd},client:{self.client_id},miss class:{negetive_class_list_client[j]} test loss distribution')
        #     sns.kdeplot(loss_false_negetive[j], label='FN')
        #     sns.kdeplot(loss_true_negetive[j], label='TN')
        #     plt.legend()
        #     plt.savefig(
        #         f'loss_fig/model:{model_name},round:{rnd},client:{self.client_id},miss class:{negetive_class_list_client[j]}_test_loss_distribution.png')
        #     plt.clf()

        for j in range(len(negetive_class_list_client)):
            tnse_Visual(feature.cpu(), flags[j].cpu(), rnd, f'model{model_name} class{negetive_class_list_client[j]} p=1')

        tnse_Visual(feature.cpu(), flag_class_one[0].cpu(), rnd, f'model{model_name} class{1} p=1')

        for j in range(len(negetive_class_list_client)):
            epoch_false_negetive_loss.append(np.array(loss_false_negetive[j]).mean())
            epoch_true_negetive_loss.append(np.array(loss_true_negetive[j]).mean())
        net.cpu()
        return epoch_false_negetive_loss, epoch_true_negetive_loss

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, client_id, args, class_neg_idx, active_class_list=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.client_id = client_id  # choose active classes
        self.annotation_num = args.annotation_num
        class_list = list(range(args.n_classes))
        if active_class_list is None:
            self.active_class_list = random.sample(class_list, self.annotation_num)
        else:
            self.active_class_list = active_class_list
        logging.info(f"Client ID: {self.client_id}, active_class_list: {self.active_class_list}")
        self.class_neg_idx = class_neg_idx

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        for i in range(len(sample['target'])):
            if i not in self.active_class_list and self.idxs[item] in self.class_neg_idx[i]:
                sample['target'][i] = 0
        return sample, self.idxs[item], self.active_class_list

    def get_num_of_each_class(self, args):
        class_sum = np.array([0.] * args.n_classes)
        for idx in self.idxs:
            class_sum += self.dataset.targets[idx]
        return class_sum.tolist()