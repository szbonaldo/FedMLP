import copy
import logging
import math
import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from itertools import chain
import seaborn as sns

from utils.evaluations import globaltest, classtest
from utils.feature_visual import tnse_Visual
from utils.FedNoRo import LogitAdjust_Multilabel, LA_KD
from utils.utils import max_m_indices, min_n_indices


class LocalUpdate(object):
    def __init__(self, args, client_id, dataset, idxs, class_pos_idx, class_neg_idx, active_class_list=None, student=None, teacher_neg=None, teacher_act=None, dataset_test=None):
        self.teacher_neg = teacher_neg
        self.teacher_act = teacher_act
        self.dataset_test = dataset_test
        self.ema_model = teacher_neg
        self.student = student
        self.args = args
        self.client_id = client_id
        self.idxs = idxs
        self.dataset = dataset
        self.active_class_list = active_class_list
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
        self.flag = True
        self.confuse_matrix = torch.zeros((8, 8)).cuda()

    def find_rows(self, tensor, up, down):
        condition = torch.all(torch.logical_or(tensor > up, tensor < down), axis=1)
        row_indices = torch.where(condition)[0]
        return row_indices

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def torch_tile(self, tensor, dim, n):
        if dim == 0:
            return tensor.unsqueeze(0).transpose(0, 1).repeat(1, n, 1).view(-1, tensor.shape[1])
        else:
            return tensor.unsqueeze(0).transpose(0, 1).repeat(1, 1, n).view(tensor.shape[0], -1)

    def get_confuse_matrix(self, logits, labels):
        source_prob = []
        for i in range(8):
            mask = self.torch_tile(torch.unsqueeze(labels[:, i], -1), 1, 8)
            logits_mask_out = logits * mask
            logits_avg = torch.sum(logits_mask_out, dim=0) / (torch.sum(labels[:, i]) + 1e-8)
            prob = torch.sigmoid(logits_avg / 2.0)
            source_prob.append(prob)
        return torch.stack(source_prob)

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def sigmoid_mse_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)

        mse_loss = (input_softmax - target_softmax) ** 2
        return mse_loss

    def kd_loss(self, source_matrix, target_matrix):
        Q = source_matrix
        P = target_matrix
        loss = (F.kl_div(Q.log(), P, None, None, 'batchmean') + F.kl_div(P.log(), Q, None, None, 'batchmean')) / 2.0
        return loss

    def train_FedNoRo(self, id, rnd, net, writer1, weight_kd = None, clean_clients=None, noisy_clients=None):
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        if rnd < self.args.rounds_FedNoRo_warmup:
            student_net = copy.deepcopy(net).cuda()
            teacher_net = copy.deepcopy(net).cuda()
            student_net.train()
            teacher_net.eval()
            # set the optimizer
            self.optimizer = torch.optim.Adam(
                student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

            # train and update
            epoch_loss = []
            for epoch in range(self.args.local_ep):
                batch_loss = []
                for k, (samples, item, active_class_list) in enumerate(self.ldr_train):
                    if k == 0:
                        active_class_list_client = []
                        negetive_class_list_client = []
                        for i in range(self.args.annotation_num):
                            active_class_list_client.append(active_class_list[i][0].item())
                        for i in range(self.args.n_classes):
                            if i not in active_class_list_client:
                                negetive_class_list_client.append(i)
                                self.class_num_list[i] = 0
                    criterion = LA_KD(cls_num_list=self.class_num_list, num=len(self.ldr_train.dataset), active_class_list_client=active_class_list_client, negative_class_list_client=negetive_class_list_client)
                    images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)

                    _, logits = student_net(images)
                    # print(logits)

                    with torch.no_grad():
                        _, teacher_output = teacher_net(images)
                        soft_label = torch.sigmoid(teacher_output / 0.8)
                    # print(teacher_output)
                    logits_sig = torch.sigmoid(logits).cuda()
                    loss = criterion(logits_sig, labels, soft_label, weight_kd)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_loss.append(loss.item())
                    self.iter_num += 1
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
        else:
            if id in clean_clients:
                net.train()
                self.optimizer = torch.optim.Adam(
                    net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
                epoch_loss = []
                ce_criterion = LogitAdjust_Multilabel(cls_num_list=self.class_num_list, num=len(self.ldr_train.dataset))
                for epoch in range(self.args.local_ep):
                    batch_loss = []
                    for k, (samples, item, active_class_list) in enumerate(self.ldr_train):
                        if k == 0:
                            active_class_list_client = []
                            negetive_class_list_client = []
                            for i in range(self.args.annotation_num):
                                active_class_list_client.append(active_class_list[i][0].item())
                            for i in range(self.args.n_classes):
                                if i not in active_class_list_client:
                                    negetive_class_list_client.append(i)
                        images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                        _, logits = net(images)
                        logits_sig = torch.sigmoid(logits).cuda()
                        loss = ce_criterion(logits_sig, labels)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        batch_loss.append(loss.item())
                        self.iter_num += 1
                    self.epoch = self.epoch + 1
                    epoch_loss.append(np.array(batch_loss).mean())
            elif id in noisy_clients:
                student_net = copy.deepcopy(net).cuda()
                teacher_net = copy.deepcopy(net).cuda()
                student_net.train()
                teacher_net.eval()
                # set the optimizer
                self.optimizer = torch.optim.Adam(
                    student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

                # train and update
                epoch_loss = []
                criterion = LA_KD(cls_num_list=self.class_num_list, num=len(self.ldr_train.dataset))

                for epoch in range(self.args.local_ep):
                    batch_loss = []
                    for k, (samples, item, active_class_list) in enumerate(self.ldr_train):
                        if k == 0:
                            active_class_list_client = []
                            negetive_class_list_client = []
                            for i in range(self.args.annotation_num):
                                active_class_list_client.append(active_class_list[i][0].item())
                            for i in range(self.args.n_classes):
                                if i not in active_class_list_client:
                                    negetive_class_list_client.append(i)
                        images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)

                        _, logits = student_net(images)
                        with torch.no_grad():
                            _, teacher_output = teacher_net(images)
                            soft_label = torch.sigmoid(teacher_output / 0.8)
                        logits_sig = torch.sigmoid(logits).cuda()
                        loss = criterion(logits_sig, labels, soft_label, weight_kd)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        batch_loss.append(loss.item())
                        self.iter_num += 1
                    self.epoch = self.epoch + 1
                    epoch_loss.append(np.array(batch_loss).mean())
        student_net.cpu()
        self.optimizer.zero_grad()
        return student_net.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

    def train_CBAFed(self, rnd, net, pt=None, tao=None):
        if rnd < self.args.rounds_CBAFed_warmup:  # stage1
            class_num_list = torch.zeros(self.args.n_classes)
            data_num = 0
            net.train()
            # set the optimizer
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            # train and update
            epoch_loss = []
            print(self.loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                                 reduction='none')  # include sigmoid
            for epoch in range(self.args.local_ep):
                print('local_epoch:', epoch)
                batch_loss = []
                for j, (samples, item, active_class_list) in enumerate(self.ldr_train):
                    if j == 0:
                        active_class_list_client = []
                        negetive_class_list_client = []
                        for i in range(self.args.annotation_num):
                            active_class_list_client.append(active_class_list[i][0].item())
                        for i in range(self.args.n_classes):
                            if i not in active_class_list_client:
                                negetive_class_list_client.append(i)
                    images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                    # class_num_list = class_num_list + torch.sum(labels, dim=0)
                    data_num = data_num + len(labels)
                    _, logits = net(images)
                    loss_sup = bce_criterion_sup(logits, labels)  # tensor(32, 5)
                    loss_sup = loss_sup[:, active_class_list_client].sum() / (
                                self.args.batch_size * self.args.annotation_num)  # supervised_loss
                    loss = loss_sup
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    self.iter_num += 1
                for id in active_class_list_client:
                    class_num_list[id] = data_num
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
            return net.state_dict(), np.array(
                epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client, class_num_list, data_num
        else:
            class_num_list = torch.zeros(self.args.n_classes)
            data_num = 0
            net.train()
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            # train and update
            epoch_loss = []

            for epoch in range(self.args.local_ep):
                print('local_epoch:', epoch)
                batch_loss = []
                for j, (samples, item, active_class_list) in enumerate(self.ldr_train):
                    idx_neg = []
                    if j == 0:
                        active_class_list_client = []
                        negetive_class_list_client = []
                        for i in range(self.args.annotation_num):
                            active_class_list_client.append(active_class_list[i][0].item())
                        for i in range(self.args.n_classes):
                            if i not in active_class_list_client:
                                negetive_class_list_client.append(i)
                    images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                    _, logits = net(images)
                    prob = torch.sigmoid(logits)
                    # print(negetive_class_list_client)
                    # print(prob)
                    for i in negetive_class_list_client:
                        noise_num = len(torch.where(prob[:, i] > tao[i])[0])
                        clean_num = len(torch.where(prob[:, i] < (1-tao[i]))[0])
                        labels[:, i] = torch.where(prob[:, i] > tao[i], 1, labels[:, i])
                        pseudo_idx = torch.where((prob[:, i] > tao[i]) | (prob[:, i] < (1-tao[i])))[0]
                        idx_neg.append(pseudo_idx)
                        class_num_list[i] = class_num_list[i] + len(pseudo_idx)
                        # print(len(pseudo_idx))
                        data_num = data_num + len(pseudo_idx)
                        if noise_num == 0:
                            self.loss_w[i] = 1
                        else:
                            self.loss_w[i] = (noise_num+clean_num) / noise_num
                    print(self.loss_w)
                    for i in active_class_list_client:
                        class_num_list[i] = class_num_list[i] + len(labels)
                    data_num = data_num + len(labels)*self.args.annotation_num
                    bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                                             reduction='none')  # include sigmoid
                    loss_sup = bce_criterion_sup(logits, labels)  # tensor(32, 5)
                    loss_sup_act = loss_sup[:, active_class_list_client].sum() / (
                            self.args.batch_size * self.args.annotation_num)  # supervised_loss
                    # loss_sup_act = loss_sup.sum() / (self.args.batch_size * self.args.n_classes)  # supervised_loss
                    loss = loss_sup_act
                    for k, i in enumerate(negetive_class_list_client):
                        if len(idx_neg[k]) != 0:
                            loss += loss_sup[idx_neg[k], i].sum() / len(idx_neg[k])  # supervised_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    self.iter_num += 1
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
            return net.state_dict(), np.array(
                epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client, class_num_list, data_num

    def train_FedIRM(self, rnd, target_matrix, writer1, negetive_class_list, active_class_list_client_i, net):    # MICCAI2021
        if rnd < self.args.rounds_FedIRM_sup:
            net.train()
            # set the optimizer
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            # train and update
            epoch_loss = []
            self.confuse_matrix = torch.zeros((8, 8)).cuda()
            print(self.loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                                     reduction='none')  # include sigmoid
            for epoch in range(self.args.local_ep):
                print('local_epoch:', epoch)
                batch_loss = []
                for j, (samples, item, active_class_list) in enumerate(self.ldr_train):
                    if j == 0:
                        active_class_list_client = []
                        negetive_class_list_client = []
                        for i in range(self.args.annotation_num):
                            active_class_list_client.append(active_class_list[i][0].item())
                        for i in range(self.args.n_classes):
                            if i not in active_class_list_client:
                                negetive_class_list_client.append(i)
                    images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(
                        self.args.device), samples["target"].to(self.args.device)
                    _, logits1 = net(images1)
                    fe2, logits2 = net(images2)
                    if rnd == self.args.rounds_FedIRM_sup - 1:  # first relation matrix
                        self.confuse_matrix = self.confuse_matrix + self.get_confuse_matrix(logits1, labels)
                    loss_sup = bce_criterion_sup(logits1, labels) + bce_criterion_sup(logits2, labels)  # tensor(32, 5)
                    loss_sup = loss_sup[:, active_class_list_client].sum() / (
                                self.args.batch_size * self.args.annotation_num)  # supervised_loss
                    self.optimizer.zero_grad()
                    loss_sup.backward()
                    self.optimizer.step()
                    batch_loss.append(loss_sup.item())
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
            if rnd == self.args.rounds_FedIRM_sup - 1:
                with torch.no_grad():
                    self.confuse_matrix = self.confuse_matrix / (1.0 * self.args.local_ep * (j + 1))
                return net.state_dict(), np.array(
                    epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client, self.confuse_matrix
            else:
                return net.state_dict(), np.array(
                        epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

        else:   # Inter-client Relation Matching
            if self.flag:
                self.ema_model.load_state_dict(net.state_dict())
                self.flag = False
                print('done')
            net.train()
            # set the optimizer
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            # train and update
            epoch_loss = []
            self.confuse_matrix = torch.zeros((8, 8)).cuda()
            print(self.loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                                     reduction='none')  # include sigmoid
            for epoch in range(self.args.local_ep):
                print('local_epoch:', epoch)
                batch_loss = []
                for j, (samples, item, active_class_list) in enumerate(self.ldr_train):
                    if j == 0:
                        active_class_list_client = []
                        negetive_class_list_client = []
                        for i in range(self.args.annotation_num):
                            active_class_list_client.append(active_class_list[i][0].item())
                        for i in range(self.args.n_classes):
                            if i not in active_class_list_client:
                                negetive_class_list_client.append(i)
                    images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(
                        self.args.device), samples["target"].to(self.args.device)
                    _, outputs = net(images1)
                    with torch.no_grad():
                        _, ema_output = self.ema_model(images2)
                    with torch.no_grad():
                        preds = torch.sigmoid(outputs).cuda()
                        uncertainty = -1.0 * (torch.sum(preds * torch.log(preds + 1e-6), dim=1) + torch.sum(
                            (1 - preds) * torch.log(1 - preds + 1e-6), dim=1))
                        uncertainty_mask = (uncertainty < 2.0)
                    with torch.no_grad():
                        activations = torch.sigmoid(outputs).cuda()
                        confidence_mask = torch.zeros(len(uncertainty_mask), dtype=bool).cuda()
                        confidence_mask[self.find_rows(activations, 0.7, 0.3)] = True
                    mask = confidence_mask * uncertainty_mask
                    if mask.sum().item() != 0:
                        pseudo_labels = activations[mask] > 0.5
                        source_matrix = self.get_confuse_matrix(outputs[mask], pseudo_labels)
                    else:
                        source_matrix = 0.5*torch.ones((8, 8)).cuda()
                    target_matrix = target_matrix.cuda()
                    print(source_matrix)
                    print(target_matrix)
                    consistency_weight = self.get_current_consistency_weight(rnd)
                    consistency_dist = torch.sum(self.sigmoid_mse_loss(outputs, ema_output)) / self.args.batch_size
                    consistency_loss = consistency_dist
                    loss = consistency_weight * consistency_loss + consistency_weight * torch.sum(
                        self.kd_loss(source_matrix, target_matrix))
                    fe2, logits2 = net(images2)
                    self.confuse_matrix = self.confuse_matrix + self.get_confuse_matrix(outputs, labels)
                    loss_sup = bce_criterion_sup(outputs, labels) + bce_criterion_sup(logits2, labels)  # tensor(32, 5)
                    loss_sup = loss_sup[:, active_class_list_client].sum() / (
                            self.args.batch_size * self.args.annotation_num)  # supervised_loss
                    loss = loss + loss_sup
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.update_ema_variables(net, self.ema_model, self.args.ema_decay, self.iter_num)
                    batch_loss.append(loss.item())
                    self.iter_num = self.iter_num + 1
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
                with torch.no_grad():
                    self.confuse_matrix = self.confuse_matrix / (1.0 * self.args.local_ep * (j + 1))
            return net.state_dict(), np.array(
                epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client, self.confuse_matrix

    def train_RoFL(self, net, f_G, epoch):
        time1 = time.time()
        sim = torch.nn.CosineSimilarity(dim=1)
        pseudo_labels = torch.zeros(len(self.dataset), self.args.n_classes, dtype=torch.float64, device=self.args.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        epoch_loss = []
        net.eval()
        f_k = torch.zeros(2*self.args.n_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(2*self.args.n_classes, 1, device=self.args.device)

        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for i, (samples, item, active_class_list) in enumerate(self.ldr_train):
                if i == 0:
                    active_class_list_client = []
                    negetive_class_list_client = []
                    for i in range(self.args.annotation_num):
                        active_class_list_client.append(active_class_list[i][0].item())
                    for i in range(self.args.n_classes):
                        if i not in active_class_list_client:
                            negetive_class_list_client.append(i)
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                feature, logit = net(images)
                probs = torch.sigmoid(logit)  # soft predict
                accuracy_th = 0.5
                preds = probs > accuracy_th  # hard predict
                preds = preds.to(torch.float64)
                pseudo_labels[item] = preds
                if epoch == 0:
                    for cls in range(self.args.n_classes):
                        f_k[2*cls] += torch.sum(feature[torch.where(labels[:, cls] == 0)[0]], dim=0)
                        f_k[2*cls+1] += torch.sum(feature[torch.where(labels[:, cls] == 1)[0]], dim=0)
                        n_labels[2*cls] += len(torch.where(labels[:, cls] == 0)[0])
                        n_labels[2*cls+1] += len(torch.where(labels[:, cls] == 1)[0])

        if epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G
        time2 = time.time()
        # print('local test time: ', time2-time1)
        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for samples, item, active_class_list in self.ldr_train:
                time4 = time.time()
                net.zero_grad()
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                feature, logit = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)

                small_loss_idxs, loss_w = self.get_small_loss_samples(logit, labels, self.args.forget_rate, negetive_class_list_client)

                y_k_tilde = torch.zeros(self.args.batch_size, self.args.n_classes, device=self.args.device)
                mask = torch.zeros(self.args.batch_size, device=self.args.device)
                for i in small_loss_idxs:
                    for cls in range(self.args.n_classes):
                        f_cls = f_k[[2*cls, 2*cls+1], :]
                        y_k_tilde[i, cls] = torch.argmax(sim(f_cls, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if torch.equal(y_k_tilde[i], labels[i]):
                        mask[i] = 1

                # When to use pseudo-labels
                if epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        pseudo_labels[item[i]] = labels[i]

                # For loss calculating
                mask_resize = mask.unsqueeze(1).repeat([1, 5])
                new_labels = mask_resize[small_loss_idxs] * labels[small_loss_idxs] + (1 - mask_resize[small_loss_idxs]) * \
                             pseudo_labels[item[small_loss_idxs]]
                new_labels = new_labels.type(torch.float).to(self.args.device)
                time5 = time.time()
                # print('batch train prepare time: ', time5-time4)
                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels, loss_w, epoch)
                # print('loss: ', loss)
                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(2*self.args.n_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(2*self.args.n_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    for cls in range(self.args.n_classes):
                        if labels[i, cls] == 0:
                            f_kj_hat[2 * cls] += feature[i]
                            n[2 * cls] += 1
                        else:
                            f_kj_hat[2 * cls + 1] += feature[i]
                            n[2 * cls + 1] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(2*self.args.n_classes, 1, device=self.args.device)
                f_k = (one - sim(f_k, f_kj_hat).reshape(2*self.args.n_classes, 1) ** 2) * f_k + (
                            sim(f_k, f_kj_hat).reshape(2*self.args.n_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())
                time6 = time.time()
                # print('batch train time: ', time6 - time5)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        time3 = time.time()
        # print('local train time: ', time3 - time2)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k

    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels, loss_w, epoch):
        mse = torch.nn.MSELoss(reduction='none')
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_w).cuda())  # include sigmoid
        L_c = bce(logit[small_loss_idxs], new_labels)
        prob = torch.sigmoid(logit).cuda()
        for i in range(self.args.n_classes):
            if i == 0:
                L_cen = torch.sum(
                    mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[(2*i+labels[small_loss_idxs, i]).cpu().numpy()]), 1))/(len(small_loss_idxs)*self.args.feature_dim)
            else:
                L_cen += torch.sum(
                    mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[(2*i+labels[small_loss_idxs, i]).cpu().numpy()]), 1))/(len(small_loss_idxs)*self.args.feature_dim)
        L_cen = L_cen / self.args.n_classes
        for i in range(self.args.n_classes):
            clsi_prob = prob[:, i].unsqueeze(1)
            clsi_prob = torch.cat((clsi_prob, 1-clsi_prob), dim=1).cuda()
            if i == 0:
                L_e = -torch.mean(torch.sum(clsi_prob[small_loss_idxs] * torch.log(clsi_prob[small_loss_idxs]), dim=1))
            else:
                L_e += -torch.mean(torch.sum(clsi_prob[small_loss_idxs] * torch.log(clsi_prob[small_loss_idxs]), dim=1))
        L_e = L_e / self.args.n_classes
        lambda_e = self.args.lambda_e
        lambda_cen = self.args.lambda_cen
        if epoch < self.args.T_pl:
            lambda_cen = (self.args.lambda_cen * epoch) / self.args.T_pl
        # print('L_c: ', L_c.item(), 'L_cen: ', L_cen.item(), 'L_e: ', L_e.item())
        if math.isnan(L_c.item()) or math.isnan(L_cen.item()) or math.isnan(L_e.item()):
            print(logit)
            print(feature)
            print(loss_w)
        print('loss: ', L_c + (lambda_cen * L_cen) + (lambda_e * L_e))
        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)

    def get_small_loss_samples(self, y_pred, y_true, forget_rate, negetive_class_list_client):
        loss_w = self.loss_w
        for i in negetive_class_list_client:
            loss_w[i] = 5.
        loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_w).cuda(), reduction='none')  # include sigmoid
        loss = torch.sum(loss_func(y_pred, y_true), dim=1)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        return ind_update, loss_w

    def train(self, rnd, net, writer1):
        # teacher_neg = deepcopy(net).to(self.args.device)    # try
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")

        net.train()
        # teacher_neg.eval()  # try
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        print(self.loss_w)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(), reduction='none')  # include sigmoid
        active_class_list_client = []
        negetive_class_list_client = []
        # mse = nn.MSELoss()  # try
        for epoch in range(self.args.local_ep):
            print('local_epoch:', epoch)
            batch_loss = []
            for k, (samples, item, active_class_list) in enumerate(self.ldr_train):
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)
                if k == 0:
                    for i in range(self.args.annotation_num):
                        active_class_list_client.append(active_class_list[i][0].item())
                    for i in range(self.args.n_classes):
                        if i not in active_class_list_client:
                            negetive_class_list_client.append(i)
                _, logits = net(images)

                # logits1_sig = torch.sigmoid(logits).cuda()  # try
                # with torch.no_grad():
                #     _, logits2 = teacher_neg(images)
                #     logits2_sig = torch.sigmoid(logits2).cuda()

                loss = bce_criterion(logits, labels)    # tensor(32, 5)
                loss = loss.sum()/(self.args.batch_size * self.args.n_classes)  # all_class_loss
                # loss = loss[:, active_class_list_client].sum()/(self.args.batch_size * self.args.annotation_num)
                # mask = torch.ones_like(loss)    # active_class_loss
                # for i in negetive_class_list_client:
                #     mask[:, i] = 0.
                # loss = (loss * mask).sum()/(self.args.batch_size * self.args.annotation_num)
                # loss += mse(logits1_sig[:, negetive_class_list_client], logits2_sig[:, negetive_class_list_client])  # try

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

            # for u in range(5):
            #     result = classtest(deepcopy(net).cuda(), test_dataset=self.dataset_test, args=self.args, classid=u)
            #     BACC, R, F1, P= result["BACC"], result["R"], result["F1"], result["P"]
            #     logging.info(
            #         "-----> BACC: %.2f, R: %.2f, F1: %.2f, P: %.2f" % (BACC * 100, R * 100, F1 * 100, P * 100))
            #     writer1.add_scalar(f'test_client_class{u}/BACC', BACC, epoch)
            #     writer1.add_scalar(f'test_client_class{u}/R', R, epoch)
            #     writer1.add_scalar(f'test_client_class{u}/F1', F1, epoch)
            #     writer1.add_scalar(f'test_client_class{u}/P', P, epoch)

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
        net.cpu()
        self.optimizer.zero_grad()
        return net.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

    def train_RSCFed(self, rnd, net):
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        self.student = deepcopy(net).to(self.args.device)
        self.teacher_neg.eval()
        self.student.train()

        # set the optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        print(self.loss_w)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(), reduction='none')  # include sigmoid
        mse_loss = nn.MSELoss()

        for epoch in range(self.args.local_ep):
            print('local_epoch:', epoch)
            batch_loss = []
            for samples, item, active_class_list in self.ldr_train:
                active_class_list_client = []
                negetive_class_list_client = []
                for i in range(self.args.annotation_num):
                    active_class_list_client.append(active_class_list[i][0].item())
                for i in range(self.args.n_classes):
                    if i not in active_class_list_client:
                        negetive_class_list_client.append(i)
                images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(self.args.device), samples["target"].to(self.args.device)
                _, logits1_stu = self.student(images1)
                logits1_stu_sig = torch.sigmoid(logits1_stu).cuda()

                with torch.no_grad():
                    _, logits2_tea = self.teacher_neg(images2)
                    # _, logits2_tea = self.teacher_neg(images1)
                    logits2_tea_sig = torch.sigmoid(logits2_tea).cuda()
                loss = bce_criterion(logits1_stu, labels)    # tensor(32, 5)
                loss_sup = loss[:, active_class_list_client].sum()/(self.args.batch_size * self.args.annotation_num)  # supervised_loss
                loss_unsup = mse_loss(logits1_stu_sig[:, negetive_class_list_client], logits2_tea_sig[:, negetive_class_list_client])
                loss = loss_sup + loss_unsup
                # loss = loss_sup

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update teacher
                state_dict1 = self.teacher_neg.state_dict()
                state_dict2 = deepcopy(self.student).state_dict()
                weight1 = 1 - 0.001
                weight2 = 0.001
                weighted_state_dict = {}
                for name in state_dict1:
                    weighted_state_dict[name] = weight1 * state_dict1[name] + weight2 * state_dict2[name]
                self.teacher_neg.load_state_dict(weighted_state_dict)
                self.teacher_neg = self.teacher_neg.to(self.args.device)
                batch_loss.append(loss.item())
                self.iter_num += 1

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        net.cpu()
        self.optimizer.zero_grad()
        return self.student.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

    def train_FixMatch(self, rnd, net):
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        net.train()

        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        print(self.loss_w)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(), reduction='none')  # include sigmoid
        bce_criterion_unsup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w_unknown).cuda(), reduction='none')  # include sigmoid

        for epoch in range(self.args.local_ep):
            print('local_epoch:', epoch)
            batch_loss = []
            for samples, item, active_class_list in self.ldr_train:
                active_class_list_client = []
                negetive_class_list_client = []
                for i in range(self.args.annotation_num):
                    active_class_list_client.append(active_class_list[i][0].item())
                for i in range(self.args.n_classes):
                    if i not in active_class_list_client:
                        negetive_class_list_client.append(i)
                images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(self.args.device), samples["target"].to(self.args.device)
                _, logits_weak = net(images1)
                logits_weak_sig = torch.sigmoid(logits_weak).cuda()
                idx = set(range(self.args.batch_size))
                for c in negetive_class_list_client:
                    idx = idx.intersection(set(torch.where(logits_weak_sig[:, c] > 0.8)[0].tolist()).union(set(torch.where(logits_weak_sig[:, c] < 0.2)[0].tolist())))
                idx = list(idx)
                logits_weak_sig = torch.where(logits_weak_sig > 0.5, 1.0, 0.0)  # turn to hard label
                _, logits_strong = net(images2)
                loss = bce_criterion(logits_weak, labels)    # tensor(32, 5)
                loss_sup = loss[:, active_class_list_client].sum()/(self.args.batch_size * self.args.annotation_num)  # supervised_loss
                if len(idx) == 0 or len(negetive_class_list_client) == 0:
                    loss = loss_sup
                else:
                    loss = bce_criterion_unsup(logits_strong, logits_weak_sig)
                    loss_unsup = loss[idx, :][:, negetive_class_list_client].sum()/(len(idx) * (self.args.n_classes - self.args.annotation_num))  # supervised_loss
                    # print('loss_sup: ', loss_sup)
                    # print('loss_unsup: ', loss_unsup)
                    loss = loss_sup + 1. * loss_unsup
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
        net.cpu()
        self.optimizer.zero_grad()
        return net.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

    def mixup_criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: (lam * criterion(pred, y_a).T).T + ((1 - lam) * criterion(pred, y_b).T).T

    def test_loss(self, rnd, net, model_name):
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

    def find_indices_in_a(self, a, b):
        return torch.where(a.unsqueeze(0) == b.unsqueeze(1))[1]

    def train_FedMLP(self, rnd, tao, Prototype, writer1, negetive_class_list, active_class_list_client_i, net):    # my method
        # assert len(self.ldr_train.dataset) == len(self.idxs)
        # print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        if rnd < self.args.rounds_FedMLP_stage1:  # stage1
            glob_model = deepcopy(net)
            net.train()
            glob_model.eval()
            # set the optimizer
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            # train and update
            epoch_loss = []
            print(self.loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                                 reduction='none')  # include sigmoid
            bce_criterion_unsup = nn.MSELoss()
            for epoch in range(self.args.local_ep):
                print('local_epoch:', epoch)
                batch_loss = []
                for j, (samples, item, active_class_list) in enumerate(self.ldr_train):
                    if j == 0:
                        active_class_list_client = []
                        negetive_class_list_client = []
                        for i in range(self.args.annotation_num):
                            active_class_list_client.append(active_class_list[i][0].item())
                        for i in range(self.args.n_classes):
                            if i not in active_class_list_client:
                                negetive_class_list_client.append(i)
                                self.class_num_list[i] = 0  # try noro
                    criterion = LogitAdjust_Multilabel(cls_num_list=self.class_num_list, num=len(self.idxs))
                    mse_loss = nn.MSELoss(reduction='none')
                    images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(
                        self.args.device), samples["target"].to(self.args.device)
                    fe1, logits1 = net(images1)
                    logits1_sig = torch.sigmoid(logits1).cuda()
                    _, logits2 = net(images2)
                    logits2_sig = torch.sigmoid(logits2).cuda()
                    # loss_sup1 = bce_criterion_sup(logits1, labels)  # tensor(32, 5)
                    # loss_sup2 = bce_criterion_sup(logits2, labels)  # tensor(32, 5)
                    with torch.no_grad():
                        _, outputs_global = glob_model(images1)
                        logits3 = torch.sigmoid(outputs_global).cuda()
                        _, outputs_global = glob_model(images2)
                        logits4 = torch.sigmoid(outputs_global).cuda()
                    loss_dis1 = mse_loss(logits1_sig, logits3).cuda()
                    loss_dis2 = mse_loss(logits2_sig, logits4).cuda()
                    loss_dis = (loss_dis1 + loss_dis2) / 2.
                    loss_sup1 = criterion(logits1_sig, labels)  # tensor(32, 5)
                    loss_sup2 = criterion(logits2_sig, labels)  # tensor(32, 5)
                    loss_sup = (loss_sup1 + loss_sup2) / 2.
                    # loss_sup = loss_sup.sum() / (self.args.batch_size * self.args.n_classes)  # supervised_loss

                    loss_sup = loss_sup[:, active_class_list_client].sum() / (
                                self.args.batch_size * self.args.annotation_num)  # supervised_loss
                    loss_dis = loss_dis[:, negetive_class_list_client].sum() / (
                                self.args.batch_size * len(negetive_class_list_client))  # supervised_loss

                    loss_unsup = bce_criterion_unsup(logits1_sig[:, negetive_class_list_client],
                                          logits2_sig[:, negetive_class_list_client])
                    loss = loss_sup + 0.0*loss_unsup + loss_dis
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    self.iter_num += 1
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
            if rnd == self.args.rounds_FedMLP_stage1 - 1:    # first tao and proto
                print('client: ', self.client_id, active_class_list_client)
                proto = torch.zeros((self.args.n_classes * 2, len(fe1[0]))) # [cls0proto0, cls0proto1, cls1proto0...]
                # proto = np.array([torch.zeros_like(fe1[0].cpu())] * self.args.n_classes * 2)
                num_proto = [0] * self.args.n_classes * 2
                t = np.array([0] * self.args.n_classes)
                test_loader = DataLoader(dataset=self.local_dataset, batch_size=self.args.batch_size * 4, shuffle=False,
                                         num_workers=8)
                net.eval()
                with torch.no_grad():
                    for samples, _, _ in test_loader:
                        images1, labels = samples["image_aug_1"].to(self.args.device), samples["target"].to(self.args.device)
                        feature, outputs = net(images1)
                        probs = torch.sigmoid(outputs)  # soft predict
                        for cls in active_class_list_client:
                            idx0 = torch.where(labels[:, cls] == 0)[0].tolist()
                            idx1 = torch.where(labels[:, cls] == 1)[0].tolist()
                            num_proto[2*cls] += len(idx0)
                            num_proto[2*cls+1] += len(idx1)
                            proto[2*cls] = feature[idx0, :].sum(0).cpu() + proto[2*cls]
                            proto[2*cls+1] = feature[idx1, :].sum(0).cpu() + proto[2*cls+1]
                            # t[cls] += torch.sum(probs[idx0, cls] < self.args.L).item() + torch.sum(
                            #     probs[idx1, cls] > self.args.U).item()
                        for cls in negetive_class_list:
                            t[cls] += torch.sum(
                                torch.logical_or(probs[:, cls] < self.args.L, probs[:, cls] > self.args.U)).item()
                for cls in active_class_list_client:
                    proto[2*cls] = proto[2*cls] / num_proto[2*cls]
                    proto[2*cls+1] = proto[2*cls+1] / num_proto[2*cls+1]
                t = t / len(self.local_dataset)
                print('local_t: ', t)
                return net.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client, t, proto
            else:
                return net.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

        else:  # stage2
            print(self.local_dataset.active_class_list)
            # find train samples for each class
            idx = []    # [[negcls1], [negcls2]]
            idxss = []
            feature = []    
            similarity = []     
            clean_idx = []  
            noise_idx = []  
            label = []  # [[negcls1], [negcls2]]
            num_train = 0
            glob_model = deepcopy(net)
            net.eval()
            class_idx = torch.tensor([])
            l = torch.tensor([]).cuda()
            f = torch.tensor([]).cuda()
            t1 = time.time()
            if rnd == self.args.rounds_FedMLP_stage1:
                # print(len(self.ldr_train.dataset))
                self.traindata_idx = []  # [[negcls1_clean_train_idx], [negcls1_noise_train_idx], [negcls2_clean_train_idx], [negcls2_noise_train_idx]] idx
                for samples, item, active_class_list in self.ldr_train:
                    class_idx = torch.cat((class_idx, item), dim=0)
                    images1, labels = samples["image_aug_1"].to(self.args.device), samples["target"].to(self.args.device)
                    with torch.no_grad():
                        features, _ = net(images1)
                    f = torch.cat((f, features), dim=0)
                    l = torch.cat((l, labels), dim=0)
                for i in range(len(negetive_class_list)):
                    feature.append(f)   # miss n classes
                    idx.append(class_idx)
                    label.append(l)
            else:
                for samples, item, active_class_list in self.ldr_train:
                    class_idx = torch.cat((class_idx, item), dim=0)
                    images1, labels = samples["image_aug_1"].to(self.args.device), samples["target"].to(self.args.device)
                    with torch.no_grad():
                        features, _ = net(images1)
                    f = torch.cat((f, features), dim=0)
                    l = torch.cat((l, labels), dim=0)
                for i in range(len(self.idxss)):
                    result_indices = self.find_indices_in_a(class_idx, torch.tensor(self.idxss[i]))
                    feature.append(f[result_indices])
                    idx.append(class_idx[result_indices])
                    label.append(l[result_indices])
            t2 = time.time()
            # print('feature_label_prepare_time: ', t2-t1)    
            for i, cls in enumerate(negetive_class_list):
                # sim = []
                proto_0 = Prototype[2*cls]
                proto_1 = Prototype[2*cls+1]
                model = CosineSimilarityFast().cuda()
                sim = (model(feature[i], torch.unsqueeze(proto_0.cuda(), dim=0)) - model(feature[i], torch.unsqueeze(proto_1.cuda(), dim=0))).tolist()
                similarity.append(sim)
            t3 = time.time()
            # print('sim_compute_time: ', t3 - t2)
            for i in range(len(negetive_class_list)):
                idx_0 = np.where(np.array(similarity[i]) >= 0)[0]
                idx_1 = np.where(np.array(similarity[i]) < 0)[0]
                clean_idx.append(idx_0.tolist())
                noise_idx.append(idx_1.tolist())
            if rnd == self.args.rounds_FedMLP_stage1:
                for i, cls in enumerate(negetive_class_list):
                    print('cls', cls, 'tao: ', tao[cls])
                    num_clean_cls = int(1 * self.args.clean_threshold * len(clean_idx[i]))
                    num_noise_cls = int(1 * self.args.noise_threshold * len(noise_idx[i]))
                    # num_clean_cls = int(tao[cls] * len(clean_idx[i]))
                    # num_noise_cls = int(tao[cls] * len(noise_idx[i]))
                    num_train = num_train + num_noise_cls + num_clean_cls
                    max_m_indices_list = np.array(max_m_indices(similarity[i], num_clean_cls))   
                    min_n_indices_list = np.array(min_n_indices(similarity[i], num_noise_cls))
                    if len(max_m_indices_list) == 0 and len(max_m_indices_list) == 0:
                        negcls_clean_train_idx = []
                        negcls_noise_train_idx = []
                    elif len(min_n_indices_list) == 0 and len(max_m_indices_list) != 0:
                        negcls_noise_train_idx = []
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                    elif len(min_n_indices_list) != 0 and len(max_m_indices_list) == 0:
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                        negcls_clean_train_idx = []
                    else:
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                    self.traindata_idx.append(negcls_clean_train_idx)
                    self.traindata_idx.append(negcls_noise_train_idx)  
            else:
                for i, cls in enumerate(negetive_class_list):
                    print('cls', cls, 'tao: ', tao[cls])
                    num_clean_cls = int(1 * self.args.clean_threshold * len(clean_idx[i]))
                    num_noise_cls = int(1 * self.args.noise_threshold * len(noise_idx[i]))
                    num_train = num_train + num_noise_cls + num_clean_cls
                    max_m_indices_list = np.array(max_m_indices(similarity[i], num_clean_cls)) 
                    min_n_indices_list = np.array(min_n_indices(similarity[i], num_noise_cls))

                    if len(max_m_indices_list) == 0 and len(max_m_indices_list) == 0:
                        negcls_clean_train_idx = []
                        negcls_noise_train_idx = []
                    elif len(min_n_indices_list) == 0 and len(max_m_indices_list) != 0:
                        negcls_noise_train_idx = []
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                    elif len(min_n_indices_list) != 0 and len(max_m_indices_list) == 0:
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                        negcls_clean_train_idx = []
                    else:
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                    self.traindata_idx[2*i].extend(negcls_clean_train_idx)
                    self.traindata_idx[2*i+1].extend(negcls_noise_train_idx) 

            t4 = time.time()
            # print('traindata_split_time: ', t4 - t3)

            for i, cls in enumerate(negetive_class_list):   
                print('class: ', cls, 'clean_train_samples: ', len(self.traindata_idx[2*i]))
                print('class: ', cls, 'noise_train_samples: ', len(self.traindata_idx[2 * i+1]))
                self.class_num_list[cls] = len(self.traindata_idx[2 * i+1]) 
                # if rnd % 10 == 0:
                #     real_clean = 0
                #     real_noise = 0
                #     # print(self.dataset[self.traindata_idx[2*i]]['target'][cls])
                #     # print(type(self.dataset[self.traindata_idx[2*i]]['target'][cls]))
                #     # input()
                #     for j in self.traindata_idx[2*i]:
                #         if self.dataset[j]['target'][cls] == 0:
                #             real_clean += 1
                #     print('clean_acc: ', real_clean/len(self.traindata_idx[2*i]))
                #     for j in self.traindata_idx[2*i+1]:
                #         if self.dataset[j]['target'][cls] == 1:
                #             real_noise += 1
                #     if len(self.traindata_idx[2 * i+1]) == 0:
                #         print('noise_acc: ', 0)
                #         writer1.add_scalar(f'noise_acc/client{self.client_id}/class{cls}',
                #                            0, rnd)
                #     else:
                #         print('noise_acc: ', real_noise/len(self.traindata_idx[2*i+1]))
                #         writer1.add_scalar(f'noise_acc/client{self.client_id}/class{cls}',
                #                            real_noise / len(self.traindata_idx[2 * i + 1]), rnd)
                #     writer1.add_scalar(f'clean_acc/client{self.client_id}/class{cls}', real_clean/len(self.traindata_idx[2*i]), rnd)
            t5 = time.time()
            # print('acc_compute_time: ', t5 - t4)   
            # train
            net.train()
            glob_model.eval()
            # set the optimizer
            self.optimizer = torch.optim.Adam(
                net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            # train and update
            epoch_loss = []
            loss_w = self.loss_w
            for i, cls in enumerate(negetive_class_list):
                if len(self.traindata_idx[2*i+1]) != 0:
                    loss_w[cls] = len(self.traindata_idx[2*i]) / len(self.traindata_idx[2*i+1])  
                else:
                    loss_w[cls] = 5.0
            print(loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_w).cuda(),
                                                     reduction='none')  # include sigmoid
            mse_loss = nn.MSELoss(reduction='none')

            for epoch in range(self.args.local_ep):
                print('local_epoch:', epoch)
                batch_loss = []
                dataset = DatasetSplit_pseudo(self.dataset, self.idxs, self.client_id, self.args,
                                       active_class_list_client_i, negetive_class_list, self.traindata_idx)
                dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                                 num_workers=8)
                for samples, item, distill_cls in dataloader:
                    distill_cls = distill_cls.cuda()
                    sup_cls = (~distill_cls.bool()).float().cuda()
                    criterion = LogitAdjust_Multilabel(cls_num_list=self.class_num_list,
                                                       num=len(self.idxs))
                    images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(self.args.device), samples["target"].to(
                            self.args.device)
                    feature, outputs = net(images1)
                    logits1 = torch.sigmoid(outputs).cuda()
                    with torch.no_grad():
                        _, outputs_global = glob_model(images1)
                        logits2 = torch.sigmoid(outputs_global).cuda()
                    # loss_sup = bce_criterion_sup(outputs, labels).cuda()
                    loss_sup = criterion(logits1, labels).cuda()
                    loss_dis = mse_loss(logits1, logits2).cuda()

                    # loss = ((loss_sup * sup_cls).sum() + (loss_dis * distill_cls).sum()) / (sup_cls.sum() + distill_cls.sum())
                    loss = (loss_sup * sup_cls).sum() / sup_cls.sum()
                    # print(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    self.iter_num += 1
                self.epoch = self.epoch + 1
                epoch_loss.append(np.array(batch_loss).mean())
            for i in range(len(self.traindata_idx) // 2):
                idxs = self.traindata_idx[2 * i] + self.traindata_idx[2 * i + 1]
                idxss.append(idxs)
            t6 = time.time()
            # print('local_train_time: ', t6 - t5)
            self.idxss = idxss
            for i in range(len(idxss)):
                self.idxss[i] = list(set(self.idxs)-set(idxss[i]))  # [[negcls1_else_idx], [negcls2_else_idx]]

            # proto = np.array(
            #     [torch.zeros_like(feature[0].cpu())] * self.args.n_classes * 2)  # [cls0proto0, cls0proto1, cls1proto0...]
            proto = torch.zeros((self.args.n_classes * 2, len(feature[0])))  # [cls0proto0, cls0proto1, cls1proto0...]
            num_proto = [0] * self.args.n_classes * 2
            t = np.array([0] * self.args.n_classes)
            # test_idx = set()   # partial_proto
            # for i in range(len(self.traindata_idx) // 2):
            #     test_idx = test_idx.union(set(self.traindata_idx[2 * i]))
            #     test_idx = test_idx.union(set(self.traindata_idx[2 * i + 1]))
            # test_dataset = DatasetSplit(self.dataset, list(test_idx), self.client_id, self.args, self.class_neg_idx,
            #                        active_class_list_client_i)
            # test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size * 4, shuffle=False,
            #                          num_workers=8)

            test_loader = DataLoader(dataset=self.local_dataset, batch_size=self.args.batch_size * 4, shuffle=False,
                                     num_workers=8)
            net.eval()
            with torch.no_grad():
                for samples, _, _ in test_loader:
                    images1, labels = samples["image_aug_1"].to(self.args.device), samples["target"].to(
                        self.args.device)
                    feature, outputs = net(images1)
                    probs = torch.sigmoid(outputs)  # soft predict
                    for cls in self.local_dataset.active_class_list:
                        idx0 = torch.where(labels[:, cls] == 0)[0]
                        idx1 = torch.where(labels[:, cls] == 1)[0]
                        num_proto[2 * cls] += len(idx0)
                        num_proto[2 * cls + 1] += len(idx1)
                        proto[2 * cls] += feature[idx0, :].sum(0).cpu()
                        proto[2 * cls + 1] += feature[idx1, :].sum(0).cpu()
                        # t[cls] += torch.sum(probs[idx0, cls] < self.args.L).item() + torch.sum(
                        #     probs[idx1, cls] > self.args.U).item()
                    for cls in negetive_class_list:
                        t[cls] += torch.sum(torch.logical_or(probs[:, cls] < self.args.L, probs[:, cls] > self.args.U)).item()
            for cls in self.local_dataset.active_class_list:
                if num_proto[2 * cls] == 0:
                    proto[2 * cls] = proto[2 * cls]
                else:
                    proto[2 * cls] = proto[2 * cls] / num_proto[2 * cls]
                if num_proto[2 * cls+1] == 0:
                    proto[2 * cls+1] = proto[2 * cls+1]
                else:
                    proto[2 * cls+1] = proto[2 * cls+1] / num_proto[2 * cls+1]
            t = t / len(self.local_dataset)
            print('local_t: ', t)
            net.cpu()
            self.optimizer.zero_grad()
            t7 = time.time()
            print('local_test_proto_time: ', t7 - t6)   
            return net.state_dict(), np.array(
                epoch_loss).mean(), _, _, negetive_class_list, self.local_dataset.active_class_list, t, proto

    def js(self, p_output, q_output):
        """
        :param predict: model prediction for original data
        :param target: model prediction for mildly augmented data
        :return: loss
        """
        KLDivLoss = nn.KLDivLoss(reduction='mean')
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

    def anti_sigmoid(self, p):
        return torch.log(p / (1 - p))
    def train_FedLSR(self, rnd, net):
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        # train and update
        epoch_loss = []
        print(self.loss_w)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda())  # include sigmoid
        for epoch in range(self.args.local_ep):
            print('local_epoch:', epoch)
            batch_loss = []
            for samples, item, active_class_list in self.ldr_train:
                active_class_list_client = []
                negetive_class_list_client = []
                for i in range(self.args.annotation_num):
                    active_class_list_client.append(active_class_list[i][0].item())
                for i in range(self.args.n_classes):
                    if i not in active_class_list_client:
                        negetive_class_list_client.append(i)
                images1, images2, labels = samples["image_aug_1"].to(self.args.device), samples["image_aug_2"].to(self.args.device), samples["target"].to(self.args.device)

                _, logits1_ori = net(images1)
                _, logits2_ori = net(images2)
                mix_1 = np.random.beta(1, 1)  # mixing predict1 and predict2
                mix_2 = 1 - mix_1
                # to further conduct self distillation, *3 means the temperature T_d is 1/3
                logits1 = torch.sigmoid(logits1_ori * 3).cuda()
                logits2 = torch.sigmoid(logits2_ori * 3).cuda()

                # for training stability to conduct clamping to avoid exploding gradients, which is also used in Symmetric CE, ICCV 2019
                logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0)

                # to mix up the two predictions
                p = torch.sigmoid(logits1_ori) * mix_1 + torch.sigmoid(logits2_ori) * mix_2
                p = self.anti_sigmoid(p)
                pred_mix = torch.sigmoid(p * 2).cuda()
                betaa = 0.4
                if (rnd < self.args.t_w):
                    betaa = 0.4 * rnd / self.args.t_w
                loss = bce_criterion(pred_mix, labels)  # to compute cross entropy loss
                # print(loss)
                loss += self.js(logits1, logits2) * betaa
                # print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
        net.cpu()
        self.optimizer.zero_grad()
        return net.state_dict(), np.array(epoch_loss).mean(), _, _, negetive_class_list_client, active_class_list_client

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, client_id, args, class_neg_idx, active_class_list=None, negative_class_list=None, corr_idx=None):
        self.dataset = dataset
        self.negative_class_list = negative_class_list
        self.corr_idx = corr_idx
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
        if self.corr_idx is not None:
            for i, class_id in enumerate(self.negative_class_list):
                if self.idxs[item] in self.corr_idx[i]:
                    sample['target'][class_id] = 1
        return sample, self.idxs[item], self.active_class_list

    def get_num_of_each_class(self, args):
        class_sum = np.array([0.] * args.n_classes)
        for idx in self.idxs:
            class_sum += self.dataset.targets[idx]
        return class_sum.tolist()


class DatasetSplit_Mixup(Dataset):
    def __init__(self, dataset, clean_idxs, noise_idxs, args, negative_class, negative_class_list, train_ratio):
        self.dataset = dataset
        self.negative_class = negative_class
        self.clean_idxs = clean_idxs
        self.noise_idxs = noise_idxs
        self.annotation_num = args.annotation_num
        self.negative_class_list = negative_class_list
        self.train_ratio = train_ratio

    def __len__(self):
        if self.train_ratio < 1:
            return int(self.train_ratio * (len(self.clean_idxs) + len(self.noise_idxs)))
        else:
            return int(len(self.clean_idxs) + len(self.noise_idxs))

    def __getitem__(self, item):
        if self.train_ratio < 1:
            item = int(item / self.train_ratio)
        if item < len(self.clean_idxs): # clean sample mixup
            flag = 0
            index = random.choice(self.clean_idxs)  
            sample1 = deepcopy(self.dataset[self.clean_idxs[item]])
            sample2 = deepcopy(self.dataset[index])
            for i in range(len(sample1['target'])):
                if i in self.negative_class_list:
                    sample1['target'][i] = 0
                    sample2['target'][i] = 0
            mixed_x, lam = self.mixup_data(sample1["image_aug_1"], sample2["image_aug_1"])
        else:   # noise sample mixup
            flag = 1
            index = random.choice(self.noise_idxs)  
            sample1 = self.dataset[self.noise_idxs[item - len(self.clean_idxs)]]
            sample2 = self.dataset[index]
            for i in range(len(sample1['target'])):
                if i in self.negative_class_list:
                    sample1['target'][i] = 0
                    sample2['target'][i] = 0
            mixed_x, lam = self.mixup_data(sample1["image_aug_1"], sample2["image_aug_1"])
            sample1['target'][self.negative_class] = 1
            sample2['target'][self.negative_class] = 1
        return mixed_x, lam, flag, sample1, sample2

    def mixup_data(self, x1, x2, alpha=1.0):
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, lam

class CosineSimilarityFast(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityFast, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        # print(x1.shape)
        # print(x2)
        # print(x2.shape)
        # input()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final = x.mul(1/x_frobenins)
        final = torch.squeeze(final, dim=1)
        return final

class DatasetSplit_pseudo(Dataset):
    def __init__(self, dataset, idxs, client_id, args, active_class_list, negative_class_list, traindata_idx):
        self.dataset = dataset
        self.negative_class_list = negative_class_list
        self.idxs = list(idxs)
        self.client_id = client_id  # choose active classes
        self.annotation_num = args.annotation_num
        self.active_class_list = active_class_list
        self.traindata_idx = traindata_idx
        self.args = args
        self.idx_conf = []
        for i in range(len(traindata_idx) // 2):
            self.idx_conf += (traindata_idx[2 * i] + traindata_idx[2 * i + 1])
        self.idx_nconf = list(set(self.idxs) - set(self.idx_conf))
        logging.info(f"Client ID: {self.client_id}, active_class_list: {self.active_class_list}")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        distill_cls = torch.zeros(self.args.n_classes)
        sample = self.dataset[self.idxs[item]]
        for i in range(len(sample['target'])):
            if i not in self.active_class_list:
                sample['target'][i] = 0
        for i in range(len(self.traindata_idx)//2):
            idx0 = self.traindata_idx[2*i]
            idx1 = self.traindata_idx[2*i+1]
            if self.idxs[item] in (idx0+idx1):
                if self.idxs[item] in idx1:
                    sample['target'][self.negative_class_list[i]] = 1
            else:
                distill_cls[self.negative_class_list[i]] = 1
        # if self.idxs[item] in self.idx_nconf:    # mixup
        #     mixed_sample = copy.deepcopy(sample)
        #     index = random.choice(self.idx_nconf)
        #     sample2 = self.dataset[index]
        #     mixed_sample["image_aug_1"], lam = self.mixup_data(sample["image_aug_1"], sample2["image_aug_1"])
        #     mixed_sample['target'] = lam * sample['target'] + (1 - lam) * sample2['target']
        #     return mixed_sample, self.idxs[item], distill_cls
        return sample, self.idxs[item], distill_cls


    def get_num_of_each_class(self, args):
        class_sum = np.array([0.] * args.n_classes)
        for idx in self.idxs:
            class_sum += self.dataset.targets[idx]
        return class_sum.tolist()
    def mixup_data(self, x1, x2, alpha=1.0):
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, lam
