import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjust_Multilabel(nn.Module):
    def __init__(self, cls_num_list, num, tau=1, weight=None):
        super(LogitAdjust_Multilabel, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        self.cls_p_list = cls_num_list / num
        self.weight = weight

    def forward(self, x, target):
        x_m = x.clone()
        # for i in range(len(self.cls_p_list)): #abu1
        #     x_m[:, i] = (x_m[:, i]*self.cls_p_list[i])/(x_m[:, i]*self.cls_p_list[i] + (1-x_m[:, i])*(1-self.cls_p_list[i]))
        # nan_mask = torch.isnan(x_m)
        # x_m[nan_mask] = 0.
        return F.binary_cross_entropy(x_m, target, weight=self.weight, reduction='none')


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, num, active_class_list_client, negative_class_list_client, tau=1, weight=None):
        super(LA_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        self.active_class_list_client = active_class_list_client
        self.negative_class_list_client = negative_class_list_client
        self.cls_p_list = cls_num_list / num
        self.weight = weight
        self.bce = LogitAdjust_Multilabel(cls_num_list, num)

    def forward(self, x, target, soft_target, w_kd):
        bceloss = self.bce(x, target)[:, self.active_class_list_client].sum()/(len(x) * len(self.active_class_list_client))
        kl = F.mse_loss(x, soft_target, reduction='none')[:, self.negative_class_list_client].sum()/(len(x) * len(self.negative_class_list_client))
        return w_kd * kl + (1 - w_kd) * bceloss


def get_output(loader, net, args, sigmoid=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, samples in enumerate(loader):
            images = samples["image"].to(args.device)
            labels = samples["target"].to(args.device)
            if sigmoid == True:
                _, outputs = net(images)
                outputs = torch.sigmoid(outputs)
            else:
                _, outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)


def DaAgg(w, dict_len, clean_clients, noisy_clients):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()
    distance = np.zeros(len(dict_len))
    for n_idx in noisy_clients:
        dis = []
        for c_idx in clean_clients:
            dis.append(model_dist(w[n_idx], w[c_idx]))
        distance[n_idx] = min(dis)
    distance = distance / distance.max()
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    # print(client_weight)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()