import copy
import math

import torch
import numpy as np

def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg

def Fed_w(w, weight):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weight[i]
        w_avg[k] = w_avg[k] / sum(weight)
    return w_avg

def RSCFed(DMA, w_locals, K, dict_len, M):
    w_sub = []
    for group in DMA:
        w_select = []
        N_total = 0
        for id in group:
            w_select.append(w_locals[id])
            N_total += dict_len[id]
        w_avg = Fed_w(w_select, [1]*K)
        w = []
        for id in group:
            a = dict_len[id] / N_total
            b = math.exp((-0.01)*(model_dist(w_locals[id], w_avg)/dict_len[id]))
            w.append(a*b)
        w_sub.append(Fed_w(w_select, w))
    w_glob = Fed_w(w_sub, [1]*M)
    return w_glob

def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1:
        dist = torch.norm(w_1[key].cpu() - w_2[key].cpu())
        dist_total += dist.cpu()
    return dist_total.cpu().item()

def FedAvg_tao(t, weight, class_active_client_list = None):
    if class_active_client_list is None:
        t_avg = np.array([0.]*len(t[0]))
        for i, tao in enumerate(t):
            t_avg += tao * float(weight[i])
        t_avg = t_avg / float(sum(weight))
        return t_avg
    else:
        t_avg = np.array([0.] * len(t[0]))
        for cls, cls_active_clients in enumerate(class_active_client_list):
            weight_sum = 0.
            for i, tao in enumerate(t):
                if i in cls_active_clients:
                    t_avg[cls] += tao[cls] * float(weight[i])
                    weight_sum += float(weight[i])
            if len(cls_active_clients) == 0:
                t_avg[cls] = 1.
            else:
                t_avg[cls] = t_avg[cls] / weight_sum
        return t_avg

def FedAvg_proto(Prototypes, weight, class_active_client_list):
    Prototype_avg = torch.zeros((len(Prototypes[0]), len(Prototypes[0][0])))
    # Prototype_avg = np.array([torch.zeros_like(Prototypes[0][0])] * len(Prototypes[0]))
    for cls, cls_active_clients in enumerate(class_active_client_list):
        Prototype_class_0_avg = torch.zeros_like(Prototypes[0][0])
        Prototype_class_1_avg = torch.zeros_like(Prototypes[0][0])
        for client_id in cls_active_clients:
            Prototype_class_0_avg = Prototypes[client_id][2*cls] * weight[client_id] + Prototype_class_0_avg
            Prototype_class_1_avg = Prototypes[client_id][2*cls+1] * weight[client_id] + Prototype_class_1_avg
        # print(cls)
        # print(cls_active_clients)
        # print(Prototype_class_0_avg)
        # print(Prototype_class_1_avg)
        Prototype_class_0_avg = Prototype_class_0_avg / np.sum(np.array(weight)[cls_active_clients])
        Prototype_class_1_avg = Prototype_class_1_avg / np.sum(np.array(weight)[cls_active_clients])
        # print(Prototype_class_0_avg)
        # print(Prototype_class_1_avg)
        Prototype_avg[2*cls] = Prototype_class_0_avg
        Prototype_avg[2*cls+1] = Prototype_class_1_avg
        # print(Prototype_avg)
        # input()
    return Prototype_avg

def FedAvg_rela(Prototypes, weight, class_active_client_list):
    Prototype_avg = torch.zeros((len(Prototypes[0]), len(Prototypes[0][0])))
    for cls, cls_active_clients in enumerate(class_active_client_list):
        Prototype_class_avg = torch.zeros_like(Prototypes[0][0])
        for client_id in cls_active_clients:
            Prototype_class_avg = Prototypes[client_id][cls] * weight[client_id] + Prototype_class_avg
        Prototype_class_avg = Prototype_class_avg / np.sum(np.array(weight)[cls_active_clients])
        Prototype_avg[cls] = Prototype_class_avg
    return Prototype_avg