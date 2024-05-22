# This file is borrowed from https://github.com/Xu-Jingyi/FedCorr/blob/main/util/sampling.py

import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train / num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)]  # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idxs, num_items, replace=False))  # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs) - dict_users[i])

    for key in dict_users.keys():
        dict_users[key] = list(dict_users[key])
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]  # indicate the clients that choose each class
    num_clients_per_class = np.array([len(x) for x in Psi])  # 每个类的客户端数量
    dict_users = {}
    for class_i in range(num_classes+1):
        # all_idxs = np.where(y_train == class_i)[0]
        n_classes_per_sample = np.sum(y_train, axis=1)
        all_idxs = np.where(n_classes_per_sample == class_i)[0]
        # else_idxs = np.where(y_train[class_i*25907:(class_i+1)*25907, class_i] != 1)[0] + class_i*25907
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[0])
        assignment = np.random.choice(Psi[0], size=len(all_idxs), p=p_dirichlet.tolist())
        # assignment_else = np.random.choice(Psi[class_i], size=len(else_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[0]:
            if client_k in dict_users:
                # dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]) | set(else_idxs[(assignment_else == client_k)]))
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                # dict_users[client_k] = set(np.concatenate((all_idxs[(assignment == client_k)], else_idxs[(assignment_else == client_k)])))
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
    for key in dict_users.keys():
        dict_users[key] = list(dict_users[key])
    return dict_users