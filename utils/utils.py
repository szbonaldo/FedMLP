import logging
import os
import random
import shutil
import sys
import heapq
import numpy as np
import torch
from tensorboardX import SummaryWriter


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# def max_m_indices(arr, m):
#     max_m_values = heapq.nlargest(m, arr)
#     max_m_indices = [arr.index(value) for value in max_m_values]
#     return max_m_indices
def max_m_indices(lst, n):
    elements_with_indices = list(enumerate(lst))
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1], reverse=True)
    top_n_elements = sorted_elements[:n]
    return [index for index, value in top_n_elements]


def min_n_indices(lst, n):
    elements_with_indices = list(enumerate(lst))
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1])
    bottom_n_elements = sorted_elements[:n]
    return [index for index, value in bottom_n_elements]
# def min_n_indices(arr, n):
#     min_n_values = heapq.nsmallest(n, arr)
#     min_n_indices = [arr.index(value) for value in min_n_values]
#     return min_n_indices


def set_output_files(args):
    outputs_dir = 'outputs_' + str(args.dataset) + '_' + str(
        args.alpha_dirichlet) + '_' + str(args.n_clients) + '_' + str(args.model) + '_' + str(args.n_classes-args.annotation_num) + '5000classmiss7_1037_iid_FedMLP_stage1glo'
    # outputs_dir = 'outputs_' + str(args.dataset) + '_' + str(args.model) + '_dataaug_' + str(
    #     args.n_classes - args.annotation_num) + 'classmiss_' + 'loss_distribution'    # demo
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp + '_' +
                           str(args.batch_size) + '_' + str(args.base_lr) + '_' +
                           str(args.rounds_warmup) + '_' +
                           str(args.rounds_corr) + '_' +
                           str(args.rounds_finetune) + '_' + str(args.local_ep))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    # if not os.path.exists(tensorboard_dir):
    #     os.mkdir(tensorboard_dir)
    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    os.mkdir(code_dir)
    # shutil.make_archives(code_dir, 'zip', base_dir='/home/szb/multilabel/')

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer1 = SummaryWriter(tensorboard_dir + 'writer1')
    return writer1, models_dir
