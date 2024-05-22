import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_num_of_each_class(args, num_samples_to_split, test_dataset):
    class_sum = np.array([0.] * args.n_classes)
    for idx in range(num_samples_to_split):
        class_sum += test_dataset.targets[idx]
    return class_sum.tolist()

def valloss(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=4)
    # 定义要划分的样本比例（例如，划分出20%的样本）
    split_ratio = 0.1

    # 计算划分的数量
    num_samples = len(test_loader.dataset)
    num_samples_to_split = int(num_samples * split_ratio)

    # 创建一个随机采样器，用于划分样本
    random_sampler = SubsetRandomSampler(range(num_samples_to_split))

    # 创建一个新的dataloader，其中包含随机划分的样本
    val_data_loader = DataLoader(
        dataset=test_loader.dataset,
        batch_size=args.batch_size*4,
        sampler=random_sampler
    )
    class_num_list = get_num_of_each_class(args, num_samples_to_split, test_dataset)
    loss_w = [num_samples_to_split / i for i in class_num_list]
    print(class_num_list, num_samples_to_split, loss_w)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_w).cuda())  # include sigmoid
    batch_loss = []
    with torch.no_grad():
        for samples in val_data_loader:
            images, labels = samples["image"].to(args.device), samples["target"].to(args.device)
            _, outputs = net(images)
            val_loss = bce_criterion(outputs, labels)
            batch_loss.append(val_loss.item())
    val_loss_mean = np.array(batch_loss).mean()
    logging.info(val_loss_mean)
    return val_loss_mean

