import logging

import numpy as np

import torch
import torch.optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, roc_curve, auc
from sklearn.metrics import average_precision_score

from utils.multilabel_metrixs import Recall, Hamming_Loss, F1Measure, Precision, BACC


def globaltest(net, test_dataset, args):
    auroc = 0
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=4)
    all_preds = np.array([])
    all_probs = []
    all_labels = np.array(test_dataset.targets)
    with torch.no_grad():
        for samples in test_loader:
            images = samples["image"].to(args.device)
            _, outputs = net(images)
            probs = torch.sigmoid(outputs)  # soft predict
            accuracy_th = 0.5
            preds = probs > accuracy_th  # hard predict
            all_probs.append(probs.detach().cpu())
            if all_preds.ndim == 1:
                all_preds = preds.detach().cpu().numpy()
            else:
                all_preds = np.concatenate([all_preds, preds.detach().cpu().numpy()], axis=0)

    all_probs = torch.cat(all_probs).numpy()
    assert all_probs.shape[0] == len(test_dataset)
    assert all_probs.shape[1] == args.n_classes
    logging.info(np.sum(all_preds, axis=0))
    logging.info(np.sum(all_labels, axis=0))

    # 初始化用于存储每个标签的AP的列表
    APs = []

    # 循环计算每个标签的AP
    for label_index in range(all_labels.shape[1]):
        true_labels = all_labels[:, label_index]
        predicted_scores = all_probs[:, label_index]

        # 计算平均精度（Average Precision）并添加到列表中
        ap = average_precision_score(true_labels, predicted_scores)
        APs.append(ap)

    # 计算mAP
    mAP = torch.tensor(APs).mean()

    bacc = BACC(all_labels, all_preds)
    R = Recall(all_labels, all_preds)
    hamming_loss = Hamming_Loss(all_labels, all_preds)
    F1 = F1Measure(all_labels, all_preds)
    P = Precision(all_labels, all_preds)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000', '#808080', '#C0C0C0', '#800000',
              '#008000', '#000080', '#808000', '#008080']
    # print(all_probs)
    for i in range(len(all_labels.T)):    # 有问题，已修改，应该按照每一类平均而不是每个样本
        fpr, tpr, th = roc_curve(all_labels.T[i], all_probs.T[i], pos_label=1)
        # ROCprint(fpr, tpr, i, colors[i])
        auroc += auc(fpr, tpr)
        # print('class: {}, auc: '.format(i), auc(fpr, tpr))
    # plt.show()
    auroc /= len(all_labels.T)

    return {"mAP": mAP,
            "BACC": bacc,
            "R": R,
            "F1": F1,
            "auc": auroc,
            "P": P,
            "hamming_loss": hamming_loss}


def ROCprint(fpr, tpr, name, colorname):
    plt.plot(fpr, tpr, lw=1, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
    plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('ROC Curve', fontsize=25)
    plt.legend(loc='lower right', fontsize=20)
    plt.savefig('multi_models_roc.png')


def classtest(net, test_dataset, args, classid):
    auroc = 0
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=4)
    all_preds = np.array([])
    all_probs = []
    all_labels = np.array(test_dataset.targets)
    with torch.no_grad():
        for samples in test_loader:
            images = samples["image"].to(args.device)
            _, outputs = net(images)
            probs = torch.sigmoid(outputs)  # soft predict
            accuracy_th = 0.5
            preds = probs > accuracy_th  # hard predict
            all_probs.append(probs.detach().cpu())
            if all_preds.ndim == 1:
                all_preds = preds.detach().cpu().numpy()
            else:
                all_preds = np.concatenate([all_preds, preds.detach().cpu().numpy()], axis=0)

    all_probs = torch.cat(all_probs).numpy()
    assert all_probs.shape[0] == len(test_dataset)
    assert all_probs.shape[1] == args.n_classes
    logging.info(np.sum(all_preds, axis=0))
    logging.info(np.sum(all_labels, axis=0))

    bacc = BACC(all_labels, all_preds, classid)
    R = Recall(all_labels, all_preds, classid)
    F1 = F1Measure(all_labels, all_preds, classid)
    P = Precision(all_labels, all_preds, classid)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000', '#808080', '#C0C0C0', '#800000',
              '#008000', '#000080', '#808000', '#008080']
    # print(all_probs)
    # for i in range(len(all_labels.T)):    # 有问题，已修改，应该按照每一类平均而不是每个样本
    #     fpr, tpr, th = roc_curve(all_labels.T[i], all_probs.T[i], pos_label=1)
    #     # ROCprint(fpr, tpr, i, colors[i])
    #     auroc += auc(fpr, tpr)
    #     # print('class: {}, auc: '.format(i), auc(fpr, tpr))
    # # plt.show()
    # auroc /= len(all_labels.T)

    return {"BACC": bacc,
            "R": R,
            "F1": F1,
            "P": P}