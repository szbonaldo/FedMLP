import logging

import numpy as np

def Hamming_Loss(y_true, y_pred, classid=None):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp / (y_true.shape[0] * y_true.shape[1])


# def Recall(y_true, y_pred):  # sample-wise
#     temp = 0
#     for i in range(y_true.shape[0]):
#         if sum(y_true[i]) == 0:
#             continue
#         temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
#     return temp / y_true.shape[0]


def Recall(y_true, y_pred, classid=None):  # class-wise
    temp = 0
    if classid is not None:
        return sum(np.logical_and(y_true.T[classid], y_pred.T[classid])) / sum(y_true.T[classid])
    else:
        for i in range(y_true.T.shape[0]):
            temp += sum(np.logical_and(y_true.T[i], y_pred.T[i])) / sum(y_true.T[i])
            # print('class: {}, recall: '.format(i), sum(np.logical_and(y_true.T[i], y_pred.T[i])) / sum(y_true.T[i]))
        return temp / y_true.T.shape[0]


def BACC(y_true, y_pred, classid=None):
    temp = 0
    if classid is not None:
        recall1 = sum(np.logical_and(y_true.T[classid], y_pred.T[classid])) / sum(y_true.T[classid])
        recall0 = sum(~np.logical_or(y_true.T[classid], y_pred.T[classid])) / (y_true.T[classid].size - np.count_nonzero(y_true.T[classid]))
        bacc = (recall0 + recall1) / 2
        return bacc
    else:
        for i in range(y_true.T.shape[0]):
            recall1 = sum(np.logical_and(y_true.T[i], y_pred.T[i])) / sum(y_true.T[i])
            recall0 = sum(~np.logical_or(y_true.T[i], y_pred.T[i])) / (y_true.T[i].size - np.count_nonzero(y_true.T[i]))
            bacc = (recall0 + recall1) / 2
            logging.info('BACC:class%d : %f' % (i, bacc))
            temp += bacc
        return temp / y_true.T.shape[0]


def Precision(y_true, y_pred, classid=None):
    temp = 0
    if classid is not None:
        return sum(np.logical_and(y_true.T[classid], y_pred.T[classid])) / sum(y_pred.T[classid])
    else:
        for i in range(y_true.T.shape[0]):
            if sum(y_pred.T[i]) == 0:
                continue
            logging.info('P:class%d : %f' % (i, sum(np.logical_and(y_true.T[i], y_pred.T[i])) / sum(y_pred.T[i])))
            temp += sum(np.logical_and(y_true.T[i], y_pred.T[i])) / sum(y_pred.T[i])
            # print('class: {}, precision: '.format(i), sum(np.logical_and(y_true.T[i], y_pred.T[i])) / sum(y_pred.T[i]))
        return temp / y_true.T.shape[0]


def F1Measure(y_true, y_pred, classid=None):
    temp = 0
    if classid is not None:
        return (2*sum(np.logical_and(y_true.T[classid], y_pred.T[classid]))) / (sum(y_true.T[classid])+sum(y_pred.T[classid]))
    else:
        for i in range(y_true.T.shape[0]):
            logging.info('f1:class%d : %f' % (i, (2*sum(np.logical_and(y_true.T[i], y_pred.T[i]))) / (sum(y_true.T[i])+sum(y_pred.T[i]))))
            temp += (2*sum(np.logical_and(y_true.T[i], y_pred.T[i]))) / (sum(y_true.T[i])+sum(y_pred.T[i]))
        return temp / y_true.T.shape[0]


