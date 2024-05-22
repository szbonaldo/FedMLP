import logging
import os
import random
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from numpy import where
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import get_dataset
from model.build_model import build_model
from utils.FedAvg import FedAvg, RSCFed, FedAvg_tao, FedAvg_proto, FedAvg_rela
from utils.FedNoRo import get_output, get_current_consistency_weight, DaAgg
from utils.evaluations import globaltest, classtest
from utils.feature_visual import tnse_Visual
from utils.local_training_init_corr import LocalUpdate
from utils.options import args_parser
from utils.utils import set_seed, set_output_files
from utils.valloss_cal import valloss

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer1, models_dir = set_output_files(args)

    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    # for key in dict_users.keys():
    #     print(len(dict_users[key]))
    #     dict_users[key] = random.sample(dict_users[key], 2500)
    # np.save(f"iid-dictusers/{str(args.dataset) + '_' + str(args.seed) + '_' + str(args.n_clients) + '5000'}.npy",
    #         dict_users, allow_pickle=True)


    logging.info(
        f"train: {np.sum(dataset_train.targets, axis=0)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {np.sum(dataset_test.targets, axis=0)}, total: {len(dataset_test.targets)}")

    row_idx_1, column_idx_1 = where(dataset_train.targets == 1)
    class_pos_idx_1 = []
    for i in range(args.n_classes):
        class_pos_idx_1.append(row_idx_1[where(column_idx_1 == i)[0]])

    p_pos_1 = 0.  # 保留的pos比例
    class_neg_idx_1 = []
    for i in range(args.n_classes):
        class_neg_idx_1.append(np.random.choice(class_pos_idx_1[i], int((1-p_pos_1)*len(class_pos_idx_1[i])), replace=False))   # list(array, ...)
    # --------------------- Partially Labelling ---------------------------
    if args.train:
        # ------------------------------ local settings ------------------------------
        user_id = list(range(args.n_clients))
        dict_len = [len(dict_users[idx]) for idx in user_id]
        trainer_locals_1 = []
        netglob = build_model(args)
        for i in user_id:
            trainer_locals_1.append(LocalUpdate(
                args, i, deepcopy(dataset_train), dict_users[i], class_pos_idx_1, class_neg_idx_1, dataset_test=dataset_test, active_class_list=[i], student=deepcopy(netglob).to(args.device),
                teacher_neg=deepcopy(netglob).to(args.device), teacher_act=deepcopy(netglob).to(args.device)))    # student initial is global
            # trainer_locals_1.append(LocalUpdate(
            #     args, i, deepcopy(dataset_train), dict_users[i], class_pos_idx_1, class_neg_idx_1, dataset_test=dataset_test, student=deepcopy(netglob).to(args.device),
            #     teacher_neg=deepcopy(netglob).to(args.device),
            #     teacher_act=deepcopy(netglob).to(args.device)))  # student initial is global


        # ------------------------------ begin training ------------------------------
        for run in range(args.runs):
            set_seed(int(run))
            netclass = build_model(args)
            logging.info(f"\n===============================> beging, run: {run} <===============================\n")
            w_class_fl = []  # 每个类一个聚合模型
            active_class_list = []  # [, , , , ]
            negetive_class_list = []    # [, , , , ]
            class_active_client_list = []
            class_negative_client_list = []

            # FeMLP
            tao = [0] * args.n_classes
            Prototype = []
            # RoFL: Initialize f_G
            f_G = torch.randn(2*args.n_classes, args.feature_dim, device=args.device)   # [[cls0_0],[cls0_1],[cls1_0]...]
            forget_rate_schedule = []
            forget_rate = args.forget_rate
            exponent = 1
            forget_rate_schedule = np.ones(args.rounds_warmup) * forget_rate
            forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** exponent, args.num_gradual)
            # ------------------------------ stage1:warm-up ------------------------------
            for rnd in range(0, args.rounds_warmup):
                # if rnd == 49:
                #     netglob.load_state_dict(torch.load(
                #         "/home/szb/multilabel/chest 5 miss model/model_warmup_globdistill_0.3_0.1_49.pth"))
                # negetive_class_list = [[1,2,3,4], [0,2,3,4], [0,1,3,4], [0,1,2,4], [0,1,2,3]]
                # active_class_list = [[0], [1], [2], [3], [4]]
                # class_active_client_list = [[0, 1, 2, 3, 4], [0, 1, 2], [0, 2, 3, 4], [0, 1, 3, 4], [1, 2, 3, 4]]
                # class_negative_client_list = [[], [3, 4], [1], [2], [0]]
                if args.exp == 'RSCFed':
                    M = 10
                    K = 6
                    DMA = []
                    for i in range(M):
                        random_numbers = random.sample(range(args.n_clients), K)
                        DMA.append(random_numbers)
                    print('DMA: ', DMA)
                if args.exp == 'RoFL':
                    if len(forget_rate_schedule) > 0:
                        args.forget_rate = forget_rate_schedule[rnd]
                        logging.info('remember_rate: %f' % (1-args.forget_rate))
                # if args.exp == 'FedNoRo' and rnd >= args.rounds_FedNoRo_warmup:
                if args.exp == 'FedNoRo':
                    weight_kd = get_current_consistency_weight(rnd, args.begin, args.end) * args.a
                logging.info("\n------------------------------> training, run: %d, round: %d <------------------------------"  % (run, rnd))
                w_locals, loss_locals = [], []
                class_num_lists, data_nums = [], []
                taos, Prototypes = [], []
                # RoFL
                f_locals = []
                for i in tqdm(user_id):  # training over the subset
                    local = trainer_locals_1[i]
                    if args.exp == 'FedAVG':
                        w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train(rnd,
                            net=deepcopy(netglob).to(args.device), writer1=writer1)
                    if args.exp == 'FedNoRo':
                        if rnd < args.rounds_FedNoRo_warmup:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_FedNoRo(
                                i, rnd,
                                net=deepcopy(netglob).to(args.device), writer1=writer1, weight_kd=weight_kd)
                        # else:
                        #     w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_FedNoRo(
                        #         i, rnd,
                        #         net=deepcopy(netglob).to(args.device), writer1=writer1, weight_kd = weight_kd, clean_clients=clean_clients, noisy_clients = noisy_clients)
                    if args.exp == 'CBAFed':
                        if rnd < args.rounds_CBAFed_warmup:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client, class_num_list, data_num = local.train_CBAFed(
                                rnd, net=deepcopy(netglob).to(args.device))
                        else:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client, class_num_list, data_num = local.train_CBAFed(
                                rnd, net=deepcopy(netglob).to(args.device), pt=pt, tao=tao)
                        class_num_lists.append(deepcopy(class_num_list))
                        data_nums.append(deepcopy(data_num))
                    if args.exp == 'FedAVG+FixMatch':
                        w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_FixMatch(rnd,
                            net=deepcopy(netglob).to(args.device))
                    if args.exp == 'FedLSR':
                        w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_FedLSR(rnd,
                            net=deepcopy(netglob).to(args.device))
                    if args.exp == 'RSCFed':
                        w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_RSCFed(rnd,
                            net=deepcopy(netglob).to(args.device))
                    if args.exp == 'RoFL':
                        w_local, loss_local, f_k = local.train_RoFL(deepcopy(netglob).to(args.device), deepcopy(f_G).to(args.device), rnd)
                    if args.exp == 'FedIRM':
                        if rnd < args.rounds_FedIRM_sup - 1:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_FedIRM(
                                rnd, Prototype, writer1, negetive_class_list=None, active_class_list_client_i=None,
                                net=deepcopy(netglob).to(args.device))
                        else:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client, Prototype_local = local.train_FedIRM(
                                rnd, Prototype, writer1, negetive_class_list[i], active_class_list[i],
                                net=deepcopy(netglob).to(args.device))
                    if args.exp == 'FeMLP':
                        if rnd < args.rounds_FeMLP_stage1-1:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client = local.train_FedMLP(
                                rnd, tao, Prototype, writer1, negetive_class_list=None, active_class_list_client_i=None, net=deepcopy(netglob).to(args.device))
                        else:
                            w_local, loss_local, loss_false_negetive, loss_true_negetive, negetive_class_list_client, active_class_list_client, tao_local, Prototype_local = local.train_FeMLP(
                                rnd, tao, Prototype, writer1, negetive_class_list[i], active_class_list[i], net=deepcopy(netglob).to(args.device))
                    if rnd == 0 and args.exp != 'RoFL':
                        active_class_list.append(active_class_list_client)
                        negetive_class_list.append(negetive_class_list_client)
                    # store every updated model
                    if args.exp == 'FedMLP' and rnd >= args.rounds_FedMLP_stage1-1:
                        taos.append(deepcopy(tao_local))
                        Prototypes.append(deepcopy(Prototype_local))
                    if args.exp == 'FedIRM' and rnd >= args.rounds_FedIRM_sup-1:
                        Prototypes.append(deepcopy(Prototype_local.detach().cpu()))
                    if args.exp == 'RoFL':
                        f_locals.append(f_k)
                    w_locals.append(deepcopy(w_local))
                    loss_locals.append(deepcopy(loss_local))
                    writer1.add_scalar(f'train_run{run}/warm-up-loss/client{i}', loss_local, rnd)
                # aggregation
                if rnd == 0 and args.exp != 'RoFL':
                    for i in range(len(active_class_list)):
                        class_active_client_list.append([])
                        class_negative_client_list.append([])
                        for j in range(len(active_class_list)):
                            if i in active_class_list[j]:
                                class_active_client_list[i].append(j)
                            if i in negetive_class_list[j]:
                                class_negative_client_list[i].append(j)
                    logging.info(class_active_client_list)
                    logging.info(class_negative_client_list)
                assert i == user_id[-1]
                assert len(w_locals) == len(dict_len) == args.n_clients
                if args.exp == 'RSCFed':
                    w_glob_fl = RSCFed(DMA, w_locals, K, dict_len, M)
                    netglob.load_state_dict(deepcopy(w_glob_fl))
                elif args.exp == 'FedMLP':
                    if rnd < args.rounds_FedMLP_stage1 - 1:
                        w_glob_fl = FedAvg(w_locals, dict_len)
                        netglob.load_state_dict(deepcopy(w_glob_fl))
                    else:
                        w_glob_fl = FedAvg(w_locals, dict_len)
                        netglob.load_state_dict(deepcopy(w_glob_fl))
                        tao = FedAvg_tao(taos, dict_len, class_negative_client_list)
                        print('avg_tao: ', tao)
                        # if args.miss_client_difficulty == 1:
                        #     tao = FedAvg_tao(taos, dict_len)
                        # else:
                        #     tao = FedAvg_tao(taos, dict_len, class_active_client_list)
                        # print('avg_tao: ', tao)
                        if rnd == args.rounds_FedMLP_stage1 - 1:
                            Prototype = FedAvg_proto(Prototypes, dict_len, class_active_client_list)
                        else:
                            lam = 1.0
                            Prototype = (1-lam)*Prototype + lam*FedAvg_proto(Prototypes, dict_len, class_active_client_list)    # 可以更新平缓一点
                        print('rnd: ', rnd, 'ok')
                    if (rnd + 1) % 10 == 0 and run == 0:
                        torch.save(netglob.state_dict(), models_dir + f'/model_warmup_globdistill_0.3_0.1_{rnd}.pth')
                elif args.exp == 'FedIRM':
                    if rnd < args.rounds_FedIRM_sup - 1:
                        w_glob_fl = FedAvg(w_locals, dict_len)
                        netglob.load_state_dict(deepcopy(w_glob_fl))
                    else:
                        w_glob_fl = FedAvg(w_locals, dict_len)
                        netglob.load_state_dict(deepcopy(w_glob_fl))
                        if rnd == args.rounds_FedIRM_sup - 1:
                            print(Prototypes)
                            Prototype = FedAvg_rela(Prototypes, dict_len, class_active_client_list)
                            print(Prototype)
                        else:
                            lam = 1.0
                            Prototype = (1-lam)*Prototype + lam*FedAvg_rela(Prototypes, dict_len, class_active_client_list)    # 可以更新平缓一点
                        print('rnd: ', rnd, 'ok')
                elif args.exp == 'RoFL':
                    w_glob_fl = FedAvg(w_locals, dict_len)
                    netglob.load_state_dict(deepcopy(w_glob_fl))
                    sim = torch.nn.CosineSimilarity(dim=1)
                    tmp = 0
                    w_sum = 0
                    for i in f_locals:
                        sim_weight = sim(f_G, i).reshape(2*args.n_classes, 1)
                        w_sum += sim_weight
                        tmp += sim_weight * i
                        # print(sim_weight)
                        # print(i)
                    for i in range(len(w_sum)):
                        if w_sum[i, 0] == 0:
                            w_sum[i, 0] = 1
                    f_G = torch.div(tmp, w_sum)
                elif args.exp == 'FedNoRo':
                    if rnd < args.rounds_FedNoRo_warmup:
                        w_glob_fl = FedAvg(w_locals, dict_len)
                        netglob.load_state_dict(deepcopy(w_glob_fl))
                elif args.exp == 'CBAFed':
                    if rnd < args.rounds_CBAFed_warmup:
                        if rnd % 5 != 0:
                            w_glob_fl = FedAvg(w_locals, dict_len)
                            netglob.load_state_dict(deepcopy(w_glob_fl))
                        else:
                            if rnd == 0:
                                w_glob_fl = FedAvg(w_locals, dict_len)
                                netglob.load_state_dict(deepcopy(w_glob_fl))
                                w_glob_res = deepcopy(w_glob_fl)
                            else:
                                w_glob_fl = FedAvg(w_locals, dict_len)
                                for k in w_glob_fl.keys():
                                    w_glob_fl[k] = 0.2*w_glob_fl[k] + 0.8*w_glob_res[k]
                                netglob.load_state_dict(deepcopy(w_glob_fl))
                                w_glob_res = deepcopy(w_glob_fl)
                    if rnd >= args.rounds_CBAFed_warmup - 1:
                        c_num = torch.zeros(args.n_classes)
                        d_num = 0
                        for s in user_id:
                            c_num += class_num_lists[s]
                            d_num += data_nums[s]
                        pt = c_num / d_num
                        avg_pt = pt.sum() / len(pt)
                        std_pt = torch.sqrt((1/(len(pt)-1))*(((pt-avg_pt)**2).sum()))
                        tao = pt + 0.45 - std_pt
                        tao = torch.where(tao > 0.95, 0.95, tao)
                        tao = torch.where(tao < 0.55, 0.55, tao)
                    if rnd >= args.rounds_CBAFed_warmup:
                        wti = (torch.tensor(data_nums) / torch.tensor(data_nums).sum()).tolist()
                        if (rnd-args.rounds_CBAFed_warmup) % 5 != 0:
                            w_glob_fl = FedAvg(w_locals, wti)
                            netglob.load_state_dict(deepcopy(w_glob_fl))
                        else:
                            if (rnd-args.rounds_CBAFed_warmup) == 0:
                                w_glob_fl = FedAvg(w_locals, wti)
                                netglob.load_state_dict(deepcopy(w_glob_fl))
                                w_glob_res = deepcopy(w_glob_fl)
                            else:
                                w_glob_fl = FedAvg(w_locals, wti)
                                for k in w_glob_fl.keys():
                                    w_glob_fl[k] = 0.5*w_glob_fl[k] + 0.5*w_glob_res[k]
                                netglob.load_state_dict(deepcopy(w_glob_fl))
                                w_glob_res = deepcopy(w_glob_fl)
                else:
                    w_glob_fl = FedAvg(w_locals, dict_len)
                    netglob.load_state_dict(deepcopy(w_glob_fl))

                # validate
                if rnd % 10 == 9:
                    logging.info(
                        "\n------------------------------> testing, run: %d, round: %d <------------------------------" % (
                            run, rnd))
                    result = globaltest(deepcopy(netglob).to(args.device), test_dataset=dataset_test, args=args)
                    mAP, BACC, R, F1, auroc, P, hamming_loss = result["mAP"], result["BACC"], result["R"], result[
                        "F1"], result["auc"], result["P"], result["hamming_loss"]
                    logging.info(
                        "-----> mAP: %.2f, BACC: %.2f, R: %.2f, F1: %.2f, auc: %.2f, P: %.2f, hamming_loss: %.2f" % (
                            mAP, BACC * 100, R * 100, F1 * 100, auroc * 100, P * 100, hamming_loss))
                    writer1.add_scalar(f'test_run{run}/mAP', mAP, rnd)
                    writer1.add_scalar(f'test_run{run}/BACC', BACC, rnd)
                    writer1.add_scalar(f'test_run{run}/R', R, rnd)
                    writer1.add_scalar(f'test_run{run}/F1', F1, rnd)
                    writer1.add_scalar(f'test_run{run}/auc', auroc, rnd)
                    writer1.add_scalar(f'test_run{run}/P', P, rnd)
                    writer1.add_scalar(f'test_run{run}/hamming_loss', hamming_loss, rnd)
                    logging.info('\n')
                if rnd == 49:
                    torch.save(netglob.state_dict(), models_dir + f'/model_{rnd}.pth')

                if rnd % 10 == 9:
                    logging.info('test')
                    logging.info("\n------------------------------> testing, run: %d, round: %d <------------------------------"  % (run, rnd))
                    result = globaltest(deepcopy(netglob).to(args.device), test_dataset=dataset_test, args=args)
                    mAP, BACC, R, F1, auroc, P, hamming_loss = result["mAP"], result["BACC"], result["R"], result["F1"], result["auc"], result["P"], result["hamming_loss"]
                    logging.info("-----> mAP: %.2f, BACC: %.2f, R: %.2f, F1: %.2f, auc: %.2f, P: %.2f, hamming_loss: %.2f"  % (mAP, BACC*100, R*100, F1*100, auroc*100, P*100, hamming_loss))
                    logging.info(np.array(loss_locals))
                    writer1.add_scalar(f'corr-test_run{run}/mAP', mAP, rnd)
                    writer1.add_scalar(f'corr-test_run{run}/BACC', BACC, rnd)
                    writer1.add_scalar(f'corr-test_run{run}/R', R, rnd)
                    writer1.add_scalar(f'corr-test_run{run}/F1', F1, rnd)
                    writer1.add_scalar(f'corr-test_run{run}/auc', auroc, rnd)
                    writer1.add_scalar(f'corr-test_run{run}/P', P, rnd)
                    writer1.add_scalar(f'corr-test_run{run}/hamming_loss', hamming_loss, rnd)
                    logging.info('\n')

                # save model
                if (rnd + 1) == args.rounds_corr:
                    torch.save(netglob.state_dict(), models_dir + f'/corr_model_{run}_{rnd}.pth')
                if rnd < 50 and (rnd + 1) % 5 == 0 and run == 0:
                    torch.save(netglob.state_dict(), models_dir + f'/corr_model_{run}_{rnd}.pth')

    else:   # test
        netglob = build_model(args)
        netglob.load_state_dict(torch.load("/home/szb/multilabel/model_warmup/model_warmup_49.pth"))
        result = classtest(deepcopy(netglob).to(args.device), test_dataset=dataset_test, args=args, classid=1)
        BACC, R, F1, P = result["BACC"], result["R"], result["F1"], result["P"]
        logging.info(
            "-----> BACC: %.2f, R: %.2f, F1: %.2f, P: %.2f" % (BACC * 100, R * 100, F1 * 100, P * 100))
        logging.info('\n')
        result = classtest(deepcopy(netglob).to(args.device), test_dataset=dataset_test, args=args, classid=4)
        BACC, R, F1, P = result["BACC"], result["R"], result["F1"], result["P"]
        logging.info(
            "-----> BACC: %.2f, R: %.2f, F1: %.2f, P: %.2f" % (BACC * 100, R * 100, F1 * 100, P * 100))
        logging.info('\n')

    torch.cuda.empty_cache()