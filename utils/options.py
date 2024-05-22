import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1037, help='random seed2023,1037')
    parser.add_argument('--gpu', type=str, default='2', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='FedMLP', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='ChestXray14', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size per gpu')
    parser.add_argument('--feature_dim', type=int,
                        default=512, help='feature_dim of ResNet18')
    parser.add_argument('--base_lr', type=float, default=3e-5,
                        help='base learning rate,ICH=3e-5,ChestXray14=3e-6')
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--train', type=int, default=1)

    # PSL setting
    parser.add_argument('--annotation_num', type=int,
                        default='1', help='The number of categories annotated by each client.')

    # for FL
    parser.add_argument('--n_clients', type=int, default=8,
                        help='number of users')
    parser.add_argument('--n_classes', type=int, default=8,
                        help='number of classes')
    parser.add_argument('--iid', type=int, default=1, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=0.5, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--rounds_warmup', type=int, default=500, help='rounds')
    parser.add_argument('--rounds_corr', type=int, default=200, help='rounds')
    parser.add_argument('--rounds_distillation', type=int, default=200, help='rounds')
    parser.add_argument('--rounds_finetune', type=int, default=50, help='rounds')
    parser.add_argument('--rounds_FedMLP_stage1', type=int, default=50, help='rounds')
    parser.add_argument('--U', type=float, default=0.7, help='tao_upper_bound')
    parser.add_argument('--L', type=float, default=0.3, help='tao_lower_bound')
    parser.add_argument('--tao_min', type=float, default=0.1, help='tao_min')
    parser.add_argument('--runs', type=int, default=1, help='training seed')

    # RoFL
    parser.add_argument('--forget_rate', type=float, default=0.2, help='forget_rate')
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--T_pl', type=int, help='T_pl: When to start using global guided pseudo labeling', default=100)
    parser.add_argument('--lambda_cen', type=float, help='lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help='lambda_e', default=0.8)

    # FedMLP_abu
    parser.add_argument('--difficulty_estimate', type=int, default=1, help='tao=1 or cal')
    parser.add_argument('--miss_client_difficulty', type=int, default=1, help='consider or not(tao agg method)')
    parser.add_argument('--mixup', type=int, default=1, help='y/n')
    parser.add_argument('--clean_threshold', type=float, default=0.005, help='clean_threshold')
    parser.add_argument('--noise_threshold', type=float, default=0.01, help='noise_threshold')

    # FedLSR
    parser.add_argument('--t_w', type=int, default=40, help='clean_threshold')
    # FedIRM
    parser.add_argument('--rounds_FedIRM_sup', type=int, default=20, help='rounds')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    # FedNoRo
    parser.add_argument('--rounds_FedNoRo_warmup', type=int, default=500, help='rounds')
    parser.add_argument('--begin', type=int, default=10, help='ramp up begin')
    parser.add_argument('--end', type=int, default=499, help='ramp up end')
    parser.add_argument('--a', type=float, default=0.8, help='a')
    #CBAFed
    parser.add_argument('--rounds_CBAFed_warmup', type=int, default=50, help='rounds')
    args = parser.parse_args()
    return args
