import argparse
import copy

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data, load_out_t, load_out_t_hidden
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from train_and_eval import distill_run_transductive, distill_run_inductive, get_PGD_inputs, advdata_hidden_out

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# get_args()函数：用于解析命令行参数并返回一个包含参数的命名空间对象
# 命令行参数通常由两部分组成：选项（Options）和参数值（Arguments）。
#       选项通常以单个短划线（-）或双短划线（--）开头，后面跟着选项的名称。参数值是选项所需的值，用于提供更具体的信息。
#       例："python program.py --input data.txt":--input 是一个选项，表示输入文件的路径，后面的 data.txt 是对应的参数值


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )
    """生成对抗扰动"""
    parser.add_argument("--PRL", type=bool, default=True, help="Set to True to include Perturbation Robustness Loss", )
    parser.add_argument("--gamma", type=float, default=0.02, help="加噪声计算的adv_loss 所占的比值", )
    parser.add_argument("--CE_adv", type=bool, default=True, help="与真实标签计算交叉熵时是否加噪声", )
    parser.add_argument("--KD_adv", type=bool, default=True, help="蒸馏logits时是否加噪声", )
    parser.add_argument("--KD_hid_adv", type=bool, default=False, help="蒸馏中间hidden时是否加噪声", )
    parser.add_argument("--adv_iters", type=float, default=5, help="迭代次数", )
    parser.add_argument("--adv_eps", type=float, default=0.01, help="对抗性扰动的幅度", )

    """Distall"""
    parser.add_argument("--lamb", type=float, default=0, help="参数平衡硬标签和教师输出的损耗，取 [0，1] 中的值", )
    parser.add_argument("--alpha", type=float, default=0.4, help="蒸馏中间层的占比，取 [0，1] 中的值", )
    parser.add_argument("--T_t", type=int, default=1, help="与教师模型最后一层蒸馏的温度T_t", )
    parser.add_argument("--T_h", type=int, default=1, help="与教师模型中间层hidden 蒸馏的温度T_h", )
    parser.add_argument("--out_t_path", type=str, default="outputs", help="Path to load teacher outputs")

    """Dataset"""
    # CPF_data =["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
    # OGB_data = ["ogbn-arxiv", "ogbn-products"]
    parser.add_argument("--dataset", type=str, default="a-photo", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """Model"""
    parser.add_argument("--model_config_path", type=str, default="./train.conf.yaml",
                        help="Path to model configeration", )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument("--num_layers", type=int, default=2, help="Student model number of layers")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Student model hidden layer dimensions", )
    parser.add_argument("--dropout_ratio", type=float, default=0.3)
    parser.add_argument("--norm_type", type=str, default="none", help="One of [none, batch, layer]")

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--fan_out", type=str, default="5,5", help="SAGE 中每层的样本数长度 = num_layers", )
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for sampler")

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500, help="Evaluate once per how many epochs")
    parser.add_argument("--patience", type=int, default=30, help="验证集的分数在多少个时期内没有提高时提前停止")  # 之前是50

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )

    args = parser.parse_args()
    return args


# run(args)函数：用于加载数据、初始化模型、运行训练和评估，并返回评估结果
def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    global score_lst, out
    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"
    print("device:", device)

    if args.feature_noise != 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "noisy_features", f"noise_{args.feature_noise}"
        )
        # Teacher is assumed to be trained on the same noisy features as well.
        args.out_t_path = args.output_path

    if args.feature_aug_k > 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        # NOTE: Teacher may or may not have augmented features, specify args.out_t_path explicitly.
        # args.out_t_path =
        args.student = f"GA{args.feature_aug_k}{args.student}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"out_t_dir: {out_t_dir}")

    """ Load data and model config"""
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )

    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    args.label_dim = labels.int().max().item() + 1

    if 0 < args.feature_noise <= 1:
        feats = (
                        1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(
            args.model_config_path, args.student, args.dataset
        )  # Note: student config
    conf = dict(args.__dict__, **conf)  # 将args.__dict__和conf字典中的内容合并到一个新的字典conf中
    print("这是参数字典conf:", conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")

    """生成扰动数据"""
    model_int = Model(conf)  # 保存原初始化模型参数
    adv_feats, out_t_hidden_adv_all, out_t_adv_all = 0, 0, 0
    if conf["PRL"]:
        # （2）通过PGD生成扰动
        conf1 = copy.deepcopy(conf)
        model1 = Model(conf1)
        adv_deltas = get_PGD_inputs(model1, feats.to(device), labels.to(device), torch.nn.NLLLoss(), conf1["adv_iters"], conf1["adv_eps"], conf1["seed"])
        adv_feats = torch.add(feats.to(device), adv_deltas)

        # （2）提取训练好的教师模型
        conf_t = conf
        conf_t["model_name"] = "SAGE"
        conf_t["dropout_ratio"] = 0
        teacher_model = Model(conf_t)
        teacher_model.load_state_dict(torch.load(out_t_dir.joinpath("model.pth")))
        # print("教师模型：", teacher_model)
        # （3）得到教师模型针对扰动的输出
        out_t_hidden_adv_all, out_t_adv_all = advdata_hidden_out(conf_t, g, teacher_model, adv_feats)

        # 验证两个模型参数是否相同
        # if are_models_equal(model1, model_int):
        #     print("模型参数相同")
        # else:
        #     print("模型参数不同")

    """ Model init """
    # model = Model(conf)
    model = model_int
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion_l = torch.nn.NLLLoss()  # CrossEntropyLoss() = NLLLoss() + softmax + log
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean")
    criterion_t_hidden = torch.nn.KLDivLoss(reduction="batchmean")
    evaluator = get_evaluator(conf["dataset"])

    """Load teacher model output"""
    out_t = load_out_t(out_t_dir)
    out_t_hidden = load_out_t_hidden(out_t_dir, args.dataset)

    logger.debug(
        f"teacher score on train data: {evaluator(out_t[idx_train], labels[idx_train])}"
    )
    logger.debug(
        f"teacher score on val data: {evaluator(out_t[idx_val], labels[idx_val])}"
    )
    logger.debug(
        f"teacher score on test data: {evaluator(out_t[idx_test], labels[idx_test])}"
    )

    """Data split and run"""
    loss_and_score = []
    if args.exp_setting == "tran":
        idx_l = idx_train
        idx_t = torch.cat([idx_train, idx_val, idx_test])
        distill_indices = (idx_l, idx_t, idx_val, idx_test)
        # teacher_indices = (idx_train, idx_val, idx_test)

        # propagate node feature
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, score_val, score_test = distill_run_transductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            out_t_hidden,
            distill_indices,
            criterion_l,
            criterion_t,
            criterion_t_hidden,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            adv_feats,
            out_t_hidden_adv_all,
            out_t_adv_all,
        )
        score_lst = [score_test]

    elif args.exp_setting == "ind":
        # Create inductive split
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, args.split_rate, args.seed
        )
        obs_idx_l = obs_idx_train
        obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
        distill_indices = (
            obs_idx_l,
            obs_idx_t,
            obs_idx_val,
            obs_idx_test,
            idx_obs,
            idx_test_ind,
        )

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        if args.feature_aug_k > 0:
            obs_g = g.subgraph(idx_obs)
            obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
            feats = feature_prop(feats, g, args.feature_aug_k)
            feats[idx_obs] = obs_feats

        out, score_val, score_test_tran, score_test_ind = distill_run_inductive(
            conf,
            model,
            # teacher_model,
            # g,
            feats,
            labels,
            out_t,
            out_t_hidden,
            distill_indices,
            criterion_l,
            criterion_t,
            criterion_t_hidden,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            adv_feats,
            out_t_hidden_adv_all,
            out_t_adv_all,
        )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving student outputs """
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss"""
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")
    # visualize(out, labels)
    # visualize(out_t, color=labels)

    return score_lst, conf


# repeat_run(args)函数：用于多次运行run(args)函数，并返回平均结果和标准差
def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


# main()函数：用于获取命令行参数，运行run(args)或repeat_run(args)函数，并将结果写入文件
def main():
    global score_str, conf
    args = get_args()
    if args.num_exp == 1:
        score, conf = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    dropout_ratio_final = conf["dropout_ratio"]
    adv_eps = conf["adv_eps"]
    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str} [alpha={args.alpha}、lamb={args.lamb}、gamma={args.gamma}、"
                f"T_t={args.T_t}、T_h={args.T_h}、dropout_ratio={dropout_ratio_final}、"
                f"adv_iters={args.adv_iters}、adv_eps={adv_eps}"
                f"PRL={args.PRL}、CE_adv={args.CE_adv}、KD_adv={args.KD_adv}、KD_hid_adv={args.KD_hid_adv}]\n")

    # for collecting aggregated results
    print(score_str)


if __name__ == "__main__":
    main()
