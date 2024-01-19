from sched import scheduler

import numpy as np
import copy
import torch
import dgl
from dgl.dataloading import GraphDataLoader
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
import time

from torch import optim
from tqdm import tqdm

from models import Model
from utils import set_seed
from torch.nn import functional as F  # 加蒸馏温度T那里用得到
import torch.utils.data as data

"""
1. Train and eval
"""


# train: GNN的全批训练函数，接受整个图作为输入
def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


# train_sage: 使用GraphSAGE训练的函数，以小批量方式处理图数据
def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


# train_mini_batch: 对于大型数据集的MLP训练函数，以小批量方式处理数据
def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


# train_mini_batch: 对于大型数据集的MLP训练函数，以小批量方式处理数据
def trainMLP_mini_batch(model, feats, labels, criterion, batch_size, optimizer, lamb=0, alpha=0, gamma=0.1,
                        T_h=1, T_t=1, hidden_teacher=0, adv_feats=0, out_t_hidden_adv=0,
                        out_t_adv=0, criterion_t_hidden=None, distill=False, PRL=False, CE_adv=False,
                        KD_adv=False, KD_hid_adv=False
                        ):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    global loss
    # # adversarial learning
    # if PRL:
    #     hidden_adv_logits, adv_logits = model.inference(None, adv_feats[idx_batch[i]])

    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)
    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function

        if lamb and (not distill):  # 与真实标签loss
            out_list, out = model(None, feats[idx_batch[i]])
            loss = criterion(F.log_softmax(out, dim=1), labels[idx_batch[i]])

            # adversarial training
            if PRL and gamma:
                loss_adv = 0
                # gamma1 = gamma
                gamma1 = 0.1
                _, adv_out = model.inference(None, adv_feats[idx_batch[i]])
                if CE_adv:
                    loss_adv = criterion(F.log_softmax(adv_out, dim=1), labels[idx_batch[i]])
                loss += gamma1 * loss_adv  # 加对抗学习loss
            #     gamma1 = 0.1
            #     adv_deltas = get_PGD_inputs(model, feats[idx_batch[i]], labels[idx_batch[i]], criterion, adv_eps)
            #     adv_feats = torch.add(feats[idx_batch[i]], adv_deltas)
            #     _, adv_logits = model(None, adv_feats)
            #     adv_out = adv_logits.log_softmax(dim=1)
            #     loss_adv = criterion(adv_out, labels[idx_batch[i]])
            #     loss += gamma1 * loss_adv  # 加对抗学习loss

            total_loss += loss.item()
            loss *= lamb

        if 1 - lamb and distill:  # 蒸馏
            beta = 1.0 - alpha
            loss_hidden = 0
            hidden_list, logits = model(None, feats[idx_batch[i]])
            hidden_student = F.log_softmax(hidden_list[-1] / T_h, dim=1)
            loss_hidden += criterion_t_hidden(hidden_student, hidden_teacher[idx_batch[i]])  # 蒸馏中间层

            logits_student = F.log_softmax(logits / T_t, dim=1)
            loss_t = criterion(logits_student, labels[idx_batch[i]])  # 蒸馏最后一层
            loss = (alpha * T_h * T_h * loss_hidden + beta * T_t * T_t * loss_t)

            # adversarial learning
            if PRL and gamma:
                hidden_adv_logits, adv_logits = model.inference(None, adv_feats[idx_batch[i]])
                loss_hidden_adv, loss_t_adv = 0, 0
                if KD_hid_adv:  # 倒数第二层
                    adv_hidden_student = F.log_softmax(hidden_adv_logits[-1] / T_h, dim=1)
                    loss_hidden_adv = criterion_t_hidden(adv_hidden_student,
                                                         out_t_hidden_adv[idx_batch[i]])  # 加噪声后和教师噪声中间层蒸馏

                if KD_adv:  # 最后一层logits
                    adv_out = F.log_softmax(adv_logits / T_t, dim=1)
                    loss_t_adv = criterion(adv_out, out_t_adv[idx_batch[i]])  # 加噪声后和教师噪声logits蒸馏

                loss += gamma * (alpha * T_h * T_h * loss_hidden_adv + beta * T_t * T_t * loss_t_adv)
            #     adv_deltas = get_PGD_inputs(model, feats[idx_batch[i]], labels[idx_batch[i]], criterion, adv_eps)
            #     adv_feats = torch.add(feats[idx_batch[i]], adv_deltas)
            #     hidden_adv_logits, adv_logits = model(None, adv_feats)
            #
            #     loss_hidden_adv = 0
            #     if KD_hid_adv:
            #         adv_hidden_student = F.log_softmax(hidden_adv_logits[0] / T_h, dim=1)
            #         loss_hidden_adv = criterion_t_hidden(adv_hidden_student, hidden_teacher[idx_batch[i]])  # 加噪声后蒸馏中间层
            #
            #     adv_out = F.log_softmax(adv_logits, dim=1)
            #     loss_t_adv = criterion(adv_out, labels[idx_batch[i]])
            #     loss += gamma * (loss_hidden_adv + loss_t_adv)  # 加对抗学习loss

            total_loss += loss.item()
            loss *= (1 - lamb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        out_logits_hidden, out = model.inference(data, feats)
        # out_logits_hidden = out_logits_hidden.log_softmax(dim=1)
        # out = out.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out_logits_hidden, out, loss.item(), score


# evaluate_mini_batch: 对于大型数据集的MLP评估函数，以小批量方式处理数据
def evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            _, logits = model.inference(None, feats[batch_size * i: batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score


# GraphDataLoader不支持直接传递node_ids参数。通过在数据集对象中进行适当的修改来实现所需的功能。
class MyDataset(data.Dataset):
    def __init__(self, graphs, labels, node_ids):
        self.graphs = graphs
        self.labels = labels
        self.node_ids = node_ids

    def __getitem__(self, index):
        graph = self.graphs[index]
        label = self.labels[index]
        node_id = self.node_ids[index]
        return graph, label, node_id

    def __len__(self):
        return len(self.graphs)


# 处理图形和节点索引
def collate_fn(samples):
    graphs, labels, node_ids = zip(*samples)  # 拆分样本元组为图数据和标签
    batched_graphs = dgl.batch(graphs)  # 将图数据批量化
    batched_labels = torch.stack(labels)  # 将标签转换为张量
    return batched_graphs, batched_labels, node_ids


"""
2. Run teacher
"""


def run_transductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    global data, feats_train, labels_train, data_eval, feats_val, feats_test, labels_val, labels_test
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(model, data, feats, labels, criterion, optimizer)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion, batch_size, evaluator
                )
            else:
                _, out, loss_train, score_train = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_train
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )

            print(
                "epoch{:3d} | loss: {:.4f}| loss_test:{:.4f} | score_train:{:.4f} | score_val:{:.4f}| score_test:{:.4f}"
                    .format(epoch, loss, loss_test, score_train, score_val, score_test))

            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_val
        )
    else:
        out_logits_hidden, out, _, score_val = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_val
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    if "MLP" in model.model_name:
        return out, score_val, score_test
    else:
        return out_logits_hidden, out, score_val, score_test


def run_inductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    global feats_train, labels_train, feats_val, labels_val, feats_test_tran, labels_test_tran, feats_test_ind, labels_test_ind, obs_data_eval, data_eval, state, obs_data, out_logits_hidden, obs_out_logits_hidden
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # create_formats_()方法：在启动采样流程之前创建 csr/coo/csc 格式
        # 这避免了在每个数据加载器进程中创建某些格式，从而节省了 momory 和 CPU
        obs_g.create_formats_()
        g.create_formats_()

        # 创建采样器
        sampler = MultiLayerNeighborSampler([eval(fanout) for fanout in conf["fan_out"].split(",")])
        # 将 conf["fan_out"] 字符串拆分成一个列表，并对列表中的每个元素应用 eval() 函数进行评估
        # conf["fan_out"]=[5,5]:表示SAGE中每层的样本数。长度=num_layers"
        sampler_eval = MultiLayerFullNeighborSampler(1)  # MultiLayerFullNeighborSampler采样器在每个节点周围采样完整的邻居节点

        # 将图转换为元组格式
        obs_dataset = MyDataset(obs_g, obs_labels, obs_idx_train)  # obs_idx_train:节点索引列表
        dataset = MyDataset(g, labels,
                            torch.arange(g.num_nodes()))  # torch.arange(obs_g.num_nodes())创建了一个包含所有节点索引的张量，确保加载了所有节点的数据

        # 创建数据加载器
        obs_dataloader = dgl.dataloading.DataLoader(  # 创建一个 GraphDataLoader 对象，用于加载观察图（obs_g）的节点数据
            graph=obs_g,
            indices=obs_idx_train,
            graph_sampler=sampler,  # 指定的采样器
            batch_size=batch_size,
            shuffle=False,  # 指定是否对数据进行洗牌
            drop_last=False,  # 指定是否丢弃最后一个不完整的批次
            num_workers=conf["num_workers"],  # 定了数据加载器的并行工作进程数
        )
        obs_dataloader_eval = dgl.dataloading.DataLoader(
            graph=obs_g,
            indices=torch.arange(obs_g.num_nodes()),
            graph_sampler=sampler_eval,  # 使用了MultiLayerFullNeighborSampler作为采样器，该采样器在每个节点周围采样完整的邻居节点
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.DataLoader(
            graph=g,
            indices=torch.arange(g.num_nodes()),
            graph_sampler=sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # # 创建数据加载器
        # obs_dataloader = GraphDataLoader(  # 创建一个 GraphDataLoader 对象，用于加载观察图（obs_g）的节点数据
        #     dataset=obs_dataset,
        #     graph=obs_g,
        #     node_ids=obs_idx_train,
        #     sampler=sampler,  # 指定的采样器
        #     batch_size=batch_size,
        #     shuffle=False,  # 指定是否对数据进行洗牌
        #     drop_last=False,  # 指定是否丢弃最后一个不完整的批次
        #     num_workers=conf["num_workers"],  # 定了数据加载器的并行工作进程数
        # )
        # obs_dataloader_eval = GraphDataLoader(
        #     dataset=obs_dataset,
        #     graph=obs_g,
        #     node_ids=torch.arange(obs_g.num_nodes()),
        #     sampler=sampler_eval,  # 使用了MultiLayerFullNeighborSampler作为采样器，该采样器在每个节点周围采样完整的邻居节点
        #     batch_size=batch_size,
        #     shuffle=False,
        #     drop_last=False,
        #     num_workers=conf["num_workers"],
        # )
        # dataloader_eval = GraphDataLoader(
        #     dataset=dataset,
        #     graph=g,
        #     node_ids=torch.arange(g.num_nodes()),
        #     sampler=sampler_eval,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     drop_last=False,
        #     num_workers=conf["num_workers"],
        # )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,
                )
                _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                _, obs_out, loss_train, score_train = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                score_test_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                _, out, loss_test_ind, score_test_ind = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )

            print(
                "epoch{:3d} | loss_test_tran:{:.4f} | loss_test_ind:{:.4f} | score_train:{:.4f} | score_val:{:.4f}| score_test_tran:{:.4f} | score_test_ind:{:.4f}"
                    .format(epoch, loss_test_tran, loss_test_ind, score_train, score_val, score_test_tran,
                            score_test_ind))

            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
        )
        out, _, score_test_ind = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
        )

    else:
        obs_out_logits_hidden, obs_out, _, score_val = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            criterion,
            evaluator,
            obs_idx_val,
        )
        out_logits_hidden, out, _, score_test_ind = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
        )

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    out_logits_hidden[idx_obs] = obs_out_logits_hidden
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out_logits_hidden, out, score_val, score_test_tran, score_test_ind


"""
3. Distill
"""


def distill_run_transductive(
        conf,
        model,
        feats,
        labels,
        out_t_all,
        out_t_hidden_all,
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
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    out_t_hidden: 教师模型倒数第二层 的logits
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t & ut_t_hidden`) respectively
    loss_and_score: Stores losses and scores.
    """
    seed = conf["seed"]
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]  # lamb: weight parameter lambda
    alpha = conf["alpha"]  # hidden隐藏层蒸馏的比重
    T_t = conf["T_t"]
    T_h = conf["T_h"]
    # 对抗训练
    PRL = conf["PRL"]
    gamma = conf["gamma"]
    CE_adv = conf["CE_adv"]
    KD_adv = conf["KD_adv"]
    KD_hid_adv = conf["KD_hid_adv"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    # 设置随机种子
    set_seed(seed)

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    out_t_hidden_all = out_t_hidden_all.to(device)

    # 为教师模型加激活函数softmax、蒸馏温度T
    out_t_hidden_all = F.softmax(out_t_hidden_all / T_h, dim=1)
    out_t_all = F.softmax(out_t_all / T_t, dim=1)

    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t, out_t_hidden = feats[idx_t], out_t_all[idx_t], out_t_hidden_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    # 处理扰动数据
    out_t_adv, out_t_hidden_adv, adv_feats_t, adv_feats_l = 0, 0, 0, 0
    if PRL and gamma:
        # adv_feats, out_t_hidden_adv_all, out_t_adv_all = advdata_hidden_out(conf, g, model, teacher_model, feats,labels, criterion_l)
        # 教师模型在扰动数据集上的acc
        adv_teacher_score = evaluator(out_t_adv_all[idx_test], labels[idx_test])
        print("教师在噪声数据上的accuracy:", adv_teacher_score)

        # trans 设置下：划分扰动数据 数据集
        adv_feats_l = adv_feats[idx_l]
        adv_feats_t, out_t_adv, out_t_hidden_adv = adv_feats[idx_t], out_t_adv_all[idx_t], out_t_hidden_adv_all[idx_t]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        """
            feats_t是feats[idx_t]：即学生模型对应教师模型节点下标的feat
            out_t 是教师模型节点预测的软标签 2485*7
                   【out_t =out_t_all[idx_t],其中idx_t=torch.cat([idx_train, idx_val, idx_test])】
        # """
        loss_t = 0
        hard_loss = 0

        if 1 - lamb:
            loss_t = trainMLP_mini_batch(model, feats_t, out_t, criterion_t, batch_size, optimizer, lamb,
                                         alpha,
                                         gamma, T_h, T_t, out_t_hidden, adv_feats_t, out_t_hidden_adv,
                                         out_t_adv, criterion_t_hidden, distill=True, PRL=PRL, KD_adv=KD_adv,
                                         KD_hid_adv=KD_hid_adv)
        if lamb:
            hard_loss = trainMLP_mini_batch(model, feats_l, labels_l, criterion_l, batch_size, optimizer, lamb,
                                            alpha, gamma, adv_feats=adv_feats_l, PRL=PRL, CE_adv=CE_adv)

        loss = hard_loss + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test, score_test = evaluate_mini_batch(
                model, feats_test, labels_test, criterion_l, batch_size, evaluator
            )

            logger.debug(
                f"Ep {epoch:3d} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            print(
                "epoch{:3d} | loss: {:.4f}| hard_loss: {:.4f}| loss_t: {:.4f}| loss_test:{:.4f} | score_l:{:.4f} | score_val:{:.4f}| score_test:{:.4f}"
                    .format(epoch, loss, hard_loss, loss_t, loss_test, score_l, score_val, score_test))

            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test


def distill_run_inductive(
        conf,
        model,
        # teacher_model,
        # g,
        feats,
        labels,
        out_t_all,
        out_t_hidden_all,
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
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    seed = conf["seed"]
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    # distill
    alpha = conf["alpha"]  # hidden隐藏层蒸馏的比重
    T_t = conf["T_t"]
    T_h = conf["T_h"]
    # 对抗训练
    PRL = conf["PRL"]
    gamma = conf["gamma"]
    CE_adv = conf["CE_adv"]
    KD_adv = conf["KD_adv"]
    KD_hid_adv = conf["KD_hid_adv"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    # 设置随机种子
    set_seed(seed)

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    out_t_hidden_all = out_t_hidden_all.to(device)

    # 为教师模型加激活函数softmax、蒸馏温度T
    out_t_hidden_all = F.softmax(out_t_hidden_all / T_h, dim=1)
    out_t_all = F.softmax(out_t_all / T_t, dim=1)

    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]
    obs_out_t_hidden = out_t_hidden_all[idx_obs]  # 240101 补充

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t, out_t_hidden = obs_feats[obs_idx_t], obs_out_t[obs_idx_t], obs_out_t_hidden[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    # 处理扰动数据
    out_t_hidden_adv, out_t_adv, adv_feats_t, adv_feats_l = 0, 0, 0, 0
    if PRL and gamma:
        # out_t_hidden_adv_all, out_t_adv_all = advdata_hidden_out(conf, g, model, teacher_model, feats,
        #                                                          labels, criterion_l)
        # 教师模型在扰动数据集上的acc
        adv_teacher_score_tran = evaluator(out_t_adv_all[obs_idx_test], labels[obs_idx_test])
        adv_teacher_score_ind = evaluator(out_t_adv_all[idx_test_ind], labels[idx_test_ind])
        print(
            "teacher在扰动数据上的accuracy: trans: {:.4f}| ind: {:.4f}".format(adv_teacher_score_tran, adv_teacher_score_ind))

        # "ind"设置下：划分扰动数据 数据集
        adv_obs_feats = adv_feats[idx_obs]
        adv_obs_out_t = out_t_adv_all[idx_obs]
        adv_obs_out_t_hidden = out_t_hidden_adv_all[idx_obs]

        adv_feats_l = adv_obs_feats[obs_idx_l]
        adv_feats_t, out_t_adv, out_t_hidden_adv = adv_obs_feats[obs_idx_t], adv_obs_out_t[obs_idx_t], \
                                                   adv_obs_out_t_hidden[obs_idx_t]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_t = 0
        hard_loss = 0
        if lamb:
            hard_loss = trainMLP_mini_batch(model, feats_l, labels_l, criterion_l, batch_size, optimizer, lamb,
                                            alpha, gamma, adv_feats=adv_feats_l, PRL=PRL, CE_adv=CE_adv)
        if 1 - lamb:
            loss_t = trainMLP_mini_batch(model, feats_t, out_t, criterion_t, batch_size, optimizer, lamb,
                                         alpha,
                                         gamma, T_h, T_t, out_t_hidden, adv_feats_t, out_t_hidden_adv, out_t_adv,
                                         criterion_t_hidden, distill=True, PRL=PRL, KD_adv=KD_adv,
                                         KD_hid_adv=KD_hid_adv)
        loss = hard_loss + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(  # "_, "用于忽略函数调用那些不需要的的返回值
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            print(
                "epoch{:3d} | loss: {:.4f}| hard_loss: {:.4f}| loss_t: {:.4f}| score_l: {:.4f}| score_val: {:.4f}| score_test_tran:{:.4f}| score_test_ind:{:.4f}"
                    .format(epoch, loss, hard_loss, loss_t, score_l, score_val, score_test_tran, score_test_ind))
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    # obs_out[obs_idx_test]表示模型在测试集上的预测结果，labels_test_tran是测试集中的真实标签
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out  # 将预测输出 obs_out 存储在名为 out 的数组或列表中的特定索引位置 idx_obs

    # logger 是一个日志记录器对象，用于在代码中记录和输出日志消息
    logger.info(  # logger.info() 用于记录一条信息级别的日志消息
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind


"""
    执行投影梯度下降（Projected Gradient Descent，PGD）方法，生成对抗性扰动
"""


def get_PGD_inputs(model, feats, labels, criterion, adv_iters, adv_eps, seed):
    set_seed(seed)
    # Set the model to evaluation mode
    model.eval()

    iters = adv_iters
    eps = adv_eps  # 表示对抗性扰动的幅度的参数
    alpha = eps / iters  # 扰动更新的步长
    # init （随机初始扰动）
    delta = torch.rand(
        feats.shape,
        requires_grad=True) * eps * 2 - eps  # 创建一个与输入特征 feats 具有相同形状的张量 delta，其中的值是在区间 [-eps, eps] 内的随机数。这个 delta 张量将被用作扰动，并且初始化为随机值
    delta = delta.to(feats.device)
    delta = torch.nn.Parameter(delta)  # 将 delta 张量转换为一个 PyTorch 参数，以便可以通过反向传播来更新它

    for i in range(iters):
        p_feats = feats + delta  # 将扰动 delta 添加到输入特征 feats 上
        _, logits = model.inference(None, p_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, labels)
        # 清零模型的梯度
        model.zero_grad()

        # Backward pass to compute the gradient of the loss with respect to the input
        loss.backward()
        # delta update
        delta.data = delta.data + alpha * delta.grad.sign()  # 根据梯度的方向和步长 alpha 来更新扰动 delta
        # Zero the gradients
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)  # 对扰动 delta 进行截断，以确保其在 [-eps, eps] 范围内

    output = delta.detach()  # 将扰动 delta 分离（detach）出来，使其不再是一个需要梯度更新的参数。
    return output  # 返回最终的扰动 output，这个扰动可以用于对输入数据进行对抗性攻击或对抗性学习中。


"""
    为教师模型生成 扰动数据 对应输出
"""


def advdata_hidden_out(
        conf,
        # batch_size, num_workers, device, T_h, T_t,
        g,
        teacher_model,
        adv_feats,
):
    # (1)教师GNN对噪声数据进行预测
    device = conf["device"]
    batch_size = conf["batch_size"]
    num_workers = conf["num_workers"]
    T_t = conf["T_t"]
    T_h = conf["T_h"]
    out_t_hidden_adv_all, out_t_adv_all = teacher_inference_adv(
        batch_size, num_workers, device,
        teacher_model,
        g,
        adv_feats,
    )
    # (2)为教师模型 新生成的预测 加激活函数softmax、蒸馏温度T
    out_t_hidden_adv_all = F.softmax(out_t_hidden_adv_all / T_h, dim=1)
    out_t_adv_all = F.softmax(out_t_adv_all / T_t, dim=1)

    return out_t_hidden_adv_all, out_t_adv_all


"""
    教师模型进行推理
"""


# 将添加扰动后的数据输入教师模型进行推理
def teacher_inference_adv(
        # conf,
        batch_size, num_workers, device,
        teacher_model,
        g,
        feats,
):
    # idx_train, idx_val, idx_test = teacher_indices

    if "SAGE" in teacher_model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        g.create_formats_()
        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        inf_data_eval = dataloader_eval
    else:
        g = g.to(device)
        inf_data_eval = g

    # 教师模型进行推理
    teacher_model.eval()
    with torch.no_grad():
        if "SAGE" in teacher_model.model_name:
            inf_out_logits_hidden, inf_out = teacher_model.inference(inf_data_eval, feats)
        else:
            inf_out_logits_hidden_list, inf_out = teacher_model.inference(inf_data_eval, feats)
            inf_out_logits_hidden = inf_out_logits_hidden_list[-1]
    # score_test = evaluator(inf_out[idx_test], labels[idx_test])

    return inf_out_logits_hidden, inf_out

