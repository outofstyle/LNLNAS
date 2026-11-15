from typing import List
import random
import pickle

import six
import numpy as np
import torch
from torch import optim
from scipy.stats import stats


def set_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def init_optimizer(params, cfg):
    if cfg:
        cfg = {k:v for k, v in six.iteritems(cfg)}
        opt_cls = getattr(optim, cfg.pop("type"))
        return opt_cls(params, **cfg)
    return None


def init_scheduler(optimizer, cfg):
    if cfg and optimizer is not None:
        cfg = {k:v for k, v in six.iteritems(cfg)}
        sch_cls = getattr(optim.lr_scheduler, cfg.pop("type"))
        return sch_cls(optimizer, **cfg)
    return None


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def test_xp(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    # calculate precision at each point
    cur_inds = np.zeros(num_archs)
    passed_set = set()
    for i_rank, rank in enumerate(ranks):
        cur_inds[i_rank] = (cur_inds[i_rank - 1] if i_rank > 0 else 0) + \
                           int(i_rank in passed_set) + int(rank <= i_rank)
        passed_set.add(rank)
    patks = cur_inds / (np.arange(num_archs) + 1)
    THRESH = 100
    p_corrs = []
    for prec in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        k = np.where(patks[THRESH:] >= prec)[0][0] + THRESH
        arch_inds = ranks[:k][ranks[:k] < k]
        p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return p_corrs


def test_xk(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks


def compare_data(
    archs, 
    accs: List[float], 
    max_compare_ratio: float = 4., 
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
):
    n_max_pairs = int(max_compare_ratio * len(archs))
    acc_diff = np.array(accs)[:, None] - np.array(accs)
    acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
    ex_thresh_inds = np.where(acc_abs_diff_matrix > compare_threshold)
    ex_thresh_num = len(ex_thresh_inds[0])
    if ex_thresh_num > n_max_pairs:
        if choose_pair_criterion == "diff":
            keep_inds = np.argpartition(
                    acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

        elif choose_pair_criterion == "random":
            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
    #archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]
    archs_1 = [archs[i] for i in ex_thresh_inds[1]]
    archs_2 = [archs[i] for i in ex_thresh_inds[0]]
    better_lst = (acc_diff > 0)[ex_thresh_inds]
    return archs_1, archs_2, better_lst


def compare_data_index(
    archs, 
    index,  # archs 对应的原始索引列表（如 [10, 20, 30, ...]）
    accs: List[float], 
    max_compare_ratio: float = 4., 
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
):
    """
    Returns:
        archs_1: 第一个架构数组
        archs_2: 第二个架构数组
        better_lst: 布尔数组，表示 archs_1 是否比 archs_2 好
        index_1: archs_1 的原始索引
        index_2: archs_2 的原始索引
    """
    n_max_pairs = int(max_compare_ratio * len(archs))
    
    # 构建准确率差矩阵
    acc_diff = np.array(accs)[:, None] - np.array(accs)
    acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
    
    # 找出超过阈值的配对
    ex_thresh_inds = np.where(acc_abs_diff_matrix > compare_threshold)
    ex_thresh_num = len(ex_thresh_inds[0])
    
    if ex_thresh_num > n_max_pairs:
        if choose_pair_criterion == "diff":
            keep_inds = np.argpartition(
                acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs
            )[-n_max_pairs:]
        elif choose_pair_criterion == "random":
            keep_inds = np.random.choice(
                np.arange(ex_thresh_num), n_max_pairs, replace=False
            )
        else:
            raise ValueError(f"Criterion {choose_pair_criterion} not supported.")
        
        ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

    # 获取架构和索引
    archs_1 = archs[ex_thresh_inds[1]]
    archs_2 = archs[ex_thresh_inds[0]]
    better_lst = (acc_diff > 0)[ex_thresh_inds]

    # 获取对应的原始索引
    index = np.array(index) 
    index_1 = index[ex_thresh_inds[1]]  # archs_1 的原始索引
    index_2 = index[ex_thresh_inds[0]]  # archs_2 的原始索引

    return archs_1, archs_2, better_lst, index_1, index_2


def valid(val_loader, model, save_path: str = None):
    """
    Test the vanilla predictor on the validation set.

    Args:
        val_loader: Validation data loader.
        model (PointwiseComparator): The vanilla architecture performance predictor.
        save_path (str): Path to save the statistics. Default: `None`.

    Returns:
        low_fidelity_corr (float): The Kendall's Tau correlation between the predicted 
                                   scores and the actual low-fidelity information.
        real_corr (float): The Kendall's Tau correlation between the predicted scores 
                           and the actual performance.
        patk: P@topk.
    """
    model.eval()

    all_scores = []
    all_low_fidelity = []
    all_real_accs = []

    for step, (archs, real_accs, low_fidelity) in enumerate(val_loader):
        archs = np.array(archs)
        low_fidelity = np.array(low_fidelity)
        real_accs = np.array(real_accs)
        n = len(archs)
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        all_low_fidelity += list(low_fidelity)
        all_real_accs += list(real_accs)
    
    low_fidelity_corr = stats.kendalltau(all_low_fidelity, all_scores).correlation
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {"real_acc": all_real_accs, "score": all_scores}
        with open(save_path, "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk  #前1%样本：k=10，8个预测正确，准确率80%，相关性0.75


def index_valid(val_loader, model, save_path: str = None):
    """
    Test the vanilla predictor on the validation set.

    Args:
        val_loader: Validation data loader.
        model (PointwiseComparator): The vanilla architecture performance predictor.
        save_path (str): Path to save the statistics. Default: `None`.

    Returns:
        low_fidelity_corr (float): The Kendall's Tau correlation between the predicted 
                                   scores and the actual low-fidelity information.
        real_corr (float): The Kendall's Tau correlation between the predicted scores 
                           and the actual performance.
        patk: P@topk.
    """
    model.eval()

    all_scores = []
    all_low_fidelity = []
    all_real_accs = []

    for step, (archs, real_accs, low_fidelity, _) in enumerate(val_loader):
        archs = np.array(archs)
        low_fidelity = np.array(low_fidelity)
        real_accs = np.array(real_accs)
        n = len(archs)
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        all_low_fidelity += list(low_fidelity)
        all_real_accs += list(real_accs)
    
    low_fidelity_corr = stats.kendalltau(all_low_fidelity, all_scores).correlation
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {"real_acc": all_real_accs, "score": all_scores}
        with open(save_path, "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk  #前1%样本：k=10，8个预测正确，准确率80%，相关性0.75


def mtl_valid(val_loader, model, save_path: str = None):
    """"
    Test the dynamic ensemble predictor on the validation set.

    Args:
        val_loader: Validation data loader.
        model (DynamicEnsemblePointwiseComparator): The architecture performance predictor.
        save_path (str): Path to save the statistics. Default: `None`.

    Returns:
        low_fidelity_corr (Dict[str, float]): The Kendall's Tau correlation between different
            low-fidelity experts' predicted scores and the actual low-fidelity information.
        real_corr (float): The Kendall's Tau correlation between the predicted scores and the
            actual performance.
        patk: P@topk.
    """
    model.eval()

    all_scores = []
    all_real_accs = []

    for step, (archs, real_accs, low_fidelity_perfs) in enumerate(val_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)

        n = len(archs)

        scores, auxiliary_scores_lst = model.mtl_predict(archs)
        scores = list(scores.cpu().data.numpy())
        all_scores += scores
        all_real_accs += list(real_accs)

        low_fidelity_types = list(low_fidelity_perfs[0].keys())

        if step == 0:
            all_low_fidelity = {low_fidelity: [] for low_fidelity in low_fidelity_types}
            all_low_fidelity_scores = {low_fidelity: [] for low_fidelity in low_fidelity_types}

        for i, low_fidelity in enumerate(low_fidelity_types):
            all_low_fidelity[low_fidelity] += [perf[low_fidelity] for perf in low_fidelity_perfs]
            all_low_fidelity_scores[low_fidelity] += list(auxiliary_scores_lst[i].cpu().data.numpy())

    low_fidelity_corr = {
        low_fidelity: stats.kendalltau(all_low_fidelity[low_fidelity],
                                       all_low_fidelity_scores[low_fidelity]).correlation
        for low_fidelity in low_fidelity_types
    }
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {
            "real_acc": all_real_accs, 
            "score": all_scores, 
            "low_fidelity": all_low_fidelity,
            "low_fidelity_score": all_low_fidelity_scores
        }
        with open(save_path, "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk


def mtl_pretrain(
    train_loader, 
    model, 
    epoch: int, 
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the first step of the dynamic ensemble training, train different low-fidelity 
    experts on different types of low-fidelity information.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()
    has_ground_truth = True
    model.train()
    for step, data in enumerate(train_loader):
        archs, accs, low_fidelity_perfs = data
        archs = np.array(archs)
        n = len(archs)

        data_lst = []

        low_fidelity_types = list(low_fidelity_perfs[0].keys())

        for low_fidelity in low_fidelity_types:
            lf_lst = np.array([perf_lst[low_fidelity] for perf_lst in low_fidelity_perfs])
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                lf_lst, 
                max_compare_ratio, 
                compare_threshold, 
                choose_pair_criterion
            )
            data_lst.append((archs_1, archs_2, better_lst))
            n_diff_pairs = len(better_lst)

        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.mtl_update_compare(data_lst)
        objs.update(loss, n_diff_pairs)
      

    return objs.avg


def mtl_pretrain_kd(
    train_loader, 
    train_loader1,
    real_data,
    model, 
    epoch: int, 
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the first step of the dynamic ensemble training, train different low-fidelity 
    experts on different types of low-fidelity information.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()
    has_ground_truth = True
    model.train()
    for step, data in enumerate(train_loader):
        archs, accs, low_fidelity_perfs = data
        #archs1, accs1, low_fidelity_perfs1 = data1
        archs = np.array(archs)
        #archs1 = np.array(archs1)
        n = len(archs)

        data_lst = []

        low_fidelity_types = list(low_fidelity_perfs[0].keys())

        for low_fidelity in low_fidelity_types:
            lf_lst = np.array([perf_lst[low_fidelity] for perf_lst in low_fidelity_perfs])
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                lf_lst, 
                max_compare_ratio, 
                compare_threshold, 
                choose_pair_criterion
            )
            data_lst.append((archs_1, archs_2, better_lst))
            n_diff_pairs = len(better_lst)

        n_diff_pairs_meter.update(float(n_diff_pairs))
        real_archs = [row[0] for row in real_data.data]
        # real_accs = [row[1] for row in real_data.data]
        # loss = model.mtl_update_compare(data_lst, real_archs, real_accs)
         
        loss = model.mtl_update_compare_kd(data_lst, real_archs, epoch)
        # if epoch % 2 == 0:
        #     loss += model.mtl_update_compare_kd2(data_lst, real_archs, epoch)
        objs.update(loss, n_diff_pairs)
        # 每K步：使用稀疏真值蒸馏
        # if step % 100 == 0 and has_ground_truth:
        #     n_max_pairs = int(max_compare_ratio * n)
        #     selected_data_list = []
           
            # 从archs中随机选取n_max_pairs个架构
           
            # sampled_idx = random.sample(range(n), min(n_max_pairs, n))
            # real_archs = archs[sampled_idx]
            # real_accs = np.array(accs)[sampled_idx]
            
            
            # t_archs_1, t_archs_2, t_better_lst = compare_data(
            #     t_archs, 
            #     t_accs, 
            #     max_compare_ratio, 
            #     compare_threshold, 
            #     choose_pair_criterion
            # )
            #selected_data_list.append((t_archs_1, t_archs_2, better_lst))
            #real_archs, real_accs = real_data.data[:, 0], real_data.data[:, 1]
            
            
            # real_archs = [row[0] for row in real_data.data]
            # real_accs = [row[1] for row in real_data.data]
            # real_archs = np.array(real_archs)
            # real_accs = np.array(real_accs)
            # loss_gt = model.update_with_ground_truth( real_archs, real_accs, alpha=0.1, beta=0.5)
        

    return objs.avg


def mtl_pretrain_kd2(
    train_loader, 
    train_loader1,
    real_data,
    model, 
    epoch: int, 
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the first step of the dynamic ensemble training, train different low-fidelity 
    experts on different types of low-fidelity information.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()
    has_ground_truth = True
    model.train()
    for step, data in enumerate(train_loader):
        archs, accs, low_fidelity_perfs = data
        #archs1, accs1, low_fidelity_perfs1 = data1
        archs = np.array(archs)
        #archs1 = np.array(archs1)
        n = len(archs)

        data_lst = []

        low_fidelity_types = list(low_fidelity_perfs[0].keys())

        for low_fidelity in low_fidelity_types:
            lf_lst = np.array([perf_lst[low_fidelity] for perf_lst in low_fidelity_perfs])
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                lf_lst, 
                max_compare_ratio, 
                compare_threshold, 
                choose_pair_criterion
            )
            data_lst.append((archs_1, archs_2, better_lst))
            n_diff_pairs = len(better_lst)

        n_diff_pairs_meter.update(float(n_diff_pairs))
        real_archs = [row[0] for row in real_data.data]
        # real_accs = [row[1] for row in real_data.data]
        # loss = model.mtl_update_compare(data_lst, real_archs, real_accs)
         
       
        loss = model.mtl_update_compare_kd2(data_lst, real_archs, epoch)
        objs.update(loss, n_diff_pairs)

        

    return objs.avg



def mtl_pretrain_same(
    train_loader, 
    model, 
    epoch: int, 
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train different low-fidelity experts using the SAME set of architecture pairs
    selected from a common reference (e.g., first low-fidelity type).

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Max number of pairs relative to number of archs.
        compare_threshold (float): Minimum performance difference to consider.
        choose_pair_criterion (str): How to select pairs ("random", "diff").

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, data in enumerate(train_loader):
        archs, accs, low_fidelity_perfs = data
        archs = np.array(archs)
        n = len(archs)

        # Step 1: 统一选取一组架构对 —— 基于第一个 low-fidelity 类型（可扩展为其他策略）
        low_fidelity_types = list(low_fidelity_perfs[0].keys())
        if len(low_fidelity_types) == 0:
            continue

        # 使用第一个 low-fidelity 类型的性能来选择架构对（作为公共参考）
        # ref_low_fidelity_key = low_fidelity_types[1]
        # ref_perf_list = np.array([perf[ref_low_fidelity_key] for perf in low_fidelity_perfs])

        # # 统一生成一组架构对（archs_1, archs_2, better_lst）
        # archs_1, archs_2, better_lst = compare_data(
        #     archs,
        #     ref_perf_list,
        #     max_compare_ratio=max_compare_ratio,
        #     compare_threshold=compare_threshold,
        #     choose_pair_criterion=choose_pair_criterion
        # )
        # n_diff_pairs = len(better_lst)
        ######################  随机选取架构
        n_max_pairs = int(max_compare_ratio * n)

        # 生成所有可能的 (i, j) 对，满足 i != j
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        # 随机打乱并采样最多 n_max_pairs 对
        sampled_pairs = random.sample(all_pairs, min(n_max_pairs, len(all_pairs)))

        # 解构成两个列表：archs_1[i] vs archs_2[i]
        idx_1 = [pair[0] for pair in sampled_pairs]  # 第一个架构索引
        idx_2 = [pair[1] for pair in sampled_pairs]  # 第二个架构索引

        archs_1 = [archs[i] for i in idx_1]
        archs_2 = [archs[i] for i in idx_2]
        n_diff_pairs = n_max_pairs #len(better_lst)
        
        #########################################################
        n_diff_pairs_meter.update(float(n_diff_pairs))

        if n_diff_pairs == 0:
            continue  # 避免空输入

        # Step 2: 所有 low-fidelity 类型都使用这一组架构对
        data_lst = []
        for low_fidelity in low_fidelity_types:
            lf_perfs = [perf[low_fidelity] for perf in low_fidelity_perfs]

            # 提取这对架构在当前 low-fidelity 上的性能
            perf_arr = np.array(lf_perfs)
            n_max_pairs = int(max_compare_ratio * len(archs))
            acc_diff = np.array(perf_arr)[:, None] - np.array(perf_arr)
            acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
            ex_thresh_inds = np.where(acc_abs_diff_matrix > compare_threshold)
            ex_thresh_num = len(ex_thresh_inds[0])
            if ex_thresh_num > n_max_pairs:
                if choose_pair_criterion == "diff":
                    keep_inds = np.argpartition(
                            acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
                    ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

                elif choose_pair_criterion == "random":
                    keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
                    ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
          
            better_lst = (acc_diff > 0)[ex_thresh_inds]

            data_lst.append((archs_1, archs_2, better_lst))

        # Step 3: 多任务更新
        loss = model.mtl_update_compare(data_lst)
        objs.update(loss, n_diff_pairs)

    return objs.avg


def mtl_train_kd(
    train_loader, 
    real_data,
    model, 
    epoch: int, 
    max_compare_ratio: float = 4., 
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the second step of the dynamic ensemble training, 
    finetune the entire predictor on the actual performance data.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()


    for step, data in enumerate(train_loader):
        archs, accs, _ = data
        archs = np.array(archs)
        accs = np.array(accs)
        n = len(archs)

        archs_1, archs_2, better_lst = compare_data(
            archs, 
            accs, 
            max_compare_ratio, 
            compare_threshold,
            choose_pair_criterion
        )
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        real_archs = [row[0] for row in real_data.data]
        loss = model.update_compare_kd(archs_1, archs_2, better_lst, real_archs, epoch)
        objs.update(loss, n_diff_pairs)
        
         

    return objs.avg



def mtl_train(
    train_loader, 
    model, 
    epoch: int, 
    max_compare_ratio: float = 4., 
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    In the second step of the dynamic ensemble training, 
    finetune the entire predictor on the actual performance data.

    Args:
        train_loader: Training data loader.
        model (DynamicEnsemblePointwiseComparator): The dynamic ensemble predictor.
        epoch (int): Current epoch number.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).
    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    has_ground_truth = True

    for step, data in enumerate(train_loader):
        archs, accs, _ = data
        archs = np.array(archs)
        accs = np.array(accs)
        n = len(archs)

        archs_1, archs_2, better_lst = compare_data(
            archs, 
            accs, 
            max_compare_ratio, 
            compare_threshold,
            choose_pair_criterion
        )
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.update_compare(archs_1, archs_2, better_lst)
        objs.update(loss, n_diff_pairs)
        
         

    return objs.avg


def train(
    train_loader, 
    model,
    epoch: int,
    compare: bool = True,
    use_low_fidelity: bool = False,
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, (archs, real_accs, low_fidelity) in enumerate(train_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if compare:
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                accs, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

    return objs.avg



def train_m(
    train_loader, 
    model,
    epoch: int,
    compare: bool = True,
    use_low_fidelity: bool = False,
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, (archs, real_accs, low_fidelity, _) in enumerate(train_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if compare:
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                accs, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

    return objs.avg


def noise_train(
    train_loader, 
    val_train_loader,
    model,
    epoch: int,
    compare: bool = True,
    use_low_fidelity: bool = False,
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, ((archs, real_accs, low_fidelity, index),(val_archs, val_real_acc, val_low_fidelity, _)) in enumerate(zip(train_loader, val_train_loader)):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if compare:
            archs_1, archs_2, better_lst, index_1, index_2 = compare_data_index(
                archs, 
                index, 
                accs, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            val_archs_1, val_archs_2, val_better_lst = compare_data(
                val_archs, 
                val_low_fidelity, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst, val_archs_1, val_archs_2, index_1, index_2)
            #loss = model.ipc_pairwise_update(archs_1, archs_2, val_better_lst, val_archs_1, val_archs_2)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

    return objs.avg


def noise_train_2(
    train_loader, 
    val_train_loader,
    model,
    epoch: int,
    compare: bool = True,
    use_low_fidelity: bool = False,
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, ((archs, real_accs, low_fidelity, index),(val_archs, _, val_low_fidelity, val_index)) in enumerate(zip(train_loader, val_train_loader)):
        archs = np.array(archs)
        val_archs = np.array(val_archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if compare:
            archs_1, archs_2, better_lst, index_1, index_2 = compare_data_index(
                archs, 
                index, 
                accs, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            val_archs_1, val_archs_2, val_better_lst, val_index_1, val_index_2 = compare_data_index(
                val_archs, 
                val_index,
                val_low_fidelity, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare_semi(archs_1, archs_2, better_lst, val_archs_1, val_archs_2, val_index_1, val_index_2)
            #loss = model.ipc_pairwise_update(archs_1, archs_2, val_better_lst, val_archs_1, val_archs_2)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

    return objs.avg


def semi_train(
    train_loader, 
    un_train_loader,
    model,
    epoch: int,
    compare: bool = True,
    use_low_fidelity: bool = False,
    max_compare_ratio: float = 4.,
    compare_threshold: float = 0.,
    choose_pair_criterion: str = "random"
) -> float:
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = AverageMeter()
    n_diff_pairs_meter = AverageMeter()

    model.train()
    for step, ((archs, real_accs, low_fidelity, index),(val_archs, _, val_low_fidelity, _)) in enumerate(zip(train_loader, un_train_loader)):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if compare:
            archs_1, archs_2, better_lst, index_1, index_2 = compare_data_index(
                archs, 
                index, 
                accs, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            val_archs_1, val_archs_2, val_better_lst = compare_data(
                val_archs, 
                val_low_fidelity, 
                max_compare_ratio, 
                compare_threshold,
                choose_pair_criterion
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare_semi_1(archs_1, archs_2, better_lst, val_archs_1, val_archs_2, index_1, index_2)
            #loss = model.ipc_pairwise_update(archs_1, archs_2, val_better_lst, val_archs_1, val_archs_2)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

    return objs.avg



#  if len(self.high_group) > 1:
#             high_feats = features[:, self.high_group, :].view(-1, H)  # [B*K, H]
#             high_feats = F.normalize(high_feats, dim=1)
#             sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp

#             labels = torch.arange(B, device=features.device).repeat_interleave(len(self.high_group))
#             pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
#             pos_mask.fill_diagonal_(False)

#             exp_sim = torch.exp(sim_mat)
#             log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
#             contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
#             contrastive_loss_total += contrastive_loss

#             # === 添加 SVD + Cov 正则化 ===
#             if svd_loss_weight > 0 or cov_loss_weight > 0:
#                 # 对高性能组特征应用正则化
#                 svd_loss_high = compute_svd_loss(high_feats, svd_loss_weight)
#                 cov_loss_high = compute_cov_loss(high_feats, cov_loss_weight)
#                 contrastive_loss_total += svd_loss_high + cov_loss_high

    
#         # 跨组对比（Low vs High 共识）
#         if len(self.high_group) > 0 and len(self.low_group) > 0:
#             high_consensus = features[:, self.high_group, :].mean(dim=1)  # [B, H]
#             low_feats = features[:, self.low_group, :].view(-1, H)        # [B*L, H]
#             low_feats = F.normalize(low_feats, dim=1)
#             high_consensus = F.normalize(high_consensus, dim=1)
#             sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp

#             labels = torch.arange(B, device=features.device).repeat_interleave(len(self.low_group))
#             exp_sim = torch.exp(sim_mat)
#             log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
#             contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
#             contrastive_loss_total += contrastive_loss

#             # === 添加 SVD + Cov 正则化 ===
#             if svd_loss_weight > 0 or cov_loss_weight > 0:
#                 # 对低绩效组特征应用正则化
#                 svd_loss_low = compute_svd_loss(low_feats, svd_loss_weight)
#                 cov_loss_low = compute_cov_loss(low_feats, cov_loss_weight)
#                 # 对高性能共识特征应用正则化
#                 svd_loss_high_consensus = compute_svd_loss(high_consensus, svd_loss_weight)
#                 cov_loss_high_consensus = compute_cov_loss(high_consensus, cov_loss_weight)
#                 contrastive_loss_total += svd_loss_low + cov_loss_low + svd_loss_high_consensus + cov_loss_high_consensus

#         total_loss += contrast_weight * contrastive_loss_total
#         self.last_total_loss = task_loss_value