"""
Networks that take architectures as inputs.
"""

import abc
from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from . import utils

def js_div(p_logits, q_logits):
    p_log_prob = F.log_softmax(p_logits, dim=1)
    q_log_prob = F.log_softmax(q_logits, dim=1)
    m_prob = 0.5 * (F.softmax(p_logits, dim=1) + F.softmax(q_logits, dim=1))
    return 0.5 * (
        F.kl_div(p_log_prob, m_prob, reduction='batchmean') +
        F.kl_div(q_log_prob, m_prob, reduction='batchmean')
    )

def add_weak_noise(x):
    """弱噪声：轻微扰动"""
    noise = torch.randn_like(x) * 0.05  # 轻微高斯噪声
    return x + noise

def add_strong_noise(x):
    """强噪声：较大扰动 + Dropout 风格掩码"""
    noise = torch.randn_like(x) * 0.2   # 较大噪声
    mask = (torch.rand_like(x) > 0.8).float()  # 只保留 80% 的信息
    return (x + noise) * mask



# 4. Barlow Twins 损失函数（去相关 + 分布控制）
def twins_off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_loss(z_a, z_b):
    N, D = z_a.shape
    c = (z_a.T @ z_b) / N  # 归一化协方差矩阵
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = twins_off_diagonal(c).pow_(2).sum()
    return on_diag + 0.05 * off_diag  # λ=0.05 推荐值


def create_pair_comp_graph(scores):
    """
    Args:
        scores: Tensor of shape (B, 2), each row is [score_A, score_B]
    Returns:
        em_targets: Tensor of shape (B, 2), soft preference labels
        sup_preds: Tensor of shape (B,), confidence in prediction
    """
    b = scores.size(0)

    # 使用 temperature 控制 softmax 锐化程度
    T = 0.5
    log_scores = F.softmax(scores, dim=-1)  # shape: (B, 2)

    # 构建 forward graph
    log_alpha = torch.full((b, 4, 2), fill_value=-float('inf'), device=scores.device)
    log_alpha[:, 0, 0] = log_scores[:, 0]
    log_alpha[:, 1, 0] = log_scores[:, 1]

    # 更新第 1 个 time step
    log_alpha[:, :2, 1] = log_alpha[:, :2, 0] + log_scores  # shape: (B, 2, 2)
    log_alpha[:, 2, 1] = log_alpha[:, 3, 0] + log_scores[:, 0]

    # 归一化 alpha 得到 forward probability
    m = torch.logsumexp(log_alpha[:, :, -1], dim=1, keepdim=True)
    alpha = torch.exp(log_alpha[:, :, -1] - m)
    sup_preds = alpha[:, :3].sum(dim=1)

    with torch.no_grad():
        log_beta = torch.full_like(log_alpha, fill_value=-float('inf'))

        log_beta[:, 0, 1] = log_scores[:, 0]
        log_beta[:, 1, 1] = log_scores[:, 1]
        log_beta[:, 2, 1] = log_scores[:, 0]

        log_beta[:, :2, 0] = log_beta[:, :2, 1] + log_scores
        log_beta[:, 3, 0] = log_beta[:, 2, 1] + log_scores[:, 1]

        # 移除不合理的减法操作
        # repeated_log_scores = log_scores.unsqueeze(1).expand(-1, 4, -1)
        # log_beta -= repeated_log_scores

        # 构造后验分布
        gamma = log_alpha + log_beta
        m = torch.logsumexp(gamma, dim=1, keepdim=True)
        gamma = torch.exp(gamma - m)

        # 只取最后一个时间步的状态
        better_states = gamma[:, :2, 1]  # shape: (B, 2)
        worse_states = gamma[:, 2:, 1].sum(dim=1, keepdim=True)  # shape: (B, 1)

        # 合并 better/worse 的权重（可选）
        better_states = better_states / better_states.sum(dim=1, keepdim=True)

        # 返回 better_states 作为 soft preference label
        em_targets = better_states  # shape: (B, 2)

    return em_targets, sup_preds


def compute_con_loss(logits_s, logits_w, reduction='mean'):
    """
    Consistency Loss: KL Divergence between weak (logits_w) and strong (logits_s) predictions.
    
    Args:
        logits_s: Strong model's output logits.
        logits_w: Weak model's output logits.
        reduction: Reduction method ('mean', 'sum', or 'none').
    
    Returns:
        Consistency loss.
    """
    # 将 logits 转换为概率分布
    probs_s = F.softmax(logits_s, dim=-1)
    probs_w = F.softmax(logits_w, dim=-1).detach()  # 停止梯度传播

    # 使用 log_softmax 计算 KL 散度
    log_probs_s = F.log_softmax(logits_s, dim=-1)
    con_loss = F.kl_div(log_probs_s, probs_w, reduction='none').sum(dim=-1)

    if reduction == 'mean':
        return con_loss.mean()
    elif reduction == 'sum':
        return con_loss.sum()
    else:
        return con_loss


def compute_em_loss(logits_s, probs_x_w, noise_matrix_col, reduction='mean'):
    """
    Args:
        logits_s: strong augmentation 输出 logits，shape (B, C)
        probs_x_w: weak augmentation 输出概率，shape (B, C)
        noise_matrix_col: noise_matrix[:, y].transpose(0, 1)，shape (B, C)
    """
    with torch.no_grad():
        # E 步：根据 probs_x_w 和 noise_matrix 推断真实分布
        em_y = probs_x_w * noise_matrix_col
        if em_y.dim() == 1:
            em_y = em_y.view(1, -1)
        em_y = em_y / em_y.sum(dim=1, keepdim=True)

    # M 步：用 em_y 作为目标训练强增强输出
    probs_s = F.softmax(logits_s, dim=0)
    probs_s = torch.clamp(probs_s, min=1e-8)
    em_loss = -torch.sum(em_y * torch.log(probs_s), dim=1)

    if reduction == 'mean':
        return em_loss.mean()
    elif reduction == 'sum':
        return em_loss.sum()
    else:
        return em_loss


# class PairwiseNoiseMatrix(nn.Module):
#     def __init__(self, num_samples, temperature=1.0, symmetric=True):
#         """
#         Args:
#             num_samples (int): 总样本数 N
#             temperature (float): 控制 softmax 温度（锐化程度）
#             symmetric (bool): 是否强制对称（即 better(i,j) = not better(j,i)）
#         """
#         super(PairwiseNoiseMatrix, self).__init__()
#         self.num_samples = num_samples
#         self.temperature = temperature
#         self.symmetric = symmetric

#         # 可学习的权重矩阵
#         self.confidence_weights = nn.Parameter(torch.randn(num_samples, num_samples))

#     def forward(self):
#         """
#         Returns:
#             confidence_matrix (Tensor): shape [N, N], confidence[i][j] 表示 i > j 的可靠性
#         """
#         # 避免自己与自己比较
#         diag_mask = ~torch.eye(self.num_samples, dtype=bool, device=self.confidence_weights.device)
#         masked_weights = self.confidence_weights * diag_mask.float()

#         # 使用 softmax 归一化，得到置信度矩阵
#         confidence_matrix = F.softmax(masked_weights / self.temperature, dim=1)

#         if self.symmetric:
#             # 强制对称性：confidence[i][j] = 1 - confidence[j][i]
#             upper_triangle = torch.triu(confidence_matrix)
#             lower_triangle = torch.tril(1 - confidence_matrix.t())
#             confidence_matrix = upper_triangle + lower_triangle

#         return confidence_matrix


def off_diagonal(x):
    """返回矩阵的非对角线元素"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_svd_loss(feat, weight=0.01, top_k=5):
    """计算 SVD 奇异值损失"""
    if weight <= 0 or feat.size(0) < feat.size(1):
        return torch.tensor(0.0, device=feat.device)
    
    try:
        # 使用 torch.linalg.svd（推荐）
        U, S, Vh = torch.linalg.svd(feat, full_matrices=False)
        if S.numel() == 0:
            return torch.tensor(0.0, device=feat.device)
        
        # 归一化奇异值
        S = S / torch.sum(S)
        # 惩罚前 top_k 个奇异值（避免集中在少数方向）
        top_k = min(top_k, len(S))
        svd_loss = -weight * torch.mean(S[:top_k])
        return svd_loss
    except:
        return torch.tensor(0.0, device=feat.device)

def compute_cov_loss(feat, weight=0.01):
    """计算协方差去相关损失"""
    if weight <= 0 or feat.size(1) <= 1:
        return torch.tensor(0.0, device=feat.device)
    
    # 中心化特征
    feat_centered = feat - feat.mean(dim=0, keepdim=True)
    # 计算协方差矩阵
    cov_matrix = torch.mm(feat_centered.t(), feat_centered) / max(feat.size(0) - 1, 1)
    # 最小化非对角线元素
    cov_loss = weight * off_diagonal(cov_matrix).pow(2).sum().div(cov_matrix.size(1))
    return cov_loss


class PairwiseNoiseMatrix(nn.Module):
    def __init__(self, num_samples, init_confidence=1.0, learnable=True):
        """
        A learnable pairwise noise/confidence matrix for ranking/pairwise learning.

        Args:
            num_samples (int): Total number of samples in training set.
            init_confidence (float): Initial confidence value (close to 1 means less noise).
            learnable (bool): Whether the matrix is learnable or not.
        """
        super(PairwiseNoiseMatrix, self).__init__()
        self.num_samples = num_samples
        self.init_confidence = init_confidence

        # 初始化一个对称的 confidence 矩阵
        conf_matrix = torch.ones(num_samples, num_samples) * init_confidence
        conf_matrix = conf_matrix.fill_diagonal_(1.0)  # 对角线为自己 vs 自己，设为 1

        if learnable:
            self.confidence = nn.Parameter(conf_matrix)
        else:
            self.register_buffer('confidence', conf_matrix)

    def forward(self):
        """
        Forward pass returns a normalized confidence matrix.
        """
        # 保证 confidence 在 [0, 1] 范围内
        conf = torch.sigmoid(self.confidence)  # 映射到 0~1

        # 强制对角线为 1（自己和自己比较总是可信）
        with torch.no_grad():
            diag_mask = torch.eye(self.num_samples, device=self.confidence.device).bool()
            conf.masked_fill_(diag_mask, 1.0)

        return conf

    def get_pair_confidence(self, index_1, index_2):
        """
        Get confidence for given pairs.

        Args:
            index_1: LongTensor, shape (batch_size), indices of first elements
            index_2: LongTensor, shape (batch_size), indices of second elements

        Returns:
            confidences: Tensor, shape (batch_size)
        """
        return self.confidence[index_1, index_2]

    def update_confidence(self, new_values, index_1, index_2):
        """
        Manually update confidence values for certain pairs.

        Args:
            new_values: Tensor, shape (batch_size), new confidence values
            index_1: LongTensor, shape (batch_size)
            index_2: LongTensor, shape (batch_size)
        """
        with torch.no_grad():
            self.confidence[index_1, index_2] = new_values.to(self.confidence.device)

def compute_prior(better_labels):
    # 自定义逻辑，例如加权平均或其他规则
    positive_ratio = torch.mean((torch.as_tensor(better_labels) == 1).float()).item()
    return 0.8 * positive_ratio + 0.2 * 0.5  # 结合数据分布和默认值

class PairwiseOverparamLoss(nn.Module):
    def __init__(self, num_examp, ratio_balance=0.1, compare_margin=0.01, margin_l2=False):
        """
        Args:
            num_examp: int, total number of samples in the dataset
            ratio_balance: float, weight for balance KL regularization
            compare_margin: float, margin for ranking loss
            margin_l2: bool, whether to use squared hinge loss
        """
        super(PairwiseOverparamLoss, self).__init__()
        self.num_examp = num_examp
        self.ratio_balance = ratio_balance
        self.compare_margin = compare_margin
        self.margin_l2 = margin_l2

        # 可学习参数：每个样本一个偏置项（u）
        self.u = nn.Parameter(torch.zeros(num_examp, 1))  # 偏移量，用于调整每个样本的输出得分

    def forward(self, index_1, index_2, score_1, score_2, better_labels):
        """
        Args:
            index_1: LongTensor, shape (batch_size,) - indices of first architecture in each pair
            index_2: LongTensor, shape (batch_size,) - indices of second architecture in each pair
            score_1: Tensor, shape (batch_size,) - predicted scores of first architecture
            score_2: Tensor, shape (batch_size,) - predicted scores of second architecture
            better_labels: List[bool] or Tensor, shape (batch_size,) - whether arch1 is better than arch2

        Returns:
            loss: scalar tensor
        """
        batch_size = len(better_labels)
        device = score_1.device

        # 将 better_labels 转换为 +1/-1 形式
        #better_pm = 2 * better_labels.float().to(device) - 1
        better_pm = 2 * torch.as_tensor(better_labels).float().to(device) - 1

        # 获取对应的 u 参数
        u_1 = self.u[index_1].squeeze()
        u_2 = self.u[index_2].squeeze()

        # 加上 u 的修正
        adjusted_score_1 = score_1 + u_1
        adjusted_score_2 = score_2 + u_2

        # 计算差值
        diff = adjusted_score_1 - adjusted_score_2

        # 排序损失核心项：margin - better_pm * diff
        zero_ = torch.tensor(0.0, device=device)
        margin = torch.tensor(self.compare_margin, device=device)

        loss_term = margin - better_pm * diff

        if not self.margin_l2:
            # Hinge Loss
            pair_loss = torch.mean(torch.max(zero_, loss_term))
        else:
            # L2 Hinge Loss
            pair_loss = torch.mean(torch.clamp(loss_term, min=0.0).pow(2) / torch.clamp(margin, min=1.0))

        loss = pair_loss

        # 平衡惩罚项（鼓励 u 分布接近 0）
        if self.ratio_balance > 0:
            avg_u = torch.sigmoid(self.u).mean()
            balance_loss = - avg_u * torch.log(avg_u) - (1 - avg_u) * torch.log(1 - avg_u)
            loss += self.ratio_balance * balance_loss

        return loss


class ArchNetwork(nn.Module):
    """
    Base class for architecture-performance predictor.
    """
    def __init__(self):
        nn.Module.__init__(self)

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass
    
    
class NoiseComparator(ArchNetwork):
    """
    Compatible to NN regression-based predictor of architecture performance.
    """

    def __init__(
        self, 
        arch_embedder_cls,
        arch_embedder_cfg: dict = None,
        mlp_hiddens = (200, 200, 200), 
        mlp_dropout: float = 0.1,
        optimizer: dict = {
            "type": "Adam",
            "lr": 0.001
        }, 
        num_total_archs = None,
        scheduler = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ):
        super(NoiseComparator, self).__init__()

        # configs
        assert compare_loss_type in ["binary_cross_entropy", "margin_linear"],\
                "comparing loss type {} not supported".format(compare_loss_type)
        self.compare_loss_type = compare_loss_type
        self.compare_margin = compare_margin
        self.margin_l2 = margin_l2
        self.max_grad_norm = max_grad_norm

        self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim
        # construct MLP from embedding to score
        self.mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout)
        self.over_loss = PairwiseOverparamLoss(
            num_examp=num_total_archs,
            ratio_balance=0.01,
            compare_margin=0.01,
            margin_l2=True
        )
        self.W1 = nn.Parameter(torch.randn(dim, 1))
        self.W2 = nn.Parameter(torch.randn(dim, 1))
        self.learnable_prior = nn.Parameter(torch.tensor(0.5)) 
        self.pair_noise_matrix = PairwiseNoiseMatrix(num_samples=num_total_archs, learnable=True)
        # init optimizer and scheduler
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

        # used for reinit optimizer and lr scheduler
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def reinit_optimizer(self, only_mlp: bool = False):
        parameters = self.mlp.parameters() if only_mlp else self.parameters()
        self.optimizer = utils.init_optimizer(parameters, self.optimizer_cfg)
    
    def reinit_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    @staticmethod
    def construct_mlp(dim: int, mlp_hiddens: Tuple[int], mlp_dropout: float, out_dim: int = 1) -> nn.Module:
        mlp = []
        for hidden_size in mlp_hiddens:
            mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace = False),
                nn.Dropout(p = mlp_dropout))
            )
            dim = hidden_size
        mlp.append(nn.Linear(dim, out_dim))
        mlp = nn.Sequential(*mlp)
        return mlp

    def predict(self, arch, sigmoid=True, tanh=False):
        score = self.mlp(self.arch_embedder(arch)).squeeze(-1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score
    
    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()

    def update_predict(self, archs, labels):
        mse_loss = self.cal_predict_loss(archs, labels)
        return self.update_step(mse_loss)

    def cal_predict_loss(self, archs, labels):
        return self._cal_predict_loss(archs, labels, self.mlp)
    
    def _cal_predict_loss(self, archs, labels, mlp):
        scores = torch.sigmoid(mlp(self.arch_embedder(archs)))
        mse_loss = F.mse_loss(
            scores.squeeze(), scores.new(labels))
        return mse_loss

    def compare(self, arch_1, arch_2):
        # pointwise score and comparen
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        return torch.sigmoid(s_2 - s_1)
    
    def cal_em_loss(self, arch_1, arch_2):
        scores_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        #log_probs_1 = F.log_softmax(scores_1, dim=-1) # shape: (B, 2, C)
        scores_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        #log_probs_2 = F.log_softmax(scores_2, dim=-1) # shape: (B, 2, C)
        log_probs = torch.stack([scores_1, scores_2], dim=1)
        # 构造 soft targets
        em_targets, sup_preds = create_pair_comp_graph(log_probs)
        
        #log_em_targets = F.log_softmax(em_targets, dim=-1)
        # 使用 em_targets 作为监督信号
        #em_loss = torch.nn.CrossEntropyLoss()(em_targets, log_probs)
        em_loss = js_div(log_probs, em_targets)
        return em_loss
    
    def update_compare_semi_1(self, arch_1, arch_2, better_labels, val_arch_1 = None, val_arch_2 = None, index_1= None, index_2= None, margin=None):
        log_weak = self.mlp(self.arch_embedder(val_arch_1)).squeeze()
        log_strong = self.mlp(self.arch_embedder(val_arch_2)).squeeze()
        #y = better_labels.long()
        y = torch.tensor(better_labels).long()
        #conf_ij = self.pair_noise_matrix.get_pair_confidence(index_1, index_2)
        probs_w = F.softmax(log_weak, dim=-1).detach()
        probs_s = F.softmax(log_strong, dim=-1)
        em_loss = self.cal_em_loss(arch_1, arch_2)
        con_loss = compute_con_loss(probs_w, probs_s)
        pair_loss = self.cal_compare_loss(arch_1, arch_2, better_labels, index_1, index_2, margin)
        pair_loss = pair_loss + 0.005*em_loss + 0.005*con_loss #+ 0.005*em_loss #0.05
       
        return self.update_step(pair_loss )
    
            

    def update_compare(self, arch_1, arch_2, better_labels, val_arch_1 = None, val_arch_2 = None, index_1= None, index_2= None, margin=None):
        pair_loss = self.cal_compare_loss(arch_1, arch_2, better_labels, index_1, index_2, margin)
        # if val_arch_1 is not None and val_arch_2 is not None:
        #     ipc_loss = self.ipc_pairwise_update(arch_1, arch_2, better_labels, val_arch_1, val_arch_2)
        #     pair_loss = pair_loss+0.01*ipc_loss  #0.1, 0.
            
            #print("em_loss", em_loss)
        return self.update_step(pair_loss)
    
    
    ###########################
    #General-Framework-Weak-Supervision
    ##############################
    def update_compare_semi(self, arch_1, arch_2, better_labels, val_arch_1 = None, val_arch_2 = None, index_1= None, index_2= None, margin=None):
        pair_loss = self.cal_compare_loss(arch_1, arch_2, better_labels, index_1, index_2, margin)
        if val_arch_1 is not None and val_arch_2 is not None:
            ipc_loss = self.ipc_pairwise_update(arch_1, arch_2, better_labels, val_arch_1, val_arch_2)
            pair_loss = pair_loss+0.1*ipc_loss  #0.1, 0.
            weak_x = val_arch_2
            strong_x = val_arch_2
            log_weak = self.mlp(self.arch_embedder(val_arch_2)).squeeze()
            log_strong = self.mlp(self.arch_embedder(val_arch_2)).squeeze()
            #y = better_labels.long()
            y = torch.tensor(better_labels).long()
            #conf_ij = self.pair_noise_matrix.get_pair_confidence(index_1, index_2)
            probs_w = F.softmax(log_weak, dim=-1).detach()
            probs_s = F.softmax(log_strong, dim=-1)

            # noise_matrix = self.pair_noise_matrix().softmax(dim=0)

            # noise_loss = self.over_loss(index_1, index_2, log_weak, log_strong, better_labels)


            #noise_matrix_col = self.pair_noise_matrix().softmax(dim=-1)[:, y].transpose(0, 1).detach()
            conf_ij = self.pair_noise_matrix.get_pair_confidence(index_1, index_2)
        
            em_loss = compute_em_loss(log_strong, probs_w, conf_ij)
            con_loss = compute_con_loss(log_strong, probs_w)

            #em_loss = compute_em_loss(strong_x, weak_x, conf_ij)
            pair_loss = pair_loss + em_loss #+ 0.01*em_loss
        return self.update_step(pair_loss)
    
    def ipc_pairwise_update(self, arch_1, arch_2, better_labels, val_arch_1, val_arch_2):
        """
        Implicit Pseudo-label Correction for Pairwise Learning
        """
        # val_feat_1 = self.arch_embedder(arch_1)
        # val_feat_2 = self.arch_embedder(arch_2)
        val_feat_1 = self.arch_embedder(arch_1)
        val_feat_2 = self.arch_embedder(arch_2)
        # 获取当前模型参数
        #W1 = model.linear.weight.data  # 假设模型是一个线性分类器

        # 前向传播计算当前得分
        score_1 = torch.mm(val_feat_1, self.W1)
        score_2 = torch.mm(val_feat_2, self.W1)

        # 计算排序损失
        #better_pm = 2 * better_labels.float().to(score_1.device) - 1  # 转换为 +1/-1
        better_pm = 2 * torch.as_tensor(better_labels).float().to(score_1.device) - 1
        margin = 0.1
        loss_rank = torch.mean(torch.clamp(margin - better_pm * (score_1 - score_2), min=0))

        # KL 惩罚项（鼓励输出分布接近先验）
        pred_mean = torch.sigmoid(score_1 - score_2).mean()
        #prior = 0.5  #均匀分布0.09-0.1
        prior = torch.sigmoid(self.learnable_prior) #0.06-0.08
        # positive_ratio = torch.mean((torch.as_tensor(better_labels) == 1).float()).item()
        # prior = positive_ratio  # 动态计算正类的先验概率  #0.06-0.09
        #prior = compute_prior(better_labels) #0.08 best, 0.07-0.09
        penalty_kl = prior * torch.log(prior / pred_mean) + (1 - prior) * torch.log((1 - prior) / (1 - pred_mean))
        loss = loss_rank + 0.1 * penalty_kl

       

        return loss
    
   

    def cal_compare_loss(self, arch_1, arch_2, better_labels, index_1= None, index_2= None, margin = None):
        return self._cal_compare_loss(arch_1, arch_2, better_labels, self.mlp, index_1, index_2, margin)

    def _cal_compare_loss(self, arch_1, arch_2, better_labels, mlp, index_1 =None, index_2 = None, margin = None):
        if self.compare_loss_type == "binary_cross_entropy":
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            compare_score = torch.sigmoid(s_2 - s_1)
            pair_loss = F.binary_cross_entropy(
                    compare_score, compare_score.new(better_labels))

        elif self.compare_loss_type == "margin_linear":
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
            zero_ = s_1.new([0.])
            margin = [self.compare_margin] if margin is None else margin
            margin = s_1.new(margin)
            # if index_1 is not None and index_2 is not None:
            #     conf_ij = self.pair_noise_matrix.get_pair_confidence(index_1, index_2)
            #     if not self.margin_l2:
            #         loss_ind = torch.max(zero_, margin - better_pm * (s_2 - s_1))
            #         weighted_loss = conf_ij * loss_ind
            #         pair_loss = weighted_loss.mean()
            #     else:
            #         loss_ind = torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
            #                 ** 2 / np.maximum(1., margin)
            #         weighted_loss = conf_ij * loss_ind
            #         pair_loss = weighted_loss.mean()
            #     with torch.no_grad():
            #         pred_diff = (s_2-s_1).sigmoid().detach()
            #         new_conf = 1.0 - pred_diff.abs()
            #         self.pair_noise_matrix.update_confidence(new_conf, index_1, index_2)
            # else:
            if not self.margin_l2:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
            else:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
                        ** 2 / np.maximum(1., margin))
            

          

      
           
            # if index_1 is not None and index_2 is not None:
            #     over_loss = self.over_loss(index_1, index_2, s_1, s_2, better_labels)
            #     pair_loss = pair_loss + 0.1*over_loss
        return pair_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location = torch.device("cpu")))

    def on_epoch_start(self, epoch):
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            print("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)



class PointwiseComparator(ArchNetwork):
    """
    Compatible to NN regression-based predictor of architecture performance.
    """

    def __init__(
        self, 
        arch_embedder_cls,
        arch_embedder_cfg: dict = None,
        mlp_hiddens = (200, 200, 200), 
        mlp_dropout: float = 0.1,
        optimizer: dict = {
            "type": "Adam",
            "lr": 0.001
        }, 
        scheduler = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ):
        super(PointwiseComparator, self).__init__()

        # configs
        assert compare_loss_type in ["binary_cross_entropy", "margin_linear"],\
                "comparing loss type {} not supported".format(compare_loss_type)
        self.compare_loss_type = compare_loss_type
        self.compare_margin = compare_margin
        self.margin_l2 = margin_l2
        self.max_grad_norm = max_grad_norm

        self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim
        # construct MLP from embedding to score
        self.mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout)
        
        # init optimizer and scheduler
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

        # used for reinit optimizer and lr scheduler
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def reinit_optimizer(self, only_mlp: bool = False):
        parameters = self.mlp.parameters() if only_mlp else self.parameters()
        self.optimizer = utils.init_optimizer(parameters, self.optimizer_cfg)
    
    def reinit_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    @staticmethod
    def construct_mlp(dim: int, mlp_hiddens: Tuple[int], mlp_dropout: float, out_dim: int = 1) -> nn.Module:
        mlp = []
        for hidden_size in mlp_hiddens:
            mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace = False),
                nn.Dropout(p = mlp_dropout))
            )
            dim = hidden_size
        mlp.append(nn.Linear(dim, out_dim))
        mlp = nn.Sequential(*mlp)
        return mlp

    def predict(self, arch, sigmoid=True, tanh=False):
        score = self.mlp(self.arch_embedder(arch)).squeeze(-1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score
    
    def predict_feature(self, arch, sigmoid=True, tanh=False):
        feats = self.arch_embedder(arch)
     
        return feats
    
    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()

    def update_predict(self, archs, labels):
        mse_loss = self.cal_predict_loss(archs, labels)
        return self.update_step(mse_loss)

    def cal_predict_loss(self, archs, labels):
        return self._cal_predict_loss(archs, labels, self.mlp)
    
    def _cal_predict_loss(self, archs, labels, mlp):
        scores = torch.sigmoid(mlp(self.arch_embedder(archs)))
        mse_loss = F.mse_loss(
            scores.squeeze(), scores.new(labels))
        return mse_loss

    def compare(self, arch_1, arch_2):
        # pointwise score and comparen
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        return torch.sigmoid(s_2 - s_1)

    def update_compare(self, arch_1, arch_2, better_labels, margin=None):
        pair_loss = self.cal_compare_loss(arch_1, arch_2, better_labels, margin)
        return self.update_step(pair_loss)

    def cal_compare_loss(self, arch_1, arch_2, better_labels, margin = None):
        return self._cal_compare_loss(arch_1, arch_2, better_labels, self.mlp, margin)

    def _cal_compare_loss(self, arch_1, arch_2, better_labels, mlp, margin = None):
        if self.compare_loss_type == "binary_cross_entropy":
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            compare_score = torch.sigmoid(s_2 - s_1)
            pair_loss = F.binary_cross_entropy(
                    compare_score, compare_score.new(better_labels))

        elif self.compare_loss_type == "margin_linear":
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
            zero_ = s_1.new([0.])
            margin = [self.compare_margin] if margin is None else margin
            margin = s_1.new(margin)
            if not self.margin_l2:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
            else:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
                        ** 2 / np.maximum(1., margin))
        return pair_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location = torch.device("cpu")))

    def on_epoch_start(self, epoch):
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            print("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


class DynamicEnsemblePointwiseComparator(PointwiseComparator):
    r"""
    Dynamic ensemble pointwise comparator.

    Args:
        auxiliary_head_num (int): Number of the low-fidelity experts.
        use_uniform_confidence (bool): Whether use uniform confidence. Default: `False`.
    """
    NAME = "dynamic_ensemble_pointwise_comparator"

    def __init__(
        self, 
        auxiliary_head_num: int,
        arch_embedder_cls,
        arch_embedder_cfg = None,
        use_uniform_confidence: bool = False,
        mlp_hiddens: Tuple[int] = (200, 200, 200), 
        mlp_dropout: float = 0.1,
        optimizer: Dict[str, Union[str, float]] = {
            "type": "Adam",
            "lr": 0.001
        }, 
        scheduler: Optional[str] = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ) -> None:
        super(DynamicEnsemblePointwiseComparator, self).__init__(
            arch_embedder_cls, 
            arch_embedder_cfg,
            mlp_hiddens, 
            mlp_dropout,
            optimizer, 
            scheduler,
            compare_loss_type, 
            compare_margin,
            margin_l2, 
            max_grad_norm 
        )

        self.auxiliary_head_num = auxiliary_head_num
        self.use_uniform_confidence = use_uniform_confidence

        if self.use_uniform_confidence:
            self.confidence = nn.Parameter(
                torch.randn((1, self.auxiliary_head_num), 
                requires_grad = True)
            )
            nn.init.constant_(self.confidence, 1. / self.auxiliary_head_num)
        else:
            self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))
            dim = self.embedding_dim = self.arch_embedder.out_dim
            # construct MLP from architecture embedding to prediction confidence score
            self.confidence_mlp = self.construct_mlp(
                dim, mlp_hiddens, mlp_dropout, auxiliary_head_num
            )

        self.module_lst = nn.ModuleList([
            PointwiseComparator(
                arch_embedder_cls, 
                arch_embedder_cfg,
                mlp_hiddens, 
                mlp_dropout,
                optimizer, 
                scheduler,
                compare_loss_type, 
                compare_margin,
                margin_l2, 
                max_grad_norm
            ) for i in range(self.auxiliary_head_num)
        ])

        # init optimizer and scheduler
        self.reinit_optimizer(only_mlp = False)
        self.reinit_scheduler()
    
    def init_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)
    
    def init_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def mtl_update_compare(self, auxiliary_datas, margin = None):
        loss = 0.
        for model, (arch_1, arch_2, better_lst) in zip(self.module_lst, auxiliary_datas):
            s_1 = model.predict(arch_1, False, False)
            s_2 = model.predict(arch_2, False, False)
            loss += self._compare_loss(s_1, s_2, better_lst, margin)
        return self.update_step(loss)
    
    def mtl_update_compare_kd(self, auxiliary_datas, real_archs, margin=None, feat_distill_weight=0.1, feat_loss_type="cosine", contrast_weight=0.1, 
                       contrast_temp=0.1,):
        """
        多任务训练 + 特征蒸馏（Feature-level Distillation）
        
        Args:
            auxiliary_datas: List of (arch_1, arch_2, better_lst) for each expert
            feat_distill_weight: 特征蒸馏损失权重
            feat_loss_type: "mse" 或 "cosine"
        """
        total_loss = 0.
        batch_size = len(auxiliary_datas[0][2])  # 假设所有batch大小一致

     
        with torch.no_grad():  # 冻结其他专家
           
                logit_1, feat_1 = self._get_student_predictions_and_features(real_archs)
               
               
              
               

        # Stack to [N, B, H] → [B, N, H]
        all_feats_1 = torch.stack(feat_1, dim=0).transpose(0, 1)  # [B, N, H]
     

        # Step 2: 对每个专家，计算主任务损失 + 特征蒸馏损失
        for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
            # 主任务：学习低保真代理标签（成对比较）
            s_1= model.predict(arch_1, sigmoid=False, tanh=False)  # [B], [B, H]
            s_2= model.predict(arch_2, sigmoid=False, tanh=False)  # [B], [B, H]
            task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
         
            # 特征蒸馏损失：向其他专家的特征学习
            feat_distill_loss = 0.0
            other_indices = [j for j in range(self.auxiliary_head_num) if j != i]
            if len(other_indices) > 0:
                # 获取其他专家的特征
                other_feats_1 = all_feats_1[:, other_indices, :]  # [B, N-1, H]
              
                # 计算平均特征教师
                mean_teacher_feat_1 = torch.mean(other_feats_1, dim=1)  # [B, H]
               

                # 选择损失函数
                if feat_loss_type == "mse":
                    
                    loss_1 = F.mse_loss(feat_1[i], mean_teacher_feat_1)
                    
                elif feat_loss_type == "cosine":
                    # 最小化 1 - cosine_similarity
                    cos_1 = F.cosine_similarity(feat_1[i], mean_teacher_feat_1, dim=1)
                  
                    loss_1 = 1.0 - cos_1.mean()
                    
                else:
                    raise ValueError(f"Unsupported feat_loss_type: {feat_loss_type}")

                feat_distill_loss = loss_1 

            # # 总损失
            # loss = task_loss + feat_distill_weight * feat_distill_loss
            # total_loss += loss

            # ✅ 对比学习损失（核心新增部分）
            if isinstance(feat_1, list):
                feat_1 = torch.stack(feat_1, dim=0)
          

            contrastive_loss = 0.0
            B = feat_1.size(0)

            # 归一化特征（推荐）
            feat_1_norm = F.normalize(feat_1, p=2, dim=1)  # [B, H]

            # 计算相似度矩阵 [B, B]
            #sim_matrix = torch.mm(feat_1_norm, feat_1_norm.t())  # [B, B]
            
                        # feat_1_norm: [B, N, H]
            expert_i_feat = feat_1_norm[:, i, :]  # [B, H] ← 选择第i个专家的特征

            # 现在可以安全转置
            sim_matrix = torch.mm(expert_i_feat, expert_i_feat.t())  # [B, B] ✅
            sim_matrix = sim_matrix / contrast_temp

            # 标签：每个样本的正样本是自己（或你指定的正样本对）
            labels = torch.arange(B, device=feat_1.device)  # [B] ← 假设正样本是自己

            # 掩码掉自己（可选：如果不想让“自己”作为正样本）
            # mask = torch.eye(B, device=feat_1.device).bool()
            # sim_matrix.masked_fill_(mask, float('-inf'))

            # 计算 InfoNCE 损失
            contrastive_loss = F.cross_entropy(sim_matrix, labels)  # [B, B] vs [B] → OK
            


            # 总损失
            loss = task_loss + feat_distill_weight * feat_distill_loss + contrast_weight * contrastive_loss
            total_loss += loss
   
        return self.update_step(total_loss)
    ###################################################################
    def update_compare(self, arch_1, arch_2, better_labels, margin = None) -> float:
        score_1 = self.predict(arch_1, False, False)
        score_2 = self.predict(arch_2, False, False)
        loss = self._compare_loss(score_1, score_2, better_labels, margin)
        return self.update_step(loss)
    
    def update_compare_kd(self, arch_1, arch_2, better_labels, margin=None, alpha=0.5, beta=0.5, T=8.0, gamma=0.1) -> float:
        """
        基于学生特征的两级蒸馏。
        
        Args:
            alpha (float): 第一级特征蒸馏损失的权重。
            beta (float): 第二级蒸馏损失的权重。
            T (float): (可选) 用于分数蒸馏的温度。
            gamma (float): 第一级特征多样性损失的权重。
        """
        # --- 第一步：获取所有模型的预测和特征 ---
        # 组领导的预测 (logits)
        score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False) # [B]
        score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False) # [B]

        # 获取所有辅助学生的预测和特征
        # lf_score_lst_1: List of [B], lf_feature_lst_1: List of [B, H]
        lf_score_lst_1, lf_feature_lst_1 = self._get_student_predictions_and_features(arch_1)
        lf_score_lst_2, lf_feature_lst_2 = self._get_student_predictions_and_features(arch_2)

        student_scores_1 = torch.stack(lf_score_lst_1, dim=1) # [B, N]
        student_scores_2 = torch.stack(lf_score_lst_2, dim=1) # [B, N]
        student_features_1 = torch.stack(lf_feature_lst_1, dim=1) # [B, N, H]
        student_features_2 = torch.stack(lf_feature_lst_2, dim=1) # [B, N, H]

        # --- 第二步：计算主任务损失 ---
        task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)

        # --- 第三步：计算第一级特征蒸馏损失 (Feature-based Peer Distillation) ---
        # 方案A: 鼓励多样性 (Diversity)
        # 计算学生特征之间的余弦相似度矩阵，并最小化其均值，迫使学生特征彼此正交/不相关。
        # 这直接对抗同质化。
        peer_feature_loss = 0.0

        # 对于 arch_1 的特征
        normed_feat_1 = F.normalize(student_features_1, p=2, dim=2) # [B, N, H]
        cosine_sim_matrix_1 = torch.bmm(normed_feat_1, normed_feat_1.transpose(1, 2)) # [B, N, N]
        # 掩盖对角线（自己和自己的相似度）
        mask = torch.eye(self.auxiliary_head_num, device=cosine_sim_matrix_1.device).bool()
        cosine_sim_matrix_1.masked_fill_(mask.unsqueeze(0), 0.0)
        diversity_loss_1 = cosine_sim_matrix_1.abs().mean() # 最小化非对角元素的绝对值

        # 对于 arch_2 的特征
        normed_feat_2 = F.normalize(student_features_2, p=2, dim=2)
        cosine_sim_matrix_2 = torch.bmm(normed_feat_2, normed_feat_2.transpose(1, 2))
        cosine_sim_matrix_2.masked_fill_(mask.unsqueeze(0), 0.0)
        diversity_loss_2 = cosine_sim_matrix_2.abs().mean()

        peer_feature_loss = (diversity_loss_1 + diversity_loss_2) / 2.0

        # --- 第四步：计算第二级蒸馏损失 (Distillation to Leader) ---
        # 方案B: 组领导的置信度应能重构“平均学生特征”
        # 获取当前架构的置信度权重
        confidence_ratio_1 = self._confidence_ratio(arch_1) # [B, N]
        confidence_ratio_2 = self._confidence_ratio(arch_2) # [B, N]

        # 计算加权后的学生特征 (即组领导“看到”的特征)
        weighted_feat_1 = torch.sum(confidence_ratio_1.unsqueeze(-1) * student_features_1, dim=1) # [B, H]
        weighted_feat_2 = torch.sum(confidence_ratio_2.unsqueeze(-1) * student_features_2, dim=1) # [B, H]

        # 计算学生特征的简单平均 (作为“共识”特征)
        mean_feat_1 = torch.mean(student_features_1, dim=1) # [B, H]
        mean_feat_2 = torch.mean(student_features_2, dim=1) # [B, H]

        # 计算MSE损失，让加权特征逼近平均特征
        leader_feat_loss_1 = F.mse_loss(weighted_feat_1, mean_feat_1)
        leader_feat_loss_2 = F.mse_loss(weighted_feat_2, mean_feat_2)
        leader_distill_loss = (leader_feat_loss_1 + leader_feat_loss_2) / 2.0

        # --- 第五步：组合总损失 ---
        # 注意：这里我们用 peer_feature_loss 替代了之前的 peer_distill_loss
        total_loss = task_loss + alpha * peer_feature_loss #+ beta * leader_distill_loss

       

        return self.update_step(total_loss)
    
    # def update_compare(self, arch_1, arch_2, better_labels, margin=None, alpha=0.5, beta=0.5, T=8.0, 
    #                    prior_margin=0.01, conf_threshold=0.1) -> float:
    #     """
    #     两级蒸馏：学生向动态权重教师和静态共识教师学习，但只在教师“高置信度”或“符合先验”时蒸馏。
        
    #     Args:
    #         alpha (float): 第一级特征多样性损失的权重。
    #         beta (float): 第二级蒸馏损失（向两位教师学习）的权重。
    #         T (float): 蒸馏温度。
    #         prior_margin (float): 先验判断时允许的性能下降容忍度。
    #         conf_threshold (float): 预测方差低于此值视为“高置信度”。
    #     """
    #     # --- 第一步：获取所有模型的预测和特征 ---
    #     lf_score_lst_1, lf_feature_lst_1 = self._get_student_predictions_and_features(arch_1)
    #     lf_score_lst_2, lf_feature_lst_2 = self._get_student_predictions_and_features(arch_2)

    #     student_scores_1 = torch.stack(lf_score_lst_1, dim=1) # [B, N]
    #     student_scores_2 = torch.stack(lf_score_lst_2, dim=1) # [B, N]
    #     student_features_1 = torch.stack(lf_feature_lst_1, dim=1) # [B, N, H]
    #     student_features_2 = torch.stack(lf_feature_lst_2, dim=1) # [B, N, H]

    #     # --- 第二步：计算主任务损失 (使用动态权重教师) ---
    #     score_1_dynamic = self.predict(arch_1, sigmoid=False, tanh=False) # [B]
    #     score_2_dynamic = self.predict(arch_2, sigmoid=False, tanh=False) # [B]
    #     task_loss = self._compare_loss(score_1_dynamic, score_2_dynamic, better_labels, margin)

    #     # --- 第三步：计算第一级特征多样性损失 ---
    #     normed_feat_1 = F.normalize(student_features_1, p=2, dim=2)
    #     cosine_sim_matrix_1 = torch.bmm(normed_feat_1, normed_feat_1.transpose(1, 2))
    #     mask = torch.eye(self.auxiliary_head_num, device=cosine_sim_matrix_1.device).bool()
    #     cosine_sim_matrix_1.masked_fill_(mask.unsqueeze(0), 0.0)
    #     diversity_loss_1 = cosine_sim_matrix_1.abs().mean()

    #     normed_feat_2 = F.normalize(student_features_2, p=2, dim=2)
    #     cosine_sim_matrix_2 = torch.bmm(normed_feat_2, normed_feat_2.transpose(1, 2))
    #     cosine_sim_matrix_2.masked_fill_(mask.unsqueeze(0), 0.0)
    #     diversity_loss_2 = cosine_sim_matrix_2.abs().mean()
    #     peer_feature_loss = (diversity_loss_1 + diversity_loss_2) / 2.0

    #     # --- 第四步：计算第二级蒸馏损失 (学生向两位教师学习，但带条件过滤) ---
    #     distill_loss = 0.0
    #     total_valid_samples = 0  # 用于归一化

    #     # 计算静态共识教师的预测
    #     mean_student_1 = torch.mean(student_scores_1, dim=1) # [B]
    #     mean_student_2 = torch.mean(student_scores_2, dim=1) # [B]


    #     # 判断教师是否“高置信度”或“符合先验”
    #     def is_teacher_reliable(scores):
    #         """
    #         判断每个样本的教师预测是否可靠。
    #         返回布尔Tensor [B]，True表示可靠。
    #         """
    #         B = scores.size(0)
    #         reliable = torch.ones(B, dtype=torch.bool, device=scores.device)

    #         # 1. 高置信度：学生预测方差小
    #         if student_scores_1.size(1) > 1:  # 至少两个学生
    #             std_1 = torch.std(student_scores_1, dim=1)  # [B]
    #             reliable &= (std_1 < conf_threshold)

           


    #         return reliable

    #     reliable_1 = is_teacher_reliable(score_1_dynamic)
    #     reliable_2 = is_teacher_reliable(score_2_dynamic)

    #     # 对每个学生模型，计算其与两位教师的蒸馏损失（仅当教师可靠时）
    #     for i in range(self.auxiliary_head_num):
    #         s1_i = student_scores_1[:, i]  # [B]
    #         s2_i = student_scores_2[:, i]  # [B]

    #         # ==== 蒸馏到动态教师 ====
    #         kl_loss_dynamic_1 = F.kl_div(
    #                 F.log_softmax(s1_i.unsqueeze(1) / T, dim=1),
    #                 F.softmax(score_1_dynamic.unsqueeze(1) / T, dim=1).detach(),
    #                 reduction='batchmean'
    #             ) * (T ** 2)
    #         # if reliable_1.any():
    #         #     kl_loss_dynamic_1 = F.kl_div(
    #         #         F.log_softmax(s1_i[reliable_1].unsqueeze(1) / T, dim=1),
    #         #         F.softmax(score_1_dynamic[reliable_1].unsqueeze(1) / T, dim=1).detach(),
    #         #         reduction='batchmean'
    #         #     ) * (T ** 2) if reliable_1.sum() > 0 else 0.0
    #         # else:
    #         #     kl_loss_dynamic_1 = 0.0

            
    #         kl_loss_dynamic_2 = F.kl_div(
    #                 F.log_softmax(s2_i.unsqueeze(1) / T, dim=1),
    #                 F.softmax(score_2_dynamic.unsqueeze(1) / T, dim=1).detach(),
    #                 reduction='batchmean'
    #             ) * (T ** 2) 
    #         # if reliable_2.any():
    #         #     kl_loss_dynamic_2 = F.kl_div(
    #         #         F.log_softmax(s2_i[reliable_2].unsqueeze(1) / T, dim=1),
    #         #         F.softmax(score_2_dynamic[reliable_2].unsqueeze(1) / T, dim=1).detach(),
    #         #         reduction='batchmean'
    #         #     ) * (T ** 2) if reliable_2.sum() > 0 else 0.0
    #         # else:
    #         #     kl_loss_dynamic_2 = 0.0

    #         # ==== 蒸馏到静态共识教师 ====
            
    #         kl_loss_static_1 = F.kl_div(
    #             F.log_softmax(s1_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(mean_student_1.unsqueeze(1) / T, dim=1).detach(),
    #             reduction='batchmean'
    #         ) * (T ** 2) 
    #         # if reliable_1.any():
    #         #     kl_loss_static_1 = F.kl_div(
    #         #         F.log_softmax(s1_i[reliable_1].unsqueeze(1) / T, dim=1),
    #         #         F.softmax(mean_student_1[reliable_1].unsqueeze(1) / T, dim=1).detach(),
    #         #         reduction='batchmean'
    #         #     ) * (T ** 2) if reliable_1.sum() > 0 else 0.0
    #         # else:
    #         #     kl_loss_static_1 = 0.0

            
    #         kl_loss_static_2 = F.kl_div(
    #                 F.log_softmax(s2_i.unsqueeze(1) / T, dim=1),
    #                 F.softmax(mean_student_2.unsqueeze(1) / T, dim=1).detach(),
    #                 reduction='batchmean'
    #             ) * (T ** 2) 
    #         # if reliable_2.any():
    #         #     kl_loss_static_2 = F.kl_div(
    #         #         F.log_softmax(s2_i[reliable_2].unsqueeze(1) / T, dim=1),
    #         #         F.softmax(mean_student_2[reliable_2].unsqueeze(1) / T, dim=1).detach(),
    #         #         reduction='batchmean'
    #         #     ) * (T ** 2) if reliable_2.sum() > 0 else 0.0
    #         # else:
    #         #     kl_loss_static_2 = 0.0

    #         # # 累加损失
    #         # sample_count = reliable_1.sum() + reliable_2.sum()
    #         # if sample_count > 0:
    #         distill_loss += (kl_loss_dynamic_1 + kl_loss_dynamic_2 + kl_loss_static_1 + kl_loss_static_2)
    #             #total_valid_samples += sample_count

    #     # # 归一化蒸馏损失
    #     # if total_valid_samples > 0:
    #     #     distill_loss = distill_loss / total_valid_samples * self.auxiliary_head_num
    #     # else:
    #     #     distill_loss = 0.0
    #     #distill_loss = distill_loss / total_valid_samples * self.auxiliary_head_num

    #     # --- 第五步：组合总损失 ---
    #     total_loss = task_loss + alpha * peer_feature_loss + beta * distill_loss

    #     return self.update_step(total_loss)
    
    #####################################
    # def update_compare(self, arch_1, arch_2, better_labels, margin=None, alpha=1.0, beta=0.5, T=8.0) -> float:
    #     """
    #     两级蒸馏：学生向动态权重教师和静态共识教师学习。
        
    #     Args:
    #         alpha (float): 第一级特征多样性损失的权重。
    #         beta (float): 第二级蒸馏损失（向两位教师学习）的权重。
    #         T (float): 蒸馏温度。
    #     """
    #     # --- 第一步：获取所有模型的预测和特征 ---
    #     # 获取所有辅助学生的预测
    #     lf_score_lst_1 = [model.predict(arch_1, sigmoid=False, tanh=False) for model in self.module_lst] # List of [B]
    #     lf_score_lst_2 = [model.predict(arch_2, sigmoid=False, tanh=False) for model in self.module_lst] # List of [B]

    #     # student_scores_1 = torch.stack(lf_score_lst_1, dim=1) # [B, N]
    #     # student_scores_2 = torch.stack(lf_score_lst_2, dim=1) # [B, N]
    #          # 获取所有辅助学生的预测和特征
    #     # lf_score_lst_1: List of [B], lf_feature_lst_1: List of [B, H]
    #     lf_score_lst_1, lf_feature_lst_1 = self._get_student_predictions_and_features(arch_1)
    #     lf_score_lst_2, lf_feature_lst_2 = self._get_student_predictions_and_features(arch_2)

    #     student_scores_1 = torch.stack(lf_score_lst_1, dim=1) # [B, N]
    #     student_scores_2 = torch.stack(lf_score_lst_2, dim=1) # [B, N]
    #     student_features_1 = torch.stack(lf_feature_lst_1, dim=1) # [B, N, H]
    #     student_features_2 = torch.stack(lf_feature_lst_2, dim=1) # [B, N, H]


    #     # --- 第二步：计算主任务损失 (使用动态权重教师) ---
    #     # 动态权重教师的预测
    #     score_1_dynamic = self.predict(arch_1, sigmoid=False, tanh=False) # [B]
    #     score_2_dynamic = self.predict(arch_2, sigmoid=False, tanh=False) # [B]
    #     task_loss = self._compare_loss(score_1_dynamic, score_2_dynamic, better_labels, margin)

    #     # --- 第三步：计算第一级特征多样性损失 (可选，用于防止同质化) ---
    #     # 如果 PointwiseComparator 支持返回特征，可以在这里计算特征多样性损失
    #     # peer_feature_loss = self._compute_feature_diversity_loss(arch_1, arch_2)
    #     # 此处为了简洁，我们先省略特征部分，专注于预测分数的蒸馏。
    #     peer_feature_loss = 0.0 # 或者您可以保留之前的特征多样性损失逻辑
    #      # 对于 arch_1 的特征
    #     normed_feat_1 = F.normalize(student_features_1, p=2, dim=2) # [B, N, H]
    #     cosine_sim_matrix_1 = torch.bmm(normed_feat_1, normed_feat_1.transpose(1, 2)) # [B, N, N]
    #     # 掩盖对角线（自己和自己的相似度）
    #     mask = torch.eye(self.auxiliary_head_num, device=cosine_sim_matrix_1.device).bool()
    #     cosine_sim_matrix_1.masked_fill_(mask.unsqueeze(0), 0.0)
    #     diversity_loss_1 = cosine_sim_matrix_1.abs().mean() # 最小化非对角元素的绝对值

    #     # 对于 arch_2 的特征
    #     normed_feat_2 = F.normalize(student_features_2, p=2, dim=2)
    #     cosine_sim_matrix_2 = torch.bmm(normed_feat_2, normed_feat_2.transpose(1, 2))
    #     cosine_sim_matrix_2.masked_fill_(mask.unsqueeze(0), 0.0)
    #     diversity_loss_2 = cosine_sim_matrix_2.abs().mean()

    #     peer_feature_loss = (diversity_loss_1 + diversity_loss_2) / 2.0

    #     # --- 第四步：计算第二级蒸馏损失 (学生向两位教师学习) ---
    #     distill_loss = 0.0

    #     # 计算静态共识教师的预测 (简单平均)
    #     mean_student_1 = torch.mean(student_scores_1, dim=1) # [B]
    #     mean_student_2 = torch.mean(student_scores_2, dim=1) # [B]

    #     # 对于每个学生模型，计算其与两位教师的蒸馏损失
    #     for i in range(self.auxiliary_head_num):
    #         # 当前学生的预测
    #         s1_i = student_scores_1[:, i] # [B]
    #         s2_i = student_scores_2[:, i] # [B]

    #         # 损失1: 向动态权重教师学习
    #         kl_loss_dynamic_1 = F.kl_div(
    #             F.log_softmax(s1_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(score_1_dynamic.unsqueeze(1) / T, dim=1).detach(), # 教师目标detach，防止反向传播影响教师
    #             reduction='batchmean'
    #         ) * (T ** 2)
    #         kl_loss_dynamic_2 = F.kl_div(
    #             F.log_softmax(s2_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(score_2_dynamic.unsqueeze(1) / T, dim=1).detach(),
    #             reduction='batchmean'
    #         ) * (T ** 2)

    #         # 损失2: 向静态共识教师学习
    #         kl_loss_static_1 = F.kl_div(
    #             F.log_softmax(s1_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(mean_student_1.unsqueeze(1) / T, dim=1).detach(),
    #             reduction='batchmean'
    #         ) * (T ** 2)
    #         kl_loss_static_2 = F.kl_div(
    #             F.log_softmax(s2_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(mean_student_2.unsqueeze(1) / T, dim=1).detach(),
    #             reduction='batchmean'
    #         ) * (T ** 2)

    #         # 将两个损失相加
    #         distill_loss += (kl_loss_dynamic_1 + kl_loss_dynamic_2 + kl_loss_static_1 + kl_loss_static_2)

    #     distill_loss = distill_loss / self.auxiliary_head_num

    #     # --- 第五步：组合总损失 ---
    #     total_loss = task_loss + alpha * peer_feature_loss + beta * distill_loss
        
    #     return self.update_step(total_loss)

    def _get_student_predictions_and_features(self, arch):
        """
        Helper function to get predictions and features from all auxiliary students.
        Returns:
            scores: List of tensors [B]
            features: List of tensors [B, H]
        """
        scores = []
        features = []
        for model in self.module_lst:
            score, feature = model.predict_score_feature(arch, sigmoid=False, tanh=False)
            scores.append(score)
            features.append(feature)
        return scores, features
    
    
    def update_with_ground_truth(self, archs, true_scores, alpha=0.5, beta=0.5, T=4.0):
        """
        使用稀疏真实标签进行蒸馏训练。
        true_scores: 真实性能分数，Tensor[B]
        """
        # 获取学生预测
     
        lf_score_lst, lf_feature_lst = self._get_student_predictions_and_features(archs)
        student_scores = torch.stack(lf_score_lst, dim=1) # [B, N]
        true_scores = torch.tensor(true_scores, device=student_scores.device, dtype=torch.float32)
        # 动态集成预测（作为教师）
        dynamic_teacher = self.predict(archs, sigmoid=False, tanh=False) # [B]

        # --- 主任务：学生向真实标签学习 ---
        task_loss = F.mse_loss(dynamic_teacher, true_scores)

        # --- 蒸馏1：所有学生向真实标签学习 ---
        distill_to_true = 0.0
        for i in range(self.auxiliary_head_num):
            s_i = student_scores[:, i]
            # KL或MSE蒸馏
            distill_to_true += F.mse_loss(s_i, true_scores.detach())
        distill_to_true /= self.auxiliary_head_num

        # --- 蒸馏2：学生向动态教师学习（一致性） ---
        distill_to_teacher = 0.0
        for i in range(self.auxiliary_head_num):
            s_i = student_scores[:, i]
            kl = F.kl_div(
                F.log_softmax(s_i.unsqueeze(1) / T, dim=1),
                F.softmax(dynamic_teacher.unsqueeze(1) / T, dim=1).detach(),
                reduction='batchmean'
            ) * (T ** 2)
            distill_to_teacher += kl
        distill_to_teacher /= self.auxiliary_head_num

        # --- 特征多样性（可选） ---
        student_features = torch.stack(lf_feature_lst, dim=1)
        normed_feat = F.normalize(student_features, p=2, dim=2)
        cosine_sim = torch.bmm(normed_feat, normed_feat.transpose(1, 2))
        mask = torch.eye(self.auxiliary_head_num, device=cosine_sim.device).bool()
        cosine_sim.masked_fill_(mask.unsqueeze(0), 0.0)
        diversity_loss = cosine_sim.abs().mean()

        # --- 总损失 ---
        total_loss = task_loss + alpha * diversity_loss + beta * (distill_to_true + distill_to_teacher)

        return self.update_step(total_loss)
    
    # def update_compare(self, arch_1, arch_2, better_labels, margin=None, alpha=0.5, beta=0.5, T=4.0) -> float:
    #     """
    #     修改后的更新方法，支持两级蒸馏。
        
    #     Args:
    #         alpha (float): 第一级蒸馏损失的权重。
    #         beta (float): 第二级蒸馏损失的权重。
    #         T (float): 蒸馏温度。
    #     """
    #     # --- 第一步：获取所有模型的原始预测 (logits) ---
    #     # 获取组领导的预测 (logits)
    #     score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False) # [B]
    #     score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False) # [B]

    #     # 获取所有辅助学生的预测 (logits)
    #     # lf_score_lst_1: List of [B], length = auxiliary_head_num
    #     _, lf_score_lst_1 = self.mtl_predict(arch_1, sigmoid=False, tanh=False)
    #     _, lf_score_lst_2 = self.mtl_predict(arch_2, sigmoid=False, tanh=False)

    #     # 将辅助学生的预测堆叠成张量
    #     # student_scores_1: [B, auxiliary_head_num]
    #     student_scores_1 = torch.stack(lf_score_lst_1, dim=1)
    #     student_scores_2 = torch.stack(lf_score_lst_2, dim=1)

    #     # --- 第二步：计算主任务损失 (Task Loss) ---
    #     task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)

    #     # --- 第三步：计算第一级蒸馏损失 (Peer Distillation among Students) ---
    #     peer_distill_loss = 0.0
    #     for i in range(self.auxiliary_head_num):
    #         # 当前学生模型的预测
    #         s1_i = student_scores_1[:, i] # [B]
    #         s2_i = student_scores_2[:, i] # [B]
            
    #         # 其他所有学生模型预测的平均值，作为软目标
    #         # 创建一个掩码，排除当前学生i
    #         mask = torch.ones(self.auxiliary_head_num, device=student_scores_1.device)
    #         mask[i] = 0
    #         mask = mask / (self.auxiliary_head_num - 1) # 归一化
            
    #         soft_target_1 = torch.sum(student_scores_1 * mask.unsqueeze(0), dim=1) # [B]
    #         soft_target_2 = torch.sum(student_scores_2 * mask.unsqueeze(0), dim=1) # [B]
            
    #         # 使用KL散度计算蒸馏损失
    #         # 注意：这里我们直接对logits进行蒸馏，也可以先用softmax(T)转换
    #         kl_loss_1 = F.kl_div(
    #             F.log_softmax(s1_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(soft_target_1.unsqueeze(1) / T, dim=1),
    #             reduction='batchmean'
    #         ) * (T ** 2)
            
    #         kl_loss_2 = F.kl_div(
    #             F.log_softmax(s2_i.unsqueeze(1) / T, dim=1),
    #             F.softmax(soft_target_2.unsqueeze(1) / T, dim=1),
    #             reduction='batchmean'
    #         ) * (T ** 2)
            
    #         peer_distill_loss += (kl_loss_1 + kl_loss_2)

    #     peer_distill_loss = peer_distill_loss / self.auxiliary_head_num

    #     # --- 第四步：计算第二级蒸馏损失 (Distillation to Leader) ---
    #     # 组领导的预测已经是加权平均，理想情况下应与学生的共识一致。
    #     # 我们可以计算组领导预测与学生预测平均值之间的KL散度。
    #     mean_student_1 = torch.mean(student_scores_1, dim=1) # [B]
    #     mean_student_2 = torch.mean(student_scores_2, dim=1) # [B]

    #     leader_kl_loss_1 = F.kl_div(
    #         F.log_softmax(score_1_leader.unsqueeze(1) / T, dim=1),
    #         F.softmax(mean_student_1.unsqueeze(1) / T, dim=1),
    #         reduction='batchmean'
    #     ) * (T ** 2)

    #     leader_kl_loss_2 = F.kl_div(
    #         F.log_softmax(score_2_leader.unsqueeze(1) / T, dim=1),
    #         F.softmax(mean_student_2.unsqueeze(1) / T, dim=1),
    #         reduction='batchmean'
    #     ) * (T ** 2)

    #     leader_distill_loss = (leader_kl_loss_1 + leader_kl_loss_2) / 2.0

    #     # --- 第五步：组合总损失并更新 ---
    #     total_loss = task_loss + beta * leader_distill_loss   #+alpha * peer_distill_loss #
    #     return self.update_step(total_loss)
       

    def mtl_predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tuple[Tensor, List[Tensor]]:
        score = self.predict(arch, sigmoid, tanh)
        lf_score_lst = [model.predict(arch, sigmoid, tanh) for model in self.module_lst]
        return score, lf_score_lst

    def separate_score_predict(self, arch) -> Tensor:
        score_lst = torch.cat([
            model.predict(arch, False, False).unsqueeze(1) 
            for model in self.module_lst],
        1)
        return score_lst

    def weighted_score(self, arch) -> Tensor:
        score_lst = self.separate_score_predict(arch)
        confidence_ratio = self._confidence_ratio(arch)
        score = (confidence_ratio * score_lst)
        return score

    def predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tensor:
        weighted_score = self.weighted_score(arch)
        score = weighted_score.sum(1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def _confidence_ratio(self, archs) -> Tensor:
        if self.use_uniform_confidence:
            confidence_ratio = torch.softmax(self.confidence, 1)
        else:
            arch_embeddings = self.arch_embedder(archs)
            confidence_ratio = torch.softmax(self.confidence_mlp(arch_embeddings), 1) # softmax or sigmoid, maybe ablation
        return confidence_ratio

    def _compare_loss(self, s_1, s_2, better_labels, margin = None):
        s_1 = s_1.squeeze()
        s_2 = s_2.squeeze()
        better_pm = 2 * s_1.new(np.array(better_labels, dtype = np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()
    


class GKD_PointwiseComparator(PointwiseComparator):
    r"""
    Dynamic ensemble pointwise comparator with feature-level distillation and dynamic grouping.
    """

    NAME = "GKD_pointwise_comparator"

    def __init__(
        self,
        auxiliary_head_num: int,
        arch_embedder_cls,
        arch_embedder_cfg=None,
        use_uniform_confidence: bool = False,
        mlp_hiddens: Tuple[int] = (200, 200, 200),
        mlp_dropout: float = 0.1,
        optimizer: Dict[str, Union[str, float]] = {
            "type": "Adam",
            "lr": 0.001
        },
        scheduler: Optional[str] = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ) -> None:
        super(GKD_PointwiseComparator, self).__init__(
            arch_embedder_cls,
            arch_embedder_cfg,
            mlp_hiddens,
            mlp_dropout,
            optimizer,
            scheduler,
            compare_loss_type,
            compare_margin,
            margin_l2,
            max_grad_norm
        )
        self.last_total_loss = None  # 用于记录上一轮总损失
         # 动态分组状态
        self.last_quality_scores = None  # 上次的质量得分
        self.group_update_interval = 5   # 最小更新间隔（防止过于频繁）
        self.group_update_counter = 0    # 更新计数器
      

        self.high_group = None
        self.low_group = None
        self.auxiliary_head_num = auxiliary_head_num
        self.use_uniform_confidence = use_uniform_confidence

        self.aux_loss_weight = 0.01
        self.gating_temp = 2.0
        self.gating_temp_min = 0.5
        self.gating_temp_decay = 0.995
        self.current_epoch = 1  # 跟踪当前 epoch

      
    


        if self.use_uniform_confidence:
            self.confidence = nn.Parameter(
                torch.randn((1, self.auxiliary_head_num), requires_grad=True)
            )
            nn.init.constant_(self.confidence, 1. / self.auxiliary_head_num)
        else:
            self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))
            dim = self.embedding_dim = self.arch_embedder.out_dim
            self.confidence_mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout, auxiliary_head_num)
        self._extract_expert_representations()
        self.group_gating_mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出 [scale_high, scale_low]
            nn.Softplus()      # 保证 >0
        )
        self.module_lst = nn.ModuleList([
            PointwiseComparator(
                arch_embedder_cls,
                arch_embedder_cfg,
                mlp_hiddens,
                mlp_dropout,
                optimizer,
                scheduler,
                compare_loss_type,
                compare_margin,
                margin_l2,
                max_grad_norm
            ) for _ in range(self.auxiliary_head_num)
        ])

        self.reinit_optimizer(only_mlp=False)
        self.reinit_scheduler()

    def _extract_expert_representations(self):
        """提取门控网络权重，并按分组存储"""
        if self.use_uniform_confidence or not hasattr(self, 'confidence_mlp'):
            self.expert_reps = None
            self.high_group_reps = None
            self.low_group_reps = None
            return
        
        # 获取专家表征
        mlp_layers = list(self.confidence_mlp.modules())
        last_linear = None
        for layer in reversed(mlp_layers):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        
        if last_linear is not None:
            self.expert_reps = last_linear.weight.detach().clone()  # [N, D]
            self.expert_reps = F.normalize(self.expert_reps, p=2, dim=1)
            
            # 按分组存储表征
            if self.high_group is not None and self.low_group is not None:
                self.high_group_reps = self.expert_reps[self.high_group]  # [K, D]
                self.low_group_reps = self.expert_reps[self.low_group]    # [L, D]
            else:
                self.high_group_reps = None
                self.low_group_reps = None
        else:
            self.expert_reps = None
            self.high_group_reps = None
            self.low_group_reps = None

    def init_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)

    def init_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def _get_grad_vector(self, model):
        """获取模型所有参数的梯度向量"""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.zeros(1)

    def compute_consistency(self, s_i, s_j):
        """计算两个子网络输出的一致性（余弦相似度）"""
        return F.cosine_similarity(s_i.unsqueeze(1), s_j.unsqueeze(1), dim=-1).mean()

    def compute_entropy(self, logits):
        """计算标签熵（用于评估标签质量）"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def compute_grad_similarity(self, model_i, model_j):
        """计算两个模型梯度方向的相似性"""
        grad_i = self._get_grad_vector(model_i)
        grad_j = self._get_grad_vector(model_j)
        return F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0), dim=-1).mean()

    def _dynamic_grouping(self, real_archs):
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)      # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]

        quality_scores = []
        for i in range(self.auxiliary_head_num):
            scores_i = scores[:, i]
            consistency_list = [
                self.compute_consistency(scores_i, scores[:, j])
                for j in range(self.auxiliary_head_num) if j != i
            ]
            consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
            entropy_i = self.compute_entropy(scores_i)
            grad_sim_i = torch.mean(
                torch.stack([
                    self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
                    for j in range(self.auxiliary_head_num) if j != i
                ])
            )
            quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
            quality_scores.append(quality_score)

        quality_scores = torch.stack(quality_scores)  # [N]
        threshold = torch.median(quality_scores)
        high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]

        return high_group, low_group, quality_scores  # ✅ 返回 quality_scores



###########################################################################################
    # def mtl_update_compare_kd(self, auxiliary_datas, real_archs, epoch, margin=None,
    #                       feat_distill_weight=0.001, feat_loss_type="cosine",
    #                       contrast_weight=0.001, contrast_temp=0.1):
        
    #     # feat_distill_weight=0.01, feat_loss_type="cosine",
    #     #                   contrast_weight=0.01, contrast_temp=0.1
    #     total_loss = 0.

    #     # --- 第一步：获取特征并动态分组 ---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)
    #     features = torch.stack(features, dim=1)

    #     if epoch % 5 == 1:  # 每10个epoch更新一次分组
    #         self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)

    #     high_group = self.high_group
    #     low_group = self.low_group

    #     # # 动态分组（只需分组，无需 quality_scores）
    #     # quality_scores = []
    #     # for i in range(self.auxiliary_head_num):
    #     #     scores_i = scores[:, i]
    #     #     consistency_list = [
    #     #         self.compute_consistency(scores_i, scores[:, j])
    #     #         for j in range(self.auxiliary_head_num) if j != i
    #     #     ]
    #     #     consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
    #     #     entropy_i = self.compute_entropy(scores_i)
    #     #     grad_sim_i = torch.mean(
    #     #         torch.stack([
    #     #             self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
    #     #             for j in range(self.auxiliary_head_num) if j != i
    #     #         ])
    #     #     )
    #     #     quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
    #     #     quality_scores.append(quality_score)

    #     # quality_scores = torch.stack(quality_scores)
    #     # threshold = torch.median(quality_scores)
    #     # high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
    #     # low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]

    #     # --- 第二步：主任务损失 ---
    #     task_loss_value = 0.0
    #     for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
    #         s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
    #         s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
    #         task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
    #         total_loss += task_loss
    #         task_loss_value += task_loss.item()
    #     task_loss_value /= len(auxiliary_datas)  # 平均主任务损失

    #     # --- 第三步：基于训练损失的动态权重 ---
    #     lambda_high = 1.0
    #     lambda_low = 0.5
    #     lambda_inter = 0.5

    #     if self.last_total_loss is not None:
    #         # 计算相对损失改进率
    #         loss_improvement = (self.last_total_loss - task_loss_value) / (self.last_total_loss + 1e-8)
            
    #         # 映射到 [0, 1] 区间（可选：使用 sigmoid 或 clamp）
    #         # 如果损失下降（improvement > 0），增强跨组蒸馏
    #         # 如果损失上升（improvement < 0），减弱跨组蒸馏
    #         adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 5.0), -1.0, 1.0).item()
            
    #         # 动态调整
    #         lambda_inter = 0.5 + 0.5 * max(adaptive_factor, 0.0)  # [0.5, 1.0]
    #         lambda_low = 0.5 - 0.4 * max(adaptive_factor, 0.0)    # [0.1, 0.5]
    #     else:
    #         # 第一轮，使用默认权重
    #         lambda_inter = 0.5
    #         lambda_low = 0.5

    #     # 更新历史损失
    #     self.last_total_loss = task_loss_value

    #     # --- 第四步：三部分协作蒸馏 ---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_intra = 0.0
    #     distill_loss_inter = 0.0
    #     B, N, H = features.shape

    #     # 1. 高性能组内蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             for j in high_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_high_intra += (1.0 - sim)

    #     # 2. 低绩效组内蒸馏
    #     if len(low_group) > 1:
    #         for i in low_group:
    #             for j in low_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_low_intra += (1.0 - sim)

    #     # 3. 跨组蒸馏（Low → High）
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         for i in low_group:
    #             for j in high_group:
    #                 feat_i = features[:, i, :]
    #                 feat_j = features[:, j, :]
    #                 sim = torch.mean(F.cosine_similarity(
    #                     F.normalize(feat_i, p=2, dim=1),
    #                     F.normalize(feat_j, p=2, dim=1),
    #                     dim=1
    #                 ))
    #                 distill_loss_inter += (1.0 - sim)

    #     # --- 第五步：组合蒸馏损失 ---
    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low * distill_loss_low_intra +
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss


    #     contrastive_loss_total = 0.0
    #     # 高性能组内对比（数值稳定版）
    #     if len(high_group) > 1:
    #         high_feats = features[:, high_group, :].view(-1, H)  # [B*K, H]
    #         high_feats = F.normalize(high_feats, dim=1)
    #         sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp  # [B*K, B*K]

    #         # 构建正样本 mask：同一样本的不同子网络
    #         labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))  # [B*K]
    #         pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B*K, B*K]
    #         pos_mask.fill_diagonal_(False)  # 排除自己

    #         # 计算 InfoNCE 损失（数值稳定）
    #         exp_sim = torch.exp(sim_mat)
    #         log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
    #         contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
    #         contrastive_loss_total += contrastive_loss

    #     # 跨组对比（Low vs High 共识）
    #     if len(high_group) > 0 and len(low_group) > 0:
    #         high_consensus = features[:, high_group, :].mean(dim=1)  # [B, H]
    #         low_feats = features[:, low_group, :].view(-1, H)        # [B*L, H]
    #         low_feats = F.normalize(low_feats, dim=1)
    #         high_consensus = F.normalize(high_consensus, dim=1)
    #         sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp  # [B*L, B]

    #         # 标签：每个低绩效特征对应同一样本的高性能共识
    #         labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))  # [B*L]

    #         # 数值稳定版
    #         exp_sim = torch.exp(sim_mat)
    #         log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
    #         contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
    #         contrastive_loss_total += contrastive_loss

    #     total_loss += contrast_weight * contrastive_loss_total
    #     self.last_total_loss = task_loss_value

    #     ######## --- 第六步：对比学习（可选）---
    #     # contrastive_loss = 0.0
    #     # if B > 1:
    #     #     feat_norm = F.normalize(features, p=2, dim=2)
    #     #     sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
    #     #     labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
    #     #     sim_matrix_flat = sim_matrix.view(-1, N)
    #     #     contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
    #     # total_loss += contrast_weight * contrastive_loss

    #     return self.update_step(total_loss)
    # # feat_distill_weight=0.0001, feat_loss_type="cosine",
    # #                   contrast_weight=0.0001 

    # def update_compare_kd(self, arch_1, arch_2, better_labels, real_archs, epoch, margin=None,
    #                   feat_distill_weight=0.001, feat_loss_type="cosine",
    #                   contrast_weight=0.001, contrast_temp=0.1): #lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0
        
    #     # feat_distill_weight=0.001, feat_loss_type="cosine",
    #     #               contrast_weight=0.001, contrast_temp=0.1, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0
    #     """
    #     基于动态分组的融合蒸馏策略
    #     """
    #     total_loss = 0.

    #     # --- 第一步：获取所有子网络在 real_archs 上的预测和特征 ---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)  # [B, N]
    #     features = torch.stack(features, dim=1)  # [B, N, H]


    #     if epoch % 5 == 1:  # 每10个epoch更新一次分组
    #         self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)
    #     #high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
    #     high_group = self.high_group
    #     low_group = self.low_group


    #     # --- 第三步：主任务损失 ---
    #     score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False)  # [B]
    #     score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False)  # [B]
    #     task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)
    #     total_loss += task_loss #0.06



    

    #     # --- 第三步：基于训练损失的动态权重 ---
    #     lambda_high = 1.0
    #     lambda_low = 0.5
    #     lambda_inter = 1.0

    #     if self.last_total_loss is not None:
    #         # 计算相对损失改进率
    #         loss_improvement = (self.last_total_loss - task_loss) / (self.last_total_loss + 1e-8)
            
    #         # 映射到 [0, 1] 区间（可选：使用 sigmoid 或 clamp）
    #         # 如果损失下降（improvement > 0），增强跨组蒸馏
    #         # 如果损失上升（improvement < 0），减弱跨组蒸馏
    #         adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 5.0), -1.0, 1.0).item()
            
    #         # 动态调整
    #         lambda_inter = 0.5 + 0.5 * max(adaptive_factor, 0.0)  # [0.5, 1.0]
    #         lambda_low = 0.5 - 0.4 * max(adaptive_factor, 0.0)    # [0.1, 0.5]
    #     else:
    #         # 第一轮，使用默认权重
    #         lambda_inter = 1.0
    #         lambda_low = 0.5

    #     # 更新历史损失
    #     self.last_total_loss = task_loss

    #     # # --- 第四步：三部分协作蒸馏 ---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_intra = 0.0
    #     distill_loss_inter = 0.0
    #     B, N, H = features.shape

    #     # 1. 高性能组内蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             for j in high_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_high_intra += (1.0 - sim)

    #     # 2. 低绩效组内蒸馏
    #     if len(low_group) > 1:
    #         for i in low_group:
    #             for j in low_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_low_intra += (1.0 - sim)

    #     # 3. 跨组蒸馏（Low → High）
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         for i in low_group:
    #             for j in high_group:
    #                 feat_i = features[:, i, :]
    #                 feat_j = features[:, j, :]
    #                 sim = torch.mean(F.cosine_similarity(
    #                     F.normalize(feat_i, p=2, dim=1),
    #                     F.normalize(feat_j, p=2, dim=1),
    #                     dim=1
    #                 ))
    #                 distill_loss_inter += (1.0 - sim)

    #     # --- 第五步：组合蒸馏损失 ---
    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low * distill_loss_low_intra +
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss


    #     contrastive_loss_total = 0.0
    #     # 高性能组内对比（数值稳定版）
    #     if len(high_group) > 1:
    #         high_feats = features[:, high_group, :].view(-1, H)  # [B*K, H]
    #         high_feats = F.normalize(high_feats, dim=1)
    #         sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp  # [B*K, B*K]

    #         # 构建正样本 mask：同一样本的不同子网络
    #         labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))  # [B*K]
    #         pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B*K, B*K]
    #         pos_mask.fill_diagonal_(False)  # 排除自己

    #         # 计算 InfoNCE 损失（数值稳定）
    #         exp_sim = torch.exp(sim_mat)
    #         log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
    #         contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
    #         contrastive_loss_total += contrastive_loss

    #     # 跨组对比（Low vs High 共识）
    #     if len(high_group) > 0 and len(low_group) > 0:
    #         high_consensus = features[:, high_group, :].mean(dim=1)  # [B, H]
    #         low_feats = features[:, low_group, :].view(-1, H)        # [B*L, H]
    #         low_feats = F.normalize(low_feats, dim=1)
    #         high_consensus = F.normalize(high_consensus, dim=1)
    #         sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp  # [B*L, B]

    #         # 标签：每个低绩效特征对应同一样本的高性能共识
    #         labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))  # [B*L]

    #         # 数值稳定版
    #         exp_sim = torch.exp(sim_mat)
    #         log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
    #         contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
    #         contrastive_loss_total += contrastive_loss

    #     total_loss += contrast_weight * contrastive_loss_total
    #     self.last_total_loss = task_loss

    #     ########--- 第六步：对比学习（可选）---
    #     contrastive_loss = 0.0
    #     if B > 1:
    #         feat_norm = F.normalize(features, p=2, dim=2)
    #         sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
    #         labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
    #         sim_matrix_flat = sim_matrix.view(-1, N)
    #         contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
    #     total_loss += contrast_weight * contrastive_loss

    #     return self.update_step(total_loss)


##############################################################################################

    def mtl_update_compare_kd(self, auxiliary_datas, real_archs, epoch, margin=None,
                          feat_distill_weight=0.001, feat_loss_type="cosine",
                          contrast_weight=0.001, contrast_temp=0.1,svd_loss_weight=0.001, cov_loss_weight=0.05):
        
        # feat_distill_weight=0.01, feat_loss_type="cosine",
        #                   contrast_weight=0.01, contrast_temp=0.1
        total_loss = 0.

        # --- 第一步：获取特征并动态分组 ---
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)  # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]
        B, N, H = features.shape

        # --- 第二步：动态分组（每5个epoch更新）---
        # if epoch % 5 == 1:
        #     quality_scores = []
        #     for i in range(self.auxiliary_head_num):
        #         scores_i = scores[:, i]  # [B]
        #         # 输出一致性
        #         consistency_list = [
        #             self.compute_consistency(scores_i, scores[:, j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ]
        #         consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #         # ✅ 用预测方差替代无效的熵（标量任务）
        #         variance_i = torch.var(scores_i)
        #         # ✅ 移除梯度相似性（高开销）
        #         quality_score = 0.7 * consistency_i + 0.3 * (1.0 / (1.0 + variance_i))
        #         quality_scores.append(quality_score)

        #     quality_scores = torch.stack(quality_scores)
        #     threshold = torch.median(quality_scores)
        #     high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        #     low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
            
        #     # 处理空组情况
        #     if len(high_group) == 0:
        #         high_group = [torch.argmax(quality_scores).item()]
        #         low_group = [i for i in range(self.auxiliary_head_num) if i != high_group[0]]
        #     if len(low_group) == 0:
        #         low_group = [torch.argmin(quality_scores).item()]
        #         high_group = [i for i in range(self.auxiliary_head_num) if i != low_group[0]]
                
        #     self.high_group = high_group
        #     self.low_group = low_group
        # else:
        #     # 使用缓存的分组
        #     high_group = self.high_group if hasattr(self, 'high_group') else list(range(self.auxiliary_head_num))
        #     low_group = self.low_group if hasattr(self, 'low_group') else []

        # # 动态分组（只需分组，无需 quality_scores）
        # quality_scores = []
        # for i in range(self.auxiliary_head_num):
        #     scores_i = scores[:, i]
        #     consistency_list = [
        #         self.compute_consistency(scores_i, scores[:, j])
        #         for j in range(self.auxiliary_head_num) if j != i
        #     ]
        #     consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #     entropy_i = self.compute_entropy(scores_i)
        #     grad_sim_i = torch.mean(
        #         torch.stack([
        #             self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ])
        #     )
        #     quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
        #     quality_scores.append(quality_score)

        # quality_scores = torch.stack(quality_scores)
        # threshold = torch.median(quality_scores)
        # high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        # low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
        
        #if epoch % 5 == 1:  # 每10个epoch更新一次分组
        self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)
        #high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
        high_group = self.high_group
        low_group = self.low_group
        # --- 第二步：主任务损失 ---
        task_loss_value = 0.0
        for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
            s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
            s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
           
            task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
           
            total_loss += task_loss
            task_loss_value += task_loss.item()
        task_loss_value /= len(auxiliary_datas)  # 平均主任务损失

        # --- 第四步：基于训练损失的动态权重 ---
        lambda_high = 1.0
        lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
        lambda_inter = 0.5

        if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
            if not hasattr(self, 'smoothed_loss'):
                self.smoothed_loss = self.last_total_loss
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
            loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
            adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
            lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
            # lambda_low_consensus 保持弱权重
        else:
            self.smoothed_loss = task_loss.item()

        self.last_total_loss = task_loss.item()

        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0

        # 1. 高性能组内蒸馏
        if len(high_group) > 1:
            for i in high_group:
                for j in high_group:
                    if i != j:
                        feat_i = features[:, i, :]
                        feat_j = features[:, j, :]
                        sim = torch.mean(F.cosine_similarity(
                            F.normalize(feat_i, p=2, dim=1),
                            F.normalize(feat_j, p=2, dim=1),
                            dim=1
                        ))
                        distill_loss_high_intra += (1.0 - sim)

        # 2. ✅ 低绩效共识教师蒸馏（安全替代）
        if len(low_group) > 1:
            low_features = features[:, low_group, :]  # [B, L, H]
            consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            for i in low_group:
                feat_i = features[:, i, :]
                sim = torch.mean(F.cosine_similarity(
                    F.normalize(feat_i, p=2, dim=1),
                    F.normalize(consensus_low, p=2, dim=1),
                    dim=1
                ))
                distill_loss_low_consensus += (1.0 - sim)
        elif len(low_group) == 1:
            distill_loss_low_consensus = 0.0

        # 3. 跨组蒸馏（Low → High）带置信度过滤
        if len(low_group) > 0 and len(high_group) > 0:
            high_scores = scores[:, high_group]  # [B, K]
            high_var = torch.var(high_scores, dim=1)  # [B]
            teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
            if teach_mask.any():
                for i in low_group:
                    for j in high_group:
                        feat_i = features[:, i, :][teach_mask]
                        feat_j = features[:, j, :][teach_mask]
                        if feat_i.size(0) > 0:
                            sim = torch.mean(F.cosine_similarity(
                                F.normalize(feat_i, p=2, dim=1),
                                F.normalize(feat_j, p=2, dim=1),
                                dim=1
                            ))
                            distill_loss_inter += (1.0 - sim)

        # --- 第六步：组合蒸馏损失 ---
        total_distill_loss = (
            lambda_high * distill_loss_high_intra +
            lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
            lambda_inter * distill_loss_inter
        )
        total_loss += feat_distill_weight * total_distill_loss

   


        contrastive_loss_total = 0.0
        # 高性能组内对比（数值稳定版）
        if len(self.high_group) > 1:
            high_feats = features[:, self.high_group, :].view(-1, H)  # [B*K, H]
            high_feats = F.normalize(high_feats, dim=1)
            sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp

            labels = torch.arange(B, device=features.device).repeat_interleave(len(self.high_group))
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask.fill_diagonal_(False)

            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
            contrastive_loss_total += contrastive_loss

            ##################### === 添加 SVD + Cov 正则化 ===
            if svd_loss_weight > 0 or cov_loss_weight > 0:
                # 对高性能组特征应用正则化
                svd_loss_high = compute_svd_loss(high_feats, svd_loss_weight)
                cov_loss_high = compute_cov_loss(high_feats, cov_loss_weight)
                contrastive_loss_total += svd_loss_high + cov_loss_high
            #####################################################################
            

    
        ####### 跨组对比（Low vs High 共识）
        if len(self.high_group) > 0 and len(self.low_group) > 0:
            high_consensus = features[:, self.high_group, :].mean(dim=1)  # [B, H]
            low_feats = features[:, self.low_group, :].view(-1, H)        # [B*L, H]
            low_feats = F.normalize(low_feats, dim=1)
            high_consensus = F.normalize(high_consensus, dim=1)
            sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp

            labels = torch.arange(B, device=features.device).repeat_interleave(len(self.low_group))
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
            contrastive_loss_total += contrastive_loss

        #     # # === 添加 SVD + Cov 正则化 ===
            if svd_loss_weight > 0 or cov_loss_weight > 0:
                # 对低绩效组特征应用正则化
                svd_loss_low = compute_svd_loss(low_feats, svd_loss_weight)
                cov_loss_low = compute_cov_loss(low_feats, cov_loss_weight)
                # 对高性能共识特征应用正则化
                svd_loss_high_consensus = compute_svd_loss(high_consensus, svd_loss_weight)
                cov_loss_high_consensus = compute_cov_loss(high_consensus, cov_loss_weight)
                contrastive_loss_total += svd_loss_low + cov_loss_low + svd_loss_high_consensus + cov_loss_high_consensus

        #     ##################################################################
         
        
          

        total_loss += contrast_weight * contrastive_loss_total
        self.last_total_loss = task_loss_value

        ######## --- 第六步：对比学习（可选）---
        # contrastive_loss = 0.0
        # if B > 1:
        #     feat_norm = F.normalize(features, p=2, dim=2)
        #     sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
        #     labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
        #     sim_matrix_flat = sim_matrix.view(-1, N)
        #     contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
        # total_loss += contrast_weight * contrastive_loss

        return self.update_step(total_loss)
    # feat_distill_weight=0.0001, feat_loss_type="cosine",
    #                   contrast_weight=0.0001 

    #    parser.add_argument(
    #     '--cov-loss-weight',
    #     type=float,
    #     default=0.04,
    # ) 
    # parser.add_argument(
    #     '--svd-loss-weight',
    #     type=float,
    #     default=0.001,
    # ) 

    def update_compare_kd(self, arch_1, arch_2, better_labels, real_archs, epoch, margin=None,
                      feat_distill_weight=0.001, feat_loss_type="cosine",
                      contrast_weight=0.001, contrast_temp=0.1):
        """
        基于动态分组的安全协作蒸馏策略
        """
        total_loss = 0.

        # --- 第一步：获取所有子网络在 real_archs 上的预测和特征 ---
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)  # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]
        B, N, H = features.shape

        # --- 第二步：动态分组（每5个epoch更新）---
        # if epoch % 5 == 1:
        #     quality_scores = []
        #     for i in range(self.auxiliary_head_num):
        #         scores_i = scores[:, i]  # [B]
        #         # 输出一致性
        #         consistency_list = [
        #             self.compute_consistency(scores_i, scores[:, j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ]
        #         consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #         # ✅ 用预测方差替代无效的熵（标量任务）
        #         variance_i = torch.var(scores_i)
        #         # ✅ 移除梯度相似性（高开销）
        #         quality_score = 0.7 * consistency_i + 0.3 * (1.0 / (1.0 + variance_i))
        #         quality_scores.append(quality_score)

        #     quality_scores = torch.stack(quality_scores)
        #     threshold = torch.median(quality_scores)
        #     high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        #     low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
            
        #     # 处理空组情况
        #     if len(high_group) == 0:
        #         high_group = [torch.argmax(quality_scores).item()]
        #         low_group = [i for i in range(self.auxiliary_head_num) if i != high_group[0]]
        #     if len(low_group) == 0:
        #         low_group = [torch.argmin(quality_scores).item()]
        #         high_group = [i for i in range(self.auxiliary_head_num) if i != low_group[0]]
                
        #     self.high_group = high_group
        #     self.low_group = low_group
        # else:
        #     # 使用缓存的分组
        #     high_group = self.high_group if hasattr(self, 'high_group') else list(range(self.auxiliary_head_num))
        #     low_group = self.low_group if hasattr(self, 'low_group') else []
        #if epoch % 5 == 1:  # 每10个epoch更新一次分组
        self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)
        #high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
        high_group = self.high_group
        low_group = self.low_group
        # --- 第三步：主任务损失 ---
        score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False)  # [B]
        score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False)  # [B]
        task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)
        total_loss += task_loss

        # --- 第四步：基于训练损失的动态权重 ---
        lambda_high = 1.0
        lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
        lambda_inter = 0.5

        if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
            if not hasattr(self, 'smoothed_loss'):
                self.smoothed_loss = self.last_total_loss
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
            loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
            adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
            lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
            # lambda_low_consensus 保持弱权重
        else:
            self.smoothed_loss = task_loss.item()

        self.last_total_loss = task_loss.item()

        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0

        # 1. 高性能组内蒸馏
        if len(high_group) > 1:
            for i in high_group:
                for j in high_group:
                    if i != j:
                        feat_i = features[:, i, :]
                        feat_j = features[:, j, :]
                        sim = torch.mean(F.cosine_similarity(
                            F.normalize(feat_i, p=2, dim=1),
                            F.normalize(feat_j, p=2, dim=1),
                            dim=1
                        ))
                        distill_loss_high_intra += (1.0 - sim)

        # 2. ✅ 低绩效共识教师蒸馏（安全替代）
        if len(low_group) > 1:
            low_features = features[:, low_group, :]  # [B, L, H]
            consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            for i in low_group:
                feat_i = features[:, i, :]
                sim = torch.mean(F.cosine_similarity(
                    F.normalize(feat_i, p=2, dim=1),
                    F.normalize(consensus_low, p=2, dim=1),
                    dim=1
                ))
                distill_loss_low_consensus += (1.0 - sim)
        elif len(low_group) == 1:
            distill_loss_low_consensus = 0.0

        # 3. 跨组蒸馏（Low → High）带置信度过滤
        if len(low_group) > 0 and len(high_group) > 0:
            high_scores = scores[:, high_group]  # [B, K]
            high_var = torch.var(high_scores, dim=1)  # [B]
            teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
            if teach_mask.any():
                for i in low_group:
                    for j in high_group:
                        feat_i = features[:, i, :][teach_mask]
                        feat_j = features[:, j, :][teach_mask]
                        if feat_i.size(0) > 0:
                            sim = torch.mean(F.cosine_similarity(
                                F.normalize(feat_i, p=2, dim=1),
                                F.normalize(feat_j, p=2, dim=1),
                                dim=1
                            ))
                            distill_loss_inter += (1.0 - sim)

        # --- 第六步：组合蒸馏损失 ---
        total_distill_loss = (
            lambda_high * distill_loss_high_intra +
            lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
            lambda_inter * distill_loss_inter
        )
        total_loss += feat_distill_weight * total_distill_loss

        # --- 第七步：对比知识蒸馏（仅组内+跨组）---
        contrastive_loss_total = 0.0

        ######### 高性能组内对比
        if len(high_group) > 1:
            high_feats = features[:, high_group, :].view(-1, H)
            high_feats = F.normalize(high_feats, dim=1)
            sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp
            labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask.fill_diagonal_(False)
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
            contrastive_loss_total += contrastive_loss

        # # 跨组对比
        if len(high_group) > 0 and len(low_group) > 0:
            high_consensus = features[:, high_group, :].mean(dim=1)
            low_feats = features[:, low_group, :].view(-1, H)
            low_feats = F.normalize(low_feats, dim=1)
            high_consensus = F.normalize(high_consensus, dim=1)
            sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp
            labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
            contrastive_loss_total += contrastive_loss

        total_loss += contrast_weight * contrastive_loss_total
        return self.update_step(total_loss)
    

    def _compute_current_quality_scores(self, scores):
        """
        计算当前批次的质量得分（用于判断是否需要更新分组）
        """
        B, N = scores.shape
        quality_scores = []
        
        for i in range(N):
            scores_i = scores[:, i]
            # 一致性
            if N > 1:
                consistency_list = [
                    self.compute_consistency(scores_i, scores[:, j])
                    for j in range(N) if j != i
                ]
                consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else torch.tensor(0.0)
            else:
                consistency_i = torch.tensor(0.0)
            
            # 稳定性（用方差替代熵）
            stability_i = 1.0 / (1.0 + torch.var(scores_i))
            
            quality_score = 0.7 * consistency_i + 0.3 * stability_i
            quality_scores.append(quality_score)
        
        return torch.stack(quality_scores, dim=0)  # [N]
    

########################################################################################################
    def _get_student_predictions_and_features(self, arch):
        """
        Helper function to get predictions and features from all auxiliary students.
        Returns:
            scores: List of tensors [B]
            features: List of tensors [B, H]
        """
        scores = []
        features = []
        for model in self.module_lst:
            feature = model.predict_feature(arch, sigmoid=False, tanh=False)
            score = model.predict(arch)
            scores.append(score)
            features.append(feature)
        return scores, features


    def mtl_predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tuple[Tensor, List[Tensor]]:
        score = self.predict(arch, sigmoid, tanh)
        lf_score_lst = [model.predict(arch, sigmoid, tanh) for model in self.module_lst]
        return score, lf_score_lst

    def separate_score_predict(self, arch) -> Tensor:
        score_lst = torch.cat([
            model.predict(arch, False, False).unsqueeze(1)
            for model in self.module_lst],
        1)
        return score_lst

    def weighted_score(self, arch) -> Tensor:
        score_lst = self.separate_score_predict(arch)
        confidence_ratio = self._confidence_ratio(arch)
        score = (confidence_ratio * score_lst)
        return score

    def predict_1(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tensor:
        weighted_score = self.weighted_score(arch)
        score = weighted_score.sum(1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def predict(self, arch, sigmoid=True, tanh=False):
        score_lst = self.separate_score_predict(arch)  # [B, N]
        base_conf = self._confidence_ratio(arch)       # [B, N]

        if self.high_group and self.low_group:
            emb = self.arch_embedder(arch)  # [B, D]
            scales = self.group_gating_mlp(emb)  # [B, 2]
            scale_high, scale_low = scales[:, 0:1], scales[:, 1:2]  # [B,1]

            group_mask = torch.zeros_like(base_conf)
            group_mask[:, self.high_group] = 1
            group_mask[:, self.low_group] = 0

            dynamic_scale = group_mask * scale_high + (1 - group_mask) * scale_low  # [B, N]
            weighted_score = base_conf * dynamic_scale * score_lst
        else:
            weighted_score = base_conf * score_lst

        score = weighted_score.sum(dim=1)
        return torch.sigmoid(score) if sigmoid else score

    def _get_expert_similarity_matrix(self):
        """获取专家相似度矩阵 [N, N]"""
        if self.expert_reps is None:
            # 无法计算多样性，返回零矩阵（无惩罚）
            N = self.auxiliary_head_num
            return torch.zeros(N, N, device=next(self.parameters()).device)
        
        # 余弦相似度矩阵
        sim_matrix = torch.mm(self.expert_reps, self.expert_reps.t())  # [N, N]
        return sim_matrix
    
    # def predict(self, arch, sigmoid=True, tanh=False, diversity_weight=0.1) -> Tensor:
    #     """
    #     多样性感知的专家集成预测
    #     """
    #     # 确保 expert_reps 在正确设备上
    #     if self.expert_reps is not None:
    #         device = next(self.parameters()).device
    #         if self.expert_reps.device != device:
    #             self.expert_reps = self.expert_reps.to(device)
        
    #     # 获取原始门控概率 [B, N]
    #     original_gating = self._confidence_ratio(arch)
        
    #     if self.expert_reps is None or diversity_weight <= 0:
    #         score_lst = self.separate_score_predict(arch)
    #         score = (original_gating * score_lst).sum(dim=1)
    #         if sigmoid:
    #             score = torch.sigmoid(score)
    #         elif tanh:
    #             score = torch.tanh(score)
    #         return score
        
    #     # 获取专家相似度矩阵 [N, N]
    #     expert_similarity = self._get_expert_similarity_matrix()
        
    #     # 确保 expert_similarity 与 original_gating 同设备
    #     if expert_similarity.device != original_gating.device:
    #         expert_similarity = expert_similarity.to(original_gating.device)
        
    #     # 计算惩罚
    #     penalties = torch.mm(original_gating, expert_similarity.t())  # [B, N]
        
    #     # 多样性调整
    #     adjusted_gating = original_gating * torch.exp(-diversity_weight * penalties)
    #     adjusted_gating = adjusted_gating / adjusted_gating.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
    #     # 集成预测
    #     score_lst = self.separate_score_predict(arch)
    #     score = (adjusted_gating * score_lst).sum(dim=1)
        
    #     if sigmoid:
    #         score = torch.sigmoid(score)
    #     elif tanh:
    #         score = torch.tanh(score)
        
    #     return score

    
    # def _confidence_ratio(self, archs) -> Tensor:
    #     """带温度退火的门控概率计算"""
    #     if self.use_uniform_confidence:
    #         confidence_ratio = torch.softmax(self.confidence, 1)
    #     else:
    #         arch_embeddings = self.arch_embedder(archs)
    #         logits = self.confidence_mlp(arch_embeddings)  # [B, N]
    #         # 应用当前门控温度
    #         confidence_ratio = torch.softmax(logits / self.gating_temp, dim=1)
    #     return confidence_ratio
    

    def _confidence_ratio(self, archs) -> Tensor:
        if self.use_uniform_confidence:
            confidence_ratio = torch.softmax(self.confidence, 1)
        else:
            arch_embeddings = self.arch_embedder(archs)
            confidence_ratio = torch.softmax(self.confidence_mlp(arch_embeddings), 1)
        return confidence_ratio

    def _compare_loss(self, s_1, s_2, better_labels, margin=None):
        s_1 = s_1.squeeze()
        s_2 = s_2.squeeze()
        better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()


            # 定期更新专家表征（例如每10步）
        # if not hasattr(self, '_rep_update_counter'):
        #     self._rep_update_counter = 0
        # self._rep_update_counter += 1
        # if self._rep_update_counter % 10 == 0:
        #     self._extract_expert_representations()
        return loss.item()

    


class GKD_PointwiseComparator_old(PointwiseComparator):
    """
    Dynamic ensemble pointwise comparator with feature-level distillation and dynamic grouping.
    """

    NAME = "GKD_pointwise_comparator"

    def __init__(
        self,
        auxiliary_head_num: int,
        arch_embedder_cls,
        arch_embedder_cfg=None,
        use_uniform_confidence: bool = False,
        mlp_hiddens: Tuple[int] = (200, 200, 200),
        mlp_dropout: float = 0.1,
        optimizer: Dict[str, Union[str, float]] = {
            "type": "Adam",
            "lr": 0.001
        },
        scheduler: Optional[str] = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ) -> None:
        super(GKD_PointwiseComparator_old, self).__init__(
            arch_embedder_cls,
            arch_embedder_cfg,
            mlp_hiddens,
            mlp_dropout,
            optimizer,
            scheduler,
            compare_loss_type,
            compare_margin,
            margin_l2,
            max_grad_norm
        )
        self.last_total_loss = None  # 用于记录上一轮总损失
        self.high_group = None
        self.low_group = None
        self.auxiliary_head_num = auxiliary_head_num
        self.use_uniform_confidence = use_uniform_confidence

        if self.use_uniform_confidence:
            self.confidence = nn.Parameter(
                torch.randn((1, self.auxiliary_head_num), requires_grad=True)
            )
            nn.init.constant_(self.confidence, 1. / self.auxiliary_head_num)
        else:
            self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))
            dim = self.embedding_dim = self.arch_embedder.out_dim
            self.confidence_mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout, auxiliary_head_num)

        self.module_lst = nn.ModuleList([
            PointwiseComparator(
                arch_embedder_cls,
                arch_embedder_cfg,
                mlp_hiddens,
                mlp_dropout,
                optimizer,
                scheduler,
                compare_loss_type,
                compare_margin,
                margin_l2,
                max_grad_norm
            ) for _ in range(self.auxiliary_head_num)
        ])

        self.reinit_optimizer(only_mlp=False)
        self.reinit_scheduler()

    def init_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)

    def init_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def _get_grad_vector(self, model):
        """获取模型所有参数的梯度向量"""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.zeros(1)

    def compute_consistency(self, s_i, s_j):
        """计算两个子网络输出的一致性（余弦相似度）"""
        return F.cosine_similarity(s_i.unsqueeze(1), s_j.unsqueeze(1), dim=-1).mean()

    def compute_entropy(self, logits):
        """计算标签熵（用于评估标签质量）"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def compute_grad_similarity(self, model_i, model_j):
        """计算两个模型梯度方向的相似性"""
        grad_i = self._get_grad_vector(model_i)
        grad_j = self._get_grad_vector(model_j)
        return F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0), dim=-1).mean()



    def _dynamic_grouping(self, real_archs):
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)      # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]

        quality_scores = []
        for i in range(self.auxiliary_head_num):
            scores_i = scores[:, i]
            consistency_list = [
                self.compute_consistency(scores_i, scores[:, j])
                for j in range(self.auxiliary_head_num) if j != i
            ]
            consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
            entropy_i = self.compute_entropy(scores_i)
            grad_sim_i = torch.mean(
                torch.stack([
                    self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
                    for j in range(self.auxiliary_head_num) if j != i
                ])
            )
            quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
            quality_scores.append(quality_score)

        quality_scores = torch.stack(quality_scores)  # [N]
        threshold = torch.median(quality_scores)
        high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]

        return high_group, low_group, quality_scores  # ✅ 返回 quality_scores
    

   




    # def mtl_update_compare_kd(self, auxiliary_datas, real_archs, margin=None,
    #                       feat_distill_weight=0.001, feat_loss_type="cosine",
    #                       contrast_weight=0.01, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0):
    #     total_loss = 0.

    #     # --- 第一步：获取特征并动态分组 ---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)
    #     features = torch.stack(features, dim=1)

    #     # 获取分组和质量得分
    #     high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)

    #     # --- 第二步：主任务损失 ---
    #     for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
    #         s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
    #         s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
    #         task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
    #         total_loss += task_loss


      
    #     # --- 第四步：三部分协作蒸馏 ---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_intra = 0.0
    #     distill_loss_inter = 0.0
    #     B, N, H = features.shape

    #     # 1. 高性能组内蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             for j in high_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_high_intra += (1.0 - sim)

    #     # 2. 低绩效组内蒸馏
    #     if len(low_group) > 1:
    #         for i in low_group:
    #             for j in low_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_low_intra += (1.0 - sim)

    #     # 3. 跨组蒸馏（Low → High）
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         for i in low_group:
    #             for j in high_group:
    #                 feat_i = features[:, i, :]
    #                 feat_j = features[:, j, :]
    #                 sim = torch.mean(F.cosine_similarity(
    #                     F.normalize(feat_i, p=2, dim=1),
    #                     F.normalize(feat_j, p=2, dim=1),
    #                     dim=1
    #                 ))
    #                 distill_loss_inter += (1.0 - sim)

    #     # --- 第五步：组合蒸馏损失 ---
    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low * distill_loss_low_intra +
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss

    #     # --- 第六步：对比学习（可选）---
    #     contrastive_loss = 0.0
    #     if B > 1:
    #         feat_norm = F.normalize(features, p=2, dim=2)
    #         sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
    #         labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
    #         sim_matrix_flat = sim_matrix.view(-1, N)
    #         contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
    #     total_loss += contrast_weight * contrastive_loss

    #     return self.update_step(total_loss)
        

    # def update_compare_kd(self, arch_1, arch_2, better_labels, real_archs, margin=None,
    #                   feat_distill_weight=0.001, feat_loss_type="cosine",
    #                   contrast_weight=0.01, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0):


    #                   #feat_distill_weight=0.01, feat_loss_type="cosine",
    #                   #contrast_weight=0.1
    #     """
    #     基于动态分组的融合蒸馏策略
    #     """
    #     total_loss = 0.

    #     # --- 第一步：获取所有子网络在 real_archs 上的预测和特征 ---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)  # [B, N]
    #     features = torch.stack(features, dim=1)  # [B, N, H]

    #     # 获取分组和质量得分
    #     high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)

    #     # --- 第三步：主任务损失 ---
    #     score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False)  # [B]
    #     score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False)  # [B]
    #     task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)
    #     total_loss += task_loss #0.06

    #     # # --- 第四步：协作蒸馏（三部分）---
       
    #     # --- 第四步：三部分协作蒸馏 ---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_intra = 0.0
    #     distill_loss_inter = 0.0
    #     B, N, H = features.shape

    #     # 1. 高性能组内蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             for j in high_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_high_intra += (1.0 - sim)

    #     # 2. 低绩效组内蒸馏
    #     if len(low_group) > 1:
    #         for i in low_group:
    #             for j in low_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_low_intra += (1.0 - sim)

    #     # 3. 跨组蒸馏（Low → High）
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         for i in low_group:
    #             for j in high_group:
    #                 feat_i = features[:, i, :]
    #                 feat_j = features[:, j, :]
    #                 sim = torch.mean(F.cosine_similarity(
    #                     F.normalize(feat_i, p=2, dim=1),
    #                     F.normalize(feat_j, p=2, dim=1),
    #                     dim=1
    #                 ))
    #                 distill_loss_inter += (1.0 - sim)

    #     # --- 第五步：组合蒸馏损失 ---
    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low * distill_loss_low_intra +
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss

    #     # --- 第六步：对比学习（可选）---
    #     contrastive_loss = 0.0
    #     if B > 1:
    #         feat_norm = F.normalize(features, p=2, dim=2)
    #         sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
    #         labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
    #         sim_matrix_flat = sim_matrix.view(-1, N)
    #         contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
    #     total_loss += contrast_weight * contrastive_loss

    #     return self.update_step(total_loss)

    ################################################################################################################
    # def mtl_update_compare_kd(self, auxiliary_datas, real_archs, margin=None,
    #                       feat_distill_weight=0.001, feat_loss_type="cosine",
    #                       contrast_weight=0.01, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0):
    #     total_loss = 0.

    #     # --- 第一步：获取特征并动态分组 ---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)
    #     features = torch.stack(features, dim=1)  # [B, N, H]

    #     # 获取分组和质量得分
    #     high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
    #     all_indices = list(range(features.size(1)))

    #     # --- 第二步：主任务损失 ---
    #     for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
    #         s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
    #         s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
    #         task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
    #         total_loss += task_loss #1.16

    #     # --- 第三步：对比蒸馏损失（替代原三部分蒸馏）---
    #     B, N, H = features.shape
    #     device = features.device

    #     # Normalize features for cosine similarity
    #     feat_norm = F.normalize(features, p=2, dim=2)  # [B, N, H]

    #     def compute_contrastive_loss(anchor_idx, positive_indices, all_indices):
    #         """
    #         Compute InfoNCE-style loss for one anchor head.
    #         anchor_idx: int
    #         positive_indices: list of int (positive samples)
    #         all_indices: list of all head indices
    #         """
    #         if len(positive_indices) == 0:
    #             return 0.0

    #         anchor_feat = feat_norm[:, anchor_idx, :]  # [B, H]
    #         # Positive similarities
    #         pos_feats = feat_norm[:, positive_indices, :]  # [B, P, H]
    #         pos_sim = torch.sum(anchor_feat.unsqueeze(1) * pos_feats, dim=2)  # [B, P]

    #         # Negative: all except self and positives
    #         negative_indices = [i for i in all_indices if i != anchor_idx and i not in positive_indices]
    #         if len(negative_indices) == 0:
    #             # No negatives → only use positives (degenerate case)
    #             logits = pos_sim
    #             labels = torch.zeros(B, dtype=torch.long, device=device)
    #         else:
    #             neg_feats = feat_norm[:, negative_indices, :]  # [B, Q, H]
    #             neg_sim = torch.sum(anchor_feat.unsqueeze(1) * neg_feats, dim=2)  # [B, Q]
    #             logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, P+Q]
    #             labels = torch.zeros(B, dtype=torch.long, device=device)  # first P are positives

    #         # Apply temperature (optional, e.g., τ=0.1)
    #         temperature = 0.1
    #         logits = logits / temperature

    #         # Mean over positives if multiple
    #         if pos_sim.size(1) > 1:
    #             # Use multi-positive InfoNCE: sum over positives in log-sum-exp
    #             # But for simplicity, we treat first positive as label (or average loss)
    #             # Here we use standard CE with single label → take mean of positive logits
    #             # Alternative: use SupCon loss. For now, average positive similarities.
    #             pos_mean = pos_sim.mean(dim=1, keepdim=True)
    #             if len(negative_indices) > 0:
    #                 logits = torch.cat([pos_mean, neg_sim], dim=1)
    #             else:
    #                 logits = pos_mean
    #             labels = torch.zeros(B, dtype=torch.long, device=device)

    #         loss = F.cross_entropy(logits, labels)
    #         return loss

    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_intra = 0.0
    #     distill_loss_inter = 0.0

    #     # 1. 高性能组内对比蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             positives = [j for j in high_group if j != i]
    #             loss = compute_contrastive_loss(i, positives, all_indices)
    #             distill_loss_high_intra += loss

    #     # 2. 低绩效组内对比蒸馏
    #     if len(low_group) > 1:
    #         for i in low_group:
    #             positives = [j for j in low_group if j != i]
    #             loss = compute_contrastive_loss(i, positives, all_indices)
    #             distill_loss_low_intra += loss

    #     # 3. 跨组对比蒸馏: Low → High (each low head treats all high heads as positives)
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         for i in low_group:
    #             positives = high_group  # all high-group heads are positive targets
    #             loss = compute_contrastive_loss(i, positives, all_indices)
    #             distill_loss_inter += loss

    #     # Normalize by number of anchors to stabilize scale
    #     num_high = max(len(high_group), 1)
    #     num_low = max(len(low_group), 1)
    #     num_inter = max(len(low_group), 1)

    #     distill_loss_high_intra /= num_high
    #     distill_loss_low_intra /= num_low
    #     distill_loss_inter /= num_inter

    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low * distill_loss_low_intra +
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss #4.72

    #     # --- 第四步：全局对比学习（可选，保留原逻辑）---
    #     contrastive_loss = 0.0
    #     if B > 1:
    #         sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
    #         labels = torch.arange(N, device=device).repeat(B)  # [B*N]
    #         sim_matrix_flat = sim_matrix.view(-1, N)
    #         contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
    #     total_loss += contrast_weight * contrastive_loss #1.61

    #     return self.update_step(total_loss)


    def mtl_update_compare_kd(self, auxiliary_datas, real_archs, epoch, margin=None,
                          feat_distill_weight=0.001, feat_loss_type="cosine",
                          contrast_weight=0.001, contrast_temp=0.1):
        
        # feat_distill_weight=0.01, feat_loss_type="cosine",
        #                   contrast_weight=0.01, contrast_temp=0.1
        total_loss = 0.

        # --- 第一步：获取特征并动态分组 ---
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)
        features = torch.stack(features, dim=1)

        if epoch % 10 == 1:  # 每10个epoch更新一次分组
            self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)

        high_group = self.high_group
        low_group = self.low_group

        # # 动态分组（只需分组，无需 quality_scores）
        # quality_scores = []
        # for i in range(self.auxiliary_head_num):
        #     scores_i = scores[:, i]
        #     consistency_list = [
        #         self.compute_consistency(scores_i, scores[:, j])
        #         for j in range(self.auxiliary_head_num) if j != i
        #     ]
        #     consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #     entropy_i = self.compute_entropy(scores_i)
        #     grad_sim_i = torch.mean(
        #         torch.stack([
        #             self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ])
        #     )
        #     quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
        #     quality_scores.append(quality_score)

        # quality_scores = torch.stack(quality_scores)
        # threshold = torch.median(quality_scores)
        # high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        # low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]

        # --- 第二步：主任务损失 ---
        task_loss_value = 0.0
        for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
            s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
            s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
            task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
            total_loss += task_loss
            task_loss_value += task_loss.item()
        task_loss_value /= len(auxiliary_datas)  # 平均主任务损失

        # # --- 第三步：基于训练损失的动态权重 ---
        # lambda_high = 1.0
        # lambda_low = 0.5
        # lambda_inter = 0.5

        # if self.last_total_loss is not None:
        #     # 计算相对损失改进率
        #     loss_improvement = (self.last_total_loss - task_loss_value) / (self.last_total_loss + 1e-8)
            
        #     # 映射到 [0, 1] 区间（可选：使用 sigmoid 或 clamp）
        #     # 如果损失下降（improvement > 0），增强跨组蒸馏
        #     # 如果损失上升（improvement < 0），减弱跨组蒸馏
        #     adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 5.0), -1.0, 1.0).item()
            
        #     # 动态调整
        #     lambda_inter = 0.5 + 0.5 * max(adaptive_factor, 0.0)  # [0.5, 1.0]
        #     lambda_low = 0.5 - 0.4 * max(adaptive_factor, 0.0)    # [0.1, 0.5]
        # else:
        #     # 第一轮，使用默认权重
        #     lambda_inter = 0.5
        #     lambda_low = 0.5

        # # 更新历史损失
        # self.last_total_loss = task_loss_value

        # # --- 第四步：三部分协作蒸馏 ---
        # distill_loss_high_intra = 0.0
        # distill_loss_low_intra = 0.0
        # distill_loss_inter = 0.0
        # B, N, H = features.shape

        # # 1. 高性能组内蒸馏
        # if len(high_group) > 1:
        #     for i in high_group:
        #         for j in high_group:
        #             if i != j:
        #                 feat_i = features[:, i, :]
        #                 feat_j = features[:, j, :]
        #                 sim = torch.mean(F.cosine_similarity(
        #                     F.normalize(feat_i, p=2, dim=1),
        #                     F.normalize(feat_j, p=2, dim=1),
        #                     dim=1
        #                 ))
        #                 distill_loss_high_intra += (1.0 - sim)

        # # 2. 低绩效组内蒸馏
        # if len(low_group) > 1:
        #     for i in low_group:
        #         for j in low_group:
        #             if i != j:
        #                 feat_i = features[:, i, :]
        #                 feat_j = features[:, j, :]
        #                 sim = torch.mean(F.cosine_similarity(
        #                     F.normalize(feat_i, p=2, dim=1),
        #                     F.normalize(feat_j, p=2, dim=1),
        #                     dim=1
        #                 ))
        #                 distill_loss_low_intra += (1.0 - sim)

        # # 3. 跨组蒸馏（Low → High）
        # if len(low_group) > 0 and len(high_group) > 0:
        #     for i in low_group:
        #         for j in high_group:
        #             feat_i = features[:, i, :]
        #             feat_j = features[:, j, :]
        #             sim = torch.mean(F.cosine_similarity(
        #                 F.normalize(feat_i, p=2, dim=1),
        #                 F.normalize(feat_j, p=2, dim=1),
        #                 dim=1
        #             ))
        #             distill_loss_inter += (1.0 - sim)

        # # --- 第五步：组合蒸馏损失 ---
        # total_distill_loss = (
        #     lambda_high * distill_loss_high_intra +
        #     lambda_low * distill_loss_low_intra +
        #     lambda_inter * distill_loss_inter
        # )
        # total_loss += feat_distill_weight * total_distill_loss

         #        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0
        B, N, H = features.shape
    

            # --- 第四步：基于训练损失的动态权重 ---
        lambda_high = 1.0
        lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
        lambda_inter = 0.5

        if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
            if not hasattr(self, 'smoothed_loss'):
                self.smoothed_loss = self.last_total_loss
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
            loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
            adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
            lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
            # lambda_low_consensus 保持弱权重
        else:
            self.smoothed_loss = task_loss.item()

        self.last_total_loss = task_loss.item()

        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0

        # 1. 高性能组内蒸馏
        if len(high_group) > 1:
            for i in high_group:
                for j in high_group:
                    if i != j:
                        feat_i = features[:, i, :]
                        feat_j = features[:, j, :]
                        sim = torch.mean(F.cosine_similarity(
                            F.normalize(feat_i, p=2, dim=1),
                            F.normalize(feat_j, p=2, dim=1),
                            dim=1
                        ))
                        distill_loss_high_intra += (1.0 - sim)

        # 2. ✅ 低绩效共识教师蒸馏（安全替代）
        if len(low_group) > 1:
            low_features = features[:, low_group, :]  # [B, L, H]
            consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            for i in low_group:
                feat_i = features[:, i, :]
                sim = torch.mean(F.cosine_similarity(
                    F.normalize(feat_i, p=2, dim=1),
                    F.normalize(consensus_low, p=2, dim=1),
                    dim=1
                ))
                distill_loss_low_consensus += (1.0 - sim)
        elif len(low_group) == 1:
            distill_loss_low_consensus = 0.0

        # 3. 跨组蒸馏（Low → High）带置信度过滤
        if len(low_group) > 0 and len(high_group) > 0:
            high_scores = scores[:, high_group]  # [B, K]
            high_var = torch.var(high_scores, dim=1)  # [B]
            teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
            if teach_mask.any():
                for i in low_group:
                    for j in high_group:
                        feat_i = features[:, i, :][teach_mask]
                        feat_j = features[:, j, :][teach_mask]
                        if feat_i.size(0) > 0:
                            sim = torch.mean(F.cosine_similarity(
                                F.normalize(feat_i, p=2, dim=1),
                                F.normalize(feat_j, p=2, dim=1),
                                dim=1
                            ))
                            distill_loss_inter += (1.0 - sim)

        # --- 第六步：组合蒸馏损失 ---
        total_distill_loss = (
            lambda_high * distill_loss_high_intra +
            lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
            lambda_inter * distill_loss_inter
        )
        total_loss += feat_distill_weight * total_distill_loss
        ###############################################################
        contrastive_loss_total = 0.0
        # 高性能组内对比（数值稳定版）
        if len(high_group) > 1:
            high_feats = features[:, high_group, :].view(-1, H)  # [B*K, H]
            high_feats = F.normalize(high_feats, dim=1)
            sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp  # [B*K, B*K]

            # 构建正样本 mask：同一样本的不同子网络
            labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))  # [B*K]
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B*K, B*K]
            pos_mask.fill_diagonal_(False)  # 排除自己

            # 计算 InfoNCE 损失（数值稳定）
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
            contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
            contrastive_loss_total += contrastive_loss

        # 跨组对比（Low vs High 共识）
        if len(high_group) > 0 and len(low_group) > 0:
            high_consensus = features[:, high_group, :].mean(dim=1)  # [B, H]
            low_feats = features[:, low_group, :].view(-1, H)        # [B*L, H]
            low_feats = F.normalize(low_feats, dim=1)
            high_consensus = F.normalize(high_consensus, dim=1)
            sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp  # [B*L, B]

            # 标签：每个低绩效特征对应同一样本的高性能共识
            labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))  # [B*L]

            # 数值稳定版
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
            contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
            contrastive_loss_total += contrastive_loss

        total_loss += contrast_weight * contrastive_loss_total
        self.last_total_loss = task_loss_value

        ######## --- 第六步：对比学习（可选）---
        # contrastive_loss = 0.0
        # if B > 1:
        #     feat_norm = F.normalize(features, p=2, dim=2)
        #     sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
        #     labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
        #     sim_matrix_flat = sim_matrix.view(-1, N)
        #     contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
        # total_loss += contrast_weight * contrastive_loss

        return self.update_step(total_loss)
    # feat_distill_weight=0.0001, feat_loss_type="cosine",
    #                   contrast_weight=0.0001 

    def update_compare_kd(self, arch_1, arch_2, better_labels, real_archs, epoch, margin=None,
                      feat_distill_weight=0.001, feat_loss_type="cosine",
                      contrast_weight=0.001, contrast_temp=0.1): #lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0
        
        # feat_distill_weight=0.001, feat_loss_type="cosine",
        #               contrast_weight=0.001, contrast_temp=0.1, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0
        """
        基于动态分组的融合蒸馏策略
        """
        total_loss = 0.

        # --- 第一步：获取所有子网络在 real_archs 上的预测和特征 ---
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)  # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]


        if epoch % 10 == 1:  # 每10个epoch更新一次分组
            self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)
        #high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
        high_group = self.high_group
        low_group = self.low_group


        # --- 第三步：主任务损失 ---
        score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False)  # [B]
        score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False)  # [B]
        task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)
        total_loss += task_loss #0.06



    

        # # --- 第三步：基于训练损失的动态权重 ---
        # lambda_high = 1.0
        # lambda_low = 0.5
        # lambda_inter = 1.0

        # if self.last_total_loss is not None:
        #     # 计算相对损失改进率
        #     loss_improvement = (self.last_total_loss - task_loss) / (self.last_total_loss + 1e-8)
            
        #     # 映射到 [0, 1] 区间（可选：使用 sigmoid 或 clamp）
        #     # 如果损失下降（improvement > 0），增强跨组蒸馏
        #     # 如果损失上升（improvement < 0），减弱跨组蒸馏
        #     adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 5.0), -1.0, 1.0).item()
            
        #     # 动态调整
        #     lambda_inter = 0.5 + 0.5 * max(adaptive_factor, 0.0)  # [0.5, 1.0]
        #     lambda_low = 0.5 - 0.4 * max(adaptive_factor, 0.0)    # [0.1, 0.5]
        # else:
        #     # 第一轮，使用默认权重
        #     lambda_inter = 0.5
        #     lambda_low = 0.5

        # # 更新历史损失
        # self.last_total_loss = task_loss

        # # # --- 第四步：三部分协作蒸馏 ---
        # distill_loss_high_intra = 0.0
        # distill_loss_low_intra = 0.0
        # distill_loss_inter = 0.0
        # B, N, H = features.shape

        # # 1. 高性能组内蒸馏
        # if len(high_group) > 1:
        #     for i in high_group:
        #         for j in high_group:
        #             if i != j:
        #                 feat_i = features[:, i, :]
        #                 feat_j = features[:, j, :]
        #                 sim = torch.mean(F.cosine_similarity(
        #                     F.normalize(feat_i, p=2, dim=1),
        #                     F.normalize(feat_j, p=2, dim=1),
        #                     dim=1
        #                 ))
        #                 distill_loss_high_intra += (1.0 - sim)

        # # 2. 低绩效组内蒸馏
        # if len(low_group) > 1:
        #     for i in low_group:
        #         for j in low_group:
        #             if i != j:
        #                 feat_i = features[:, i, :]
        #                 feat_j = features[:, j, :]
        #                 sim = torch.mean(F.cosine_similarity(
        #                     F.normalize(feat_i, p=2, dim=1),
        #                     F.normalize(feat_j, p=2, dim=1),
        #                     dim=1
        #                 ))
        #                 distill_loss_low_intra += (1.0 - sim)

        # # 3. 跨组蒸馏（Low → High）
        # if len(low_group) > 0 and len(high_group) > 0:
        #     for i in low_group:
        #         for j in high_group:
        #             feat_i = features[:, i, :]
        #             feat_j = features[:, j, :]
        #             sim = torch.mean(F.cosine_similarity(
        #                 F.normalize(feat_i, p=2, dim=1),
        #                 F.normalize(feat_j, p=2, dim=1),
        #                 dim=1
        #             ))
        #             distill_loss_inter += (1.0 - sim)

        # # --- 第五步：组合蒸馏损失 ---
        # total_distill_loss = (
        #     lambda_high * distill_loss_high_intra +
        #     lambda_low * distill_loss_low_intra +
        #     lambda_inter * distill_loss_inter
        # )
        # total_loss += feat_distill_weight * total_loss
        ############################################################
        #        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0
        B, N, H = features.shape
    

            # --- 第四步：基于训练损失的动态权重 ---
        lambda_high = 1.0
        lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
        lambda_inter = 0.5

        if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
            if not hasattr(self, 'smoothed_loss'):
                self.smoothed_loss = self.last_total_loss
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
            loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
            adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
            lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
            # lambda_low_consensus 保持弱权重
        else:
            self.smoothed_loss = task_loss.item()

        self.last_total_loss = task_loss.item()

        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0

        # 1. 高性能组内蒸馏
        if len(high_group) > 1:
            for i in high_group:
                for j in high_group:
                    if i != j:
                        feat_i = features[:, i, :]
                        feat_j = features[:, j, :]
                        sim = torch.mean(F.cosine_similarity(
                            F.normalize(feat_i, p=2, dim=1),
                            F.normalize(feat_j, p=2, dim=1),
                            dim=1
                        ))
                        distill_loss_high_intra += (1.0 - sim)

        # 2. ✅ 低绩效共识教师蒸馏（安全替代）
        if len(low_group) > 1:
            low_features = features[:, low_group, :]  # [B, L, H]
            consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            for i in low_group:
                feat_i = features[:, i, :]
                sim = torch.mean(F.cosine_similarity(
                    F.normalize(feat_i, p=2, dim=1),
                    F.normalize(consensus_low, p=2, dim=1),
                    dim=1
                ))
                distill_loss_low_consensus += (1.0 - sim)
        elif len(low_group) == 1:
            distill_loss_low_consensus = 0.0

        # 3. 跨组蒸馏（Low → High）带置信度过滤
        if len(low_group) > 0 and len(high_group) > 0:
            high_scores = scores[:, high_group]  # [B, K]
            high_var = torch.var(high_scores, dim=1)  # [B]
            teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
            if teach_mask.any():
                for i in low_group:
                    for j in high_group:
                        feat_i = features[:, i, :][teach_mask]
                        feat_j = features[:, j, :][teach_mask]
                        if feat_i.size(0) > 0:
                            sim = torch.mean(F.cosine_similarity(
                                F.normalize(feat_i, p=2, dim=1),
                                F.normalize(feat_j, p=2, dim=1),
                                dim=1
                            ))
                            distill_loss_inter += (1.0 - sim)

        # --- 第六步：组合蒸馏损失 ---
        total_distill_loss = (
            lambda_high * distill_loss_high_intra +
            lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
            lambda_inter * distill_loss_inter
        )
        total_loss += feat_distill_weight * total_distill_loss

   
    

        contrastive_loss_total = 0.0
        # 高性能组内对比（数值稳定版）
        if len(high_group) > 1:
            high_feats = features[:, high_group, :].view(-1, H)  # [B*K, H]
            high_feats = F.normalize(high_feats, dim=1)
            sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp  # [B*K, B*K]

            # 构建正样本 mask：同一样本的不同子网络
            labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))  # [B*K]
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B*K, B*K]
            pos_mask.fill_diagonal_(False)  # 排除自己

            # 计算 InfoNCE 损失（数值稳定）
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
            contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
            contrastive_loss_total += contrastive_loss

        # 跨组对比（Low vs High 共识）
        if len(high_group) > 0 and len(low_group) > 0:
            high_consensus = features[:, high_group, :].mean(dim=1)  # [B, H]
            low_feats = features[:, low_group, :].view(-1, H)        # [B*L, H]
            low_feats = F.normalize(low_feats, dim=1)
            high_consensus = F.normalize(high_consensus, dim=1)
            sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp  # [B*L, B]

            # 标签：每个低绩效特征对应同一样本的高性能共识
            labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))  # [B*L]

            # 数值稳定版
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True))
            contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
            contrastive_loss_total += contrastive_loss

        total_loss += contrast_weight * contrastive_loss_total
        self.last_total_loss = task_loss

        #--- 第六步：对比学习（可选）---
        # contrastive_loss = 0.0
        # if B > 1:
        #     feat_norm = F.normalize(features, p=2, dim=2)
        #     sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
        #     labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
        #     sim_matrix_flat = sim_matrix.view(-1, N)
        #     contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
        # total_loss += contrast_weight * contrastive_loss

        return self.update_step(total_loss)
    





    #############################################################################################



    def _get_student_predictions_and_features(self, arch):
        """
        Helper function to get predictions and features from all auxiliary students.
        Returns:
            scores: List of tensors [B]
            features: List of tensors [B, H]
        """
        scores = []
        features = []
        for model in self.module_lst:
            feature = model.predict_feature(arch, sigmoid=False, tanh=False)
            score = model.predict(arch)
            scores.append(score)
            features.append(feature)
        return scores, features


    def mtl_predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tuple[Tensor, List[Tensor]]:
        score = self.predict(arch, sigmoid, tanh)
        lf_score_lst = [model.predict(arch, sigmoid, tanh) for model in self.module_lst]
        return score, lf_score_lst

    def separate_score_predict(self, arch) -> Tensor:
        score_lst = torch.cat([
            model.predict(arch, False, False).unsqueeze(1)
            for model in self.module_lst],
        1)
        return score_lst

    def weighted_score(self, arch) -> Tensor:
        score_lst = self.separate_score_predict(arch)
        confidence_ratio = self._confidence_ratio(arch)
        score = (confidence_ratio * score_lst)
        return score

    def predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tensor:
        weighted_score = self.weighted_score(arch)
        score = weighted_score.sum(1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def _confidence_ratio(self, archs) -> Tensor:
        if self.use_uniform_confidence:
            confidence_ratio = torch.softmax(self.confidence, 1)
        else:
            arch_embeddings = self.arch_embedder(archs)
            confidence_ratio = torch.softmax(self.confidence_mlp(arch_embeddings), 1)
        return confidence_ratio

    def _compare_loss(self, s_1, s_2, better_labels, margin=None):
        s_1 = s_1.squeeze()
        s_2 = s_2.squeeze()
        better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()

    #################### mix up -data
    # def semi_update_compare(self, arch_1, arch_2, better_labels, acc, prob, index, mlp, margin=None):
    #     if index == 1:
    #         arch_embedder = self.arch_embedder_1
    #     else:
    #         arch_embedder = self.arch_embedder_2
    #     s_1 = mlp(arch_embedder(arch_1)).squeeze()
    #     s_2 = mlp(arch_embedder(arch_2)).squeeze()
    #     better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
    #     zero_ = s_1.new([0.])
    #     margin = [self.compare_margin] if margin is None else margin
    #     margin = s_1.new(margin)
    #     px_1 = torch.softmax(s_1, dim=-1)
    #     px_2 = torch.softmax(s_2, dim=-1)
        
    #     pref_prob1 = torch.sigmoid(s_2 - s_1)
        
        
        
        
    #     # 判断一致性
       
    #     #match1 = ((s_1_1 - s_1_2) > 0) == ((s_2_1 - s_2_2) > 0)
       
        

    #     # 更新权重
    #     # prob = torch.tensor(prob).to(s_1.device)
    #     # prob[match1] = 1
        
        
        
    #     confidence_mask = ((s_1 - s_2).sigmoid() > 0.95).float()
        
                        
    #     pseudo_preference = confidence_mask.unsqueeze(1) * acc + \
    #         (1 - confidence_mask).unsqueeze(1) * px_1.detach()
            
    #     idx_chosen_pair1 = torch.where(prob == 1)[0]
    #     idx_unchosen_pair1 = torch.where(prob == 0)[0]
        
        
    #     # high_conf_pair1 = pref_prob1 > 0.95
    #     # pseudo_preference1 = pref_prob1 * high_conf_pair1.float()
        
    #     # high_conf_pair2 = pref_prob2 > 0.95
    #     # pseudo_preference2 = pref_prob2 * high_conf_pair2.float()
        
    #     # train_labels_v1[index_1] = pseudo_preference1
    #     # train_labels_v2[index_2] = 1 - pseudo_preference1  # B better 的概率
        
        
        
                        
    #     # MixUp 成对数据
    #     l = np.random.beta(4, 4)
    #     l = max(l, 1 - l)

    #     # 随机打乱配对
    #     idx = torch.randperm(len(arch_1))
    #     archs_1_rand = arch_1[idx]
    #     archs_2_rand = arch_2[idx]
    #     better_labels_rand = better_labels[idx]

    #     # 构造混合数据
    #     archs_1_mix = l * arch_1 + (1 - l) * archs_1_rand
    #     archs_2_mix = l * arch_2 + (1 - l) * archs_2_rand
    #     better_labels_mix = l * better_labels + (1 - l) * better_labels_rand
        
        
    #     mix_s_1 = self.mlp(arch_embedder(archs_1_mix)).squeeze()
    #     mix_s_2 = self.mlp(arch_embedder(archs_2_mix)).squeeze()
    #     if not self.margin_l2:
    #         mix_pair_loss_chosen = torch.mean(torch.max(zero_, margin - better_labels_mix * (mix_s_1[idx_chosen_pair1] - mix_s_2[idx_chosen_pair1])))
    #         mix_pair_loss_unchosen = torch.mean(torch.max(zero_, margin - better_labels_mix * (mix_s_1[idx_unchosen_pair1] - mix_s_2[idx_unchosen_pair1])))
    #     else:
    #         mix_pair_loss_chosen = torch.mean(torch.max(zero_, margin - better_labels_mix * (mix_s_1[idx_chosen_pair1] - mix_s_2[idx_chosen_pair1])) \
    #                 ** 2 / np.maximum(1., margin))
    #         mix_pair_loss_unchosen = torch.mean(torch.max(zero_, margin - better_labels_mix * (mix_s_1[idx_unchosen_pair1] - mix_s_2[idx_unchosen_pair1])) \
    #                 ** 2 / np.maximum(1., margin))
                        
    #     if not self.margin_l2:
    #         pair_loss = torch.max(zero_, margin - better_pm * (s_2 - s_1))
    #     else:
    #         pair_loss = torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
    #                 ** 2 / np.maximum(1., margin)
                
    #     return pair_loss + 0.1* (mix_pair_loss_unchosen + mix_pair_loss_chosen)
    
    
    
    ########################################################################
    ############################################################################################################################



    # def mtl_update_compare_kd(self, auxiliary_datas, real_archs, epoch, margin=None,
    #                       feat_distill_weight=0.001, feat_loss_type="cosine",
    #                       contrast_weight=0.001, contrast_temp=0.1):
    #     total_loss = 0.

    #     # --- 第一步：获取特征并动态分组（每5个epoch更新）---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)      # [B, N]
    #     features = torch.stack(features, dim=1)  # [B, N, H]
        
    #     if epoch % 5 == 1:
    #         quality_scores = []
    #         for i in range(self.auxiliary_head_num):
    #             scores_i = scores[:, i]  # [B]
    #             # 输出一致性：与其他子网络的平均一致性
    #             consistency_list = [
    #                 self.compute_consistency(scores_i, scores[:, j])
    #                 for j in range(self.auxiliary_head_num) if j != i
    #             ]
    #             consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
                
    #             # ✅ 用预测方差替代无效的熵（标量任务）
    #             variance_i = torch.var(scores_i)  # 方差越小，预测越稳定
                
    #             # ✅ 移除梯度相似性（高开销且不稳定）
    #             quality_score = 0.7 * consistency_i + 0.3 * (1.0 / (1.0 + variance_i))
    #             quality_scores.append(quality_score)

    #         quality_scores = torch.stack(quality_scores)
    #         threshold = torch.median(quality_scores)
    #         high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
    #         low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
            
    #         # 处理空组情况
    #         if len(high_group) == 0:
    #             high_group = [torch.argmax(quality_scores).item()]
    #             low_group = [i for i in range(self.auxiliary_head_num) if i != high_group[0]]
    #         if len(low_group) == 0:
    #             low_group = [torch.argmin(quality_scores).item()]
    #             high_group = [i for i in range(self.auxiliary_head_num) if i != low_group[0]]
                
    #         self.high_group = high_group
    #         self.low_group = low_group

    #     # if epoch % 5 == 1:
    #     #     self.high_group, self.low_group, quality_scores = self._dynamic_grouping(real_archs)
    #     high_group = self.high_group
    #     low_group =self.low_group

    #     # --- 第二步：主任务损失 ---
    #     task_loss_value = 0.0
    #     for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
    #         s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
    #         s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
    #         task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
    #         total_loss += task_loss
    #         task_loss_value += task_loss.item()
    #     task_loss_value /= len(auxiliary_datas)

    #     # --- 第三步：基于训练损失的动态权重（确保生效）---
    #     lambda_high = 1.0
    #     lambda_low = 0.5
    #     lambda_inter = 0.5

    #     if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
    #         # 使用滑动平均避免震荡
    #         if not hasattr(self, 'smoothed_loss'):
    #             self.smoothed_loss = self.last_total_loss
    #         self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss_value
            
    #         loss_improvement = (self.last_total_loss - task_loss_value) / (self.last_total_loss + 1e-8)
    #         adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
    #         # ✅ 确保所有权重都动态调整
    #         lambda_high = 1.0  # 高性能组内蒸馏保持强一致
    #         lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
    #         lambda_low = 0.7 - 0.4 * max(adaptive_factor, 0.0)    # [0.3, 0.7]
    #     else:
    #         self.smoothed_loss = task_loss_value

    #     self.last_total_loss = task_loss_value

    #     # --- 第三步：动态权重 ---
    #     lambda_high = 1.0
    #     lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
    #     lambda_inter = 0.5

    #     if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
    #         if not hasattr(self, 'smoothed_loss'):
    #             self.smoothed_loss = self.last_total_loss
    #         self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss_value
            
    #         loss_improvement = (self.last_total_loss - task_loss_value) / (self.last_total_loss + 1e-8)
    #         adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
    #         lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)
    #         # lambda_low_consensus 保持弱权重，避免主导
    #     else:
    #         self.smoothed_loss = task_loss_value

    #     self.last_total_loss = task_loss_value

    #     # --- 第四步：协作蒸馏（安全版）---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
    #     distill_loss_inter = 0.0
    #     B, N, H = features.shape

    #     # 1. 高性能组内蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             for j in high_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_high_intra += (1.0 - sim)

    #     # 2. ✅ 低绩效共识教师蒸馏（安全替代方案）
    #     if len(low_group) > 1:
    #         # 构建低绩效共识教师：组平均特征
    #         low_features = features[:, low_group, :]  # [B, L, H]
    #         consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            
    #         # 每个低绩效子网络向共识教师学习
    #         for i in low_group:
    #             feat_i = features[:, i, :]
    #             sim = torch.mean(F.cosine_similarity(
    #                 F.normalize(feat_i, p=2, dim=1),
    #                 F.normalize(consensus_low, p=2, dim=1),
    #                 dim=1
    #             ))
    #             distill_loss_low_consensus += (1.0 - sim)
    #     elif len(low_group) == 1:
    #         # 单个低绩效子网络，无需蒸馏
    #         distill_loss_low_consensus = 0.0

    #     # 3. 跨组蒸馏（Low → High）带置信度过滤
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         high_scores = scores[:, high_group]
    #         high_var = torch.var(high_scores, dim=1)
    #         teach_mask = high_var < torch.median(high_var + 1e-8)
            
    #         if teach_mask.any():
    #             for i in low_group:
    #                 for j in high_group:
    #                     feat_i = features[:, i, :][teach_mask]
    #                     feat_j = features[:, j, :][teach_mask]
    #                     if feat_i.size(0) > 0:
    #                         sim = torch.mean(F.cosine_similarity(
    #                             F.normalize(feat_i, p=2, dim=1),
    #                             F.normalize(feat_j, p=2, dim=1),
    #                             dim=1
    #                         ))
    #                         distill_loss_inter += (1.0 - sim)

    #     # --- 第五步：组合蒸馏损失 ---
    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss

    #     # --- 第六步：对比知识蒸馏（仅保留有效部分）---
    #     contrastive_loss_total = 0.0

    #     # 1. 高性能组内对比（数值稳定版）
    #     if len(self.high_group) > 1:
    #         high_feats = features[:, self.high_group, :].view(-1, H)
    #         high_feats = F.normalize(high_feats, dim=1)
    #         sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp

    #         labels = torch.arange(B, device=features.device).repeat_interleave(len(self.high_group))
    #         pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    #         pos_mask.fill_diagonal_(False)

    #         exp_sim = torch.exp(sim_mat)
    #         log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    #         contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
    #         contrastive_loss_total += contrastive_loss

    #     # 2. 跨组对比（Low vs High 共识）
    #     if len(self.high_group) > 0 and len(self.low_group) > 0:
    #         high_consensus = features[:, self.high_group, :].mean(dim=1)  # [B, H]
    #         low_feats = features[:, self.low_group, :].view(-1, H)        # [B*L, H]
    #         low_feats = F.normalize(low_feats, dim=1)
    #         high_consensus = F.normalize(high_consensus, dim=1)
    #         sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp

    #         labels = torch.arange(B, device=features.device).repeat_interleave(len(self.low_group))
    #         exp_sim = torch.exp(sim_mat)
    #         log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    #         contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
    #         contrastive_loss_total += contrastive_loss

    #     # ✅ 移除冗余的全局对比损失（原第六步）
    #     total_loss += contrast_weight * contrastive_loss_total

    #     return self.update_step(total_loss)
    

    # def update_compare_kd(self, arch_1, arch_2, better_labels, real_archs, epoch, margin=None,
    #                   feat_distill_weight=0.001, feat_loss_type="cosine",
    #                   contrast_weight=0.001, contrast_temp=0.1, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0): #lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0
        
    #     # feat_distill_weight=0.001, feat_loss_type="cosine",
    #     #               contrast_weight=0.001, contrast_temp=0.1, lambda_high=1.0, lambda_low=0.5, lambda_inter=1.0
    #     """
    #     基于动态分组的融合蒸馏策略
    #     """
    #     total_loss = 0.

    #     # --- 第一步：获取所有子网络在 real_archs 上的预测和特征 ---
    #     scores, features = self._get_student_predictions_and_features(real_archs)
    #     scores = torch.stack(scores, dim=1)  # [B, N]
    #     features = torch.stack(features, dim=1)  # [B, N, H]

    #     if epoch % 5 == 1:
    #         quality_scores = []
    #         for i in range(self.auxiliary_head_num):
    #             scores_i = scores[:, i]  # [B]
    #             # 输出一致性：与其他子网络的平均一致性
    #             consistency_list = [
    #                 self.compute_consistency(scores_i, scores[:, j])
    #                 for j in range(self.auxiliary_head_num) if j != i
    #             ]
    #             consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
                
    #             # ✅ 用预测方差替代无效的熵（标量任务）
    #             variance_i = torch.var(scores_i)  # 方差越小，预测越稳定
                
    #             # ✅ 移除梯度相似性（高开销且不稳定）
    #             quality_score = 0.7 * consistency_i + 0.3 * (1.0 / (1.0 + variance_i))
    #             quality_scores.append(quality_score)

    #         quality_scores = torch.stack(quality_scores)
    #         threshold = torch.median(quality_scores)
    #         high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
    #         low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
            
    #         # 处理空组情况
    #         if len(high_group) == 0:
    #             high_group = [torch.argmax(quality_scores).item()]
    #             low_group = [i for i in range(self.auxiliary_head_num) if i != high_group[0]]
    #         if len(low_group) == 0:
    #             low_group = [torch.argmin(quality_scores).item()]
    #             high_group = [i for i in range(self.auxiliary_head_num) if i != low_group[0]]
                
    #         self.high_group = high_group
    #         self.low_group = low_group
    #     # if epoch % 5 == 1:
    #     #     self.high_group, self.low_group, quality_scores = self._dynamic_grouping(real_archs)
    #     high_group = self.high_group
    #     low_group =self.low_group


    #     # --- 第三步：主任务损失 ---
    #     score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False)  # [B]
    #     score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False)  # [B]
    #     task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)
    #     total_loss += task_loss #0.06


    #        # --- 第五步：安全协作蒸馏 ---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
    #     distill_loss_inter = 0.0
    #     B, N, H = features.shape
    

    #         # --- 第四步：基于训练损失的动态权重 ---
    #     lambda_high = 1.0
    #     lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
    #     lambda_inter = 0.5

    #     if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
    #         if not hasattr(self, 'smoothed_loss'):
    #             self.smoothed_loss = self.last_total_loss
    #         self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
    #         loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
    #         adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
    #         lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
    #         # lambda_low_consensus 保持弱权重
    #     else:
    #         self.smoothed_loss = task_loss.item()

    #     self.last_total_loss = task_loss.item()

    #     # --- 第五步：安全协作蒸馏 ---
    #     distill_loss_high_intra = 0.0
    #     distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
    #     distill_loss_inter = 0.0

    #     # 1. 高性能组内蒸馏
    #     if len(high_group) > 1:
    #         for i in high_group:
    #             for j in high_group:
    #                 if i != j:
    #                     feat_i = features[:, i, :]
    #                     feat_j = features[:, j, :]
    #                     sim = torch.mean(F.cosine_similarity(
    #                         F.normalize(feat_i, p=2, dim=1),
    #                         F.normalize(feat_j, p=2, dim=1),
    #                         dim=1
    #                     ))
    #                     distill_loss_high_intra += (1.0 - sim)

    #     # 2. ✅ 低绩效共识教师蒸馏（安全替代）
    #     if len(low_group) > 1:
    #         low_features = features[:, low_group, :]  # [B, L, H]
    #         consensus_low = torch.mean(low_features, dim=1)  # [B, H]
    #         for i in low_group:
    #             feat_i = features[:, i, :]
    #             sim = torch.mean(F.cosine_similarity(
    #                 F.normalize(feat_i, p=2, dim=1),
    #                 F.normalize(consensus_low, p=2, dim=1),
    #                 dim=1
    #             ))
    #             distill_loss_low_consensus += (1.0 - sim)
    #     elif len(low_group) == 1:
    #         distill_loss_low_consensus = 0.0

    #     # 3. 跨组蒸馏（Low → High）带置信度过滤
    #     if len(low_group) > 0 and len(high_group) > 0:
    #         high_scores = scores[:, high_group]  # [B, K]
    #         high_var = torch.var(high_scores, dim=1)  # [B]
    #         teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
    #         if teach_mask.any():
    #             for i in low_group:
    #                 for j in high_group:
    #                     feat_i = features[:, i, :][teach_mask]
    #                     feat_j = features[:, j, :][teach_mask]
    #                     if feat_i.size(0) > 0:
    #                         sim = torch.mean(F.cosine_similarity(
    #                             F.normalize(feat_i, p=2, dim=1),
    #                             F.normalize(feat_j, p=2, dim=1),
    #                             dim=1
    #                         ))
    #                         distill_loss_inter += (1.0 - sim)

    #     # --- 第六步：组合蒸馏损失 ---
    #     total_distill_loss = (
    #         lambda_high * distill_loss_high_intra +
    #         lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
    #         lambda_inter * distill_loss_inter
    #     )
    #     total_loss += feat_distill_weight * total_distill_loss

    #     # --- 第七步：对比知识蒸馏（仅组内+跨组）---
    #     contrastive_loss_total = 0.0

    #     ###############高性能组内对比
    #     # if len(high_group) > 1:
    #     #     high_feats = features[:, high_group, :].view(-1, H)
    #     #     high_feats = F.normalize(high_feats, dim=1)
    #     #     sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp
    #     #     labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))
    #     #     pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    #     #     pos_mask.fill_diagonal_(False)
    #     #     exp_sim = torch.exp(sim_mat)
    #     #     log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    #     #     contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
    #     #     contrastive_loss_total += contrastive_loss

    #     ############## 跨组对比
    #     # if len(high_group) > 0 and len(low_group) > 0:
    #     #     high_consensus = features[:, high_group, :].mean(dim=1)
    #     #     low_feats = features[:, low_group, :].view(-1, H)
    #     #     low_feats = F.normalize(low_feats, dim=1)
    #     #     high_consensus = F.normalize(high_consensus, dim=1)
    #     #     sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp
    #     #     labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))
    #     #     exp_sim = torch.exp(sim_mat)
    #     #     log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    #     #     contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
    #     #     contrastive_loss_total += contrastive_loss

    #     # total_loss += contrast_weight * contrastive_loss_total
    #     return self.update_step(total_loss)



class GKD_PointwiseComparator_1(PointwiseComparator):
    r"""
    Dynamic ensemble pointwise comparator with feature-level distillation and dynamic grouping.
    """

    NAME = "GKD_pointwise_comparator"

    def __init__(
        self,
        auxiliary_head_num: int,
        arch_embedder_cls,
        arch_embedder_cfg=None,
        use_uniform_confidence: bool = False,
        mlp_hiddens: Tuple[int] = (200, 200, 200),
        mlp_dropout: float = 0.1,
        optimizer: Dict[str, Union[str, float]] = {
            "type": "Adam",
            "lr": 0.001
        },
        scheduler: Optional[str] = None,
        compare_loss_type: str = "margin_linear",
        compare_margin: float = 0.01,
        margin_l2: bool = False,
        max_grad_norm: float = None
    ) -> None:
        super(GKD_PointwiseComparator, self).__init__(
            arch_embedder_cls,
            arch_embedder_cfg,
            mlp_hiddens,
            mlp_dropout,
            optimizer,
            scheduler,
            compare_loss_type,
            compare_margin,
            margin_l2,
            max_grad_norm
        )
        self.last_total_loss = None  # 用于记录上一轮总损失
         # 动态分组状态
        self.last_quality_scores = None  # 上次的质量得分
        self.group_update_interval = 5   # 最小更新间隔（防止过于频繁）
        self.group_update_counter = 0    # 更新计数器
      

        self.high_group = None
        self.low_group = None
        self.auxiliary_head_num = auxiliary_head_num
        self.use_uniform_confidence = use_uniform_confidence

        self.aux_loss_weight = 0.01
        self.gating_temp = 2.0
        self.gating_temp_min = 0.5
        self.gating_temp_decay = 0.995
        self.current_epoch = 1  # 跟踪当前 epoch

      
    


        if self.use_uniform_confidence:
            self.confidence = nn.Parameter(
                torch.randn((1, self.auxiliary_head_num), requires_grad=True)
            )
            nn.init.constant_(self.confidence, 1. / self.auxiliary_head_num)
        else:
            self.arch_embedder = arch_embedder_cls(**(arch_embedder_cfg or {}))
            dim = self.embedding_dim = self.arch_embedder.out_dim
            self.confidence_mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout, auxiliary_head_num)
        self._extract_expert_representations()
        self.group_gating_mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出 [scale_high, scale_low]
            nn.Softplus()      # 保证 >0
        )
        self.module_lst = nn.ModuleList([
            PointwiseComparator(
                arch_embedder_cls,
                arch_embedder_cfg,
                mlp_hiddens,
                mlp_dropout,
                optimizer,
                scheduler,
                compare_loss_type,
                compare_margin,
                margin_l2,
                max_grad_norm
            ) for _ in range(self.auxiliary_head_num)
        ])

        self.reinit_optimizer(only_mlp=False)
        self.reinit_scheduler()

    def _extract_expert_representations(self):
        """提取门控网络权重，并按分组存储"""
        if self.use_uniform_confidence or not hasattr(self, 'confidence_mlp'):
            self.expert_reps = None
            self.high_group_reps = None
            self.low_group_reps = None
            return
        
        # 获取专家表征
        mlp_layers = list(self.confidence_mlp.modules())
        last_linear = None
        for layer in reversed(mlp_layers):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        
        if last_linear is not None:
            self.expert_reps = last_linear.weight.detach().clone()  # [N, D]
            self.expert_reps = F.normalize(self.expert_reps, p=2, dim=1)
            
            # 按分组存储表征
            if self.high_group is not None and self.low_group is not None:
                self.high_group_reps = self.expert_reps[self.high_group]  # [K, D]
                self.low_group_reps = self.expert_reps[self.low_group]    # [L, D]
            else:
                self.high_group_reps = None
                self.low_group_reps = None
        else:
            self.expert_reps = None
            self.high_group_reps = None
            self.low_group_reps = None

    def init_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)

    def init_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def _get_grad_vector(self, model):
        """获取模型所有参数的梯度向量"""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.zeros(1)

    def compute_consistency(self, s_i, s_j):
        """计算两个子网络输出的一致性（余弦相似度）"""
        return F.cosine_similarity(s_i.unsqueeze(1), s_j.unsqueeze(1), dim=-1).mean()

    def compute_entropy(self, logits):
        """计算标签熵（用于评估标签质量）"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def compute_grad_similarity(self, model_i, model_j):
        """计算两个模型梯度方向的相似性"""
        grad_i = self._get_grad_vector(model_i)
        grad_j = self._get_grad_vector(model_j)
        return F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0), dim=-1).mean()

    def _dynamic_grouping(self, real_archs):
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)      # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]

        quality_scores = []
        for i in range(self.auxiliary_head_num):
            scores_i = scores[:, i]
            consistency_list = [
                self.compute_consistency(scores_i, scores[:, j])
                for j in range(self.auxiliary_head_num) if j != i
            ]
            consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
            entropy_i = self.compute_entropy(scores_i)
            grad_sim_i = torch.mean(
                torch.stack([
                    self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
                    for j in range(self.auxiliary_head_num) if j != i
                ])
            )
            quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
            quality_scores.append(quality_score)

        quality_scores = torch.stack(quality_scores)  # [N]
        threshold = torch.median(quality_scores)
        high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]

        return high_group, low_group, quality_scores  # ✅ 返回 quality_scores






##############################################################################################

    def mtl_update_compare_kd(self, auxiliary_datas, real_archs, epoch, margin=None,
                          feat_distill_weight=0.001, feat_loss_type="cosine",
                          contrast_weight=0.001, contrast_temp=0.1,svd_loss_weight=0.001, cov_loss_weight=0.05):
        
        # feat_distill_weight=0.01, feat_loss_type="cosine",
        #                   contrast_weight=0.01, contrast_temp=0.1
        total_loss = 0.

        # --- 第一步：获取特征并动态分组 ---
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)  # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]
        B, N, H = features.shape

        # --- 第二步：动态分组（每5个epoch更新）---
        # if epoch % 5 == 1:
        #     quality_scores = []
        #     for i in range(self.auxiliary_head_num):
        #         scores_i = scores[:, i]  # [B]
        #         # 输出一致性
        #         consistency_list = [
        #             self.compute_consistency(scores_i, scores[:, j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ]
        #         consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #         # ✅ 用预测方差替代无效的熵（标量任务）
        #         variance_i = torch.var(scores_i)
        #         # ✅ 移除梯度相似性（高开销）
        #         quality_score = 0.7 * consistency_i + 0.3 * (1.0 / (1.0 + variance_i))
        #         quality_scores.append(quality_score)

        #     quality_scores = torch.stack(quality_scores)
        #     threshold = torch.median(quality_scores)
        #     high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        #     low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
            
        #     # 处理空组情况
        #     if len(high_group) == 0:
        #         high_group = [torch.argmax(quality_scores).item()]
        #         low_group = [i for i in range(self.auxiliary_head_num) if i != high_group[0]]
        #     if len(low_group) == 0:
        #         low_group = [torch.argmin(quality_scores).item()]
        #         high_group = [i for i in range(self.auxiliary_head_num) if i != low_group[0]]
                
        #     self.high_group = high_group
        #     self.low_group = low_group
        # else:
        #     # 使用缓存的分组
        #     high_group = self.high_group if hasattr(self, 'high_group') else list(range(self.auxiliary_head_num))
        #     low_group = self.low_group if hasattr(self, 'low_group') else []

        # # 动态分组（只需分组，无需 quality_scores）
        # quality_scores = []
        # for i in range(self.auxiliary_head_num):
        #     scores_i = scores[:, i]
        #     consistency_list = [
        #         self.compute_consistency(scores_i, scores[:, j])
        #         for j in range(self.auxiliary_head_num) if j != i
        #     ]
        #     consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #     entropy_i = self.compute_entropy(scores_i)
        #     grad_sim_i = torch.mean(
        #         torch.stack([
        #             self.compute_grad_similarity(self.module_lst[i], self.module_lst[j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ])
        #     )
        #     quality_score = 0.4 * consistency_i + 0.3 * (1 - entropy_i) + 0.3 * grad_sim_i
        #     quality_scores.append(quality_score)

        # quality_scores = torch.stack(quality_scores)
        # threshold = torch.median(quality_scores)
        # high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        # low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
        
        #if epoch % 5 == 1:  # 每10个epoch更新一次分组
        self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)
        #high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
        high_group = self.high_group
        low_group = self.low_group
        # --- 第二步：主任务损失 ---
        task_loss_value = 0.0
        for i, (model, (arch_1, arch_2, better_lst)) in enumerate(zip(self.module_lst, auxiliary_datas)):
            s_1 = model.predict(arch_1, sigmoid=False, tanh=False)
            s_2 = model.predict(arch_2, sigmoid=False, tanh=False)
           
            task_loss = self._compare_loss(s_1, s_2, better_lst, margin)
           
            total_loss += task_loss
            task_loss_value += task_loss.item()
        task_loss_value /= len(auxiliary_datas)  # 平均主任务损失

        # --- 第四步：基于训练损失的动态权重 ---
        lambda_high = 1.0
        lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
        lambda_inter = 0.5

        if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
            if not hasattr(self, 'smoothed_loss'):
                self.smoothed_loss = self.last_total_loss
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
            loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
            adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
            lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
            # lambda_low_consensus 保持弱权重
        else:
            self.smoothed_loss = task_loss.item()

        self.last_total_loss = task_loss.item()

        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0

        # 1. 高性能组内蒸馏
        if len(high_group) > 1:
            for i in high_group:
                for j in high_group:
                    if i != j:
                        feat_i = features[:, i, :]
                        feat_j = features[:, j, :]
                        sim = torch.mean(F.cosine_similarity(
                            F.normalize(feat_i, p=2, dim=1),
                            F.normalize(feat_j, p=2, dim=1),
                            dim=1
                        ))
                        distill_loss_high_intra += (1.0 - sim)

        # 2. ✅ 低绩效共识教师蒸馏（安全替代）
        if len(low_group) > 1:
            low_features = features[:, low_group, :]  # [B, L, H]
            consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            for i in low_group:
                feat_i = features[:, i, :]
                sim = torch.mean(F.cosine_similarity(
                    F.normalize(feat_i, p=2, dim=1),
                    F.normalize(consensus_low, p=2, dim=1),
                    dim=1
                ))
                distill_loss_low_consensus += (1.0 - sim)
        elif len(low_group) == 1:
            distill_loss_low_consensus = 0.0

        # 3. 跨组蒸馏（Low → High）带置信度过滤
        if len(low_group) > 0 and len(high_group) > 0:
            high_scores = scores[:, high_group]  # [B, K]
            high_var = torch.var(high_scores, dim=1)  # [B]
            teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
            if teach_mask.any():
                for i in low_group:
                    for j in high_group:
                        feat_i = features[:, i, :][teach_mask]
                        feat_j = features[:, j, :][teach_mask]
                        if feat_i.size(0) > 0:
                            sim = torch.mean(F.cosine_similarity(
                                F.normalize(feat_i, p=2, dim=1),
                                F.normalize(feat_j, p=2, dim=1),
                                dim=1
                            ))
                            distill_loss_inter += (1.0 - sim)

        # --- 第六步：组合蒸馏损失 ---
        total_distill_loss = (
            lambda_high * distill_loss_high_intra +
            lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
            lambda_inter * distill_loss_inter
        )
        total_loss += feat_distill_weight * total_distill_loss

   


        contrastive_loss_total = 0.0
        # 高性能组内对比（数值稳定版）
        if len(self.high_group) > 1:
            high_feats = features[:, self.high_group, :].view(-1, H)  # [B*K, H]
            high_feats = F.normalize(high_feats, dim=1)
            sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp

            labels = torch.arange(B, device=features.device).repeat_interleave(len(self.high_group))
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask.fill_diagonal_(False)

            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
            contrastive_loss_total += contrastive_loss

            ##################### === 添加 SVD + Cov 正则化 ===
            # if svd_loss_weight > 0 or cov_loss_weight > 0:
            #     # 对高性能组特征应用正则化
            #     svd_loss_high = compute_svd_loss(high_feats, svd_loss_weight)
            #     cov_loss_high = compute_cov_loss(high_feats, cov_loss_weight)
            #     contrastive_loss_total += svd_loss_high + cov_loss_high
            #####################################################################
            

    
        ####### 跨组对比（Low vs High 共识）
        if len(self.high_group) > 0 and len(self.low_group) > 0:
            high_consensus = features[:, self.high_group, :].mean(dim=1)  # [B, H]
            low_feats = features[:, self.low_group, :].view(-1, H)        # [B*L, H]
            low_feats = F.normalize(low_feats, dim=1)
            high_consensus = F.normalize(high_consensus, dim=1)
            sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp

            labels = torch.arange(B, device=features.device).repeat_interleave(len(self.low_group))
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
            contrastive_loss_total += contrastive_loss

        #     # # === 添加 SVD + Cov 正则化 ===
        #     # if svd_loss_weight > 0 or cov_loss_weight > 0:
        #     #     # 对低绩效组特征应用正则化
        #     #     svd_loss_low = compute_svd_loss(low_feats, svd_loss_weight)
        #     #     cov_loss_low = compute_cov_loss(low_feats, cov_loss_weight)
        #     #     # 对高性能共识特征应用正则化
        #     #     svd_loss_high_consensus = compute_svd_loss(high_consensus, svd_loss_weight)
        #     #     cov_loss_high_consensus = compute_cov_loss(high_consensus, cov_loss_weight)
        #     #     contrastive_loss_total += svd_loss_low + cov_loss_low + svd_loss_high_consensus + cov_loss_high_consensus

        #     ##################################################################
         
        
          

        total_loss += contrast_weight * contrastive_loss_total
        self.last_total_loss = task_loss_value

        ######## --- 第六步：对比学习（可选）---
        # contrastive_loss = 0.0
        # if B > 1:
        #     feat_norm = F.normalize(features, p=2, dim=2)
        #     sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
        #     labels = torch.arange(N, device=feat_norm.device).repeat(B)   # [B*N]
        #     sim_matrix_flat = sim_matrix.view(-1, N)
        #     contrastive_loss = F.cross_entropy(sim_matrix_flat, labels)
        # total_loss += contrast_weight * contrastive_loss

        return self.update_step(total_loss)
    
   

    def update_compare_kd(self, arch_1, arch_2, better_labels, real_archs, epoch, margin=None,
                      feat_distill_weight=0.001, feat_loss_type="cosine",
                      contrast_weight=0.001, contrast_temp=0.1):
        """
        基于动态分组的安全协作蒸馏策略
        """
        total_loss = 0.

        # --- 第一步：获取所有子网络在 real_archs 上的预测和特征 ---
        scores, features = self._get_student_predictions_and_features(real_archs)
        scores = torch.stack(scores, dim=1)  # [B, N]
        features = torch.stack(features, dim=1)  # [B, N, H]
        B, N, H = features.shape

        # --- 第二步：动态分组（每5个epoch更新）---
        # if epoch % 5 == 1:
        #     quality_scores = []
        #     for i in range(self.auxiliary_head_num):
        #         scores_i = scores[:, i]  # [B]
        #         # 输出一致性
        #         consistency_list = [
        #             self.compute_consistency(scores_i, scores[:, j])
        #             for j in range(self.auxiliary_head_num) if j != i
        #         ]
        #         consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else 0.0
        #         # ✅ 用预测方差替代无效的熵（标量任务）
        #         variance_i = torch.var(scores_i)
        #         # ✅ 移除梯度相似性（高开销）
        #         quality_score = 0.7 * consistency_i + 0.3 * (1.0 / (1.0 + variance_i))
        #         quality_scores.append(quality_score)

        #     quality_scores = torch.stack(quality_scores)
        #     threshold = torch.median(quality_scores)
        #     high_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] > threshold]
        #     low_group = [i for i in range(self.auxiliary_head_num) if quality_scores[i] <= threshold]
            
        #     # 处理空组情况
        #     if len(high_group) == 0:
        #         high_group = [torch.argmax(quality_scores).item()]
        #         low_group = [i for i in range(self.auxiliary_head_num) if i != high_group[0]]
        #     if len(low_group) == 0:
        #         low_group = [torch.argmin(quality_scores).item()]
        #         high_group = [i for i in range(self.auxiliary_head_num) if i != low_group[0]]
                
        #     self.high_group = high_group
        #     self.low_group = low_group
        # else:
        #     # 使用缓存的分组
        #     high_group = self.high_group if hasattr(self, 'high_group') else list(range(self.auxiliary_head_num))
        #     low_group = self.low_group if hasattr(self, 'low_group') else []
        #if epoch % 5 == 1:  # 每10个epoch更新一次分组
        self.high_group, self.low_group, _ = self._dynamic_grouping(real_archs)
        #high_group, low_group, quality_scores = self._dynamic_grouping(real_archs)
        high_group = self.high_group
        low_group = self.low_group
        # --- 第三步：主任务损失 ---
        score_1_leader = self.predict(arch_1, sigmoid=False, tanh=False)  # [B]
        score_2_leader = self.predict(arch_2, sigmoid=False, tanh=False)  # [B]
        task_loss = self._compare_loss(score_1_leader, score_2_leader, better_labels, margin)
        total_loss += task_loss

        # --- 第四步：基于训练损失的动态权重 ---
        lambda_high = 1.0
        lambda_low_consensus = 0.3  # 低绩效共识蒸馏权重（弱）
        lambda_inter = 0.5

        if hasattr(self, 'last_total_loss') and self.last_total_loss is not None:
            if not hasattr(self, 'smoothed_loss'):
                self.smoothed_loss = self.last_total_loss
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * task_loss.item()
            
            loss_improvement = (self.last_total_loss - task_loss.item()) / (self.last_total_loss + 1e-8)
            adaptive_factor = torch.clamp(torch.tensor(loss_improvement * 2.0), -1.0, 1.0).item()
            
            lambda_inter = 0.3 + 0.7 * max(adaptive_factor, 0.0)  # [0.3, 1.0]
            # lambda_low_consensus 保持弱权重
        else:
            self.smoothed_loss = task_loss.item()

        self.last_total_loss = task_loss.item()

        # --- 第五步：安全协作蒸馏 ---
        distill_loss_high_intra = 0.0
        distill_loss_low_consensus = 0.0  # ✅ 替代低绩效组内蒸馏
        distill_loss_inter = 0.0

        # 1. 高性能组内蒸馏
        if len(high_group) > 1:
            for i in high_group:
                for j in high_group:
                    if i != j:
                        feat_i = features[:, i, :]
                        feat_j = features[:, j, :]
                        sim = torch.mean(F.cosine_similarity(
                            F.normalize(feat_i, p=2, dim=1),
                            F.normalize(feat_j, p=2, dim=1),
                            dim=1
                        ))
                        distill_loss_high_intra += (1.0 - sim)

        # 2. ✅ 低绩效共识教师蒸馏（安全替代）
        if len(low_group) > 1:
            low_features = features[:, low_group, :]  # [B, L, H]
            consensus_low = torch.mean(low_features, dim=1)  # [B, H]
            for i in low_group:
                feat_i = features[:, i, :]
                sim = torch.mean(F.cosine_similarity(
                    F.normalize(feat_i, p=2, dim=1),
                    F.normalize(consensus_low, p=2, dim=1),
                    dim=1
                ))
                distill_loss_low_consensus += (1.0 - sim)
        elif len(low_group) == 1:
            distill_loss_low_consensus = 0.0

        # 3. 跨组蒸馏（Low → High）带置信度过滤
        if len(low_group) > 0 and len(high_group) > 0:
            high_scores = scores[:, high_group]  # [B, K]
            high_var = torch.var(high_scores, dim=1)  # [B]
            teach_mask = high_var < torch.median(high_var + 1e-8)  # 高置信样本
            
            if teach_mask.any():
                for i in low_group:
                    for j in high_group:
                        feat_i = features[:, i, :][teach_mask]
                        feat_j = features[:, j, :][teach_mask]
                        if feat_i.size(0) > 0:
                            sim = torch.mean(F.cosine_similarity(
                                F.normalize(feat_i, p=2, dim=1),
                                F.normalize(feat_j, p=2, dim=1),
                                dim=1
                            ))
                            distill_loss_inter += (1.0 - sim)

        # --- 第六步：组合蒸馏损失 ---
        total_distill_loss = (
            lambda_high * distill_loss_high_intra +
            lambda_low_consensus * distill_loss_low_consensus +  # ✅ 安全替代
            lambda_inter * distill_loss_inter
        )
        total_loss += feat_distill_weight * total_distill_loss

        # --- 第七步：对比知识蒸馏（仅组内+跨组）---
        contrastive_loss_total = 0.0

        ######### 高性能组内对比
        if len(high_group) > 1:
            high_feats = features[:, high_group, :].view(-1, H)
            high_feats = F.normalize(high_feats, dim=1)
            sim_mat = torch.mm(high_feats, high_feats.t()) / contrast_temp
            labels = torch.arange(B, device=features.device).repeat_interleave(len(high_group))
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask.fill_diagonal_(False)
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1))
            contrastive_loss_total += contrastive_loss

        # # 跨组对比
        if len(high_group) > 0 and len(low_group) > 0:
            high_consensus = features[:, high_group, :].mean(dim=1)
            low_feats = features[:, low_group, :].view(-1, H)
            low_feats = F.normalize(low_feats, dim=1)
            high_consensus = F.normalize(high_consensus, dim=1)
            sim_mat = torch.mm(low_feats, high_consensus.t()) / contrast_temp
            labels = torch.arange(B, device=features.device).repeat_interleave(len(low_group))
            exp_sim = torch.exp(sim_mat)
            log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = -torch.mean(log_prob[torch.arange(len(labels)), labels])
            contrastive_loss_total += contrastive_loss

        total_loss += contrast_weight * contrastive_loss_total
        return self.update_step(total_loss)
    

    def _compute_current_quality_scores(self, scores):
        """
        计算当前批次的质量得分（用于判断是否需要更新分组）
        """
        B, N = scores.shape
        quality_scores = []
        
        for i in range(N):
            scores_i = scores[:, i]
            # 一致性
            if N > 1:
                consistency_list = [
                    self.compute_consistency(scores_i, scores[:, j])
                    for j in range(N) if j != i
                ]
                consistency_i = torch.mean(torch.stack(consistency_list)) if consistency_list else torch.tensor(0.0)
            else:
                consistency_i = torch.tensor(0.0)
            
            # 稳定性（用方差替代熵）
            stability_i = 1.0 / (1.0 + torch.var(scores_i))
            
            quality_score = 0.7 * consistency_i + 0.3 * stability_i
            quality_scores.append(quality_score)
        
        return torch.stack(quality_scores, dim=0)  # [N]
    

########################################################################################################
    def _get_student_predictions_and_features(self, arch):
        """
        Helper function to get predictions and features from all auxiliary students.
        Returns:
            scores: List of tensors [B]
            features: List of tensors [B, H]
        """
        scores = []
        features = []
        for model in self.module_lst:
            feature = model.predict_feature(arch, sigmoid=False, tanh=False)
            score = model.predict(arch)
            scores.append(score)
            features.append(feature)
        return scores, features


    def mtl_predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tuple[Tensor, List[Tensor]]:
        score = self.predict(arch, sigmoid, tanh)
        lf_score_lst = [model.predict(arch, sigmoid, tanh) for model in self.module_lst]
        return score, lf_score_lst

    def separate_score_predict(self, arch) -> Tensor:
        score_lst = torch.cat([
            model.predict(arch, False, False).unsqueeze(1)
            for model in self.module_lst],
        1)
        return score_lst

    def weighted_score(self, arch) -> Tensor:
        score_lst = self.separate_score_predict(arch)
        confidence_ratio = self._confidence_ratio(arch)
        score = (confidence_ratio * score_lst)
        return score

    def predict_1(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tensor:
        weighted_score = self.weighted_score(arch)
        score = weighted_score.sum(1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def predict(self, arch, sigmoid=True, tanh=False):
        score_lst = self.separate_score_predict(arch)  # [B, N]
        base_conf = self._confidence_ratio(arch)       # [B, N]

        if self.high_group and self.low_group:
            emb = self.arch_embedder(arch)  # [B, D]
            scales = self.group_gating_mlp(emb)  # [B, 2]
            scale_high, scale_low = scales[:, 0:1], scales[:, 1:2]  # [B,1]

            group_mask = torch.zeros_like(base_conf)
            group_mask[:, self.high_group] = 1
            group_mask[:, self.low_group] = 0

            dynamic_scale = group_mask * scale_high + (1 - group_mask) * scale_low  # [B, N]
            weighted_score = base_conf * dynamic_scale * score_lst
        else:
            weighted_score = base_conf * score_lst

        score = weighted_score.sum(dim=1)
        return torch.sigmoid(score) if sigmoid else score

    def _get_expert_similarity_matrix(self):
        """获取专家相似度矩阵 [N, N]"""
        if self.expert_reps is None:
            # 无法计算多样性，返回零矩阵（无惩罚）
            N = self.auxiliary_head_num
            return torch.zeros(N, N, device=next(self.parameters()).device)
        
        # 余弦相似度矩阵
        sim_matrix = torch.mm(self.expert_reps, self.expert_reps.t())  # [N, N]
        return sim_matrix
    
    

    
    
    

    def _confidence_ratio(self, archs) -> Tensor:
        if self.use_uniform_confidence:
            confidence_ratio = torch.softmax(self.confidence, 1)
        else:
            arch_embeddings = self.arch_embedder(archs)
            confidence_ratio = torch.softmax(self.confidence_mlp(arch_embeddings), 1)
        return confidence_ratio

    def _compare_loss(self, s_1, s_2, better_labels, margin=None):
        s_1 = s_1.squeeze()
        s_2 = s_2.squeeze()
        better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()


      
        return loss.item()
    
    
