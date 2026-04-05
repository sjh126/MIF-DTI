# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((labels.shape[0], 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
class HybridLoss(nn.Module):
    """
    混合损失：Weighted CE + FocalLoss
    Args:
        weight_ce: 分类权重，用于 CrossEntropyLoss
        alpha_focal: FocalLoss alpha，类别权重
        gamma: FocalLoss gamma，关注难样本
        lambda_focal: Focal部分占比，0->纯CE, 1->纯Focal
        DEVICE: 设备
    """
    def __init__(self, weight_ce=None, alpha_focal=1.0, gamma=2.0, lambda_focal=0.5, DEVICE='cuda'):
        super(HybridLoss, self).__init__()
        self.weight_ce = weight_ce
        self.alpha_focal = alpha_focal
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self.DEVICE = DEVICE

        # CE with reduction='none' to compute per-sample loss
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weight_ce, reduction='none')

    def forward(self, logits, labels):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size] long tensor
        """
        # 1. 计算 CE loss
        ce = self.ce_loss_fn(logits, labels)  # [batch_size]

        # 2. 计算 pt = 模型对真实类别的预测概率
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.view(-1, 1)).squeeze(1)  # [batch_size]

        # 3. 计算 Focal Loss
        focal_factor = (1 - pt) ** self.gamma
        focal_loss = self.alpha_focal * focal_factor * ce  # [batch_size]

        # 4. 混合 CE + Focal
        loss = (1 - self.lambda_focal) * ce + self.lambda_focal * focal_loss

        # 5. 返回平均 loss
        return loss.mean()