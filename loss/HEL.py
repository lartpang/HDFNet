# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 下午10:02
# @Author  : Lart Pang
# @FileName: HEL.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import torch.nn.functional as F
from torch import nn


class HEL(nn.Module):
    def __init__(self):
        super(HEL, self).__init__()
        print("You are using `HEL`!")
        self.eps = 1e-6

    def edge_loss(self, pred, target):
        edge = target - F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        # input, kernel_size, stride=None, padding=0
        numerator = (edge * (pred - target).abs_()).sum([2, 3])
        denominator = edge.sum([2, 3]) + self.eps
        return numerator / denominator

    def region_loss(self, pred, target):
        # 该部分损失更强调前景区域内部或者背景区域内部的预测一致性
        numerator_fore = (target - target * pred).sum([2, 3])
        denominator_fore = target.sum([2, 3]) + self.eps

        numerator_back = ((1 - target) * pred).sum([2, 3])
        denominator_back = (1 - target).sum([2, 3]) + self.eps
        return numerator_fore / denominator_fore + numerator_back / denominator_back

    def forward(self, pred, target):
        edge_loss = self.edge_loss(pred, target)
        region_loss = self.region_loss(pred, target)
        return (edge_loss + region_loss).mean()
