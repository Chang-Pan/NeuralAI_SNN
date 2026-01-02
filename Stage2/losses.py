import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from network import psp


class SpikeLoss(torch.nn.Module):
    """
    SNN 中常用的三种损失形式封装：
    1) spike count loss
    2) spike kernel loss
    3) spike softmax loss
    """
    def __init__(self, network_config):
        super(SpikeLoss, self).__init__()
        self.network_config = network_config
        self.criterion = torch.nn.CrossEntropyLoss()

    def spike_count(self, outputs, target, network_config, layer_config):
        """
        基于脉冲发放次数的损失
        """
        delta = loss_count.apply(outputs, target, network_config, layer_config)
        return 0.5 * torch.sum(delta ** 2)

    def spike_kernel(self, outputs, target, network_config):
        """
        基于 PSP kernel 的逐时间步误差
        """
        delta = loss_kernel.apply(outputs, target, network_config)
        return 0.5 * torch.sum(delta ** 2)

    def spike_soft_max(self, outputs, target):
        """
        将时间维度上的 spike 累加后，
        使用 softmax + cross entropy
        """
        logits = outputs.sum(dim=4).squeeze(-1).squeeze(-1)
        log_probs = f.log_softmax(logits, dim=1)
        return self.criterion(log_probs, target)


class loss_count(torch.autograd.Function):
    """
    Spike count loss 的自定义梯度
    目标是：控制神经元的发放次数，而不是每个时间点的精确值
    """
    @staticmethod
    def forward(ctx, outputs, target, network_config, layer_config):
        desired_count = network_config['desired_count']
        undesired_count = network_config['undesired_count']

        batch_size, channels, height, width, n_steps = outputs.shape

        # 实际脉冲发放次数（沿时间维度求和）
        output_spike_count = torch.sum(outputs, dim=4)

        # 计数误差（按时间归一化）
        delta = (output_spike_count - target) / n_steps

        # ---------- 第一类 mask ----------
        # undesired 神经元：只惩罚「发放过多」的情况
        mask = torch.ones_like(output_spike_count)
        mask[target == undesired_count] = 0
        mask[delta < 0] = 0
        delta[mask == 1] = 0

        # ---------- 第二类 mask ----------
        # desired 神经元：只惩罚「发放不足」的情况
        mask = torch.ones_like(output_spike_count)
        mask[target == desired_count] = 0
        mask[delta > 0] = 0
        delta[mask == 1] = 0

        # 将计数误差扩展到每一个时间步
        delta = delta.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)

        return delta

    @staticmethod
    def backward(ctx, grad_output):
        # 该 loss 对 outputs 是“恒等反传”
        return grad_output, None, None, None


class loss_kernel(torch.autograd.Function):
    """
    基于 PSP kernel 的逐时间误差
    """
    @staticmethod
    def forward(ctx, outputs, target, network_config):
        # 将目标 spike 转换为 PSP 形式
        target_psp = psp(target, network_config)

        # PSP 层面的逐时间误差
        delta = outputs - target_psp
        return delta

    @staticmethod
    def backward(ctx, grad_output):
        # 直接将梯度传回 outputs
        return grad_output, None, None
