import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_


def init(n_t, ts):
    """
    初始化突触响应核 synaptic alpha kernel（离散指数衰减）
    用于时间卷积 / PSP 计算

    参数:
        n_t : 时间步数
        ts  : 突触时间常数 tau_s

    返回:
        synaptic_kernel : shape = (1, 1, 1, 1, n_t)
    """
    n_steps = n_t
    tau_s = ts

    # synaptic alpha kernel，放在 GPU 上
    synaptic_kernel = torch.zeros(1, 1, 1, 1, n_steps).cuda()

    # t = 0 时刻的初始脉冲响应
    synaptic_kernel[..., 0] = 1.0

    # 离散时间指数衰减：a[t+1] = a[t] - a[t] / tau_s
    for t in range(n_steps - 1):
        synaptic_kernel[..., t + 1] = (
            synaptic_kernel[..., t]
            - synaptic_kernel[..., t] / tau_s
        )

    # 归一化，符合 PSP 的离散近似形式
    synaptic_kernel /= tau_s

    return synaptic_kernel


def get_mnist(data_path, network_config):
    """
    加载 MNIST 数据集，并构造 DataLoader

    参数:
        data_path      : MNIST 数据存储路径
        network_config : 包含 batch_size 的配置字典

    返回:
        trainloader, testloader
    """
    print("loading MNIST")

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = network_config['batch_size']

    # MNIST 标准预处理：ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        data_path, train=True, transform=transform, download=True
    )
    testset = torchvision.datasets.MNIST(
        data_path, train=False, transform=transform, download=True
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return trainloader, testloader


def psp(inputs, network_config):
    """
    计算突触后电位（Post-Synaptic Potential, PSP）

    离散形式：
        syn[t] = syn[t-1] - syn[t-1] / tau_s + input[t]
        psp[t] = syn[t] / tau_s

    参数:
        inputs : shape = (B, C, H, W, T)，脉冲输入
        network_config : 包含 n_steps, tau_s 的配置字典

    返回:
        synaptic_outputs : shape = (B, C, H, W, T)
    """
    batch_size, channels, height, width, _ = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']

    # 当前时刻的突触状态（积分变量）
    synaptic_state = torch.zeros(
        batch_size, channels, height, width
    ).cuda()

    # 保存所有时间步的 PSP 输出
    synaptic_outputs = torch.zeros(
        batch_size, channels, height, width, n_steps
    ).cuda()

    for t in range(n_steps):
        # 突触动力学更新：指数衰减 + 当前脉冲输入
        synaptic_state = (
            synaptic_state
            - synaptic_state / tau_s
            + inputs[..., t]
        )

        # PSP 输出（离散近似）
        synaptic_outputs[..., t] = synaptic_state / tau_s

    return synaptic_outputs
