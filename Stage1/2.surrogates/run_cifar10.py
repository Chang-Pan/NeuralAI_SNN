import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import sys
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spikingjelly.activation_based import neuron, functional, surrogate
import atexit

# --- 简单日志：标准输出 + 文件 ---
log_filename = "snn_search_result.log"
_log_file = open(log_filename, "a", encoding="utf-8")


def log(msg: str):
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()


@atexit.register
def _close_log_file():
    try:
        _log_file.close()
    except Exception:
        pass


log(f"PyTorch version: {torch.__version__}")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"Current device: {torch.cuda.get_device_name(0)}")
    log(f"Device count: {torch.cuda.device_count()}")


# --- 辅助函数：阶跃函数 ---
@torch.jit.script
def heaviside(x: torch.Tensor):
    """
    前向传播用的阶跃函数：x >= 0 时输出 1，否则输出 0
    """
    return (x >= 0).float()

# =========================================================
# 1. SuperSpike 实现
# 公式: h(x) = 1 / (beta * |x| + 1)^2
# =========================================================
class SuperSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        
        # 实现图片中的公式: 1 / (beta * |x| + 1)^2
        denom = (alpha * x.abs() + 1.0)
        grad_x = grad_output * (1.0 / (denom * denom))
        
        return grad_x, None

class SuperSpike(nn.Module):
    def __init__(self, alpha=100.0, spiking=True):
        """
        SuperSpike 替代梯度
        :param alpha: 对应公式中的 beta，控制梯度的陡峭程度
        """
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            return SuperSpikeFunction.apply(x, self.alpha)
        else:
            return heaviside(x)

# =========================================================
# 2. Sigmoid' (Image Version) 实现
# 公式: h(x) = s(x)(1 - s(x)), 其中 s(x) = sigmoid(beta * x)
# =========================================================
class SigmoidDerivativeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        
        # 计算 s(x) = sigmoid(beta * x)
        sigmoid_x = torch.sigmoid(alpha * x)
        
        # 实现: h(x) = s(x) * (1 - s(x))
        grad_x = grad_output * sigmoid_x * (1.0 - sigmoid_x)
        
        return grad_x, None

class SigmoidDerivative(nn.Module):
    def __init__(self, alpha=4.0, spiking=True):
        """
        Sigmoid' 替代梯度 (图片版本)
        :param alpha: 对应公式中的 beta
        """
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            return SigmoidDerivativeFunction.apply(x, self.alpha)
        return heaviside(x)

# =========================================================
# 3. Esser et al. 实现
# 公式: h(x) = max(0, 1.0 - beta * |x|)
# =========================================================
class EsserFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        
        # 实现图片公式: max(0, 1.0 - beta * |x|)
        grad_x = grad_output * torch.clamp(1.0 - alpha * x.abs(), min=0.0)
        
        return grad_x, None

class Esser(nn.Module):
    def __init__(self, alpha=1.0, spiking=True):
        """
        Esser et al. 替代梯度
        :param alpha: 对应公式中的 beta，通常设为 1.0 或更大
        """
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            return EsserFunction.apply(x, self.alpha)
        return heaviside(x)

# ----------------------------------------
# 1. 定义超参数和设置
# ----------------------------------------

T = 8             # 仿真总时长 (SNN 的关键参数)
BATCH_SIZE = 64   # 批处理大小
EPOCHS = 10       # 控制总 epoch 预算（3×3×15×3=405 < 500）
LR = 1e-3         # 学习率
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BETA = 10.0       # 替代梯度中的超参数, 论文中规定值
BASELINE_ACC = 70.0

# 网格搜索范围：alpha 统一对数插值 15 点 (0.05 ~ 1e3)，lr 对数插值 6 点 (1e-4 ~ 1e-1)
LR_GRID = [float(x) for x in np.logspace(-4, -1, 6)]
ALPHA_GRID: Dict[str, List[float]] = {
    "SuperSpike": [float(x) for x in np.logspace(np.log10(0.05), 3, 15)],
    "Sigmoid": [float(x) for x in np.logspace(np.log10(0.05), 3, 15)],
    "Esser": [float(x) for x in np.logspace(np.log10(0.05), 3, 15)],
}
ARTIFACT_DIR = "artifacts"

log(f"--- 实验设置 ---")
log(f"设备 (DEVICE): {DEVICE}")
log(f"仿真时长 (T): {T}")
log(f"批大小 (BATCH_SIZE): {BATCH_SIZE}")
log(f"训练轮数 (EPOCHS): {EPOCHS}")
log(f"------------------\n")

# ----------------------------------------
# 2. 加载和预处理 CIFAR10 数据集
# ----------------------------------------
log("正在加载 CIFAR10 数据集...")
# CIFAR10 图像的均值和标准差 (用于归一化)
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 简单数据增强：随机翻转
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

# 加载数据
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

log("数据集加载完毕。\n")

# ----------------------------------------
# 3. 定义基础的卷积 SNN 模型
# ----------------------------------------
# 使用 nn.Sequential 快速搭建一个简单的 CNN 结构
# 关键在于在激活函数的位置换上 SNN 的脉冲神经元

class BasicCSNN(nn.Module):
    # 增加 surrogate_function 参数
    def __init__(self, T, surrogate_function=surrogate.Sigmoid()):
        super().__init__()
        self.T = T  # 保存仿真时长
        # logging.info(f"Initializing Network with Surrogate: {surrogate_function.__class__.__name__}")

        # 定义网络结构
        # 结构：[卷积 -> 脉冲 -> 池化] x 2 -> [展平 -> 全连接 -> 脉冲] -> [全连接]
        self.net = nn.Sequential(
            # 块 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # --- 核心：使用 LIF 神经元 ---
            # 激活驱动:LIFNode 在前向传播时模拟 LIF 神经元动力学，在反向传播时，SpikingJelly 会自动使用“替代梯度”进行计算。
            neuron.LIFNode(surrogate_function=surrogate_function),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # 块 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            neuron.LIFNode(surrogate_function=surrogate_function),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # 展平
            nn.Flatten(),

            # 全连接层 1
            nn.Linear(64 * 8 * 8, 128), # 64 * 8 * 8 = 4096
            neuron.LIFNode(surrogate_function=surrogate_function),

            # 输出层 (全连接层 2)
            # 输出层通常不使用脉冲神经元，而是直接输出膜电位或累积电流
            # 这样可以方便地与交叉熵损失配合使用
            nn.Linear(128, 10) # 10个类别
        )

    def forward(self, x):
        # --- SNN 算法思路的核心 ---
        # SNN 神经元是有状态的（例如膜电位 V），在处理一个新样本前必须重置
        # 1. 重置网络中所有神经元的状态
        functional.reset_net(self)

        # 准备一个列表来收集 T 个时间步的输出
        # (T, N, C)，T=时间步, N=BatchSize, C=类别数
        outputs_over_time = []

        # 2. SNN 的时间步循环
        # 对于静态图像 (如CIFAR10)，我们在 T 个时间步内输入 *相同* 的图像 x
        # 神经元会在这 T 步内不断累积输入并发放脉冲
        for t in range(self.T):
            # 运行一步前向传播
            out_t = self.net(x)
            outputs_over_time.append(out_t)

        # 3. 聚合 T 个时间步的输出
        # (T, N, 10) -> (T, N, 10)
        outputs_stack = torch.stack(outputs_over_time)
        
        # 4. 解码：计算 T 步内的平均输出
        # (T, N, 10) -> (N, 10)
        # 我们取所有时间步输出的平均值，作为最终的分类 "logits"
        # 这是一种常见的 SNN 解码方式（Rate Coding / Mean Output）
        return outputs_stack.mean(dim=0)

# ----------------------------------------
# 4. 准备实验配置 (Optuna 版本)
# ----------------------------------------

# 定义要对比的替代梯度名称
surrogate_types = ["Sigmoid", "Esser"]

# 辅助函数：根据名称和 alpha 创建 surrogate 实例
def get_surrogate_func(name, alpha):
    if name == "SuperSpike":
        return SuperSpike(alpha=alpha)
    elif name == "Sigmoid":
        return SigmoidDerivative(alpha=alpha)
    elif name == "Esser":
        return Esser(alpha=alpha)
    raise ValueError(f"Unknown surrogate type: {name}")

# 定义损失函数 (全局使用)
criterion = nn.CrossEntropyLoss()

# ----------------------------------------
# 5. 训练和评估循环
# ----------------------------------------

# --- 训练函数 (Train Loop) ---
def train_epoch(model, optimizer, epoch, model_name):
    model.train()  # 设置为训练模式
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs) # 这里的 model 是参数传进来的
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    end_time = time.time()
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / total
    log(f"[{model_name}] Epoch {epoch+1} Train | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {end_time - start_time:.2f}s")
    return avg_loss, acc

# --- 评估函数 (Eval Loop) ---
def test_epoch(model, epoch, model_name):
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    # 评估时不需要计算梯度
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 统计准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    log(f"[{model_name}] Epoch {epoch+1} Test  | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    return avg_loss, acc

# ----------------------------------------
# 6. 网格搜索 (学习率 & alpha)
# ----------------------------------------
log(f"=== 开始对比实验 (总 Epochs: {EPOCHS}) ===\n")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_single_config(surrogate_name: str, alpha: float, lr: float):
    surr_func = get_surrogate_func(surrogate_name, alpha)
    model = BasicCSNN(T=T, surrogate_function=surr_func).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    baseline_epoch: Optional[int] = None
    history = []

    for epoch in range(EPOCHS):
        label = f"{surrogate_name}_a={alpha:.3g}_lr={lr:.0e}"
        train_loss, train_acc = train_epoch(model, optimizer, epoch, label)
        test_loss, test_acc = test_epoch(model, epoch, label)

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

        if test_acc > best_acc:
            best_acc = test_acc
        if baseline_epoch is None and test_acc >= BASELINE_ACC:
            baseline_epoch = epoch + 1

    return {
        "best_acc": best_acc,
        "baseline_epoch": baseline_epoch,
        "history": history,
    }


def run_grid_search(surrogate_name: str, lr_grid: List[float], alpha_grid: List[float]):
    log(f"\n>>> 开始 {surrogate_name} 的网格搜索 (lr x alpha = {len(lr_grid) * len(alpha_grid)})")
    records = []

    for lr in lr_grid:
        for alpha in alpha_grid:
            metrics = run_single_config(surrogate_name, alpha, lr)
            record = {
                "surrogate": surrogate_name,
                "lr": lr,
                "alpha": alpha,
                "best_acc": metrics["best_acc"],
                "baseline_epoch": metrics["baseline_epoch"],
            }
            records.append(record)
            log(
                f"{surrogate_name} | lr={lr:.0e}, alpha={alpha:.3g} | best_acc={metrics['best_acc']:.2f}% | baseline_epoch={metrics['baseline_epoch']}"
            )

    return records


def plot_heatmap(records: List[Dict], surrogate_name: str, outfile: str):
    # 确保目录存在
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="alpha", columns="lr", values="best_acc")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, origin="lower", cmap="viridis")
    plt.title(f"{surrogate_name} Accuracy Heatmap")
    plt.xlabel("Learning Rate")
    plt.ylabel("Alpha")
    plt.xticks(ticks=range(len(pivot.columns)), labels=[f"{lr:.0e}" for lr in pivot.columns])
    plt.yticks(ticks=range(len(pivot.index)), labels=[f"{a:.3g}" for a in pivot.index])
    plt.colorbar(im, label="Best Test Accuracy (%)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    log(f"保存热力图: {outfile}")


def summarize_baseline(records: List[Dict], surrogate_name: str):
    reached = [r for r in records if r["baseline_epoch"] is not None]
    if not reached:
        log(f"{surrogate_name}: 没有配置达到 {BASELINE_ACC}% 基线")
        return None
    winner = min(reached, key=lambda r: r["baseline_epoch"])
    log(
        f"{surrogate_name}: 最快达到基线的配置 lr={winner['lr']:.0e}, alpha={winner['alpha']:.3g}, epoch={winner['baseline_epoch']}"
    )
    return winner


all_records: List[Dict] = []
baseline_summary = {}

for name in surrogate_types:
    records = run_grid_search(name, LR_GRID, ALPHA_GRID[name])
    all_records.extend(records)
    heatmap_path = os.path.join(ARTIFACT_DIR, f"heatmap_{name}.png")
    plot_heatmap(records, name, heatmap_path)
    baseline_summary[name] = summarize_baseline(records, name)

# 保存表格数据，便于复现或外部可视化
results_csv = os.path.join(ARTIFACT_DIR, "snn_lr_alpha_grid.csv")
if os.path.exists(results_csv):
    pd.DataFrame(all_records).to_csv(results_csv, mode='a', header=False, index=False)
    log(f"结果已追加保存: {results_csv}")
else:
    pd.DataFrame(all_records).to_csv(results_csv, index=False)
    log(f"完整结果已保存: {results_csv}")

log("\n=== 所有实验完成，最终结果汇总 ===")
for name in surrogate_types:
    winner = baseline_summary.get(name)
    if winner is None:
        log(f"模型: {name:<10} | 未达到 {BASELINE_ACC}% 基线")
    else:
        log(
            f"模型: {name:<10} | 最快基线 lr={winner['lr']:.0e}, alpha={winner['alpha']:.3g}, epoch={winner['baseline_epoch']} | best_acc={winner['best_acc']:.2f}%"
        )
