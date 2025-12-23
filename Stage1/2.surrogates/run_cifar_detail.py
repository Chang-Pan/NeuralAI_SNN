import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from spikingjelly.activation_based import neuron, functional, surrogate

# --- 实验配置 ---
ARTIFACT_DIR = "physics_of_snn_phase_transition_v2"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置
T = 4
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10 
# Alpha 扫描范围：从极宽(平缓)到极窄(陡峭)
ALPHA_LIST = np.logspace(np.log10(0.5), np.log10(20), 12).tolist() 
SEEDS = 3

class PhysicsProbe:
    """物理探针：测量梯度流(反向)和发放率(前向)"""
    def __init__(self, model):
        self.grad_norms = {}
        self.firing_rates = {}
        self.hooks = []
        self.model = model
        
        # 注册钩子
        for name, module in model.named_modules():
            # 监控卷积层的梯度
            if isinstance(module, nn.Conv2d):
                self._register_grad_hook(name, module)
            # 监控神经元的发放率
            if isinstance(module, neuron.LIFNode):
                self._register_firing_hook(name, module)
    
    def _register_grad_hook(self, name, module):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                # 记录传入该层的梯度的 L2 范数
                norm = grad_output[0].norm().item()
                if name not in self.grad_norms: self.grad_norms[name] = []
                self.grad_norms[name].append(norm)
        self.hooks.append(module.register_full_backward_hook(hook))

    def _register_firing_hook(self, name, module):
        def hook(module, input, output):
            # output 是 spike tensor [T, B, C, H, W] 或 [B, C...]
            # 计算平均发放率
            fr = output.mean().item()
            if name not in self.firing_rates: self.firing_rates[name] = []
            self.firing_rates[name].append(fr)
        self.hooks.append(module.register_forward_hook(hook))

    def get_metrics(self):
        """计算序参量"""
        # 1. 梯度传输率 (Input / Output)
        # 排序：确保 conv1 在 conv2 之前
        grad_keys = sorted(list(self.grad_norms.keys()), key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        
        chi = 0.0
        if len(grad_keys) >= 2:
            first_layer = grad_keys[0]  # 浅层
            last_layer = grad_keys[-1]  # 深层
            
            g_in = np.mean(self.grad_norms[first_layer])
            g_out = np.mean(self.grad_norms[last_layer])
            
            # 避免除以零
            if g_out > 1e-9:
                chi = g_in / g_out
            else:
                chi = 0.0 # 梯度完全消失
        
        # 2. 平均发放率 (所有层平均)
        avg_fr = 0.0
        if self.firing_rates:
            all_rates = [np.mean(v) for v in self.firing_rates.values()]
            avg_fr = np.mean(all_rates)

        return chi, avg_fr

    def clear(self):
        self.grad_norms = {}
        self.firing_rates = {}
        
    def remove(self):
        for h in self.hooks: h.remove()

class DynamicDepthSNN(nn.Module):
    def __init__(self, depth_type, alpha):
        super().__init__()
        self.T = T
        # 关键：Sigmoid 的 alpha 参数控制陡峭程度
        # alpha 越大，导数越陡，梯度窗口越窄 -> 对应相变理论中的"有序/冻结"倾向
        surr_func = surrogate.Sigmoid(alpha=alpha)
        
        layers = []
        in_channels = 3
        
        # 增加深度对比度
        if depth_type == "Shallow":
            configs = [64, 'M', 128] # 2层卷积
        elif depth_type == "Deep":
            # 7层卷积，足以产生指数级梯度消失
            configs = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
            
        current_channels = 3
        for i, cfg in enumerate(configs):
            if cfg == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.Conv2d(current_channels, cfg, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(cfg)) # BN 有助于缓解，但在极深SNN中仍会遇到问题
                layers.append(neuron.LIFNode(surrogate_function=surr_func, detach_reset=True))
                current_channels = cfg
                
        self.features = nn.Sequential(*layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            # 这里需要处理一下 T 维度，或者直接用 functional 模拟一次
            # 简单起见，假设不随时间改变 shape
            out_dummy = self.features(dummy)
            flat_dim = out_dummy.flatten().shape[0]
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 10)
        )

    def forward(self, x):
        functional.reset_net(self)
        # 静态图输入，复制 T 次
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) 
        
        # 直接通过 sequential
        # SpikingJelly 的 sequential 自动处理时间步 (如果是 LIFNode)
        # 但为了保险和监控，我们手动循环或者用 functional.multi_step_forward
        outputs = []
        for t in range(self.T):
            feature_out = self.features(x) # 广播机制或 reuse
            out = self.classifier(feature_out)
            outputs.append(out)
        
        return torch.stack(outputs).mean(dim=0)

# --- 数据 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 使用 Subset 加速调试，正式跑去掉 Subset
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 减少数据量以符合 1000 epoch 预算 (如果需要更快，可以下采样)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

results = []

print("=== SNN Physics Phase Transition Experiment ===")

for depth_type in ["Shallow", "Deep"]:
    for alpha in ALPHA_LIST:
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = DynamicDepthSNN(depth_type, alpha).to(DEVICE)
            probe = PhysicsProbe(model)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()
            
            print(f"[{depth_type} | Alpha={alpha:.1f} | Seed={seed}]", end=" ")
            
            # --- Phase 1: Initialization Metrics (Before Training) ---
            # 测量“零时刻”的梯度流，这是纯粹的物理属性，未被优化污染
            model.zero_grad()
            probe.clear()
            # 取一个 batch 测初始化状态
            imgs, labels = next(iter(train_loader))
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            
            init_chi, init_fr = probe.get_metrics()
            print(f"Init: Chi={init_chi:.2e}, FR={init_fr:.2f} | ", end="")
            
            # --- Phase 2: Training Dynamics ---
            best_acc = 0.0
            train_chis = []
            
            for epoch in range(EPOCHS):
                model.train()
                probe.clear()
                
                # 训练 50 个 step 采样
                for i, (imgs, labels) in enumerate(train_loader):
                    if i > 50: break 
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
                
                # 记录训练中的动力学
                epoch_chi, epoch_fr = probe.get_metrics()
                train_chis.append(epoch_chi)
                
                # 验证
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                        out = model(imgs)
                        _, pred = out.max(1)
                        total += labels.size(0)
                        correct += pred.eq(labels).sum().item()
                
                acc = 100. * correct / total
                if acc > best_acc: best_acc = acc
            
            print(f"Best Acc: {best_acc:.2f}%")
            
            results.append({
                "Depth": depth_type,
                "Alpha": alpha,
                "Seed": seed,
                "Init_Chi": init_chi,      # 初始梯度传输率
                "Init_FR": init_fr,        # 初始发放率
                "Train_Chi_Mean": np.mean(train_chis), # 训练态梯度传输率
                "BestAccuracy": best_acc
            })
            
            probe.remove()

# --- 绘图分析 ---
df = pd.DataFrame(results)
df.to_csv(os.path.join(ARTIFACT_DIR, "snn_phase_data.csv"), index=False)

sns.set_context("paper", font_scale=1.5)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: 准确率相变 (验证冻结相和可训练相)
sns.lineplot(data=df, x="Alpha", y="BestAccuracy", hue="Depth", marker="o", ax=axes[0])
axes[0].set_xscale("log")
axes[0].set_title("Accuracy Phase Transition")
axes[0].set_ylabel("Test Accuracy (%)")

# Plot 2: 梯度传输率 (验证 Ganguli 理论)
# 理论预测：Deep 网络在 Alpha 过大时，Chi 会指数级衰减到 0
sns.lineplot(data=df, x="Alpha", y="Init_Chi", hue="Depth", marker="s", ax=axes[1])
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("Gradient Flow ($\\chi$) at Init")
axes[1].axhline(1.0, color='r', linestyle='--', label='Critical Line')
axes[1].set_ylabel("Gradient Transmission Ratio")

# Plot 3: 发放率 (验证 Rossbroich 理论)
# 理论预测：Alpha 影响有效发放，过大 Alpha 可能导致静息
sns.lineplot(data=df, x="Alpha", y="Init_FR", hue="Depth", marker="^", ax=axes[2])
axes[2].set_xscale("log")
axes[2].set_title("Firing Rate Phase")
axes[2].set_ylabel("Avg Firing Rate")

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "physics_analysis.png"))
print("Done.")