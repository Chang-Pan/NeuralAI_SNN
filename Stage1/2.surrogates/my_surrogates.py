import torch
import torch.nn as nn

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