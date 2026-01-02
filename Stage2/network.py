import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from data_utils import psp


class EventDrivenNeuron(torch.autograd.Function):
    """
    事件驱动 LIF 神经元的自定义 autograd Function
    前向：模拟 LIF + 突触动力学
    反向：基于事件的时间反向传播（event-driven BPTT）
    """

    @staticmethod
    def forward(ctx, inputs, net_cfg, layer_cfg, syn_kernel):
        """
        inputs: [B, C, H, W, T]
        """
        batch_size, channels, height, width, n_steps = inputs.shape

        # 网络与层参数
        tau_m = net_cfg['tau_m']
        tau_s = net_cfg['tau_s']
        threshold = layer_cfg['threshold']

        theta_m = 1.0 / tau_m
        theta_s = 1.0 / tau_s

        # 状态变量
        membrane_potential = torch.zeros(
            batch_size, channels, height, width, device=inputs.device
        )
        synaptic_current = torch.zeros_like(membrane_potential)

        # 用于反向传播的时间序列缓存
        membrane_traces = []
        membrane_updates = []
        spike_traces = []
        synaptic_traces = []

        for t in range(n_steps):
            # 膜电位更新（LIF 离散形式）
            membrane_update = (-theta_m) * membrane_potential + inputs[..., t]
            membrane_potential = membrane_potential + membrane_update

            # 脉冲发放
            spike = (membrane_potential > threshold).float()

            membrane_traces.append(membrane_potential.clone())
            membrane_updates.append(membrane_update)
            spike_traces.append(spike)

            # 发放后膜电位重置
            membrane_potential = membrane_potential * (1.0 - spike)

            # 突触电流更新（指数衰减）
            synaptic_current = synaptic_current + (spike - synaptic_current) * theta_s
            synaptic_traces.append(synaptic_current.clone())

        # 堆叠时间维度
        membrane_traces = torch.stack(membrane_traces, dim=4)
        membrane_updates = torch.stack(membrane_updates, dim=4)
        spike_traces = torch.stack(spike_traces, dim=4)
        synaptic_traces = torch.stack(synaptic_traces, dim=4)

        # 保存反向传播所需变量
        ctx.save_for_backward(
            membrane_updates,
            spike_traces,
            membrane_traces,
            synaptic_traces,
            torch.tensor([threshold, tau_s, theta_m], device=inputs.device),
            syn_kernel
        )

        return synaptic_traces

    @staticmethod
    def backward(ctx, grad_synaptic):
        """
        grad_synaptic: 上游对 synaptic_traces 的梯度
        """
        (
            membrane_updates,
            spike_traces,
            membrane_traces,
            synaptic_traces,
            others,
            syn_kernel
        ) = ctx.saved_tensors

        batch_size, channels, height, width, n_steps = grad_synaptic.shape
        threshold, tau_s, theta_m = others.tolist()

        # 判定是否使用 surrogate 梯度的阈值
        synaptic_threshold = 1.0 / (4.0 * tau_s)

        grad_inputs = torch.zeros_like(grad_synaptic)

        # 突触核对 spike 的偏导
        partial_a = syn_kernel / (-tau_s)
        partial_a = partial_a.repeat(batch_size, channels, height, width, 1)

        # 时间递推的误差状态（θ）
        theta_state = torch.zeros(
            batch_size, channels, height, width, device=grad_synaptic.device
        )

        # 反向时间循环
        for t in range(n_steps - 1, -1, -1):
            time_window = n_steps - t
            spike = spike_traces[..., t]

            # -------- 对膜电位 u 的偏导（事件驱动）--------
            partial_u = torch.clamp(
                -1.0 / membrane_updates[..., t],
                min=-8.0,
                max=0.0
            ) * spike

            # spike → synaptic 的时间卷积梯度
            partial_a_u = (
                partial_u.unsqueeze(-1).repeat(1, 1, 1, 1, time_window)
                * partial_a[..., :time_window]
            )

            grad_current = torch.sum(
                partial_a_u * grad_synaptic[..., t:n_steps] * tau_s,
                dim=4
            )

            # 时间递推误差项
            if t != n_steps - 1:
                grad_current += (
                    theta_state
                    * membrane_traces[..., t]
                    * (-1.0)
                    * theta_m
                    * partial_u
                )
                grad_current += (
                    theta_state
                    * (1.0 - theta_m)
                    * (1.0 - spike)
                )

            # 更新时间误差状态
            theta_state = (
                grad_current * spike
                + theta_state * (1.0 - spike) * (1.0 - theta_m)
            )

            # -------- surrogate 梯度部分 --------
            grad_a = torch.sum(
                syn_kernel[..., :time_window] * grad_synaptic[..., t:n_steps],
                dim=-1
            )

            a = 0.2
            surrogate = torch.clamp(
                (-membrane_traces[..., t] + threshold) / a,
                min=-8.0,
                max=8.0
            )
            surrogate = torch.exp(surrogate)
            surrogate = surrogate / ((1.0 + surrogate) ** 2 * a)

            grad_a = grad_a * surrogate

            synaptic_value = synaptic_traces[..., t]
            grad_current[synaptic_value < synaptic_threshold] = grad_a[
                synaptic_value < synaptic_threshold
            ]

            grad_inputs[..., t] = grad_current

        return grad_inputs, None, None, None

class LinearLayer(nn.Linear):
    """
    支持事件驱动神经元的线性层
    """
    def __init__(self, net_cfg, layer_cfg, name, in_shape):
        in_features = layer_cfg['n_inputs']
        out_features = layer_cfg['n_outputs']

        self.layer_cfg = layer_cfg
        self.net_cfg = net_cfg
        self.name = name
        self.type = layer_cfg['type']
        self.in_shape = in_shape
        self.out_shape = [out_features, 1, 1]

        if not isinstance(in_features, int):
            raise Exception(f'inFeatures 应为 int，实际为: {in_features}')
        if not isinstance(out_features, int):
            raise Exception(f'outFeatures 应为 int，实际为: {out_features}')

        super().__init__(in_features, out_features, bias=False)

        weight_scale = layer_cfg.get('weight_scale', 1.0)
        nn.init.kaiming_normal_(self.weight)
        self.weight = nn.Parameter(
            weight_scale * self.weight.cuda(),
            requires_grad=True
        )

        print(
            f"Linear层: {self.name}, "
            f"输入形状: {self.in_shape}, "
            f"输出形状: {self.out_shape}, "
            f"权重形状: {list(self.weight.shape)}"
        )

    def forward(self, x):
        # [B, C, H, W, T] → [B, T, C*H*W]
        x = x.view(x.shape[0], -1, x.shape[4]).transpose(1, 2)
        y = f.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)
        return y.view(y.shape[0], y.shape[1], 1, 1, y.shape[2])

    def forward_pass(self, x, epoch, syn_kernel):
        y = self.forward(x)
        y = EventDrivenNeuron.apply(y, self.net_cfg, self.layer_cfg, syn_kernel)
        return y

    def get_parameters(self):
        return self.weight

    def weight_clipper(self):
        self.weight.data.clamp_(-4.0, 4.0)

class Network(nn.Module):
    """
    多层事件驱动 SNN
    """
    def __init__(self, net_cfg, layers_cfg, input_shape):
        super().__init__()
        self.layers = []
        self.net_cfg = net_cfg

        parameters = []
        print("网络结构:")

        for name, cfg in layers_cfg.items():
            if cfg['type'] == 'linear':
                layer = LinearLayer(net_cfg, cfg, name, input_shape)
                self.layers.append(layer)
                input_shape = layer.out_shape
                parameters.append(layer.get_parameters())
            else:
                raise Exception(f'未定义的层类型: {cfg["type"]}')

        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, spike_input, epoch, is_train, syn_kernel):
        # spike_input → PSP → 事件驱动神经元
        spikes = psp(spike_input, self.net_cfg)
        assert self.net_cfg['model'] == "LIF"

        for layer in self.layers:
            spikes = layer.forward_pass(spikes, epoch, syn_kernel)

        return spikes

    def get_parameters(self):
        return self.my_parameters

    def weight_clipper(self):
        for layer in self.layers:
            layer.weight_clipper()
