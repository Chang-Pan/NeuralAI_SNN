## 2.surrogates

`my_surrogates.py`: 实现三种常见的脉冲神经网络(SNN)替代梯度函数，包括SuperSpike、Sigmoid导数和Esser等方法。

`run_cifar10_surrogates_251129_by_jxq.ipynb`: 在CIFAR-10数据集上测试不同替代梯度函数对SNN模型训练性能的影响。

`run_cifar10`: 上者的脚本化

`artifacts/`: 内有热力图, 表格数据

最终结果汇总
模型: SuperSpike | 最快基线 lr=2e-03, alpha=7.07, epoch=5 | best_acc=72.45%
模型: Sigmoid    | 最快基线 lr=2e-03, alpha=3.49, epoch=6 | best_acc=72.26%
模型: Esser      | 最快基线 lr=2e-03, alpha=0.847, epoch=6 | best_acc=72.94%
