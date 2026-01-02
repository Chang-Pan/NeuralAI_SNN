import numpy as np
import torch
import matplotlib.pyplot as plt


class LearningStat:
    """
    单个阶段（training / testing）的统计信息
    """
    def __init__(self):
        # 累积量
        self.loss_sum = 0.0
        self.correct_samples = 0
        self.num_samples = 0

        # 历史最优
        self.min_loss = None
        self.max_accuracy = None

        # 日志
        self.loss_log = []
        self.accuracy_log = []

        # 是否刷新最优标志
        self.best_loss = False
        self.best_accuracy = False

    def reset(self):
        """在一个 epoch 结束后清空累计量"""
        self.loss_sum = 0.0
        self.correct_samples = 0
        self.num_samples = 0

    def loss(self):
        """平均 loss"""
        if self.num_samples > 0:
            return self.loss_sum / self.num_samples
        return None

    def accuracy(self):
        """分类准确率"""
        if self.num_samples > 0 and self.correct_samples > 0:
            return self.correct_samples / self.num_samples
        return None

    def update(self):
        """
        在 epoch 结束时调用：
        - 记录 loss / accuracy
        - 更新历史最优
        """
        current_loss = self.loss()
        self.loss_log.append(current_loss)

        if self.min_loss is None:
            self.min_loss = current_loss
        else:
            if current_loss < self.min_loss:
                self.min_loss = current_loss
                self.best_loss = True
            else:
                self.best_loss = False

        current_accuracy = self.accuracy()
        self.accuracy_log.append(current_accuracy)

        if self.max_accuracy is None:
            self.max_accuracy = current_accuracy
        else:
            if current_accuracy > self.max_accuracy:
                self.max_accuracy = current_accuracy
                self.best_accuracy = True
            else:
                self.best_accuracy = False

    def display_string(self):
        """
        控制台打印用字符串
        """
        loss = self.loss()
        accuracy = self.accuracy()

        if loss is None:
            return 'No testing results'

        if accuracy is None:
            if self.min_loss is None:
                return f'loss = {loss:<11.5g}'
            return f'loss = {loss:<11.5g} (min = {self.min_loss:<11.5g})'

        return (
            f'loss = {loss:<11.5g} (min = {self.min_loss:<11.5g})    '
            f'accuracy = {accuracy * 100:.2f}% (max = {self.max_accuracy * 100:.2f}%)'
        )


class LearningStats:
    """
    训练 + 测试 的整体统计管理
    """
    def __init__(self):
        self.lines_printed = 0
        self.training = LearningStat()
        self.testing = LearningStat()

    def update(self):
        """
        一个 epoch 结束时调用
        """
        self.training.update()
        self.training.reset()
        self.testing.update()
        self.testing.reset()

    def print(self, epoch, iteration=None, time_elapsed=None, header=None, footer=None):
        """
        动态刷新控制台输出
        """
        print('\033[%dA' % self.lines_printed)
        self.lines_printed = 1

        epoch_str = f'Epoch : {epoch:10d}'
        iter_str = '' if iteration is None else f'(i = {iteration:7d})'
        time_str = '' if time_elapsed is None else f', {time_elapsed:12.4f} s elapsed'

        if header is not None:
            for h in header:
                print('\033[2K' + str(h))
                self.lines_printed += 1

        print(epoch_str + iter_str + time_str)
        print(self.training.display_string())
        print(self.testing.display_string())
        self.lines_printed += 3

        if footer is not None:
            for f in footer:
                print('\033[2K' + str(f))
                self.lines_printed += 1

    def plot(self, figures=(1, 2), save_fig=False, path=''):
        """
        绘制 loss / accuracy 曲线
        """
        # Loss 曲线
        plt.figure(figures[0])
        plt.cla()
        if self.training.loss_log:
            plt.semilogy(self.training.loss_log, label='Training')
        if self.testing.loss_log:
            plt.semilogy(self.testing.loss_log, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        if save_fig:
            plt.savefig(path + 'loss.png')

        # Accuracy 曲线
        plt.figure(figures[1])
        plt.cla()
        if self.training.accuracy_log:
            plt.plot(self.training.accuracy_log, label='Training')
        if self.testing.accuracy_log:
            plt.plot(self.testing.accuracy_log, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        if save_fig:
            plt.savefig(path + 'accuracy.png')

    def save(self, filename=''):
        """
        保存 loss / accuracy 日志到文本
        """
        with open(filename + 'loss.txt', 'wt') as f:
            f.write('#%11s %11s\n' % ('Train', 'Test'))
            for i in range(len(self.training.loss_log)):
                f.write(
                    '%12.6g %12.6g\n'
                    % (self.training.loss_log[i], self.testing.loss_log[i])
                )

        with open(filename + 'accuracy.txt', 'wt') as f:
            f.write('#%11s %11s\n' % ('Train', 'Test'))
            for i in range(len(self.training.accuracy_log)):
                test_acc = (
                    self.testing.accuracy_log[i]
                    if self.testing.accuracy_log[i] is not None else 0
                )
                f.write(
                    '%12.6g %12.6g\n'
                    % (self.training.accuracy_log[i], test_acc)
                )

    def load(self, filename='', num_epoch=None, modulo=1):
        """
        从文本加载历史训练记录
        """
        accuracy = np.loadtxt(filename + 'accuracy.txt')
        loss = np.loadtxt(filename + 'loss.txt')

        if num_epoch is None:
            num_epoch = loss.shape[0] // modulo * modulo + 1

        self.training.loss_log = loss[:num_epoch, 0].tolist()
        self.testing.loss_log = loss[:num_epoch, 1].tolist()
        self.training.min_loss = loss[:num_epoch, 0].min()
        self.testing.min_loss = loss[:num_epoch, 1].min()

        self.training.accuracy_log = accuracy[:num_epoch, 0].tolist()
        self.testing.accuracy_log = accuracy[:num_epoch, 1].tolist()
        self.training.max_accuracy = accuracy[:num_epoch, 0].max()
        self.testing.max_accuracy = accuracy[:num_epoch, 1].max()

        return num_epoch

class EarlyStopping:
    """
    基于验证指标的 Early Stopping 机制

    - 当验证指标在 patience 个 epoch 内没有显著提升时，触发 early stop
    - 每当指标刷新最优时，自动保存模型 checkpoint
    """
    def __init__(self, patience=50, verbose=False, delta=0):
        # 允许连续多少个 epoch 无提升
        self.patience = patience

        # 是否打印提示信息
        self.verbose = verbose

        # 当前连续未提升的 epoch 计数
        self.counter = 0

        # 记录历史最优指标
        self.best_score = None

        # 是否触发 early stop
        self.early_stop = False

        # 记录历史最小（或最优）验证值，用于打印
        self.val_min = np.inf

        # 判定“提升”的最小阈值
        self.delta = delta

    def __call__(self, val, model, epoch):
        """
        每个 epoch 调用一次

        val   : 当前验证指标
        model : 当前模型
        epoch : 当前 epoch
        """
        current_score = val

        # 第一次调用，直接记录并保存
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model, val, epoch)
            return

        # 若当前指标没有达到“显著提升”
        if current_score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(
                    f'EarlyStopping counter: '
                    f'{self.counter} out of {self.patience}'
                )

            # 超过耐心阈值，触发 early stop
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 指标提升，更新最优并保存模型
            self.best_score = current_score
            self.save_checkpoint(model, val, epoch)
            self.counter = 0

    def save_checkpoint(self, network, val, epoch):
        """
        保存当前最优模型
        """
        if self.verbose:
            print(
                f'Validation metric improved '
                f'({self.val_min:.6f} --> {val:.6f}). '
                f'Saving model ...'
            )

        checkpoint = {
            'net': network.state_dict(),
            'loss': val,
            'epoch': epoch,
        }

        import os
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(checkpoint, './checkpoint/ckpt.pth')

        # 更新历史最优记录
        self.val_min = val
