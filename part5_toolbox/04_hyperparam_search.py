"""
自动生成自: part5_toolbox\04_hyperparam_search.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────
# LR Finder：找到最优学习率范围
# 原理：从极小 lr 开始指数增大，记录 loss 变化
# 最优 lr ≈ loss 下降最快的点（斜率最大处）
# ─────────────────────────────────────────────────────────

class LRFinder:
    """
    学习率范围测试

    使用方法：
        finder = LRFinder(model, optimizer, criterion)
        finder.range_test(dataloader, start_lr=1e-7, end_lr=10, num_iter=100)
        finder.plot()
        best_lr = finder.suggest_lr()
    """

    def __init__(self, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # 保存初始状态，测试后可恢复
        self._model_state = deepcopy(model.state_dict())
        self._optimizer_state = deepcopy(optimizer.state_dict())

        self.lrs = []
        self.losses = []

    def range_test(self, dataloader,
                   start_lr: float = 1e-7,
                   end_lr: float = 10.0,
                   num_iter: int = 100,
                   smooth_f: float = 0.05,
                   diverge_th: float = 5.0):
        """
        执行 LR 范围测试

        smooth_f: 损失平滑系数（指数移动平均）
        diverge_th: 当 loss > best_loss * diverge_th 时停止
        """
        # 设置初始 lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = start_lr

        # 指数增长因子
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        best_loss = float('inf')
        avg_loss = 0.0
        self.lrs = []
        self.losses = []

        self.model.train()
        data_iter = iter(dataloader)

        for i in range(num_iter):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            # 指数移动平均平滑 loss
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            smoothed = avg_loss / (1 - (1 - smooth_f) ** (i + 1))

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            self.losses.append(smoothed)

            if smoothed < best_loss:
                best_loss = smoothed

            # 发散则停止
            if smoothed > diverge_th * best_loss:
                print(f"Loss 发散，在 lr={current_lr:.2e} 处停止")
                break

            # 更新 lr
            for pg in self.optimizer.param_groups:
                pg['lr'] *= lr_mult

        # 恢复初始状态
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)
        print(f"LR 范围测试完成，共 {len(self.lrs)} 步")

    def plot(self, skip_start: int = 5, skip_end: int = 5, figsize=(10, 4)):
        """绘制 LR vs Loss 曲线"""
        if not self.lrs:
            print("请先运行 range_test()")
            return

        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 线性 x 轴
        axes[0].plot(lrs, losses, 'b-', linewidth=1.5)
        axes[0].set_xlabel('学习率')
        axes[0].set_ylabel('Loss（平滑）')
        axes[0].set_title('LR Finder（线性 x 轴）', fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 对数 x 轴（更常用）
        axes[1].plot(lrs, losses, 'b-', linewidth=1.5)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('学习率（log scale）')
        axes[1].set_ylabel('Loss（平滑）')
        axes[1].set_title('LR Finder（log x 轴）\n选 loss 下降最快的点', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # 标注建议 lr
        best_lr = self.suggest_lr(skip_start, skip_end)
        if best_lr:
            for ax in axes:
                ax.axvline(best_lr, color='red', linestyle='--',
                           label=f'建议 lr={best_lr:.2e}')
                ax.legend(fontsize=9)

        plt.suptitle('学习率范围测试（LR Finder）', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lr_finder.png', dpi=150, bbox_inches='tight')
        plt.show()

    def suggest_lr(self, skip_start: int = 5, skip_end: int = 5) -> Optional[float]:
        """返回建议的学习率（loss 下降最快的点）"""
        if len(self.lrs) < skip_start + skip_end + 2:
            return None

        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        # 计算梯度（loss 对 lr 的变化率）
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        best_lr = lrs[min_grad_idx]
        print(f"建议学习率: {best_lr:.2e}（loss 下降最快处）")
        return best_lr


# ─────────────────────────────────────────────────────────
# 学习率调度策略对比
# ─────────────────────────────────────────────────────────

def compare_lr_schedules(base_lr: float = 0.1, n_epochs: int = 100,
                          steps_per_epoch: int = 50):
    """
    可视化并对比常用 LR 调度策略
    """
    total_steps = n_epochs * steps_per_epoch

    def get_lr_curve(scheduler_fn):
        """模拟 lr 变化曲线"""
        model = nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=base_lr)
        scheduler = scheduler_fn(opt)
        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        return lrs

    schedules = {
        'StepLR (每30epoch×0.1)': lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=30 * steps_per_epoch, gamma=0.1),
        'CosineAnnealing': lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps),
        'OneCycleLR': lambda opt: torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=base_lr, total_steps=total_steps),
        'ExponentialLR (γ=0.99)': lambda opt: torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=0.99 ** (1 / steps_per_epoch)),
        'CyclicLR': lambda opt: torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=base_lr * 0.01, max_lr=base_lr,
            step_size_up=5 * steps_per_epoch, mode='triangular2'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (name, sched_fn) in enumerate(schedules.items()):
        lrs = get_lr_curve(sched_fn)
        epochs = np.linspace(0, n_epochs, total_steps)
        axes[i].plot(epochs, lrs, 'b-', linewidth=1.5)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('学习率')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(bottom=0)

    # 最后一个子图：所有策略对比
    ax = axes[-1]
    for name, sched_fn in schedules.items():
        lrs = get_lr_curve(sched_fn)
        epochs = np.linspace(0, n_epochs, total_steps)
        ax.plot(epochs, lrs, linewidth=1.2, label=name[:20], alpha=0.8)
    ax.set_title('所有策略对比', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('学习率')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'学习率调度策略对比（base_lr={base_lr}）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lr_schedules.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 超参敏感性分析
# ─────────────────────────────────────────────────────────

def hyperparam_sensitivity_analysis(results: list, metric: str = 'val_acc'):
    """
    分析哪个超参数对结果影响最大

    results: HyperparamSearch.results 格式的列表
             [{'params': {...}, 'score': float}, ...]
    """
    import itertools

    valid = [r for r in results if r.get('score', float('-inf')) != float('-inf')]
    if not valid:
        print("没有有效结果")
        return

    param_keys = list(valid[0]['params'].keys())
    scores = np.array([r['score'] for r in valid])

    print(f"\n超参敏感性分析（共 {len(valid)} 个试验）")
    print("=" * 50)

    sensitivities = {}
    for key in param_keys:
        values = [r['params'][key] for r in valid]
        unique_vals = sorted(set(values))

        if len(unique_vals) < 2:
            continue

        # 计算每个参数值对应的平均分数
        group_means = []
        for v in unique_vals:
            group_scores = [r['score'] for r in valid if r['params'][key] == v]
            group_means.append(np.mean(group_scores))

        # 敏感性 = 最大组均值 - 最小组均值
        sensitivity = max(group_means) - min(group_means)
        sensitivities[key] = sensitivity
        print(f"  {key:20s}: 敏感性={sensitivity:.4f}  (各值均值: {dict(zip(unique_vals, [f'{m:.3f}' for m in group_means]))})")

    # 可视化
    if sensitivities:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 敏感性条形图
        keys = list(sensitivities.keys())
        vals = [sensitivities[k] for k in keys]
        colors = plt.cm.RdYlGn(np.array(vals) / max(vals))
        axes[0].barh(keys, vals, color=colors)
        axes[0].set_xlabel('敏感性（最大组均值 - 最小组均值）')
        axes[0].set_title(f'超参数敏感性排名\n（{metric}）', fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 最重要超参数的详细分布
        most_important = max(sensitivities, key=sensitivities.get)
        values = [r['params'][most_important] for r in valid]
        axes[1].scatter(values, scores, alpha=0.6, s=60, c=scores, cmap='RdYlGn')
        axes[1].set_xlabel(most_important)
        axes[1].set_ylabel(metric)
        axes[1].set_title(f'最重要超参数: {most_important}', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('超参数敏感性分析', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('hyperparam_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.show()

    return sensitivities


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────


def print_learning_guide():
    print("""
学习导读：超参搜索的目标不是碰运气找到最高分，而是缩小“不稳定、欠拟合、过拟合”的可能范围。

1. LR Finder 怎么看
   - 横轴最好看 log scale，因为学习率通常跨越多个数量级。
   - 选择 loss 下降最快点附近，但不要选已经开始反弹或发散的点。
   - 如果曲线从头到尾不下降，先查数据、标签和 loss；如果突然爆炸，学习率上界太高。

2. 学习率调度图怎么看
   - StepLR 是阶梯式下降，适合有明确阶段的传统训练。
   - CosineAnnealing 平滑退火，适合通用默认策略。
   - OneCycle/Cyclic 会先升后降，常用于快速探索，但对峰值学习率更敏感。

3. 敏感性分析怎么看
   - 条形图越长，说明这个超参数改变后验证分数波动越大。
   - 散点图看的是参数值和指标的关系，不要只盯最高点，还要看最高点周围是否稳定。
   - 如果某个“最优点”周围全是低分，它可能只是随机噪声。

工程经验：
   学习率优先用对数尺度粗搜，例如 1e-4、3e-4、1e-3、3e-3；dropout 先看 0.0 到 0.5；
   hidden size 先试 2 到 3 个量级点。真实项目里不要用测试集调参，测试集只用于最终报告。

真实踩坑：
   我见过一次搜索把验证集和测试集混用，最后线上准确率比离线低很多。排查时发现“最好参数”其实是对测试集噪声过拟合。
   正确流程是训练集训练、验证集选择、测试集最后一次评估，并保存每次失败配置。

进阶思考：
   随机搜索为什么在高维空间里常比网格搜索更划算？如果最高分配置训练成本高 3 倍，但只提升 0.1%，你会怎么取舍？
""".strip())


def demo_hyperparam_tools():
    print_learning_guide()
    torch.manual_seed(42)

    # 1. LR 调度策略对比
    print("1. 学习率调度策略对比")
    compare_lr_schedules(base_lr=0.1, n_epochs=100, steps_per_epoch=50)

    # 2. 模拟超参搜索结果，做敏感性分析
    print("\n2. 超参敏感性分析（模拟数据）")
    import random
    random.seed(42)

    # 模拟一组搜索结果
    mock_results = []
    for lr in [0.001, 0.01, 0.1]:
        for hidden in [32, 64, 128]:
            for dropout in [0.0, 0.3, 0.5]:
                # 模拟：lr=0.01 最好，hidden=64 最好，dropout 影响小
                score = (
                    0.8
                    - abs(np.log10(lr) - np.log10(0.01)) * 0.1
                    - abs(hidden - 64) / 200
                    - dropout * 0.05
                    + random.gauss(0, 0.02)
                )
                mock_results.append({
                    'params': {'lr': lr, 'hidden_size': hidden, 'dropout': dropout},
                    'score': score,
                    'metrics': {'val_acc': score},
                })

    sensitivities = hyperparam_sensitivity_analysis(mock_results, metric='val_acc')

    return sensitivities

sensitivities = demo_hyperparam_tools()
