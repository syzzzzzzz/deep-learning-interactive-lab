# 第九章：调试工具箱 — 梯度监控 + 训练动态 + 超参搜索

---

## 9.1 梯度监控工具（生产级）

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import Dict, List, Optional
import warnings

# ─────────────────────────────────────────────────────────
# 梯度监控器：实时追踪每层梯度健康状况
# ─────────────────────────────────────────────────────────

class GradientMonitor:
    """
    生产级梯度监控工具

    功能：
    1. 实时追踪每层梯度的均值、标准差、最大值
    2. 自动检测梯度消失/爆炸
    3. 绘制梯度流动图
    4. 生成诊断报告

    使用方法：
        monitor = GradientMonitor(model)
        for epoch in range(epochs):
            loss.backward()
            monitor.record()   # 每次 backward 后调用
            optimizer.step()
        monitor.plot()
        monitor.report()
    """

    VANISHING_THRESHOLD = 1e-6
    EXPLODING_THRESHOLD = 100.0

    def __init__(self, model: nn.Module, watch_layers: Optional[List[str]] = None):
        self.model = model
        self.watch_layers = watch_layers
        self.history: Dict[str, List[dict]] = defaultdict(list)
        self.step = 0

    def record(self):
        """在 backward() 之后调用，记录当前梯度状态"""
        self.step += 1
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if self.watch_layers and not any(w in name for w in self.watch_layers):
                continue

            grad = param.grad.detach()
            self.history[name].append({
                'step': self.step,
                'mean': grad.abs().mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'norm': grad.norm().item(),
            })

    def plot(self, metric='mean', last_n_steps: int = None, figsize=(16, 10)):
        """
        可视化梯度历史

        metric: 'mean' | 'std' | 'max' | 'norm'
        last_n_steps: 只显示最近 N 步
        """
        if not self.history:
            print("没有梯度数据，请先调用 record()")
            return

        names = list(self.history.keys())
        n_layers = len(names)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # 1. 梯度随训练步数的变化（折线图）
        ax1 = fig.add_subplot(gs[0, :])
        for name in names[:8]:  # 最多显示8层
            data = self.history[name]
            if last_n_steps:
                data = data[-last_n_steps:]
            steps = [d['step'] for d in data]
            values = [d[metric] for d in data]
            ax1.plot(steps, values, label=name.replace('.weight', '').replace('.bias', ''),
                     alpha=0.8, linewidth=1.5)

        ax1.axhline(y=self.VANISHING_THRESHOLD, color='blue', linestyle='--',
                    alpha=0.5, label=f'消失阈值 ({self.VANISHING_THRESHOLD})')
        ax1.axhline(y=self.EXPLODING_THRESHOLD, color='red', linestyle='--',
                    alpha=0.5, label=f'爆炸阈值 ({self.EXPLODING_THRESHOLD})')
        ax1.set_title(f'梯度 {metric} 随训练步数的变化', fontsize=12)
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel(f'梯度 {metric}')
        ax1.set_yscale('log')
        ax1.legend(fontsize=7, loc='upper right', ncol=2)
        ax1.grid(True, alpha=0.3)

        # 2. 最终梯度大小（条形图，按层排序）
        ax2 = fig.add_subplot(gs[1, 0])
        final_values = []
        layer_names = []
        for name in names:
            if self.history[name]:
                final_values.append(self.history[name][-1][metric])
                layer_names.append(name.replace('.weight', '').replace('.bias', ''))

        colors = ['red' if v < self.VANISHING_THRESHOLD else
                  'orange' if v > self.EXPLODING_THRESHOLD else
                  'steelblue' for v in final_values]

        bars = ax2.barh(range(len(layer_names)), final_values, color=colors)
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels(layer_names, fontsize=8)
        ax2.set_xlabel(f'梯度 {metric}')
        ax2.set_title('各层最终梯度大小', fontsize=11)
        ax2.axvline(x=self.VANISHING_THRESHOLD, color='blue', linestyle='--', alpha=0.5)
        ax2.axvline(x=self.EXPLODING_THRESHOLD, color='red', linestyle='--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        # 3. 梯度热力图（层 × 时间步）
        ax3 = fig.add_subplot(gs[1, 1])
        n_show = min(10, n_layers)
        n_steps_show = min(50, self.step)
        heatmap_data = np.zeros((n_show, n_steps_show))

        for i, name in enumerate(names[:n_show]):
            data = self.history[name][-n_steps_show:]
            for j, d in enumerate(data):
                heatmap_data[i, j] = np.log10(d[metric] + 1e-10)

        im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                        vmin=-8, vmax=2)
        ax3.set_yticks(range(n_show))
        ax3.set_yticklabels([n.replace('.weight', '') for n in names[:n_show]], fontsize=8)
        ax3.set_xlabel('训练步数（最近）')
        ax3.set_title('梯度热力图（log10 scale）\n绿=正常，红=消失，黄=爆炸', fontsize=10)
        plt.colorbar(im, ax=ax3, label='log10(梯度)')

        plt.suptitle('梯度监控面板', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('gradient_monitor.png', dpi=150, bbox_inches='tight')
        plt.show()

    def report(self) -> dict:
        """生成梯度健康诊断报告"""
        print("\n" + "=" * 65)
        print("梯度健康诊断报告")
        print("=" * 65)
        print(f"{'层名':35s} {'均值':10s} {'最大值':10s} {'状态':10s}")
        print("-" * 65)

        issues = {'vanishing': [], 'exploding': [], 'healthy': []}

        for name, data in self.history.items():
            if not data:
                continue
            recent = data[-min(10, len(data)):]
            avg_mean = np.mean([d['mean'] for d in recent])
            avg_max = np.max([d['max'] for d in recent])

            if avg_mean < self.VANISHING_THRESHOLD:
                status = "⚠️  梯度消失"
                issues['vanishing'].append(name)
            elif avg_max > self.EXPLODING_THRESHOLD:
                status = "🔥 梯度爆炸"
                issues['exploding'].append(name)
            else:
                status = "✅ 正常"
                issues['healthy'].append(name)

            short_name = name[:33] + '..' if len(name) > 35 else name
            print(f"{short_name:35s} {avg_mean:10.2e} {avg_max:10.2e} {status}")

        print("\n总结：")
        print(f"  正常层: {len(issues['healthy'])}")
        print(f"  梯度消失: {len(issues['vanishing'])}")
        print(f"  梯度爆炸: {len(issues['exploding'])}")

        if issues['vanishing']:
            print("\n建议（梯度消失）：")
            print("  → 使用 ReLU/GELU 替代 Sigmoid/Tanh")
            print("  → 添加残差连接（ResNet 风格）")
            print("  → 使用 He 初始化")
            print("  → 添加 BatchNorm/LayerNorm")

        if issues['exploding']:
            print("\n建议（梯度爆炸）：")
            print("  → 使用梯度裁剪: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
            print("  → 降低学习率")
            print("  → 使用 Xavier/He 初始化")

        return issues

    def remove_hooks(self):
        pass  # 本实现使用 record() 手动调用，无需移除钩子


# ─────────────────────────────────────────────────────────
# 训练动态可视化
# ─────────────────────────────────────────────────────────

class TrainingDynamicsVisualizer:
    """
    实时追踪并可视化训练动态：
    - 损失曲线（训练/验证）
    - 准确率曲线
    - 学习率变化
    - 权重分布变化
    - 激活值分布
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.history = defaultdict(list)
        self.weight_snapshots = {}  # 每隔一段时间保存权重分布
        self.activation_snapshots = {}

    def log(self, **kwargs):
        """记录任意指标"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.history[key].append(value)

    def snapshot_weights(self, step: int):
        """保存当前权重分布快照"""
        self.weight_snapshots[step] = {}
        for name, param in self.model.named_parameters():
            self.weight_snapshots[step][name] = param.detach().cpu().numpy().flatten()

    def plot_training_curves(self, figsize=(15, 5)):
        """绘制训练曲线"""
        metrics = list(self.history.keys())
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("没有记录的指标")
            return

        fig, axes = plt.subplots(1, min(n_metrics, 4), figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 4:
            axes = axes[:4]

        for ax, metric in zip(axes, metrics[:4]):
            values = self.history[metric]
            steps = range(len(values))
            ax.plot(steps, values, 'b-', linewidth=1.5, alpha=0.8)

            # 平滑曲线
            if len(values) > 10:
                window = max(1, len(values) // 20)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(values)), smoothed, 'r-',
                        linewidth=2, label='平滑')

            ax.set_title(metric, fontsize=11)
            ax.set_xlabel('步数')
            ax.grid(True, alpha=0.3)
            if 'loss' in metric.lower():
                ax.set_yscale('log')
            if 'acc' in metric.lower():
                ax.set_ylim(0, 1.05)

        plt.suptitle('训练动态', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_dynamics.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_weight_evolution(self, layer_name: str = None):
        """可视化权重分布随训练的变化"""
        if not self.weight_snapshots:
            print("没有权重快照，请先调用 snapshot_weights()")
            return

        steps = sorted(self.weight_snapshots.keys())

        # 选择要显示的层
        if layer_name is None:
            all_names = list(self.weight_snapshots[steps[0]].keys())
            layer_name = [n for n in all_names if 'weight' in n][0]

        fig, axes = plt.subplots(1, min(len(steps), 5), figsize=(15, 4))
        if len(steps) == 1:
            axes = [axes]

        for ax, step in zip(axes, steps[:5]):
            weights = self.weight_snapshots[step].get(layer_name, np.array([]))
            if len(weights) == 0:
                continue
            ax.hist(weights, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
            ax.set_title(f'Step {step}\nmean={weights.mean():.3f}\nstd={weights.std():.3f}',
                         fontsize=9)
            ax.set_xlabel('权重值')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'权重分布演化: {layer_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('weight_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────
# 超参搜索框架
# ─────────────────────────────────────────────────────────

class HyperparamSearch:
    """
    轻量级超参搜索框架

    支持：
    - 网格搜索（Grid Search）
    - 随机搜索（Random Search）
    - 结果可视化
    - 最优参数报告

    使用方法：
        searcher = HyperparamSearch(train_fn)
        searcher.grid_search({
            'lr': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
        })
        searcher.plot_results()
        best = searcher.best_params()
    """

    def __init__(self, train_fn, metric='val_acc', maximize=True):
        """
        train_fn: 接受超参数字典，返回指标字典的函数
        metric: 优化目标指标名
        maximize: True=最大化指标，False=最小化
        """
        self.train_fn = train_fn
        self.metric = metric
        self.maximize = maximize
        self.results = []

    def grid_search(self, param_grid: dict, verbose=True):
        """网格搜索：遍历所有参数组合"""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        print(f"网格搜索：共 {len(combinations)} 个参数组合")
        print("=" * 60)

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            if verbose:
                print(f"\n[{i+1}/{len(combinations)}] 参数: {params}")

            try:
                metrics = self.train_fn(params)
                result = {'params': params, 'metrics': metrics,
                          'score': metrics.get(self.metric, float('-inf'))}
                self.results.append(result)

                if verbose:
                    print(f"  结果: {metrics}")
            except Exception as e:
                print(f"  失败: {e}")
                self.results.append({'params': params, 'metrics': {}, 'score': float('-inf')})

        print(f"\n搜索完成！共 {len(self.results)} 个结果")
        return self

    def random_search(self, param_distributions: dict, n_trials=20, verbose=True):
        """随机搜索：从分布中随机采样参数"""
        print(f"随机搜索：共 {n_trials} 次试验")
        print("=" * 60)

        for i in range(n_trials):
            params = {}
            for key, dist in param_distributions.items():
                if isinstance(dist, list):
                    params[key] = np.random.choice(dist)
                elif isinstance(dist, tuple) and len(dist) == 2:
                    lo, hi = dist
                    if isinstance(lo, float) or isinstance(hi, float):
                        params[key] = np.random.uniform(lo, hi)
                    else:
                        params[key] = np.random.randint(lo, hi + 1)
                elif callable(dist):
                    params[key] = dist()

            if verbose:
                print(f"\n[{i+1}/{n_trials}] 参数: {params}")

            try:
                metrics = self.train_fn(params)
                result = {'params': params, 'metrics': metrics,
                          'score': metrics.get(self.metric, float('-inf'))}
                self.results.append(result)
                if verbose:
                    print(f"  结果: {metrics}")
            except Exception as e:
                print(f"  失败: {e}")

        return self

    def best_params(self) -> dict:
        """返回最优参数"""
        if not self.results:
            return {}
        valid = [r for r in self.results if r['score'] != float('-inf')]
        if not valid:
            return {}
        best = max(valid, key=lambda r: r['score']) if self.maximize else \
               min(valid, key=lambda r: r['score'])
        print(f"\n最优参数: {best['params']}")
        print(f"最优指标 ({self.metric}): {best['score']:.4f}")
        return best['params']

    def plot_results(self, top_k=10):
        """可视化搜索结果"""
        if not self.results:
            print("没有搜索结果")
            return

        valid = [r for r in self.results if r['score'] != float('-inf')]
        valid.sort(key=lambda r: r['score'], reverse=self.maximize)

        scores = [r['score'] for r in valid]
        param_keys = list(valid[0]['params'].keys()) if valid else []

        n_params = len(param_keys)
        fig, axes = plt.subplots(1, min(n_params + 1, 5), figsize=(min(n_params + 1, 5) * 4, 4))
        if not hasattr(axes, '__len__'):
            axes = [axes]

        # 总体分数分布
        axes[0].hist(scores, bins=min(20, len(scores)), color='steelblue',
                     alpha=0.7, edgecolor='white')
        axes[0].axvline(x=max(scores) if self.maximize else min(scores),
                        color='red', linestyle='--', label='最优')
        axes[0].set_title(f'{self.metric} 分布', fontsize=11)
        axes[0].set_xlabel(self.metric)
        axes[0].set_ylabel('频次')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 每个超参数 vs 分数
        for ax, key in zip(axes[1:], param_keys[:4]):
            param_values = [r['params'][key] for r in valid]
            ax.scatter(param_values, scores, alpha=0.6, s=50)
            ax.set_title(f'{key} vs {self.metric}', fontsize=11)
            ax.set_xlabel(key)
            ax.set_ylabel(self.metric)
            ax.grid(True, alpha=0.3)

            # 如果是数值型，画趋势线
            try:
                pv = np.array(param_values, dtype=float)
                sv = np.array(scores)
                z = np.polyfit(pv, sv, 1)
                p = np.poly1d(z)
                x_line = np.linspace(pv.min(), pv.max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='趋势')
                ax.legend(fontsize=8)
            except Exception:
                pass

        plt.suptitle(f'超参搜索结果（共 {len(valid)} 次试验）',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('hyperparam_search.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 打印 Top-K 结果
        print(f"\nTop-{min(top_k, len(valid))} 结果：")
        print("-" * 60)
        for i, r in enumerate(valid[:top_k]):
            print(f"  #{i+1}: {self.metric}={r['score']:.4f}  params={r['params']}")


# ─────────────────────────────────────────────────────────
# 完整使用示例
# ─────────────────────────────────────────────────────────

def demo_all_tools():
    """演示所有调试工具的使用"""
    torch.manual_seed(42)

    # 构建一个简单模型
    model = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid(),
    )

    # 生成数据
    X = torch.randn(200, 20)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    # 初始化工具
    grad_monitor = GradientMonitor(model)
    dynamics = TrainingDynamicsVisualizer(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # 训练循环
    for step in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        acc = ((pred > 0.5) == y).float().mean().item()

        optimizer.zero_grad()
        loss.backward()

        # 记录梯度
        grad_monitor.record()

        optimizer.step()

        # 记录训练动态
        dynamics.log(loss=loss.item(), accuracy=acc)

        # 每20步保存权重快照
        if step % 20 == 0:
            dynamics.snapshot_weights(step)

    # 可视化
    print("绘制梯度监控面板...")
    grad_monitor.plot()
    grad_monitor.report()

    print("\n绘制训练动态...")
    dynamics.plot_training_curves()
    dynamics.plot_weight_evolution()

    # 超参搜索示例
    print("\n运行超参搜索...")

    def quick_train(params):
        """快速训练函数，用于超参搜索"""
        torch.manual_seed(42)
        m = nn.Sequential(
            nn.Linear(20, params.get('hidden_size', 64)), nn.ReLU(),
            nn.Linear(params.get('hidden_size', 64), 1), nn.Sigmoid(),
        )
        opt = torch.optim.Adam(m.parameters(), lr=params.get('lr', 0.01))
        for _ in range(50):
            p = m(X)
            l = criterion(p, y)
            opt.zero_grad()
            l.backward()
            opt.step()
        with torch.no_grad():
            final_acc = ((m(X) > 0.5) == y).float().mean().item()
        return {'val_acc': final_acc, 'final_loss': l.item()}

    searcher = HyperparamSearch(quick_train, metric='val_acc', maximize=True)
    searcher.grid_search({
        'lr': [0.001, 0.01, 0.1],
        'hidden_size': [32, 64, 128],
    })
    searcher.plot_results()
    best = searcher.best_params()

    return grad_monitor, dynamics, searcher

grad_monitor, dynamics, searcher = demo_all_tools()
```

---

## 小结

| 工具 | 用途 | 关键方法 |
|------|------|----------|
| GradientMonitor | 检测梯度消失/爆炸 | `record()`, `plot()`, `report()` |
| TrainingDynamicsVisualizer | 追踪训练过程 | `log()`, `snapshot_weights()`, `plot_training_curves()` |
| HyperparamSearch | 自动搜索最优超参 | `grid_search()`, `random_search()`, `best_params()` |
