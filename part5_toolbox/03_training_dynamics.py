try:
    """
    自动生成自: part5_toolbox\03_training_dynamics.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from collections import defaultdict
    from typing import Dict, List, Optional

    # ─────────────────────────────────────────────────────────
    # 分布追踪器：记录权重/激活的统计量随训练的变化
    # ─────────────────────────────────────────────────────────

    class DistributionTracker:
        """
        追踪权重和激活值的分布随训练步数的变化

        使用方法：
            tracker = DistributionTracker(model)
            for step, (x, y) in enumerate(dataloader):
                loss = criterion(model(x), y)
                loss.backward()
                if step % 10 == 0:
                    tracker.record_weights(step)
                optimizer.step()
            tracker.plot_weight_evolution()
            tracker.plot_activation_saturation()
        """

        def __init__(self, model: nn.Module):
            self.model = model
            self.weight_history: Dict[str, List[dict]] = defaultdict(list)
            self.activation_history: Dict[str, List[dict]] = defaultdict(list)
            self._hooks = []

        def record_weights(self, step: int):
            """记录当前所有参数的分布统计量"""
            for name, param in self.model.named_parameters():
                if 'weight' not in name:
                    continue
                w = param.detach().cpu().numpy().flatten()
                self.weight_history[name].append({
                    'step': step,
                    'mean': float(w.mean()),
                    'std': float(w.std()),
                    'abs_mean': float(np.abs(w).mean()),
                    'p5': float(np.percentile(w, 5)),
                    'p95': float(np.percentile(w, 95)),
                })

        def register_activation_hooks(self, layer_types=(nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
            """注册钩子，自动记录激活层的输出分布"""
            def make_hook(name):
                def hook(module, inp, out):
                    a = out.detach().cpu().numpy().flatten()
                    # 饱和率：对于 ReLU 是死亡神经元比例；对于 Tanh/Sigmoid 是接近饱和的比例
                    if isinstance(module, nn.ReLU):
                        saturation = float((a == 0).mean())
                    elif isinstance(module, (nn.Tanh, nn.Sigmoid)):
                        saturation = float((np.abs(a) > 0.9).mean())
                    else:
                        saturation = 0.0

                    self.activation_history[name].append({
                        'mean': float(a.mean()),
                        'std': float(a.std()),
                        'saturation': saturation,
                        'abs_mean': float(np.abs(a).mean()),
                    })
                return hook

            for name, module in self.model.named_modules():
                if isinstance(module, layer_types):
                    h = module.register_forward_hook(make_hook(name))
                    self._hooks.append(h)

        def remove_hooks(self):
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

        def plot_weight_evolution(self, figsize=(16, 10)):
            """可视化权重分布随训练的演化"""
            if not self.weight_history:
                print("没有权重记录，请先调用 record_weights()")
                return

            names = list(self.weight_history.keys())
            n_layers = min(6, len(names))

            fig, axes = plt.subplots(2, n_layers, figsize=figsize)
            if n_layers == 1:
                axes = axes.reshape(2, 1)

            for col, name in enumerate(names[:n_layers]):
                data = self.weight_history[name]
                steps = [d['step'] for d in data]
                means = [d['mean'] for d in data]
                stds = [d['std'] for d in data]
                p5 = [d['p5'] for d in data]
                p95 = [d['p95'] for d in data]

                # 上行：均值 ± 标准差
                ax = axes[0, col]
                ax.plot(steps, means, 'b-', linewidth=1.5, label='均值')
                ax.fill_between(steps,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.3, color='blue', label='±std')
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                short = name.replace('.weight', '')[:20]
                ax.set_title(short, fontsize=9)
                ax.set_xlabel('步数')
                if col == 0:
                    ax.set_ylabel('权重值')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

                # 下行：5th-95th 百分位范围
                ax2 = axes[1, col]
                ax2.fill_between(steps, p5, p95, alpha=0.4, color='green')
                ax2.plot(steps, p5, 'g--', linewidth=1, alpha=0.7, label='P5')
                ax2.plot(steps, p95, 'g-', linewidth=1, alpha=0.7, label='P95')
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_xlabel('步数')
                if col == 0:
                    ax2.set_ylabel('权重百分位')
                ax2.legend(fontsize=7)
                ax2.grid(True, alpha=0.3)

            plt.suptitle('权重分布演化（均值/标准差/百分位）', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('weight_distribution_evolution.png', dpi=150, bbox_inches='tight')
            plt.show()

        def plot_activation_saturation(self, figsize=(14, 5)):
            """可视化激活饱和率随训练的变化"""
            if not self.activation_history:
                print("没有激活记录，请先注册钩子并运行前向传播")
                return

            names = list(self.activation_history.keys())
            n = len(names)

            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # 左图：各层饱和率
            for name in names:
                data = self.activation_history[name]
                steps = range(len(data))
                sat = [d['saturation'] for d in data]
                axes[0].plot(steps, sat, label=name[:20], alpha=0.8, linewidth=1.5)

            axes[0].set_title('激活饱和率随训练步数的变化\n(ReLU=死亡比例, Tanh/Sigmoid=|a|>0.9比例)',
                              fontsize=10)
            axes[0].set_xlabel('前向传播次数')
            axes[0].set_ylabel('饱和率')
            axes[0].set_ylim(0, 1)
            axes[0].legend(fontsize=7, loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%警戒线')

            # 右图：最终各层饱和率对比
            final_sat = [self.activation_history[n][-1]['saturation'] for n in names]
            colors = ['red' if s > 0.5 else 'orange' if s > 0.2 else 'steelblue'
                      for s in final_sat]
            axes[1].barh(range(len(names)), final_sat, color=colors)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels([n[:20] for n in names], fontsize=8)
            axes[1].set_xlabel('最终饱和率')
            axes[1].set_title('各层最终激活饱和率\n红=严重(>50%), 橙=警告(>20%)', fontsize=10)
            axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5)
            axes[1].axvline(0.2, color='orange', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3)

            plt.suptitle('激活饱和分析', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('activation_saturation.png', dpi=150, bbox_inches='tight')
            plt.show()


    # ─────────────────────────────────────────────────────────
    # 学习率 vs 梯度比值分析（更新幅度监控）
    # ─────────────────────────────────────────────────────────

    class UpdateRatioMonitor:
        """
        监控每层的"更新幅度比"：lr * grad_norm / weight_norm

        理想范围：~1e-3（太大=不稳定，太小=学习太慢）
        参考：Andrej Karpathy 的建议值约为 1e-3
        """

        def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
            self.model = model
            self.optimizer = optimizer
            self.history: Dict[str, List[float]] = defaultdict(list)
            self.step = 0

        def record(self):
            """在 optimizer.step() 之前调用"""
            self.step += 1
            lr = self.optimizer.param_groups[0]['lr']

            for name, param in self.model.named_parameters():
                if param.grad is None or 'weight' not in name:
                    continue
                grad_norm = param.grad.detach().norm().item()
                weight_norm = param.detach().norm().item()
                if weight_norm > 1e-8:
                    ratio = lr * grad_norm / weight_norm
                    self.history[name].append(ratio)

        def plot(self, figsize=(14, 5)):
            """可视化更新幅度比"""
            if not self.history:
                print("没有记录，请先调用 record()")
                return

            names = list(self.history.keys())
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # 左图：各层更新比随训练步数变化
            for name in names[:8]:
                data = self.history[name]
                axes[0].plot(data, label=name.replace('.weight', '')[:20],
                             alpha=0.7, linewidth=1.2)

            axes[0].axhline(1e-3, color='green', linestyle='--', linewidth=2,
                            label='理想值 (1e-3)')
            axes[0].axhline(1e-2, color='orange', linestyle='--', alpha=0.7,
                            label='警告上限 (1e-2)')
            axes[0].axhline(1e-4, color='blue', linestyle='--', alpha=0.7,
                            label='警告下限 (1e-4)')
            axes[0].set_yscale('log')
            axes[0].set_title('更新幅度比 (lr × ‖grad‖ / ‖weight‖)\n理想值约 1e-3', fontsize=10)
            axes[0].set_xlabel('训练步数')
            axes[0].set_ylabel('更新比（log scale）')
            axes[0].legend(fontsize=7, loc='upper right')
            axes[0].grid(True, alpha=0.3)

            # 右图：最终各层更新比
            final_ratios = [self.history[n][-1] if self.history[n] else 0 for n in names]
            colors = ['red' if r > 1e-2 or r < 1e-4 else 'steelblue' for r in final_ratios]
            axes[1].barh(range(len(names)), final_ratios, color=colors)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels([n.replace('.weight', '')[:25] for n in names], fontsize=8)
            axes[1].set_xscale('log')
            axes[1].axvline(1e-3, color='green', linestyle='--', linewidth=2)
            axes[1].set_title('各层最终更新幅度比\n红=异常，蓝=正常', fontsize=10)
            axes[1].grid(True, alpha=0.3)

            plt.suptitle('更新幅度监控（诊断学习率是否合适）', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('update_ratio.png', dpi=150, bbox_inches='tight')
            plt.show()

            # 打印诊断
            print("\n更新幅度诊断：")
            for name, ratio in zip(names, final_ratios):
                if ratio > 1e-2:
                    status = "⚠️  过大（学习率可能太高）"
                elif ratio < 1e-4:
                    status = "⚠️  过小（学习率可能太低）"
                else:
                    status = "✅ 正常"
                short = name.replace('.weight', '')[:30]
                print(f"  {short:30s}: {ratio:.2e}  {status}")


    # ─────────────────────────────────────────────────────────
    # 完整演示
    # ─────────────────────────────────────────────────────────


    def print_learning_guide():
        print("""
    学习导读：训练动态页要把 loss、权重分布、激活饱和和更新幅度放在一起看。

    1. 权重分布演化怎么看
       - 上排是均值和标准差：均值长期偏离 0，可能表示某些层被推到单侧区域。
       - 下排是 P5 到 P95 百分位范围：范围快速变宽常见于学习率过高，范围长期不变常见于学习率过低或梯度传不过来。

    2. 激活饱和率怎么看
       - ReLU 的饱和率表示输出为 0 的比例，也就是“死亡神经元”比例。
       - Tanh/Sigmoid 的饱和率表示 |a| > 0.9 的比例，越高越容易让梯度变小。
       - 超过 20% 就该留意，超过 50% 通常要查初始化、输入尺度或激活函数选择。

    3. 更新幅度比怎么看
       - 更新幅度比 = lr * grad_norm / weight_norm，比单看梯度更接近“参数这一步被改了多少”。
       - 经验健康带大致在 1e-4 到 1e-2，中心参考值约 1e-3。
       - 太大：训练曲线常震荡或发散；太小：loss 像没有训练一样缓慢。

    工程坑案例：
       一个模型验证集迟迟不升，表面看像欠拟合；更新幅度图显示大部分层都低于 1e-4，根因是学习率和梯度裁剪一起太保守。
       先把学习率提高到 3 倍，再放宽 grad_clip，曲线才开始下降。

    进阶思考：
       为什么更新幅度比比梯度范数更能判断学习率？如果某层权重很小但梯度正常，它的更新比会给出什么警告？
    """.strip())


    def demo_training_dynamics():
        print_learning_guide()
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        X = torch.randn(200, 20)
        y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        tracker = DistributionTracker(model)
        tracker.register_activation_hooks()
        ratio_monitor = UpdateRatioMonitor(model, optimizer)

        for step in range(100):
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()

            ratio_monitor.record()

            if step % 10 == 0:
                tracker.record_weights(step)

            optimizer.step()

        tracker.remove_hooks()

        print("绘制权重分布演化...")
        tracker.plot_weight_evolution()

        print("\n绘制激活饱和率...")
        tracker.plot_activation_saturation()

        print("\n绘制更新幅度比...")
        ratio_monitor.plot()

        return tracker, ratio_monitor

    tracker, ratio_monitor = demo_training_dynamics()
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part5_toolbox/03_training_dynamics.py", e)
