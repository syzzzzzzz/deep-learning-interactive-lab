MODULE_TITLE = "数据集与优化器"
MODULE_SUMMARY = "理解数据划分、批训练、SGD、Adam 和优化轨迹。"
MODULE_TAGS = ["基础", "数据集", "优化器", "训练"]

try:
    """
    自动生成自: part1_foundations\03_datasets_optimizers.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # ─────────────────────────────────────────────────────────
    # 在 Rosenbrock 函数上对比优化器
    # ─────────────────────────────────────────────────────────

    def rosenbrock(x, y, a=1, b=100):
        """Rosenbrock 函数：经典优化测试函数，最小值在 (a, a^2)"""
        return (a - x)**2 + b * (y - x**2)**2

    def compare_optimizers():
        """在 2D 损失曲面上可视化不同优化器的轨迹"""

        # 创建损失曲面
        x_range = np.linspace(-2, 2, 300)
        y_range = np.linspace(-1, 3, 300)
        X, Y = np.meshgrid(x_range, y_range)
        Z = rosenbrock(X, Y)
        Z_log = np.log(Z + 1)  # 对数变换，便于可视化

        # 优化器配置
        optimizer_configs = {
            'SGD (lr=0.001)':      lambda p: torch.optim.SGD(p, lr=0.001),
            'SGD+Momentum':        lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
            'Adam (lr=0.01)':      lambda p: torch.optim.Adam(p, lr=0.01),
            'RMSprop':             lambda p: torch.optim.RMSprop(p, lr=0.01),
            'AdaGrad':             lambda p: torch.optim.Adagrad(p, lr=0.1),
            'AdamW':               lambda p: torch.optim.AdamW(p, lr=0.01),
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        colors = plt.cm.rainbow(np.linspace(0, 1, len(optimizer_configs)))

        for ax, (name, opt_fn), color in zip(axes, optimizer_configs.items(), colors):
            # 初始化参数（从同一起点出发）
            params = torch.tensor([-1.5, 1.5], requires_grad=True)
            optimizer = opt_fn([params])

            trajectory = [params.detach().numpy().copy()]
            losses = []

            for step in range(1000):
                x, y = params[0], params[1]
                loss = (1 - x)**2 + 100 * (y - x**2)**2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trajectory.append(params.detach().numpy().copy())
                losses.append(loss.item())

                if loss.item() < 1e-6:
                    break

            trajectory = np.array(trajectory)

            # 绘制损失曲面
            ax.contourf(X, Y, Z_log, levels=50, cmap='viridis', alpha=0.6)
            ax.contour(X, Y, Z_log, levels=20, colors='white', alpha=0.3, linewidths=0.5)

            # 绘制轨迹
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-',
                    color=color, linewidth=1.5, alpha=0.8)
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                       c='white', s=100, zorder=5, marker='o', label='起点')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                       c='red', s=100, zorder=5, marker='*', label='终点')
            ax.scatter(1, 1, c='yellow', s=200, zorder=6, marker='*', label='最优点(1,1)')

            final_loss = losses[-1] if losses else float('inf')
            ax.set_title(f'{name}\n步数={len(losses)}, 最终Loss={final_loss:.2e}',
                         fontsize=10)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1, 3)
            ax.legend(fontsize=7, loc='upper right')

        plt.suptitle('不同优化器在 Rosenbrock 函数上的轨迹对比',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


    # ─────────────────────────────────────────────────────────
    # 学习率调度器可视化
    # ─────────────────────────────────────────────────────────

    def visualize_lr_schedulers():
        """可视化不同学习率调度策略"""
        initial_lr = 0.1
        n_epochs = 100

        # 创建虚拟模型和优化器
        model = nn.Linear(1, 1)

        schedulers = {
            'StepLR (step=30, γ=0.1)': lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=30, gamma=0.1),
            'CosineAnnealing': lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs),
            'ExponentialLR (γ=0.95)': lambda opt: torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=0.95),
            'OneCycleLR': lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=initial_lr, total_steps=n_epochs),
            'CyclicLR': lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, base_lr=0.001, max_lr=initial_lr, step_size_up=20),
            'ReduceLROnPlateau': None,  # 特殊处理
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for ax, (name, sched_fn) in zip(axes, schedulers.items()):
            if sched_fn is None:
                # ReduceLROnPlateau 模拟
                lrs = [initial_lr]
                lr = initial_lr
                patience_count = 0
                best_loss = float('inf')
                for epoch in range(n_epochs):
                    # 模拟损失（先下降后震荡）
                    fake_loss = 1.0 / (epoch + 1) + 0.1 * np.sin(epoch * 0.3)
                    if fake_loss < best_loss - 0.01:
                        best_loss = fake_loss
                        patience_count = 0
                    else:
                        patience_count += 1
                    if patience_count >= 10:
                        lr *= 0.5
                        patience_count = 0
                    lrs.append(lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=initial_lr)
                scheduler = sched_fn(opt)
                lrs = []
                for epoch in range(n_epochs):
                    lrs.append(opt.param_groups[0]['lr'])
                    scheduler.step()

            ax.plot(lrs, 'b-', linewidth=2)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('学习率')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        plt.suptitle('学习率调度策略对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lr_schedulers.png', dpi=150, bbox_inches='tight')
        plt.show()


    # ============================================================
    # 代码段 2
    # ============================================================

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # ─────────────────────────────────────────────────────────
    # 损失函数可视化
    # ─────────────────────────────────────────────────────────

    def visualize_loss_functions():
        """可视化常用损失函数的特性"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        # 1. MSE vs MAE vs Huber（回归损失）
        y_true = torch.tensor(0.0)
        y_pred = torch.linspace(-3, 3, 200)

        mse_loss = (y_pred - y_true) ** 2
        mae_loss = torch.abs(y_pred - y_true)
        huber_loss = nn.HuberLoss(reduction='none')(y_pred, y_true.expand_as(y_pred))

        axes[0].plot(y_pred.numpy(), mse_loss.numpy(), 'b-', linewidth=2, label='MSE')
        axes[0].plot(y_pred.numpy(), mae_loss.numpy(), 'r-', linewidth=2, label='MAE')
        axes[0].plot(y_pred.numpy(), huber_loss.detach().numpy(), 'g-', linewidth=2, label='Huber')
        axes[0].set_title('回归损失函数对比', fontsize=11)
        axes[0].set_xlabel('预测值 - 真实值')
        axes[0].set_ylabel('损失值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 5)

        # 2. BCE Loss（二分类）
        p = torch.linspace(0.001, 0.999, 200)
        bce_pos = -torch.log(p)       # 真实标签=1
        bce_neg = -torch.log(1 - p)   # 真实标签=0

        axes[1].plot(p.numpy(), bce_pos.numpy(), 'b-', linewidth=2, label='真实=1: -log(p)')
        axes[1].plot(p.numpy(), bce_neg.numpy(), 'r-', linewidth=2, label='真实=0: -log(1-p)')
        axes[1].set_title('二元交叉熵损失 (BCE)', fontsize=11)
        axes[1].set_xlabel('预测概率 p')
        axes[1].set_ylabel('损失值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 5)

        # 3. Cross-Entropy（多分类）
        # 展示 softmax + cross-entropy 的效果
        logits = torch.tensor([[2.0, 1.0, 0.1],   # 正确预测
                                [0.1, 0.1, 2.0],   # 错误预测
                                [1.0, 1.0, 1.0]])  # 不确定

        probs = torch.softmax(logits, dim=1)
        true_labels = torch.tensor([0, 0, 0])  # 真实类别都是0
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, true_labels)

        x_pos = np.arange(3)
        width = 0.25
        labels = ['正确预测', '错误预测', '不确定']
        colors = ['green', 'red', 'orange']

        for i, (prob, label, color) in enumerate(zip(probs, labels, colors)):
            axes[2].bar(x_pos + i * width, prob.numpy(), width,
                        label=f'{label} (Loss={ce_loss[i]:.2f})',
                        color=color, alpha=0.7)

        axes[2].set_title('交叉熵损失：不同预测情况', fontsize=11)
        axes[2].set_xlabel('类别')
        axes[2].set_ylabel('预测概率')
        axes[2].set_xticks(x_pos + width)
        axes[2].set_xticklabels(['类别0\n(真实)', '类别1', '类别2'])
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # 4. KL 散度
        x = np.linspace(-3, 3, 200)
        p_dist = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)  # 标准正态
        q_dist1 = np.exp(-(x-1)**2 / 2) / np.sqrt(2 * np.pi)  # 偏移
        q_dist2 = np.exp(-x**2 / (2 * 4)) / np.sqrt(2 * np.pi * 4)  # 更宽

        axes[3].plot(x, p_dist, 'b-', linewidth=2, label='P（目标分布）')
        axes[3].plot(x, q_dist1, 'r--', linewidth=2, label='Q1（偏移）')
        axes[3].plot(x, q_dist2, 'g--', linewidth=2, label='Q2（更宽）')
        axes[3].fill_between(x, p_dist, q_dist1, alpha=0.2, color='red', label='KL(P||Q1)')
        axes[3].set_title('KL 散度：分布差异度量', fontsize=11)
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('概率密度')
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)

        # 5. 损失函数选择指南
        axes[4].axis('off')
        guide_text = """
    损失函数选择指南：

    回归任务：
      MSE    → 对异常值敏感，梯度大
      MAE    → 对异常值鲁棒，梯度恒定
      Huber  → MSE+MAE 的折中

    分类任务：
      BCE    → 二分类（输出sigmoid）
      CE     → 多分类（输出softmax）
      Focal  → 类别不平衡时使用

    生成模型：
      KL散度  → VAE
      对抗损失 → GAN
      感知损失 → 图像生成

    序列任务：
      CTC    → 语音识别
      NLL    → 语言模型
    """
        axes[4].text(0.05, 0.95, guide_text, transform=axes[4].transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 6. 损失曲面形状对比
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x_range, y_range)

        # 凸函数（MSE）
        Z_convex = X**2 + Y**2
        # 非凸函数（多个局部最小值）
        Z_nonconvex = np.sin(3*X) * np.cos(3*Y) + X**2 * 0.1 + Y**2 * 0.1

        axes[5].contourf(X, Y, Z_nonconvex, levels=30, cmap='viridis')
        axes[5].set_title('非凸损失曲面\n（存在多个局部最小值）', fontsize=11)
        axes[5].set_xlabel('参数1')
        axes[5].set_ylabel('参数2')

        plt.suptitle('损失函数全景', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('loss_functions.png', dpi=150, bbox_inches='tight')
        plt.show()


    # ============================================================
    # 代码段 3
    # ============================================================

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    def batch_size_experiment():
        """
        实验：不同批次大小对训练的影响
        小批次 → 噪声大，但泛化好
        大批次 → 稳定，但可能陷入尖锐最小值
        """
        torch.manual_seed(42)

        # 生成数据
        n = 1000
        X = torch.randn(n, 10)
        w_true = torch.randn(10)
        y = (X @ w_true > 0).float()

        batch_sizes = [1, 8, 32, 256, 1000]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for bs in batch_sizes:
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()

            losses = []
            for epoch in range(100):
                epoch_loss = 0
                n_batches = 0
                for i in range(0, n, bs):
                    X_batch = X[i:i+bs]
                    y_batch = y[i:i+bs].unsqueeze(1)

                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                losses.append(epoch_loss / n_batches)

            axes[0].plot(losses, label=f'BS={bs}', alpha=0.8)

        axes[0].set_title('不同批次大小的训练损失曲线', fontsize=12)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 批次大小 vs 训练时间（理论分析）
        batch_sizes_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        # 理论：每个 epoch 的更新次数
        updates_per_epoch = [n // bs for bs in batch_sizes_range]
        # 理论：每次更新的计算量（正比于 batch size）
        compute_per_update = batch_sizes_range

        axes[1].plot(batch_sizes_range, updates_per_epoch, 'b-o', label='每epoch更新次数')
        axes[1].set_xlabel('批次大小')
        axes[1].set_ylabel('每 Epoch 更新次数')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title('批次大小 vs 更新频率', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('batch_size_experiment.png', dpi=150, bbox_inches='tight')
        plt.show()

    if __name__ == '__main__':
        value = rosenbrock(np.array([1.0]), np.array([1.0]))[0]
        assert np.isclose(value, 0.0)
        print("Dataset and optimizer demos loaded. Call visualization functions explicitly to generate figures.")
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part1_foundations/03_datasets_optimizers.py", e)
