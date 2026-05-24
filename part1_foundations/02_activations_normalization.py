"""
自动生成自: part1_foundations\02_activations_normalization.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# 所有常用激活函数可视化
# ─────────────────────────────────────────────────────────

def plot_all_activations():
    x = torch.linspace(-5, 5, 500)

    activations = {
        'Sigmoid':    (torch.sigmoid(x),    '1/(1+e^{-x})',    '蓝色'),
        'Tanh':       (torch.tanh(x),        'tanh(x)',          '橙色'),
        'ReLU':       (torch.relu(x),        'max(0,x)',         '绿色'),
        'LeakyReLU':  (nn.LeakyReLU(0.1)(x),'max(0.1x,x)',     '红色'),
        'ELU':        (nn.ELU()(x),          'ELU(x)',           '紫色'),
        'GELU':       (nn.GELU()(x),         'GELU(x)',          '棕色'),
        'Swish':      (x * torch.sigmoid(x), 'x·σ(x)',          '粉色'),
        'Mish':       (x * torch.tanh(nn.Softplus()(x)), 'Mish', '青色'),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, (y, formula, color)) in enumerate(activations.items()):
        ax = axes[idx]
        ax.plot(x.numpy(), y.detach().numpy(), linewidth=2.5, label=formula)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_title(f'{name}\n{formula}', fontsize=11)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # 标注关键特性
        if name == 'Sigmoid':
            ax.annotate('饱和区（梯度≈0）', xy=(-4, 0.02), fontsize=8, color='red')
        elif name == 'ReLU':
            ax.annotate('死亡ReLU区域', xy=(-3, 0.1), fontsize=8, color='red')

    plt.suptitle('常用激活函数对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activations.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_all_activations()

# ─────────────────────────────────────────────────────────
# 激活函数的梯度（导数）
# ─────────────────────────────────────────────────────────

def plot_activation_gradients():
    x = torch.linspace(-5, 5, 500, requires_grad=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, act_fn) in zip(axes, [
        ('Sigmoid', torch.sigmoid),
        ('Tanh', torch.tanh),
        ('ReLU', torch.relu),
    ]):
        x_req = x.clone().requires_grad_(True)
        y = act_fn(x_req)
        y.sum().backward()
        grad = x_req.grad.detach().numpy()

        ax.plot(x.numpy(), act_fn(x).detach().numpy(),
                'b-', linewidth=2, label='函数值')
        ax.plot(x.numpy(), grad, 'r--', linewidth=2, label='导数（梯度）')
        ax.set_title(f'{name} 及其导数', fontsize=12)
        ax.set_xlim(-5, 5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 标注最大梯度
        max_grad = grad.max()
        ax.annotate(f'最大梯度={max_grad:.3f}',
                    xy=(0, max_grad), fontsize=9, color='red',
                    xytext=(1, max_grad + 0.1))

    plt.suptitle('激活函数的梯度（导数）', fontsize=13)
    plt.tight_layout()
    plt.savefig('activation_gradients.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_activation_gradients()

# ============================================================
# 代码段 2
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────
# 不同初始化方法对比
# ─────────────────────────────────────────────────────────

def compare_initializations():
    """
    展示不同初始化方法对信号传播的影响
    关键问题：信号在深层网络中是否会消失或爆炸？
    """
    depth = 50
    width = 256
    n_samples = 1000

    init_methods = {
        '全零初始化': lambda: torch.zeros(width, width),
        '随机小值': lambda: torch.randn(width, width) * 0.01,
        '随机大值': lambda: torch.randn(width, width) * 1.0,
        'Xavier初始化': lambda: nn.init.xavier_normal_(torch.empty(width, width)),
        'He初始化': lambda: nn.init.kaiming_normal_(torch.empty(width, width)),
    }

    fig, axes = plt.subplots(1, len(init_methods), figsize=(20, 4))

    for ax, (name, init_fn) in zip(axes, init_methods.items()):
        # 模拟信号通过深层网络
        x = torch.randn(n_samples, width)
        activations_std = [x.std().item()]

        for layer in range(depth):
            W = init_fn()
            x = torch.relu(x @ W)
            activations_std.append(x.std().item())

            # 防止数值溢出
            if x.std() > 1e6 or x.std() < 1e-10:
                activations_std.extend([activations_std[-1]] * (depth - layer - 1))
                break

        ax.plot(activations_std, 'b-o', markersize=3)
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel('层数')
        ax.set_ylabel('激活值标准差')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理想值=1')
        ax.legend(fontsize=8)

        # 诊断
        final_std = activations_std[-1]
        if final_std < 1e-6:
            ax.set_facecolor('#ffe6e6')  # 红色背景：梯度消失
            ax.set_title(f'{name}\n⚠️ 信号消失', fontsize=10, color='red')
        elif final_std > 1e3:
            ax.set_facecolor('#fff3e6')  # 橙色背景：梯度爆炸
            ax.set_title(f'{name}\n🔥 信号爆炸', fontsize=10, color='orange')
        else:
            ax.set_facecolor('#e6ffe6')  # 绿色背景：正常
            ax.set_title(f'{name}\n✅ 信号稳定', fontsize=10, color='green')

    plt.suptitle('不同初始化方法对信号传播的影响（50层网络）',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initialization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_initializations()

# ─────────────────────────────────────────────────────────
# Xavier 和 He 初始化的数学原理
# ─────────────────────────────────────────────────────────

def explain_initialization_math():
    print("=" * 60)
    print("初始化方法的数学原理")
    print("=" * 60)

    print("""
Xavier 初始化（适合 Sigmoid/Tanh）：
  目标：保持每层输出的方差不变
  W ~ Uniform(-√(6/(n_in+n_out)), √(6/(n_in+n_out)))
  或 W ~ Normal(0, √(2/(n_in+n_out)))

He 初始化（适合 ReLU）：
  ReLU 会将一半的神经元置零，所以需要更大的方差
  W ~ Normal(0, √(2/n_in))

直觉：
  - 如果权重太小 → 信号逐层衰减 → 梯度消失
  - 如果权重太大 → 信号逐层放大 → 梯度爆炸
  - 好的初始化 → 信号在各层保持相似的方差
""")

    # 验证
    n_in, n_out = 256, 256
    xavier_std = np.sqrt(2 / (n_in + n_out))
    he_std = np.sqrt(2 / n_in)

    print(f"Xavier std = {xavier_std:.4f}")
    print(f"He std = {he_std:.4f}")

    # 实际使用
    linear = nn.Linear(256, 256)
    print(f"\nPyTorch 默认初始化 std = {linear.weight.std().item():.4f}")

    nn.init.xavier_normal_(linear.weight)
    print(f"Xavier 初始化后 std = {linear.weight.std().item():.4f}")

    nn.init.kaiming_normal_(linear.weight)
    print(f"He 初始化后 std = {linear.weight.std().item():.4f}")

explain_initialization_math()

# ============================================================
# 代码段 3
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────
# 归一化方法对比可视化
# ─────────────────────────────────────────────────────────

def visualize_normalization():
    """
    可视化 BatchNorm、LayerNorm、InstanceNorm 的区别
    关键：它们在哪个维度上计算均值和方差？
    """
    # 模拟一个 batch: [B=4, C=3, H=2, W=2]
    # B=batch, C=channel, H=height, W=width
    torch.manual_seed(42)
    x = torch.randn(4, 3, 2, 2) * 3 + 2  # 均值≠0, 方差≠1

    print("输入张量 shape:", x.shape)
    print(f"输入统计: mean={x.mean():.3f}, std={x.std():.3f}")

    # BatchNorm: 在 [B, H, W] 维度上归一化（每个通道独立）
    bn = nn.BatchNorm2d(3, affine=False)
    x_bn = bn(x)
    print(f"\nBatchNorm 后: mean={x_bn.mean():.3f}, std={x_bn.std():.3f}")
    print(f"  每通道均值: {x_bn.mean(dim=[0,2,3]).detach().numpy()}")

    # LayerNorm: 在 [C, H, W] 维度上归一化（每个样本独立）
    ln = nn.LayerNorm([3, 2, 2], elementwise_affine=False)
    x_ln = ln(x)
    print(f"\nLayerNorm 后: mean={x_ln.mean():.3f}, std={x_ln.std():.3f}")
    print(f"  每样本均值: {x_ln.mean(dim=[1,2,3]).detach().numpy()}")

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    norm_methods = [
        ('原始输入', x),
        ('BatchNorm', x_bn),
        ('LayerNorm', x_ln),
    ]

    for col, (name, tensor) in enumerate(norm_methods):
        data = tensor.detach().numpy().flatten()

        # 直方图
        axes[0, col].hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        axes[0, col].set_title(f'{name}\n分布', fontsize=11)
        axes[0, col].set_xlabel('值')
        axes[0, col].set_ylabel('频次')
        axes[0, col].axvline(x=data.mean(), color='r', linestyle='--',
                              label=f'均值={data.mean():.2f}')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        # 热力图（第一个样本，第一个通道）
        sample = tensor[0, 0].detach().numpy()
        im = axes[1, col].imshow(sample, cmap='RdBu', aspect='auto')
        axes[1, col].set_title(f'{name}\n样本[0]通道[0]', fontsize=11)
        plt.colorbar(im, ax=axes[1, col])

    # 归一化方法示意图
    axes[0, 3].axis('off')
    axes[0, 3].text(0.1, 0.7, """
归一化方法对比：

BatchNorm:
  在 [B, H, W] 上计算
  适合 CNN，依赖 batch size
  训练/推理行为不同

LayerNorm:
  在 [C, H, W] 上计算
  适合 Transformer/RNN
  不依赖 batch size

InstanceNorm:
  在 [H, W] 上计算
  适合风格迁移
  每个样本每个通道独立
""", transform=axes[0, 3].transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[1, 3].axis('off')

    plt.suptitle('归一化方法对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_normalization()

# ============================================================
# 代码段 4
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# 过拟合可视化实验
# ─────────────────────────────────────────────────────────

def overfitting_experiment():
    """
    用多项式拟合演示过拟合
    真实函数: y = sin(x) + noise
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成数据
    n_train = 20
    n_test = 200
    x_train = torch.linspace(-3, 3, n_train).unsqueeze(1)
    y_train = torch.sin(x_train) + torch.randn_like(x_train) * 0.3
    x_test = torch.linspace(-3, 3, n_test).unsqueeze(1)
    y_test = torch.sin(x_test)

    def make_poly_features(x, degree):
        """生成多项式特征"""
        return torch.cat([x ** i for i in range(1, degree + 1)], dim=1)

    degrees = [1, 3, 9, 15]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for col, degree in enumerate(degrees):
        X_train = make_poly_features(x_train, degree)
        X_test = make_poly_features(x_test, degree)

        # 标准化
        mean = X_train.mean(0)
        std = X_train.std(0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # 训练模型
        model = nn.Linear(degree, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_losses = []
        test_losses = []

        for epoch in range(2000):
            pred_train = model(X_train_norm)
            loss_train = nn.MSELoss()(pred_train, y_train)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                pred_test = model(X_test_norm)
                loss_test = nn.MSELoss()(pred_test, y_test)

            train_losses.append(loss_train.item())
            test_losses.append(loss_test.item())

        # 绘制拟合曲线
        with torch.no_grad():
            y_pred = model(X_test_norm).numpy()

        axes[0, col].scatter(x_train.numpy(), y_train.numpy(),
                              c='blue', s=30, zorder=5, label='训练数据')
        axes[0, col].plot(x_test.numpy(), y_test.numpy(),
                           'g-', linewidth=2, label='真实函数')
        axes[0, col].plot(x_test.numpy(), y_pred,
                           'r-', linewidth=2, label='模型预测')
        axes[0, col].set_title(f'{degree}次多项式\n'
                                f'训练Loss={train_losses[-1]:.3f}\n'
                                f'测试Loss={test_losses[-1]:.3f}',
                                fontsize=10)
        axes[0, col].set_ylim(-3, 3)
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.3)

        # 绘制损失曲线
        axes[1, col].plot(train_losses, 'b-', label='训练Loss', alpha=0.8)
        axes[1, col].plot(test_losses, 'r-', label='测试Loss', alpha=0.8)
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('MSE Loss')
        axes[1, col].set_yscale('log')
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        # 标注过拟合
        if test_losses[-1] > train_losses[-1] * 2:
            axes[0, col].set_facecolor('#ffe6e6')
            axes[1, col].set_facecolor('#ffe6e6')

    plt.suptitle('过拟合演示：多项式次数越高，训练误差越小，但测试误差可能更大',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('overfitting_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

overfitting_experiment()

# ─────────────────────────────────────────────────────────
# Dropout 正则化演示
# ─────────────────────────────────────────────────────────

def dropout_demo():
    """演示 Dropout 如何防止过拟合"""
    torch.manual_seed(42)

    # 生成数据
    n = 100
    X = torch.randn(n, 20)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    # 分割训练/测试集
    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y[:60], y[60:]

    def make_model(use_dropout=False, dropout_p=0.5):
        layers = [nn.Linear(20, 64), nn.ReLU()]
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers.extend([nn.Linear(64, 64), nn.ReLU()])
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(64, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    results = {}
    for name, use_dropout in [('无Dropout', False), ('有Dropout(p=0.5)', True)]:
        model = make_model(use_dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        train_accs, test_accs = [], []
        for epoch in range(500):
            model.train()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                train_acc = ((model(X_train) > 0.5) == y_train).float().mean().item()
                test_acc = ((model(X_test) > 0.5) == y_test).float().mean().item()
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        results[name] = (train_accs, test_accs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, (train_accs, test_accs)) in zip(axes, results.items()):
        ax.plot(train_accs, 'b-', label='训练准确率', alpha=0.8)
        ax.plot(test_accs, 'r-', label='测试准确率', alpha=0.8)
        ax.set_title(f'{name}', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('准确率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        gap = np.mean(train_accs[-50:]) - np.mean(test_accs[-50:])
        ax.text(0.05, 0.05, f'过拟合差距: {gap:.3f}',
                transform=ax.transAxes, fontsize=10,
                color='red' if gap > 0.1 else 'green')

    plt.suptitle('Dropout 防止过拟合效果对比', fontsize=13)
    plt.tight_layout()
    plt.savefig('dropout_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

dropout_demo()
