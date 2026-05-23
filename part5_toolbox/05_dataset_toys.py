"""
自动生成自: part5_toolbox\05_dataset_toys.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons, make_circles, make_blobs
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────
# 数据集生成器
# ─────────────────────────────────────────────────────────

class DatasetFactory:
    """
    数据集工厂：一行代码生成各种玩具数据集

    支持：
    - 二分类：月牙、圆环、螺旋、XOR、高斯混合
    - 多分类：多中心高斯、同心圆
    - 回归：正弦、多项式、带噪声线性
    - 序列：正弦波、随机游走、周期信号
    """

    @staticmethod
    def moons(n: int = 500, noise: float = 0.1, seed: int = 42) -> Tuple:
        """月牙形二分类（非线性可分）"""
        np.random.seed(seed)
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def circles(n: int = 500, noise: float = 0.05, factor: float = 0.5,
                seed: int = 42) -> Tuple:
        """同心圆二分类"""
        X, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=seed)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def spiral(n: int = 500, n_classes: int = 2, noise: float = 0.1,
               seed: int = 42) -> Tuple:
        """螺旋形分类（最难的非线性数据集之一）"""
        np.random.seed(seed)
        X_list, y_list = [], []
        for c in range(n_classes):
            t = np.linspace(0, 4 * np.pi, n // n_classes)
            r = t / (4 * np.pi)
            angle = t + c * (2 * np.pi / n_classes)
            x1 = r * np.cos(angle) + np.random.randn(len(t)) * noise
            x2 = r * np.sin(angle) + np.random.randn(len(t)) * noise
            X_list.append(np.stack([x1, x2], axis=1))
            y_list.append(np.full(len(t), c))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        idx = np.random.permutation(len(X))
        return (torch.tensor(X[idx], dtype=torch.float32),
                torch.tensor(y[idx], dtype=torch.long))

    @staticmethod
    def xor(n: int = 500, noise: float = 0.1, seed: int = 42) -> Tuple:
        """XOR 问题（线性不可分的经典例子）"""
        np.random.seed(seed)
        X = np.random.randn(n, 2).astype(np.float32)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
        X += np.random.randn(n, 2).astype(np.float32) * noise
        return torch.from_numpy(X), torch.from_numpy(y)

    @staticmethod
    def gaussian_blobs(n: int = 500, n_classes: int = 3,
                       cluster_std: float = 0.5, seed: int = 42) -> Tuple:
        """多中心高斯分布（线性可分）"""
        X, y = make_blobs(n_samples=n, centers=n_classes,
                          cluster_std=cluster_std, random_state=seed)
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    @staticmethod
    def sine_regression(n: int = 300, noise: float = 0.1,
                        freq: float = 1.0, seed: int = 42) -> Tuple:
        """正弦回归（测试模型拟合非线性函数的能力）"""
        np.random.seed(seed)
        X = np.linspace(-3, 3, n).astype(np.float32)
        y = np.sin(freq * X) + np.random.randn(n).astype(np.float32) * noise
        return (torch.from_numpy(X).unsqueeze(1),
                torch.from_numpy(y).unsqueeze(1))

    @staticmethod
    def polynomial_regression(n: int = 200, degree: int = 3,
                               noise: float = 0.3, seed: int = 42) -> Tuple:
        """多项式回归"""
        np.random.seed(seed)
        X = np.linspace(-2, 2, n).astype(np.float32)
        y = sum((-1)**i * X**(i+1) for i in range(degree)).astype(np.float32)
        y += np.random.randn(n).astype(np.float32) * noise
        return (torch.from_numpy(X).unsqueeze(1),
                torch.from_numpy(y).unsqueeze(1))

    @staticmethod
    def imbalanced(n: int = 500, ratio: float = 0.1, seed: int = 42) -> Tuple:
        """不平衡数据集（测试模型对类别不平衡的处理）"""
        np.random.seed(seed)
        n_pos = int(n * ratio)
        n_neg = n - n_pos
        X_pos = np.random.randn(n_pos, 2).astype(np.float32) + np.array([2, 2])
        X_neg = np.random.randn(n_neg, 2).astype(np.float32)
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
        idx = np.random.permutation(n)
        return torch.from_numpy(X[idx]), torch.from_numpy(y[idx])


# ─────────────────────────────────────────────────────────
# 决策边界可视化
# ─────────────────────────────────────────────────────────

def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                            title: str = '', resolution: int = 200,
                            figsize=(8, 6)):
    """
    可视化二维分类模型的决策边界

    model: 输入 [N, 2]，输出 [N, n_classes] logits
    X: [N, 2] 特征
    y: [N] 标签
    """
    model.eval()
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min.item(), x1_max.item(), resolution),
        np.linspace(x2_min.item(), x2_max.item(), resolution)
    )
    grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)

    with torch.no_grad():
        logits = model(grid)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).long()
        else:
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

    n_classes = int(y.max().item()) + 1
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 决策边界
    Z = preds.numpy().reshape(xx1.shape)
    axes[0].contourf(xx1, xx2, Z, alpha=0.3, cmap='Set1',
                     levels=np.arange(-0.5, n_classes + 0.5, 1))
    axes[0].contour(xx1, xx2, Z, colors='gray', linewidths=0.5, alpha=0.5)
    for c in range(n_classes):
        mask = y == c
        axes[0].scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.7,
                        color=colors[c], label=f'类别 {c}', edgecolors='white', linewidth=0.3)
    axes[0].set_title(f'决策边界\n{title}', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # 置信度热力图（仅二分类）
    if n_classes == 2:
        conf = probs[:, 1].numpy().reshape(xx1.shape) if probs.dim() > 1 else probs.numpy().reshape(xx1.shape)
        im = axes[1].contourf(xx1, xx2, conf, levels=20, cmap='RdBu_r', alpha=0.8)
        plt.colorbar(im, ax=axes[1], label='P(类别=1)')
        for c in range(n_classes):
            mask = y == c
            axes[1].scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.7,
                            color=colors[c], edgecolors='white', linewidth=0.3)
        axes[1].set_title('预测置信度热力图', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.2)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title[:10].replace(" ", "_")}.png',
                dpi=120, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 数据集难度对比实验
# ─────────────────────────────────────────────────────────

def dataset_difficulty_experiment():
    """
    在所有玩具数据集上训练同一个 MLP，对比难度
    """
    datasets = {
        '高斯混合（易）':  DatasetFactory.gaussian_blobs(n=400, n_classes=2),
        'XOR（中）':       DatasetFactory.xor(n=400),
        '月牙形（中）':    DatasetFactory.moons(n=400, noise=0.15),
        '同心圆（难）':    DatasetFactory.circles(n=400, noise=0.05),
        '螺旋（极难）':    DatasetFactory.spiral(n=400, n_classes=2),
    }

    def make_mlp():
        return nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    results = {}
    fig, axes = plt.subplots(2, len(datasets), figsize=(len(datasets) * 3.5, 7))

    for col, (name, (X, y)) in enumerate(datasets.items()):
        torch.manual_seed(42)
        model = make_mlp()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(300):
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 计算准确率
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        results[name] = {'acc': acc, 'final_loss': losses[-1]}

        # 上行：数据分布
        colors = ['#4C72B0', '#DD8452']
        for c in range(2):
            mask = y == c
            axes[0, col].scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.6,
                                  color=colors[c])
        axes[0, col].set_title(f'{name}\n准确率={acc:.1%}', fontsize=9, fontweight='bold')
        axes[0, col].axis('off')

        # 下行：决策边界
        x1_min, x1_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
        x2_min, x2_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                                np.linspace(x2_min, x2_max, 100))
        grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)
        with torch.no_grad():
            Z = model(grid).argmax(1).numpy().reshape(xx1.shape)
        axes[1, col].contourf(xx1, xx2, Z, alpha=0.3, cmap='Set1',
                               levels=[-0.5, 0.5, 1.5])
        for c in range(2):
            mask = y == c
            axes[1, col].scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.5,
                                  color=colors[c])
        axes[1, col].set_title('决策边界', fontsize=8)
        axes[1, col].axis('off')

    plt.suptitle('同一 MLP 在不同数据集上的表现\n（上=数据分布，下=学到的决策边界）',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_difficulty.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n各数据集最终准确率：")
    for name, res in results.items():
        bar = '█' * int(res['acc'] * 20)
        print(f"  {name:15s}: {res['acc']:.1%}  {bar}")

    return results


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_dataset_toys():
    print("数据集难度对比实验")
    results = dataset_difficulty_experiment()

    print("\n单独演示：螺旋数据集 + 决策边界")
    X, y = DatasetFactory.spiral(n=600, n_classes=3)

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 3),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(500):
        loss = nn.CrossEntropyLoss()(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    acc = (model(X).argmax(1) == y).float().mean().item()
    plot_decision_boundary(model, X, y, title=f'螺旋3分类 acc={acc:.1%}')

    return results

results = demo_dataset_toys()
