"""
自动生成自: part1_foundations\01_tensors_gradients.md
可独立运行的 Python 源码
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────
# 张量基础操作演示
# ─────────────────────────────────────────────────────────

def tensor_basics_demo():
    print("=" * 60)
    print("张量基础演示")
    print("=" * 60)

    # 创建张量
    scalar = torch.tensor(42.0)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # 模拟一张 RGB 图像 (3通道, 4x4像素)
    image = torch.randn(3, 4, 4)
    # 模拟一批图像 (32张, 3通道, 224x224)
    batch = torch.randn(32, 3, 224, 224)

    print(f"标量: {scalar}, shape={scalar.shape}, ndim={scalar.ndim}")
    print(f"向量: shape={vector.shape}, ndim={vector.ndim}")
    print(f"矩阵: shape={matrix.shape}, ndim={matrix.ndim}")
    print(f"图像: shape={image.shape}, ndim={image.ndim}")
    print(f"批次: shape={batch.shape}, ndim={batch.ndim}")

    # 张量运算
    print("\n--- 张量运算 ---")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")           # 逐元素乘法
    print(f"a @ b = {a @ b}")           # 点积 (内积)
    print(f"a.sum() = {a.sum()}")
    print(f"a.mean() = {a.mean()}")

    # 矩阵乘法
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = A @ B  # 矩阵乘法
    print(f"\n矩阵乘法: ({A.shape}) @ ({B.shape}) = {C.shape}")

    # 广播机制
    print("\n--- 广播机制 ---")
    x = torch.ones(3, 1)   # shape: (3, 1)
    y = torch.ones(1, 4)   # shape: (1, 4)
    z = x + y              # 广播为 (3, 4)
    print(f"广播: {x.shape} + {y.shape} = {z.shape}")

    return scalar, vector, matrix, image, batch

tensor_basics_demo()

# ============================================================
# 代码段 2
# ============================================================

import torch

# ─────────────────────────────────────────────────────────
# 梯度直观演示
# ─────────────────────────────────────────────────────────

def gradient_intuition():
    """
    用最简单的例子理解梯度
    目标：找到 f(x) = (x - 3)^2 的最小值
    答案显然是 x = 3
    """
    print("=" * 60)
    print("梯度直观演示：找 f(x) = (x-3)^2 的最小值")
    print("=" * 60)

    # requires_grad=True 告诉 PyTorch 追踪这个张量的梯度
    x = torch.tensor(0.0, requires_grad=True)

    history = {'x': [], 'loss': [], 'grad': []}

    lr = 0.1  # 学习率
    for step in range(50):
        # 前向传播：计算损失
        loss = (x - 3) ** 2

        # 反向传播：计算梯度
        loss.backward()

        # 记录历史
        history['x'].append(x.item())
        history['loss'].append(loss.item())
        history['grad'].append(x.grad.item())

        # 梯度下降更新
        with torch.no_grad():
            x -= lr * x.grad

        # 清零梯度（重要！否则梯度会累积）
        x.grad.zero_()

        if step % 10 == 0:
            print(f"Step {step:3d}: x={x.item():.4f}, "
                  f"loss={loss.item():.4f}, "
                  f"grad={history['grad'][-1]:.4f}")

    print(f"\n最终结果: x = {x.item():.6f} (目标: 3.0)")
    return history

history = gradient_intuition()

# 可视化梯度下降过程
def plot_gradient_descent(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    steps = range(len(history['x']))

    # x 的变化
    axes[0].plot(steps, history['x'], 'b-o', markersize=3)
    axes[0].axhline(y=3, color='r', linestyle='--', label='目标 x=3')
    axes[0].set_title('参数 x 的变化', fontsize=12)
    axes[0].set_xlabel('训练步数')
    axes[0].set_ylabel('x 的值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 损失的变化
    axes[1].plot(steps, history['loss'], 'r-o', markersize=3)
    axes[1].set_title('损失函数的变化', fontsize=12)
    axes[1].set_xlabel('训练步数')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    # 梯度的变化
    axes[2].plot(steps, history['grad'], 'g-o', markersize=3)
    axes[2].axhline(y=0, color='r', linestyle='--', label='梯度=0')
    axes[2].set_title('梯度的变化', fontsize=12)
    axes[2].set_xlabel('训练步数')
    axes[2].set_ylabel('梯度值')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 在损失曲面上画出轨迹
    fig2, ax = plt.subplots(figsize=(8, 5))
    x_range = np.linspace(-1, 7, 200)
    y_range = (x_range - 3) ** 2
    ax.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = (x-3)²')
    ax.scatter(history['x'], history['loss'], c=range(len(history['x'])),
               cmap='Reds', s=50, zorder=5, label='梯度下降轨迹')
    ax.set_title('梯度下降在损失曲面上的轨迹', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_descent.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存: gradient_descent.png")

plot_gradient_descent(history)

# ============================================================
# 代码段 3
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────
# 手动实现反向传播（理解原理）
# ─────────────────────────────────────────────────────────

class ManualNeuron:
    """
    手动实现单个神经元的前向和反向传播
    y = sigmoid(w*x + b)
    """
    def __init__(self):
        self.w = 0.5
        self.b = 0.1
        # 缓存前向传播的中间值（反向传播需要）
        self.x = None
        self.z = None
        self.a = None
        self.y = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.x = x
        self.z = self.w * x + self.b    # 线性变换
        self.a = self.z                  # 这里 a = z（简化）
        self.y = self.sigmoid(self.a)    # 激活函数
        return self.y

    def backward(self, dy):
        """
        dy: ∂L/∂y（从上层传来的梯度）
        返回: ∂L/∂x（传给下层的梯度）
        """
        # ∂L/∂a = ∂L/∂y · ∂y/∂a = dy · sigmoid'(a)
        da = dy * self.sigmoid_grad(self.a)

        # ∂L/∂w = ∂L/∂a · ∂a/∂w = da · x
        dw = da * self.x

        # ∂L/∂b = ∂L/∂a · ∂a/∂b = da · 1
        db = da * 1

        # ∂L/∂x = ∂L/∂a · ∂a/∂x = da · w
        dx = da * self.w

        return dx, dw, db

# 验证手动实现与 PyTorch 自动微分一致
def verify_backprop():
    print("=" * 60)
    print("验证手动反向传播 vs PyTorch 自动微分")
    print("=" * 60)

    x_val = 2.0
    w_val = 0.5
    b_val = 0.1
    target = 0.8

    # 手动实现
    neuron = ManualNeuron()
    neuron.w = w_val
    neuron.b = b_val
    y_manual = neuron.forward(x_val)
    loss_manual = (y_manual - target) ** 2
    dy = 2 * (y_manual - target)  # MSE 的梯度
    dx, dw, db = neuron.backward(dy)

    print(f"手动实现:")
    print(f"  y = {y_manual:.6f}")
    print(f"  loss = {loss_manual:.6f}")
    print(f"  ∂L/∂w = {dw:.6f}")
    print(f"  ∂L/∂b = {db:.6f}")

    # PyTorch 自动微分
    x = torch.tensor(x_val)
    w = torch.tensor(w_val, requires_grad=True)
    b = torch.tensor(b_val, requires_grad=True)
    t = torch.tensor(target)

    z = w * x + b
    y = torch.sigmoid(z)
    loss = (y - t) ** 2
    loss.backward()

    print(f"\nPyTorch 自动微分:")
    print(f"  y = {y.item():.6f}")
    print(f"  loss = {loss.item():.6f}")
    print(f"  ∂L/∂w = {w.grad.item():.6f}")
    print(f"  ∂L/∂b = {b.grad.item():.6f}")

    print(f"\n差异:")
    print(f"  ∂L/∂w 差异: {abs(dw - w.grad.item()):.2e}")
    print(f"  ∂L/∂b 差异: {abs(db - b.grad.item()):.2e}")

verify_backprop()

# ============================================================
# 代码段 4
# ============================================================

import torch
from torch.autograd import Variable

def visualize_computation_graph():
    """
    可视化计算图（需要 torchviz: pip install torchviz）
    这里用文字描述代替
    """
    print("\n计算图示意：")
    print("""
    x ──────────────────────────────────────────┐
                                                 ↓
    w ──→ [w*x] ──→ z ──→ [z+b] ──→ a ──→ [sigmoid] ──→ y ──→ [MSE] ──→ L
                                ↑
    b ──────────────────────────┘

    反向传播（梯度流动方向相反）：
    L ──→ [∂L/∂y] ──→ y ──→ [∂y/∂a] ──→ a ──→ [∂a/∂z] ──→ z ──→ [∂z/∂w] ──→ w
                                                              └──→ [∂z/∂b] ──→ b
                                                              └──→ [∂z/∂x] ──→ x
    """)

    # 用 PyTorch 展示梯度流
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)

    z = w * x
    a = z + b
    y = torch.sigmoid(a)
    L = (y - 0.8) ** 2

    L.backward()

    print("各节点的梯度：")
    print(f"  ∂L/∂w = {w.grad:.6f}")
    print(f"  ∂L/∂b = {b.grad:.6f}")
    print(f"  ∂L/∂x = {x.grad:.6f}")

visualize_computation_graph()

# ============================================================
# 代码段 5
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────
# 单层神经网络：学习 XOR 问题
# ─────────────────────────────────────────────────────────

class TinyNet(nn.Module):
    """
    两层神经网络解决 XOR 问题
    XOR 是线性不可分的，需要隐藏层
    """
    def __init__(self, hidden_size=4):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        h = self.activation(self.layer1(x))  # 隐藏层
        y = self.activation(self.layer2(h))  # 输出层
        return y

def train_xor():
    # XOR 数据
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

    model = TinyNet(hidden_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(5000):
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # 测试
    with torch.no_grad():
        pred = model(X)
        print("XOR 预测结果：")
        for i, (xi, yi, pi) in enumerate(zip(X, y, pred)):
            print(f"  {xi.tolist()} → 真实:{yi.item():.0f}, "
                  f"预测:{pi.item():.4f} ({'✓' if abs(pi.item()-yi.item())<0.5 else '✗'})")

    # 可视化决策边界
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(losses)
    axes[0].set_title('训练损失曲线', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # 决策边界
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                          np.linspace(-0.5, 1.5, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)

    axes[1].contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    colors = ['blue', 'red', 'red', 'blue']
    labels = ['(0,0)→0', '(0,1)→1', '(1,0)→1', '(1,1)→0']
    for i, (xi, ci, li) in enumerate(zip(X, colors, labels)):
        axes[1].scatter(xi[0], xi[1], c=ci, s=200, zorder=5, label=li)
    axes[1].set_title('XOR 决策边界', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('xor_decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

    return model, losses

model, losses = train_xor()

# ============================================================
# 代码段 6
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class GradientFlowVisualizer:
    """
    可视化神经网络中梯度的流动
    帮助诊断梯度消失/爆炸问题
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_history = defaultdict(list)
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self.grad_history[n].append(
                        grad.abs().mean().item()
                    )
                )
                self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def plot(self, title="梯度流动"):
        if not self.grad_history:
            print("没有梯度数据，请先运行训练")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 最终梯度大小（按层）
        names = list(self.grad_history.keys())
        final_grads = [self.grad_history[n][-1] for n in names]

        axes[0].barh(range(len(names)), final_grads, color='steelblue')
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels([n.replace('.', '\n') for n in names], fontsize=8)
        axes[0].set_xlabel('梯度绝对均值')
        axes[0].set_title(f'{title} - 各层梯度大小', fontsize=12)
        axes[0].axvline(x=1e-4, color='r', linestyle='--', label='消失阈值')
        axes[0].axvline(x=10, color='orange', linestyle='--', label='爆炸阈值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 梯度随训练步数的变化
        for name in names[:4]:  # 只显示前4层
            grads = self.grad_history[name]
            axes[1].plot(grads, label=name, alpha=0.8)
        axes[1].set_xlabel('训练步数')
        axes[1].set_ylabel('梯度绝对均值')
        axes[1].set_title('梯度随训练的变化', fontsize=12)
        axes[1].legend(fontsize=8)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
        plt.show()

    def diagnose(self):
        """诊断梯度问题"""
        print("\n梯度诊断报告：")
        print("-" * 50)
        for name, grads in self.grad_history.items():
            if not grads:
                continue
            avg_grad = np.mean(grads[-10:])  # 最近10步的平均
            if avg_grad < 1e-6:
                status = "⚠️  梯度消失"
            elif avg_grad > 100:
                status = "🔥 梯度爆炸"
            else:
                status = "✅ 正常"
            print(f"  {name:30s}: {avg_grad:.2e}  {status}")

# 使用示例
class DeepNet(nn.Module):
    def __init__(self, depth=5):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(32, 32), nn.Sigmoid()])
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def demo_gradient_flow():
    model = DeepNet(depth=5)
    visualizer = GradientFlowVisualizer(model)

    X = torch.randn(100, 32)
    y = torch.randn(100, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(50):
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    visualizer.plot("深层网络（Sigmoid激活）")
    visualizer.diagnose()
    visualizer.remove_hooks()

demo_gradient_flow()
