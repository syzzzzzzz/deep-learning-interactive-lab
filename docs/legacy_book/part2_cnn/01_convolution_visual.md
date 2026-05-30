# 第一章：卷积、池化、填充、步幅——逐步数字计算与可视化

## 来源标注版：卷积到底在图像上找什么

**这一节真正要学的不是“卷积核在图片上滑动”这句口号，而是：同一个局部探测器在整张图上重复使用，把局部像素模式变成特征图上的响应。** 本文对照 CS231n、PyTorch `Conv2d` 文档、D2L 和 LeNet 论文；页面中的 Sobel/Laplacian 等固定核只是教学直觉，不代表训练出的 CNN 一定学到同样权重。[S1][S2][S3][S4]

来源标符：

- [S1] CS231n Convolutional Neural Networks: https://cs231n.github.io/convolutional-networks/
- [S2] PyTorch `torch.nn.Conv2d`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- [S3] Dive into Deep Learning, CNN chapters: https://d2l.ai/chapter_convolutional-neural-networks/index.html
- [S4] LeCun et al., Gradient-Based Learning Applied to Document Recognition: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

### 1. 卷积核：一个会重复使用的局部探测器

卷积核可以先理解成一个小模板。它每次覆盖输入图像的一小块，把窗口里的像素和自己的权重逐项相乘再求和，得到输出特征图上的一个数。如果卷积核像竖直边缘探测器，它滑到竖直边缘附近时响应更强；滑到平坦区域时响应更弱。[S1][S3]

> 操作建议：切换不同卷积核，先猜它会突出边缘、模糊还是锐化，再观察特征图哪里变亮。

### 2. stride 和 padding：控制窗口怎么走、边界怎么看

`stride` 决定窗口每次移动几格；`padding` 决定是否给图像边缘补值。stride 大会让输出变小、计算减少，但可能跳过细节；padding 可以让边界像素也被卷积核充分看到，同时影响输出尺寸。[S1][S2]

```text
H_out = floor((H_in + 2P - K) / S) + 1
```

> 操作建议：把 stride 从 1 调到 2，再打开/关闭 padding。观察输出网格尺寸和边缘响应为什么变化。

### 3. 多通道：彩色图是一组局部证据，不是一张平面

彩色图像常见输入形状是 `[B, C_in, H, W]`。卷积核不是只看单张灰度图，而是跨通道汇总局部证据，生成 `[B, C_out, H_out, W_out]` 这样的输出。每个输出通道可以理解成一种探测器的响应图。[S2][S3]

> 操作建议：把特征图当成“证据强度图”看。问自己：亮的地方像边缘、纹理，还是亮暗变化？

### 4. CNN 为什么有效：局部连接 + 参数共享

图像有明显的空间结构：附近像素往往有关。同一个边缘或纹理探测器也可能出现在图像任意位置。卷积层利用这两个事实，通过局部连接和参数共享减少参数量，也让模型更适合视觉任务。[S1][S4]

> 操作建议：看完卷积动画后，去中央控制台比较 Linear 和 Conv 预设。思考为什么同样处理图像，卷积通常更省参数。

## 1.1 卷积的本质

卷积核在输入上滑动，每次做**逐元素乘法再求和**（互相关运算）。
不同的卷积核提取不同的特征：边缘、纹理、颜色梯度……

## 1.2 逐步数字计算（输入3×32×32，卷积核3×3，步幅1，填充1）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

# ─────────────────────────────────────────────────────────
# 手动卷积计算，逐步打印每一步
# ─────────────────────────────────────────────────────────

def manual_conv2d_step_by_step():
    """
    用具体数字演示卷积的每一步计算

    输入:  [1, 1, 5, 5]  (batch=1, channel=1, H=5, W=5)
    卷积核: [1, 1, 3, 3]  (out_ch=1, in_ch=1, kH=3, kW=3)
    步幅=1, 填充=0
    输出:  [1, 1, 3, 3]  → (5-3+0*2)/1+1 = 3
    """
    # 5×5 输入（简单数字，便于手算）
    x = torch.tensor([[[[1., 2., 3., 0., 1.],
                         [4., 5., 6., 1., 0.],
                         [7., 8., 9., 2., 1.],
                         [0., 1., 2., 3., 4.],
                         [1., 0., 1., 2., 3.]]]])
    # 3×3 Sobel 水平边缘检测核
    kernel = torch.tensor([[[[-1., -2., -1.],
                               [ 0.,  0.,  0.],
                               [ 1.,  2.,  1.]]]])

    print("=" * 60)
    print("手动卷积计算演示")
    print("=" * 60)
    print(f"\n输入 x (5×5):\n{x[0,0].numpy()}")
    print(f"\n卷积核 (3×3):\n{kernel[0,0].numpy()}")

    # 手动计算输出的每个位置
    print("\n逐位置计算（步幅=1，填充=0）：")
    print("输出形状 = ((5-3+0*2)/1+1, (5-3+0*2)/1+1) = (3, 3)")
    print()

    output_manual = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            patch = x[0, 0, i:i+3, j:j+3].numpy()
            k = kernel[0, 0].numpy()
            val = (patch * k).sum()
            output_manual[i, j] = val
            print(f"  输出[{i},{j}] = patch[{i}:{i+3},{j}:{j+3}] ⊙ kernel = {val:.1f}")
            if i == 0 and j == 0:
                print(f"    patch:\n{patch}")
                print(f"    逐元素乘:\n{patch * k}")
                print(f"    求和: {val:.1f}")

    print(f"\n手动计算输出:\n{output_manual}")

    # 用 PyTorch 验证
    output_torch = F.conv2d(x, kernel, padding=0, stride=1)
    print(f"\nPyTorch 计算输出:\n{output_torch[0,0].numpy()}")
    print(f"\n误差: {np.abs(output_manual - output_torch[0,0].numpy()).max():.8f}  ✓")

    return x, kernel, output_manual


def output_shape_formula():
    """
    演示输出形状公式：
    H_out = (H_in + 2*padding - kernel_size) / stride + 1
    """
    print("\n" + "=" * 60)
    print("输出形状公式演示")
    print("H_out = (H_in + 2*P - K) / S + 1")
    print("=" * 60)

    configs = [
        # (H_in, K, P, S, 描述)
        (32, 3, 0, 1, "32×32输入，3×3核，无填充，步幅1"),
        (32, 3, 1, 1, "32×32输入，3×3核，填充1，步幅1  ← 保持尺寸"),
        (32, 3, 1, 2, "32×32输入，3×3核，填充1，步幅2  ← 下采样"),
        (32, 5, 2, 1, "32×32输入，5×5核，填充2，步幅1  ← 保持尺寸"),
        (28, 5, 0, 1, "28×28输入，5×5核，无填充，步幅1  ← LeNet"),
        (224, 11, 2, 4, "224×224输入，11×11核，填充2，步幅4  ← AlexNet第一层"),
    ]

    for H_in, K, P, S, desc in configs:
        H_out = (H_in + 2*P - K) // S + 1
        print(f"  {desc}")
        print(f"    ({H_in} + 2×{P} - {K}) / {S} + 1 = {H_out}×{H_out}")
        print()


x, kernel, output_manual = manual_conv2d_step_by_step()
output_shape_formula()
```

## 1.3 三种经典卷积核：边缘检测、模糊、锐化

```python
# ─────────────────────────────────────────────────────────
# 三种经典卷积核的完整实现与效果对比
# ─────────────────────────────────────────────────────────

CLASSIC_KERNELS = {
    # ── 边缘检测 ──────────────────────────────────────────
    'Sobel_水平边缘': {
        'kernel': np.array([[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]]),
        'desc': '检测水平方向的亮度变化（水平边缘）\n原理：上下像素差值，中间行权重为0',
        'cmap': 'RdBu',
    },
    'Sobel_垂直边缘': {
        'kernel': np.array([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]),
        'desc': '检测垂直方向的亮度变化（垂直边缘）\n原理：左右像素差值，中间列权重为0',
        'cmap': 'RdBu',
    },
    'Laplacian_全方向边缘': {
        'kernel': np.array([[ 0., -1.,  0.],
                             [-1.,  4., -1.],
                             [ 0., -1.,  0.]]),
        'desc': '检测所有方向的边缘\n原理：中心像素 - 四邻域均值，二阶导数',
        'cmap': 'RdBu',
    },
    # ── 模糊 ──────────────────────────────────────────────
    '均值模糊': {
        'kernel': np.ones((3, 3)) / 9,
        'desc': '简单平均，去除噪声\n原理：每个像素替换为邻域均值',
        'cmap': 'gray',
    },
    '高斯模糊': {
        'kernel': np.array([[1., 2., 1.],
                             [2., 4., 2.],
                             [1., 2., 1.]]) / 16,
        'desc': '加权平均，中心权重更大\n原理：模拟高斯分布，更自然的模糊',
        'cmap': 'gray',
    },
    # ── 锐化 ──────────────────────────────────────────────
    '锐化': {
        'kernel': np.array([[ 0., -1.,  0.],
                             [-1.,  5., -1.],
                             [ 0., -1.,  0.]]),
        'desc': '增强边缘，使图像更清晰\n原理：原图 + 边缘（Laplacian）',
        'cmap': 'gray',
    },
    '浮雕效果': {
        'kernel': np.array([[-2., -1., 0.],
                             [-1.,  1., 1.],
                             [ 0.,  1., 2.]]),
        'desc': '产生3D浮雕感\n原理：对角方向的差值',
        'cmap': 'gray',
    },
}


def make_test_image(size=64):
    """生成包含各种特征的测试图像"""
    img = np.zeros((size, size), dtype=np.float32)
    # 矩形
    img[8:28, 8:28] = 1.0
    # 圆形
    for i in range(size):
        for j in range(size):
            if (i-48)**2 + (j-48)**2 < 12**2:
                img[i, j] = 0.8
    # 对角线
    for i in range(size):
        img[i, min(i, size-1)] = 1.0
    # 水平线
    img[40:42, :] = 0.9
    # 垂直线
    img[:, 40:42] = 0.7
    # 噪声
    img += np.random.randn(size, size).astype(np.float32) * 0.03
    return np.clip(img, 0, 1)


def apply_and_visualize_kernels():
    """应用所有经典卷积核并可视化效果"""
    np.random.seed(42)
    img = make_test_image(64)
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # [1,1,64,64]

    n_kernels = len(CLASSIC_KERNELS)
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, n_kernels + 1, hspace=0.4, wspace=0.3)

    # 原图
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax_orig.set_title('原始图像\n(64×64)', fontsize=11, fontweight='bold')
    ax_orig.axis('off')

    for col, (name, info) in enumerate(CLASSIC_KERNELS.items()):
        k = info['kernel']
        k_tensor = torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0)
        result = F.conv2d(x, k_tensor, padding=1)[0, 0].numpy()

        # 第一行：卷积核可视化
        ax_k = fig.add_subplot(gs[0, col + 1])
        vmax = max(abs(k.min()), abs(k.max())) + 1e-8
        im = ax_k.imshow(k, cmap='RdBu', vmin=-vmax, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax_k.text(j, i, f'{k[i,j]:.2f}', ha='center', va='center',
                          fontsize=8, fontweight='bold',
                          color='white' if abs(k[i,j]) > vmax*0.5 else 'black')
        ax_k.set_title(name.replace('_', '\n'), fontsize=8, fontweight='bold')
        ax_k.axis('off')

        # 第二行：卷积结果
        ax_r = fig.add_subplot(gs[1, col + 1])
        ax_r.imshow(result, cmap=info['cmap'],
                    vmin=result.min(), vmax=result.max())
        ax_r.set_title('卷积结果', fontsize=7)
        ax_r.axis('off')

        # 第三行：激活值分布
        ax_h = fig.add_subplot(gs[2, col + 1])
        ax_h.hist(result.flatten(), bins=30, color='steelblue',
                  edgecolor='white', alpha=0.8)
        ax_h.set_title(f'分布\n均值={result.mean():.2f}', fontsize=7)
        ax_h.tick_params(labelsize=6)
        ax_h.grid(True, alpha=0.3)

    plt.suptitle('经典卷积核效果对比（第一行=核，第二行=结果，第三行=分布）',
                 fontsize=13, fontweight='bold')
    plt.savefig('classic_kernels.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印每个核的详细说明
    print("\n各卷积核说明：")
    for name, info in CLASSIC_KERNELS.items():
        print(f"\n{name}:")
        print(f"  核:\n{info['kernel']}")
        print(f"  原理: {info['desc']}")


apply_and_visualize_kernels()
```

## 1.4 感受野（Receptive Field）可视化

```python
def visualize_receptive_field():
    """
    可视化不同深度下的感受野大小

    感受野公式（每层 3×3 卷积，步幅1）：
    第1层后：3×3
    第2层后：5×5  （每层增加 2*(kernel-1)/2 = 2）
    第n层后：(2n+1)×(2n+1)

    两个 3×3 = 一个 5×5 的感受野，但参数更少：
    2×(3×3) = 18 参数  vs  5×5 = 25 参数
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    input_size = 13
    for ax_idx, n_layers in enumerate([1, 2, 3, 4]):
        rf_size = 2 * n_layers + 1
        grid = np.zeros((input_size, input_size))
        center = input_size // 2
        half = rf_size // 2
        r_start = max(0, center - half)
        r_end   = min(input_size, center + half + 1)
        c_start = max(0, center - half)
        c_end   = min(input_size, center + half + 1)
        grid[r_start:r_end, c_start:c_end] = 0.4
        grid[center, center] = 1.0

        axes[ax_idx].imshow(grid, cmap='Blues', vmin=0, vmax=1)
        axes[ax_idx].set_title(f'{n_layers}层 3×3 卷积\n感受野={rf_size}×{rf_size}',
                                fontsize=11, fontweight='bold')
        for i in range(input_size):
            for j in range(input_size):
                if grid[i, j] > 0:
                    axes[ax_idx].text(j, i, '●' if grid[i,j]==1 else '·',
                                      ha='center', va='center', fontsize=8,
                                      color='white' if grid[i,j] > 0.5 else 'steelblue')
        axes[ax_idx].set_xticks([]); axes[ax_idx].set_yticks([])

    plt.suptitle('感受野随网络深度的增长\n（蓝色区域=当前输出位置能"看到"的输入范围）',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('receptive_field.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("感受野大小 vs 层数（3×3卷积，步幅1）：")
    for n in range(1, 8):
        rf = 2*n + 1
        params_stack = n * 9
        params_single = rf * rf
        print(f"  {n}层: 感受野={rf}×{rf}={rf**2}像素  "
              f"堆叠参数={params_stack}  等效单核参数={params_single}  "
              f"节省={params_single-params_stack}个参数")


visualize_receptive_field()
```

## 1.5 池化操作详解

```python
def demonstrate_pooling():
    """演示 MaxPool 和 AvgPool 的计算过程"""
    x = torch.tensor([[[[1., 3., 2., 4.],
                         [5., 6., 7., 8.],
                         [3., 2., 1., 0.],
                         [9., 4., 3., 2.]]]])
    print("输入 (4×4):")
    print(x[0, 0].numpy())

    max_pool = F.max_pool2d(x, kernel_size=2, stride=2)
    avg_pool = F.avg_pool2d(x, kernel_size=2, stride=2)

    print(f"\nMaxPool2d(2×2, stride=2) → 输出形状: {tuple(max_pool.shape)}")
    print(max_pool[0, 0].numpy())
    print("原理：每个2×2区域取最大值")
    print("  左上[1,3,5,6]→6  右上[2,4,7,8]→8  左下[3,2,9,4]→9  右下[1,0,3,2]→3")

    print(f"\nAvgPool2d(2×2, stride=2) → 输出形状: {tuple(avg_pool.shape)}")
    print(avg_pool[0, 0].numpy())
    print("原理：每个2×2区域取平均值")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[0, 0].numpy(), cmap='Blues', vmin=0, vmax=9)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{x[0,0,i,j]:.0f}', ha='center', va='center',
                         fontsize=14, fontweight='bold')
    axes[0].set_title('输入 (4×4)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(max_pool[0, 0].numpy(), cmap='Reds', vmin=0, vmax=9)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{max_pool[0,0,i,j]:.0f}', ha='center', va='center',
                         fontsize=18, fontweight='bold', color='white')
    axes[1].set_title('MaxPool (2×2)\n取最大值', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(avg_pool[0, 0].numpy(), cmap='Greens', vmin=0, vmax=9)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f'{avg_pool[0,0,i,j]:.1f}', ha='center', va='center',
                         fontsize=18, fontweight='bold', color='white')
    axes[2].set_title('AvgPool (2×2)\n取平均值', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('池化操作对比', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pooling_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


demonstrate_pooling()
```

## 1.6 CNN 10个常见调试问题

```python
CNN_DEBUG_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║              CNN 10 大常见调试问题                                    ║
╚══════════════════════════════════════════════════════════════════════╝

问题1：特征图全黑（激活值全为0）
  现象：可视化特征图时，某些通道完全是黑色
  原因：① ReLU 死亡（Dead ReLU）：权重初始化不当导致所有输入为负
        ② 学习率过大导致权重爆炸后崩溃
        ③ BatchNorm 参数异常
  解决：① 换用 LeakyReLU 或 GELU
        ② 使用 He 初始化（nn.init.kaiming_normal_）
        ③ 检查 BN 的 running_mean/running_var 是否正常
  诊断代码：
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # 注册 hook 检查死亡比例
            pass

问题2：训练损失不下降
  现象：loss 从第一个 epoch 就不动，或在某个值卡住
  原因：① 学习率太小（<1e-6）或太大（>0.1）
        ② 数据归一化问题（像素值0-255未归一化到0-1）
        ③ 标签错误（one-hot vs 整数索引）
        ④ 模型输出维度与类别数不匹配
  解决：① 用 LR Finder 找最优学习率
        ② 确保输入归一化：transforms.Normalize(mean, std)
        ③ CrossEntropyLoss 需要整数标签，BCELoss 需要 float
  快速检查：
    print(model(x[:1]).shape)  # 检查输出形状
    print(y[:5])               # 检查标签格式

问题3：过拟合（训练准确率高，验证准确率低）
  现象：训练集 99%，验证集 60%，差距随训练增大
  原因：模型容量过大，数据量不足
  解决：① 添加 Dropout（卷积后用 nn.Dropout2d，全连接后用 nn.Dropout）
        ② 数据增强（随机翻转、裁剪、颜色抖动）
        ③ 减小模型（减少通道数或层数）
        ④ L2 正则化（optimizer weight_decay=1e-4）
        ⑤ 早停（patience=10）

问题4：梯度消失（深层网络训练不动）
  现象：浅层的梯度范数接近0，只有最后几层在更新
  原因：① 没有残差连接
        ② 使用 Sigmoid/Tanh 激活（饱和区梯度≈0）
        ③ 权重初始化不当
  解决：① 使用残差连接（ResNet 风格）
        ② 换用 ReLU/GELU
        ③ 使用 He 初始化
  诊断：
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f'{name}: grad_norm={p.grad.norm():.6f}')

问题5：梯度爆炸
  现象：loss 突然变成 NaN，或 loss 剧烈震荡
  原因：学习率过大，或网络太深没有归一化
  解决：① 梯度裁剪：nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ② 降低学习率
        ③ 添加 BatchNorm

问题6：输出形状错误（RuntimeError: size mismatch）
  现象：forward 时报 size mismatch 错误
  原因：全连接层输入维度计算错误
  解决：用 AdaptiveAvgPool2d(1) 替代固定 flatten，
        或先打印中间形状：
    x = torch.randn(1, 3, 32, 32)
    for name, layer in model.named_children():
        x = layer(x)
        print(f'{name}: {tuple(x.shape)}')

问题7：BatchNorm 在推理时表现差
  现象：训练时准确率高，model.eval() 后准确率骤降
  原因：① 忘记调用 model.eval()（BN 会用 batch 统计而非 running 统计）
        ② 训练 batch 太小（<8），running 统计不稳定
        ③ 训练和测试数据分布差异大
  解决：① 推理前务必 model.eval()
        ② batch_size >= 16
        ③ 考虑用 GroupNorm 替代 BatchNorm

问题8：数据增强导致验证集泄露
  现象：验证集准确率异常高，但测试集很低
  原因：对验证集也做了随机增强
  解决：
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),  # 验证集只做基础变换！
    ])

问题9：类别不平衡导致模型偏向多数类
  现象：模型总是预测同一个类别，准确率虚高
  原因：数据集中某类样本远多于其他类
  解决：
    # 方法1：加权损失
    weights = torch.tensor([1.0, 10.0, 5.0])  # 少数类权重更大
    criterion = nn.CrossEntropyLoss(weight=weights)
    # 方法2：过采样少数类（WeightedRandomSampler）

问题10：卷积核学不到有意义的特征
  现象：可视化卷积核，全是随机噪声，没有边缘/纹理模式
  原因：① 训练轮数不够
        ② 学习率太小
        ③ 数据量不足
        ④ 模型太深，浅层梯度太小
  解决：① 增加训练轮数
        ② 使用预训练权重（迁移学习）
        ③ 检查梯度流（见问题4）
"""

print(CNN_DEBUG_GUIDE)
```

## 小结

| 概念 | 公式 | 关键点 |
|------|------|--------|
| 输出尺寸 | (H+2P-K)/S+1 | 填充P=K//2时尺寸不变 |
| 感受野 | 2n+1（n层3×3） | 深度换感受野，参数更少 |
| 边缘检测 | Sobel/Laplacian | 差分运算，检测亮度变化 |
| 模糊 | 均值/高斯 | 平均运算，去除噪声 |
| 锐化 | 原图+边缘 | 增强高频信息 |

## 内容可信度与来源

**可信度：已校对。** 本节关于卷积滑窗、卷积核、stride、padding、输出尺寸和浅层滤波器直觉的说明，已对照 CS231n、PyTorch `Conv2d` 文档、D2L 与早期 CNN 论文来源检查。页面中的固定 Sobel/Laplacian 核用于建立图像处理直觉，不代表训练后的 CNN 一定学到完全相同的核。

参考来源：

- CS231n Convolutional Neural Networks: https://cs231n.github.io/convolutional-networks/
- PyTorch `torch.nn.Conv2d`: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- Dive into Deep Learning, convolution chapters: https://d2l.ai/chapter_convolutional-neural-networks/index.html
- LeCun et al., Gradient-Based Learning Applied to Document Recognition: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

边界说明：

- 输出尺寸公式以 PyTorch Conv2d 与 CS231n 的卷积定义为准；不同框架在 padding 字符串、ceil/floor 策略上可能有接口差异。
- 本节图像尺寸很小，是为了让滑窗过程看得清楚；真实 CNN 通常在多通道、高分辨率和多层结构中运行。
- 固定滤波器只能说明“局部模式匹配”的思想，训练出的卷积核会随数据、任务和优化过程变化。
