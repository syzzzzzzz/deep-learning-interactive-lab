"""
自动生成自: part2_cnn\07_advanced_convolution.md
可独立运行的 Python 源码
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

MODULE_TITLE = "高级卷积"
MODULE_SUMMARY = "用空洞卷积、转置卷积、棋盘伪影、分组卷积和可变形卷积解释 CNN 如何改变感受野、上采样方式和采样位置。"
MODULE_TAGS = ["CNN", "空洞卷积", "转置卷积", "分组卷积", "可变形卷积", "上采样"]
MODULE_RELATED_TOPICS = ["part2/04_debug_panel", "part2/06_modern_architectures", "part2/08_visualization_gradcam", "part5/01_feature_visualization"]
PRACTICE_TARGET = "调整空洞率、分组数、上采样方式和偏移强度，解释感受野、参数量、棋盘伪影和采样点为什么变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


def demonstrate_dilated_conv():
    """
    对比不同空洞率的卷积效果
    """
    # 生成测试图像：同心圆
    size = 64
    img = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - size // 2) ** 2 + (j - size // 2) ** 2)
            img[i, j] = np.sin(dist * 0.5)

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    # 同一个 3×3 边缘检测核，不同空洞率
    kernel = torch.tensor([[[[-1., -1., -1.],
                              [-1.,  8., -1.],
                              [-1., -1., -1.]]]])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原图', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    for idx, dilation in enumerate([1, 2, 3, 5]):
        # 空洞卷积：padding 需要跟着调整
        padding = dilation  # 保持输出尺寸不变
        out = F.conv2d(x, kernel, padding=padding, dilation=dilation)

        rf = 2 * dilation + 1  # 感受野大小
        axes[idx + 1].imshow(out[0, 0].numpy(), cmap='RdBu')
        axes[idx + 1].set_title(f'dilation={dilation}\n感受野={rf}×{rf}',
                                 fontsize=11, fontweight='bold')
        axes[idx + 1].axis('off')

    plt.suptitle('空洞卷积：不增加参数，扩大感受野', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dilated_conv.png', dpi=150)
    plt.show()


# demonstrate_dilated_conv()  # 协议化后由 compute_advanced_convolution() 控制执行

# ============================================================
# 代码段 2
# ============================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    DeepLabV3 的核心模块
    并行多个不同空洞率的卷积，捕获多尺度上下文
    """
    def __init__(self, in_ch, out_ch=256, rates=[1, 6, 12, 18]):
        super().__init__()
        self.branches = nn.ModuleList()

        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        # 全局池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 融合
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        res = [branch(x) for branch in self.branches]
        res.append(F.interpolate(self.global_pool(x), size=(h, w), mode='bilinear'))
        return self.project(torch.cat(res, dim=1))

# ============================================================
# 代码段 3
# ============================================================

def visualize_transposed_conv():
    """可视化转置卷积的上采样过程"""
    # 输入 2×2 → 转置卷积 → 输出 4×4
    x = torch.tensor([[[[1., 2.],
                         [3., 4.]]]])  # [1, 1, 2, 2]

    kernel = torch.tensor([[[[1., 0.],
                              [0., 1.]]]])  # 2×2 核

    # 转置卷积：stride=2, 无 padding
    out = F.conv_transpose2d(x, kernel, stride=2, padding=0)
    # 输出尺寸: (2-1)*2 - 0 + 2 + 0 = 4

    print(f"输入形状: {tuple(x.shape)}")
    print(f"输入:\n{x[0,0].numpy()}")
    print(f"\n核:\n{kernel[0,0].numpy()}")
    print(f"\n输出形状: {tuple(out.shape)}")
    print(f"输出:\n{out[0,0].numpy()}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(x[0, 0].numpy(), cmap='Blues', vmin=0, vmax=5)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{x[0,0,i,j]:.0f}', ha='center', va='center',
                         fontsize=14, fontweight='bold')
    axes[0].set_title('输入 (2×2)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(kernel[0, 0].numpy(), cmap='Oranges', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{kernel[0,0,i,j]:.0f}', ha='center', va='center',
                         fontsize=14, fontweight='bold')
    axes[1].set_title('转置卷积核 (2×2)\nstride=2', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(out[0, 0].numpy(), cmap='Greens', vmin=0, vmax=5)
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f'{out[0,0,i,j]:.0f}', ha='center', va='center',
                         fontsize=11, fontweight='bold', color='white' if out[0,0,i,j] > 2 else 'black')
    axes[2].set_title('输出 (4×4)\n上采样 2×', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('转置卷积：2×2 → 4×4', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('transposed_conv.png', dpi=150)
    plt.show()


# visualize_transposed_conv()  # 协议化后由 compute_advanced_convolution() 控制执行

# ============================================================
# 代码段 4
# ============================================================

def demonstrate_checkerboard_artifacts():
    """
    转置卷积的棋盘效应

    问题：转置卷积会产生重叠，导致棋盘状伪影
    解决：先上采样（最近邻/双线性），再卷积
    """
    x = torch.randn(1, 1, 4, 4)

    # 转置卷积（有棋盘效应）
    conv_t = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
    with torch.no_grad():
        out_conv_t = conv_t(x)

    # 上采样 + 卷积（无棋盘效应）
    upsample_conv = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(1, 1, 3, padding=1),
    )
    with torch.no_grad():
        out_up_conv = upsample_conv(x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(x[0, 0].detach().numpy(), cmap='gray')
    axes[0].set_title('输入 (4×4)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(out_conv_t[0, 0].detach().numpy(), cmap='gray')
    axes[1].set_title('转置卷积 (8×8)\n注意棋盘效应', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(out_up_conv[0, 0].detach().numpy(), cmap='gray')
    axes[2].set_title('上采样+卷积 (8×8)\n无棋盘效应', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('棋盘效应对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('checkerboard.png', dpi=150)
    plt.show()


# demonstrate_checkerboard_artifacts()  # 协议化后由 compute_advanced_convolution() 控制执行

# ============================================================
# 代码段 5
# ============================================================

def visualize_group_conv_types():
    """
    可视化三种卷积类型的连接模式

    标准卷积:   每个输出通道看所有输入通道
    分组卷积:   输入通道分组，每组独立卷积
    深度卷积:   分组数=通道数（每组1个通道）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    C_in, C_out = 4, 4

    # 标准卷积：全连接
    ax = axes[0]
    for i in range(C_out):
        for j in range(C_in):
            ax.plot([j, i], [0, 1], 'b-', alpha=0.3, linewidth=1)
    for j in range(C_in):
        ax.plot(j, 0, 'o', color='steelblue', markersize=15)
    for i in range(C_out):
        ax.plot(i, 1, 'o', color='indianred', markersize=15)
    ax.set_title('标准卷积\n每个输出看所有输入', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.3, 1.3)
    ax.text(-0.5, 0, '输入通道', fontsize=10, ha='right')
    ax.text(-0.5, 1, '输出通道', fontsize=10, ha='right')

    # 分组卷积 (groups=2)
    ax = axes[1]
    groups = 2
    for g in range(groups):
        start_in = g * (C_in // groups)
        end_in = (g + 1) * (C_in // groups)
        start_out = g * (C_out // groups)
        end_out = (g + 1) * (C_out // groups)
        for i in range(start_out, end_out):
            for j in range(start_in, end_in):
                ax.plot([j, i], [0, 1], 'b-', alpha=0.5, linewidth=1.5)
    for j in range(C_in):
        ax.plot(j, 0, 'o', color='steelblue', markersize=15)
    for i in range(C_out):
        ax.plot(i, 1, 'o', color='indianred', markersize=15)
    ax.set_title('分组卷积 (groups=2)\n每组独立卷积', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.3, 1.3)

    # 深度卷积 (groups=C_in)
    ax = axes[2]
    for i in range(C_out):
        ax.plot([i, i], [0, 1], 'b-', alpha=0.8, linewidth=2)
    for j in range(C_in):
        ax.plot(j, 0, 'o', color='steelblue', markersize=15)
    for i in range(C_out):
        ax.plot(i, 1, 'o', color='indianred', markersize=15)
    ax.set_title('深度卷积 (groups=C_in)\n每个通道独立', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.3, 1.3)

    plt.suptitle('卷积连接模式对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('group_conv_types.png', dpi=150)
    plt.show()


# visualize_group_conv_types()  # 协议化后由 compute_advanced_convolution() 控制执行

# ============================================================
# 代码段 6
# ============================================================

class ResNeXtBlock(nn.Module):
    """
    ResNeXt 块：分组卷积 + 残差

    核心思想：增加"基数"（分组数）比增加深度/宽度更有效
    原论文：101层 ResNeXt > 200层 ResNet
    """
    def __init__(self, in_ch, out_ch, stride=1, groups=32):
        super().__init__()
        mid_ch = out_ch // 2

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # 分组卷积：groups 参数是关键
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch, 3, stride=stride, padding=1,
            groups=groups, bias=False  # 32 个分组
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + residual)


def compare_group_conv_cost():
    """对比不同 groups 的参数量"""
    in_ch, out_ch, kernel_size = 64, 128, 3

    print("分组卷积参数量对比")
    print("=" * 50)
    print(f"{'groups':>6s} {'参数量':>10s} {'压缩比':>8s}")
    print("-" * 50)

    base_params = in_ch * out_ch * kernel_size * kernel_size
    for g in [1, 2, 4, 8, 16, 32, 64]:
        params = (in_ch // g) * (out_ch // g) * kernel_size * kernel_size * g
        ratio = base_params / params
        print(f"{g:6d} {params:10,d} {ratio:8.1f}x")


# compare_group_conv_cost()  # 协议化后由 compute_advanced_convolution() 控制执行

# ============================================================
# 代码段 7
# ============================================================

class DeformableConv2d(nn.Module):
    """
    可变形卷积简化实现

    原理：
    1. 额外卷积层预测每个采样点的 (dx, dy) 偏移
    2. 使用 grid_sample 在偏移位置采样
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        # 偏移量预测：2 * K * K 个输出 (每个采样点的 dx, dy)
        self.offset_conv = nn.Conv2d(
            in_ch, 2 * kernel_size * kernel_size,
            kernel_size, padding=padding
        )
        # 初始偏移为零
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # 主卷积
        self.regular_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, padding=padding
        )

    def forward(self, x):
        # 预测偏移量
        offsets = self.offset_conv(x)  # [B, 2*K*K, H, W]

        # 生成标准网格
        B, _, H, W = x.shape
        K = self.kernel_size

        # 标准采样位置
        y_std = torch.arange(-K // 2 + 1, K // 2 + 1, device=x.device)
        x_std = torch.arange(-K // 2 + 1, K // 2 + 1, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_std, x_std, indexing='ij')
        y_grid = y_grid.flatten()  # [K*K]
        x_grid = x_grid.flatten()  # [K*K]

        # 像素坐标
        y_pix = torch.arange(0, H, device=x.device, dtype=torch.float32)
        x_pix = torch.arange(0, W, device=x.device, dtype=torch.float32)
        y_pix, x_pix = torch.meshgrid(y_pix, x_pix, indexing='ij')

        # 构建采样网格
        # 简化版本：只加偏移到标准位置
        grid_y = y_pix.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        grid_x = x_pix.unsqueeze(0).expand(B, -1, -1)

        # 加上偏移（取第一个采样点做演示）
        dy = offsets[:, 0, :, :]  # [B, H, W]
        dx = offsets[:, 1, :, :]

        grid_y = grid_y + dy
        grid_x = grid_x + dx

        # 归一化到 [-1, 1]（grid_sample 要求）
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        grid_x = 2.0 * grid_x / (W - 1) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

        # 使用 grid_sample 采样
        x_sampled = F.grid_sample(x, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

        # 在采样位置上做标准卷积
        return self.regular_conv(x_sampled)


def demonstrate_deformable_conv():
    """可视化可变形卷积的采样位置"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    K = 3
    # 标准卷积采样
    y = np.arange(-K // 2 + 1, K // 2 + 1)
    x = np.arange(-K // 2 + 1, K // 2 + 1)
    yy, xx = np.meshgrid(y, x, indexing='ij')

    axes[0].scatter(xx.flatten(), -yy.flatten(), c='steelblue', s=200, zorder=5)
    axes[0].scatter(0, 0, c='red', s=300, zorder=6, marker='*')
    axes[0].set_title('标准卷积采样\n固定 3×3 网格', fontsize=12, fontweight='bold')
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # 可变形卷积采样（学习后偏移）
    np.random.seed(42)
    dx = np.random.randn(K * K) * 0.3
    dy = np.random.randn(K * K) * 0.3

    axes[1].scatter((xx.flatten() + dx), -(yy.flatten() + dy),
                     c='indianred', s=200, zorder=5)
    # 连线：原始 → 偏移
    for i in range(K * K):
        axes[1].plot([xx.flatten()[i], xx.flatten()[i] + dx[i]],
                      [-yy.flatten()[i], -(yy.flatten()[i] + dy[i])],
                      'k-', alpha=0.3)
    axes[1].scatter(0, 0, c='red', s=300, zorder=6, marker='*')
    axes[1].set_title('可变形卷积采样\n偏移随物体形状变化', fontsize=12, fontweight='bold')
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    # 应用示意：圆形物体
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 1.5
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)
    axes[2].plot(circle_x, circle_y, 'b-', linewidth=2)

    # 可变形卷积采样沿圆弧分布
    n_points = 9
    theta_pts = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pt_x = r * np.cos(theta_pts)
    pt_y = r * np.sin(theta_pts)
    axes[2].scatter(pt_x, pt_y, c='indianred', s=200, zorder=5)
    axes[2].scatter(0, 0, c='red', s=300, zorder=6, marker='*')
    axes[2].set_title('适应物体形状\n采样点沿圆弧分布', fontsize=12, fontweight='bold')
    axes[2].set_xlim(-2.5, 2.5)
    axes[2].set_ylim(-2.5, 2.5)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('可变形卷积：自适应采样', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deformable_conv.png', dpi=150)
    plt.show()


# demonstrate_deformable_conv()  # 协议化后由 compute_advanced_convolution() 控制执行

# ============================================================
# 代码段 8
# ============================================================

def compare_advanced_conv_params():
    """对比高级卷积的参数量和感受野"""
    in_ch, out_ch = 64, 128

    results = {
        '标准 3×3': {
            'params': in_ch * out_ch * 9,
            'rf': 3,
            '特点': '基准'
        },
        '空洞 3×3 (d=2)': {
            'params': in_ch * out_ch * 9,
            'rf': 5,
            '特点': '同样参数，更大感受野'
        },
        '深度可分离': {
            'params': in_ch * 9 + in_ch * out_ch,
            'rf': 3,
            '特点': '参数 ~1/9'
        },
        '分组 (g=4)': {
            'params': in_ch * out_ch * 9 // 4,
            'rf': 3,
            '特点': '参数 1/4'
        },
        '可变形 3×3': {
            'params': in_ch * out_ch * 9 + in_ch * 18,
            'rf': 3,
            '特点': '额外偏移参数'
        },
    }

    print("高级卷积技术对比")
    print("=" * 70)
    print(f"{'类型':20s} {'参数量':>10s} {'感受野':>6s} {'特点':25s}")
    print("-" * 70)
    for name, info in results.items():
        print(f"{name:20s} {info['params']:10,d} {info['rf']:6d} {info['特点']:25s}")


# compare_advanced_conv_params()  # 协议化后由 compute_advanced_convolution() 控制执行


def _make_ring_image(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    image = np.sin(dist * 0.45)
    return image.astype(np.float32)


def _plot_dilated_conv(dilation: int, image_size: int) -> tuple[object, dict[str, int]]:
    image = _make_ring_image(image_size)
    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]])
    rates = sorted(set([1, 2, dilation, max(dilation + 1, 3)]))[:4]
    with safe_mpl_figure(figsize=(11, 3.6)) as fig:
        axes = fig.subplots(1, len(rates) + 1)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("输入纹理", fontsize=9, fontweight="bold")
        axes[0].axis("off")
        for ax, rate in zip(axes[1:], rates):
            out = F.conv2d(x, kernel, padding=rate, dilation=rate)[0, 0].numpy()
            ax.imshow(out, cmap="RdBu")
            ax.set_title(f"d={rate}\n感受野 {2 * rate + 1}x{2 * rate + 1}", fontsize=8, fontweight="bold")
            ax.axis("off")
        fig.suptitle("空洞卷积：不增加参数，扩大感受野", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, {"effective_receptive_field": 2 * dilation + 1}


def _plot_transposed_and_checkerboard(stride: int, seed: int) -> tuple[object, dict[str, tuple[int, ...]]]:
    torch.manual_seed(seed)
    stride = clamp_int(stride, 1, 4, "转置卷积 stride")
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    kernel = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    transposed = F.conv_transpose2d(x, kernel, stride=stride)
    noisy = torch.randn(1, 1, 4, 4)
    conv_t = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
    upsample_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(1, 1, 3, padding=1))
    with torch.no_grad():
        checker = conv_t(noisy)[0, 0].numpy()
        smoother = upsample_conv(noisy)[0, 0].numpy()
    with safe_mpl_figure(figsize=(11, 4.1)) as fig:
        axes = fig.subplots(1, 4)
        axes[0].imshow(x[0, 0].numpy(), cmap="Blues")
        axes[0].set_title("输入 2x2", fontsize=9, fontweight="bold")
        axes[1].imshow(transposed[0, 0].numpy(), cmap="Greens")
        axes[1].set_title(f"转置卷积\nstride={stride}", fontsize=9, fontweight="bold")
        axes[2].imshow(checker, cmap="gray")
        axes[2].set_title("转置卷积伪影", fontsize=9, fontweight="bold")
        axes[3].imshow(smoother, cmap="gray")
        axes[3].set_title("上采样+卷积", fontsize=9, fontweight="bold")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("上采样调试：转置卷积为什么可能出现棋盘纹", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, {"transposed_shape": tuple(transposed.shape), "checker_shape": checker.shape}


def _plot_group_conv(groups: int, in_channels: int, out_channels: int) -> tuple[object, dict[str, float]]:
    groups = clamp_int(groups, 1, min(in_channels, out_channels), "分组数")
    standard = in_channels * out_channels * 9
    grouped = standard / groups
    depthwise = in_channels * 9 + in_channels * out_channels
    with safe_mpl_figure(figsize=(10, 4.1)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.bar(["标准", f"分组 g={groups}", "深度可分离"], [standard, grouped, depthwise], color=["#bf3f5b", "#00f0ff", "#00ff88"], alpha=0.85)
        ax1.set_title("参数量对比", fontsize=10, fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.25)
        connection = np.zeros((out_channels, in_channels))
        for group in range(groups):
            in_start = group * in_channels // groups
            in_end = (group + 1) * in_channels // groups
            out_start = group * out_channels // groups
            out_end = (group + 1) * out_channels // groups
            connection[out_start:out_end, in_start:in_end] = 1
        ax2.imshow(connection, cmap="Greens", aspect="auto")
        ax2.set_xlabel("输入通道")
        ax2.set_ylabel("输出通道")
        ax2.set_title("通道连接模式", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig, {"grouped_params": float(grouped), "standard_params": float(standard), "group_compression": float(standard / max(grouped, 1))}


def _plot_deformable(offset_strength: float, seed: int) -> tuple[object, dict[str, float]]:
    rng = np.random.default_rng(seed)
    k = 3
    coords = np.arange(-k // 2 + 1, k // 2 + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dx = rng.normal(0, offset_strength, k * k)
    dy = rng.normal(0, offset_strength, k * k)
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1.5
    with safe_mpl_figure(figsize=(11, 3.8)) as fig:
        axes = fig.subplots(1, 3)
        axes[0].scatter(xx.flatten(), -yy.flatten(), c="#00f0ff", s=130)
        axes[0].scatter(0, 0, c="#bf3f5b", s=220, marker="*")
        axes[0].set_title("标准固定采样", fontsize=9, fontweight="bold")
        axes[1].scatter(xx.flatten() + dx, -(yy.flatten() + dy), c="#b000ff", s=130)
        for i in range(k * k):
            axes[1].plot([xx.flatten()[i], xx.flatten()[i] + dx[i]], [-yy.flatten()[i], -(yy.flatten()[i] + dy[i])], color="black", alpha=0.25)
        axes[1].set_title("可变形偏移采样", fontsize=9, fontweight="bold")
        axes[2].plot(radius * np.cos(theta), radius * np.sin(theta), color="#00f0ff", linewidth=2)
        theta_pts = np.linspace(0, 2 * np.pi, k * k, endpoint=False)
        axes[2].scatter(radius * np.cos(theta_pts), radius * np.sin(theta_pts), c="#b000ff", s=130)
        axes[2].set_title("贴合物体形状", fontsize=9, fontweight="bold")
        for ax in axes:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.25)
        fig.suptitle("可变形卷积：采样点从固定网格变成可学习位置", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, {"offset_mean": float(np.sqrt(dx**2 + dy**2).mean())}


def _advanced_param_table(in_channels: int, out_channels: int, groups: int, dilation: int) -> dict[str, dict[str, object]]:
    return {
        "标准 3x3": {"params": in_channels * out_channels * 9, "rf": 3, "特点": "基准"},
        f"空洞 3x3 d={dilation}": {"params": in_channels * out_channels * 9, "rf": 2 * dilation + 1, "特点": "同参数扩大感受野"},
        "深度可分离": {"params": in_channels * 9 + in_channels * out_channels, "rf": 3, "特点": "移动端轻量"},
        f"分组 g={groups}": {"params": in_channels * out_channels * 9 // max(groups, 1), "rf": 3, "特点": "减少通道连接"},
        "可变形 3x3": {"params": in_channels * out_channels * 9 + in_channels * 18, "rf": 3, "特点": "额外学习偏移"},
    }


def compute_advanced_convolution(
    dilation: int = 3,
    transpose_stride: int = 2,
    groups: int = 4,
    in_channels: int = 64,
    out_channels: int = 128,
    offset_strength: float = 0.3,
    image_size: int = 64,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute advanced convolution visuals without top-level plotting."""

    dilation = clamp_int(dilation, 1, 8, "空洞率")
    transpose_stride = clamp_int(transpose_stride, 1, 4, "转置卷积 stride")
    in_channels = clamp_int(in_channels, 4, 256, "输入通道")
    out_channels = clamp_int(out_channels, 4, 512, "输出通道")
    groups = clamp_int(groups, 1, min(in_channels, out_channels), "分组数")
    offset_strength = clamp_float(offset_strength, 0.0, 1.2, "偏移强度")
    image_size = clamp_int(image_size, 32, 96, "图像尺寸")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dilated_fig, dilated_stats = _plot_dilated_conv(dilation, image_size)
    transpose_fig, transpose_stats = _plot_transposed_and_checkerboard(transpose_stride, seed)
    group_fig, group_stats = _plot_group_conv(groups, in_channels, out_channels)
    deform_fig, deform_stats = _plot_deformable(offset_strength, seed)
    table = _advanced_param_table(in_channels, out_channels, groups, dilation)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("高级卷积协议化计算")
        print(f"dilation={dilation}, transpose_stride={transpose_stride}, groups={groups}, offset_strength={offset_strength:.2f}")
        for name, info in table.items():
            print(f"{name}: 参数={info['params']:,}, 感受野={info['rf']}, 特点={info['特点']}")
        print(f"分组卷积压缩比={group_stats['group_compression']:.2f}x, 偏移平均距离={deform_stats['offset_mean']:.3f}")
    figures = [
        ("advanced_dilated_conv.png", dilated_fig),
        ("advanced_transposed_checkerboard.png", transpose_fig),
        ("advanced_group_conv.png", group_fig),
        ("advanced_deformable_conv.png", deform_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {**dilated_stats, **transpose_stats, **group_stats, **deform_stats}
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "table": table, "log": log_buffer.getvalue()}


def _go_to_gradcam() -> None:
    import streamlit as st

    st.query_params["module"] = "part2_cnn/08_visualization_gradcam"
    st.rerun()


def render() -> None:
    """Render the advanced convolution lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_convolution_particle_flow, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成高级卷积演示：空洞、上采样、分组、可变形采样")
        with st.sidebar:
            dilation = st.slider("空洞率 dilation", 1, 8, 3, 1)
            transpose_stride = st.slider("转置卷积 stride", 1, 4, 2, 1)
            groups = st.slider("分组数 groups", 1, 16, 4, 1)
            in_channels = st.slider("输入通道", 4, 256, 64, 4)
            out_channels = st.slider("输出通道", 4, 512, 128, 4)
            offset_strength = st.slider("偏移强度", 0.0, 1.2, 0.3, 0.05)
            image_size = st.slider("图像尺寸", 32, 96, 64, 8)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("继续看：Grad-CAM 可视化", width="stretch"):
                _go_to_gradcam()
        data = compute_advanced_convolution(dilation, transpose_stride, groups, in_channels, out_channels, offset_strength, image_size, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_convolution_particle_flow()
        st.markdown(
            """
            **零基础直觉：**高级卷积不是把普通卷积复杂化，而是在问四个工程问题：
            如何看得更远？如何把小图放大？如何减少通道连接？如何让采样点贴合物体形状？
            """
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("有效感受野", f"{stats['effective_receptive_field']}x{stats['effective_receptive_field']}")
        c2.metric("分组压缩比", f"{stats['group_compression']:.1f}x")
        c3.metric("转置输出形状", str(stats["transposed_shape"]))
        c4.metric("平均偏移", f"{stats['offset_mean']:.2f}")
        explainers = [
            ("空洞卷积", "空洞率越大，同一个 3x3 核覆盖越远，但采样会变稀，可能漏掉细节。"),
            ("转置卷积与棋盘伪影", "转置卷积可以学习上采样，但重叠不均匀时容易出现棋盘纹；上采样+卷积更稳。"),
            ("分组卷积", "分组越多，通道连接越少，参数越少；但组之间交流也变弱，需要 1x1 卷积补通道混合。"),
            ("可变形卷积", "采样点不再固定在方格上，而是学习偏移，适合物体形状弯曲或姿态变化大的场景。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请只调整对应控件，观察图中最先变化的位置。思考：这是感受野、参数量、上采样重叠，还是采样位置在变？")
        with st.expander("参数表与控制台输出", expanded=False):
            st.dataframe(data["table"], width="stretch")
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/07_advanced_convolution.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_advanced_convolution(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_advanced_convolution(dilation=2, transpose_stride=2, groups=2, in_channels=8, out_channels=16, image_size=32, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["effective_receptive_field"] == 5 and data["stats"]["group_compression"] > 1


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_advanced_convolution))
