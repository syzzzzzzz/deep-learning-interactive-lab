"""
自动生成自: part2_cnn\04_debug_panel.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from typing import Dict, Optional

# ─────────────────────────────────────────────────────────
# 预定义经典卷积核
# ─────────────────────────────────────────────────────────

KERNELS: Dict[str, np.ndarray] = {
    '恒等（不变）': np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=np.float32),

    '均值模糊': np.ones((3, 3), dtype=np.float32) / 9,

    '高斯模糊': np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=np.float32) / 16,

    'Sobel 水平边缘': np.array([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=np.float32),

    'Sobel 垂直边缘': np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=np.float32),

    'Laplacian 锐化': np.array([[ 0, -1,  0],
                                  [-1,  4, -1],
                                  [ 0, -1,  0]], dtype=np.float32),

    '浮雕效果': np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]], dtype=np.float32),

    '水平线检测': np.array([[-1, -1, -1],
                             [ 2,  2,  2],
                             [-1, -1, -1]], dtype=np.float32),

    '垂直线检测': np.array([[-1, 2, -1],
                             [-1, 2, -1],
                             [-1, 2, -1]], dtype=np.float32),

    '对角线检测': np.array([[ 2, -1, -1],
                             [-1,  2, -1],
                             [-1, -1,  2]], dtype=np.float32),
}


def apply_kernel(image: np.ndarray, kernel: np.ndarray,
                 padding: int = 1) -> np.ndarray:
    """
    对图像应用卷积核

    image: [H, W] 灰度图
    kernel: [3, 3] 卷积核
    """
    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    k = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
    out = F.conv2d(x, k, padding=padding)
    return out[0, 0].numpy()


class KernelDebugPanel:
    """
    卷积核交互调试面板

    功能：
    - 从预定义核中选择
    - 手动调整 9 个核参数（滑块）
    - 实时看到卷积结果
    - 对比多个核的效果

    使用方法：
        panel = KernelDebugPanel(image)
        panel.show_all_kernels()          # 静态对比所有核
        panel.interactive()               # 交互式调试（需要 GUI 环境）
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        if image is None:
            # 生成测试图像：带边缘和纹理的合成图
            self.image = self._make_test_image()
        else:
            self.image = image.astype(np.float32)
            if self.image.max() > 1.0:
                self.image = self.image / 255.0

    def _make_test_image(self, size: int = 64) -> np.ndarray:
        """生成包含各种特征的测试图像"""
        img = np.zeros((size, size), dtype=np.float32)
        # 矩形
        img[10:30, 10:30] = 1.0
        # 圆形
        cx, cy, r = 48, 16, 10
        for i in range(size):
            for j in range(size):
                if (i - cy)**2 + (j - cx)**2 < r**2:
                    img[i, j] = 0.8
        # 对角线
        for i in range(size):
            if i < size:
                img[i, i] = 1.0
                if i + 1 < size:
                    img[i, i+1] = 0.5
        # 水平线
        img[45:47, :] = 0.9
        # 垂直线
        img[:, 45:47] = 0.7
        # 添加轻微噪声
        img += np.random.randn(size, size).astype(np.float32) * 0.05
        return np.clip(img, 0, 1)

    def show_all_kernels(self, figsize=(20, 12)):
        """静态显示所有预定义核的效果"""
        n = len(KERNELS)
        n_cols = 5
        n_rows = (n + n_cols - 1) // n_cols

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_cols * 2,
                               hspace=0.4, wspace=0.3)

        for idx, (name, kernel) in enumerate(KERNELS.items()):
            row = idx // n_cols
            col = (idx % n_cols) * 2

            result = apply_kernel(self.image, kernel)

            # 核的可视化
            ax_k = fig.add_subplot(gs[row, col])
            vmax = max(abs(kernel.min()), abs(kernel.max()))
            ax_k.imshow(kernel, cmap='RdBu', vmin=-vmax, vmax=vmax)
            for i in range(3):
                for j in range(3):
                    ax_k.text(j, i, f'{kernel[i,j]:.1f}',
                              ha='center', va='center', fontsize=7,
                              color='black' if abs(kernel[i,j]) < vmax*0.5 else 'white')
            ax_k.set_title(name, fontsize=8, fontweight='bold')
            ax_k.axis('off')

            # 卷积结果
            ax_r = fig.add_subplot(gs[row, col + 1])
            ax_r.imshow(result, cmap='gray', vmin=result.min(), vmax=result.max())
            ax_r.set_title('结果', fontsize=7)
            ax_r.axis('off')

        plt.suptitle('经典卷积核效果对比（左=核，右=结果）',
                     fontsize=14, fontweight='bold')
        plt.savefig('kernel_effects.png', dpi=120, bbox_inches='tight')
        plt.show()

    def show_kernel_detail(self, kernel_name: str):
        """详细展示单个核的效果"""
        if kernel_name not in KERNELS:
            print(f"未知核: {kernel_name}，可用: {list(KERNELS.keys())}")
            return

        kernel = KERNELS[kernel_name]
        result = apply_kernel(self.image, kernel)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 原图
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('原始图像', fontsize=11)
        axes[0].axis('off')

        # 卷积核
        vmax = max(abs(kernel.min()), abs(kernel.max())) + 1e-8
        im = axes[1].imshow(kernel, cmap='RdBu', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=axes[1])
        for i in range(3):
            for j in range(3):
                axes[1].text(j, i, f'{kernel[i,j]:.2f}',
                             ha='center', va='center', fontsize=11,
                             fontweight='bold',
                             color='black' if abs(kernel[i,j]) < vmax*0.5 else 'white')
        axes[1].set_title(f'卷积核: {kernel_name}', fontsize=11)
        axes[1].axis('off')

        # 卷积结果
        axes[2].imshow(result, cmap='gray')
        axes[2].set_title('卷积结果', fontsize=11)
        axes[2].axis('off')

        # 激活值分布
        axes[3].hist(result.flatten(), bins=40, color='steelblue',
                     edgecolor='white', alpha=0.8)
        axes[3].set_title('激活值分布', fontsize=11)
        axes[3].set_xlabel('激活值')
        axes[3].set_ylabel('频次')
        axes[3].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(f'卷积核详细分析: {kernel_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'kernel_detail_{kernel_name[:10].replace(" ", "_")}.png',
                    dpi=120, bbox_inches='tight')
        plt.show()

    def compare_padding_stride(self, kernel_name: str = 'Sobel 水平边缘'):
        """对比不同 padding 和 stride 的效果"""
        kernel = KERNELS[kernel_name]
        x = torch.from_numpy(self.image).float().unsqueeze(0).unsqueeze(0)
        k = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)

        configs = [
            ('padding=0, stride=1', dict(padding=0, stride=1)),
            ('padding=1, stride=1', dict(padding=1, stride=1)),
            ('padding=1, stride=2', dict(padding=1, stride=2)),
            ('padding=2, stride=1', dict(padding=2, stride=1)),
        ]

        fig, axes = plt.subplots(1, len(configs) + 1, figsize=(18, 4))

        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title(f'原图 {self.image.shape}', fontsize=10)
        axes[0].axis('off')

        for i, (label, cfg) in enumerate(configs):
            out = F.conv2d(x, k, **cfg)[0, 0].numpy()
            axes[i+1].imshow(out, cmap='gray')
            axes[i+1].set_title(f'{label}\n输出: {out.shape}', fontsize=9)
            axes[i+1].axis('off')

        plt.suptitle(f'Padding & Stride 效果对比（核: {kernel_name}）',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('padding_stride_compare.png', dpi=120, bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────
# 多层 CNN 逐层输出查看器
# ─────────────────────────────────────────────────────────

class LayerByLayerViewer:
    """
    逐层查看 CNN 的输出变化

    使用方法：
        viewer = LayerByLayerViewer(model)
        viewer.show_pipeline(input_tensor)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def show_pipeline(self, x: torch.Tensor, max_channels: int = 4,
                      figsize_scale: float = 2.0):
        """
        显示输入经过每一层后的变化

        每行 = 一层，每列 = 一个通道
        """
        self.model.eval()
        activations = [('输入', x[0].detach().cpu())]

        current = x
        with torch.no_grad():
            for name, module in self.model.named_children():
                current = module(current)
                if current.dim() == 4:
                    activations.append((name, current[0].detach().cpu()))
                elif current.dim() == 2:
                    activations.append((name, current[0].detach().cpu().unsqueeze(0)))

        n_rows = len(activations)
        n_cols = max_channels + 1  # +1 for label column

        fig, axes = plt.subplots(n_rows, max_channels,
                                  figsize=(max_channels * figsize_scale,
                                           n_rows * figsize_scale))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row, (layer_name, feat) in enumerate(activations):
            n_ch = min(feat.shape[0], max_channels)
            for col in range(max_channels):
                ax = axes[row, col]
                if col < n_ch:
                    if feat.dim() == 3:
                        ax.imshow(feat[col].numpy(), cmap='viridis', aspect='auto')
                        ax.set_title(f'ch{col}', fontsize=7)
                    else:
                        ax.bar(range(len(feat[0])), feat[0].numpy(), color='steelblue')
                    ax.axis('off')
                else:
                    ax.axis('off')

            # 在第一列左侧标注层名
            axes[row, 0].set_ylabel(
                f'{layer_name}\n{tuple(feat.shape)}',
                fontsize=8, rotation=0, labelpad=60, va='center'
            )
            axes[row, 0].yaxis.set_label_position('left')

        plt.suptitle('CNN 逐层特征图（每行=一层，每列=一个通道）',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('layer_by_layer.png', dpi=120, bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_debug_panel():
    torch.manual_seed(42)

    panel = KernelDebugPanel()

    print("1. 所有卷积核效果对比")
    panel.show_all_kernels()

    print("\n2. 单核详细分析")
    panel.show_kernel_detail('Sobel 水平边缘')
    panel.show_kernel_detail('高斯模糊')

    print("\n3. Padding & Stride 对比")
    panel.compare_padding_stride('Sobel 垂直边缘')

    print("\n4. 逐层输出查看")
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
    )

    x = torch.from_numpy(panel.image).float().unsqueeze(0).unsqueeze(0)
    viewer = LayerByLayerViewer(model)
    viewer.show_pipeline(x, max_channels=4)

    return panel, viewer

panel, viewer = demo_debug_panel()
