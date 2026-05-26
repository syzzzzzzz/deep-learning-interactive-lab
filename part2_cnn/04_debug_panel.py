"""
自动生成自: part2_cnn\04_debug_panel.md
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
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from typing import Dict, Optional

MODULE_TITLE = "CNN 调试面板"
MODULE_SUMMARY = "用卷积核效果、padding/stride 输出尺寸和逐层特征图解释 CNN 调试时到底该看哪里。"
MODULE_TAGS = ["CNN", "调试", "卷积核", "Padding", "Stride", "特征图"]
MODULE_RELATED_TOPICS = ["part2/01_convolution_visual", "part2/02_feature_maps", "part2/03_classic_architectures", "part5/02_gradient_monitor"]
PRACTICE_TARGET = "切换卷积核、padding、stride 和显示通道数，解释输出尺寸、边缘响应和逐层特征图为什么变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure

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

# panel, viewer = demo_debug_panel()  # 协议化后由 render()/compute_cnn_debug_panel() 控制执行


def _plot_kernel_detail(image: np.ndarray, kernel_name: str) -> tuple[object, dict[str, float]]:
    if kernel_name not in KERNELS:
        raise ValueError(f"未知卷积核: {kernel_name}")
    kernel = KERNELS[kernel_name]
    result = apply_kernel(image, kernel, padding=1)
    with safe_mpl_figure(figsize=(11, 3.8)) as fig:
        axes = fig.subplots(1, 4)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("输入图像", fontsize=10, fontweight="bold")
        axes[0].axis("off")
        vmax = max(abs(float(kernel.min())), abs(float(kernel.max())), 1e-8)
        im = axes[1].imshow(kernel, cmap="RdBu", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.03)
        for i in range(3):
            for j in range(3):
                axes[1].text(j, i, f"{kernel[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
        axes[1].set_title(kernel_name, fontsize=10, fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(result, cmap="gray")
        axes[2].set_title("卷积输出", fontsize=10, fontweight="bold")
        axes[2].axis("off")
        axes[3].hist(result.flatten(), bins=32, color="#00f0ff", edgecolor="white", alpha=0.82)
        axes[3].axvline(0, color="#bf3f5b", linestyle="--", alpha=0.75)
        axes[3].set_title("激活分布", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("响应值")
        axes[3].grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        stats = {
            "response_mean": float(result.mean()),
            "response_std": float(result.std()),
            "response_max": float(result.max()),
            "response_min": float(result.min()),
        }
        return fig, stats


def _plot_padding_stride(image: np.ndarray, kernel_name: str, padding: int, stride: int) -> tuple[object, dict[str, object]]:
    kernel = KERNELS[kernel_name]
    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    k = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
    configs = [
        ("当前设置", {"padding": padding, "stride": stride}),
        ("不补边", {"padding": 0, "stride": 1}),
        ("保尺寸", {"padding": 1, "stride": 1}),
        ("下采样", {"padding": 1, "stride": 2}),
    ]
    outputs: list[tuple[str, np.ndarray]] = []
    for label, cfg in configs:
        outputs.append((f"{label}\np={cfg['padding']}, s={cfg['stride']}", F.conv2d(x, k, **cfg)[0, 0].numpy()))
    with safe_mpl_figure(figsize=(11, 3.6)) as fig:
        axes = fig.subplots(1, len(outputs) + 1)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"输入\n{image.shape}", fontsize=9, fontweight="bold")
        axes[0].axis("off")
        for ax, (label, output) in zip(axes[1:], outputs):
            ax.imshow(output, cmap="gray")
            ax.set_title(f"{label}\n{output.shape}", fontsize=8, fontweight="bold")
            ax.axis("off")
        fig.suptitle("Padding / Stride 调试：输出尺寸和边界信息如何变化", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, {"current_output_shape": outputs[0][1].shape, "current_output_mean": float(outputs[0][1].mean())}


def _plot_layer_pipeline(image: np.ndarray, max_channels: int) -> tuple[object, list[tuple[str, tuple[int, ...]]]]:
    max_channels = clamp_int(max_channels, 1, 8, "显示通道数")
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
    )
    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    activations: list[tuple[str, torch.Tensor]] = [("输入", x[0].detach())]
    current = x
    with torch.no_grad():
        for index, layer in enumerate(model):
            current = layer(current)
            if current.dim() == 4:
                activations.append((f"{index}:{layer.__class__.__name__}", current[0].detach()))
    with safe_mpl_figure(figsize=(max_channels * 1.8, len(activations) * 1.65)) as fig:
        axes = fig.subplots(len(activations), max_channels)
        axes = np.atleast_2d(axes)
        for row, (name, feat) in enumerate(activations):
            channels = min(max_channels, feat.shape[0])
            for col in range(max_channels):
                ax = axes[row, col]
                if col < channels:
                    ax.imshow(feat[col].numpy(), cmap="viridis", aspect="auto")
                    ax.set_title(f"ch{col}", fontsize=7)
                ax.axis("off")
            axes[row, 0].set_ylabel(f"{name}\n{tuple(feat.shape)}", fontsize=8, rotation=0, labelpad=55, va="center")
        fig.suptitle("逐层输出查看：每过一层，图像如何变成特征图", fontsize=12, fontweight="bold")
        fig.tight_layout()
        shapes = [(name, tuple(feat.shape)) for name, feat in activations]
        return fig, shapes


def compute_cnn_debug_panel(
    kernel_name: str = "Sobel 水平边缘",
    image_size: int = 64,
    padding: int = 1,
    stride: int = 1,
    max_channels: int = 4,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute CNN debugging visuals without launching Matplotlib GUI widgets."""

    if kernel_name not in KERNELS:
        raise ValueError("kernel_name 必须来自 KERNELS")
    image_size = clamp_int(image_size, 32, 96, "图像尺寸")
    padding = clamp_int(padding, 0, 4, "padding")
    stride = clamp_int(stride, 1, 4, "stride")
    max_channels = clamp_int(max_channels, 1, 8, "显示通道数")
    np.random.seed(seed)
    torch.manual_seed(seed)
    panel = KernelDebugPanel()
    image = panel._make_test_image(image_size)
    kernel_fig, kernel_stats = _plot_kernel_detail(image, kernel_name)
    padding_fig, padding_stats = _plot_padding_stride(image, kernel_name, padding, stride)
    layer_fig, layer_shapes = _plot_layer_pipeline(image, max_channels)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("CNN 调试面板协议化计算")
        print(f"卷积核={kernel_name}, image_size={image_size}, padding={padding}, stride={stride}, max_channels={max_channels}")
        print(f"卷积响应均值={kernel_stats['response_mean']:.4f}, 标准差={kernel_stats['response_std']:.4f}")
        print(f"当前输出尺寸={padding_stats['current_output_shape']}")
        for name, shape in layer_shapes:
            print(f"{name}: {shape}")
        print("诊断建议：边缘核看响应位置，padding/stride 看尺寸变化，逐层图看是否出现全黑、全亮或通道坍缩。")
    figures = [
        ("cnn_debug_kernel_detail.png", kernel_fig),
        ("cnn_debug_padding_stride.png", padding_fig),
        ("cnn_debug_layer_pipeline.png", layer_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {**kernel_stats, **padding_stats, "layer_count": len(layer_shapes)}
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "layer_shapes": layer_shapes, "log": log_buffer.getvalue()}


def _go_to_feature_maps() -> None:
    import streamlit as st

    st.query_params["module"] = "part2_cnn/02_feature_maps"
    st.rerun()


def render() -> None:
    """Render the CNN debugging panel lesson."""

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
        render_loading_bar("正在生成卷积核、padding/stride 和逐层特征图诊断")
        with st.sidebar:
            kernel_name = st.selectbox("卷积核", list(KERNELS.keys()), index=list(KERNELS.keys()).index("Sobel 水平边缘"))
            image_size = st.slider("图像尺寸", 32, 96, 64, 8)
            padding = st.slider("padding", 0, 4, 1, 1)
            stride = st.slider("stride", 1, 4, 1, 1)
            max_channels = st.slider("显示通道数", 1, 8, 4, 1)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("继续看：特征图可视化", width="stretch"):
                _go_to_feature_maps()
        data = compute_cnn_debug_panel(kernel_name, image_size, padding, stride, max_channels, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_convolution_particle_flow()
        st.markdown(
            """
            **零基础直觉：**CNN 调试就是问三个问题：卷积核在找什么？padding/stride 有没有把尺寸弄错？
            多层网络走到后面是不是还保留有用信号？本页把这三件事拆成三张图，让你不用盯控制台猜。
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("响应标准差", f"{stats['response_std']:.3f}")
        c2.metric("当前输出尺寸", str(stats["current_output_shape"]))
        c3.metric("可视化层数", str(stats["layer_count"]))
        explainers = [
            ("卷积核详情", "核里的正负权重决定它更喜欢亮边、暗边还是模糊区域；输出图越亮，表示该位置越符合这个核的模式。"),
            ("Padding / Stride 对比", "padding 决定边界是否保留，stride 决定滑窗跳多远；stride 越大，输出越小，信息越被压缩。"),
            ("逐层输出", "浅层通常保留边缘和纹理，越往后越抽象；如果某层全黑，常见原因是激活死亡、权重异常或输入归一化错误。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请只改一个控件，观察输出尺寸、亮区位置或通道数量如何变化。思考：这是卷积核作用，还是 padding/stride 造成的？")
        with st.expander("控制台输出与排查清单", expanded=False):
            st.markdown(
                """
                - **输出尺寸不对**：先查 kernel、padding、stride。
                - **图像边缘异常**：先查 padding 是否过小或过大。
                - **特征图全黑**：查 ReLU 死亡、学习率过大、输入是否归一化。
                - **通道都很像**：查初始化、训练是否充分，或卷积核是否学到重复模式。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/04_debug_panel.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_cnn_debug_panel(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_cnn_debug_panel(kernel_name="高斯模糊", image_size=32, padding=1, stride=1, max_channels=2, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["layer_count"] > 0 and data["stats"]["current_output_shape"][0] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_cnn_debug_panel))
