MODULE_TITLE = "卷积直觉"
MODULE_SUMMARY = "用滑窗、卷积核和边缘检测建立 CNN 的局部特征直觉。"
MODULE_TAGS = ["CNN", "卷积", "可视化", "视觉"]

import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


CLASSIC_KERNELS = {
    "Sobel_水平边缘": {
        "kernel": np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]),
        "desc": "检测水平方向的亮度变化（水平边缘）\n原理：上下像素差值，中间行权重为0",
        "cmap": "RdBu",
    },
    "Sobel_垂直边缘": {
        "kernel": np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
        "desc": "检测垂直方向的亮度变化（垂直边缘）\n原理：左右像素差值，中间列权重为0",
        "cmap": "RdBu",
    },
    "Laplacian_全方向边缘": {
        "kernel": np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]),
        "desc": "检测所有方向的边缘\n原理：中心像素 - 四邻域均值，二阶导数",
        "cmap": "RdBu",
    },
    "均值模糊": {
        "kernel": np.ones((3, 3)) / 9,
        "desc": "简单平均，去除噪声\n原理：每个像素替换为邻域均值",
        "cmap": "gray",
    },
    "高斯模糊": {
        "kernel": np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16,
        "desc": "加权平均，中心权重更大\n原理：模拟高斯分布，更自然的模糊",
        "cmap": "gray",
    },
    "锐化": {
        "kernel": np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]),
        "desc": "增强边缘，使图像更清晰\n原理：原图 + 边缘（Laplacian）",
        "cmap": "gray",
    },
    "浮雕效果": {
        "kernel": np.array([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]),
        "desc": "产生3D浮雕感\n原理：对角方向的差值",
        "cmap": "gray",
    },
}


CNN_DEBUG_GUIDE = """
CNN 10 大常见调试问题

1. 特征图全黑：检查 ReLU 死亡、学习率、BatchNorm running 统计。
2. 训练损失不下降：检查学习率、数据归一化、标签格式、输出维度。
3. 过拟合：使用 Dropout、数据增强、减小模型、L2 正则、早停。
4. 梯度消失：增加残差连接，换用 ReLU/GELU，检查初始化。
5. 梯度爆炸：使用梯度裁剪、降低学习率、加入归一化。
6. 输出形状错误：逐层打印 shape，或用 AdaptiveAvgPool2d(1)。
7. BatchNorm 推理表现差：推理前调用 model.eval()，确认 batch 统计稳定。
8. 验证集泄露：验证集不要做随机增强。
9. 类别不平衡：使用加权损失或 WeightedRandomSampler。
10. 卷积核学不到特征：增加训练轮数、使用预训练、检查梯度流。
"""


def manual_conv2d_step_by_step() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    x = torch.tensor(
        [[[[1.0, 2.0, 3.0, 0.0, 1.0],
           [4.0, 5.0, 6.0, 1.0, 0.0],
           [7.0, 8.0, 9.0, 2.0, 1.0],
           [0.0, 1.0, 2.0, 3.0, 4.0],
           [1.0, 0.0, 1.0, 2.0, 3.0]]]]
    )
    kernel = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])

    print("=" * 60)
    print("手动卷积计算演示")
    print("=" * 60)
    print(f"\n输入 x (5×5):\n{x[0, 0].numpy()}")
    print(f"\n卷积核 (3×3):\n{kernel[0, 0].numpy()}")
    print("\n逐位置计算（步幅=1，填充=0）：")
    output_manual = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            patch = x[0, 0, i:i + 3, j:j + 3].numpy()
            k = kernel[0, 0].numpy()
            val = (patch * k).sum()
            output_manual[i, j] = val
            print(f"  输出[{i},{j}] = patch[{i}:{i+3},{j}:{j+3}] ⊙ kernel = {val:.1f}")
            if i == 0 and j == 0:
                print(f"    patch:\n{patch}")
                print(f"    逐元素乘:\n{patch * k}")
                print(f"    求和: {val:.1f}")

    output_torch = F.conv2d(x, kernel, padding=0, stride=1)
    print(f"\n手动计算输出:\n{output_manual}")
    print(f"\nPyTorch 计算输出:\n{output_torch[0, 0].numpy()}")
    print(f"\n误差: {np.abs(output_manual - output_torch[0, 0].numpy()).max():.8f}  ✓")
    return x, kernel, output_manual


def output_shape_formula() -> list[tuple[int, int, int, int, int, str]]:
    print("\n" + "=" * 60)
    print("输出形状公式演示")
    print("H_out = (H_in + 2*P - K) / S + 1")
    configs = [
        (32, 3, 0, 1, "32×32输入，3×3核，无填充，步幅1"),
        (32, 3, 1, 1, "32×32输入，3×3核，填充1，步幅1  ← 保持尺寸"),
        (32, 3, 1, 2, "32×32输入，3×3核，填充1，步幅2  ← 下采样"),
        (32, 5, 2, 1, "32×32输入，5×5核，填充2，步幅1  ← 保持尺寸"),
        (28, 5, 0, 1, "28×28输入，5×5核，无填充，步幅1  ← LeNet"),
        (224, 11, 2, 4, "224×224输入，11×11核，填充2，步幅4  ← AlexNet第一层"),
    ]
    rows = []
    for H_in, K, P, S, desc in configs:
        H_out = (H_in + 2 * P - K) // S + 1
        rows.append((H_in, K, P, S, H_out, desc))
        print(f"  {desc}")
        print(f"    ({H_in} + 2×{P} - {K}) / {S} + 1 = {H_out}×{H_out}")
    return rows


def make_test_image(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    img[8:28, 8:28] = 1.0
    for i in range(size):
        for j in range(size):
            if (i - 48) ** 2 + (j - 48) ** 2 < 12 ** 2:
                img[i, j] = 0.8
    for i in range(size):
        img[i, min(i, size - 1)] = 1.0
    img[40:42, :] = 0.9
    img[:, 40:42] = 0.7
    img += np.random.randn(size, size).astype(np.float32) * 0.03
    return np.clip(img, 0, 1)


def plot_classic_kernels() -> plt.Figure:
    np.random.seed(42)
    img = make_test_image(64)
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    n_kernels = len(CLASSIC_KERNELS)
    with safe_mpl_figure(figsize=(20, 14)) as fig:
        gs = fig.add_gridspec(3, n_kernels + 1, hspace=0.4, wspace=0.3)
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_orig.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax_orig.set_title("原始图像\n(64×64)", fontsize=11, fontweight="bold")
        ax_orig.axis("off")
        for col, (name, info) in enumerate(CLASSIC_KERNELS.items()):
            k = info["kernel"]
            result = F.conv2d(x, torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0), padding=1)[0, 0].numpy()
            ax_k = fig.add_subplot(gs[0, col + 1])
            vmax = max(abs(k.min()), abs(k.max())) + 1e-8
            ax_k.imshow(k, cmap="RdBu", vmin=-vmax, vmax=vmax)
            for i in range(3):
                for j in range(3):
                    ax_k.text(j, i, f"{k[i, j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold", color="white" if abs(k[i, j]) > vmax * 0.5 else "black")
            ax_k.set_title(name.replace("_", "\n"), fontsize=8, fontweight="bold")
            ax_k.axis("off")
            ax_r = fig.add_subplot(gs[1, col + 1])
            ax_r.imshow(result, cmap=str(info["cmap"]), vmin=result.min(), vmax=result.max())
            ax_r.set_title("卷积结果", fontsize=7)
            ax_r.axis("off")
            ax_h = fig.add_subplot(gs[2, col + 1])
            ax_h.hist(result.flatten(), bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            ax_h.set_title(f"分布\n均值={result.mean():.2f}", fontsize=7)
            ax_h.grid(True, alpha=0.3)
        fig.suptitle("经典卷积核效果对比（第一行=核，第二行=结果，第三行=分布）", fontsize=13, fontweight="bold")
        return fig


def print_kernel_descriptions() -> None:
    print("\n各卷积核说明：")
    for name, info in CLASSIC_KERNELS.items():
        print(f"\n{name}:")
        print(f"  核:\n{info['kernel']}")
        print(f"  原理: {info['desc']}")


def plot_receptive_field() -> plt.Figure:
    input_size = 13
    with safe_mpl_figure(figsize=(16, 4)) as fig:
        axes = fig.subplots(1, 4)
        for ax_idx, n_layers in enumerate([1, 2, 3, 4]):
            rf_size = 2 * n_layers + 1
            grid = np.zeros((input_size, input_size))
            center = input_size // 2
            half = rf_size // 2
            grid[center - half:center + half + 1, center - half:center + half + 1] = 0.4
            grid[center, center] = 1.0
            axes[ax_idx].imshow(grid, cmap="Blues", vmin=0, vmax=1)
            axes[ax_idx].set_title(f"{n_layers}层 3×3 卷积\n感受野={rf_size}×{rf_size}", fontsize=11, fontweight="bold")
            for i in range(input_size):
                for j in range(input_size):
                    if grid[i, j] > 0:
                        axes[ax_idx].text(j, i, "●" if grid[i, j] == 1 else "·", ha="center", va="center", fontsize=8, color="white" if grid[i, j] > 0.5 else "steelblue")
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])
        fig.suptitle('感受野随网络深度的增长\n（蓝色区域=当前输出位置能"看到"的输入范围）', fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig


def print_receptive_field_table() -> None:
    print("感受野大小 vs 层数（3×3卷积，步幅1）：")
    for n in range(1, 8):
        rf = 2 * n + 1
        params_stack = n * 9
        params_single = rf * rf
        print(f"  {n}层: 感受野={rf}×{rf}={rf**2}像素  堆叠参数={params_stack}  等效单核参数={params_single}  节省={params_single - params_stack}个参数")


def plot_pooling_demo() -> plt.Figure:
    x = torch.tensor([[[[1.0, 3.0, 2.0, 4.0], [5.0, 6.0, 7.0, 8.0], [3.0, 2.0, 1.0, 0.0], [9.0, 4.0, 3.0, 2.0]]]])
    max_pool = F.max_pool2d(x, kernel_size=2, stride=2)
    avg_pool = F.avg_pool2d(x, kernel_size=2, stride=2)
    print("输入 (4×4):")
    print(x[0, 0].numpy())
    print(f"\nMaxPool2d(2×2, stride=2) → 输出形状: {tuple(max_pool.shape)}")
    print(max_pool[0, 0].numpy())
    print("原理：每个2×2区域取最大值")
    print(f"\nAvgPool2d(2×2, stride=2) → 输出形状: {tuple(avg_pool.shape)}")
    print(avg_pool[0, 0].numpy())
    print("原理：每个2×2区域取平均值")
    with safe_mpl_figure(figsize=(12, 4)) as fig:
        axes = fig.subplots(1, 3)
        panels = [
            (x[0, 0].numpy(), "Blues", "输入 (4×4)", "{:.0f}", 14),
            (max_pool[0, 0].numpy(), "Reds", "MaxPool (2×2)\n取最大值", "{:.0f}", 18),
            (avg_pool[0, 0].numpy(), "Greens", "AvgPool (2×2)\n取平均值", "{:.1f}", 18),
        ]
        for ax, (data, cmap, title, fmt, size) in zip(axes, panels):
            ax.imshow(data, cmap=cmap, vmin=0, vmax=9)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=size, fontweight="bold", color="white" if data.shape[0] == 2 else "black")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")
        fig.suptitle("池化操作对比", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def compute_卷积可视化(save_artifacts: bool = False) -> dict[str, object]:
    """Compute convolution demonstrations without Streamlit calls."""

    artifacts: list[Path] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        _, _, output_manual = manual_conv2d_step_by_step()
        formula_rows = output_shape_formula()
        classic_fig = plot_classic_kernels()
        print_kernel_descriptions()
        receptive_fig = plot_receptive_field()
        print_receptive_field_table()
        pooling_fig = plot_pooling_demo()
        print(CNN_DEBUG_GUIDE)
    figures = [("classic_kernels.png", classic_fig), ("receptive_field.png", receptive_fig), ("pooling_demo.png", pooling_fig)]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {"log": log_buffer.getvalue(), "figures": figures, "artifacts": artifacts, "manual_output": output_manual, "formula_rows": formula_rows}


def render_卷积可视化() -> None:
    """Render the convolution lesson in Streamlit."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_convolution_particle_flow, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("卷积演示加载：像素粒子、卷积核和输出特征图将同步出现")
        render_convolution_particle_flow()
        data = compute_卷积可视化(save_artifacts=True)
        st.subheader("手算卷积输出")
        st.dataframe(data["manual_output"], width="stretch")
        st.subheader("输出形状公式")
        st.dataframe([{"输入": h, "核": k, "填充": p, "步幅": s, "输出": out, "说明": desc} for h, k, p, s, out, desc in data["formula_rows"]], width="stretch")
        for title, (filename, fig) in zip(("经典卷积核效果", "感受野增长", "池化操作对比"), data["figures"]):
            st.subheader(title)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"已保存产物：{get_artifact_path(filename)}")
        with st.expander("控制台讲解与 CNN 调试指南", expanded=False):
            st.code(str(data["log"])[-16000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/01_convolution_visual.py", exc)


render = render_卷积可视化


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_卷积可视化(save_artifacts=False)
    return bool(data["figures"]) and bool(data["formula_rows"])


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_stdio()
    if _running_under_streamlit():
        render_卷积可视化()
    else:
        try:
            result = compute_卷积可视化(save_artifacts=True)
            print(result["log"])
            for path in result["artifacts"]:
                print(f"图像已保存: {path}")
        except Exception as e:
            traceback.print_exception(e)
            raise SystemExit(1)
