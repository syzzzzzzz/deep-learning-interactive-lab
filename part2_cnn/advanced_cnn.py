"""
Advanced CNN visual lab.

Run:
    streamlit run part2_cnn/advanced_cnn.py
or:
    python main.py part2_cnn/advanced_cnn
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from components.visual_system import (
    render_advanced_conv_comparison,
    render_beginner_hint,
    render_motion_note,
    render_visual_system,
)


torch.set_num_threads(1)

st.set_page_config(
    page_title="CNN 深度拓展",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #18242c;
        --muted: #5c6972;
        --line: #d8dde2;
        --panel: #ffffff;
        --paper: #f8f7f3;
        --teal: #0f8b8d;
        --rose: #bf3f5b;
        --amber: #c4871f;
        --green: #3f7d58;
    }
    .stApp {
        background: linear-gradient(180deg, #fbfaf7 0%, #eef5f3 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] {
        background: #eef4f2;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
    }
    .intro {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .intro h1 {
        margin: 0;
        font-size: clamp(2rem, 3vw, 3.25rem);
        line-height: 1.05;
    }
    .intro p {
        color: var(--muted);
        max-width: 980px;
        line-height: 1.75;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid var(--teal);
        background: rgba(255,255,255,0.72);
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 0.85rem;
        color: #2c3941;
        line-height: 1.68;
        margin: 0.25rem 0 0.8rem 0;
    }
    .formula {
        font-family: Consolas, Menlo, monospace;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
        margin: 0.35rem 0 0.7rem 0;
        color: #22313a;
        overflow-x: auto;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


render_visual_system("light")
render_advanced_conv_comparison()


@dataclass(frozen=True)
class ConvResult:
    output: torch.Tensor
    formula: str
    params: int
    effective_kernel: int


def make_demo_input(size: int = 32, channels: int = 4, seed: int = 7) -> torch.Tensor:
    """Create a small multi-channel image with edges, blobs, stripes and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    yy = y / max(size - 1, 1)
    xx = x / max(size - 1, 1)

    square = np.zeros((size, size), dtype=np.float32)
    square[size // 6 : size // 2, size // 7 : size // 2] = 1.0

    circle = (((x - size * 0.68) ** 2 + (y - size * 0.63) ** 2) < (size * 0.18) ** 2).astype(np.float32)
    diagonal = np.exp(-((xx - yy) ** 2) / 0.0025).astype(np.float32)
    stripes = (0.5 + 0.5 * np.sin(2 * np.pi * (xx * 4.0 + yy * 1.4))).astype(np.float32)

    base = [square, circle, diagonal, stripes]
    while len(base) < channels:
        base.append((0.5 * base[-1] + 0.5 * rng.random((size, size))).astype(np.float32))

    arr = np.stack(base[:channels]).astype(np.float32)
    arr += rng.normal(0, 0.025, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def make_kernel_bank(out_channels: int, in_channels: int, kernel_size: int) -> torch.Tensor:
    """Deterministic kernels: blur, edge, sharpen-like and diagonal responses."""
    center = kernel_size // 2
    y, x = np.mgrid[0:kernel_size, 0:kernel_size]
    dist2 = (x - center) ** 2 + (y - center) ** 2
    sigma = max(kernel_size / 3.0, 0.8)

    blur = np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)
    blur /= blur.sum()

    horizontal = np.zeros_like(blur)
    horizontal[:center, :] = -1
    horizontal[center + 1 :, :] = 1
    horizontal -= horizontal.mean()
    horizontal /= np.abs(horizontal).sum() + 1e-8

    vertical = horizontal.T
    laplace = -np.ones_like(blur)
    laplace[center, center] = kernel_size * kernel_size - 1
    laplace /= np.abs(laplace).sum() + 1e-8

    diag = np.eye(kernel_size, dtype=np.float32) - np.fliplr(np.eye(kernel_size, dtype=np.float32))
    diag /= np.abs(diag).sum() + 1e-8
    patterns = [blur, horizontal, vertical, laplace, diag]

    weight = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
    for oc in range(out_channels):
        for ic in range(in_channels):
            pattern = patterns[(oc + ic) % len(patterns)]
            weight[oc, ic] = pattern * (1.0 + 0.12 * ic)
    return torch.from_numpy(weight)


def make_pointwise_weight(out_channels: int, in_channels: int) -> torch.Tensor:
    weight = np.zeros((out_channels, in_channels, 1, 1), dtype=np.float32)
    for oc in range(out_channels):
        for ic in range(in_channels):
            weight[oc, ic, 0, 0] = math.cos((oc + 1) * (ic + 1)) / max(in_channels, 1)
    return torch.from_numpy(weight)


def standard_output_dim(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def transposed_output_dim(size: int, kernel: int, stride: int, padding: int, dilation: int, output_padding: int) -> int:
    return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def apply_conv(
    x: torch.Tensor,
    conv_type: str,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    out_channels: int,
    output_padding: int,
) -> ConvResult:
    in_channels = x.shape[1]

    if conv_type == "1x1 卷积":
        kernel_size = 1
        dilation = 1
        padding = 0
        weight = make_pointwise_weight(out_channels, in_channels)
        out = F.conv2d(x, weight, stride=stride, padding=padding)
        params = out_channels * in_channels
        effective = 1
        formula = f"floor((H + 2P - K) / S + 1) = floor(({x.shape[-2]} + 0 - 1) / {stride} + 1)"
        return ConvResult(out, formula, params, effective)

    if conv_type == "转置卷积":
        weight = make_kernel_bank(in_channels, out_channels, kernel_size)
        max_output_padding = max(stride - 1, 0)
        output_padding = min(output_padding, max_output_padding)
        out = F.conv_transpose2d(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
        )
        params = in_channels * out_channels * kernel_size * kernel_size
        effective = dilation * (kernel_size - 1) + 1
        formula = (
            f"(H - 1)S - 2P + D(K - 1) + OP + 1 = "
            f"({x.shape[-2]} - 1){stride} - 2*{padding} + {dilation}*({kernel_size} - 1) + {output_padding} + 1"
        )
        return ConvResult(out, formula, params, effective)

    if conv_type == "空洞卷积":
        groups = 1
        dilation = max(dilation, 2)

    if conv_type == "深度可分离卷积":
        depthwise_weight = make_kernel_bank(in_channels, 1, kernel_size)
        depthwise = F.conv2d(
            x,
            depthwise_weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        pointwise = make_pointwise_weight(out_channels, in_channels)
        out = F.conv2d(depthwise, pointwise)
        params = in_channels * kernel_size * kernel_size + in_channels * out_channels
        effective = dilation * (kernel_size - 1) + 1
        formula = (
            f"depthwise: floor((H + 2P - D(K - 1) - 1) / S + 1), "
            f"pointwise: 1x1 channel mixing"
        )
        return ConvResult(out, formula, params, effective)

    if conv_type == "分组卷积":
        valid_groups = [g for g in [1, 2, 4] if in_channels % g == 0 and out_channels % g == 0]
        groups = groups if groups in valid_groups else valid_groups[-1]
    else:
        groups = 1

    weight = make_kernel_bank(out_channels, in_channels // groups, kernel_size)
    out = F.conv2d(x, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    params = out_channels * (in_channels // groups) * kernel_size * kernel_size
    effective = dilation * (kernel_size - 1) + 1
    formula = (
        f"floor((H + 2P - D(K - 1) - 1) / S + 1) = "
        f"floor(({x.shape[-2]} + 2*{padding} - {dilation}*({kernel_size} - 1) - 1) / {stride} + 1)"
    )
    return ConvResult(out, formula, params, effective)


def plot_feature_maps(x: torch.Tensor, y: torch.Tensor, max_channels: int = 4) -> plt.Figure:
    in_img = x[0].mean(dim=0).numpy()
    out = y[0].detach().cpu()
    n = min(max_channels, out.shape[0])

    fig, axes = plt.subplots(2, max(n, 2), figsize=(3.1 * max(n, 2), 5.5))
    axes = np.asarray(axes)

    axes[0, 0].imshow(in_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"输入均值图\n{tuple(x.shape)}", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    if x.shape[1] > 1:
        axes[0, 1].imshow(x[0, 0].numpy(), cmap="Blues", vmin=0, vmax=1)
        axes[0, 1].set_title("输入通道 0", fontsize=10)
        axes[0, 1].axis("off")
    for j in range(2, axes.shape[1]):
        axes[0, j].axis("off")

    for i in range(n):
        fmap = out[i].numpy()
        axes[1, i].imshow(fmap, cmap="viridis")
        axes[1, i].set_title(f"输出通道 {i}\n{fmap.shape[0]}x{fmap.shape[1]}", fontsize=10)
        axes[1, i].axis("off")
    for j in range(n, axes.shape[1]):
        axes[1, j].axis("off")

    fig.tight_layout()
    return fig


def plot_conv_type_overview(x: torch.Tensor) -> plt.Figure:
    configs = [
        ("1x1", "1x1 卷积", 1, 1, 0, 1, 1),
        ("3x3", "标准卷积", 3, 1, 1, 1, 1),
        ("5x5", "标准卷积", 5, 1, 2, 1, 1),
        ("转置", "转置卷积", 3, 2, 1, 1, 0),
        ("空洞", "空洞卷积", 3, 1, 2, 2, 1),
        ("深度可分离", "深度可分离卷积", 3, 1, 1, 1, 1),
        ("分组", "分组卷积", 3, 1, 1, 1, 2),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.ravel()
    axes[0].imshow(x[0].mean(dim=0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("输入", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for ax, (title, conv_type, k, s, p, d, g) in zip(axes[1:], configs):
        result = apply_conv(x, conv_type, k, s, p, d, g, out_channels=4, output_padding=1)
        fmap = result.output[0, 0].detach().numpy()
        ax.imshow(fmap, cmap="viridis")
        ax.set_title(
            f"{title}\n{tuple(result.output.shape[-2:])}, 参数 {result.params}",
            fontsize=10,
            fontweight="bold",
        )
        ax.axis("off")
    fig.suptitle("不同卷积类型对同一输入的响应", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def pool2d_numpy(img: np.ndarray, kernel: int, stride: int, mode: str) -> np.ndarray:
    out_h = math.floor((img.shape[0] - kernel) / stride + 1)
    out_w = math.floor((img.shape[1] - kernel) / stride + 1)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i * stride : i * stride + kernel, j * stride : j * stride + kernel]
            out[i, j] = patch.max() if mode == "max" else patch.mean()
    return out


def plot_pooling_demo(kernel: int, stride: int) -> plt.Figure:
    img = make_demo_input(12, 1, seed=3)[0, 0].numpy()
    max_pool = pool2d_numpy(img, kernel, stride, "max")
    avg_pool = pool2d_numpy(img, kernel, stride, "avg")
    gap = np.array([[img.mean()]], dtype=np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    items = [
        ("输入", img, "gray"),
        ("最大池化", max_pool, "YlOrRd"),
        ("平均池化", avg_pool, "YlGnBu"),
        ("全局平均池化", gap, "Greens"),
    ]
    for ax, (title, arr, cmap) in zip(axes, items):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(f"{title}\n{arr.shape[0]}x{arr.shape[1]}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr.shape[0] <= 8 and arr.shape[1] <= 8:
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_bn_demo(batch_size: int, channels: int, seed: int) -> plt.Figure:
    rng = np.random.default_rng(seed)
    means = np.linspace(-2.5, 2.0, channels).reshape(1, channels, 1, 1)
    scales = np.linspace(0.35, 2.2, channels).reshape(1, channels, 1, 1)
    x = rng.normal(size=(batch_size, channels, 8, 8)).astype(np.float32) * scales + means
    x_t = torch.from_numpy(x)
    y = F.batch_norm(x_t, running_mean=None, running_var=None, training=True, eps=1e-5)
    x_flat = x_t.permute(1, 0, 2, 3).reshape(channels, -1).numpy()
    y_flat = y.permute(1, 0, 2, 3).reshape(channels, -1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(x_flat.ravel(), bins=50, alpha=0.78, color="#bf3f5b", label="BN 前")
    axes[0].hist(y_flat.ravel(), bins=50, alpha=0.68, color="#0f8b8d", label="BN 后")
    axes[0].set_title("整体分布", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    width = 0.36
    idx = np.arange(channels)
    axes[1].bar(idx - width / 2, x_flat.mean(axis=1), width, label="BN 前", color="#bf3f5b")
    axes[1].bar(idx + width / 2, y_flat.mean(axis=1), width, label="BN 后", color="#0f8b8d")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("每通道均值", fontweight="bold")
    axes[1].set_xlabel("通道")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].bar(idx - width / 2, x_flat.std(axis=1), width, label="BN 前", color="#bf3f5b")
    axes[2].bar(idx + width / 2, y_flat.std(axis=1), width, label="BN 后", color="#0f8b8d")
    axes[2].axhline(1, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("每通道标准差", fontweight="bold")
    axes[2].set_xlabel("通道")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_dropout_demo(drop_prob: float, training: bool, seed: int) -> tuple[plt.Figure, float]:
    torch.manual_seed(seed)
    x = make_demo_input(32, 1, seed=seed)
    if training:
        y = F.dropout(x, p=drop_prob, training=True)
    else:
        y = F.dropout(x, p=drop_prob, training=False)

    zero_ratio = float((y == 0).float().mean().item())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].imshow(x[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("输入激活", fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(y[0, 0].numpy(), cmap="magma")
    axes[1].set_title("Dropout 输出", fontweight="bold")
    axes[1].axis("off")
    axes[2].hist(x.flatten().numpy(), bins=35, alpha=0.7, label="输入", color="#5c6972")
    axes[2].hist(y.flatten().numpy(), bins=35, alpha=0.7, label="输出", color="#c4871f")
    axes[2].set_title("数值分布", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, zero_ratio


def render_advanced_cnn_learning_map() -> None:
    st.markdown(
        """
        <div class="note">
        <strong>学习地图：</strong>这个页面不是在罗列高级名词，而是在拆 CNN 的四个工程问题：
        <strong>卷积实验台</strong>回答“输出尺寸和感受野怎么变”，
        <strong>卷积类型总览</strong>回答“不同卷积为什么省参数或扩视野”，
        <strong>池化层</strong>回答“为什么丢掉一部分空间信息反而更稳”，
        <strong>BatchNorm / Dropout</strong>回答“训练为什么会稳定、为什么不容易死记训练集”。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        > 互动顺序：先在左侧固定“随机种子”，逐个调“卷积核大小”“步长 stride”“填充 padding”“空洞率 dilation”，观察“输出形状”“有效感受野”“参数量”和输出特征图如何一起变化；再切换到池化、BN、Dropout 页签，把这些局部操作放回完整 CNN 训练流程里。
        >
        > 进阶思考：CNN 看起来只是图片变小、通道变多，但每一次尺寸变化都在做取舍。请一边调参一边问：这一层是在保留细节、扩大视野、节省参数，还是在提高训练稳定性？
        """
    )


def render_conv_experiment_guide(
    conv_type: str,
    input_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    output_padding: int,
    result: ConvResult,
) -> None:
    st.markdown(
        f"""
        **图怎么看：**上排显示输入的均值图和第 0 个输入通道，下排显示若干输出通道。颜色越亮表示该卷积核在这个位置响应越强；响应强不等于“模型理解了语义”，它只说明当前核在该局部窗口上计算出了较大数值。指标卡里的“输出形状”告诉你特征图还有多少空间位置，“有效感受野”告诉你单个输出格子看到多大输入区域，“参数量”告诉你这一层要学习多少权重。

        **参数怎么调：**当前卷积类型是 **{conv_type}**，输入为 **{in_channels} x {input_size} x {input_size}**，输出通道数为 **{out_channels}**，卷积核大小为 **{kernel_size}**，步长为 **{stride}**，填充为 **{padding}**，空洞率为 **{dilation}**，分组数为 **{groups}**，转置卷积 output_padding 为 **{output_padding}**。当前有效感受野是 **{result.effective_kernel} x {result.effective_kernel}**，参数量是 **{result.params:,}**。

        **取值区间：**“输入尺寸 H=W”的范围是 12 到 64；“输入通道数”可选 1、2、4；“输出通道数”可选 1、2、4、8；“卷积核大小”可选 1、3、5；“步长 stride”的范围是 1 到 4；“填充 padding”的范围是 0 到 6；“空洞率 dilation”的范围是 1 到 4。工程上，3x3、stride=1、padding=1 是最常见的稳定基线；stride=2 常用于下采样；stride=4 通常要谨慎，因为空间细节会被快速丢掉。

        > 反例实验 1：把“步长 stride”调到 4，再观察输出形状和特征图。你会发现输出网格明显变小，这是因为卷积窗口跳得太快，很多空间位置没有被细致读取；在检测和分割任务里，这会伤害小目标。
        >
        > 反例实验 2：选择“空洞卷积”，把“空洞率 dilation”调到 4。有效感受野会变大，但输出可能出现更稀疏的响应。思考：为什么空洞卷积能看得更远，却可能漏掉细密纹理？
        >
        > 工程经验：如果只是做普通图像分类，先用 3x3 卷积、padding=1、stride=1 建基线；需要降采样时再把 stride 设为 2。不要一开始就堆大核、大步幅和大空洞率，否则很难判断性能变化来自哪一个因素。
        """
    )


def render_conv_overview_guide() -> None:
    st.markdown(
        """
        **图怎么看：**这张总览图把同一个输入送进多种卷积。每个小图的标题都包含输出空间尺寸和参数量：尺寸变化说明信息网格变大或变小，参数量变化说明这一层的学习成本。1x1 卷积几乎不改变空间邻域，但会混合通道；转置卷积会扩大输出网格；空洞卷积不增加参数也能扩大感受野；深度可分离和分组卷积主要是在减少跨通道连接。

        **参数怎么调：**这一页签不额外使用滑块，它使用固定输入来对比卷积类型。请回到“卷积实验台”，用“卷积类型”选择同名操作，再调“卷积核大小”“步长 stride”“填充 padding”“空洞率 dilation”“分组数 groups”，把总览中的现象复现出来。

        > 进阶思考：如果两个卷积输出图看起来相似，但参数量差很多，你会选哪一个？移动端模型通常更在意参数量和 FLOPs，医学影像或分割任务则更在意空间细节是否保住。
        """
    )


def render_pooling_guide(pool_kernel: int, pool_stride: int) -> None:
    st.markdown(
        f"""
        **图怎么看：**输入图保留原始局部数值；最大池化图只保留每个窗口里的最强响应；平均池化图保留窗口平均趋势；全局平均池化把整张图压成 1 个数。池化后的图越小，空间位置越粗，但对小幅平移和噪声越不敏感。

        **参数怎么调：**当前“池化核”为 **{pool_kernel}**，“池化步长”为 **{pool_stride}**。池化核范围是 2 到 4，步长范围是 1 到 4。核越大、步长越大，输出越小，信息压缩越强；步长小于核时窗口会重叠，输出更平滑但计算更多。

        > 反例实验 3：把“池化核”设为 4，把“池化步长”也设为 4。观察输出尺寸和数值变化：这会非常激进地压缩空间信息。分类任务可能还能接受，但分割、定位、小目标检测会很容易丢细节。
        >
        > 工程经验：早期 CNN 常频繁池化，现代 CNN 更常用 stride 卷积、残差和特征金字塔控制尺寸。池化不是越强越好，它是在“稳健性”和“定位精度”之间做交换。
        """
    )


def render_bn_guide(batch_size: int, channels: int) -> None:
    st.markdown(
        f"""
        **图怎么看：**左图比较 BN 前后的整体分布，中图比较每个通道的均值，右图比较每个通道的标准差。理想情况下，BN 后每个通道会更接近均值 0、标准差 1，所以中图的青色柱靠近 0，右图的青色柱靠近虚线 1。

        **参数怎么调：**当前 batch size 为 **{batch_size}**，通道数为 **{channels}**。batch size 范围是 2 到 64，通道数范围是 2 到 12。batch 太小时，均值和方差估计会抖；通道越多，你越能看到“每个通道单独标准化”的效果。

        > 互动：把“batch size”从 2 调到 64，观察 BN 后的均值和标准差是否更稳定。思考：为什么小 batch 训练里 BN 可能不如 GroupNorm 或 LayerNorm 稳？
        >
        > 工程经验：BN 训练和推理行为不同。训练时用当前 batch 统计量，推理时用滑动平均统计量；如果忘记切换 eval 模式，线上输出可能会抖动。
        """
    )


def render_dropout_guide(drop_prob: float, training: bool, zero_ratio: float) -> None:
    mode_text = "训练模式" if training else "推理模式"
    st.markdown(
        f"""
        **图怎么看：**左图是原始激活，中图是 Dropout 后的输出，右图显示数值分布变化。当前处于 **{mode_text}**，丢弃概率 p 为 **{drop_prob:.2f}**，输出零值比例为 **{zero_ratio:.1%}**。训练模式下会随机屏蔽一部分激活；推理模式下不会随机丢弃，因此输出更稳定。

        **参数怎么调：**“丢弃概率 p”的范围是 0.0 到 0.9，默认值是 0.35；“训练模式：开启 Dropout”决定是否真的执行随机丢弃；“Dropout 随机种子”只改变本次随机掩码。p=0 表示不丢弃，p 接近 0.9 会让大部分激活变成 0，模型很难保留足够信息。

        > 极端值实验：把“丢弃概率 p”调到 0.9，再打开训练模式。观察中图是否大片变暗。思考：为什么过强 Dropout 会从“防止死记”变成“让模型看不清输入”？
        >
        > 工程经验：CNN 里 Dropout 常放在分类头或较高层，卷积特征早期层过强 Dropout 可能破坏局部纹理。现代架构还会使用数据增强、权重衰减、随机深度等替代或补充它。
        """
    )


st.markdown(
    """
    <div class="intro">
      <h1>CNN 深度拓展</h1>
      <p>用可调参数和特征图把高级卷积、池化、批归一化和 Dropout 拆开观察。左侧控制实验参数，主区域实时展示输出尺寸、参数量和响应图。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_beginner_hint(
    "先把高级卷积当成不同的取样方式",
    "标准卷积、1x1、转置卷积、空洞卷积、深度可分离卷积和分组卷积，都在回答同一个问题：用多少参数、看多大范围、保留多少空间细节。",
    action="先固定随机种子，只改一个参数；看输出形状、有效感受野和参数量哪一个先变。",
)
render_motion_note(
    "动效在说明什么",
    "上方高级卷积动画把不同卷积的扫描方式放在一起对比；亮块移动代表卷积窗口取样，通道变化代表参数连接方式改变。",
)
st.markdown(
    """
    **这是什么？** 这页把 CNN 里常见的高级组件拆开观察：不同卷积负责改变取样方式和连接方式，池化负责压缩空间信息，BatchNorm 负责稳定数值分布，Dropout 负责让模型训练时不要太依赖少数激活。

    **生活类比：** 像用不同镜头看同一张照片：标准卷积是普通放大镜，1x1 卷积是在调色盘里混合颜色，空洞卷积是隔着格子看更大范围，转置卷积像把小图重新铺大，池化像做摘要，BN 像把不同批次的音量调到接近水平，Dropout 像训练时随机遮住一部分线索。

    **一句话直觉：** 高级 CNN 层不是神秘技巧，而是在控制“看多大、算多少、保留多少细节、训练稳不稳”。

    **图中每个元素代表什么：** 特征图里的每个小图是一张输入或输出通道；颜色越亮表示该位置响应越强；指标卡显示输入形状、输出形状、有效感受野和参数量；池化图中的数字是窗口汇总后的值；BN 直方图和柱状图分别表示归一化前后的分布、均值和标准差；Dropout 图中变暗或为零的位置表示被随机屏蔽的激活。
    """
)
render_advanced_cnn_learning_map()

tab_conv, tab_overview, tab_pool, tab_bn, tab_dropout = st.tabs(
    ["卷积实验台", "卷积类型总览", "池化层", "BatchNorm", "Dropout"]
)

with st.sidebar:
    st.header("卷积实验参数")
    conv_type = st.selectbox(
        "卷积类型",
        ["标准卷积", "1x1 卷积", "转置卷积", "空洞卷积", "深度可分离卷积", "分组卷积"],
        index=0,
    )
    input_size = st.slider("输入尺寸 H=W", 12, 64, 32, 4)
    in_channels = st.select_slider("输入通道数", options=[1, 2, 4], value=4)
    out_channels = st.select_slider("输出通道数", options=[1, 2, 4, 8], value=4)
    kernel_size = st.select_slider("卷积核大小", options=[1, 3, 5], value=3)
    stride = st.slider("步长 stride", 1, 4, 1)
    padding = st.slider("填充 padding", 0, 6, 1)
    dilation = st.slider("空洞率 dilation", 1, 4, 1)
    groups = st.select_slider("分组数 groups", options=[1, 2, 4], value=2)
    output_padding = st.slider("转置卷积 output_padding", 0, 3, 0)
    seed = st.slider("随机种子", 0, 99, 7)

input_tensor = make_demo_input(input_size, in_channels, seed=seed)

with tab_conv:
    st.subheader("可调卷积与输出尺寸")
    st.markdown(
        '<div class="note">输出尺寸由输入尺寸、卷积核、步长、填充和空洞率共同决定；分组和深度可分离卷积主要改变连接方式与参数量。</div>',
        unsafe_allow_html=True,
    )
    try:
        result = apply_conv(
            input_tensor,
            conv_type,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            out_channels,
            output_padding,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("输入形状", f"{in_channels} x {input_size} x {input_size}")
        c2.metric("输出形状", f"{result.output.shape[1]} x {result.output.shape[2]} x {result.output.shape[3]}")
        c3.metric("有效感受野", f"{result.effective_kernel} x {result.effective_kernel}")
        c4.metric("参数量", f"{result.params:,}")
        st.markdown(f'<div class="formula">{result.formula}</div>', unsafe_allow_html=True)
        st.pyplot(plot_feature_maps(input_tensor, result.output), clear_figure=True)
        render_conv_experiment_guide(
            conv_type,
            input_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
            result,
        )
    except RuntimeError as exc:
        st.error(f"当前参数无法完成卷积计算：{exc}")
    except ValueError as exc:
        st.error(f"当前参数组合无效：{exc}")

with tab_overview:
    st.subheader("各种卷积类型可视化")
    left, right = st.columns([2, 1])
    with left:
        st.pyplot(plot_conv_type_overview(make_demo_input(32, 4, seed=11)), clear_figure=True)
    with right:
        st.markdown(
            """
            <div class="small-muted">
            <b>1x1 卷积</b>：只混合通道，不扩大空间邻域。<br><br>
            <b>3x3 / 5x5 卷积</b>：核越大，单层看到的局部范围越大，参数也更多。<br><br>
            <b>转置卷积</b>：常用于上采样，输出尺寸按反向尺寸公式扩大。<br><br>
            <b>空洞卷积</b>：在核元素之间插空，不增加参数也能扩大感受野。<br><br>
            <b>深度可分离卷积</b>：先逐通道空间卷积，再用 1x1 混合通道。<br><br>
            <b>分组卷积</b>：把通道分成若干组，各组独立卷积，减少跨组连接。
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_conv_overview_guide()

with tab_pool:
    st.subheader("池化层过程演示")
    pool_cols = st.columns([1, 1, 3])
    with pool_cols[0]:
        pool_kernel = st.slider("池化核", 2, 4, 2)
    with pool_cols[1]:
        pool_stride = st.slider("池化步长", 1, 4, 2)
    st.pyplot(plot_pooling_demo(pool_kernel, pool_stride), clear_figure=True)
    st.markdown(
        '<div class="note">最大池化保留局部最强响应，平均池化保留局部平均趋势，全局平均池化把每个通道压成一个数，常用于分类头前替代大规模全连接层。</div>',
        unsafe_allow_html=True,
    )
    render_pooling_guide(pool_kernel, pool_stride)

with tab_bn:
    st.subheader("批归一化对数据分布的影响")
    bn_cols = st.columns(3)
    with bn_cols[0]:
        bn_batch = st.slider("batch size", 2, 64, 16)
    with bn_cols[1]:
        bn_channels = st.slider("通道数", 2, 12, 6)
    with bn_cols[2]:
        bn_seed = st.slider("BN 随机种子", 0, 99, 12)
    st.pyplot(plot_bn_demo(bn_batch, bn_channels, bn_seed), clear_figure=True)
    st.markdown(
        '<div class="note">训练模式下，BN 在每个通道上用当前 batch 的均值和方差做标准化，使各通道更接近均值 0、标准差 1，再由可学习参数缩放和平移。</div>',
        unsafe_allow_html=True,
    )
    render_bn_guide(bn_batch, bn_channels)

with tab_dropout:
    st.subheader("Dropout 开启 / 关闭对比")
    d_cols = st.columns(3)
    with d_cols[0]:
        drop_prob = st.slider("丢弃概率 p", 0.0, 0.9, 0.35, 0.05)
    with d_cols[1]:
        training = st.toggle("训练模式：开启 Dropout", value=True)
    with d_cols[2]:
        drop_seed = st.slider("Dropout 随机种子", 0, 99, 5)
    fig, zero_ratio = plot_dropout_demo(drop_prob, training, drop_seed)
    c1, c2 = st.columns(2)
    c1.metric("当前模式", "训练，随机丢弃" if training else "推理，直接通过")
    c2.metric("输出零值比例", f"{zero_ratio:.1%}")
    st.pyplot(fig, clear_figure=True)
    st.markdown(
        '<div class="note">训练时 Dropout 随机屏蔽一部分激活，并按保留概率缩放剩余激活；推理时关闭随机屏蔽，输出保持稳定。</div>',
        unsafe_allow_html=True,
    )
    render_dropout_guide(drop_prob, training, zero_ratio)
