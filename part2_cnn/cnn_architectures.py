"""
Classic CNN architecture visual lab.

Run:
    streamlit run part2_cnn/cnn_architectures.py
or:
    python main.py part2_cnn/cnn_architectures
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots


torch.set_num_threads(1)

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(
    page_title="经典 CNN 架构与高级应用",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172026;
        --muted: #596772;
        --line: #d8dee3;
        --paper: #f8f6f1;
        --panel: #ffffff;
        --teal: #0f8b8d;
        --rose: #bf3f5b;
        --amber: #c4871f;
        --green: #3f7d58;
        --blue: #3268a8;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(238,246,243,0.96) 100%),
            #fbfaf6;
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.2rem;
    }
    h1, h2, h3 { letter-spacing: 0; }
    section[data-testid="stSidebar"] {
        background: #eef4f2;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 1.08;
        margin: 0;
    }
    .hero p {
        color: var(--muted);
        max-width: 980px;
        line-height: 1.75;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid var(--teal);
        background: rgba(255,255,255,0.74);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        color: #26343b;
        line-height: 1.7;
        margin: 0.35rem 0 0.85rem 0;
    }
    .callout {
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        color: #2b3941;
        line-height: 1.68;
        margin: 0.35rem 0 0.75rem 0;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 0.85rem 0;
    }
    .mini-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 116px;
    }
    .mini-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .mini-card p {
        color: var(--muted);
        margin: 0;
        line-height: 1.62;
        font-size: 0.92rem;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.58;
    }
    @media (max-width: 900px) {
        .mini-grid { grid-template-columns: 1fr; }
        .mini-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class Architecture:
    name: str
    year: int
    params_m: float
    top1_error: float | None
    input_size: str
    key_idea: str
    tradeoff: str
    stages: tuple[tuple[str, str, str], ...]
    forward: tuple[tuple[str, str, str, str], ...]
    notes: tuple[str, ...]


ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture(
        name="LeNet-5",
        year=1998,
        params_m=0.06,
        top1_error=None,
        input_size="32x32 灰度",
        key_idea="卷积 + 池化 + 全连接，把手写数字识别拆成局部特征逐级组合。",
        tradeoff="参数极少，适合小图像；深度和表达能力有限。",
        stages=(
            ("Input", "32x32x1", "输入图像"),
            ("C1", "28x28x6", "5x5 卷积"),
            ("S2", "14x14x6", "平均池化"),
            ("C3", "10x10x16", "5x5 卷积"),
            ("S4", "5x5x16", "平均池化"),
            ("FC", "120 -> 84 -> 10", "分类器"),
        ),
        forward=(
            ("输入", "1 x 32 x 32", "0.001M", "像素网格"),
            ("Conv 5x5", "6 x 28 x 28", "0.0002M", "边缘/笔画"),
            ("AvgPool", "6 x 14 x 14", "0", "平移容忍"),
            ("Conv 5x5", "16 x 10 x 10", "0.002M", "笔画组合"),
            ("AvgPool", "16 x 5 x 5", "0", "紧凑表示"),
            ("FC", "10", "0.058M", "类别得分"),
        ),
        notes=("第一代成功 CNN。", "证明局部连接和权值共享可以大幅减少参数。"),
    ),
    Architecture(
        name="AlexNet",
        year=2012,
        params_m=61.0,
        top1_error=36.7,
        input_size="224x224 RGB",
        key_idea="更深更宽的卷积网络，ReLU、Dropout、数据增强和 GPU 训练共同放大规模。",
        tradeoff="准确率跃迁明显，但全连接层很大，参数和显存开销高。",
        stages=(
            ("Input", "224x224x3", "ImageNet 图像"),
            ("Conv1", "55x55x96", "11x11 stride 4"),
            ("Pool", "27x27x96", "最大池化"),
            ("Conv2", "27x27x256", "5x5 卷积"),
            ("Conv3-5", "13x13x384/256", "连续 3x3 卷积"),
            ("FC", "4096 -> 4096 -> 1000", "Dropout 分类"),
        ),
        forward=(
            ("输入", "3 x 224 x 224", "0", "RGB 图像"),
            ("Conv 11x11/s4", "96 x 55 x 55", "0.035M", "大感受野低级特征"),
            ("Pool + Conv", "256 x 27 x 27", "0.615M", "纹理和局部模式"),
            ("Conv 堆叠", "256 x 13 x 13", "2.2M", "部件级特征"),
            ("Flatten", "43264", "0", "展开空间特征"),
            ("FC 分类器", "1000", "58M+", "类别得分"),
        ),
        notes=("ImageNet 时代的转折点。", "ReLU 让训练速度明显快于 tanh/sigmoid。"),
    ),
    Architecture(
        name="VGGNet",
        year=2014,
        params_m=138.0,
        top1_error=28.5,
        input_size="224x224 RGB",
        key_idea="只用 3x3 卷积，通过重复堆叠增加深度和非线性。",
        tradeoff="结构极其规整，迁移学习友好；参数量巨大，计算昂贵。",
        stages=(
            ("Input", "224x224x3", "输入图像"),
            ("Block1", "224x224x64", "2 个 3x3 Conv"),
            ("Block2", "112x112x128", "2 个 3x3 Conv"),
            ("Block3", "56x56x256", "3 个 3x3 Conv"),
            ("Block4/5", "28/14x512", "更深卷积堆叠"),
            ("FC", "4096 -> 4096 -> 1000", "大分类头"),
        ),
        forward=(
            ("输入", "3 x 224 x 224", "0", "RGB 图像"),
            ("Conv Block 1", "64 x 112 x 112", "0.039M", "边缘"),
            ("Conv Block 2", "128 x 56 x 56", "0.22M", "纹理"),
            ("Conv Block 3", "256 x 28 x 28", "1.5M", "局部部件"),
            ("Conv Block 4/5", "512 x 7 x 7", "12.9M", "语义部件"),
            ("FC 分类器", "1000", "123M+", "类别得分"),
        ),
        notes=("3x3 卷积成为长期默认选择。", "VGG 特征曾是检测、分割和风格迁移的常用骨干。"),
    ),
    Architecture(
        name="GoogLeNet/Inception",
        year=2014,
        params_m=6.8,
        top1_error=30.2,
        input_size="224x224 RGB",
        key_idea="Inception 模块并行使用 1x1、3x3、5x5 和池化，自动融合多尺度特征。",
        tradeoff="参数效率高，但模块分支复杂，工程实现不如 VGG 简洁。",
        stages=(
            ("Stem", "112x112x64", "7x7 卷积 + 池化"),
            ("Early", "56x56x192", "1x1/3x3 卷积"),
            ("Inception 3", "28x28x480", "多分支融合"),
            ("Inception 4", "14x14x832", "多尺度语义"),
            ("Inception 5", "7x7x1024", "高层语义"),
            ("GAP", "1000", "全局平均池化"),
        ),
        forward=(
            ("输入", "3 x 224 x 224", "0", "RGB 图像"),
            ("Stem", "192 x 28 x 28", "0.12M", "基础特征"),
            ("Inception 3a/3b", "480 x 14 x 14", "0.75M", "多尺度局部模式"),
            ("Inception 4x", "832 x 7 x 7", "3.8M", "对象部件"),
            ("Inception 5x", "1024 x 7 x 7", "2.0M", "类别语义"),
            ("GAP + Linear", "1000", "1.0M", "轻量分类头"),
        ),
        notes=("用 1x1 卷积先降维再做大核卷积。", "全局平均池化减少了大 FC 层。"),
    ),
    Architecture(
        name="ResNet",
        year=2015,
        params_m=25.6,
        top1_error=24.7,
        input_size="224x224 RGB",
        key_idea="残差连接学习 F(x)，输出 F(x)+x，让深层网络更容易优化。",
        tradeoff="深度可扩展性极强；训练和推理仍有较大计算量。",
        stages=(
            ("Stem", "112x112x64", "7x7 卷积"),
            ("Stage1", "56x56x256", "残差块 x3"),
            ("Stage2", "28x28x512", "残差块 x4"),
            ("Stage3", "14x14x1024", "残差块 x6"),
            ("Stage4", "7x7x2048", "残差块 x3"),
            ("Head", "1000", "GAP + FC"),
        ),
        forward=(
            ("输入", "3 x 224 x 224", "0", "RGB 图像"),
            ("Stem", "64 x 56 x 56", "0.009M", "初级特征"),
            ("Residual Stage 1", "256 x 56 x 56", "0.21M", "浅层残差"),
            ("Residual Stage 2/3", "1024 x 14 x 14", "6.0M", "中高层残差"),
            ("Residual Stage 4", "2048 x 7 x 7", "14.9M", "语义特征"),
            ("GAP + FC", "1000", "2.0M", "类别得分"),
        ),
        notes=("解决了非常深网络的退化问题。", "现代视觉骨干网络的共同祖先之一。"),
    ),
    Architecture(
        name="DenseNet",
        year=2017,
        params_m=8.0,
        top1_error=25.0,
        input_size="224x224 RGB",
        key_idea="每一层都接收前面所有层的特征，使用 concat 强化特征复用。",
        tradeoff="参数效率好，梯度流顺畅；特征拼接带来显存访问压力。",
        stages=(
            ("Stem", "112x112x64", "7x7 卷积"),
            ("DenseBlock1", "56x56x256", "6 层密集连接"),
            ("Transition", "28x28x128", "1x1 + AvgPool"),
            ("DenseBlock2/3", "14x14x1024", "特征持续累积"),
            ("DenseBlock4", "7x7x1024", "高层融合"),
            ("Head", "1000", "GAP + FC"),
        ),
        forward=(
            ("输入", "3 x 224 x 224", "0", "RGB 图像"),
            ("Stem", "64 x 56 x 56", "0.009M", "初级特征"),
            ("Dense Block 1", "256 x 56 x 56", "0.33M", "复用浅层特征"),
            ("Dense Block 2/3", "1024 x 14 x 14", "5.7M", "密集语义积累"),
            ("Dense Block 4", "1024 x 7 x 7", "1.9M", "最终特征池"),
            ("GAP + FC", "1000", "1.0M", "类别得分"),
        ),
        notes=("concat 而不是 add。", "growth rate 控制每层新增通道数。"),
    ),
    Architecture(
        name="MobileNet",
        year=2017,
        params_m=4.2,
        top1_error=29.4,
        input_size="224x224 RGB",
        key_idea="深度可分离卷积把空间卷积和通道混合拆开，大幅降低参数和 FLOPs。",
        tradeoff="移动端友好；在极小模型下精度和表达能力会下降。",
        stages=(
            ("Stem", "112x112x32", "标准 3x3 卷积"),
            ("DW+PW", "112x112x64", "深度卷积 + 1x1"),
            ("Downsample", "56/28x128/256", "逐步降采样"),
            ("Bottleneck", "14x14x512", "重复轻量块"),
            ("Final", "7x7x1024", "高层特征"),
            ("Head", "1000", "GAP + FC"),
        ),
        forward=(
            ("输入", "3 x 224 x 224", "0", "RGB 图像"),
            ("Conv Stem", "32 x 112 x 112", "0.0009M", "初级特征"),
            ("Depthwise", "32 x 112 x 112", "0.0003M", "逐通道空间滤波"),
            ("Pointwise", "64 x 112 x 112", "0.002M", "通道混合"),
            ("DW/PW 堆叠", "1024 x 7 x 7", "3.1M", "轻量语义提取"),
            ("GAP + FC", "1000", "1.0M", "类别得分"),
        ),
        notes=("标准 3x3 卷积参数约为 Cin*Cout*9。", "深度可分离卷积约为 Cin*9 + Cin*Cout。"),
    ),
)

ARCH_BY_NAME = {arch.name: arch for arch in ARCHITECTURES}

DETECTION_ROWS = [
    {
        "路线": "R-CNN",
        "核心流程": "候选区域 -> 每个区域跑 CNN -> SVM/回归",
        "特点": "准确但很慢，训练流程分散。",
    },
    {
        "路线": "Fast R-CNN",
        "核心流程": "整图 CNN -> RoI Pooling -> 分类和框回归",
        "特点": "共享卷积特征，速度大幅提升。",
    },
    {
        "路线": "Faster R-CNN",
        "核心流程": "RPN 生成候选框 -> RoI Head 精修",
        "特点": "两阶段检测的经典范式。",
    },
    {
        "路线": "YOLO",
        "核心流程": "单次前向直接预测类别和框",
        "特点": "一阶段检测，速度快，端到端。",
    },
    {
        "路线": "YOLOv3-v8+",
        "核心流程": "多尺度特征金字塔 + 解耦头/anchor-free 等改进",
        "特点": "实时检测主力路线之一。",
    },
]

SEGMENTATION_ROWS = [
    {
        "任务": "语义分割",
        "代表": "FCN",
        "输出": "每个像素一个类别，不区分同类实例。",
        "关键点": "把分类网络的全连接层换成卷积层，再上采样回原图。",
    },
    {
        "任务": "语义分割",
        "代表": "U-Net",
        "输出": "医学/小数据场景常用的像素分类。",
        "关键点": "编码器-解码器 + 跳跃连接，保留细节位置。",
    },
    {
        "任务": "实例分割",
        "代表": "Mask R-CNN",
        "输出": "每个目标一个框、类别和独立 mask。",
        "关键点": "Faster R-CNN 上增加 mask 分支，RoIAlign 改善对齐。",
    },
]


def architecture_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "网络": arch.name,
                "年份": arch.year,
                "参数量(M)": arch.params_m,
                "典型输入": arch.input_size,
                "核心思想": arch.key_idea,
            }
            for arch in ARCHITECTURES
        ]
    )


def plot_timeline() -> go.Figure:
    years = [arch.year for arch in ARCHITECTURES]
    names = [arch.name for arch in ARCHITECTURES]
    params = [arch.params_m for arch in ARCHITECTURES]
    y = [1 + 0.16 * ((i % 2) * 2 - 1) for i in range(len(ARCHITECTURES))]
    colors = ["#0f8b8d", "#bf3f5b", "#c4871f", "#3268a8", "#3f7d58", "#6b5fb5", "#5f7f3a"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=y,
            mode="lines+markers+text",
            line=dict(color="#87939b", width=2),
            marker=dict(size=[10 + math.log10(p + 1) * 8 for p in params], color=colors),
            text=names,
            textposition=["top center" if i % 2 == 0 else "bottom center" for i in range(len(names))],
            hovertemplate="<b>%{text}</b><br>%{x}<extra></extra>",
        )
    )
    for arch, yy, color in zip(ARCHITECTURES, y, colors, strict=True):
        fig.add_annotation(
            x=arch.year,
            y=yy + (0.22 if yy > 1 else -0.22),
            text=arch.key_idea,
            showarrow=False,
            font=dict(size=11, color="#3f4c54"),
            align="center",
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
        )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="", tickmode="linear", dtick=2, range=[1996, 2019]),
        yaxis=dict(visible=False, range=[0.45, 1.55]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_param_compare() -> go.Figure:
    df = architecture_dataframe()
    fig = go.Figure(
        go.Bar(
            x=df["网络"],
            y=df["参数量(M)"],
            marker_color=["#0f8b8d", "#bf3f5b", "#c4871f", "#3268a8", "#3f7d58", "#6b5fb5", "#5f7f3a"],
            text=[f"{v:g}M" if v >= 1 else f"{v * 1000:.0f}K" for v in df["参数量(M)"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>参数量: %{y:g}M<extra></extra>",
        )
    )
    fig.update_yaxes(type="log", title="参数量，log scale")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
    )
    return fig


def plot_architecture_blocks(arch: Architecture, active_idx: int | None = None) -> go.Figure:
    n = len(arch.stages)
    fig = go.Figure()
    palette = ["#0f8b8d", "#3268a8", "#c4871f", "#bf3f5b", "#3f7d58", "#6b5fb5"]

    for i, (name, shape, desc) in enumerate(arch.stages):
        x0 = i * 1.55
        x1 = x0 + 1.18
        color = palette[i % len(palette)]
        opacity = 0.92 if active_idx is None or i == active_idx else 0.35
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0.22,
            y1=0.88,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=opacity,
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.55,
            text=f"<b>{name}</b><br>{shape}<br><span style='font-size:11px'>{desc}</span>",
            showarrow=False,
            font=dict(size=12, color="#ffffff"),
            align="center",
        )
        if i < n - 1:
            fig.add_annotation(
                x=x1 + 0.18,
                y=0.55,
                ax=x1 - 0.02,
                ay=0.55,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#52616b",
            )

    fig.update_xaxes(visible=False, range=[-0.25, (n - 1) * 1.55 + 1.35])
    fig.update_yaxes(visible=False, range=[0, 1.05])
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def forward_dataframe(arch: Architecture) -> pd.DataFrame:
    rows = []
    for idx, (stage, shape, params, meaning) in enumerate(arch.forward, 1):
        rows.append({"步骤": idx, "层/模块": stage, "输出张量": shape, "本段参数": params, "学到什么": meaning})
    return pd.DataFrame(rows)


def make_demo_image(size: int = 64, variant: str = "几何图形") -> torch.Tensor:
    y, x = np.mgrid[0:size, 0:size]
    xx = x / (size - 1)
    yy = y / (size - 1)
    img = np.zeros((size, size), dtype=np.float32)

    if variant == "几何图形":
        img[10:34, 8:30] = 0.85
        circle = ((x - 44) ** 2 + (y - 42) ** 2) < 13**2
        img[circle] = 1.0
        img += 0.35 * np.exp(-((xx - yy) ** 2) / 0.002)
    elif variant == "条纹纹理":
        img = 0.45 + 0.35 * np.sin(2 * np.pi * (xx * 6 + yy * 1.4))
        img += 0.2 * np.sin(2 * np.pi * yy * 11)
    else:
        img = 0.3 * xx + 0.25 * yy
        img += np.exp(-((xx - 0.33) ** 2 + (yy - 0.35) ** 2) / 0.012)
        img += 0.75 * np.exp(-((xx - 0.70) ** 2 + (yy - 0.62) ** 2) / 0.018)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return torch.from_numpy(img.astype(np.float32))[None, None]


def kernel_bank() -> torch.Tensor:
    kernels = np.array(
        [
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
            [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
        ],
        dtype=np.float32,
    )
    kernels[3] /= 9.0
    kernels[:3] /= np.abs(kernels[:3]).sum(axis=(1, 2), keepdims=True)
    kernels[4:] /= np.abs(kernels[4:]).sum(axis=(1, 2), keepdims=True)
    return torch.from_numpy(kernels[:, None])


def normalize_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def compute_feature_layers(image: torch.Tensor) -> dict[str, torch.Tensor]:
    bank = kernel_bank()
    early = F.relu(F.conv2d(image, bank, padding=1))
    pooled = F.avg_pool2d(early, 2)

    mix_weight = torch.zeros(8, early.shape[1], 3, 3)
    base = kernel_bank()
    for out_ch in range(8):
        for in_ch in range(early.shape[1]):
            mix_weight[out_ch, in_ch] = base[(out_ch + in_ch) % base.shape[0], 0] / max(early.shape[1], 1)
    middle = F.relu(F.conv2d(pooled, mix_weight, padding=1))
    deep = F.avg_pool2d(middle, 2)
    deep_weight = torch.zeros(6, middle.shape[1], 3, 3)
    for out_ch in range(deep_weight.shape[0]):
        for in_ch in range(deep_weight.shape[1]):
            deep_weight[out_ch, in_ch] = base[(out_ch * 2 + in_ch) % base.shape[0], 0] / max(middle.shape[1], 1)
    deep = F.relu(F.conv2d(deep, torch.flip(deep_weight, dims=[2]), padding=1))
    return {
        "输入": image,
        "浅层：边缘/方向": early,
        "中层：纹理/局部部件": middle,
        "深层：语义雏形": deep,
    }


def feature_maps_figure(features: torch.Tensor, title: str, max_channels: int = 6) -> plt.Figure:
    arr = features.detach().cpu().numpy()[0]
    n = min(max_channels, arr.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.25))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(normalize_np(arr[i]), cmap="magma")
        ax.set_title(f"ch {i + 1}", fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_residual_demo(scale: float, use_projection: bool) -> go.Figure:
    t = np.linspace(-2.2, 2.2, 160)
    x = np.tanh(t)
    residual = scale * (0.55 * np.sin(2.4 * t) + 0.28 * t)
    shortcut = 0.78 * x if use_projection else x
    y_plain = residual
    y_res = residual + shortcut

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, name="输入 x", line=dict(color="#596772", width=2)))
    fig.add_trace(go.Scatter(x=t, y=y_plain, name="普通层 F(x)", line=dict(color="#bf3f5b", width=2)))
    fig.add_trace(go.Scatter(x=t, y=y_res, name="残差层 F(x)+x", line=dict(color="#0f8b8d", width=3)))
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="样本位置",
        yaxis_title="激活值",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def plot_residual_block(use_projection: bool) -> go.Figure:
    fig = go.Figure()
    boxes = [
        ("x", 0.1, 0.5, "#596772"),
        ("Conv 3x3\nBN + ReLU", 1.2, 0.5, "#3268a8"),
        ("Conv 3x3\nBN", 2.4, 0.5, "#3268a8"),
        ("Add", 3.45, 0.5, "#c4871f"),
        ("ReLU", 4.35, 0.5, "#3f7d58"),
    ]
    if use_projection:
        boxes.append(("1x1 投影", 2.35, 0.12, "#6b5fb5"))
    for text, x, y, color in boxes:
        fig.add_shape(type="rect", x0=x - 0.42, x1=x + 0.42, y0=y - 0.16, y1=y + 0.16, fillcolor=color, line=dict(color=color))
        fig.add_annotation(x=x, y=y, text=text.replace("\n", "<br>"), showarrow=False, font=dict(color="#ffffff", size=12))

    arrows = [((0.52, 0.5), (0.78, 0.5)), ((1.62, 0.5), (1.98, 0.5)), ((2.82, 0.5), (3.08, 0.5)), ((3.84, 0.5), (3.95, 0.5))]
    for (x0, y0), (x1, y1) in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#52616b")
    shortcut_y = 0.12 if use_projection else 0.26
    fig.add_annotation(x=3.02, y=shortcut_y, ax=0.52, ay=shortcut_y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#0f8b8d")
    fig.add_shape(type="line", x0=0.52, x1=0.52, y0=0.5, y1=shortcut_y, line=dict(color="#0f8b8d", width=2))
    fig.add_shape(type="line", x0=3.02, x1=3.02, y0=shortcut_y, y1=0.5, line=dict(color="#0f8b8d", width=2))

    fig.update_xaxes(visible=False, range=[-0.45, 4.9])
    fig.update_yaxes(visible=False, range=[-0.05, 0.8])
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def inception_outputs(image: torch.Tensor) -> dict[str, torch.Tensor]:
    blur5 = torch.ones(1, 1, 5, 5) / 25.0
    edge3 = kernel_bank()[1:2]
    lap3 = kernel_bank()[0:1]
    pointwise = image * 0.9 + 0.05
    return {
        "1x1：保留/重混合": pointwise,
        "3x3：边缘纹理": F.relu(F.conv2d(image, edge3, padding=1)),
        "5x5：更大上下文": F.relu(F.conv2d(image, blur5, padding=2)),
        "Pool+1x1：稳健摘要": F.avg_pool2d(image, 3, stride=1, padding=1),
        "Concat：通道拼接": torch.cat(
            [
                pointwise,
                F.relu(F.conv2d(image, edge3, padding=1)),
                F.relu(F.conv2d(image, blur5, padding=2)),
                F.avg_pool2d(image, 3, stride=1, padding=1),
            ],
            dim=1,
        ),
    }


def inception_figure(outputs: dict[str, torch.Tensor]) -> plt.Figure:
    names = list(outputs.keys())[:4]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.7))
    for ax, name in zip(axes, names, strict=True):
        arr = outputs[name].detach().cpu().numpy()[0, 0]
        ax.imshow(normalize_np(arr), cmap="viridis")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_inception_block() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.05, x1=0.75, y0=0.38, y1=0.62, fillcolor="#596772", line=dict(color="#596772"))
    fig.add_annotation(x=0.4, y=0.5, text="输入特征图", showarrow=False, font=dict(color="white"))
    branches = [
        ("1x1", 1.55, 0.78, "#0f8b8d"),
        ("1x1 -> 3x3", 1.55, 0.58, "#3268a8"),
        ("1x1 -> 5x5", 1.55, 0.38, "#bf3f5b"),
        ("Pool -> 1x1", 1.55, 0.18, "#c4871f"),
    ]
    for text, x, y, color in branches:
        fig.add_shape(type="rect", x0=x - 0.48, x1=x + 0.48, y0=y - 0.08, y1=y + 0.08, fillcolor=color, line=dict(color=color))
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color="white", size=12))
        fig.add_annotation(x=x - 0.52, y=y, ax=0.75, ay=0.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.8, arrowcolor="#52616b")
        fig.add_annotation(x=2.58, y=0.5, ax=x + 0.5, ay=y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.8, arrowcolor="#52616b")
    fig.add_shape(type="rect", x0=2.62, x1=3.32, y0=0.38, y1=0.62, fillcolor="#3f7d58", line=dict(color="#3f7d58"))
    fig.add_annotation(x=2.97, y=0.5, text="Concat<br>通道拼接", showarrow=False, font=dict(color="white"))
    fig.update_xaxes(visible=False, range=[-0.1, 3.55])
    fig.update_yaxes(visible=False, range=[0.02, 0.92])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_detection_concepts() -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("两阶段：R-CNN / Faster R-CNN", "一阶段：YOLO"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    two_stage = [("图像", 0.2), ("候选框/RPN", 1.25), ("RoI 特征", 2.35), ("分类 + 框回归", 3.55)]
    one_stage = [("图像", 0.2), ("网格/特征金字塔", 1.45), ("直接预测框+类别", 2.9)]
    for col, seq, color in [(1, two_stage, "#3268a8"), (2, one_stage, "#bf3f5b")]:
        for i, (label, x) in enumerate(seq):
            fig.add_shape(type="rect", x0=x - 0.38, x1=x + 0.38, y0=0.35, y1=0.65, fillcolor=color, line=dict(color=color), row=1, col=col)
            fig.add_annotation(x=x, y=0.5, text=label, showarrow=False, font=dict(color="white", size=12), row=1, col=col)
            if i > 0:
                fig.add_annotation(x=x - 0.42, y=0.5, ax=seq[i - 1][1] + 0.42, ay=0.5, xref=f"x{col if col > 1 else ''}", yref=f"y{col if col > 1 else ''}", axref=f"x{col if col > 1 else ''}", ayref=f"y{col if col > 1 else ''}", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#52616b")
    fig.update_xaxes(visible=False, row=1, col=1, range=[-0.35, 4.1])
    fig.update_xaxes(visible=False, row=1, col=2, range=[-0.35, 3.4])
    fig.update_yaxes(visible=False, row=1, col=1, range=[0.1, 0.9])
    fig.update_yaxes(visible=False, row=1, col=2, range=[0.1, 0.9])
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def segmentation_demo_figure() -> plt.Figure:
    size = 80
    y, x = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = 0.18 + x / size * 0.25
    img[..., 1] = 0.22 + y / size * 0.25
    img[..., 2] = 0.28
    circle = ((x - 26) ** 2 + (y - 42) ** 2) < 15**2
    square = (x > 46) & (x < 68) & (y > 22) & (y < 56)
    img[circle] = [0.88, 0.28, 0.34]
    img[square] = [0.20, 0.62, 0.86]

    semantic = np.zeros((size, size), dtype=np.float32)
    semantic[circle] = 1
    semantic[square] = 2
    instance = np.zeros((size, size, 3), dtype=np.float32)
    instance[circle] = [0.9, 0.25, 0.3]
    instance[square] = [0.2, 0.55, 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img)
    axes[0].set_title("输入图像", fontsize=10)
    axes[1].imshow(semantic, cmap="viridis", vmin=0, vmax=2)
    axes[1].set_title("语义分割：像素类别", fontsize=10)
    axes[2].imshow(img, alpha=0.55)
    axes[2].imshow(instance, alpha=0.55)
    axes[2].set_title("实例分割：目标级 mask", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("导航")
    selected_arch_name = st.selectbox("选择网络", [arch.name for arch in ARCHITECTURES], index=4)
    selected_arch = ARCH_BY_NAME[selected_arch_name]
    section = st.radio(
        "查看内容",
        ["架构演进", "网络结构与前向传播", "特征图可视化", "残差连接", "Inception 模块", "检测与分割概览"],
        index=0,
    )

    st.divider()
    st.caption("所有演示均为本地确定性可视化，不下载预训练权重。")


st.markdown(
    """
    <div class="hero">
        <h1>经典 CNN 架构与高级应用</h1>
        <p>从 LeNet-5 到 MobileNet，观察 CNN 如何在深度、宽度、多尺度、残差连接和轻量化之间演进，并把这些骨干网络连接到检测和分割任务。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if section == "架构演进":
    st.subheader("1. 经典架构演进图")
    st.plotly_chart(plot_timeline(), width="stretch", config=PLOT_CONFIG)

    st.subheader("2. 参数量对比")
    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(plot_param_compare(), width="stretch", config=PLOT_CONFIG)
    with right:
        st.dataframe(architecture_dataframe(), width="stretch", hide_index=True)

    st.markdown(
        """
        <div class="note">
        <strong>读图要点：</strong>AlexNet 和 VGG 主要靠规模和深度推进；GoogLeNet 用多分支和 1x1 降维提高参数效率；
        ResNet 解决深层优化；DenseNet 强调特征复用；MobileNet 则把目标转向移动端计算效率。
        </div>
        """,
        unsafe_allow_html=True,
    )

elif section == "网络结构与前向传播":
    st.subheader(f"{selected_arch.name} 架构图")
    st.plotly_chart(plot_architecture_blocks(selected_arch), width="stretch", config=PLOT_CONFIG)

    m1, m2, m3 = st.columns(3)
    m1.metric("年份", selected_arch.year)
    m2.metric("参数量", f"{selected_arch.params_m:g}M" if selected_arch.params_m >= 1 else f"{selected_arch.params_m * 1000:.0f}K")
    m3.metric("典型输入", selected_arch.input_size)

    st.markdown(f"<div class='callout'><strong>核心思想：</strong>{selected_arch.key_idea}<br><strong>取舍：</strong>{selected_arch.tradeoff}</div>", unsafe_allow_html=True)

    step = st.slider("前向传播步骤", 1, len(selected_arch.forward), len(selected_arch.forward), key="forward_step")
    st.plotly_chart(plot_architecture_blocks(selected_arch, active_idx=min(step - 1, len(selected_arch.stages) - 1)), width="stretch", config=PLOT_CONFIG)

    df = forward_dataframe(selected_arch)
    styled = df.style.apply(lambda row: ["background-color: #e8f4f2" if row["步骤"] == step else "" for _ in row], axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    st.markdown("**结构备注**")
    for note in selected_arch.notes:
        st.write(f"- {note}")

elif section == "特征图可视化":
    st.subheader("3. 特征图可视化：不同层学到的特征")
    input_variant = st.selectbox("输入模式", ["几何图形", "条纹纹理", "亮斑目标"], index=0)
    layer_name = st.select_slider("观察层级", options=["输入", "浅层：边缘/方向", "中层：纹理/局部部件", "深层：语义雏形"], value="浅层：边缘/方向")
    image = make_demo_image(64, input_variant)
    layers = compute_feature_layers(image)

    cols = st.columns([0.95, 1.7])
    with cols[0]:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.imshow(image.detach().numpy()[0, 0], cmap="gray")
        ax.set_title("合成输入", fontsize=11)
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
    with cols[1]:
        st.pyplot(feature_maps_figure(layers[layer_name], layer_name), clear_figure=True)

    st.markdown(
        """
        <div class="note">
        浅层通道通常响应边缘、方向和颜色/亮度变化；中层开始组合纹理和局部部件；深层空间尺寸变小，但语义抽象更强。
        这里用手工卷积核做教学演示，因此显示的是“层级效果”而不是某个真实预训练模型的权重。
        </div>
        """,
        unsafe_allow_html=True,
    )

elif section == "残差连接":
    st.subheader("4. 残差连接的原理演示")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        projection = st.checkbox("输入/输出维度不一致时使用 1x1 投影", value=False)
        st.plotly_chart(plot_residual_block(projection), width="stretch", config=PLOT_CONFIG)
    with col_b:
        scale = st.slider("残差分支 F(x) 强度", 0.0, 1.5, 0.7, 0.05)
        st.plotly_chart(plot_residual_demo(scale, projection), width="stretch", config=PLOT_CONFIG)

    st.markdown(
        """
        <div class="mini-grid">
            <div class="mini-card"><strong>普通层</strong><p>直接学习 y = F(x)。如果层数很深，优化过程可能把已有有用表示破坏掉。</p></div>
            <div class="mini-card"><strong>残差层</strong><p>学习 y = F(x) + x。最差也可以让 F(x) 接近 0，使模块近似恒等映射。</p></div>
            <div class="mini-card"><strong>梯度路径</strong><p>shortcut 给反向传播提供直接路径，缓解深层网络的退化和梯度衰减问题。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"y = F(x) + x,\quad \frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1")

elif section == "Inception 模块":
    st.subheader("5. Inception 模块的多尺度特征融合")
    st.plotly_chart(plot_inception_block(), width="stretch", config=PLOT_CONFIG)
    variant = st.selectbox("输入模式", ["几何图形", "条纹纹理", "亮斑目标"], index=2, key="inception_input")
    image = make_demo_image(64, variant)
    outputs = inception_outputs(image)
    st.pyplot(inception_figure(outputs), clear_figure=True)
    concat = outputs["Concat：通道拼接"]
    c1, c2, c3 = st.columns(3)
    c1.metric("输入通道数", image.shape[1])
    c2.metric("分支数", 4)
    c3.metric("拼接后通道数", concat.shape[1])
    st.markdown(
        """
        <div class="note">
        Inception 的关键不是“选一个核大小”，而是让不同尺度的分支同时工作，再在通道维拼接。
        1x1 卷积常用于降维和通道重混合，使 3x3/5x5 分支的计算成本可控。
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.subheader("6. 目标检测概览：R-CNN 系列与 YOLO 系列")
    st.plotly_chart(plot_detection_concepts(), width="stretch", config=PLOT_CONFIG)
    st.dataframe(pd.DataFrame(DETECTION_ROWS), width="stretch", hide_index=True)

    st.markdown(
        """
        <div class="note">
        两阶段检测先找候选区域再分类和精修，通常更重但定位精细；一阶段检测把定位和分类合并为一次密集预测，更适合实时场景。
        CNN 骨干网络负责提取特征，检测头负责把特征变成框、类别和置信度。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("7. 图像分割概览：语义分割与实例分割")
    left, right = st.columns([1.1, 1.15])
    with left:
        st.pyplot(segmentation_demo_figure(), clear_figure=True)
    with right:
        st.dataframe(pd.DataFrame(SEGMENTATION_ROWS), width="stretch", hide_index=True)
        st.markdown(
            """
            <div class="callout">
            <strong>语义分割</strong>回答“每个像素属于哪一类”；<strong>实例分割</strong>还要回答“这是第几个独立目标”。
            U-Net 的跳跃连接偏向恢复空间细节，Mask R-CNN 则把检测框和像素 mask 结合起来。
            </div>
            """,
            unsafe_allow_html=True,
        )
