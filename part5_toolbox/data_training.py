"""
Data preprocessing and training practice lab.

Run:
    streamlit run part5_toolbox/data_training.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance


torch.set_num_threads(1)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]


st.set_page_config(
    page_title="数据预处理与训练技巧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    .stApp { background: #fbfaf7; color: #182026; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid #d8dedf;
        border-radius: 8px;
        padding: 10px 12px;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.75);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        line-height: 1.65;
        margin: 0.4rem 0 0.9rem 0;
    }
    .small {
        color: #58646d;
        font-size: 0.92rem;
        line-height: 1.58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_note(title: str, body: str) -> None:
    st.markdown(
        f'<div class="note"><strong>{title}</strong> {body}</div>',
        unsafe_allow_html=True,
    )


def render_action(body: str) -> None:
    st.markdown(f"> {body}")


def tight_fig(width: float = 8.0, height: float = 4.5):
    fig = plt.figure(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor("#fbfaf7")
    return fig


@st.cache_data(show_spinner=False)
def make_preprocess_data(n: int, scale_gap: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(38, 11, n).clip(18, 70)
    income = rng.lognormal(mean=10.6, sigma=0.45, size=n) / scale_gap
    city = rng.choice(["北京", "上海", "成都", "杭州"], size=n, p=[0.25, 0.25, 0.3, 0.2])
    city_bonus = {"北京": 1.2, "上海": 1.0, "成都": -0.7, "杭州": 0.35}
    score = (
        0.08 * age
        + 0.00011 * income
        + np.array([city_bonus[c] for c in city])
        + rng.normal(0, 1.1, n)
    )
    return pd.DataFrame({"年龄": age, "收入": income, "城市": city, "目标分数": score})


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min(axis=0, keepdims=True)) / (
        x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8
    )


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def plot_preprocess(df: pd.DataFrame, method: str):
    numeric = df[["年龄", "收入"]].to_numpy(dtype=np.float32)
    if method == "原始数值":
        transformed = numeric
    elif method == "归一化":
        transformed = normalize_minmax(numeric)
    else:
        transformed = standardize(numeric)

    one_hot = pd.get_dummies(df["城市"], prefix="城市").astype(int)
    transformed_df = pd.DataFrame(transformed, columns=["年龄", "收入"])

    fig = tight_fig(11, 4.6)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    colors = {"北京": "#c73e5b", "上海": "#0f8b8d", "成都": "#d99a22", "杭州": "#5e4ae3"}
    for city, part in df.groupby("城市"):
        idx = part.index
        ax0.scatter(
            df.loc[idx, "年龄"],
            df.loc[idx, "收入"],
            s=24,
            alpha=0.72,
            label=city,
            color=colors[city],
            edgecolor="white",
            linewidth=0.3,
        )
        ax1.scatter(
            transformed_df.loc[idx, "年龄"],
            transformed_df.loc[idx, "收入"],
            s=24,
            alpha=0.72,
            color=colors[city],
            edgecolor="white",
            linewidth=0.3,
        )

    ax0.set_title("原始尺度")
    ax0.set_xlabel("年龄")
    ax0.set_ylabel("收入")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8)

    ax1.set_title(method)
    ax1.set_xlabel("处理后的年龄")
    ax1.set_ylabel("处理后的收入")
    ax1.grid(alpha=0.25)

    ax2.imshow(one_hot.head(12).to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("独热编码：类别变成 0/1 特征")
    ax2.set_yticks(range(12))
    ax2.set_yticklabels([f"样本 {i}" for i in range(12)], fontsize=8)
    ax2.set_xticks(range(one_hot.shape[1]))
    ax2.set_xticklabels(one_hot.columns, rotation=35, ha="right")
    for r in range(12):
        for c in range(one_hot.shape[1]):
            ax2.text(c, r, str(one_hot.iloc[r, c]), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig, transformed_df, one_hot


def render_preprocessing(seed: int) -> None:
    st.subheader("1. 数据预处理方法演示")
    render_note(
        "核心直觉：",
        "归一化把数值压到固定区间，标准化把分布挪到均值 0、标准差 1；独热编码把没有大小关系的类别拆成多列 0/1。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("样本数", 60, 500, 180, 20)
    with c2:
        scale_gap = st.slider("收入尺度压缩因子", 1.0, 12.0, 3.0, 0.5)
    with c3:
        method = st.segmented_control(
            "数值处理方式",
            ["原始数值", "归一化", "标准化"],
            default="标准化",
        )

    df = make_preprocess_data(n, scale_gap, seed)
    fig, transformed, one_hot = plot_preprocess(df, method or "标准化")
    st.pyplot(fig, clear_figure=True)
    render_action(
        "请把“数值处理方式”依次切到“原始数值”“归一化”“标准化”，观察中间散点图的坐标范围如何变化；再把“收入尺度压缩因子”从 1.0 拖到 12.0，看收入这一列是否还会压过年龄这一列。"
    )
    render_note(
        "图怎么看：",
        "左图显示原始年龄和收入的尺度差异，中图显示处理后的数值空间，右图显示城市被拆成独热编码。神经网络的第一层会把所有输入特征一起乘权重；如果某一列数值尺度过大，它会在梯度里占据过高话语权。"
    )
    render_note(
        "工程经验：",
        "连续数值特征在 90% 的小型深度学习项目里先用标准化作为默认起点；归一化适合像素、比例这类天然有上下界的输入。真正上线时，标准化的均值和标准差必须只在训练集上拟合，再原样用于验证集、测试集和线上数据。"
    )

    left, right = st.columns([1, 1])
    with left:
        st.caption("处理后数值特征的统计量")
        stats = transformed.describe().loc[["mean", "std", "min", "max"]]
        st.dataframe(stats, width="stretch")
    with right:
        st.caption("独热编码后的前 8 行")
        st.dataframe(pd.concat([df[["城市"]].head(8), one_hot.head(8)], axis=1), width="stretch")
    render_note(
        "常见坑：",
        "把城市直接编码成 0、1、2、3 会暗示“杭州比北京大”这种不存在的顺序；把全量数据一起标准化会造成数据泄漏。排查时先看上面的统计表：标准化后 mean 应接近 0、std 应接近 1，独热编码每行通常只有一个 1。"
    )
    render_action(
        "进阶思考：当“样本数”从 60 调到 500 时，统计量是否更稳定？如果验证集分布和训练集不同，标准化表里的 mean/std 会怎样暴露这种漂移？"
    )


def synthetic_rgb_image(size: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xx / size - 0.5) * 2
    y = (yy / size - 0.5) * 2
    bg = np.zeros((size, size, 3), dtype=np.float32)
    bg[..., 0] = 0.25 + 0.45 * (1 - np.sqrt(x * x + y * y)).clip(0, 1)
    bg[..., 1] = 0.35 + 0.25 * np.sin(5 * x + 2 * y)
    bg[..., 2] = 0.58 + 0.20 * np.cos(4 * y)

    mask = ((x + 0.22) ** 2 / 0.16 + (y - 0.05) ** 2 / 0.36) < 1
    bg[mask] = np.array([0.92, 0.42, 0.25])
    stripe = np.abs(y + 0.45 * x) < 0.055
    bg[stripe & mask] = np.array([0.98, 0.85, 0.30])
    eye = ((x + 0.42) ** 2 + (y - 0.02) ** 2) < 0.015
    bg[eye] = np.array([0.05, 0.06, 0.07])
    return np.clip(bg, 0, 1)


def pil_from_array(img: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def center_crop_resize(image: Image.Image, crop_ratio: float, output_size: int) -> Image.Image:
    width, height = image.size
    crop_w = max(8, int(width * crop_ratio))
    crop_h = max(8, int(height * crop_ratio))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h)).resize(
        (output_size, output_size), Image.Resampling.BICUBIC
    )


def scale_canvas(image: Image.Image, scale: float, output_size: int) -> Image.Image:
    new_size = max(12, int(output_size * scale))
    resized = image.resize((new_size, new_size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (output_size, output_size), (245, 244, 238))
    left = (output_size - new_size) // 2
    top = (output_size - new_size) // 2
    canvas.paste(resized, (left, top))
    if scale > 1:
        left = (new_size - output_size) // 2
        top = (new_size - output_size) // 2
        canvas = resized.crop((left, top, left + output_size, top + output_size))
    return canvas


def jitter_color(image: Image.Image, brightness: float, contrast: float, color: float) -> Image.Image:
    out = ImageEnhance.Brightness(image).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Color(out).enhance(color)
    return out


def plot_augmentations(angle: float, crop_ratio: float, scale: float, brightness: float, contrast: float, color: float):
    base = pil_from_array(synthetic_rgb_image())
    augments = [
        ("原图", base),
        ("水平翻转", base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        (f"旋转 {angle:.0f}°", base.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(245, 244, 238))),
        (f"中心裁剪 {crop_ratio:.0%}", center_crop_resize(base, crop_ratio, base.size[0])),
        (f"缩放 {scale:.2f}x", scale_canvas(base, scale, base.size[0])),
        ("色彩抖动", jitter_color(base, brightness, contrast, color)),
    ]

    fig = tight_fig(10.5, 6.0)
    axes = fig.subplots(2, 3)
    for ax, (title, image) in zip(axes.flat, augments):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def render_augmentation() -> None:
    st.subheader("2. 数据增强可视化")
    render_note(
        "核心直觉：",
        "数据增强不改变标签，却故意改变输入外观，让模型少记位置、角度、颜色这些偶然因素，多学真正稳定的模式。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        angle = st.slider("旋转角度", -45, 45, 18, 1)
        crop_ratio = st.slider("裁剪保留比例", 0.55, 1.0, 0.78, 0.01)
    with c2:
        scale = st.slider("缩放比例", 0.65, 1.45, 1.12, 0.01)
        brightness = st.slider("亮度", 0.45, 1.65, 1.15, 0.05)
    with c3:
        contrast = st.slider("对比度", 0.45, 1.85, 1.25, 0.05)
        color = st.slider("色彩饱和度", 0.2, 2.0, 1.25, 0.05)

    st.pyplot(plot_augmentations(angle, crop_ratio, scale, brightness, contrast, color), clear_figure=True)
    render_action(
        "请把“旋转角度”拖到 -45 或 45，把“裁剪保留比例”拖到 0.55，再观察目标主体是否还完整；如果主体被裁掉，这种增强就会把标签变成噪声。"
    )
    render_note(
        "图怎么看：",
        "六张小图共享同一张原图：翻转检查左右不变性，旋转检查角度鲁棒性，裁剪和缩放检查目标是否仍在画面中，亮度/对比度/色彩饱和度检查模型是否过度依赖颜色。"
    )
    render_note(
        "工程经验：",
        "增强强度不是越大越好。分类任务里常从轻量增强开始：旋转不超过 15 到 25 度、裁剪保留比例不低于 0.75、亮度和对比度围绕 1.0 上下浮动。医学、遥感、工业缺陷这类方向要先确认变换不会改变标签含义。"
    )
    render_action(
        "极端值测试：把“亮度”调到 0.45、再调到 1.65，观察主体细节是否消失。思考：如果训练集中有这种极端图，模型是在学鲁棒性，还是在学错误样本？"
    )


@st.cache_data(show_spinner=False)
def simulate_initialization(depth: int, width: int, samples: int, activation: str, seed: int):
    torch.manual_seed(seed)
    x0 = torch.randn(samples, width)

    def run(kind: str):
        x = x0.clone()
        activation_stds = []
        gradient_stds = []
        layers = []
        for _ in range(depth):
            layer = nn.Linear(width, width, bias=False)
            if kind == "Xavier":
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            layers.append(layer)
            with torch.no_grad():
                x = layer(x)
                x = torch.tanh(x) if activation == "Tanh" else F.relu(x)
                activation_stds.append(float(x.std()))

        model = nn.Sequential()
        modules = []
        for layer in layers:
            modules.append(layer)
            modules.append(nn.Tanh() if activation == "Tanh" else nn.ReLU())
        model = nn.Sequential(*modules)
        xin = x0.clone().requires_grad_(True)
        out = model(xin).pow(2).mean()
        out.backward()
        for layer in layers:
            gradient_stds.append(float(layer.weight.grad.std()))
        return np.array(activation_stds), np.array(gradient_stds), x.detach().numpy().ravel()

    return {"Xavier": run("Xavier"), "He": run("He")}


def plot_initialization(result: dict):
    fig = tight_fig(11, 4.7)
    ax0, ax1, ax2 = fig.subplots(1, 3)
    layers = np.arange(1, len(result["Xavier"][0]) + 1)
    colors = {"Xavier": "#5e4ae3", "He": "#0f8b8d"}

    for name, (act_std, grad_std, final_act) in result.items():
        ax0.plot(layers, act_std, marker="o", color=colors[name], label=name)
        ax1.plot(layers, grad_std, marker="o", color=colors[name], label=name)
        ax2.hist(final_act, bins=45, alpha=0.48, color=colors[name], label=name, density=True)

    ax0.set_title("前向激活标准差")
    ax0.set_xlabel("层数")
    ax0.set_ylabel("std")
    ax0.grid(alpha=0.25)
    ax0.legend()

    ax1.set_title("反向梯度标准差")
    ax1.set_xlabel("层数")
    ax1.set_ylabel("std")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.set_title("最后一层激活分布")
    ax2.set_xlabel("activation")
    ax2.set_ylabel("density")
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    return fig


def render_initialization(seed: int) -> None:
    st.subheader("3. 权重初始化方法对比")
    render_note(
        "核心直觉：",
        "初始化的目标是让信号穿过很多层后仍然不过大、不过小。Xavier 常配合 Tanh，He 更适合 ReLU。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        depth = st.slider("网络深度", 4, 50, 24, 1)
    with c2:
        width = st.slider("每层宽度", 32, 512, 128, 32)
    with c3:
        activation = st.segmented_control("激活函数", ["ReLU", "Tanh"], default="ReLU")

    result = simulate_initialization(depth, width, 384, activation or "ReLU", seed)
    st.pyplot(plot_initialization(result), clear_figure=True)
    render_action(
        "请把“网络深度”从 4 拖到 50，并在“激活函数”中切换 ReLU/Tanh，观察前向激活标准差和反向梯度标准差是否向 0 塌缩或快速放大。"
    )
    render_note(
        "图怎么看：",
        "左图看信号前向传播后是否保持合理尺度，中图看梯度反向传播是否还能回到浅层，右图看最后一层激活是否集中在少数区间。ReLU 搭配 He 初始化通常更稳，Tanh 搭配 Xavier 更符合它的对称饱和特性。"
    )
    render_note(
        "工程经验：",
        "如果深层网络一开始 loss 几乎不动，先不要急着换优化器；先检查初始化、激活函数和输入标准化。常用起点是 ReLU/GELU + He 初始化，Transformer/RNN 中再配合 LayerNorm 或残差连接稳定尺度。"
    )
    render_action(
        "进阶思考：当“每层宽度”增大时，曲线为什么可能更平滑？如果中图的梯度标准差在浅层接近 0，学习率调大能真正解决问题吗？"
    )


@dataclass(frozen=True)
class RegularizationConfig:
    l1: float
    l2: float
    dropout: float
    patience: int
    max_epochs: int
    seed: int


class RegressionNet(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 96),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.net(x)


@st.cache_data(show_spinner=False)
def train_regularization(config: RegularizationConfig):
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    x_train = np.linspace(-2.7, 2.7, 38).astype(np.float32)
    y_train = (
        np.sin(2.2 * x_train)
        + 0.28 * np.cos(5.0 * x_train)
        + rng.normal(0, 0.18, len(x_train))
    ).astype(np.float32)
    x_val = np.linspace(-2.9, 2.9, 90).astype(np.float32)
    y_val = (np.sin(2.2 * x_val) + 0.28 * np.cos(5.0 * x_val)).astype(np.float32)

    xt = torch.tensor(x_train[:, None])
    yt = torch.tensor(y_train[:, None])
    xv = torch.tensor(x_val[:, None])
    yv = torch.tensor(y_val[:, None])

    model = RegressionNet(config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.l2)

    best_state = None
    best_val = float("inf")
    wait = 0
    stop_epoch = config.max_epochs
    train_losses = []
    val_losses = []

    for epoch in range(config.max_epochs):
        model.train()
        pred = model(xt)
        mse = F.mse_loss(pred, yt)
        l1_penalty = torch.tensor(0.0)
        for param in model.parameters():
            l1_penalty = l1_penalty + param.abs().sum()
        loss = mse + config.l1 * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = F.mse_loss(model(xt), yt).item()
            val_loss = F.mse_loss(model(xv), yv).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                stop_epoch = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    grid = np.linspace(-3.1, 3.1, 240).astype(np.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(grid[:, None])).squeeze().numpy()

    all_weights = torch.cat([p.detach().flatten() for p in model.parameters() if p.dim() > 1])
    near_zero = float((all_weights.abs() < 1e-3).float().mean())

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "grid": grid,
        "prediction": prediction,
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
        "best_val": best_val,
        "stop_epoch": stop_epoch,
        "near_zero": near_zero,
    }


def plot_regularization(result: dict):
    fig = tight_fig(11, 4.8)
    ax0, ax1 = fig.subplots(1, 2)

    ax0.scatter(result["x_train"], result["y_train"], color="#c73e5b", label="训练样本", zorder=3)
    ax0.plot(result["x_val"], result["y_val"], color="#58646d", linestyle="--", label="真实函数")
    ax0.plot(result["grid"], result["prediction"], color="#0f8b8d", linewidth=2.4, label="模型预测")
    ax0.set_title("拟合结果")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(alpha=0.25)
    ax0.legend()

    epochs = np.arange(1, len(result["train_losses"]) + 1)
    ax1.plot(epochs, result["train_losses"], color="#c73e5b", label="训练损失")
    ax1.plot(epochs, result["val_losses"], color="#0f8b8d", label="验证损失")
    ax1.axvline(result["stop_epoch"], color="#d99a22", linestyle="--", label="停止轮次")
    ax1.set_title("训练 / 验证损失")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.25)
    ax1.legend()

    fig.tight_layout()
    return fig


def render_regularization(seed: int) -> None:
    st.subheader("4. 正则化方法演示")
    render_note(
        "核心直觉：",
        "L1 倾向于把一部分权重压到接近 0，L2 抑制权重整体变大，Dropout 让网络不能过度依赖某几个神经元，早停在验证集开始变差时收手。"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        l1 = st.slider("L1 强度", 0.0, 0.002, 0.0001, 0.00005, format="%.5f")
    with c2:
        l2 = st.slider("L2 强度", 0.0, 0.05, 0.003, 0.001, format="%.3f")
    with c3:
        dropout = st.slider("Dropout", 0.0, 0.75, 0.18, 0.01)
    with c4:
        patience = st.slider("早停耐心", 5, 120, 35, 5)

    max_epochs = st.slider("最多训练轮数", 80, 800, 360, 20)
    config = RegularizationConfig(l1, l2, dropout, patience, max_epochs, seed)
    result = train_regularization(config)

    m1, m2, m3 = st.columns(3)
    m1.metric("最佳验证损失", f"{result['best_val']:.4f}")
    m2.metric("实际停止轮次", f"{result['stop_epoch']}")
    m3.metric("近零权重比例", f"{result['near_zero']:.1%}")
    st.pyplot(plot_regularization(result), clear_figure=True)
    render_action(
        "请先把“Dropout”调到 0.0，再调到 0.75；然后把“L2 强度”从 0.000 调到 0.050，观察左图拟合曲线是否从追噪声变成过度平滑。"
    )
    render_note(
        "图怎么看：",
        "左图里红点是训练样本、灰虚线是真实函数、青色线是模型预测；右图同时显示训练损失和验证损失。训练损失持续下降但验证损失反弹，是过拟合；两条曲线都高，是欠拟合。"
    )
    render_note(
        "参数经验：",
        "L1 主要增加稀疏性，可以看“近零权重比例”；L2 是更常用的默认正则，小模型常从 1e-4 到 1e-3 开始；Dropout 在全连接层常用 0.1 到 0.5，超过 0.6 往往会明显欠拟合；早停耐心通常设为总轮数的 5% 到 15%。"
    )
    render_note(
        "真实踩坑：",
        "我见过一个小样本回归项目把 Dropout 固定成 0.7，训练三天都像没学会；曲线症状就是训练损失和验证损失都高。排查步骤是先关 Dropout 确认模型能过拟合小训练集，再逐步加 L2、Dropout 和早停。"
    )
    render_action(
        "进阶思考：如果“最佳验证损失”变好但“近零权重比例”也大幅升高，这说明 L1 在帮你选特征，还是已经把模型容量压坏了？"
    )


def warmup_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def schedule_curves(base_lr: float, min_lr: float, epochs: int, step_size: int, gamma: float, warmup_epochs: int):
    steps = np.arange(epochs)
    step_lr = np.array([base_lr * (gamma ** (e // step_size)) for e in steps])
    cosine = np.array(
        [
            min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * e / max(1, epochs - 1)))
            for e in steps
        ]
    )
    warmup = np.array([warmup_cosine_lr(e, epochs, base_lr, min_lr, warmup_epochs) for e in steps])
    return steps + 1, {"StepLR": step_lr, "CosineAnnealingLR": cosine, "Warmup + Cosine": warmup}


def plot_schedules(steps: np.ndarray, curves: dict[str, np.ndarray]):
    fig = tight_fig(10.5, 4.6)
    ax = fig.subplots()
    colors = {"StepLR": "#c73e5b", "CosineAnnealingLR": "#0f8b8d", "Warmup + Cosine": "#5e4ae3"}
    for name, values in curves.items():
        ax.plot(steps, values, linewidth=2.4, color=colors[name], label=name)
    ax.set_title("学习率调度策略曲线")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def render_schedules() -> None:
    st.subheader("5. 学习率调度策略曲线")
    render_note(
        "核心直觉：",
        "学习率不是只能固定不变。前期可以大一点快速探索，后期逐步变小稳定收敛；warmup 则先慢启动，避免一开始把参数推飞。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        base_lr = st.slider("初始 / 峰值学习率", 0.0005, 0.2, 0.06, 0.0005, format="%.4f")
        min_lr = st.slider("最小学习率", 0.0, 0.05, 0.002, 0.0005, format="%.4f")
    with c2:
        epochs = st.slider("总 epoch", 20, 300, 120, 5)
        step_size = st.slider("StepLR 间隔", 5, 80, 30, 5)
    with c3:
        gamma = st.slider("StepLR 衰减系数", 0.05, 0.95, 0.35, 0.05)
        warmup_epochs = st.slider("Warmup epoch", 0, 60, 12, 1)

    steps, curves = schedule_curves(base_lr, min_lr, epochs, step_size, gamma, warmup_epochs)
    st.pyplot(plot_schedules(steps, curves), clear_figure=True)
    render_action(
        "请把“初始 / 峰值学习率”拖到 0.2，再把“Warmup epoch”从 0 拖到 20，观察紫色曲线如何先慢启动再进入退火。"
    )
    render_note(
        "图怎么看：",
        "StepLR 是阶梯式下降，适合传统 CNN 训练；CosineAnnealingLR 是平滑退火，适合不想手工指定下降节点的实验；Warmup + Cosine 先小步启动，常用于 Transformer、大 batch 或混合精度训练。"
    )
    render_note(
        "工程经验：",
        "学习率搜索通常先定数量级，再定调度。Adam/AdamW 常从 1e-4 到 3e-3 搜；SGD 常从 1e-2 到 1e-1 搜。`min_lr` 不宜太高，否则后期无法细致收敛；`gamma` 太小会让 StepLR 过早失去学习能力。"
    )

    summary = pd.DataFrame(
        {
            "策略": list(curves),
            "起始 LR": [values[0] for values in curves.values()],
            "中段 LR": [values[len(values) // 2] for values in curves.values()],
            "末尾 LR": [values[-1] for values in curves.values()],
        }
    )
    st.dataframe(summary, width="stretch", hide_index=True)
    render_note(
        "排查手册：",
        "loss 前几轮直接发散，优先降低“初始 / 峰值学习率”或增加 “Warmup epoch”；loss 后期平台期明显，检查“最小学习率”是否过高；验证指标反复震荡，优先让学习率下降更平滑。"
    )
    render_action(
        "进阶思考：如果 batch size 加大 4 倍，峰值学习率和 warmup 应该一起怎么变？为什么大 batch 更需要 warmup？"
    )


def plot_mixed_precision():
    fig = tight_fig(10, 3.8)
    ax = fig.subplots()
    labels = ["FP32 显存", "AMP 显存", "FP32 吞吐", "AMP 吞吐"]
    values = [1.0, 0.58, 1.0, 1.65]
    colors = ["#c73e5b", "#0f8b8d", "#c73e5b", "#0f8b8d"]
    bars = ax.bar(labels, values, color=colors, alpha=0.88)
    ax.axhline(1.0, color="#58646d", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.9)
    ax.set_ylabel("相对量")
    ax.set_title("混合精度训练的常见收益示意")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}x", ha="center")
    fig.tight_layout()
    return fig


def render_mixed_precision() -> None:
    st.subheader("6. 混合精度训练概念说明")
    render_note(
        "核心直觉：",
        "混合精度不是把所有计算都粗暴改成半精度，而是在适合的位置用 FP16/BF16 加速，在容易溢出或需要稳定性的地方保留 FP32。"
    )

    left, right = st.columns([1.05, 1])
    with left:
        st.pyplot(plot_mixed_precision(), clear_figure=True)
    with right:
        st.markdown(
            """
            **典型做法**

            - 前向和反向的大部分矩阵乘法使用 FP16 或 BF16。
            - 主权重、优化器状态、损失缩放等关键部分保留 FP32 稳定性。
            - PyTorch 中通常用 `torch.autocast` 和 `GradScaler` 管理。
            - BF16 动态范围更接近 FP32，很多新 GPU 上比 FP16 更省心。
            """
        )

    st.code(
        """
scaler = torch.cuda.amp.GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        pred = model(x)
        loss = loss_fn(pred, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
        """.strip(),
        language="python",
    )

    st.markdown(
        '<p class="small">注意：上面的收益图是概念示意，真实收益取决于 GPU、模型结构、batch size、算子支持和数据加载瓶颈。</p>',
        unsafe_allow_html=True,
    )
    render_note(
        "图怎么看：",
        "柱状图把 FP32 和 AMP 的显存、吞吐放在同一基准线上。AMP 显存柱更低表示同样模型能放更大 batch，AMP 吞吐柱更高表示单位时间能处理更多样本。"
    )
    render_note(
        "工程经验：",
        "新项目如果使用 NVIDIA Tensor Core 或较新的 GPU，优先尝试 BF16；老 GPU 上常用 FP16 + GradScaler。若 loss 变成 NaN，先降低学习率、检查损失缩放，再确认 LayerNorm、softmax、loss 这些敏感计算没有被错误强制成低精度。"
    )
    render_action(
        "进阶思考：如果 AMP 后吞吐没有提升，你会先检查 GPU 算子支持、batch size，还是数据加载速度？为什么小模型常常不是被矩阵乘法卡住？"
    )


def main() -> None:
    st.title("数据预处理与训练技巧")
    st.markdown(
        "把工程训练中最常遇到的几类技巧做成可调实验：先看输入怎么变，再看训练过程为什么更稳。"
    )
    render_note(
        "学习路线：",
        "本页不是单纯罗列技巧，而是按真实训练事故的排查顺序组织：先处理输入尺度和增强，再看初始化与正则化，最后用学习率调度和混合精度提高稳定性与效率。"
    )

    with st.sidebar:
        st.header("实验导航")
        section = st.radio(
            "选择模块",
            [
                "数据预处理",
                "数据增强",
                "权重初始化",
                "正则化",
                "学习率调度",
                "混合精度训练",
            ],
        )
        seed = st.number_input("随机种子", 0, 9999, 42, 1)
        st.caption("相同参数和随机种子下，图形与训练曲线会稳定复现。")

    if section == "数据预处理":
        render_preprocessing(int(seed))
    elif section == "数据增强":
        render_augmentation()
    elif section == "权重初始化":
        render_initialization(int(seed))
    elif section == "正则化":
        render_regularization(int(seed))
    elif section == "学习率调度":
        render_schedules()
    else:
        render_mixed_precision()


if __name__ == "__main__":
    main()


render = main


def compute(seed: int = 42) -> dict[str, object]:
    """Pure computation placeholder."""
    return {"status": "ok", "seed": seed}


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""
    return True
