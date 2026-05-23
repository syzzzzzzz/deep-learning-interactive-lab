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

    left, right = st.columns([1, 1])
    with left:
        st.caption("处理后数值特征的统计量")
        stats = transformed.describe().loc[["mean", "std", "min", "max"]]
        st.dataframe(stats, width="stretch")
    with right:
        st.caption("独热编码后的前 8 行")
        st.dataframe(pd.concat([df[["城市"]].head(8), one_hot.head(8)], axis=1), width="stretch")


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

    summary = pd.DataFrame(
        {
            "策略": list(curves),
            "起始 LR": [values[0] for values in curves.values()],
            "中段 LR": [values[len(values) // 2] for values in curves.values()],
            "末尾 LR": [values[-1] for values in curves.values()],
        }
    )
    st.dataframe(summary, width="stretch", hide_index=True)


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


def main() -> None:
    st.title("数据预处理与训练技巧")
    st.markdown(
        "把工程训练中最常遇到的几类技巧做成可调实验：先看输入怎么变，再看训练过程为什么更稳。"
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
