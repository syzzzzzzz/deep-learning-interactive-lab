"""
深度学习可视化实验台。

运行:
    streamlit run part6_universal_framework/06_streamlit_demo.py
"""

from __future__ import annotations

import html as html_lib
import math
from dataclasses import dataclass
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots


torch.set_num_threads(1)

st.set_page_config(
    page_title="深度学习可视化实验台",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172026;
        --muted: #58646d;
        --line: #d7dde1;
        --paper: #f7f4ee;
        --panel: #ffffff;
        --teal: #0f8b8d;
        --rose: #c73e5b;
        --amber: #d99a22;
        --violet: #5e4ae3;
        --green: #477b44;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.90) 0%, rgba(237,243,240,0.94) 100%),
            linear-gradient(90deg, rgba(15,139,141,0.08) 0%, transparent 24%, transparent 76%, rgba(199,62,91,0.07) 100%),
            #fbfaf6;
        color: var(--ink);
    }
    h1, h2, h3 { letter-spacing: 0; }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background: #eef4f1;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px;
    }
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.70);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.95rem 1rem 0.85rem 1rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #172026;
        background: #172026;
        color: #ffffff;
        min-height: 2.65rem;
        font-weight: 700;
    }
    .stButton > button:hover {
        border-color: #0f8b8d;
        background: #0f8b8d;
        color: #ffffff;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding: 0.1rem 0 1rem 0;
        margin-bottom: 0.85rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 1.05;
        margin: 0;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.75;
        max-width: 920px;
        margin: 0.45rem 0 0 0;
    }
    .scene-title {
        margin: 0.35rem 0 0.3rem 0;
        font-size: 1.35rem;
        font-weight: 750;
        color: var(--ink);
    }
    .scene-lead {
        color: var(--muted);
        max-width: 960px;
        margin: 0 0 0.75rem 0;
        line-height: 1.75;
        font-size: 1rem;
    }
    .lesson-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.75rem 0 1rem 0;
    }
    .lesson-card {
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 118px;
    }
    .lesson-card strong {
        display: block;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
        color: #23313a;
    }
    .lesson-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.62;
        font-size: 0.9rem;
    }
    .insight {
        border-left: 4px solid var(--teal);
        background: rgba(255,255,255,0.68);
        padding: 0.72rem 0.9rem;
        margin: 0.35rem 0 0.9rem 0;
        border-radius: 0 8px 8px 0;
        color: #26343b;
        line-height: 1.68;
    }
    .insight strong {
        color: #102027;
    }
    .caption {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .mini-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.2rem 0 0.8rem 0;
        font-size: 0.92rem;
    }
    .mini-table td {
        border-bottom: 1px solid rgba(215,221,225,0.9);
        padding: 0.42rem 0.35rem;
        color: var(--muted);
        vertical-align: top;
    }
    .mini-table td:first-child {
        color: var(--ink);
        font-weight: 650;
        width: 34%;
    }
    .kernel-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.35rem;
        max-width: 240px;
        margin: 0.2rem 0 0.8rem 0;
    }
    .kernel-cell {
        text-align: center;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.42rem 0.2rem;
        font-variant-numeric: tabular-nums;
        color: var(--ink);
    }
    @media (max-width: 900px) {
        .lesson-grid { grid-template-columns: 1fr; }
        .lesson-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DATASET_LABELS = {
    "双月": "moons",
    "同心环": "rings",
    "螺旋": "spiral",
    "异或": "xor",
}

DATASET_GUIDE = {
    "双月": "两类点像两个月牙，必须弯出一条非直线边界。",
    "同心环": "外圈和内圈相互包围，线性模型会非常吃力。",
    "螺旋": "边界需要不断转弯，能看出深度和宽度的作用。",
    "异或": "四个象限交错分类，是理解非线性最经典的小玩具。",
}

CAPACITY_SETTINGS = {
    "简单": {"depth": 1, "hidden_width": 8, "epochs": 140, "learning_rate": 0.016},
    "适中": {"depth": 3, "hidden_width": 32, "epochs": 240, "learning_rate": 0.012},
    "很强": {"depth": 5, "hidden_width": 96, "epochs": 360, "learning_rate": 0.008},
}

CAPACITY_GUIDE = {
    "简单": "参数少，学习快，但只能画比较朴素的边界。",
    "适中": "通常最适合观察：有足够表达力，又不至于过度追噪声。",
    "很强": "能画复杂边界，也更容易把样本里的偶然噪声当成规律。",
}

KERNEL_GUIDE = {
    "边缘": "中心像素和周围像素差很多时会亮起来，所以边缘最明显。",
    "锐化": "保留中心，同时压低周围，让局部变化更突出。",
    "浮雕": "给不同方向不同权重，等于把方向性变化刻出来。",
    "平滑": "把附近像素做平均，细碎变化会被抹平，整体结构更柔和。",
}

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    n_samples: int
    noise: float
    hidden_width: int
    depth: int
    learning_rate: float
    epochs: int
    activation: str
    seed: int


def render_scene_intro(title: str, lead: str, cards: list[tuple[str, str]]) -> None:
    card_html = "".join(
        '<div class="lesson-card">'
        f"<strong>{html_lib.escape(card_title)}</strong>"
        f"<p>{html_lib.escape(body)}</p>"
        "</div>"
        for card_title, body in cards
    )
    st.markdown(
        f"""
        <div class="scene-title">{html_lib.escape(title)}</div>
        <p class="scene-lead">{html_lib.escape(lead)}</p>
        <div class="lesson-grid">{card_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
            <strong>{html_lib.escape(title)}</strong> {html_lib.escape(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_table(rows: list[tuple[str, str]]) -> None:
    body = "".join(
        "<tr>"
        f"<td>{html_lib.escape(label)}</td>"
        f"<td>{html_lib.escape(text)}</td>"
        "</tr>"
        for label, text in rows
    )
    st.markdown(f'<table class="mini-table">{body}</table>', unsafe_allow_html=True)


def render_kernel_matrix(matrix: np.ndarray) -> None:
    cells = "".join(
        f'<div class="kernel-cell">{value:+.2f}</div>'
        for value in matrix.reshape(-1)
    )
    st.markdown(f'<div class="kernel-grid">{cells}</div>', unsafe_allow_html=True)


def make_dataset(name: str, n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    if name == "moons":
        t0 = rng.uniform(0, math.pi, n0)
        t1 = rng.uniform(0, math.pi, n1)
        x0 = np.c_[np.cos(t0), np.sin(t0)]
        x1 = np.c_[1.0 - np.cos(t1), 0.45 - np.sin(t1)]
    elif name == "rings":
        t0 = rng.uniform(0, 2 * math.pi, n0)
        t1 = rng.uniform(0, 2 * math.pi, n1)
        x0 = np.c_[0.75 * np.cos(t0), 0.75 * np.sin(t0)]
        x1 = np.c_[1.45 * np.cos(t1), 1.45 * np.sin(t1)]
    elif name == "spiral":
        t0 = np.linspace(0.4, 3.7 * math.pi, n0)
        t1 = t0 + math.pi
        r0 = np.linspace(0.15, 1.65, n0)
        r1 = np.linspace(0.15, 1.65, n1)
        x0 = np.c_[r0 * np.cos(t0), r0 * np.sin(t0)]
        x1 = np.c_[r1 * np.cos(t1[:n1]), r1 * np.sin(t1[:n1])]
    elif name == "xor":
        x = rng.uniform(-1.6, 1.6, size=(n_samples, 2))
        y = ((x[:, 0] * x[:, 1]) > 0).astype(np.float32)
        x += rng.normal(0, noise, size=x.shape)
        return x.astype(np.float32), y.astype(np.float32)
    else:
        raise ValueError(f"unknown dataset: {name}")

    x = np.vstack([x0, x1])
    y = np.r_[np.zeros(n0), np.ones(n1)]
    x += rng.normal(0, noise, size=x.shape)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    order = rng.permutation(len(x))
    return x[order].astype(np.float32), y[order].astype(np.float32)


class TinyMLP(nn.Module):
    def __init__(self, hidden_width: int, depth: int, activation: str):
        super().__init__()
        activation_cls = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "GELU": nn.GELU}[activation]
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(activation_cls())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        activations = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, (nn.ReLU, nn.Tanh, nn.GELU)):
                activations.append(x)
        if return_activations:
            return x, activations
        return x


@st.cache_data(show_spinner=False)
def run_experiment(config: ExperimentConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x_np, y_np = make_dataset(config.dataset, config.n_samples, config.noise, config.seed)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np).unsqueeze(1)

    model = TinyMLP(config.hidden_width, config.depth, config.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    losses = []
    accuracies = []
    history_steps = []
    checkpoints = max(1, config.epochs // 90)

    for epoch in range(config.epochs):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % checkpoints == 0 or epoch == config.epochs - 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                acc = ((probs > 0.5).float() == y).float().mean().item()
            history_steps.append(epoch + 1)
            losses.append(float(loss.item()))
            accuracies.append(float(acc))

    grid_size = 152
    grid_x, grid_y = np.meshgrid(
        np.linspace(-2.6, 2.6, grid_size, dtype=np.float32),
        np.linspace(-2.6, 2.6, grid_size, dtype=np.float32),
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()].astype(np.float32)

    with torch.no_grad():
        grid_t = torch.from_numpy(grid)
        grid_logits, activations = model(grid_t, return_activations=True)
        probs = torch.sigmoid(grid_logits).reshape(grid_x.shape).numpy()
        final_logits = model(x)
        final_probs = torch.sigmoid(final_logits)
        final_loss = loss_fn(final_logits, y).item()
        final_acc = ((final_probs > 0.5).float() == y).float().mean().item()

    first_activation = activations[0].numpy()
    variances = first_activation.var(axis=0)
    map_count = min(6, first_activation.shape[1])
    neuron_ids = np.argsort(variances)[-map_count:][::-1]
    activation_maps = [
        first_activation[:, neuron_id].reshape(grid_x.shape)
        for neuron_id in neuron_ids
    ]

    return {
        "x": x_np,
        "y": y_np,
        "history_steps": np.array(history_steps, dtype=np.int16),
        "losses": np.array(losses, dtype=np.float32),
        "accuracies": np.array(accuracies, dtype=np.float32),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "probs": probs,
        "activation_maps": np.array(activation_maps),
        "neuron_ids": neuron_ids,
        "final_loss": float(final_loss),
        "final_acc": float(final_acc),
        "params": sum(p.numel() for p in model.parameters()),
    }


def apply_plot_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 28, "r": 18, "t": 48, "b": 34},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font=PLOT_FONT,
        hoverlabel={"bgcolor": "#172026", "font_color": "#ffffff"},
    )
    return fig


def plot_decision_boundary(result: dict) -> go.Figure:
    x = result["x"]
    y = result["y"]
    grid_x = result["grid_x"]
    grid_y = result["grid_y"]

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=grid_x[0],
            y=grid_y[:, 0],
            z=result["probs"],
            contours={"start": 0, "end": 1, "size": 0.05, "coloring": "heatmap"},
            colorscale=[
                [0.0, "#12343b"],
                [0.45, "#fff7e6"],
                [0.55, "#fff7e6"],
                [1.0, "#c73e5b"],
            ],
            colorbar={"title": "红类概率", "thickness": 12},
            hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>红类概率=%{z:.2f}<extra></extra>",
            name="预测概率",
        )
    )
    fig.add_trace(
        go.Contour(
            x=grid_x[0],
            y=grid_y[:, 0],
            z=result["probs"],
            contours={"start": 0.5, "end": 0.5, "size": 0.5, "coloring": "lines"},
            line={"color": "#d99a22", "width": 4},
            showscale=False,
            hoverinfo="skip",
            name="分界线",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x[y == 0, 0],
            y=x[y == 0, 1],
            mode="markers",
            name="蓝绿色样本",
            marker={"size": 8, "color": "#0f8b8d", "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<extra>蓝绿色样本</extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x[y == 1, 0],
            y=x[y == 1, 1],
            mode="markers",
            name="玫红样本",
            marker={"size": 8, "color": "#c73e5b", "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<extra>玫红样本</extra>",
        )
    )
    fig.update_xaxes(title="x1", range=[-2.6, 2.6], zeroline=False, gridcolor="rgba(88,100,109,0.18)")
    fig.update_yaxes(title="x2", range=[-2.6, 2.6], zeroline=False, gridcolor="rgba(88,100,109,0.18)", scaleanchor="x", scaleratio=1)
    fig.update_layout(title="模型学出的分类地形", legend={"orientation": "h", "y": -0.18})
    return apply_plot_layout(fig, 520)


def plot_training_curves(result: dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    steps = result["history_steps"]
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result["losses"],
            mode="lines",
            line={"color": "#c73e5b", "width": 3, "shape": "spline"},
            name="损失",
            hovertemplate="轮数=%{x}<br>损失=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=result["accuracies"],
            mode="lines",
            line={"color": "#0f8b8d", "width": 3, "shape": "spline"},
            name="准确率",
            hovertemplate="轮数=%{x}<br>准确率=%{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title="训练轮数", gridcolor="rgba(88,100,109,0.18)")
    fig.update_yaxes(title="损失", secondary_y=False, gridcolor="rgba(88,100,109,0.18)")
    fig.update_yaxes(title="准确率", range=[0, 1.02], secondary_y=True)
    fig.update_layout(title="训练时发生了什么", legend={"orientation": "h", "y": -0.25})
    return apply_plot_layout(fig, 330)


def plot_activation_maps(result: dict) -> go.Figure:
    maps = result["activation_maps"]
    neuron_ids = result["neuron_ids"]
    titles = [f"神经元 {int(neuron_id)}" for neuron_id in neuron_ids]
    titles.extend([""] * (6 - len(titles)))
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles)

    for index in range(6):
        row = index // 3 + 1
        col = index % 3 + 1
        if index < len(maps):
            fig.add_trace(
                go.Heatmap(
                    z=maps[index],
                    colorscale="Magma",
                    showscale=index == 0,
                    colorbar={"thickness": 10, "title": "响应"} if index == 0 else None,
                    hovertemplate="位置=(%{x}, %{y})<br>响应=%{z:.2f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="隐藏层正在把平面切成哪些小区域")
    return apply_plot_layout(fig, 470)


def boundary_reading(result: dict, dataset_label: str, capacity: str, noise: float) -> str:
    acc = result["final_acc"]
    if acc >= 0.95 and noise < 0.35:
        opening = "现在的边界已经抓住了主要形状。"
    elif noise >= 0.35:
        opening = "现在样本故意被弄得很乱，模型分错一些点是正常现象。"
    else:
        opening = "现在的边界还没有完全贴住数据，说明模型能力、学习率或训练轮数还可以继续调。"
    return (
        f"{opening}{DATASET_GUIDE[dataset_label]}"
        f"{CAPACITY_GUIDE[capacity]}黄色线是 50% 概率线，它就是模型真正拿来做决定的边界。"
    )


@st.cache_data(show_spinner=False)
def synthetic_image(kind: str, size: int = 88) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    if kind == "环形结构":
        image = np.exp(-((np.sqrt(xx**2 + yy**2) - 0.48) ** 2) / 0.012)
        image += 0.65 * np.exp(-((xx + yy) ** 2) / 0.018)
    elif kind == "斜向纹理":
        image = 0.5 + 0.5 * np.sin(18 * (xx + 0.55 * yy))
        image *= np.exp(-0.35 * (xx**2 + yy**2))
    else:
        image = np.exp(-((xx + 0.35) ** 2 + (yy - 0.15) ** 2) / 0.08)
        image += 0.8 * np.exp(-((xx - 0.28) ** 2 + (yy + 0.28) ** 2) / 0.035)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image.astype(np.float32)


def kernel_matrix_for(name: str, strength: float) -> np.ndarray:
    kernels = {
        "边缘": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "锐化": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "浮雕": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    }
    if name == "平滑":
        center_weight = max(1.0, 7.0 - strength * 2.4)
        k = np.array([[1, 1, 1], [1, center_weight, 1], [1, 1, 1]], dtype=np.float32)
        return k / k.sum()
    return np.array(kernels[name], dtype=np.float32) * strength


def kernel_for(name: str, strength: float) -> torch.Tensor:
    k = torch.tensor(kernel_matrix_for(name, strength), dtype=torch.float32)
    return k.view(1, 1, 3, 3)


@st.cache_data(show_spinner=False)
def convolve_image(image: np.ndarray, kernel_name: str, strength: float) -> np.ndarray:
    x = torch.from_numpy(image).view(1, 1, *image.shape)
    y = F.conv2d(x, kernel_for(kernel_name, strength), padding=1)
    out = y.squeeze().numpy()
    return (out - out.min()) / (out.max() - out.min() + 1e-8)


def plot_convolution_pair(image: np.ndarray, filtered: np.ndarray) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["输入图像", "卷积后的特征图"],
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Heatmap(z=image, colorscale="Viridis", showscale=False, hovertemplate="强度=%{z:.2f}<extra>输入</extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=filtered, colorscale="Inferno", showscale=False, hovertemplate="响应=%{z:.2f}<extra>特征</extra>"),
        row=1,
        col=2,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="同一张图，被一个小滤镜重新阅读")
    return apply_plot_layout(fig, 500)


def deterministic_vectors(tokens: list[str], dim: int, seed: int) -> np.ndarray:
    vectors = []
    for token in tokens:
        local_seed = (sum(ord(ch) for ch in token) + seed * 131) % (2**32 - 1)
        rng = np.random.default_rng(local_seed)
        vectors.append(rng.normal(0, 1, size=dim))
    return np.vstack(vectors).astype(np.float32)


@st.cache_data(show_spinner=False)
def attention_matrix(text: str, temperature: float, dim: int, seed: int) -> tuple[list[str], np.ndarray]:
    tokens = [token for token in text.strip().replace("，", " ").replace("。", " ").split(" ") if token]
    if len(tokens) < 2:
        tokens = list(text.strip()) or list("深度学习")
    tokens = tokens[:12]
    x = deterministic_vectors(tokens, dim, seed)
    rng = np.random.default_rng(seed)
    wq = rng.normal(0, 1 / math.sqrt(dim), size=(dim, dim))
    wk = rng.normal(0, 1 / math.sqrt(dim), size=(dim, dim))
    q = x @ wq
    k = x @ wk
    score = q @ k.T / math.sqrt(dim)
    score = score / temperature
    score = score - score.max(axis=1, keepdims=True)
    attn = np.exp(score)
    attn = attn / attn.sum(axis=1, keepdims=True)
    return tokens, attn


def plot_attention(tokens: list[str], attn: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=attn,
                x=tokens,
                y=tokens,
                colorscale="Magma",
                colorbar={"title": "权重", "thickness": 12},
                hovertemplate="正在看的词=%{y}<br>被看的词=%{x}<br>权重=%{z:.1%}<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title="被看的词", tickangle=28)
    fig.update_yaxes(title="正在看的词", autorange="reversed")
    fig.update_layout(title="每个词把注意力分给了谁")
    return apply_plot_layout(fig, 560)


def strongest_attention_pairs(tokens: list[str], attn: np.ndarray, count: int = 4) -> list[tuple[str, str, float]]:
    pairs = []
    for row in range(len(tokens)):
        for col in range(len(tokens)):
            if row != col:
                pairs.append((tokens[row], tokens[col], float(attn[row, col])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:count]


st.markdown(
    """
    <div class="hero">
      <h1>神经网络实验室</h1>
      <p>不用先背公式。先调数据、模型、卷积核和注意力温度，看神经网络怎样把点、图像和词语变成可以学习的结构。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scene = st.segmented_control(
    "选择一个现象",
    ["边界怎么学出来", "卷积看到了什么", "注意力看向哪里"],
    default="边界怎么学出来",
)

with st.sidebar:
    st.header("实验设置")
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=7, step=1)
    st.caption("随机种子会改变样本、初始权重和注意力投影。相同设置下结果会稳定复现。")
    st.divider()
    st.caption("常用旋钮放在页面里；训练类参数点“重新训练”后才会生效，拖动时不会反复卡住。")

if scene == "边界怎么学出来":
    render_scene_intro(
        "边界怎么学出来",
        "分类不是模型“看懂了标签名”，而是在平面上学出一张概率地形。背景越红，模型越相信这里属于红类；越蓝绿，越相信属于蓝绿色类。",
        [
            ("你正在看什么", "散点是训练样本，彩色背景是模型对每个位置的预测概率，黄色线是模型的决策边界。"),
            ("试试看", "先把模型能力调成简单，再切到螺旋或同心环；然后换成很强，观察边界怎样变弯。"),
            ("为什么会这样", "隐藏层相当于一组可学习的小切刀。层数和宽度越多，组合出的弯曲边界越复杂。"),
        ],
    )

    with st.form("boundary_controls"):
        control_a, control_b, control_c = st.columns(3)
        with control_a:
            dataset_label = st.segmented_control(
                "数据形状",
                list(DATASET_LABELS.keys()),
                default="双月",
            )
        with control_b:
            noise = st.slider("混乱程度", 0.0, 0.55, 0.18, 0.01)
        with control_c:
            capacity = st.segmented_control(
                "模型能力",
                list(CAPACITY_SETTINGS.keys()),
                default="适中",
            )

        selected = CAPACITY_SETTINGS[capacity or "适中"]
        with st.expander("细节参数", expanded=False):
            detail_a, detail_b, detail_c = st.columns(3)
            with detail_a:
                n_samples = st.slider("样本数", 120, 1000, 420, 20)
                activation = st.selectbox("激活函数", ["ReLU", "Tanh", "GELU"], index=0)
            with detail_b:
                depth = st.slider("隐藏层数", 1, 5, selected["depth"], 1)
                hidden_width = st.slider("每层宽度", 4, 128, selected["hidden_width"], 4)
            with detail_c:
                learning_rate = st.slider(
                    "学习率",
                    0.0005,
                    0.08,
                    float(selected["learning_rate"]),
                    0.0005,
                    format="%.4f",
                )
                epochs = st.slider("训练轮数", 20, 800, selected["epochs"], 20)
        st.form_submit_button("重新训练")

    dataset_label = dataset_label or "双月"
    capacity = capacity or "适中"
    dataset = DATASET_LABELS[dataset_label]
    config = ExperimentConfig(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        hidden_width=hidden_width,
        depth=depth,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        seed=int(seed),
    )

    with st.spinner("正在训练小模型..."):
        result = run_experiment(config)

    left, mid, right = st.columns(3)
    left.metric("分对了多少", f"{result['final_acc']:.1%}")
    mid.metric("还差多少", f"{result['final_loss']:.4f}")
    right.metric("可调参数", f"{result['params']:,}")

    render_insight("读图重点", boundary_reading(result, dataset_label, capacity, noise))

    col_boundary, col_side = st.columns([1.35, 1])
    with col_boundary:
        st.plotly_chart(plot_decision_boundary(result), width="stretch", config=PLOT_CONFIG)
        st.markdown(
            '<p class="caption">把鼠标放到背景上，可以看到这个位置被判成红类的概率。'
            "边界不是老师画上去的线，而是模型训练后自然长出来的 50% 等概率线。</p>",
            unsafe_allow_html=True,
        )
    with col_side:
        st.plotly_chart(plot_training_curves(result), width="stretch", config=PLOT_CONFIG)
        render_mini_table(
            [
                ("混乱程度", "给样本坐标加噪声。越大，规律越不干净，模型越难做到全对。"),
                ("模型能力", "用层数和宽度控制表达力。能力太小会欠拟合，太强可能追着噪声跑。"),
                ("学习率", "每一步改参数的幅度。太小学得慢，太大容易在好答案附近来回跳。"),
            ]
        )

    with st.expander("进一步看：隐藏层的神经元在亮哪里", expanded=False):
        st.markdown(
            "每张小图是一枚隐藏神经元对整个平面的响应。亮的地方表示这个神经元更兴奋。"
            "多个神经元的响应叠起来，最后组合成上面的分类边界。"
        )
        st.plotly_chart(plot_activation_maps(result), width="stretch", config=PLOT_CONFIG)

elif scene == "卷积看到了什么":
    render_scene_intro(
        "卷积看到了什么",
        "卷积层不是一次看完整张图，而是拿一个很小的权重模板在图上滑动。哪里和模板匹配，特征图哪里就亮。",
        [
            ("你正在看什么", "左边是输入图像，右边是同一个图像经过 3x3 卷积核后的响应。"),
            ("试试看", "把观察方式从边缘切到平滑，再调效果强度。注意细节会被放大还是被抹掉。"),
            ("为什么会这样", "卷积核里的数字决定它偏爱哪种局部模式：边缘、纹理、方向变化或平滑区域。"),
        ],
    )

    control_a, control_b, control_c = st.columns(3)
    with control_a:
        image_kind = st.segmented_control(
            "输入图像",
            ["环形结构", "斜向纹理", "双峰能量"],
            default="环形结构",
        )
    with control_b:
        kernel_name = st.segmented_control(
            "观察方式",
            ["边缘", "锐化", "浮雕", "平滑"],
            default="边缘",
        )
    with control_c:
        strength = st.slider("效果强度", 0.2, 2.5, 1.0, 0.1)

    image_kind = image_kind or "环形结构"
    kernel_name = kernel_name or "边缘"
    image = synthetic_image(image_kind)
    filtered = convolve_image(image, kernel_name, strength)
    kernel_matrix = kernel_matrix_for(kernel_name, strength)

    render_insight(
        "读图重点",
        f"{KERNEL_GUIDE[kernel_name]}右图越亮，表示这个局部区域越符合当前卷积核正在寻找的模式。",
    )

    col_plot, col_text = st.columns([1.35, 1])
    with col_plot:
        st.plotly_chart(plot_convolution_pair(image, filtered), width="stretch", config=PLOT_CONFIG)
    with col_text:
        st.markdown("**当前 3x3 卷积核**")
        render_kernel_matrix(kernel_matrix)
        render_mini_table(
            [
                ("卷积核", "一个小权重模板。它滑过图像，每次只看附近几个像素。"),
                ("特征图", "卷积核在每个位置的响应记录。亮处表示模式匹配得更强。"),
                ("共享权重", "同一个卷积核扫完整张图，所以它能在任何位置寻找同一种结构。"),
            ]
        )
        st.markdown(
            '<p class="caption">真实 CNN 会同时学习很多个卷积核。前几层常学到边缘和纹理，'
            "后面的层会把这些小结构组合成部件和物体。</p>",
            unsafe_allow_html=True,
        )

else:
    render_scene_intro(
        "注意力看向哪里",
        "Transformer 的注意力不是一句“重点关注”。它是一张权重表：每个词都会把自己的注意力分给句子里的其他词。",
        [
            ("你正在看什么", "每一行表示一个正在思考的词；这一行里的亮格子表示它把注意力分给了谁。"),
            ("试试看", "把注意力集中程度调低，权重会更尖；调高，注意力会更平均。"),
            ("为什么会这样", "Query 和 Key 做相似度比较，再经过 softmax，得到一行加起来为 1 的注意力分布。"),
        ],
    )
    st.markdown(
        dedent(
            """
            **这张实验图要这样读：**横轴是“被看的词”，纵轴是“正在看的词”。某一行越亮，表示这一行对应的词越依赖那一列的词。注意力不是把一句话压成一个结果，而是让每个词都拥有一份自己的上下文读取清单。

            > 互动：先保持默认文本，观察“深度”“学习”“表征”等词之间的亮格子；再改写“输入一串词”，加入“因为”“所以”这样的连接词。思考：如果词与词之间的关系变了，为什么注意力分布也应该跟着变？
            """
        )
    )

    control_a, control_b = st.columns([1.4, 1])
    with control_a:
        text = st.text_input("输入一串词", "深度 学习 把 数据 变成 表征 再 变成 判断")
    with control_b:
        temperature = st.slider("注意力集中程度", 0.2, 3.0, 1.0, 0.1)

    with st.sidebar.expander("注意力实验细节"):
        dim = st.slider("向量维度", 8, 96, 32, 8)

    tokens, attn = attention_matrix(text, temperature, dim, int(seed))

    if temperature <= 0.7:
        temp_note = "现在温度较低，softmax 会把差异放大，注意力更像聚光灯。"
    elif temperature >= 1.8:
        temp_note = "现在温度较高，权重被摊得更平，注意力更像柔光。"
    else:
        temp_note = "现在温度适中，既能看见偏好，也不会过度压扁其他关系。"
    render_insight("读图重点", temp_note)
    st.markdown(
        dedent(
            f"""
            **参数含义：**“注意力集中程度”实际控制 softmax 前分数的温度，温度越低，最大分数越容易压过其他候选；温度越高，多个候选会被保留下来。“向量维度”现在是 **{dim}**，它决定每个词向量有多少个特征方向。

            > 互动：把“注意力集中程度”从 0.2 拖到 3.0，观察热力图从尖锐变平滑；再把侧边栏“向量维度”从 8 调到 96，观察随机教学向量带来的关系纹理变化。思考：为什么真实模型既需要表达能力，也要控制计算成本？
            """
        )
    )

    col_heatmap, col_notes = st.columns([1.25, 1])
    with col_heatmap:
        st.plotly_chart(plot_attention(tokens, attn), width="stretch", config=PLOT_CONFIG)
    with col_notes:
        pairs = strongest_attention_pairs(tokens, attn)
        render_mini_table(
            [
                ("Query", "正在发问的词：我现在需要从谁那里取信息？"),
                ("Key", "被比较的词：我这里有什么特征值得别人看？"),
                ("Value", "真正被汇总的信息。注意力权重决定每个 Value 被拿走多少。"),
            ]
        )
        rows = [(f"{source} -> {target}", f"权重 {weight:.1%}") for source, target, weight in pairs]
        st.markdown("**当前最强的几条注意力连接**")
        render_mini_table(rows)
        st.markdown(
            '<p class="caption">这里的词向量是为了教学随机生成的，所以它不等于真实大模型的语言理解。'
            "但热图的计算方式和注意力机制的形状是一致的。</p>",
            unsafe_allow_html=True,
        )
