"""
Interactive RNN and sequence model teaching lab.

Run:
    streamlit run part3_rnn/sequence_models.py
or:
    python main.py part3_rnn/sequence_models
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


torch.set_num_threads(1)

st.set_page_config(
    page_title="RNN 与序列模型",
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
        --panel: #ffffff;
        --teal: #0f8b8d;
        --rose: #bf3f5b;
        --amber: #c4871f;
        --green: #3f7d58;
        --blue: #3268a8;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(240,247,246,0.96) 100%),
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
    .formula {
        font-family: Consolas, monospace;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.72rem 0.9rem;
        line-height: 1.7;
        color: #1e2b32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PALETTE = {
    "input": "#3268a8",
    "hidden": "#0f8b8d",
    "output": "#bf3f5b",
    "memory": "#c4871f",
    "gate": "#3f7d58",
    "muted": "#596772",
}


@dataclass
class ForecastResult:
    losses: list[float]
    input_seq: np.ndarray
    target_seq: np.ndarray
    pred_seq: np.ndarray
    future: np.ndarray
    test_loss: float


class SequencePredictor(nn.Module):
    def __init__(self, cell_type: str, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        cells = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        self.rnn = cells[cell_type](
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out)


class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        return self.fc(out), hidden


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#52616b") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.8,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: int = 10,
) -> None:
    rect = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color,
        edgecolor="white",
        linewidth=2,
        alpha=0.92,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
    )


def clean_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def plot_rnn_unroll(seq_len: int, active_step: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.3))
    clean_axis(ax, (-0.7, seq_len + 0.45), (-0.2, 3.4))

    add_box(ax, (-0.55, 1.4), 0.8, 0.62, "RNN\ncell", PALETTE["hidden"], 10)
    add_arrow(ax, (-0.1, 1.4), (-0.1, 0.9), PALETTE["input"])
    add_arrow(ax, (-0.1, 2.02), (-0.1, 2.54), PALETTE["output"])
    ax.text(-0.1, 0.62, "x_t", ha="center", va="center", fontsize=11, color=PALETTE["input"])
    ax.text(-0.1, 2.78, "y_t", ha="center", va="center", fontsize=11, color=PALETTE["output"])
    loop = FancyArrowPatch(
        (0.24, 1.92),
        (0.24, 1.52),
        connectionstyle="arc3,rad=-1.35",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=PALETTE["hidden"],
    )
    ax.add_patch(loop)
    ax.text(0.55, 1.7, "h_{t-1} -> h_t", fontsize=10, color=PALETTE["hidden"], va="center")

    ax.plot([0.95, 0.95], [0.1, 3.15], color="#c9d1d6", linestyle="--", linewidth=1.2)
    ax.text(1.04, 3.05, "按时间展开", fontsize=11, color=PALETTE["muted"])

    for t in range(seq_len):
        x = t + 1.35
        color = PALETTE["hidden"] if t + 1 == active_step else "#8aa9aa"
        alpha = 1.0 if t + 1 == active_step else 0.62
        add_box(ax, (x - 0.35, 1.35), 0.7, 0.62, f"h{t + 1}", color, 10)
        ax.patches[-1].set_alpha(alpha)
        add_arrow(ax, (x, 0.82), (x, 1.35), PALETTE["input"])
        add_arrow(ax, (x, 1.97), (x, 2.52), PALETTE["output"])
        ax.text(x, 0.56, f"x{t + 1}", ha="center", fontsize=10, color=PALETTE["input"])
        ax.text(x, 2.76, f"y{t + 1}", ha="center", fontsize=10, color=PALETTE["output"])
        if t > 0:
            add_arrow(ax, (x - 1.0 + 0.35, 1.66), (x - 0.35, 1.66), PALETTE["hidden"])
        if t + 1 == active_step:
            ax.text(
                x,
                3.18,
                "当前时间步",
                ha="center",
                va="center",
                fontsize=10,
                color=PALETTE["hidden"],
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


def plot_hidden_dynamics(seq_len: int, hidden_size: int, recurrent_scale: float, input_scale: float) -> plt.Figure:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 4 * np.pi, seq_len)
    x = np.stack([np.sin(t), np.cos(0.7 * t), np.sin(1.7 * t)], axis=1) * input_scale
    w_x = rng.normal(0, 0.8, size=(3, hidden_size))
    q, _ = np.linalg.qr(rng.normal(size=(hidden_size, hidden_size)))
    w_h = q * recurrent_scale
    h = np.zeros(hidden_size)
    states = []
    for i in range(seq_len):
        h = np.tanh(x[i] @ w_x + h @ w_h)
        states.append(h.copy())
    states_arr = np.array(states)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8), gridspec_kw={"height_ratios": [1, 1.35, 1]})
    for i in range(x.shape[1]):
        axes[0].plot(x[:, i], label=f"输入维度 {i + 1}", linewidth=1.8)
    axes[0].set_title("输入序列")
    axes[0].set_xlabel("时间步")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncols=3, fontsize=9)

    im = axes[1].imshow(states_arr.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("隐藏状态热力图")
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("隐藏单元")
    fig.colorbar(im, ax=axes[1], shrink=0.88)

    for i in range(min(5, hidden_size)):
        axes[2].plot(states_arr[:, i], label=f"h[{i}]", linewidth=1.8)
    axes[2].set_title("部分隐藏单元随时间变化")
    axes[2].set_xlabel("时间步")
    axes[2].set_ylim(-1.08, 1.08)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(ncols=min(5, hidden_size), fontsize=9)
    fig.tight_layout()
    return fig


def plot_gradient_flow(seq_len: int, jacobian_scale: float, activation_saturation: float) -> tuple[plt.Figure, dict[str, float]]:
    steps = np.arange(seq_len + 1)
    local_derivative = max(1e-4, 1.0 - activation_saturation)
    effective_gain = jacobian_scale * local_derivative
    grad_norm = np.power(effective_gain, steps)
    grad_norm = np.clip(grad_norm, 1e-12, 1e12)

    clipped = np.minimum(grad_norm, 1.0)
    clipped_high = np.minimum(grad_norm, 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].semilogy(steps, grad_norm, color=PALETTE["rose"], linewidth=2.5, label="原始梯度范数")
    axes[0].semilogy(steps, clipped, color=PALETTE["green"], linestyle="--", linewidth=2, label="clip=1")
    axes[0].semilogy(steps, clipped_high, color=PALETTE["blue"], linestyle=":", linewidth=2, label="clip=10")
    axes[0].axhline(1e-4, color="#7a858c", linestyle="--", alpha=0.55)
    axes[0].axhline(1e4, color="#7a858c", linestyle="--", alpha=0.55)
    axes[0].set_title("反向传播中的矩阵连乘")
    axes[0].set_xlabel("反向穿过的时间步数")
    axes[0].set_ylabel("梯度范数 log scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    colors = ["#3268a8" if 0.3 <= v <= 3.0 else "#bf3f5b" for v in grad_norm[:: max(1, seq_len // 20)]]
    sample_steps = steps[:: max(1, seq_len // 20)]
    sample_vals = grad_norm[:: max(1, seq_len // 20)]
    axes[1].bar(sample_steps, np.log10(sample_vals + 1e-12), color=colors, width=max(0.8, seq_len / 35))
    axes[1].axhline(0, color="#52616b", linewidth=1)
    axes[1].set_title("log10(梯度范数)")
    axes[1].set_xlabel("时间距离")
    axes[1].set_ylabel("log10 norm")
    axes[1].grid(True, axis="y", alpha=0.25)

    metrics = {
        "effective_gain": effective_gain,
        "final_norm": float(grad_norm[-1]),
        "local_derivative": local_derivative,
    }
    fig.tight_layout()
    return fig, metrics


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_step_demo(forget_bias: float, input_bias: float, output_bias: float, candidate_strength: float) -> dict[str, np.ndarray]:
    units = 8
    base = np.linspace(-1.2, 1.2, units)
    c_prev = np.tanh(np.linspace(-1.6, 1.6, units))
    f = sigmoid(forget_bias + 0.75 * base)
    i = sigmoid(input_bias - 0.55 * base)
    g = np.tanh(candidate_strength * np.sin(np.linspace(0, 1.75 * np.pi, units)))
    o = sigmoid(output_bias + 0.45 * np.cos(np.linspace(0, 2 * np.pi, units)))
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return {"c_prev": c_prev, "f": f, "i": i, "g": g, "o": o, "c": c, "h": h}


def plot_lstm_flow(values: dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    clean_axis(ax, (-0.2, 10.6), (0.2, 6.3))

    add_box(ax, (0.3, 4.75), 1.25, 0.62, "c_{t-1}", PALETTE["memory"])
    add_box(ax, (2.25, 4.75), 1.35, 0.62, "遗忘门\nf_t", "#d96363")
    add_box(ax, (4.15, 4.75), 1.35, 0.62, "输入门\ni_t", "#3268a8")
    add_box(ax, (5.95, 4.75), 1.35, 0.62, "候选记忆\ng_t", "#3f7d58")
    add_box(ax, (7.85, 4.75), 1.25, 0.62, "c_t", PALETTE["memory"])
    add_box(ax, (7.85, 2.25), 1.25, 0.62, "tanh(c_t)", "#6b5fb5")
    add_box(ax, (5.95, 2.25), 1.35, 0.62, "输出门\no_t", "#bf3f5b")
    add_box(ax, (9.15, 2.25), 1.2, 0.62, "h_t", PALETTE["hidden"])

    add_arrow(ax, (1.55, 5.06), (2.25, 5.06), PALETTE["memory"])
    add_arrow(ax, (3.6, 5.06), (7.85, 5.06), PALETTE["memory"])
    add_arrow(ax, (5.5, 5.06), (5.95, 5.06), PALETTE["green"])
    add_arrow(ax, (7.3, 5.06), (7.85, 5.06), PALETTE["green"])
    add_arrow(ax, (8.48, 4.75), (8.48, 2.87), PALETTE["memory"])
    add_arrow(ax, (7.3, 2.56), (7.85, 2.56), PALETTE["output"])
    add_arrow(ax, (9.1, 2.56), (9.15, 2.56), PALETTE["hidden"])

    ax.text(4.45, 5.55, "保留旧记忆", fontsize=10, color="#6b3a3a")
    ax.text(5.38, 4.35, "写入新内容", fontsize=10, color="#245064")
    ax.text(7.15, 1.82, "决定暴露多少记忆", fontsize=10, color="#713446")
    ax.text(3.65, 3.45, "c_t = f_t * c_{t-1} + i_t * g_t", fontsize=13, color="#1e2b32", fontweight="bold")
    ax.text(6.6, 1.2, "h_t = o_t * tanh(c_t)", fontsize=13, color="#1e2b32", fontweight="bold")

    ax_heat = fig.add_subplot(gs[1, 0])
    rows = ["c_prev", "f", "i", "g", "o", "c", "h"]
    matrix = np.vstack([values[k] for k in rows])
    im = ax_heat.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_heat.set_yticks(range(len(rows)))
    ax_heat.set_yticklabels(["旧记忆", "遗忘门", "输入门", "候选", "输出门", "新记忆", "隐藏态"])
    ax_heat.set_xlabel("隐藏单元")
    ax_heat.set_title("门值和状态值")
    fig.colorbar(im, ax=ax_heat, shrink=0.86)

    ax_bar = fig.add_subplot(gs[1, 1])
    means = [values["f"].mean(), values["i"].mean(), values["o"].mean()]
    ax_bar.bar(["遗忘门", "输入门", "输出门"], means, color=["#d96363", "#3268a8", "#bf3f5b"])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("平均门值")
    ax_bar.set_title("门的开合程度")
    ax_bar.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def gru_step_demo(update_bias: float, reset_bias: float, candidate_strength: float) -> dict[str, np.ndarray]:
    units = 8
    base = np.linspace(-1.25, 1.25, units)
    h_prev = np.tanh(np.linspace(-1.4, 1.4, units))
    z = sigmoid(update_bias + 0.7 * base)
    r = sigmoid(reset_bias - 0.55 * base)
    n = np.tanh(candidate_strength * np.sin(np.linspace(0, 1.5 * np.pi, units)) + r * h_prev * 0.4)
    h = (1.0 - z) * h_prev + z * n
    return {"h_prev": h_prev, "z": z, "r": r, "n": n, "h": h}


def plot_gru_flow(values: dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    clean_axis(ax, (-0.2, 10.3), (0.5, 5.5))

    add_box(ax, (0.35, 3.65), 1.25, 0.62, "h_{t-1}", PALETTE["hidden"])
    add_box(ax, (2.2, 4.4), 1.35, 0.62, "更新门\nz_t", "#3268a8")
    add_box(ax, (2.2, 2.85), 1.35, 0.62, "重置门\nr_t", "#bf3f5b")
    add_box(ax, (4.6, 2.85), 1.45, 0.62, "候选状态\nn_t", "#3f7d58")
    add_box(ax, (7.05, 3.65), 1.35, 0.62, "插值混合", "#c4871f")
    add_box(ax, (8.9, 3.65), 1.15, 0.62, "h_t", PALETTE["hidden"])

    add_arrow(ax, (1.6, 3.96), (2.2, 4.7), PALETTE["hidden"])
    add_arrow(ax, (1.6, 3.96), (2.2, 3.16), PALETTE["hidden"])
    add_arrow(ax, (3.55, 3.16), (4.6, 3.16), PALETTE["output"])
    add_arrow(ax, (6.05, 3.16), (7.05, 3.8), PALETTE["green"])
    add_arrow(ax, (3.55, 4.7), (7.05, 4.1), PALETTE["input"])
    add_arrow(ax, (1.6, 3.96), (7.05, 3.96), PALETTE["hidden"])
    add_arrow(ax, (8.4, 3.96), (8.9, 3.96), PALETTE["hidden"])
    ax.text(5.5, 4.8, "h_t = (1 - z_t) * h_{t-1} + z_t * n_t", fontsize=13, color="#1e2b32", fontweight="bold")
    ax.text(3.0, 2.35, "r_t 控制旧状态进入候选状态的比例", fontsize=10, color="#713446")
    ax.text(4.0, 1.65, "GRU 没有独立细胞状态，把记忆直接放在 h_t 中", fontsize=11, color=PALETTE["muted"])

    ax_heat = fig.add_subplot(gs[1, 0])
    rows = ["h_prev", "z", "r", "n", "h"]
    matrix = np.vstack([values[k] for k in rows])
    im = ax_heat.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_heat.set_yticks(range(len(rows)))
    ax_heat.set_yticklabels(["旧隐藏态", "更新门", "重置门", "候选状态", "新隐藏态"])
    ax_heat.set_xlabel("隐藏单元")
    ax_heat.set_title("GRU 门值和状态值")
    fig.colorbar(im, ax=ax_heat, shrink=0.86)

    ax_bar = fig.add_subplot(gs[1, 1])
    means = [values["z"].mean(), values["r"].mean()]
    ax_bar.bar(["更新门", "重置门"], means, color=["#3268a8", "#bf3f5b"])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("平均门值")
    ax_bar.set_title("门的开合程度")
    ax_bar.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_bidirectional_rnn(seq_len: int, merge_mode: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    clean_axis(ax, (-0.5, seq_len + 0.5), (-0.2, 4.2))
    for t in range(seq_len):
        x = t + 0.3
        add_box(ax, (x - 0.28, 0.35), 0.56, 0.42, f"x{t + 1}", PALETTE["input"], 9)
        add_box(ax, (x - 0.32, 1.45), 0.64, 0.48, f"→h{t + 1}", PALETTE["hidden"], 9)
        add_box(ax, (x - 0.32, 2.45), 0.64, 0.48, f"←h{t + 1}", "#6b5fb5", 9)
        add_box(ax, (x - 0.34, 3.45), 0.68, 0.46, f"y{t + 1}", PALETTE["output"], 9)
        add_arrow(ax, (x, 0.77), (x, 1.45), PALETTE["input"])
        add_arrow(ax, (x, 2.45), (x, 1.93), "#6b5fb5")
        add_arrow(ax, (x, 1.93), (x, 3.45), PALETTE["output"])
        if t > 0:
            add_arrow(ax, (x - 1 + 0.32, 1.69), (x - 0.32, 1.69), PALETTE["hidden"])
        if t < seq_len - 1:
            add_arrow(ax, (x + 1 - 0.32, 2.69), (x + 0.32, 2.69), "#6b5fb5")

    ax.text(-0.35, 1.68, "正向", ha="right", va="center", color=PALETTE["hidden"], fontweight="bold")
    ax.text(-0.35, 2.68, "反向", ha="right", va="center", color="#6b5fb5", fontweight="bold")
    ax.text(
        seq_len / 2,
        4.08,
        f"每个位置的输出 = {merge_mode}(正向上下文, 反向上下文)",
        ha="center",
        fontsize=12,
        color="#1e2b32",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def make_sine_dataset(seq_len: int, noise: float, n_samples: int = 180) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(42)
    t = np.linspace(0, 8 * np.pi, n_samples + seq_len + 1)
    signal = np.sin(t) + 0.35 * np.sin(2.3 * t + 0.4) + noise * rng.normal(size=len(t))
    x, y = [], []
    for i in range(n_samples):
        x.append(signal[i : i + seq_len])
        y.append(signal[i + 1 : i + seq_len + 1])
    return (
        torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1),
    )


@st.cache_data(show_spinner=False)
def train_forecast_model(
    cell_type: str,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    lr: float,
    epochs: int,
    noise: float,
    dropout: float,
) -> ForecastResult:
    torch.manual_seed(42)
    np.random.seed(42)
    x, y = make_sine_dataset(seq_len, noise)
    split = int(0.78 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = SequencePredictor(cell_type, hidden_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test)
        test_loss = float(F.mse_loss(pred_test, y_test).item())
        sample_x = x_test[0:1].clone()
        sample_pred = pred_test[0, :, 0].detach().numpy()
        target = y_test[0, :, 0].detach().numpy()
        input_seq = sample_x[0, :, 0].detach().numpy()

        rolling = sample_x.clone()
        future = []
        for _ in range(30):
            out = model(rolling)
            next_value = out[:, -1:, :]
            future.append(float(next_value.item()))
            rolling = torch.cat([rolling[:, 1:, :], next_value], dim=1)

    return ForecastResult(losses, input_seq, target, sample_pred, np.array(future), test_loss)


def plot_forecast_result(result: ForecastResult) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(result.losses, color=PALETTE["blue"], linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE log scale")
    axes[0].grid(True, alpha=0.25)

    t = np.arange(len(result.input_seq))
    future_t = np.arange(len(result.input_seq), len(result.input_seq) + len(result.future))
    axes[1].plot(t, result.input_seq, color="#7a858c", linewidth=1.8, label="输入窗口")
    axes[1].plot(t, result.target_seq, color=PALETTE["green"], linewidth=2.2, label="真实下一步")
    axes[1].plot(t, result.pred_seq, color=PALETTE["rose"], linestyle="--", linewidth=2.2, label="模型预测")
    axes[1].plot(future_t, result.future, color=PALETTE["amber"], linewidth=2.2, label="滚动预测")
    axes[1].axvline(len(t) - 1, color="#7a858c", linestyle=":", linewidth=1.4)
    axes[1].set_title("时间序列预测")
    axes[1].set_xlabel("时间步")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


TEXT_CORPORA = {
    "唐诗风格小样本": "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少海上生明月天涯共此时",
    "英文名字": "alice bob charlie diana edward fiona george helen ivan julia kevin laura michael nancy oliver ",
    "技术短句": "rnn remembers sequences lstm controls memory gru is compact attention reads context ",
}


@st.cache_resource(show_spinner=False)
def train_char_model(corpus_name: str, hidden_size: int, epochs: int, lr: float) -> dict[str, object]:
    torch.manual_seed(11)
    text = TEXT_CORPORA[corpus_name]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    x = ids[:-1].unsqueeze(0)
    y = ids[1:]

    model = CharLanguageModel(len(chars), embed_size=24, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.squeeze(0), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return {"model": model, "chars": chars, "stoi": stoi, "itos": itos, "losses": losses}


def generate_text(
    model: CharLanguageModel,
    prefix: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    length: int,
    temperature: float,
) -> str:
    model.eval()
    valid_prefix = "".join(c for c in prefix if c in stoi)
    if not valid_prefix:
        valid_prefix = next(iter(stoi))
    ids = torch.tensor([[stoi[c] for c in valid_prefix]], dtype=torch.long)
    generated = list(valid_prefix)
    with torch.no_grad():
        logits, hidden = model(ids)
        for _ in range(length):
            last_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = int(torch.multinomial(probs[0], 1).item())
            generated.append(itos[next_id])
            logits, hidden = model(torch.tensor([[next_id]], dtype=torch.long), hidden)
    return "".join(generated)


def plot_char_training(losses: list[float], generated: str, chars: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(losses, color=PALETTE["blue"], linewidth=2)
    axes[0].set_title("字符模型训练损失")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].grid(True, alpha=0.25)

    counts = {c: generated.count(c) for c in chars}
    axes[1].bar(list(counts.keys()), list(counts.values()), color=PALETTE["green"])
    axes[1].set_title("生成文本的字符分布")
    axes[1].set_xlabel("字符")
    axes[1].set_ylabel("出现次数")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def render_sequence_learning_map() -> None:
    st.markdown(
        """
        <div class="note">
        <strong>学习地图：</strong>序列模型和普通前馈网络最大的区别，是它必须回答“前面发生过什么”。
        RNN 用隐藏状态 h_t 压缩历史，LSTM 额外维护细胞状态 c_t 来保留长期记忆，GRU 用更少的门把记忆直接放在 h_t 里。
        左侧“查看内容”可以按机制顺序阅读：先看 RNN 展开和隐藏状态，再看梯度问题，接着比较 LSTM/GRU 门控，最后进入预测和文本生成任务。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        > 互动顺序：先在“RNN 展开”拖动“展开时间步”和“隐藏单元数”，观察状态如何沿时间传递；再到“梯度问题”把“反向传播时间距离”调长，观察远处梯度为什么会消失或爆炸；最后在“时间序列预测”和“文本生成”里调模型类型、窗口长度、学习率、temperature，把结构问题和训练现象连起来。
        >
        > 进阶思考：如果一个模型只看到当前输入 x_t，却没有隐藏状态 h_{t-1}，它还能判断“前面已经出现过什么”吗？这就是序列模型为什么需要状态。
        """
    )
    st.code(
        """hidden = None
for x, y in loader:
    pred, hidden = model(x, hidden)      # 前向：状态跨时间传递
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()                      # BPTT：沿时间反向传播
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    hidden = hidden.detach()             # 截断 BPTT：保留数值，截断梯度图""",
        language="python",
    )


def render_rnn_unroll_guide(
    seq_len: int,
    active_step: int,
    recurrent_scale: float,
    hidden_size: int,
    input_scale: float,
) -> None:
    st.markdown(
        f"""
        **图怎么看：**上方展开图把同一个 RNN cell 沿时间复制成 **{seq_len}** 个时间步。每个 h 都不是新模型，而是同一组参数反复使用；箭头 `h_(t-1) -> h_t` 表示历史摘要向后传。当前高亮的是第 **{active_step}** 步，它同时接收当前输入 `x_t` 和前一步隐藏状态。

        **隐藏状态为什么必要：**如果没有 h，模型每一步只能看当前输入，无法记住前面出现过的节奏、词语或事件。下方隐藏状态热力图中，横轴是时间，纵轴是隐藏单元；颜色持续变化说明模型正在把历史压缩进一组状态值。当前隐藏单元数为 **{hidden_size}**，输入强度为 **{input_scale:.1f}**，循环权重尺度为 **{recurrent_scale:.2f}**。

        **参数怎么调：**“展开时间步”的范围是 3 到 12，控制结构图长度；“当前时间步”只改变高亮位置；“循环权重尺度”的范围是 0.10 到 1.50，决定历史状态影响下一步的强弱；“隐藏单元数”的范围是 4 到 16，决定状态容量；“输入强度”的范围是 0.2 到 2.0，决定新输入对状态的冲击。

        > 极端值实验 1：把“隐藏单元数”调到 4，再把“输入强度”调高。观察热力图是否更容易挤成少数强响应。隐藏维度过小的症状是：模型只能记住粗略趋势，长序列预测会变钝。
        >
        > 极端值实验 2：把“循环权重尺度”调到 0.10，再调到 1.50。前者会让历史很快淡掉，后者会让状态更容易饱和或震荡。真实训练中，这会表现为长期依赖学不住或 loss 曲线不稳定。
        """
    )


def render_gradient_issue_guide(seq_len: int, jacobian_scale: float, saturation: float, metrics: dict[str, float]) -> None:
    st.markdown(
        f"""
        **图怎么看：**左图用 log 坐标展示梯度穿过时间步时的范数变化；右图把同一件事画成柱状图。当前反向传播时间距离是 **{seq_len}**，循环 Jacobian 尺度是 **{jacobian_scale:.2f}**，tanh 饱和程度是 **{saturation:.2f}**，单步有效增益为 **{metrics['effective_gain']:.3f}**。单步有效增益略小于 1，几十步后也会指数衰减；略大于 1，几十步后也可能爆炸。

        **症状与排查：**梯度消失时，早期时间步几乎学不到，表现为长依赖任务准确率低、预测只依赖最近输入；梯度爆炸时，loss 会突然变成很大、NaN，或者参数更新后模型输出乱跳。排查顺序是：先看序列长度是否过长，再看激活是否饱和，再看学习率，最后打开梯度裁剪。

        **梯度裁剪和截断 BPTT：**图中的 clip 曲线说明梯度裁剪能限制爆炸，但不能把已经消失的梯度恢复回来。截断 BPTT 的做法是把长序列分段训练，段间传递隐藏状态数值，但用 `detach()` 截断梯度图；它节省显存，也减少极长链路上的梯度问题。

        > 极端值实验 3：把“反向传播时间距离”调到 120，再把“循环 Jacobian 尺度”调到 1.20。观察梯度如何爆炸。然后把尺度调到 0.85，观察梯度如何消失。思考：为什么 RNN 训练比普通前馈网络更怕时间链太长？
        >
        > 工程经验：真实项目里通常同时使用较小学习率、梯度裁剪、LSTM/GRU、LayerNorm/权重初始化，以及 Truncated BPTT。只加梯度裁剪，通常只能救爆炸，救不了长期记忆。
        """
    )


def render_lstm_guide(forget_bias: float, input_bias: float, output_bias: float, candidate_strength: float) -> None:
    st.markdown(
        f"""
        **图怎么看：**LSTM 有两条状态线：细胞状态 c_t 更像长期记忆，隐藏状态 h_t 更像当前要暴露给外部的短期摘要。热力图中“遗忘门”决定旧记忆保留多少，“输入门”决定新候选写入多少，“输出门”决定暴露多少记忆到 h_t。当前遗忘门偏置 **{forget_bias:.1f}**，输入门偏置 **{input_bias:.1f}**，输出门偏置 **{output_bias:.1f}**，候选记忆强度 **{candidate_strength:.1f}**。

        **RNN 与 LSTM 的差异：**普通 RNN 只有 h_t 一条状态，所有历史都被挤在同一条通道里；LSTM 用 c_t 建立更直接的记忆通道，再用门控决定保留、写入和输出。这就是 LSTM 更擅长长依赖的原因。

        **参数怎么调：**四个滑块范围分别是 -3.0 到 3.0、-3.0 到 3.0、-3.0 到 3.0、0.1 到 2.5。偏置越高，对应门平均越开；候选记忆强度越高，新写入内容越强。工程上常把遗忘门偏置初始化为正值，让模型一开始更愿意保留旧记忆。

        > 互动：把“遗忘门偏置”调到 -3.0，再调到 3.0。观察“新记忆”一行如何变化。思考：为什么遗忘门太关会让长期信息断掉，太开又可能把旧噪声一直带下去？
        """
    )


def render_gru_guide(update_bias: float, reset_bias: float, candidate_strength: float) -> None:
    st.markdown(
        f"""
        **图怎么看：**GRU 没有独立的 c_t，它把记忆直接放在 h_t 中。更新门 z_t 决定旧隐藏态和候选状态的混合比例，重置门 r_t 决定计算候选状态时看多少旧信息。当前更新门偏置 **{update_bias:.1f}**，重置门偏置 **{reset_bias:.1f}**，候选状态强度 **{candidate_strength:.1f}**。

        **LSTM 与 GRU 的差异：**LSTM 有输入门、遗忘门、输出门和候选记忆，控制更细；GRU 把输入/遗忘合并成更新门，参数更少、训练更快。页面指标卡用“4 组门”和“3 组门”说明了这个结构差异。

        **参数怎么调：**“更新门偏置”和“重置门偏置”的范围都是 -3.0 到 3.0；“候选状态强度”的范围是 0.1 到 2.5。更新门越开，GRU 越愿意采用新候选；重置门越关，候选状态越少依赖旧历史。

        > 互动：把“更新门偏置”调到 -3.0，再调到 3.0。观察新隐藏态更像旧隐藏态还是候选状态。思考：为什么 GRU 常作为 LSTM 的轻量替代，但不是所有长记忆任务都一定优于 LSTM？
        """
    )


def render_bidirectional_guide(seq_len: int, merge_mode: str) -> None:
    st.markdown(
        f"""
        **图怎么看：**每个位置同时有正向隐藏态和反向隐藏态。正向读左侧历史，反向读右侧未来，最后按 **{merge_mode}** 合并成当前位置输出。当前序列长度为 **{seq_len}**。

        **为什么需要双向：**文本分类、序列标注、实体抽取等任务通常已经看到了完整句子，因此一个词的左侧和右侧都可能提供线索。单向 RNN 只能利用过去，双向 RNN 能同时利用两边上下文。

        **边界条件：**双向 RNN 不适合严格在线预测和自回归生成，因为这些场景不能偷看未来 token。做实时语音、实时传感器预测时，通常只能用单向或有限延迟的双向结构。

        > 互动：把“序列长度”调长，观察每个位置都多了一条反向信息路径。思考：为什么双向结构适合文本分类，却不适合一步一步生成下一个字符？
        """
    )


def render_forecast_guide(
    cell_type: str,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    lr: float,
    epochs: int,
    noise: float,
    dropout: float,
    test_loss: float,
) -> None:
    st.markdown(
        f"""
        **图怎么看：**左图是训练损失，越平滑下降说明优化越稳定；右图把输入窗口、真实下一步、模型预测和滚动预测放在一起。滚动预测会把模型自己的输出继续喂回去，所以错误会逐步累积。当前模型类型是 **{cell_type}**，窗口长度 **{seq_len}**，隐藏单元 **{hidden_size}**，层数 **{num_layers}**，学习率 **{lr}**，训练 **{epochs}** epoch，噪声强度 **{noise:.2f}**，dropout **{dropout:.2f}**，测试 MSE **{test_loss:.5f}**。

        **RNN / LSTM / GRU 怎么选：**RNN 参数少但长依赖弱；LSTM 稳定但更重；GRU 介于两者之间，常作为速度和效果的折中。隐藏单元太少会欠拟合，窗口过长会增加 BPTT 难度，学习率太大容易让损失曲线抖动。

        **Teacher Forcing 的影子：**训练时模型看到的是干净的输入窗口，滚动预测时看到的是自己上一步预测，这和 Seq2Seq 中 Teacher Forcing 的训练/推理差异很像。症状是单步预测还可以，但长距离滚动预测逐渐漂移。

        > 极端值实验：把“隐藏单元”设为 8、把“窗口长度”设为 60，再把“学习率”设为 0.01。观察损失曲线和滚动预测是否更不稳定。排查时先降低学习率，再缩短窗口或换 LSTM/GRU，最后再增加隐藏维度。
        """
    )


def render_text_generation_guide(
    corpus_name: str,
    hidden_size: int,
    epochs: int,
    lr: float,
    length: int,
    temperature: float,
) -> None:
    st.markdown(
        f"""
        **图怎么看：**生成结果展示模型按字符一步步预测下一个字符；左图是训练交叉熵，右图是生成文本中的字符分布。当前语料是 **{corpus_name}**，隐藏单元 **{hidden_size}**，训练 **{epochs}** epoch，学习率 **{lr}**，生成长度 **{length}**，temperature **{temperature:.2f}**。

        **Teacher Forcing 与推理差异：**训练字符模型时，每一步通常用真实前缀预测下一个真实字符，这就是 Teacher Forcing 的简化形式；生成时，模型只能使用自己刚生成的字符。如果训练时太依赖真实前缀，生成时一个错误会带偏后续很多步。

        **参数怎么调：**temperature 低时概率分布更尖锐，输出更保守、更容易重复；temperature 高时更随机，但也更容易乱码。小语料和大 epoch 会让模型快速记忆训练文本，学习率太大则可能让 loss 震荡。

        > 极端值实验：把 temperature 调到 0.2，再调到 2.0。观察生成文本从重复保守到随机发散的变化。思考：生成质量差时，是模型没学会，还是采样温度把分布推得太极端？
        """
    )


with st.sidebar:
    st.header("导航")
    section = st.radio(
        "查看内容",
        [
            "RNN 展开",
            "梯度问题",
            "LSTM 门控",
            "GRU 门控",
            "双向 RNN",
            "时间序列预测",
            "文本生成",
        ],
        index=0,
    )
    st.divider()
    st.caption("所有演示均为本地小规模模型或确定性模拟，重点展示结构和动态。")


st.markdown(
    """
    <div class="hero">
        <h1>RNN 与序列模型</h1>
        <p>从循环状态、梯度传播、LSTM/GRU 门控，到双向结构、时间序列预测和字符级文本生成，把序列模型的核心机制拆成可调的可视化实验。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_sequence_learning_map()


if section == "RNN 展开":
    st.subheader("1. RNN 的基本结构和循环机制")
    c1, c2, c3 = st.columns(3)
    seq_len = c1.slider("展开时间步", 3, 12, 7, 1)
    active_step = c2.slider("当前时间步", 1, seq_len, min(4, seq_len), 1)
    recurrent_scale = c3.slider("循环权重尺度", 0.1, 1.5, 0.85, 0.05)
    st.pyplot(plot_rnn_unroll(seq_len, active_step), clear_figure=True)
    st.markdown(
        """
        <div class="formula">
        h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)<br>
        y_t = W_hy h_t + b_y
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("隐藏状态如何携带历史信息")
    d1, d2 = st.columns(2)
    hidden_size = d1.slider("隐藏单元数", 4, 16, 8, 1)
    input_scale = d2.slider("输入强度", 0.2, 2.0, 1.0, 0.1)
    st.pyplot(plot_hidden_dynamics(seq_len * 6, hidden_size, recurrent_scale, input_scale), clear_figure=True)
    render_rnn_unroll_guide(seq_len, active_step, recurrent_scale, hidden_size, input_scale)

elif section == "梯度问题":
    st.subheader("2. 梯度消失和梯度爆炸")
    c1, c2, c3 = st.columns(3)
    seq_len = c1.slider("反向传播时间距离", 10, 120, 60, 5)
    jacobian_scale = c2.slider("循环 Jacobian 尺度", 0.70, 1.30, 0.98, 0.01)
    saturation = c3.slider("tanh 饱和程度", 0.00, 0.70, 0.12, 0.01)
    fig, metrics = plot_gradient_flow(seq_len, jacobian_scale, saturation)
    st.pyplot(fig, clear_figure=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("单步有效增益", f"{metrics['effective_gain']:.3f}")
    m2.metric("激活局部导数", f"{metrics['local_derivative']:.3f}")
    m3.metric("最终梯度范数", f"{metrics['final_norm']:.2e}")
    st.markdown(
        """
        <div class="note">
        RNN 反向传播会反复乘以相似的局部 Jacobian。单步有效增益小于 1 时，远处梯度指数衰减；大于 1 时，梯度可能指数放大。梯度裁剪能限制爆炸，但不能真正恢复已经消失的远距离信号。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_gradient_issue_guide(seq_len, jacobian_scale, saturation, metrics)

elif section == "LSTM 门控":
    st.subheader("3. LSTM 门控机制")
    c1, c2, c3, c4 = st.columns(4)
    forget_bias = c1.slider("遗忘门偏置", -3.0, 3.0, 1.0, 0.1)
    input_bias = c2.slider("输入门偏置", -3.0, 3.0, 0.0, 0.1)
    output_bias = c3.slider("输出门偏置", -3.0, 3.0, 0.3, 0.1)
    candidate_strength = c4.slider("候选记忆强度", 0.1, 2.5, 1.1, 0.1)
    values = lstm_step_demo(forget_bias, input_bias, output_bias, candidate_strength)
    st.pyplot(plot_lstm_flow(values), clear_figure=True)
    st.markdown(
        """
        <div class="formula">
        f_t = sigmoid(...), i_t = sigmoid(...), g_t = tanh(...), o_t = sigmoid(...)<br>
        c_t = f_t * c_{t-1} + i_t * g_t<br>
        h_t = o_t * tanh(c_t)
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_lstm_guide(forget_bias, input_bias, output_bias, candidate_strength)

elif section == "GRU 门控":
    st.subheader("4. GRU 的简化门控机制")
    c1, c2, c3 = st.columns(3)
    update_bias = c1.slider("更新门偏置", -3.0, 3.0, 0.4, 0.1)
    reset_bias = c2.slider("重置门偏置", -3.0, 3.0, 0.0, 0.1)
    candidate_strength = c3.slider("候选状态强度", 0.1, 2.5, 1.1, 0.1)
    values = gru_step_demo(update_bias, reset_bias, candidate_strength)
    st.pyplot(plot_gru_flow(values), clear_figure=True)
    lstm_params = 4 * (64 * 32 + 64 * 64 + 64)
    gru_params = 3 * (64 * 32 + 64 * 64 + 64)
    m1, m2, m3 = st.columns(3)
    m1.metric("LSTM 参数比例", "4 组门")
    m2.metric("GRU 参数比例", "3 组门")
    m3.metric("GRU / LSTM", f"{gru_params / lstm_params:.0%}")
    st.markdown(
        """
        <div class="note">
        GRU 把 LSTM 的输入门和遗忘门合并为更新门，并取消独立细胞状态。它通常参数更少、训练更快，但在需要精细长期记忆控制的任务上不一定总是优于 LSTM。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_gru_guide(update_bias, reset_bias, candidate_strength)

elif section == "双向 RNN":
    st.subheader("5. 双向 RNN 结构")
    c1, c2 = st.columns(2)
    seq_len = c1.slider("序列长度", 4, 12, 8, 1)
    merge_mode = c2.selectbox("双向合并方式", ["concat", "sum", "mean"], index=0)
    st.pyplot(plot_bidirectional_rnn(seq_len, merge_mode), clear_figure=True)
    st.markdown(
        """
        <div class="note">
        双向 RNN 对每个位置同时读取左侧历史和右侧未来，适合分类、标注、抽取等已经看到完整输入的任务。它不适合严格在线预测，因为在线场景不能提前读取未来 token。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_bidirectional_guide(seq_len, merge_mode)

elif section == "时间序列预测":
    st.subheader("6. 简单时间序列预测 demo")
    c1, c2, c3, c4 = st.columns(4)
    cell_type = c1.selectbox("模型类型", ["RNN", "LSTM", "GRU"], index=1)
    seq_len = c2.slider("窗口长度", 12, 60, 32, 4)
    hidden_size = c3.slider("隐藏单元", 8, 96, 32, 8)
    epochs = c4.slider("训练 epoch", 20, 220, 90, 10)
    d1, d2, d3, d4 = st.columns(4)
    num_layers = d1.slider("层数", 1, 3, 1, 1)
    lr = d2.select_slider("学习率", options=[0.0005, 0.001, 0.003, 0.01], value=0.003)
    noise = d3.slider("噪声强度", 0.0, 0.4, 0.08, 0.02)
    dropout = d4.slider("dropout", 0.0, 0.5, 0.0, 0.05)
    with st.spinner("训练小型序列模型..."):
        result = train_forecast_model(cell_type, seq_len, hidden_size, num_layers, lr, epochs, noise, dropout)
    st.pyplot(plot_forecast_result(result), clear_figure=True)
    st.metric("测试 MSE", f"{result.test_loss:.5f}")
    render_forecast_guide(cell_type, seq_len, hidden_size, num_layers, lr, epochs, noise, dropout, result.test_loss)

elif section == "文本生成":
    st.subheader("7. 简单文本生成 demo")
    c1, c2, c3, c4 = st.columns(4)
    corpus_name = c1.selectbox("训练语料", list(TEXT_CORPORA), index=0)
    hidden_size = c2.slider("隐藏单元", 16, 128, 64, 16)
    epochs = c3.slider("训练 epoch", 80, 700, 260, 20)
    lr = c4.select_slider("学习率", options=[0.001, 0.003, 0.01, 0.03], value=0.01)
    d1, d2, d3 = st.columns(3)
    prefix = d1.text_input("起始文本", value=TEXT_CORPORA[corpus_name][0])
    length = d2.slider("生成长度", 10, 120, 50, 5)
    temperature = d3.slider("temperature", 0.2, 2.0, 0.8, 0.05)

    with st.spinner("训练字符级 GRU 语言模型..."):
        bundle = train_char_model(corpus_name, hidden_size, epochs, lr)
    generated = generate_text(
        bundle["model"],
        prefix,
        bundle["stoi"],
        bundle["itos"],
        length,
        temperature,
    )
    st.markdown(f"<div class='callout'><strong>生成结果：</strong><br>{generated}</div>", unsafe_allow_html=True)
    st.pyplot(plot_char_training(bundle["losses"], generated, bundle["chars"]), clear_figure=True)
    st.markdown(
        """
        <div class="note">
        这是字符级模型：每一步只根据已经生成的字符预测下一个字符。temperature 越低越保守，越高越随机；小语料会很快记住训练文本，因此这里主要用于观察序列生成机制。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_text_generation_guide(corpus_name, hidden_size, epochs, lr, length, temperature)
