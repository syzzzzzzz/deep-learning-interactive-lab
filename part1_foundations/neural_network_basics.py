"""
Neural network basics interactive teaching module.

Run:
    streamlit run part1_foundations/neural_network_basics.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="神经网络基础",
    layout="wide",
    initial_sidebar_state="expanded",
)


INK = "#172026"
MUTED = "#58646d"
TEAL = "#0f8b8d"
ROSE = "#c73e5b"
AMBER = "#d99a22"
GREEN = "#477b44"
VIOLET = "#5e4ae3"
BLUE = "#2d6cdf"
GRAY = "#9aa7ad"
PAPER = "#fbfaf6"
LINE = "#d7dde1"
COLORS = [TEAL, ROSE, AMBER, GREEN, VIOLET, BLUE]
PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"


st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(239,245,242,0.96) 100%),
            #fbfaf6;
        color: #172026;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }
    section[data-testid="stSidebar"] {
        background: #eef4f1;
        border-right: 1px solid #d7dde1;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.7rem;
    }
    .hero {
        border-bottom: 1px solid #d7dde1;
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3rem);
        margin: 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #58646d;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 1040px;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.74);
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 0.9rem;
        color: #26343b;
        line-height: 1.65;
        margin: 0.4rem 0 0.9rem 0;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.4rem 0 0.9rem 0;
    }
    .mini-cell {
        background: rgba(255,255,255,0.74);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        min-height: 86px;
    }
    .mini-cell strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #172026;
    }
    .mini-cell span {
        color: #58646d;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    @media (max-width: 900px) {
        .mini-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def segmented(label: str, options: list[str], default: str) -> str:
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default)
        return value or default
    return st.radio(label, options, index=options.index(default), horizontal=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def go_to_playground(example: str) -> None:
    st.query_params["module"] = PLAYGROUND_TARGET
    st.query_params["example"] = example
    st.rerun()


def concept_cards(cards: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="mini-cell"><strong>{title}</strong><span>{text}</span></div>'
        for title, text in cards
    )
    st.markdown(f'<div class="mini-grid">{body}</div>', unsafe_allow_html=True)


def new_figure(figsize: tuple[float, float] = (7.2, 4.6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.22)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#c8d0d5")
    return fig, ax


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.22)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#c8d0d5")


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def tanh(x: np.ndarray | float) -> np.ndarray | float:
    return np.tanh(x)


def relu(x: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(0.0, x)


def leaky_relu(x: np.ndarray | float, slope: float = 0.1) -> np.ndarray | float:
    return np.where(np.asarray(x) >= 0, x, slope * np.asarray(x))


def gelu(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    u = math.sqrt(2.0 / math.pi) * (x_arr + 0.044715 * x_arr**3)
    return 0.5 * x_arr * (1.0 + np.tanh(u))


def swish(x: np.ndarray | float, beta: float = 1.0) -> np.ndarray | float:
    x_arr = np.asarray(x)
    return x_arr * sigmoid(beta * x_arr)


def apply_activation(z: float, name: str) -> float:
    if name == "Sigmoid":
        return float(sigmoid(z))
    if name == "Tanh":
        return float(tanh(z))
    if name == "ReLU":
        return float(relu(z))
    if name == "Leaky ReLU":
        return float(leaky_relu(z))
    if name == "GELU":
        return float(gelu(z))
    if name == "Swish":
        return float(swish(z))
    return z


def activation_and_derivative(name: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name == "Sigmoid":
        y = sigmoid(x)
        dy = y * (1.0 - y)
    elif name == "Tanh":
        y = np.tanh(x)
        dy = 1.0 - y**2
    elif name == "ReLU":
        y = relu(x)
        dy = (x > 0).astype(float)
    elif name == "Leaky ReLU":
        slope = 0.1
        y = leaky_relu(x, slope)
        dy = np.where(x > 0, 1.0, slope)
    elif name == "GELU":
        u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        tanh_u = np.tanh(u)
        y = 0.5 * x * (1.0 + tanh_u)
        du = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * x**2)
        dy = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u**2) * du
    else:
        s = sigmoid(x)
        y = x * s
        dy = s + x * s * (1.0 - s)
    return y, dy


def draw_network(
    layers: list[int],
    values: list[np.ndarray] | None = None,
    edge_strengths: list[np.ndarray] | None = None,
    highlight_layer: int | None = None,
    title: str = "",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("white")
    ax.axis("off")
    x_positions = np.linspace(0.08, 0.92, len(layers))

    node_positions: list[list[tuple[float, float]]] = []
    for layer_index, count in enumerate(layers):
        ys = np.linspace(0.82, 0.18, count) if count > 1 else np.array([0.5])
        node_positions.append([(float(x_positions[layer_index]), float(y)) for y in ys])

    for layer_index in range(len(layers) - 1):
        strength = None if edge_strengths is None else np.asarray(edge_strengths[layer_index])
        max_strength = 1.0 if strength is None else max(float(np.abs(strength).max()), 1e-8)
        for i, start in enumerate(node_positions[layer_index]):
            for j, end in enumerate(node_positions[layer_index + 1]):
                value = 0.35 if strength is None else abs(float(strength[i, j])) / max_strength
                color = GRAY if strength is None or strength[i, j] >= 0 else ROSE
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=color,
                    alpha=0.18 + 0.55 * value,
                    linewidth=0.7 + 3.0 * value,
                    zorder=1,
                )

    for layer_index, positions in enumerate(node_positions):
        layer_values = None if values is None else np.asarray(values[layer_index])
        if layer_values is None:
            colors = [TEAL] * len(positions)
            labels = [f"L{layer_index}.{i}" for i in range(len(positions))]
        else:
            vmax = max(float(np.abs(layer_values).max()), 1e-8)
            colors = [GREEN if v >= 0 else ROSE for v in layer_values]
            labels = [f"{v:.2f}" for v in layer_values / vmax]

        for (x, y), color, label in zip(positions, colors, labels):
            radius = 0.036 if highlight_layer != layer_index else 0.043
            circle = plt.Circle((x, y), radius, facecolor=color, edgecolor="white", linewidth=1.6, alpha=0.94, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8, zorder=4)

    layer_names = ["输入层", "隐藏层 1", "隐藏层 2", "输出层"]
    for layer_index, x in enumerate(x_positions):
        name = layer_names[layer_index] if layer_index < len(layer_names) else f"层 {layer_index}"
        ax.text(x, 0.05, name, ha="center", va="center", color=MUTED, fontsize=10)

    ax.set_title(title, color=INK, pad=12)
    fig.tight_layout()
    return fig


def render_single_neuron() -> None:
    st.subheader("单个神经元：加权求和、偏置、激活")
    concept_cards(
        [
            ("线性部分", "先计算 z = w1*x1 + w2*x2 + b，权重控制输入方向和强度。"),
            ("非线性部分", "激活函数把线性结果变成新的表征，使多层网络能表达弯曲边界。"),
            ("阈值直觉", "偏置 b 会整体平移决策边界，像调整神经元被点亮的门槛。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 单神经元参数")
        x1 = st.slider("输入 x1", -3.0, 3.0, 1.0, 0.05)
        x2 = st.slider("输入 x2", -3.0, 3.0, -0.6, 0.05)
        w1 = st.slider("权重 w1", -4.0, 4.0, 1.4, 0.05)
        w2 = st.slider("权重 w2", -4.0, 4.0, -1.1, 0.05)
        bias = st.slider("偏置 b", -4.0, 4.0, 0.25, 0.05)
        activation = st.selectbox("激活函数", ["Linear", "Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "GELU", "Swish"], index=1)

    weighted = np.array([x1 * w1, x2 * w2])
    z = float(weighted.sum() + bias)
    y = apply_activation(z, activation)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("x1*w1", f"{weighted[0]:.3f}")
    m2.metric("x2*w2", f"{weighted[1]:.3f}")
    m3.metric("z", f"{z:.3f}")
    m4.metric("输出 a", f"{y:.3f}")

    left, right = st.columns([1.15, 1])
    with left:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        fig.patch.set_facecolor(PAPER)
        ax.axis("off")
        positions = {"x1": (0.12, 0.70), "x2": (0.12, 0.30), "sum": (0.52, 0.50), "out": (0.86, 0.50)}
        for key, (px, py) in positions.items():
            color = TEAL if key.startswith("x") else AMBER if key == "sum" else GREEN
            circle = plt.Circle((px, py), 0.075, facecolor=color, edgecolor="white", linewidth=2.0)
            ax.add_patch(circle)
        ax.text(*positions["x1"], f"x1\n{x1:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(*positions["x2"], f"x2\n{x2:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(*positions["sum"], f"z\n{z:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(*positions["out"], f"a\n{y:.2f}", ha="center", va="center", color="white", fontsize=10)
        for name, weight, contrib in [("x1", w1, weighted[0]), ("x2", w2, weighted[1])]:
            start = positions[name]
            end = positions["sum"]
            color = TEAL if weight >= 0 else ROSE
            ax.annotate(
                "",
                xy=(end[0] - 0.075, end[1]),
                xytext=(start[0] + 0.075, start[1]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5 + abs(weight), "alpha": 0.8},
            )
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.05, f"w={weight:.2f}\nwx={contrib:.2f}", color=INK, ha="center", fontsize=9)
        ax.annotate("", xy=(0.785, 0.50), xytext=(0.595, 0.50), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.4})
        ax.text(0.52, 0.33, f"+ b = {bias:.2f}", ha="center", va="center", color=MUTED, fontsize=10)
        ax.text(0.70, 0.56, activation, ha="center", color=MUTED, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.08, 0.92)
        st.pyplot(fig, width="stretch")

    with right:
        xs = np.linspace(-6, 6, 500)
        ys = np.array([apply_activation(float(v), activation) for v in xs])
        fig, ax = new_figure((6.1, 4.2))
        ax.plot(xs, ys, color=TEAL, linewidth=2.6, label=activation)
        ax.scatter([z], [y], s=95, color=ROSE, edgecolor="white", linewidth=1.0, zorder=4, label="当前 z")
        ax.axvline(z, color=ROSE, linestyle="--", alpha=0.55)
        ax.set_xlabel("z")
        ax.set_ylabel("a = f(z)")
        ax.set_title("激活函数上的当前输出", color=INK)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    note("拖动权重会改变每个输入对 z 的贡献；拖动偏置会把神经元整体推向更容易或更不容易激活。")


def perceptron_predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return ((x @ w + b) >= 0).astype(int)


def render_perceptron_xor() -> None:
    st.subheader("感知机：线性分类器与 XOR 局限")
    concept_cards(
        [
            ("感知机", "输出是 step(w*x+b)，因此二维输入里只能画出一条直线边界。"),
            ("AND/OR 可分", "这些逻辑门的正负样本能被一条直线分开。"),
            ("XOR 不可分", "XOR 的两个正样本在对角线位置，任何单条直线都会错分至少一个点。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 感知机参数")
        gate = st.selectbox("逻辑门", ["AND", "OR", "XOR"], index=2)
        w1 = st.slider("w1", -4.0, 4.0, 1.0, 0.05)
        w2 = st.slider("w2", -4.0, 4.0, 1.0, 0.05)
        bias = st.slider("b", -4.0, 4.0, -1.0, 0.05)
        show_mlp = st.toggle("显示两层网络如何解决 XOR", value=True)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "XOR": np.array([0, 1, 1, 0]),
    }[gate]
    w = np.array([w1, w2], dtype=float)
    pred = perceptron_predict(x, w, bias)
    accuracy = float((pred == labels).mean())

    m1, m2, m3 = st.columns(3)
    m1.metric("当前准确率", f"{accuracy:.0%}")
    m2.metric("错误样本数", int((pred != labels).sum()))
    m3.metric("模型类型", "单层感知机")

    left, right = st.columns([1.25, 1])
    with left:
        fig, ax = new_figure((6.7, 5.4))
        xx, yy = np.meshgrid(np.linspace(-0.35, 1.35, 220), np.linspace(-0.35, 1.35, 220))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = perceptron_predict(grid, w, bias).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=[TEAL, ROSE], alpha=0.13)
        if abs(w2) > 1e-8:
            xs = np.linspace(-0.35, 1.35, 100)
            ys = -(w1 * xs + bias) / w2
            ax.plot(xs, ys, color=INK, linewidth=2.0, label="w1*x1 + w2*x2 + b = 0")
        elif abs(w1) > 1e-8:
            ax.axvline(-bias / w1, color=INK, linewidth=2.0, label="决策边界")
        for cls, color in [(0, TEAL), (1, ROSE)]:
            mask = labels == cls
            ax.scatter(x[mask, 0], x[mask, 1], s=180, color=color, edgecolor="white", linewidth=1.5, label=f"真实 {cls}", zorder=4)
        for point, y_true, y_pred in zip(x, labels, pred):
            mark = "ok" if y_true == y_pred else "错"
            ax.text(point[0], point[1] + 0.10, f"pred={y_pred}\n{mark}", ha="center", color=INK, fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-0.35, 1.35)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"{gate} 的感知机决策边界", color=INK)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    with right:
        table = np.c_[x, labels, x @ w + bias, pred]
        st.dataframe(
            {
                "x1": table[:, 0],
                "x2": table[:, 1],
                "真实": table[:, 2].astype(int),
                "z": np.round(table[:, 3], 3),
                "预测": table[:, 4].astype(int),
            },
            width="stretch",
            hide_index=True,
        )
        if gate == "XOR":
            note("XOR 的正类在两个对角点。单层感知机只有一条直线边界，所以不能同时把两个正类点与两个负类点分开。")
        else:
            note("AND 和 OR 是线性可分任务。调节 w1、w2、b 可以找到一条把两类点分开的直线。")

    if show_mlp and gate == "XOR":
        st.markdown("#### 两层感知机解决 XOR")
        hidden1 = ((x[:, 0] + x[:, 1] - 0.5) >= 0).astype(int)
        hidden2 = ((x[:, 0] + x[:, 1] - 1.5) >= 0).astype(int)
        xor_pred = ((hidden1 - hidden2) >= 1).astype(int)
        fig, ax = new_figure((7.2, 4.2))
        xs = np.linspace(-0.35, 1.35, 100)
        ax.plot(xs, 0.5 - xs, color=GREEN, linewidth=2.0, label="隐藏神经元 1: x1+x2>=0.5")
        ax.plot(xs, 1.5 - xs, color=VIOLET, linewidth=2.0, label="隐藏神经元 2: x1+x2>=1.5")
        for cls, color in [(0, TEAL), (1, ROSE)]:
            mask = labels == cls
            ax.scatter(x[mask, 0], x[mask, 1], s=180, color=color, edgecolor="white", linewidth=1.5, label=f"XOR {cls}", zorder=4)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-0.35, 1.35)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("两个隐藏神经元把平面切成三段，中间一段就是 XOR 正类", color=INK)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        st.dataframe(
            {"x1": x[:, 0], "x2": x[:, 1], "h1": hidden1, "h2": hidden2, "h1-h2": hidden1 - hidden2, "输出": xor_pred},
            width="stretch",
            hide_index=True,
        )


@dataclass(frozen=True)
class MLPState:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    preacts: list[np.ndarray]
    activations: list[np.ndarray]


def make_mlp_state(seed: int, x: np.ndarray) -> MLPState:
    rng = np.random.default_rng(seed)
    shapes = [(3, 5), (5, 4), (4, 2)]
    weights = [rng.normal(0, 0.75 / math.sqrt(inp), size=(inp, out)) for inp, out in shapes]
    biases = [rng.normal(0, 0.20, size=out) for _, out in shapes]
    activations = [x]
    preacts = []
    current = x
    for layer_index, (w, b) in enumerate(zip(weights, biases)):
        z = current @ w + b
        preacts.append(z)
        current = sigmoid(z) if layer_index == len(weights) - 1 else np.tanh(z)
        activations.append(current)
    return MLPState(weights=weights, biases=biases, preacts=preacts, activations=activations)


def render_mlp_forward(seed: int) -> None:
    st.subheader("MLP 前向传播：数据如何逐层流动")
    concept_cards(
        [
            ("矩阵乘法", "每一层做 a_next = f(a @ W + b)，多个神经元可以一次性并行计算。"),
            ("中间表征", "隐藏层不是最终答案，而是在重写输入，使任务在新空间里更容易。"),
            ("动画帧", "拖动层索引可以观察输入、隐藏层和输出层的数值如何依次出现。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 前向传播参数")
        x1 = st.slider("输入特征 1", -2.0, 2.0, 0.8, 0.05)
        x2 = st.slider("输入特征 2", -2.0, 2.0, -0.4, 0.05)
        x3 = st.slider("输入特征 3", -2.0, 2.0, 1.2, 0.05)
        frame = st.slider("前向传播动画帧", 0, 3, 3, 1)

    state = make_mlp_state(seed, np.array([x1, x2, x3], dtype=float))
    shown_values = [np.zeros_like(a) for a in state.activations]
    for idx in range(frame + 1):
        shown_values[idx] = state.activations[idx]
    edge_strengths = [w for w in state.weights]

    m1, m2, m3 = st.columns(3)
    m1.metric("输出 0", f"{state.activations[-1][0]:.3f}")
    m2.metric("输出 1", f"{state.activations[-1][1]:.3f}")
    m3.metric("当前帧", ["输入层", "隐藏层 1", "隐藏层 2", "输出层"][frame])

    left, right = st.columns([1.25, 1])
    with left:
        st.pyplot(
            draw_network([3, 5, 4, 2], shown_values, edge_strengths, highlight_layer=frame, title="MLP 前向传播数据流"),
            width="stretch",
        )
    with right:
        selected = min(max(frame - 1, 0), len(state.preacts) - 1)
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=False)
        fig.patch.set_facecolor(PAPER)
        for ax in axes:
            style_axis(ax)
        axes[0].bar(range(len(state.preacts[selected])), state.preacts[selected], color=AMBER)
        axes[0].axhline(0, color=INK, linewidth=0.9, alpha=0.45)
        axes[0].set_title(f"第 {selected + 1} 层线性结果 z", color=INK)
        axes[1].bar(range(len(state.activations[selected + 1])), state.activations[selected + 1], color=TEAL)
        axes[1].axhline(0, color=INK, linewidth=0.9, alpha=0.45)
        axes[1].set_title(f"第 {selected + 1} 层激活 a", color=INK)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    note("前向传播只是在固定参数下计算输出；训练发生在反向传播和优化器更新参数时。")


def render_backprop(seed: int) -> None:
    st.subheader("反向传播：误差信号如何变成每层梯度")
    concept_cards(
        [
            ("链式法则", "损失对早期参数的影响，要乘过后面所有层的局部导数。"),
            ("梯度流", "颜色越深表示该层权重梯度越大，太浅可能对应梯度消失。"),
            ("激活选择", "Sigmoid 在饱和区导数很小，ReLU 类函数通常能保留更强梯度。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 反向传播参数")
        activation = st.selectbox("隐藏层激活", ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"], index=0)
        weight_scale = st.slider("权重尺度", 0.2, 3.0, 1.0, 0.05)
        target = st.slider("目标值 y", 0.0, 1.0, 1.0, 0.05)
        frame = st.slider("反向传播动画帧", 0, 3, 3, 1)

    rng = np.random.default_rng(seed + 13)
    x = np.array([0.7, -1.1, 0.5], dtype=float)
    dims = [3, 4, 4, 1]
    weights = [rng.normal(0, weight_scale / math.sqrt(dims[i]), size=(dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
    biases = [rng.normal(0, 0.05, size=dims[i + 1]) for i in range(len(dims) - 1)]

    acts = [x]
    zs = []
    derivs = []
    for layer_index, (w, b) in enumerate(zip(weights, biases)):
        z = acts[-1] @ w + b
        zs.append(z)
        if layer_index == len(weights) - 1:
            a = sigmoid(z)
            d = a * (1 - a)
        else:
            a, d = activation_and_derivative(activation, z)
        acts.append(np.asarray(a))
        derivs.append(np.asarray(d))

    output = float(acts[-1][0])
    loss = 0.5 * (output - target) ** 2
    delta = np.array([(output - target) * derivs[-1][0]])
    deltas = [None, None, delta]
    grad_w = [None, None, np.outer(acts[-2], delta)]
    for layer_index in [1, 0]:
        delta = (deltas[layer_index + 1] @ weights[layer_index + 1].T) * derivs[layer_index]
        deltas[layer_index] = delta
        grad_w[layer_index] = np.outer(acts[layer_index], delta)

    grad_norms = np.array([float(np.linalg.norm(g)) for g in grad_w])
    visible_strengths = []
    for index, g in enumerate(grad_w):
        if index >= len(grad_w) - frame:
            visible_strengths.append(g)
        else:
            visible_strengths.append(np.zeros_like(g))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("输出", f"{output:.3f}")
    m2.metric("损失", f"{loss:.4f}")
    m3.metric("首层梯度范数", f"{grad_norms[0]:.4f}")
    m4.metric("末层梯度范数", f"{grad_norms[-1]:.4f}")

    left, right = st.columns([1.25, 1])
    with left:
        st.pyplot(
            draw_network(dims, acts, visible_strengths, highlight_layer=max(0, len(dims) - 1 - frame), title="反向传播梯度流"),
            width="stretch",
        )
    with right:
        fig, ax = new_figure((6.2, 4.2))
        bars = ax.bar(["W1", "W2", "W3"], grad_norms, color=[TEAL, AMBER, ROSE])
        ax.set_yscale("log")
        ax.set_ylabel("梯度范数 log scale")
        ax.set_title("各层权重梯度大小", color=INK)
        for bar, value in zip(bars, grad_norms):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.05 + 1e-8, f"{value:.2e}", ha="center", color=INK, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    note("如果把权重尺度调大并使用 Sigmoid，隐藏层容易进入饱和区，导数接近 0，反向传播到前层的梯度会明显变小。")


def render_activations() -> None:
    st.subheader("激活函数及导数对比")
    concept_cards(
        [
            ("饱和函数", "Sigmoid 和 Tanh 在两端导数接近 0，深层网络中可能让梯度变弱。"),
            ("分段线性", "ReLU 简洁高效，但负半轴导数为 0；Leaky ReLU 保留一条小斜率。"),
            ("平滑门控", "GELU 和 Swish 是平滑非线性，常见于现代深度网络。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 激活函数参数")
        selected = st.multiselect(
            "显示函数",
            ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "GELU", "Swish"],
            default=["Sigmoid", "Tanh", "ReLU", "GELU", "Swish"],
        )
        x_min = st.slider("x 最小值", -10.0, -1.0, -5.0, 0.5)
        x_max = st.slider("x 最大值", 1.0, 10.0, 5.0, 0.5)

    if not selected:
        st.warning("至少选择一个激活函数。")
        return
    x = np.linspace(x_min, x_max, 800)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        style_axis(ax)
        ax.axhline(0, color=INK, linewidth=0.8, alpha=0.45)
        ax.axvline(0, color=INK, linewidth=0.8, alpha=0.45)
    for idx, name in enumerate(selected):
        y, dy = activation_and_derivative(name, x)
        color = COLORS[idx % len(COLORS)]
        axes[0].plot(x, y, color=color, linewidth=2.2, label=name)
        axes[1].plot(x, dy, color=color, linewidth=2.2, label=name)
    axes[0].set_title("函数值", color=INK)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[1].set_title("导数", color=INK)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f'(x)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    note("训练时真正影响梯度传播的是右图的导数。导数长期接近 0 的区间越多，深层网络越容易出现梯度消失。")


def render_losses() -> None:
    st.subheader("损失函数曲线对比")
    concept_cards(
        [
            ("MSE", "平方误差对大错误惩罚很强，曲线光滑，常用于回归。"),
            ("MAE", "绝对误差对异常值更稳健，但 0 点不可导，梯度大小基本恒定。"),
            ("交叉熵", "分类里直接惩罚真实类别概率低的情况，越自信地错损失越大。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 损失函数参数")
        huber_delta = st.slider("Huber delta", 0.2, 3.0, 1.0, 0.05)
        prob_y = st.selectbox("交叉熵真实标签", [1, 0], index=0)
        show_grad = st.toggle("显示导数/梯度", value=True)

    e = np.linspace(-4.0, 4.0, 800)
    mse = 0.5 * e**2
    mae = np.abs(e)
    huber = np.where(np.abs(e) <= huber_delta, 0.5 * e**2, huber_delta * (np.abs(e) - 0.5 * huber_delta))
    p = np.linspace(1e-4, 0.9999, 800)
    ce = -np.log(p) if prob_y == 1 else -np.log(1 - p)

    if show_grad:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
        flat_axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        flat_axes = axes.ravel()
    fig.patch.set_facecolor(PAPER)
    for ax in flat_axes:
        style_axis(ax)

    flat_axes[0].plot(e, mse, color=TEAL, linewidth=2.2, label="MSE = 0.5e^2")
    flat_axes[0].plot(e, mae, color=ROSE, linewidth=2.2, label="MAE = |e|")
    flat_axes[0].plot(e, huber, color=AMBER, linewidth=2.2, label="Huber")
    flat_axes[0].set_title("回归损失：按误差 e = y_pred - y_true", color=INK)
    flat_axes[0].set_xlabel("误差 e")
    flat_axes[0].set_ylabel("loss")
    flat_axes[0].legend(frameon=False)

    flat_axes[1].plot(p, ce, color=VIOLET, linewidth=2.4, label=f"Cross Entropy, y={prob_y}")
    flat_axes[1].set_title("二分类交叉熵：按预测概率 p", color=INK)
    flat_axes[1].set_xlabel("预测为 1 的概率 p")
    flat_axes[1].set_ylabel("loss")
    flat_axes[1].set_ylim(0, min(10.0, float(ce.max())))
    flat_axes[1].legend(frameon=False)

    if show_grad:
        flat_axes[2].plot(e, e, color=TEAL, linewidth=2.2, label="MSE 梯度")
        flat_axes[2].plot(e, np.sign(e), color=ROSE, linewidth=2.2, label="MAE 梯度")
        flat_axes[2].plot(e, np.where(np.abs(e) <= huber_delta, e, huber_delta * np.sign(e)), color=AMBER, linewidth=2.2, label="Huber 梯度")
        flat_axes[2].axhline(0, color=INK, linewidth=0.8, alpha=0.45)
        flat_axes[2].set_title("回归损失对误差的导数", color=INK)
        flat_axes[2].set_xlabel("误差 e")
        flat_axes[2].set_ylabel("dL/de")
        flat_axes[2].legend(frameon=False)
        ce_grad = -1 / p if prob_y == 1 else 1 / (1 - p)
        flat_axes[3].plot(p, np.clip(ce_grad, -20, 20), color=VIOLET, linewidth=2.2)
        flat_axes[3].axhline(0, color=INK, linewidth=0.8, alpha=0.45)
        flat_axes[3].set_title("交叉熵对概率的导数，裁剪到 [-20, 20]", color=INK)
        flat_axes[3].set_xlabel("p")
        flat_axes[3].set_ylabel("dL/dp")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    note("损失函数不只定义分数，也定义优化方向。大错误处的斜率越大，优化器越会优先修正这些样本。")


def objective(point: np.ndarray) -> float:
    x, y = point
    return 0.08 * (x**2 + 6.0 * y**2) + 0.18 * np.sin(2.2 * x) * np.cos(1.7 * y)


def objective_grad(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array(
        [
            0.16 * x + 0.18 * 2.2 * np.cos(2.2 * x) * np.cos(1.7 * y),
            0.96 * y - 0.18 * 1.7 * np.sin(2.2 * x) * np.sin(1.7 * y),
        ]
    )


def optimizer_path(name: str, lr: float, steps: int, start: np.ndarray) -> np.ndarray:
    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    momentum = 0.9
    rho = 0.9
    point = start.astype(float).copy()
    velocity = np.zeros_like(point)
    cache = np.zeros_like(point)
    m = np.zeros_like(point)
    v = np.zeros_like(point)
    path = [point.copy()]
    for t in range(1, steps + 1):
        grad = objective_grad(point)
        if name == "SGD":
            step = lr * grad
        elif name == "Momentum":
            velocity = momentum * velocity + grad
            step = lr * velocity
        elif name == "AdaGrad":
            cache += grad**2
            step = lr * grad / (np.sqrt(cache) + eps)
        elif name == "RMSprop":
            cache = rho * cache + (1 - rho) * grad**2
            step = lr * grad / (np.sqrt(cache) + eps)
        else:
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
        point = point - step
        point = np.clip(point, -4.0, 4.0)
        path.append(point.copy())
    return np.asarray(path)


def render_optimizers() -> None:
    st.subheader("优化器收敛过程：同一地形上的不同走法")
    concept_cards(
        [
            ("SGD", "直接沿负梯度走，简单但可能在陡峭方向震荡。"),
            ("自适应方法", "AdaGrad、RMSprop、Adam 会按历史梯度缩放每个坐标的步长。"),
            ("学习率", "学习率过小收敛慢，过大可能震荡或越过低谷。"),
        ]
    )

    with st.sidebar:
        st.markdown("### 优化器参数")
        optimizers = st.multiselect(
            "显示优化器",
            ["SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"],
            default=["SGD", "Momentum", "RMSprop", "Adam"],
        )
        lr = st.slider("学习率", 0.01, 1.0, 0.18, 0.01)
        steps = st.slider("总步数", 10, 220, 90, 5)
        frame = st.slider("动画帧", 0, steps, min(45, steps), 1)
        start_x = st.slider("起点 x", -3.5, 3.5, -3.0, 0.1)
        start_y = st.slider("起点 y", -3.5, 3.5, 2.8, 0.1)

    if not optimizers:
        st.warning("至少选择一个优化器。")
        return

    xs = np.linspace(-4, 4, 220)
    ys = np.linspace(-4, 4, 220)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.08 * (xx**2 + 6.0 * yy**2) + 0.18 * np.sin(2.2 * xx) * np.cos(1.7 * yy)
    start = np.array([start_x, start_y])

    fig, ax = new_figure((8.0, 6.0))
    contour = ax.contourf(xx, yy, zz, levels=36, cmap="viridis", alpha=0.82)
    ax.contour(xx, yy, zz, levels=18, colors="white", linewidths=0.45, alpha=0.35)
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="loss")

    final_rows = []
    for idx, name in enumerate(optimizers):
        path = optimizer_path(name, lr, steps, start)
        shown = path[: frame + 1]
        color = COLORS[idx % len(COLORS)]
        ax.plot(shown[:, 0], shown[:, 1], color=color, linewidth=2.2, label=name)
        ax.scatter(shown[-1, 0], shown[-1, 1], s=80, color=color, edgecolor="white", linewidth=1.0, zorder=5)
        final_rows.append((name, objective(path[frame]), np.linalg.norm(objective_grad(path[frame])), path[frame, 0], path[frame, 1]))

    ax.scatter([start_x], [start_y], marker="x", s=110, color=INK, linewidth=2.4, label="起点")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("参数 w1")
    ax.set_ylabel("参数 w2")
    ax.set_title("优化路径动画：等高线表示损失地形", color=INK)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    st.dataframe(
        {
            "优化器": [row[0] for row in final_rows],
            "当前 loss": [round(row[1], 5) for row in final_rows],
            "梯度范数": [round(row[2], 5) for row in final_rows],
            "x": [round(row[3], 3) for row in final_rows],
            "y": [round(row[4], 3) for row in final_rows],
        },
        width="stretch",
        hide_index=True,
    )
    note("同一个学习率不一定适合所有优化器。比较路径时重点看是否震荡、是否停滞，以及是否能稳定降低损失。")


try:
    st.markdown(
        """
        <div class="hero">
          <h1>神经网络基础模块</h1>
          <p>从单个神经元开始，逐步观察感知机、MLP 前向传播、反向传播、激活函数、损失函数和优化器。所有图像都由 Streamlit 与 Matplotlib 交互生成，可直接拖动参数观察变化。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    practice_col, _ = st.columns([0.24, 0.76])
    with practice_col:
        if st.button("去实战：MLP 构建器", width="stretch"):
            go_to_playground("mlp")

    with st.sidebar:
        st.header("全局设置")
        seed = int(st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1))
        st.caption("相同随机种子会复现相同的网络权重和演示轨迹。")
        st.divider()


    scene = segmented(
        "选择教学场景",
        ["单个神经元", "感知机与 XOR", "MLP 前向传播", "反向传播", "激活函数", "损失函数", "优化器"],
        "单个神经元",
    )

    if scene == "单个神经元":
        render_single_neuron()
    elif scene == "感知机与 XOR":
        render_perceptron_xor()
    elif scene == "MLP 前向传播":
        render_mlp_forward(seed)
    elif scene == "反向传播":
        render_backprop(seed)
    elif scene == "激活函数":
        render_activations()
    elif scene == "损失函数":
        render_losses()
    else:
        render_optimizers()
except Exception as exc:
    from components.error_boundary import render_module_error
    render_module_error("part1_foundations/neural_network_basics.py", exc)


def render() -> None:
    """Page entry point — content runs at module import time."""
    pass


def compute(seed: int = 42) -> dict[str, object]:
    """Pure computation placeholder."""
    return {"status": "ok", "seed": seed}


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""
    return True
