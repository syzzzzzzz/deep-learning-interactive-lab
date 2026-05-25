"""
Graph neural network interactive introduction.

Run:
    streamlit run part4_transformer/gnn_intro.py
or:
    python main.py part4_transformer/gnn_intro
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "ink": "#172026",
    "muted": "#596772",
    "line": "#d8dee3",
    "blue": "#3268a8",
    "teal": "#0f8b8d",
    "rose": "#bf3f5b",
    "amber": "#c4871f",
    "green": "#3f7d58",
    "violet": "#7353ba",
    "paper": "#fbfaf6",
}

CLASS_COLORS = ["#3268a8", "#bf3f5b", "#3f7d58"]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="图神经网络入门",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(241,246,244,0.96) 100%), #fbfaf6;
        color: #172026;
    }
    .block-container { padding-top: 1.25rem; padding-bottom: 2.2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    section[data-testid="stSidebar"] {
        background: #eef4f2;
        border-right: 1px solid #d8dee3;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #d8dee3;
        border-radius: 8px;
        padding: 0.7rem;
    }
    .hero {
        border-bottom: 1px solid #d8dee3;
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3.15rem);
        line-height: 1.1;
        margin: 0;
    }
    .hero p {
        color: #596772;
        max-width: 980px;
        line-height: 1.75;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.76);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        color: #26343b;
        line-height: 1.7;
        margin: 0.35rem 0 0.85rem 0;
    }
    .formula {
        font-family: Consolas, Menlo, monospace;
        background: rgba(255,255,255,0.82);
        border: 1px solid #d8dee3;
        border-radius: 8px;
        padding: 0.72rem 0.9rem;
        line-height: 1.72;
        color: #1e2b32;
        overflow-x: auto;
        margin-bottom: 0.75rem;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 0.9rem 0;
    }
    .mini-card {
        background: rgba(255,255,255,0.8);
        border: 1px solid #d8dee3;
        border-radius: 8px;
        padding: 0.74rem 0.82rem;
        min-height: 114px;
    }
    .mini-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .mini-card p {
        color: #596772;
        margin: 0;
        line-height: 1.62;
        font-size: 0.92rem;
    }
    @media (max-width: 1000px) {
        .mini-grid { grid-template-columns: 1fr; }
        .mini-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>图神经网络入门</h1>
            <p>
            图神经网络把节点特征和图结构一起作为输入。这个页面用小图演示邻接矩阵、边列表、特征矩阵，
            再把 GCN 的邻居平均和 GAT 的注意力加权拆开，最后做一个节点分类的可视化 demo。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_concept_cards() -> None:
    cards = [
        ("图数据", "图通常写成 G=(V,E)，节点 V 带特征 X，边 E 形成邻接矩阵 A，也可以附带边权或边类型。"),
        ("消息传递", "每层 GNN 都做一次邻居聚合：收集邻居消息、合并、更新节点表示。层数越深，感受野越大。"),
        ("GCN", "GCN 使用归一化邻接矩阵做加权平均，再接线性变换和非线性，相当于结构感知的特征平滑。"),
        ("GAT", "GAT 不把所有邻居等权看待，而是学习注意力系数，让关键邻居贡献更大。"),
    ]
    html = '<div class="mini-grid">'
    for title, body in cards:
        html += f'<div class="mini-card"><strong>{title}</strong><p>{body}</p></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def build_graph(kind: str) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray, list[str]]:
    if kind == "社交网络":
        coords = np.array(
            [
                [-1.5, 0.8],
                [-0.8, 1.25],
                [-0.65, 0.35],
                [-1.35, -0.25],
                [0.65, 0.95],
                [1.45, 0.45],
                [0.75, -0.15],
                [1.65, -0.65],
                [0.05, 0.35],
                [-0.05, -0.75],
            ]
        )
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (4, 8), (2, 8), (3, 9), (6, 9), (8, 9)]
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    elif kind == "论文引用":
        coords = np.array(
            [
                [-1.7, 0.95],
                [-0.95, 0.35],
                [-1.15, -0.75],
                [-0.15, 1.15],
                [0.1, -0.15],
                [0.95, 0.75],
                [1.55, -0.05],
                [0.85, -0.95],
                [1.85, 0.95],
                [-0.05, -1.25],
            ]
        )
        edges = [(0, 1), (1, 2), (0, 3), (3, 4), (1, 4), (4, 5), (5, 6), (6, 7), (5, 8), (4, 9), (2, 9), (7, 9), (3, 5)]
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 1, 2])
    else:
        theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        coords = np.c_[np.cos(theta), np.sin(theta)]
        coords[:4] += np.array([-0.65, 0.2])
        coords[4:8] += np.array([0.55, 0.15])
        coords[8:] += np.array([0.0, -0.75])
        edges = [(i, (i + 1) % 12) for i in range(12)]
        edges += [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (2, 9), (6, 10), (3, 4)]
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    names = [f"v{i}" for i in range(len(coords))]
    features = make_features(coords, labels)
    return coords, edges, labels, features, names


def make_features(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_hint = np.zeros((len(labels), 3))
    label_hint[np.arange(len(labels)), labels] = 1.0
    feature = np.c_[
        coords[:, 0],
        coords[:, 1],
        np.linalg.norm(coords, axis=1),
        label_hint[:, 0] * 0.7 + 0.15,
        label_hint[:, 1] * 0.7 + 0.15,
        label_hint[:, 2] * 0.7 + 0.15,
    ]
    return feature.astype(float)


def adjacency_matrix(n: int, edges: list[tuple[int, int]], self_loop: bool = False) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=float)
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    if self_loop:
        adjacency += np.eye(n)
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(1 / np.sqrt(degree + 1e-9))
    return inv_sqrt @ adjacency @ inv_sqrt


def draw_graph(
    ax: plt.Axes,
    coords: np.ndarray,
    edges: list[tuple[int, int]],
    node_values: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    names: list[str] | None = None,
    highlighted: int | None = None,
    edge_weights: dict[tuple[int, int], float] | None = None,
    title: str = "",
) -> None:
    segments = []
    widths = []
    colors = []
    for i, j in edges:
        segments.append([coords[i], coords[j]])
        key = (min(i, j), max(i, j))
        weight = 0.35 if edge_weights is None else edge_weights.get(key, 0.18)
        widths.append(1.2 + 5.0 * weight)
        colors.append(PALETTE["amber"] if edge_weights is not None and weight > 0.2 else "#9aa7b1")
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, alpha=0.75, zorder=1))

    if node_values is not None:
        colors_node = [CLASS_COLORS[int(v) % len(CLASS_COLORS)] for v in node_values]
    elif labels is not None:
        colors_node = [CLASS_COLORS[int(v) % len(CLASS_COLORS)] for v in labels]
    else:
        colors_node = PALETTE["blue"]

    edge_color = ["#111827" if i == highlighted else "white" for i in range(len(coords))]
    linewidth = [3.0 if i == highlighted else 1.6 for i in range(len(coords))]
    ax.scatter(coords[:, 0], coords[:, 1], s=650, c=colors_node, edgecolors=edge_color, linewidths=linewidth, zorder=3, alpha=0.95)
    for i, (x, y) in enumerate(coords):
        text = names[i] if names is not None else str(i)
        ax.text(x, y, text, ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=4)
    ax.set_title(title, fontweight="bold", color=PALETTE["ink"])
    ax.set_aspect("equal")
    margin = 0.45
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
    ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
    ax.axis("off")


def plot_graph_representation(kind: str, self_loop: bool, weighted: bool) -> plt.Figure:
    coords, edges, labels, features, names = build_graph(kind)
    n = len(coords)
    adjacency = adjacency_matrix(n, edges, self_loop=self_loop)
    if weighted:
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] and i != j:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    adjacency[i, j] = 1 / (1 + distance)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0, 1.1]})
    draw_graph(axes[0], coords, edges, labels=labels, names=names, title="图 G=(V,E)")
    im = axes[1].imshow(adjacency, cmap="Blues", vmin=0, vmax=max(1, float(adjacency.max())))
    axes[1].set_title("邻接矩阵 A", fontweight="bold")
    axes[1].set_xlabel("目标节点 j")
    axes[1].set_ylabel("源节点 i")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(names, rotation=90)
    axes[1].set_yticklabels(names)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(features, cmap="RdBu", aspect="auto")
    axes[2].set_title("节点特征矩阵 X", fontweight="bold")
    axes[2].set_xlabel("特征维度")
    axes[2].set_ylabel("节点")
    axes[2].set_yticks(range(n))
    axes[2].set_yticklabels(names)
    axes[2].set_xticks(range(features.shape[1]))
    axes[2].set_xticklabels(["x", "y", "r", "c0", "c1", "c2"])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def add_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.8,
        edgecolor="white",
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color="white", fontweight="bold", fontsize=10)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#52616b") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            shrinkA=5,
            shrinkB=5,
        )
    )


def gcn_forward(features: np.ndarray, adjacency: np.ndarray, hidden_dim: int, seed: int, use_relu: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a_hat = normalize_adjacency(adjacency + np.eye(adjacency.shape[0]))
    weight = rng.normal(0, 1 / math.sqrt(features.shape[1]), (features.shape[1], hidden_dim))
    hidden = a_hat @ features @ weight
    return np.maximum(hidden, 0) if use_relu else hidden


def plot_gcn_demo(kind: str, hidden_dim: int, seed: int, selected_node: int) -> plt.Figure:
    coords, edges, labels, features, names = build_graph(kind)
    adjacency = adjacency_matrix(len(coords), edges)
    a_hat = normalize_adjacency(adjacency + np.eye(len(coords)))
    hidden = gcn_forward(features, adjacency, hidden_dim, seed)
    neighbor_mix = a_hat @ features
    score = hidden[:, 0] if hidden.shape[1] else hidden.mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    draw_graph(axes[0], coords, edges, labels=labels, names=names, highlighted=selected_node, title="一跳邻居聚合")
    for j in np.where(a_hat[selected_node] > 0)[0]:
        if j == selected_node:
            continue
        axes[0].plot([coords[j, 0], coords[selected_node, 0]], [coords[j, 1], coords[selected_node, 1]], color=PALETTE["amber"], linewidth=4, alpha=0.45)
    im = axes[1].imshow(neighbor_mix, cmap="RdBu", aspect="auto")
    axes[1].set_title("A_hat X: 结构平滑后的特征", fontweight="bold")
    axes[1].set_xlabel("特征维度")
    axes[1].set_ylabel("节点")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(hidden, cmap="viridis", aspect="auto")
    axes[2].set_title("H = ReLU(A_hat X W)", fontweight="bold")
    axes[2].set_xlabel("隐藏维度")
    axes[2].set_ylabel("节点")
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels(names)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"选中节点 {names[selected_node]} 的一跳聚合；隐藏维度0范围 {score.min():.2f} 到 {score.max():.2f}", fontsize=12, color=PALETTE["muted"])
    fig.tight_layout()
    return fig


def gat_attention(features: np.ndarray, adjacency: np.ndarray, selected_node: int, sharpness: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1 / math.sqrt(features.shape[1]), (features.shape[1], 4))
    projected = features @ projection
    scores = np.full(adjacency.shape[0], -1e9)
    neighbors = np.where(adjacency[selected_node] > 0)[0]
    neighbors = np.r_[neighbors, selected_node]
    for j in neighbors:
        raw = np.dot(projected[selected_node], projected[j])
        raw += 0.65 / (1 + np.linalg.norm(features[selected_node] - features[j]))
        scores[j] = sharpness * raw
    valid = scores > -1e8
    shifted = scores[valid] - scores[valid].max()
    weights = np.zeros_like(scores)
    weights[valid] = np.exp(shifted) / np.exp(shifted).sum()
    return weights


def plot_gat_demo(kind: str, selected_node: int, sharpness: float, seed: int) -> plt.Figure:
    coords, edges, labels, features, names = build_graph(kind)
    adjacency = adjacency_matrix(len(coords), edges)
    selected_node = min(selected_node, len(coords) - 1)
    weights = gat_attention(features, adjacency, selected_node, sharpness, seed)
    edge_weights = {}
    for j, value in enumerate(weights):
        if j == selected_node or value <= 0:
            continue
        edge_weights[(min(selected_node, j), max(selected_node, j))] = float(value)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.12, 0.92, 1]})
    draw_graph(
        axes[0],
        coords,
        edges,
        labels=labels,
        names=names,
        highlighted=selected_node,
        edge_weights=edge_weights,
        title=f"GAT 对 {names[selected_node]} 的邻居注意力",
    )
    axes[1].bar(np.arange(len(weights)), weights, color=[PALETTE["rose"] if i == selected_node else PALETTE["blue"] for i in range(len(weights))])
    axes[1].set_title("注意力系数 alpha_ij", fontweight="bold")
    axes[1].set_xlabel("邻居节点 j")
    axes[1].set_ylabel("权重")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=90)
    axes[1].grid(True, axis="y", alpha=0.25)

    attention_matrix = np.full((len(names), len(names)), np.nan)
    for i in range(len(names)):
        attention_matrix[i] = gat_attention(features, adjacency, i, sharpness, seed)
    im = axes[2].imshow(attention_matrix, cmap="YlOrRd", vmin=0, vmax=max(0.35, float(np.nanmax(attention_matrix))))
    axes[2].set_title("所有节点的一头注意力矩阵", fontweight="bold")
    axes[2].set_xlabel("被关注节点 j")
    axes[2].set_ylabel("查询节点 i")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_yticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=90)
    axes[2].set_yticklabels(names)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def label_propagation(labels: np.ndarray, adjacency: np.ndarray, labeled_nodes: list[int], steps: int, alpha: float) -> np.ndarray:
    num_classes = int(labels.max()) + 1
    y = np.ones((len(labels), num_classes)) / num_classes
    for idx in labeled_nodes:
        y[idx] = 0
        y[idx, labels[idx]] = 1

    transition = adjacency + np.eye(len(labels))
    transition = transition / (transition.sum(axis=1, keepdims=True) + 1e-9)
    seed_y = y.copy()
    mask = np.zeros(len(labels), dtype=bool)
    mask[labeled_nodes] = True
    for _ in range(steps):
        y = alpha * (transition @ y) + (1 - alpha) * seed_y
        y[mask] = seed_y[mask]
    return y


def plot_node_classification(kind: str, labeled_per_class: int, steps: int, alpha: float) -> plt.Figure:
    coords, edges, labels, _features, names = build_graph(kind)
    adjacency = adjacency_matrix(len(coords), edges)
    labeled_nodes = []
    for cls in range(int(labels.max()) + 1):
        labeled_nodes.extend(np.where(labels == cls)[0][:labeled_per_class].tolist())
    probabilities = label_propagation(labels, adjacency, labeled_nodes, steps, alpha)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accuracy = float((prediction == labels).mean())

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.05, 1.05, 1]})
    draw_graph(axes[0], coords, edges, node_values=prediction, names=names, title=f"节点分类预测 准确率 {accuracy:.0%}")
    for idx in labeled_nodes:
        axes[0].scatter([coords[idx, 0]], [coords[idx, 1]], s=900, facecolors="none", edgecolors="#111827", linewidths=2.8, zorder=5)
        axes[0].text(coords[idx, 0], coords[idx, 1] - 0.28, "L", ha="center", va="center", color="#111827", fontsize=9, fontweight="bold", zorder=6)

    draw_graph(axes[1], coords, edges, node_values=labels, names=names, title="真实类别")
    im = axes[2].imshow(probabilities, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("每个节点的类别概率", fontweight="bold")
    axes[2].set_xlabel("类别")
    axes[2].set_ylabel("节点")
    axes[2].set_xticks(range(probabilities.shape[1]))
    axes[2].set_xticklabels([f"class {i}" for i in range(probabilities.shape[1])])
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels([f"{name}  conf={confidence[i]:.2f}" for i, name in enumerate(names)])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_representation_tab(kind: str) -> None:
    st.subheader("1. 图数据的表示方法")
    col1, col2 = st.columns([0.23, 0.77])
    with col1:
        self_loop = st.checkbox("显示自环", value=False)
        weighted = st.checkbox("按距离生成边权", value=False)
        st.markdown(
            """
            <div class="formula">
            V: 节点集合<br>
            E: 边集合<br>
            A: 邻接矩阵<br>
            X: 节点特征矩阵
            </div>
            """,
            unsafe_allow_html=True,
        )
        coords, edges, labels, features, _names = build_graph(kind)
        st.metric("节点数 |V|", len(coords))
        st.metric("边数 |E|", len(edges))
        st.metric("特征维度", features.shape[1])
    with col2:
        st.pyplot(plot_graph_representation(kind, self_loop, weighted), clear_figure=True)


def render_gcn_tab(kind: str, seed: int) -> None:
    st.subheader("2. GCN 的基本原理")
    st.markdown(
        """
        <div class="formula">
        A_hat = D^(-1/2) (A + I) D^(-1/2)<br>
        H^(l+1) = sigma( A_hat H^(l) W^(l) )
        </div>
        """,
        unsafe_allow_html=True,
    )
    coords, _edges, _labels, _features, names = build_graph(kind)
    col1, col2 = st.columns([0.23, 0.77])
    with col1:
        hidden_dim = st.select_slider("隐藏维度", options=[2, 4, 8, 16], value=4)
        selected_node = st.selectbox("观察节点", list(range(len(coords))), format_func=lambda i: names[i], index=min(4, len(coords) - 1))
        st.markdown(
            '<div class="note">GCN 的一层会把当前节点和一跳邻居混合。堆叠两层后，节点就能间接看到二跳邻居。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(plot_gcn_demo(kind, hidden_dim, seed, selected_node), clear_figure=True)


def render_gat_tab(kind: str, seed: int) -> None:
    st.subheader("3. GAT 的基本原理")
    st.markdown(
        """
        <div class="formula">
        e_ij = LeakyReLU( a^T [W h_i || W h_j] )<br>
        alpha_ij = softmax_j(e_ij)<br>
        h_i' = sigma( sum_j alpha_ij W h_j )
        </div>
        """,
        unsafe_allow_html=True,
    )
    coords, _edges, _labels, _features, names = build_graph(kind)
    col1, col2 = st.columns([0.23, 0.77])
    with col1:
        selected_node = st.selectbox("查询节点", list(range(len(coords))), format_func=lambda i: names[i], index=min(4, len(coords) - 1), key="gat_node")
        sharpness = st.slider("注意力锐度", 0.3, 4.0, 1.4, 0.1)
        st.markdown(
            '<div class="note">锐度越高，softmax 越容易把权重集中到少数邻居；锐度越低，GAT 越接近平均聚合。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(plot_gat_demo(kind, selected_node, sharpness, seed), clear_figure=True)


def render_classification_tab(kind: str) -> None:
    st.subheader("4. 节点分类可视化 demo")
    col1, col2 = st.columns([0.23, 0.77])
    with col1:
        labeled_per_class = st.select_slider("每类已标注节点数", options=[1, 2], value=1)
        steps = st.slider("传播步数", 0, 20, 6)
        alpha = st.slider("邻居传播强度", 0.0, 1.0, 0.82, 0.02)
        st.markdown(
            '<div class="note">这个 demo 用标签传播模拟节点分类：少数已标注节点提供监督信号，类别概率沿边扩散。真实 GNN 会同时学习特征变换和分类器。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(plot_node_classification(kind, labeled_per_class, steps, alpha), clear_figure=True)


def main() -> None:
    try:
        render_hero()
        render_concept_cards()

        st.sidebar.header("交互参数")
        kind = st.sidebar.selectbox("图类型", ["社交网络", "论文引用", "知识图谱"])
        seed = st.sidebar.slider("随机种子", 0, 99, 13)

        tabs = st.tabs(["图表示", "GCN", "GAT", "节点分类"])
        with tabs[0]:
            render_representation_tab(kind)
        with tabs[1]:
            render_gcn_tab(kind, seed)
        with tabs[2]:
            render_gat_tab(kind, seed)
        with tabs[3]:
            render_classification_tab(kind)
    except Exception as exc:
        from components.error_boundary import render_module_error
        render_module_error("part4_transformer/gnn_intro.py", exc)


if __name__ == "__main__":
    main()
