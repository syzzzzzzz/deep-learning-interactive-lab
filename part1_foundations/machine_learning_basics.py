"""
机器学习基础可视化教学模块。

运行:
    streamlit run part1_foundations/machine_learning_basics.py
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


st.set_page_config(
    page_title="机器学习基础",
    layout="wide",
    initial_sidebar_state="expanded",
)


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}

INK = "#172026"
MUTED = "#58646d"
TEAL = "#0f8b8d"
ROSE = "#c73e5b"
AMBER = "#d99a22"
GREEN = "#477b44"
VIOLET = "#5e4ae3"
GRAY = "#9aa7ad"


st.markdown(
    """
    <style>
    :root {
        --ink: #172026;
        --muted: #58646d;
        --line: #d7dde1;
        --paper: #fbfaf6;
        --teal: #0f8b8d;
        --rose: #c73e5b;
        --amber: #d99a22;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(239,245,242,0.96) 100%),
            linear-gradient(90deg, rgba(15,139,141,0.06) 0%, transparent 32%, rgba(199,62,91,0.05) 100%),
            var(--paper);
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.4rem;
    }
    h1, h2, h3 { letter-spacing: 0; }
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
    .hero {
        border-bottom: 1px solid var(--line);
        padding: 0.1rem 0 1rem 0;
        margin-bottom: 0.9rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3.1rem);
        line-height: 1.05;
        margin: 0;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.75;
        max-width: 1000px;
        margin: 0.5rem 0 0 0;
    }
    .scene-title {
        margin: 0.35rem 0 0.3rem 0;
        font-size: 1.35rem;
        font-weight: 750;
        color: var(--ink);
    }
    .scene-lead {
        color: var(--muted);
        max-width: 1040px;
        margin: 0 0 0.75rem 0;
        line-height: 1.75;
        font-size: 1rem;
    }
    .lesson-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
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
        background: rgba(255,255,255,0.70);
        padding: 0.72rem 0.9rem;
        margin: 0.35rem 0 0.9rem 0;
        border-radius: 0 8px 8px 0;
        color: #26343b;
        line-height: 1.68;
    }
    .insight strong { color: #102027; }
    .mini-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.2rem 0 0.8rem 0;
        font-size: 0.92rem;
    }
    .mini-table td {
        border-bottom: 1px solid rgba(215,221,225,0.9);
        padding: 0.45rem 0.35rem;
        color: var(--muted);
        vertical-align: top;
    }
    .mini-table td:first-child {
        color: var(--ink);
        font-weight: 650;
        width: 34%;
    }
    @media (max-width: 1050px) {
        .lesson-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
        .lesson-grid { grid-template-columns: 1fr; }
        .lesson-card { min-height: auto; }
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


def apply_plot_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 36, "r": 24, "t": 56 if title else 32, "b": 42},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.74)",
        font=PLOT_FONT,
        hoverlabel={"bgcolor": INK, "font_color": "#ffffff"},
    )
    return fig


def render_scene_intro(title: str, lead: str, cards: list[tuple[str, str]]) -> None:
    card_html = "".join(
        '<div class="lesson-card">'
        f"<strong>{html.escape(card_title)}</strong>"
        f"<p>{html.escape(body)}</p>"
        "</div>"
        for card_title, body in cards
    )
    st.markdown(
        f"""
        <div class="scene-title">{html.escape(title)}</div>
        <p class="scene-lead">{html.escape(lead)}</p>
        <div class="lesson-grid">{card_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
            <strong>{html.escape(title)}</strong> {html.escape(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(rows: list[tuple[str, str]]) -> None:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{html.escape(text)}</td>"
        "</tr>"
        for label, text in rows
    )
    st.markdown(f'<table class="mini-table">{body}</table>', unsafe_allow_html=True)


def render_matplotlib(fig: plt.Figure) -> None:
    try:
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def normalize_xy(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def scatter_trace(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    color: str,
    symbol: str = "circle",
    size: int = 8,
) -> go.Scattergl:
    return go.Scattergl(
        x=x[:, 0],
        y=x[:, 1],
        mode="markers",
        name=name,
        marker={
            "size": size,
            "color": color,
            "symbol": symbol,
            "line": {"color": "#ffffff", "width": 1},
        },
        hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<extra>" + name + "</extra>",
    )


@st.cache_data(show_spinner=False)
def learning_type_data(seed: int, label_fraction: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_cls, y_cls = make_moons(n_samples=260, noise=0.18, random_state=seed)
    x_cls = normalize_xy(x_cls)
    labeled = rng.random(len(y_cls)) < label_fraction

    x_cluster, _ = make_blobs(
        n_samples=280,
        centers=[[-1.8, -0.6], [0.0, 1.25], [1.75, -0.45]],
        cluster_std=[0.46, 0.52, 0.42],
        random_state=seed + 11,
    )
    x_cluster = normalize_xy(x_cluster)
    cluster_y = KMeans(n_clusters=3, random_state=seed, n_init=10).fit_predict(x_cluster)

    values = np.zeros((5, 5), dtype=float)
    goal = np.array([4, 4])
    trap = np.array([1, 3])
    for row in range(5):
        for col in range(5):
            pos = np.array([row, col])
            values[row, col] = 1.0 - 0.17 * np.abs(pos - goal).sum()
            values[row, col] -= 0.45 * np.exp(-np.square(pos - trap).sum() / 1.3)

    policy = np.zeros((5, 5, 2), dtype=float)
    for row in range(5):
        for col in range(5):
            vec = goal - np.array([row, col])
            norm = np.linalg.norm(vec) + 1e-8
            policy[row, col] = np.array([vec[1], -vec[0]]) / norm
    policy[4, 4] = 0

    return {
        "x_cls": x_cls,
        "y_cls": y_cls,
        "labeled": labeled,
        "x_cluster": x_cluster,
        "cluster_y": cluster_y,
        "values": values,
        "policy": policy,
    }


def plot_learning_types(data: dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "监督学习: 样本有答案",
            "无监督学习: 只看结构",
            "半监督学习: 少量标签带大量未标注点",
            "强化学习: 状态、动作、奖励",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    x_cls = data["x_cls"]
    y_cls = data["y_cls"]
    for cls, color in [(0, TEAL), (1, ROSE)]:
        mask = y_cls == cls
        fig.add_trace(scatter_trace(x_cls[mask], y_cls[mask], f"类别 {cls}", color), row=1, col=1)

    x_cluster = data["x_cluster"]
    cluster_y = data["cluster_y"]
    for cls, color in [(0, TEAL), (1, AMBER), (2, VIOLET)]:
        mask = cluster_y == cls
        fig.add_trace(scatter_trace(x_cluster[mask], cluster_y[mask], f"簇 {cls + 1}", color), row=1, col=2)

    labeled = data["labeled"]
    fig.add_trace(scatter_trace(x_cls[~labeled], y_cls[~labeled], "未标注", "#b8c2c8", "circle-open", 7), row=2, col=1)
    for cls, color in [(0, TEAL), (1, ROSE)]:
        mask = labeled & (y_cls == cls)
        fig.add_trace(scatter_trace(x_cls[mask], y_cls[mask], f"少量标签 {cls}", color, "diamond", 10), row=2, col=1)

    values = data["values"]
    policy = data["policy"]
    fig.add_trace(
        go.Heatmap(
            z=values,
            colorscale=[[0, "#f4e6df"], [0.5, "#fff9ec"], [1, "#1c7c78"]],
            showscale=False,
            hovertemplate="状态(%{x},%{y})<br>价值=%{z:.2f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    yy, xx = np.mgrid[0:5, 0:5]
    labels = [
        "目标" if (r == 4 and c == 4) else "陷阱" if (r == 1 and c == 3) else ""
        for r, c in zip(yy.ravel(), xx.ravel())
    ]
    fig.add_trace(
        go.Scatter(
            x=xx.ravel(),
            y=yy.ravel(),
            mode="markers+text",
            marker={"size": 26, "color": "rgba(255,255,255,0.68)", "line": {"color": "#d7dde1", "width": 1}},
            text=labels,
            textfont={"size": 11, "color": INK},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    for r in range(5):
        for c in range(5):
            dx, dy = policy[r, c] * 0.32
            if abs(dx) + abs(dy) > 0:
                fig.add_annotation(
                    x=c + dx,
                    y=r + dy,
                    ax=c - dx,
                    ay=r - dy,
                    xref="x4",
                    yref="y4",
                    axref="x4",
                    ayref="y4",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.4,
                    arrowcolor=INK,
                )

    for idx in range(1, 5):
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1
        fig.update_xaxes(showgrid=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, row=row, col=col)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_layout(legend={"orientation": "h", "y": -0.08})
    return apply_plot_layout(fig, 680, "四类学习范式看的不是同一种反馈信号")


@st.cache_data(show_spinner=False)
def task_comparison_data(seed: int, cluster_count: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    x_reg = np.linspace(-3, 3, 95)
    y_reg = 0.35 * x_reg**3 - 0.9 * x_reg + rng.normal(0, 1.0, size=len(x_reg))
    coef = np.polyfit(x_reg, y_reg, deg=3)
    x_line = np.linspace(-3.2, 3.2, 220)
    y_line = np.polyval(coef, x_line)

    x_cls, y_cls = make_classification(
        n_samples=240,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.25,
        random_state=seed + 9,
    )
    x_cls = StandardScaler().fit_transform(x_cls)
    clf = LogisticRegression(random_state=seed).fit(x_cls, y_cls)
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    probs = clf.predict_proba(np.c_[grid_x.ravel(), grid_y.ravel()])[:, 1].reshape(grid_x.shape)

    x_cluster, _ = make_blobs(n_samples=260, centers=cluster_count, cluster_std=0.62, random_state=seed + 19)
    x_cluster = StandardScaler().fit_transform(x_cluster)
    km = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit(x_cluster)

    return {
        "x_reg": x_reg,
        "y_reg": y_reg,
        "x_line": x_line,
        "y_line": y_line,
        "x_cls": x_cls,
        "y_cls": y_cls,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "probs": probs,
        "x_cluster": x_cluster,
        "cluster_labels": km.labels_,
        "centers": km.cluster_centers_,
    }


def plot_task_comparison(data: dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["回归: 预测连续数值", "分类: 预测离散类别", "聚类: 发现相似分组"],
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=data["x_reg"],
            y=data["y_reg"],
            mode="markers",
            name="训练样本",
            marker={"color": TEAL, "size": 8, "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra>样本</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["x_line"],
            y=data["y_line"],
            mode="lines",
            name="拟合曲线",
            line={"color": ROSE, "width": 3},
            hovertemplate="x=%{x:.2f}<br>预测=%{y:.2f}<extra>回归</extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Contour(
            x=data["grid_x"][0],
            y=data["grid_y"][:, 0],
            z=data["probs"],
            contours={"start": 0, "end": 1, "size": 0.05, "coloring": "heatmap"},
            colorscale=[[0, "#12343b"], [0.5, "#fff7e6"], [1, ROSE]],
            showscale=False,
            hovertemplate="P(类别1)=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    for cls, color in [(0, TEAL), (1, ROSE)]:
        mask = data["y_cls"] == cls
        fig.add_trace(scatter_trace(data["x_cls"][mask], data["y_cls"][mask], f"类别 {cls}", color), row=1, col=2)

    colors = [TEAL, AMBER, ROSE, VIOLET]
    for cls in sorted(np.unique(data["cluster_labels"])):
        mask = data["cluster_labels"] == cls
        fig.add_trace(
            scatter_trace(data["x_cluster"][mask], data["cluster_labels"][mask], f"簇 {cls + 1}", colors[int(cls) % len(colors)]),
            row=1,
            col=3,
        )
    fig.add_trace(
        go.Scatter(
            x=data["centers"][:, 0],
            y=data["centers"][:, 1],
            mode="markers",
            name="簇中心",
            marker={"symbol": "x", "size": 16, "color": INK, "line": {"width": 3}},
            hovertemplate="中心 x1=%{x:.2f}<br>x2=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    for col in range(1, 4):
        fig.update_xaxes(gridcolor="rgba(88,100,109,0.16)", zeroline=False, row=1, col=col)
        fig.update_yaxes(gridcolor="rgba(88,100,109,0.16)", zeroline=False, row=1, col=col)
    fig.update_layout(legend={"orientation": "h", "y": -0.22})
    return apply_plot_layout(fig, 520, "同样是学习，输出目标完全不同")


@dataclass(frozen=True)
class FitConfig:
    degree: int
    noise: float
    train_size: int
    seed: int


@st.cache_data(show_spinner=False)
def overfit_data(config: FitConfig) -> dict[str, np.ndarray | float | int | str]:
    rng = np.random.default_rng(config.seed)
    x_all = rng.uniform(-3, 3, size=360)
    y_clean = np.sin(1.45 * x_all) + 0.28 * np.cos(3.1 * x_all)
    y_all = y_clean + rng.normal(0, config.noise, size=len(x_all))

    x_train, x_val, y_train, y_val = train_test_split(
        x_all.reshape(-1, 1),
        y_all,
        train_size=config.train_size,
        random_state=config.seed,
    )

    degrees = np.arange(1, 19)
    train_errors: list[float] = []
    val_errors: list[float] = []
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(int(degree), include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-2),
        )
        model.fit(x_train, y_train)
        train_errors.append(float(np.mean((model.predict(x_train) - y_train) ** 2)))
        val_errors.append(float(np.mean((model.predict(x_val) - y_val) ** 2)))

    selected = make_pipeline(
        PolynomialFeatures(config.degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1e-2),
    )
    selected.fit(x_train, y_train)
    x_line = np.linspace(-3.2, 3.2, 320).reshape(-1, 1)
    y_true_line = np.sin(1.45 * x_line[:, 0]) + 0.28 * np.cos(3.1 * x_line[:, 0])
    y_pred_line = selected.predict(x_line)

    train_errors_np = np.array(train_errors)
    val_errors_np = np.array(val_errors)
    selected_train = float(train_errors_np[config.degree - 1])
    selected_val = float(val_errors_np[config.degree - 1])
    best_degree = int(degrees[np.argmin(val_errors_np)])

    if selected_train > 0.42 and selected_val > 0.42:
        status = "欠拟合"
    elif selected_val > max(selected_train * 2.2, val_errors_np.min() * 1.35) and config.degree > best_degree:
        status = "过拟合"
    else:
        status = "相对合适"

    return {
        "x_train": x_train[:, 0],
        "y_train": y_train,
        "x_val": x_val[:, 0],
        "y_val": y_val,
        "x_line": x_line[:, 0],
        "y_true_line": y_true_line,
        "y_pred_line": y_pred_line,
        "degrees": degrees,
        "train_errors": train_errors_np,
        "val_errors": val_errors_np,
        "selected_degree": config.degree,
        "best_degree": best_degree,
        "final_train": selected_train,
        "final_val": selected_val,
        "status": status,
    }


def plot_overfit(result: dict[str, np.ndarray | float | int | str]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["当前复杂度下的拟合曲线", "复杂度曲线: 训练误差 vs 验证误差"],
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Scatter(
            x=result["x_train"],
            y=result["y_train"],
            mode="markers",
            name="训练数据",
            marker={"color": TEAL, "size": 8, "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra>训练</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result["x_val"],
            y=result["y_val"],
            mode="markers",
            name="验证数据",
            marker={"color": GRAY, "size": 7, "symbol": "circle-open"},
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra>验证</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result["x_line"],
            y=result["y_true_line"],
            mode="lines",
            name="真实规律",
            line={"color": "#7b8790", "width": 2, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result["x_line"],
            y=result["y_pred_line"],
            mode="lines",
            name="模型预测",
            line={"color": ROSE, "width": 3},
            hovertemplate="x=%{x:.2f}<br>预测=%{y:.2f}<extra>模型</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result["degrees"],
            y=result["train_errors"],
            mode="lines+markers",
            name="训练误差",
            line={"color": TEAL, "width": 3},
            hovertemplate="复杂度=%{x}<br>MSE=%{y:.4f}<extra>训练</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=result["degrees"],
            y=result["val_errors"],
            mode="lines+markers",
            name="验证误差",
            line={"color": ROSE, "width": 3},
            hovertemplate="复杂度=%{x}<br>MSE=%{y:.4f}<extra>验证</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=int(result["selected_degree"]),
        line_dash="dot",
        line_color=AMBER,
        annotation_text="当前",
        row=1,
        col=2,
    )
    fig.add_vline(
        x=int(result["best_degree"]),
        line_dash="dash",
        line_color=GREEN,
        annotation_text="验证最优",
        row=1,
        col=2,
    )
    fig.update_xaxes(title="x", gridcolor="rgba(88,100,109,0.16)", zeroline=False, row=1, col=1)
    fig.update_yaxes(title="y", gridcolor="rgba(88,100,109,0.16)", zeroline=False, row=1, col=1)
    fig.update_xaxes(title="多项式次数", dtick=2, row=1, col=2)
    fig.update_yaxes(title="均方误差 MSE", row=1, col=2)
    fig.update_layout(legend={"orientation": "h", "y": -0.18})
    return apply_plot_layout(fig, 520, "泛化能力要同时看训练集和验证集")


@dataclass(frozen=True)
class EvalConfig:
    threshold: float
    class_sep: float
    positive_rate: float
    seed: int


@st.cache_data(show_spinner=False)
def evaluation_data(config: EvalConfig) -> dict[str, np.ndarray | float]:
    x, y = make_classification(
        n_samples=720,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[1 - config.positive_rate, config.positive_rate],
        class_sep=config.class_sep,
        flip_y=0.05,
        random_state=config.seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.38,
        stratify=y,
        random_state=config.seed,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.seed))
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= config.threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_test, prob)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, prob)

    return {
        "cm": confusion_matrix(y_test, pred),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_thresholds": pr_thresholds,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "acc": accuracy_score(y_test, pred),
        "auc": auc(fpr, tpr),
    }


def plot_evaluation(result: dict[str, np.ndarray | float], threshold: float) -> go.Figure:
    cm = result["cm"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["混淆矩阵: 错在什么地方", "ROC 曲线: 扫过所有阈值"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["预测 0", "预测 1"],
            y=["真实 0", "真实 1"],
            colorscale=[[0, "#f6f1e8"], [1, TEAL]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22, "color": INK},
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>数量=%{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result["fpr"],
            y=result["tpr"],
            mode="lines",
            name=f"ROC AUC={float(result['auc']):.3f}",
            line={"color": ROSE, "width": 4},
            hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="随机猜测",
            line={"color": GRAY, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    thresholds = result["roc_thresholds"]
    finite_thresholds = np.where(np.isfinite(thresholds), thresholds, 1.01)
    closest = int(np.argmin(np.abs(finite_thresholds - threshold)))
    fig.add_trace(
        go.Scatter(
            x=[result["fpr"][closest]],
            y=[result["tpr"][closest]],
            mode="markers",
            name=f"当前阈值 {threshold:.2f}",
            marker={"color": AMBER, "size": 13, "line": {"color": "#ffffff", "width": 2}},
            hovertemplate="当前阈值<br>FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title="假阳性率 FPR", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="真阳性率 TPR", range=[0, 1], row=1, col=2)
    fig.update_layout(legend={"orientation": "h", "y": -0.2})
    return apply_plot_layout(fig, 470, "分类器评估不能只看准确率")


def plot_metric_bars(result: dict[str, np.ndarray | float]) -> go.Figure:
    names = ["Precision", "Recall", "F1", "Accuracy"]
    values = [float(result["precision"]), float(result["recall"]), float(result["f1"]), float(result["acc"])]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[TEAL, ROSE, AMBER, GREEN],
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            hovertemplate="%{x}=%{y:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1.05], title="分数")
    return apply_plot_layout(fig, 315, "精确率、召回率、F1 的权衡")


def plot_threshold_tradeoff(result: dict[str, np.ndarray | float]) -> plt.Figure:
    thresholds = np.asarray(result["pr_thresholds"])
    precision = np.asarray(result["precision_curve"])[:-1]
    recall = np.asarray(result["recall_curve"])[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(thresholds, precision, color=TEAL, linewidth=2.2, label="Precision")
    ax.plot(thresholds, recall, color=ROSE, linewidth=2.2, label="Recall")
    ax.plot(thresholds, f1, color=AMBER, linewidth=2.2, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class CVConfig:
    folds: int
    shuffle: bool
    seed: int


@st.cache_data(show_spinner=False)
def cross_validation_data(config: CVConfig) -> dict[str, np.ndarray | list[float]]:
    x, y = make_classification(
        n_samples=180,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.1,
        flip_y=0.08,
        random_state=config.seed,
    )
    order = np.argsort(y)
    x = x[order]
    y = y[order]

    splitter = StratifiedKFold(
        n_splits=config.folds,
        shuffle=config.shuffle,
        random_state=config.seed if config.shuffle else None,
    )
    split_matrix = np.zeros((config.folds, len(y)), dtype=int)
    scores: list[float] = []
    fold_sizes: list[int] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
        split_matrix[fold, train_idx] = 1
        split_matrix[fold, val_idx] = 2
        fold_sizes.append(len(val_idx))

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.seed))
        model.fit(x[train_idx], y[train_idx])
        scores.append(float(model.score(x[val_idx], y[val_idx])))

    holdout_scores: list[float] = []
    rng = np.random.default_rng(config.seed)
    for index in range(18):
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=1 / config.folds,
            stratify=y,
            random_state=int(rng.integers(0, 100000)),
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.seed + index))
        model.fit(x[train_idx], y[train_idx])
        holdout_scores.append(float(model.score(x[val_idx], y[val_idx])))

    return {
        "split_matrix": split_matrix,
        "scores": scores,
        "fold_sizes": np.array(fold_sizes),
        "holdout_scores": holdout_scores,
        "y": y,
    }


def plot_cross_validation(result: dict[str, np.ndarray | list[float]]) -> go.Figure:
    matrix = result["split_matrix"]
    scores = result["scores"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["K 折交叉验证如何轮换验证集", "每一折的验证分数"],
        column_widths=[0.64, 0.36],
        horizontal_spacing=0.11,
    )
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            colorscale=[
                [0.0, "#f4f1ea"],
                [0.33, "#f4f1ea"],
                [0.34, TEAL],
                [0.66, TEAL],
                [0.67, ROSE],
                [1.0, ROSE],
            ],
            showscale=False,
            hovertemplate="折=%{y}<br>样本=%{x}<br>角色=%{z}<extra>1训练 2验证</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[f"Fold {i + 1}" for i in range(len(scores))],
            y=scores,
            marker_color=[TEAL if i % 2 == 0 else AMBER for i in range(len(scores))],
            text=[f"{s:.1%}" for s in scores],
            textposition="outside",
            hovertemplate="%{x}<br>验证分数=%{y:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=float(np.mean(scores)),
        line_dash="dash",
        line_color=ROSE,
        annotation_text=f"平均 {np.mean(scores):.1%}",
        row=1,
        col=2,
    )
    fig.update_xaxes(title="样本索引", row=1, col=1)
    fig.update_yaxes(title="折编号", row=1, col=1)
    fig.update_yaxes(title="Accuracy", range=[0, 1.05], row=1, col=2)
    return apply_plot_layout(fig, 510, "同一份数据被轮流拿来验证")


def plot_cv_distribution(result: dict[str, np.ndarray | list[float]]) -> go.Figure:
    cv_scores = result["scores"]
    holdout_scores = result["holdout_scores"]
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=holdout_scores,
            name="多次随机留出",
            marker_color=GRAY,
            boxmean=True,
            hovertemplate="分数=%{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Box(
            y=cv_scores,
            name="K 折结果",
            marker_color=TEAL,
            boxmean=True,
            hovertemplate="分数=%{y:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1.05], title="验证分数")
    return apply_plot_layout(fig, 330, "交叉验证让评估不那么依赖一次随机划分")


st.markdown(
    """
    <div class="hero">
      <h1>机器学习基础</h1>
      <p>从学习范式、任务类型、泛化误差、评估指标到交叉验证，把机器学习入门最常用的几块地基放在同一个可调实验台里观察。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("全局设置")
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)
    st.caption("相同随机种子会复现相同的数据、划分和模型结果。")
    st.divider()
    st.caption("拖动控件会重新计算当前场景；较重的计算已经缓存。")


scene = segmented(
    "选择教学场景",
    ["学习范式", "任务类型", "过拟合与欠拟合", "模型评估", "交叉验证"],
    "学习范式",
)


if scene == "学习范式":
    render_scene_intro(
        "监督、无监督、半监督、强化学习",
        "机器学习的核心差异不在算法名字，而在训练时能看到什么反馈：完整答案、没有答案、少量答案，或者来自环境的奖励。",
        [
            ("监督学习", "每个样本都有目标标签，模型直接学习输入到答案的映射。"),
            ("无监督学习", "没有人工标签，模型从样本相似性、密度或结构里找规律。"),
            ("半监督学习", "少量样本有标签，大量样本无标签，用数据结构帮助标签传播。"),
            ("强化学习", "智能体选择动作，环境返回奖励，目标是学到长期收益高的策略。"),
        ],
    )
    label_fraction = st.slider("半监督场景中的标注比例", 0.04, 0.50, 0.16, 0.02)
    data = learning_type_data(int(seed), label_fraction)
    render_insight(
        "读图重点",
        "同一批点在监督学习里是带答案的训练题，在无监督学习里是待发现的结构。强化学习不再是一批独立样本，而是状态、动作和奖励之间的循环。",
    )
    st.plotly_chart(plot_learning_types(data), width="stretch", config=PLOT_CONFIG)
    render_table(
        [
            ("监督学习例子", "房价预测、图像分类、垃圾邮件识别。关键是有可靠标签。"),
            ("无监督学习例子", "用户分群、异常检测、表示学习。关键是发现结构而不是对答案。"),
            ("半监督学习例子", "医学图像和语音数据常见：未标注样本多，专家标签贵。"),
            ("强化学习例子", "游戏 AI、机器人控制、推荐系统中的长期反馈优化。"),
        ]
    )

elif scene == "任务类型":
    render_scene_intro(
        "回归、分类、聚类的可视化对比",
        "机器学习任务最先要问：输出是什么。连续数值是回归，离散标签是分类，没有标签但要分组就是聚类。",
        [
            ("回归", "输出是连续数值，错误大小可以用距离衡量。"),
            ("分类", "输出是离散类别，模型常先给概率，再按阈值或最大概率做决定。"),
            ("聚类", "训练时没有类别答案，算法根据相似性把样本分到若干簇。"),
            ("共同点", "三者都在寻找输入空间里的结构，只是目标形式不同。"),
        ],
    )
    cluster_count = st.slider("聚类簇数", 2, 4, 3, 1)
    data = task_comparison_data(int(seed), cluster_count)
    render_insight(
        "读图重点",
        "回归图里红线的纵坐标就是预测值；分类图的背景颜色是类别概率；聚类图里的颜色来自算法发现的簇，不是人工标签。",
    )
    st.plotly_chart(plot_task_comparison(data), width="stretch", config=PLOT_CONFIG)

elif scene == "过拟合与欠拟合":
    render_scene_intro(
        "模型复杂度与泛化",
        "训练误差低不等于模型好。欠拟合是规律没学够，过拟合是把训练集噪声也背下来了，泛化能力要看验证误差。",
        [
            ("欠拟合", "模型太简单，训练误差和验证误差都偏高。"),
            ("相对合适", "训练误差下降，验证误差也稳定下降或处在低位。"),
            ("过拟合", "训练误差继续下降，验证误差反而升高或明显高于训练误差。"),
            ("复杂度旋钮", "这里用多项式次数模拟模型容量，次数越高越容易弯曲。"),
        ],
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        degree = st.slider("模型复杂度: 多项式次数", 1, 18, 5, 1)
    with c2:
        noise = st.slider("数据噪声", 0.05, 0.80, 0.28, 0.01)
    with c3:
        train_size = st.slider("训练样本数", 20, 180, 70, 5)

    result = overfit_data(FitConfig(degree=degree, noise=noise, train_size=train_size, seed=int(seed)))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前状态", str(result["status"]))
    m2.metric("训练 MSE", f"{float(result['final_train']):.4f}")
    m3.metric("验证 MSE", f"{float(result['final_val']):.4f}")
    m4.metric("验证最优复杂度", f"{int(result['best_degree'])}")
    if result["status"] == "欠拟合":
        note = "模型没有足够能力追上真实曲线。通常可以提高复杂度、增加有效特征，或换用更合适的模型。"
    elif result["status"] == "过拟合":
        note = "训练误差很低但验证误差明显更高。通常需要降低复杂度、增加数据、加强正则化或早停。"
    else:
        note = "训练误差和验证误差处在相对平衡的位置。这个区间通常比单纯追求训练误差更重要。"
    render_insight("读图重点", note)
    st.plotly_chart(plot_overfit(result), width="stretch", config=PLOT_CONFIG)

elif scene == "模型评估":
    render_scene_intro(
        "混淆矩阵、ROC、精确率、召回率、F1",
        "分类模型输出概率以后，阈值会决定哪些样本被判成正类。阈值一变，精确率、召回率和混淆矩阵都会变。",
        [
            ("混淆矩阵", "把预测结果拆成 TN、FP、FN、TP，能看清错在什么类型。"),
            ("ROC 曲线", "扫过所有阈值，观察真阳性率和假阳性率的整体权衡。"),
            ("精确率", "预测为正的样本里，有多少真的为正。"),
            ("召回率 / F1", "召回率看正类找回多少；F1 是精确率和召回率的折中。"),
        ],
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.slider("分类阈值", 0.05, 0.95, 0.50, 0.01)
    with c2:
        class_sep = st.slider("类别可分程度", 0.45, 2.20, 1.05, 0.05)
    with c3:
        positive_rate = st.slider("正类比例", 0.10, 0.70, 0.35, 0.05)

    result = evaluation_data(
        EvalConfig(
            threshold=threshold,
            class_sep=class_sep,
            positive_rate=positive_rate,
            seed=int(seed),
        )
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{float(result['precision']):.1%}")
    m2.metric("Recall", f"{float(result['recall']):.1%}")
    m3.metric("F1", f"{float(result['f1']):.1%}")
    m4.metric("ROC AUC", f"{float(result['auc']):.3f}")
    render_insight(
        "读图重点",
        "提高阈值通常会减少误报、提高精确率，但可能漏掉更多正类、降低召回率。ROC AUC 不固定在某个阈值，而是评估排序能力。",
    )
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(plot_evaluation(result, threshold), width="stretch", config=PLOT_CONFIG)
    with right:
        st.plotly_chart(plot_metric_bars(result), width="stretch", config=PLOT_CONFIG)
        render_matplotlib(plot_threshold_tradeoff(result))
        render_table(
            [
                ("Precision", "适合误报代价高的场景，例如把普通邮件误判为垃圾邮件。"),
                ("Recall", "适合漏报代价高的场景，例如疾病筛查或风险预警。"),
                ("F1", "当精确率和召回率都重要时，用一个数观察折中表现。"),
            ]
        )

else:
    render_scene_intro(
        "交叉验证的可视化演示",
        "一次训练/验证划分可能刚好运气好或运气差。K 折交叉验证让每个样本轮流做验证集，用多次结果的平均值估计泛化能力。",
        [
            ("训练集", "每一折里用来拟合模型的样本。"),
            ("验证集", "当前折里暂时不参与训练，只用来评估。"),
            ("K 折平均", "把 K 次验证分数平均，比单次留出法更稳定。"),
            ("注意点", "交叉验证仍然不能替代最终测试集，调参太多也会过拟合验证流程。"),
        ],
    )
    c1, c2 = st.columns(2)
    with c1:
        folds = st.slider("K 折数量", 3, 10, 5, 1)
    with c2:
        shuffle = st.toggle("划分前打乱样本", value=True)

    result = cross_validation_data(CVConfig(folds=folds, shuffle=shuffle, seed=int(seed)))
    scores = np.array(result["scores"])
    m1, m2, m3 = st.columns(3)
    m1.metric("平均验证分数", f"{scores.mean():.1%}")
    m2.metric("折间标准差", f"{scores.std():.2%}")
    m3.metric("每折验证样本", f"{int(np.mean(result['fold_sizes']))}")
    render_insight(
        "读图重点",
        "左图中青色表示训练、红色表示验证。每一行是一折；红色窗口移动一次，就完成一次新的验证。",
    )
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(plot_cross_validation(result), width="stretch", config=PLOT_CONFIG)
    with right:
        st.plotly_chart(plot_cv_distribution(result), width="stretch", config=PLOT_CONFIG)
        render_table(
            [
                ("为什么更稳定", "每个样本都有机会进入验证集，减少某一次随机划分带来的偶然性。"),
                ("什么时候少用", "数据量极大、训练成本很高时，可以用留出集或时间切分等更便宜的方案。"),
                ("时间序列提醒", "有时间顺序的数据不能随便打乱，要使用按时间前后切分的验证方法。"),
            ]
        )
