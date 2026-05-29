"""
Training process visualization demo.

Run:
    streamlit run part6_universal_framework/training_demo.py
or:
    python main.py part6/training_demo
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from html import escape
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from components.visual_system import (
    render_beginner_hint,
    render_loading_bar,
    render_motion_note,
    render_neon_metric_card,
    render_training_dashboard_gauges,
    render_visual_system,
)


MODULE_TITLE = "训练过程可视化演示"
MODULE_SUMMARY = "用轻量数据集演示完整训练循环，实时展示损失、准确率、学习率和梯度范数。"
MODULE_TAGS = ["训练", "可视化", "演示", "实战"]

MAX_EPOCHS = 50
SAMPLE_COUNT = 200
INPUT_DIM = 6
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}


torch.set_num_threads(1)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    epochs: int
    hidden_size: int
    seed: int


@dataclass
class TrainingHistory:
    epochs: list[int]
    losses: list[float]
    accuracies: list[float]
    learning_rates: list[float]
    grad_norms: dict[str, list[float]]


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def go_home() -> None:
    st.query_params.clear()
    st.rerun()


def module_url(target: str) -> str:
    return f"/?module={quote(target, safe='')}"


def build_dataset(seed: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    x, y = make_classification(
        n_samples=SAMPLE_COUNT,
        n_features=INPUT_DIM,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.25,
        random_state=seed,
    )
    x = StandardScaler().fit_transform(x).astype(np.float32)
    y = y.astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y), x[:, :2]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def layer_grad_norms(model: nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        layer_name = name.rsplit(".", 1)[0]
        value = parameter.grad.detach().norm(2).item()
        norms[layer_name] = float((norms.get(layer_name, 0.0) ** 2 + value**2) ** 0.5)
    return norms


def empty_history() -> TrainingHistory:
    return TrainingHistory(epochs=[], losses=[], accuracies=[], learning_rates=[], grad_norms={})


def make_line_chart(
    x_values: list[int],
    y_values: list[float],
    title: str,
    y_title: str,
    color: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line={"color": color, "width": 3},
            marker={"size": 6},
            name=y_title,
        )
    )
    fig.update_layout(
        title=title,
        height=300,
        margin={"l": 40, "r": 20, "t": 48, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_grad_chart(history: TrainingHistory) -> go.Figure:
    fig = go.Figure()
    colors = ["#0f8b8d", "#bf3f5b", "#3268a8", "#c4871f"]
    for index, (layer, values) in enumerate(history.grad_norms.items()):
        fig.add_trace(
            go.Scatter(
                x=history.epochs,
                y=values,
                mode="lines+markers",
                line={"color": colors[index % len(colors)], "width": 2.5},
                marker={"size": 5},
                name=layer,
            )
        )
    fig.update_layout(
        title="每层梯度 L2 范数",
        height=330,
        margin={"l": 40, "r": 20, "t": 48, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="梯度范数",
        legend={"orientation": "h", "y": -0.24},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_dataset_chart(features_2d: np.ndarray, labels: torch.Tensor) -> go.Figure:
    label_array = labels.numpy()
    fig = go.Figure()
    for cls, color in ((0, "#3268a8"), (1, "#bf3f5b")):
        mask = label_array == cls
        fig.add_trace(
            go.Scatter(
                x=features_2d[mask, 0],
                y=features_2d[mask, 1],
                mode="markers",
                marker={"color": color, "size": 8, "opacity": 0.78},
                name=f"类别 {cls}",
            )
        )
    fig.update_layout(
        title="make_classification 生成的 200 个样本（前两维投影）",
        height=320,
        margin={"l": 36, "r": 16, "t": 48, "b": 32},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        font=PLOT_FONT,
        xaxis_title="特征 1",
        yaxis_title="特征 2",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def render_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172026;
            --muted: #596772;
            --line: #d8dee3;
            --paper: #fbfaf6;
            --teal: #0f8b8d;
            --rose: #bf3f5b;
            --amber: #c4871f;
            --blue: #3268a8;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(239,246,243,0.97) 100%),
                #fbfaf6;
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.2rem;
        }
        h1, h2, h3 { letter-spacing: 0; }
        section[data-testid="stSidebar"] {
            background: #eef4f2;
            border-right: 1px solid var(--line);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.7rem;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #172026;
            background: #172026;
            color: white;
            min-height: 2.45rem;
            font-weight: 750;
        }
        .stButton > button:hover {
            border-color: var(--teal);
            background: var(--teal);
            color: white;
        }
        .demo-hero {
            border-bottom: 1px solid var(--line);
            padding: 0.05rem 0 0.95rem 0;
            margin-bottom: 0.9rem;
        }
        .demo-hero h1 {
            font-size: clamp(2.05rem, 3vw, 3.25rem);
            line-height: 1.08;
            margin: 0;
        }
        .demo-hero p {
            color: var(--muted);
            max-width: 960px;
            line-height: 1.75;
            margin: 0.45rem 0 0.75rem 0;
        }
        .demo-badge {
            display: inline-block;
            border: 1px solid rgba(15,139,141,0.34);
            border-radius: 999px;
            background: rgba(15,139,141,0.10);
            color: #0b6264;
            padding: 0.22rem 0.62rem;
            font-size: 0.84rem;
            font-weight: 800;
            margin-right: 0.42rem;
        }
        .note {
            border-left: 4px solid var(--teal);
            background: rgba(255,255,255,0.74);
            border-radius: 0 8px 8px 0;
            padding: 0.75rem 0.9rem;
            color: #26343b;
            line-height: 1.68;
            margin: 0.4rem 0 1rem 0;
        }
        .related-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
        }
        .related-card {
            display: block;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255,255,255,0.78);
            padding: 0.78rem 0.85rem;
            text-decoration: none !important;
            color: var(--ink) !important;
            min-height: 92px;
        }
        .related-card:hover {
            border-color: var(--teal);
            box-shadow: 0 8px 22px rgba(15,139,141,0.12);
        }
        .related-card strong {
            display: block;
            margin-bottom: 0.32rem;
        }
        .related-card span {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }
        @media (max-width: 900px) {
            .related-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    tags = "".join(f'<span class="demo-badge">{escape(tag)}</span>' for tag in MODULE_TAGS)
    st.markdown(
        f"""
        <div class="demo-hero">
          <h1>{escape(MODULE_TITLE)}</h1>
          <p>{escape(MODULE_SUMMARY)}</p>
          <div><span class="demo-badge">教学演示</span>{tags}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_related_links() -> None:
    links = [
        (
            "训练动态",
            "part5_toolbox/03_training_dynamics",
            "观察训练曲线、收敛速度和过拟合迹象。",
        ),
        (
            "梯度监控",
            "part5_toolbox/02_gradient_monitor",
            "进一步分析梯度范数、梯度爆炸和消失。",
        ),
        (
            "特征可视化",
            "part5_toolbox/01_feature_visualization",
            "理解模型中间表示和特征空间变化。",
        ),
    ]
    cards = []
    for title, target, description in links:
        cards.append(
            f"""
            <a class="related-card" href="{module_url(target)}" target="_self">
              <strong>{escape(title)}</strong>
              <span>{escape(description)}</span>
            </a>
            """
        )
    st.subheader("相关知识点")
    st.markdown(
        '<div class="note">本演示是轻量教学版本，完整训练实验请前往工具箱章节。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="related-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def train_demo(config: TrainingConfig) -> Iterator[tuple[TrainingHistory, dict[str, float], TinyMLP, np.ndarray, torch.Tensor]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x_tensor, y_tensor, features_2d = build_dataset(config.seed)
    model = TinyMLP(INPUT_DIM, config.hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    history = empty_history()

    start_time = time.perf_counter()
    for epoch in range(1, min(config.epochs, MAX_EPOCHS) + 1):
        model.train()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        grad_snapshot = layer_grad_norms(model)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y_tensor).float().mean().item()

        history.epochs.append(epoch)
        history.losses.append(float(loss.item()))
        history.accuracies.append(float(accuracy))
        history.learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        for layer, norm in grad_snapshot.items():
            history.grad_norms.setdefault(layer, []).append(norm)

        yield history, {
            "elapsed": time.perf_counter() - start_time,
            "loss": history.losses[-1],
            "accuracy": history.accuracies[-1],
        }, model, features_2d, y_tensor


def render_training_panel(config: TrainingConfig) -> None:
    metric_cols = st.columns(4)
    loss_slot = metric_cols[0].empty()
    acc_slot = metric_cols[1].empty()
    lr_slot = metric_cols[2].empty()
    time_slot = metric_cols[3].empty()

    progress = st.progress(0, text="等待训练开始")
    left, right = st.columns(2)
    loss_chart = left.empty()
    accuracy_chart = right.empty()
    lr_chart = left.empty()
    grad_chart = right.empty()

    final_history: TrainingHistory | None = None
    final_stats: dict[str, float] | None = None
    final_model: TinyMLP | None = None

    with st.spinner("正在训练轻量二分类 MLP，并逐 epoch 更新指标..."):
        for history, stats, model, _features_2d, _labels in train_demo(config):
            final_history = history
            final_stats = stats
            final_model = model

            epoch = history.epochs[-1]
            progress.progress(epoch / config.epochs, text=f"训练进度：Epoch {epoch}/{config.epochs}")
            loss_slot.metric("当前损失", f"{stats['loss']:.4f}")
            acc_slot.metric("当前准确率", f"{stats['accuracy'] * 100:.1f}%")
            lr_slot.metric("当前学习率", f"{history.learning_rates[-1]:.5f}")
            time_slot.metric("已用时间", f"{stats['elapsed']:.2f}s")

            loss_chart.plotly_chart(
                make_line_chart(history.epochs, history.losses, "损失曲线", "CrossEntropy Loss", "#bf3f5b"),
                width="stretch",
                config=PLOT_CONFIG,
            )
            accuracy_chart.plotly_chart(
                make_line_chart(history.epochs, history.accuracies, "准确率曲线", "Accuracy", "#0f8b8d"),
                width="stretch",
                config=PLOT_CONFIG,
            )
            lr_chart.plotly_chart(
                make_line_chart(history.epochs, history.learning_rates, "学习率曲线（CosineAnnealingLR）", "Learning Rate", "#3268a8"),
                width="stretch",
                config=PLOT_CONFIG,
            )
            grad_chart.plotly_chart(make_grad_chart(history), width="stretch", config=PLOT_CONFIG)

    progress.progress(1.0, text="训练完成")
    if final_history is None or final_stats is None or final_model is None:
        st.warning("训练没有产生有效结果，请调整参数后重试。")
        return

    st.subheader("训练完成")
    final_cols = st.columns(4)
    final_cols[0].metric("最终损失", f"{final_stats['loss']:.4f}")
    final_cols[1].metric("最终准确率", f"{final_stats['accuracy'] * 100:.1f}%")
    final_cols[2].metric("训练时间", f"{final_stats['elapsed']:.2f}s")
    final_cols[3].metric("参数量", f"{count_parameters(final_model):,}")

    with st.expander("参数量明细", expanded=False):
        rows = [
            {"层": name, "形状": tuple(parameter.shape), "参数量": int(parameter.numel())}
            for name, parameter in final_model.named_parameters()
        ]
        st.table(rows)


def render_app() -> None:
    st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
    render_style()
    render_visual_system("light")
    render_header()
    render_loading_bar("训练仪表盘加载：指标指针会随着训练状态给出直观信号")
    render_beginner_hint(
        "先看 loss，再看 accuracy",
        "loss 是优化器真正最小化的目标；accuracy 是给人看的结果指标。两者一起看，才能判断模型是在学习还是在记答案。",
        action="训练前先记住当前学习率和 epoch 数，点击开始训练后观察 loss 是否平滑下降。",
    )
    render_motion_note(
        "仪表盘不是装饰",
        "Loss 看优化方向，Accuracy 看任务效果，LR 看步子大小，梯度范数看更新是否过猛或过弱。",
    )
    render_training_dashboard_gauges()

    top_left, top_right = st.columns([0.78, 0.22])
    with top_left:
        st.markdown(
            '<div class="note">这个页面固定使用 CPU 和 200 个 sklearn 样本，重点展示训练循环里哪些信号应该被持续观察。</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("返回主界面", width="stretch"):
            go_home()

    with st.sidebar:
        st.header("训练参数")
        learning_rate = st.slider("学习率", min_value=0.001, max_value=0.200, value=0.030, step=0.001)
        epochs = st.slider("Epoch 数", min_value=5, max_value=MAX_EPOCHS, value=25, step=1)
        hidden_size = st.slider("隐藏层大小", min_value=4, max_value=64, value=16, step=4)
        seed = st.slider("随机种子", min_value=0, max_value=99, value=7, step=1)
        run_clicked = st.button("开始训练", width="stretch", type="primary")

    config = TrainingConfig(
        learning_rate=float(learning_rate),
        epochs=min(int(epochs), MAX_EPOCHS),
        hidden_size=int(hidden_size),
        seed=int(seed),
    )

    x_tensor, y_tensor, features_2d = build_dataset(config.seed)
    overview_cols = st.columns([0.58, 0.42])
    with overview_cols[0]:
        st.plotly_chart(make_dataset_chart(features_2d, y_tensor), width="stretch", config=PLOT_CONFIG)
    with overview_cols[1]:
        st.subheader("演示设置")
        metric_cols = st.columns(2)
        with metric_cols[0]:
            render_neon_metric_card("样本数", str(SAMPLE_COUNT), caption="教学页固定小样本，保证训练反馈足够快。")
        with metric_cols[1]:
            render_neon_metric_card("输入维度", str(INPUT_DIM), caption="每个样本进入 MLP 前有 6 个特征。")
        st.write(f"样本数：{SAMPLE_COUNT}")
        st.write(f"输入特征：{INPUT_DIM}")
        st.write("模型：Linear → ReLU → Linear")
        st.write("优化器：Adam")
        st.write("Scheduler：CosineAnnealingLR")
        st.caption("每个 epoch 做一次完整前向、反向、参数更新和指标记录。")

    if run_clicked:
        render_training_panel(config)
    else:
        st.info("调整左侧参数后点击“开始训练”。Epoch 上限固定为 50，避免教学页面长时间占用本机资源。")

    st.divider()
    render_related_links()


if __name__ == "__main__":
    try:
        render_app()
    except Exception as error:
        try:
            st.error("训练过程可视化演示暂时无法运行。")
            st.warning("请返回主界面重新进入，或调小学习率、Epoch 数后重试。")
            if st.button("返回主界面"):
                go_home()
            with st.expander("错误详情", expanded=False):
                st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)), language="text")
        except Exception:
            raise
