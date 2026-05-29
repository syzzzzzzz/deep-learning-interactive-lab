"""Plotly chart adapters for neural network playground training history."""

from __future__ import annotations

import math

try:
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # quality_check may run without heavy dependencies.
    np = go = make_subplots = None  # type: ignore[assignment]

from components.playground_training import PlaygroundTrainingHistory
from components.playground_training import attention_token_labels

PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}


def make_loss_curve(history: PlaygroundTrainingHistory) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.epochs,
            y=history.losses,
            mode="lines+markers",
            line={"color": "#0f8b8d", "width": 3},
            marker={"size": 7},
            name="loss",
        )
    )
    fig.update_layout(
        title="损失曲线：真实轻量训练是否正在变好",
        height=320,
        margin={"l": 42, "r": 18, "t": 52, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_gradient_flow_chart(history: PlaygroundTrainingHistory) -> go.Figure:
    fig = go.Figure()
    colors = ["#0f8b8d", "#3268a8", "#bf3f5b", "#c4871f", "#46535d", "#6f5da8"]
    for index, (layer, values) in enumerate(history.grad_norms.items()):
        fig.add_trace(
            go.Scatter(
                x=history.epochs,
                y=values,
                mode="lines+markers",
                line={"color": colors[index % len(colors)], "width": 2.4},
                marker={"size": 5},
                name=layer,
            )
        )
    fig.update_layout(
        title="梯度流：每一层到底有没有收到学习信号",
        height=360,
        margin={"l": 42, "r": 18, "t": 52, "b": 68},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="Gradient L2 norm",
        legend={"orientation": "h", "y": -0.28},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_update_ratio_chart(history: PlaygroundTrainingHistory) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=history.epochs,
            y=[value * 100 for value in history.update_ratios],
            marker_color="#bf3f5b",
            name="参数更新幅度比",
        )
    )
    fig.update_layout(
        title="参数更新动画：每轮参数移动了多少",
        height=320,
        margin={"l": 42, "r": 18, "t": 52, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="更新 / 参数范数（%）",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_cnn_feature_map(feature_maps: np.ndarray) -> go.Figure:
    channel_count = int(feature_maps.shape[0])
    cols = min(3, channel_count)
    rows = int(math.ceil(channel_count / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"通道 {index}" for index in range(channel_count)],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )
    for index, channel in enumerate(feature_maps):
        row = index // cols + 1
        col = index % cols + 1
        fig.add_trace(
            go.Heatmap(z=channel, colorscale="Viridis", showscale=index == 0),
            row=row,
            col=col,
        )
    fig.update_layout(
        title="CNN 特征图：卷积层把输入图像改写成哪些响应",
        height=max(300, 210 * rows),
        margin={"l": 28, "r": 20, "t": 72, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def make_attention_heatmap(history: PlaygroundTrainingHistory) -> go.Figure:
    heatmap = history.attention_heatmap
    if heatmap is None:
        raise ValueError("缺少注意力热力图。")
    tokens = history.attention_tokens or attention_token_labels(int(heatmap.shape[0]))
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=heatmap,
                x=tokens,
                y=tokens,
                colorscale="YlGnBu",
                colorbar={"title": "权重"},
            )
        ]
    )
    title = "注意力热力图：query token 正在看向谁"
    if history.attention_is_simulated:
        title += "（教学模拟）"
    fig.update_layout(
        title=title,
        height=420,
        margin={"l": 78, "r": 22, "t": 58, "b": 78},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="被看的 key token",
        yaxis_title="当前 query token",
    )
    return fig

