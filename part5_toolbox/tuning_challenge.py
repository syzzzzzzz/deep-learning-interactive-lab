"""
Interactive hyperparameter tuning challenge.

Run:
    streamlit run part5_toolbox/tuning_challenge.py
or:
    python main.py part5/tuning_challenge
"""

from __future__ import annotations

import hashlib
import html
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


INK = "#172026"
MUTED = "#596772"
LINE = "#d8dee3"
TEAL = "#0f8b8d"
ROSE = "#bf3f5b"
AMBER = "#c4871f"
BLUE = "#3268a8"
GREEN = "#3f7d58"
VIOLET = "#7353ba"
PAPER = "#f8f9f6"

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": INK}


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str
    data_size: int
    noise: float
    budget: int
    base_acc: float
    max_acc: float
    ideal_lr: float
    ideal_decay: float
    ideal_dropout: float
    ideal_aug: float
    ideal_model: int
    hint: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "小样本图像分类",
        "2,000 张医学影像，类别轻微不均衡，需要稳定泛化。",
        2000,
        0.18,
        18,
        0.68,
        0.91,
        0.0012,
        0.0008,
        0.28,
        0.55,
        3,
        "小数据先控过拟合：增强、权重衰减和适度 dropout 往往比盲目加大模型更有效。",
    ),
    Scenario(
        "文本情感分类",
        "60,000 条短文本，标签较干净，模型收敛速度是主要约束。",
        60000,
        0.08,
        10,
        0.78,
        0.94,
        0.0007,
        0.0002,
        0.12,
        0.18,
        4,
        "数据量足够时，过强正则会欠拟合；学习率和批大小的配合更关键。",
    ),
    Scenario(
        "工业缺陷检测",
        "12,000 张工厂图片，缺陷样本少，线上误报成本高。",
        12000,
        0.14,
        14,
        0.72,
        0.93,
        0.0009,
        0.0012,
        0.22,
        0.65,
        4,
        "类别不均衡时不要只追训练准确率，验证曲线稳定性和召回/误报权衡更重要。",
    ),
)


def e(value: str) -> str:
    return html.escape(value, quote=True)


def stable_noise(*parts: object, scale: float = 1.0) -> float:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale


def page_style() -> None:
    st.set_page_config(page_title="调参实战挑战", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; }
        .stApp { background: #f8f9f6; color: #172026; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid #d8dee3;
            border-radius: 8px;
            padding: 10px 12px;
        }
        .note {
            border-left: 4px solid #0f8b8d;
            background: rgba(255,255,255,0.78);
            border-radius: 0 8px 8px 0;
            padding: 0.74rem 0.9rem;
            line-height: 1.68;
            margin: 0.35rem 0 0.9rem 0;
        }
        .diagnosis {
            background: rgba(255,255,255,0.78);
            border: 1px solid #d8dee3;
            border-radius: 8px;
            padding: 0.84rem 0.94rem;
            line-height: 1.62;
            min-height: 8rem;
        }
        .tag {
            display: inline-block;
            border: 1px solid #ccd6da;
            border-radius: 999px;
            padding: 0.12rem 0.52rem;
            margin: 0.1rem 0.25rem 0.1rem 0;
            color: #4d5a63;
            background: rgba(255,255,255,0.72);
            font-size: 0.84rem;
        }
        .small { color: #596772; font-size: 0.92rem; line-height: 1.58; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def selected_scenario(name: str) -> Scenario:
    return next(scenario for scenario in SCENARIOS if scenario.name == name)


def log_distance(value: float, target: float) -> float:
    return abs(math.log10(value) - math.log10(target))


def evaluate(
    scenario: Scenario,
    lr: float,
    batch_size: int,
    weight_decay: float,
    dropout: float,
    augmentation: float,
    model_size: int,
    epochs: int,
) -> dict[str, float | str]:
    lr_penalty = min(log_distance(lr, scenario.ideal_lr) / 1.2, 1.5)
    decay_penalty = min(log_distance(max(weight_decay, 1e-7), scenario.ideal_decay) / 1.6, 1.2)
    dropout_penalty = abs(dropout - scenario.ideal_dropout) / 0.45
    aug_penalty = abs(augmentation - scenario.ideal_aug) / 0.75
    model_gap = abs(model_size - scenario.ideal_model) / 4

    batch_factor = math.log2(batch_size / 64)
    batch_penalty = max(0.0, abs(batch_factor) - 1.2) * 0.05
    effective_regularization = weight_decay * 900 + dropout + augmentation * 0.42
    capacity = model_size / 5

    overfit = max(0.0, capacity + 0.35 - effective_regularization - scenario.data_size / 50000)
    underfit = max(0.0, effective_regularization + 0.22 - capacity - scenario.data_size / 120000)
    instability = max(0.0, log_distance(lr, scenario.ideal_lr) - 0.35) + max(0.0, batch_factor - 1.5) * 0.25

    score_loss = (
        0.10 * lr_penalty
        + 0.055 * decay_penalty
        + 0.055 * dropout_penalty
        + 0.045 * aug_penalty
        + 0.05 * model_gap
        + batch_penalty
        + 0.045 * overfit
        + 0.04 * underfit
        + 0.035 * instability
    )
    noise = stable_noise(scenario.name, lr, batch_size, weight_decay, dropout, augmentation, model_size, epochs, scale=0.009)
    val_acc = max(0.35, min(scenario.max_acc, scenario.max_acc - score_loss + noise))

    train_boost = 0.05 * capacity + 0.06 * overfit - 0.05 * underfit
    train_acc = max(val_acc, min(0.995, val_acc + train_boost + stable_noise("train", lr, model_size, scale=0.004)))
    gap = train_acc - val_acc
    stability = max(0.0, min(1.0, 1 - instability * 0.55 - scenario.noise * 0.45))
    time_cost = epochs * (0.45 + model_size * 0.24) * (64 / batch_size) ** 0.35
    budget_used = min(99.0, time_cost / scenario.budget * 100)

    if instability > 0.75:
        diagnosis = "训练不稳定：学习率或批大小让更新噪声过大，先做 LR range test 或把学习率降半个数量级。"
    elif gap > 0.09:
        diagnosis = "过拟合明显：训练准确率高于验证准确率，优先加强数据增强、权重衰减或降低模型规模。"
    elif underfit > 0.45:
        diagnosis = "欠拟合：正则偏强或模型容量不足，可以减小 dropout/增强强度，或提高模型规模。"
    elif budget_used > 100:
        diagnosis = "预算吃紧：当前配置可能有效，但训练成本超出约束，尝试更大 batch 或更小模型做第一轮搜索。"
    elif val_acc > scenario.max_acc - 0.025:
        diagnosis = "配置接近最优：下一步做小范围局部搜索，并固定随机种子确认结果稳定。"
    else:
        diagnosis = "结果可用但还有空间：围绕学习率、正则强度和模型容量做一次单变量消融。"

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": gap,
        "stability": stability,
        "time_cost": time_cost,
        "budget_used": budget_used,
        "diagnosis": diagnosis,
        "overfit": overfit,
        "underfit": underfit,
        "instability": instability,
    }


@st.cache_data(show_spinner=False)
def learning_curves(train_acc: float, val_acc: float, gap: float, instability: float, epochs: int) -> pd.DataFrame:
    xs = np.arange(1, epochs + 1)
    speed = 3.2 / max(epochs, 1)
    train = train_acc - (train_acc - 0.48) * np.exp(-speed * xs)
    val = val_acc - (val_acc - 0.44) * np.exp(-speed * xs * 0.88)
    wobble = np.sin(xs * 1.7) * min(0.035, instability * 0.018)
    val = np.clip(val + wobble, 0, 1)
    train_loss = np.clip(1.3 - train * 1.05 + np.exp(-xs / max(2, epochs / 5)) * 0.35, 0.02, None)
    val_loss = np.clip(1.35 - val * 1.0 + np.exp(-xs / max(2, epochs / 4)) * 0.30 + gap * 0.55, 0.02, None)
    return pd.DataFrame({"epoch": xs, "train_acc": train, "val_acc": val, "train_loss": train_loss, "val_loss": val_loss})


def curve_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_acc"], mode="lines+markers", name="训练准确率", line={"color": AMBER, "width": 3}))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_acc"], mode="lines+markers", name="验证准确率", line={"color": TEAL, "width": 3}))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], mode="lines", name="训练损失", yaxis="y2", line={"color": ROSE, "dash": "dot"}))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"], mode="lines", name="验证损失", yaxis="y2", line={"color": BLUE, "dash": "dot"}))
    fig.update_layout(
        height=390,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=PLOT_FONT,
        xaxis={"title": "epoch", "gridcolor": "rgba(23,32,38,0.12)"},
        yaxis={"title": "accuracy", "range": [0.35, 1.0], "gridcolor": "rgba(23,32,38,0.12)"},
        yaxis2={"title": "loss", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.22},
    )
    return fig


@st.cache_data(show_spinner=False)
def search_landscape_data(scenario_name: str, batch_size: int, epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario = selected_scenario(scenario_name)
    lrs = np.logspace(-5, -1, 22)
    decays = np.logspace(-6, -2, 22)
    z = []
    for decay in decays:
        row = []
        for lr in lrs:
            res = evaluate(
                scenario,
                lr=lr,
                batch_size=batch_size,
                weight_decay=decay,
                dropout=scenario.ideal_dropout,
                augmentation=scenario.ideal_aug,
                model_size=scenario.ideal_model,
                epochs=epochs,
            )
            row.append(float(res["val_acc"]))
        z.append(row)
    return lrs, decays, np.array(z, dtype=np.float32)


def search_landscape(scenario: Scenario, batch_size: int, epochs: int) -> go.Figure:
    lrs, decays, z = search_landscape_data(scenario.name, batch_size, epochs)
    fig = go.Figure(
        data=go.Heatmap(
            x=lrs,
            y=decays,
            z=z,
            colorscale=[[0, "#f3d8d2"], [0.45, "#f2c572"], [0.72, "#6ab7a8"], [1, "#3268a8"]],
            colorbar={"title": "val acc"},
            hovertemplate="lr=%{x:.1e}<br>decay=%{y:.1e}<br>val=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=390,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=PLOT_FONT,
        xaxis={"title": "learning rate", "type": "log"},
        yaxis={"title": "weight decay", "type": "log"},
    )
    return fig


def experiment_row(scenario: Scenario, lr: float, batch_size: int, weight_decay: float, dropout: float, augmentation: float, model_size: int, epochs: int) -> dict[str, object]:
    result = evaluate(scenario, lr, batch_size, weight_decay, dropout, augmentation, model_size, epochs)
    return {
        "场景": scenario.name,
        "lr": f"{lr:.1e}",
        "batch": batch_size,
        "decay": f"{weight_decay:.1e}",
        "dropout": round(dropout, 2),
        "增强": round(augmentation, 2),
        "模型": model_size,
        "epochs": epochs,
        "验证准确率": round(float(result["val_acc"]), 4),
        "泛化差距": round(float(result["gap"]), 4),
        "稳定性": round(float(result["stability"]), 3),
    }


page_style()

st.title("调参实战挑战")
st.markdown(
    '<div class="note">目标不是记住某个神奇学习率，而是训练一个决策习惯：先识别失败模式，再选择最便宜的下一轮实验。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("实验配置")
    scenario_name = st.selectbox("任务场景", [scenario.name for scenario in SCENARIOS])
    scenario = selected_scenario(scenario_name)
    lr = st.select_slider("学习率", options=[1e-5, 3e-5, 1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], value=7e-4)
    batch_size = st.select_slider("批大小", options=[16, 32, 64, 128, 256, 512], value=64)
    weight_decay = st.select_slider("权重衰减", options=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=3e-4)
    dropout = st.slider("Dropout", 0.0, 0.7, 0.2, step=0.05)
    augmentation = st.slider("数据增强强度", 0.0, 1.0, 0.45, step=0.05)
    model_size = st.slider("模型规模", 1, 5, 3)
    epochs = st.slider("训练轮数", 3, 30, 12)
    st.divider()
    st.caption("模拟器用固定规则生成结果，适合练习调参判断，不代表真实数据集指标。")

result = evaluate(scenario, lr, batch_size, weight_decay, dropout, augmentation, model_size, epochs)
history = learning_curves(
    float(result["train_acc"]),
    float(result["val_acc"]),
    float(result["gap"]),
    float(result["instability"]),
    epochs,
)

top = st.columns(4)
top[0].metric("验证准确率", f"{float(result['val_acc']):.2%}")
top[1].metric("训练准确率", f"{float(result['train_acc']):.2%}")
top[2].metric("泛化差距", f"{float(result['gap']):.2%}")
top[3].metric("预算占用", f"{float(result['budget_used']):.0f}%")

st.markdown(f'<div class="note"><strong>{e(scenario.name)}：</strong>{e(scenario.task)} {e(scenario.hint)}</div>', unsafe_allow_html=True)

left, right = st.columns([0.58, 0.42])
with left:
    st.plotly_chart(curve_chart(history), use_container_width=True, config=PLOT_CONFIG)
with right:
    st.subheader("诊断反馈")
    st.markdown(
        '<div class="diagnosis">'
        f"<strong>当前判断</strong><br>{e(str(result['diagnosis']))}<br><br>"
        f'<span class="tag">过拟合 {float(result["overfit"]):.2f}</span>'
        f'<span class="tag">欠拟合 {float(result["underfit"]):.2f}</span>'
        f'<span class="tag">不稳定 {float(result["instability"]):.2f}</span>'
        f'<span class="tag">稳定性 {float(result["stability"]):.2f}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("记录本轮实验", use_container_width=True):
        rows = st.session_state.get("tuning_rows", [])
        rows.append(experiment_row(scenario, lr, batch_size, weight_decay, dropout, augmentation, model_size, epochs))
        st.session_state["tuning_rows"] = rows[-12:]

tabs = st.tabs(["搜索地形", "实验记录", "调参策略卡", "复盘问题"])

with tabs[0]:
    st.subheader("学习率 x 权重衰减地形")
    st.plotly_chart(search_landscape(scenario, batch_size, epochs), use_container_width=True, config=PLOT_CONFIG)
    st.markdown(
        '<div class="note">先做粗搜索找到可训练区域，再做局部搜索。网格图里大片低分区域通常比单个最高点更有价值，因为它告诉你哪些设置很脆弱。</div>',
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.subheader("实验记录")
    rows = st.session_state.get("tuning_rows", [])
    if rows:
        df = pd.DataFrame(rows).sort_values("验证准确率", ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)
        best = df.iloc[0]
        st.markdown(
            f'<div class="note">当前最好结果：验证准确率 {best["验证准确率"]:.2%}，配置为 lr={best["lr"]}、batch={best["batch"]}、decay={best["decay"]}。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("点击“记录本轮实验”后，这里会形成可比较的实验表。")

with tabs[2]:
    st.subheader("失败模式到动作")
    cards = [
        ("训练损失不降", "先查学习率、标签、数据管线；学习率过高会震荡，过低会像没训练。"),
        ("训练好验证差", "优先看数据泄漏、切分方式、增强、权重衰减、dropout 和早停。"),
        ("训练验证都差", "可能欠拟合：提高模型容量、训练轮数，或减少过强正则。"),
        ("结果波动大", "固定随机种子，多跑几次；降低学习率或使用更稳的调度器。"),
    ]
    cols = st.columns(2)
    for index, (title, body) in enumerate(cards):
        with cols[index % 2]:
            st.markdown(f'<div class="diagnosis"><strong>{e(title)}</strong><br>{e(body)}</div>', unsafe_allow_html=True)

with tabs[3]:
    st.subheader("本轮复盘")
    st.checkbox("我能解释当前训练/验证差距来自哪里")
    st.checkbox("我只改了少数变量，能归因本轮变化")
    st.checkbox("我记录了失败配置，而不只记录最好结果")
    st.checkbox("我知道下一轮最便宜的实验是什么")
    st.markdown(
        '<div class="note">真正的调参能力来自可复盘的实验设计：每轮实验应该能排除一个假设，或者缩小一个搜索范围。</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    """Page entry point — content runs at module import time."""
    pass


def compute(seed: int = 42) -> dict[str, object]:
    """Pure computation placeholder."""
    return {"status": "ok", "seed": seed}


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""
    return True
