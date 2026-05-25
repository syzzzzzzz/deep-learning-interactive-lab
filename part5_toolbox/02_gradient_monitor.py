"""
梯度监控与超参搜索工具 — 统一视觉系统版

Run:
    streamlit run part5_toolbox/02_gradient_monitor.py
or:
    python main.py part5_toolbox/02_gradient_monitor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from components.visual_system import NEON_BLUE, NEON_GREEN, NEON_PURPLE, render_visual_system


def print_learning_guide():
    print("""
学习导读：梯度监控不是"看一堆数字"，而是在给训练过程做体检。

1. 梯度健康仪表盘怎么看
   - 每根柱子代表一层的梯度范数，颜色和脉冲编码健康状态。
   - 绿色脉冲 = 正常（1e-6 ~ 100），蓝色暗淡 = 消失（< 1e-6），红色闪烁 = 爆炸（> 100）。
   - 只看单个 batch 不够，至少观察最近 10~50 步的趋势。

2. 和训练曲线怎么联合诊断
   - loss 发散 + 梯度 max 暴涨 → 学习率过高、标签异常或 loss 实现错误。
   - loss 不动 + 前层 mean 接近 0 → 梯度消失，优先查初始化、激活函数、残差/归一化。
   - loss 正常下降但某层长期异常 → 该层可能被冻结或没有接入计算图。

3. 梯度爆炸 vs 梯度消失
   - 梯度消失：深层网络常见，Sigmoid/Tanh 是主要元凶；解决方案包括 ReLU/GELU、残差连接、He 初始化、BatchNorm。
   - 梯度爆炸：梯度范数超过 100 就要警惕；解决方案包括梯度裁剪 clip_grad_norm_、降低学习率、Xavier/He 初始化。

工程坑案例：
   在一个文本分类项目里只看验证准确率以为模型欠拟合，连续加大模型；
   后来梯度图显示 embedding 层长期接近 0，根因是 tokenizer 把大部分词都映射成了 UNK。
   先看梯度，可以少走很多弯路。

进阶思考：
   如果只有最后一层梯度很大，你会先查标签和 loss，还是先重写模型？
   如果只有前几层梯度消失，残差连接和归一化各解决什么问题？
""".strip())


st.set_page_config(
    page_title="梯度监控与超参搜索",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_visual_system("dark")

# ─────────────────────────────────────────────────────────
# 页面级样式
# ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2.2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    .hero { border-bottom: 1px solid var(--vs-line); padding-bottom: 0.85rem; margin-bottom: 0.85rem; }
    .hero h1 { font-size: clamp(2rem, 3vw, 3.2rem); line-height: 1.08; margin: 0; }
    .hero p { color: var(--vs-muted); max-width: 980px; line-height: 1.75; margin: 0.45rem 0 0 0; }
    .note {
        border-left: 4px solid var(--vs-blue);
        background: var(--vs-panel-soft);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        color: var(--vs-ink);
        line-height: 1.7;
        margin: 0.35rem 0 0.85rem 0;
    }
    .callout {
        background: var(--vs-panel-soft);
        border: 1px solid var(--vs-line);
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        color: var(--vs-ink);
        line-height: 1.68;
        margin: 0.35rem 0 0.75rem 0;
    }
    .grad-dashboard {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 0.6rem;
        padding: 1.5rem 0.5rem 0.5rem;
        min-height: 180px;
    }
    .grad-stage {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.3rem;
    }
    .grad-bar {
        width: 100%;
        max-width: 48px;
        height: 100px;
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
        overflow: hidden;
        position: relative;
    }
    .grad-fill {
        position: absolute;
        bottom: 0;
        width: 100%;
        border-radius: 6px;
    }
    .grad-healthy .grad-fill {
        height: 65%;
        background: linear-gradient(to top, rgba(0,255,136,0.3), """ + NEON_GREEN + """);
        box-shadow: 0 0 16px rgba(0,255,136,0.4);
        animation: grad-pulse-healthy 2s ease-in-out infinite;
    }
    .grad-vanishing .grad-fill {
        height: 8%;
        background: linear-gradient(to top, rgba(0,240,255,0.2), """ + NEON_BLUE + """);
        box-shadow: 0 0 8px rgba(0,240,255,0.2);
        animation: grad-pulse-vanishing 3s ease-in-out infinite;
    }
    .grad-exploding .grad-fill {
        height: 95%;
        background: linear-gradient(to top, rgba(255,51,85,0.3), #ff3355);
        box-shadow: 0 0 24px rgba(255,51,85,0.5);
        animation: grad-pulse-exploding 0.8s ease-in-out infinite;
    }
    .grad-norm {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.72rem;
        color: var(--vs-muted);
    }
    .grad-name {
        font-size: 0.78rem;
        color: var(--vs-ink);
        font-weight: 600;
    }
    .grad-label { font-size: 0.72rem; font-weight: 700; }
    .grad-legend {
        display: flex;
        gap: 1.2rem;
        margin-top: 0.6rem;
        font-size: 0.82rem;
        color: var(--vs-muted);
    }
    .grad-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.3rem;
        vertical-align: middle;
    }
    @keyframes grad-pulse-healthy {
        0%, 100% { opacity: 0.7; filter: brightness(0.9); }
        50% { opacity: 1; filter: brightness(1.3); }
    }
    @keyframes grad-pulse-vanishing {
        0%, 100% { opacity: 0.3; filter: brightness(0.5); }
        50% { opacity: 0.6; filter: brightness(0.8); }
    }
    @keyframes grad-pulse-exploding {
        0%, 100% { opacity: 0.8; filter: brightness(1.0); }
        50% { opacity: 1; filter: brightness(1.6); }
    }
    @media (max-width: 760px) {
        .grad-dashboard { flex-wrap: wrap; }
        .grad-stage { min-width: 60px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# 梯度健康仪表盘动效
# ─────────────────────────────────────────────────────────


def render_gradient_health_dashboard(
    gradient_norms: list[float] | None = None,
    layer_names: list[str] | None = None,
) -> None:
    """梯度健康仪表盘：正常/消失/爆炸三种状态的可视化。"""
    if gradient_norms is None:
        gradient_norms = [0.85, 0.42, 0.003, 0.000001, 0.95, 120.5, 0.67]
    if layer_names is None:
        layer_names = ["Conv1", "Conv2", "Conv3", "Conv4", "FC1", "FC2", "FC3"]

    stages: list[str] = []
    for name, norm in zip(layer_names, gradient_norms):
        if norm < 1e-6:
            status, color, label = "vanishing", NEON_BLUE, "消失"
        elif norm > 100:
            status, color, label = "exploding", "#ff3355", "爆炸"
        else:
            status, color, label = "healthy", NEON_GREEN, "正常"
        stages.append(
            f'<div class="grad-stage grad-{status}">'
            f'<div class="grad-norm">{norm:.2e}</div>'
            f'<div class="grad-bar"><div class="grad-fill"></div></div>'
            f'<div class="grad-name">{name}</div>'
            f'<div class="grad-label" style="color:{color}">{label}</div>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="vs-card" style="padding:1rem">
          <div style="font-weight:850;margin-bottom:.7rem;color:var(--vs-ink)">
            <i class="fa-solid fa-heart-pulse"></i> 梯度健康仪表盘
          </div>
          <div class="grad-dashboard">{"".join(stages)}</div>
          <div class="grad-legend">
            <span><span class="grad-dot" style="background:{NEON_GREEN}"></span> 正常 (1e-6 ~ 100)</span>
            <span><span class="grad-dot" style="background:{NEON_BLUE}"></span> 消失 (&lt; 1e-6)</span>
            <span><span class="grad-dot" style="background:#ff3355"></span> 爆炸 (&gt; 100)</span>
          </div>
          <p style="color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0">
            每根柱子代表一层的梯度范数。绿色脉冲 = 健康；蓝色暗淡 = 梯度消失（参数难以更新）；红色闪烁 = 梯度爆炸（训练不稳定）。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# 梯度监控器：实时追踪每层梯度健康状况
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class GradientMonitor:
    """
    生产级梯度监控工具

    功能：
    1. 实时追踪每层梯度的均值、标准差、最大值
    2. 自动检测梯度消失/爆炸
    3. 绘制梯度流动图
    4. 生成诊断报告

    使用方法：
        monitor = GradientMonitor(model)
        for epoch in range(epochs):
            loss.backward()
            monitor.record()   # 每次 backward 后调用
            optimizer.step()
        monitor.plot()
        monitor.report()
    """

    VANISHING_THRESHOLD = 1e-6
    EXPLODING_THRESHOLD = 100.0

    def __init__(self, model: nn.Module, watch_layers: list[str] | None = None):
        self.model = model
        self.watch_layers = watch_layers
        self.history: dict[str, list[dict]] = {}
        self.step = 0

    def record(self):
        """在 backward() 之后调用，记录当前梯度状态"""
        self.step += 1
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if self.watch_layers and not any(w in name for w in self.watch_layers):
                continue

            grad = param.grad.detach()
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(
                {
                    "step": self.step,
                    "mean": grad.abs().mean().item(),
                    "std": grad.std().item(),
                    "max": grad.abs().max().item(),
                    "norm": grad.norm().item(),
                }
            )

    def diagnose(self) -> dict[str, list[str]]:
        """返回诊断结果：正常/消失/爆炸层名列表。"""
        issues: dict[str, list[str]] = {"healthy": [], "vanishing": [], "exploding": []}
        for name, data in self.history.items():
            if not data:
                continue
            recent = data[-min(10, len(data)) :]
            avg_mean = np.mean([d["mean"] for d in recent])
            avg_max = np.max([d["max"] for d in recent])
            if avg_mean < self.VANISHING_THRESHOLD:
                issues["vanishing"].append(name)
            elif avg_max > self.EXPLODING_THRESHOLD:
                issues["exploding"].append(name)
            else:
                issues["healthy"].append(name)
        return issues

    def latest_norms(self) -> tuple[list[str], list[float]]:
        """返回最新一步各层名称和梯度范数。"""
        names, norms = [], []
        for name, data in self.history.items():
            if data:
                names.append(name.replace(".weight", "").replace(".bias", ""))
                norms.append(data[-1]["norm"])
        return names, norms

    def report(self) -> dict[str, list[str]]:
        """生成梯度健康诊断报告（Streamlit 版）。"""
        issues = self.diagnose()

        st.markdown("### 梯度健康诊断报告")
        rows = []
        for name, data in self.history.items():
            if not data:
                continue
            recent = data[-min(10, len(data)) :]
            avg_mean = np.mean([d["mean"] for d in recent])
            avg_max = np.max([d["max"] for d in recent])
            if avg_mean < self.VANISHING_THRESHOLD:
                status = "⚠️ 消失"
            elif avg_max > self.EXPLODING_THRESHOLD:
                status = "🔥 爆炸"
            else:
                status = "✅ 正常"
            rows.append(
                {"层名": name, "均值": f"{avg_mean:.2e}", "最大值": f"{avg_max:.2e}", "状态": status}
            )

        if rows:
            import pandas as pd

            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("正常层数", len(issues["healthy"]))
        c2.metric("梯度消失", len(issues["vanishing"]))
        c3.metric("梯度爆炸", len(issues["exploding"]))

        if issues["vanishing"]:
            st.markdown(
                '<div class="callout"><strong>梯度消失建议：</strong>'
                "使用 ReLU/GELU 替代 Sigmoid/Tanh · 添加残差连接 · 使用 He 初始化 · 添加 BatchNorm/LayerNorm</div>",
                unsafe_allow_html=True,
            )
        if issues["exploding"]:
            st.markdown(
                '<div class="callout"><strong>梯度爆炸建议：</strong>'
                "使用梯度裁剪 <code>clip_grad_norm_(params, 1.0)</code> · 降低学习率 · 使用 Xavier/He 初始化</div>",
                unsafe_allow_html=True,
            )

        return issues


# ─────────────────────────────────────────────────────────
# 训练动态可视化
# ─────────────────────────────────────────────────────────


class TrainingDynamicsVisualizer:
    """实时追踪并可视化训练动态。"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.history: dict[str, list[float]] = {}
        self.weight_snapshots: dict[int, dict[str, np.ndarray]] = {}

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def snapshot_weights(self, step: int):
        self.weight_snapshots[step] = {}
        for name, param in self.model.named_parameters():
            self.weight_snapshots[step][name] = param.detach().cpu().numpy().flatten()


# ─────────────────────────────────────────────────────────
# 页面主体
# ─────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>梯度监控与超参搜索</h1>
        <p>用可视化诊断梯度消失/爆炸，用超参搜索找到最佳训练配置。梯度是训练的体检表——先看梯度，再改模型。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 学习导读 ──
st.markdown(
    """
    <div class="note">
    <strong>学习地图：</strong>上方仪表盘实时展示各层梯度健康状态；
    下方演示区运行一个简单模型的训练循环，记录梯度并生成诊断报告。
    先观察仪表盘中正常/消失/爆炸三种状态的视觉差异，再运行训练看真实梯度流动。
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 交互式梯度健康仪表盘 ──
st.subheader("1. 梯度健康仪表盘")

col_ctrl, _ = st.columns([1, 1])
with col_ctrl:
    scenario = st.radio(
        "模拟场景",
        ["混合状态（含消失和爆炸）", "全部正常", "严重消失", "严重爆炸"],
        index=0,
        horizontal=True,
    )

SCENARIOS = {
    "混合状态（含消失和爆炸）": (
        ["Conv1", "Conv2", "Conv3", "Conv4", "FC1", "FC2", "FC3"],
        [0.85, 0.42, 0.003, 0.000001, 0.95, 120.5, 0.67],
    ),
    "全部正常": (
        ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"],
        [0.52, 0.38, 0.61, 0.44, 0.29],
    ),
    "严重消失": (
        ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"],
        [0.12, 0.005, 0.00003, 0.0000001, 0.000000001],
    ),
    "严重爆炸": (
        ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"],
        [1.2, 45.0, 320.0, 1500.0, 8900.0],
    ),
}

names, norms = SCENARIOS[scenario]
render_gradient_health_dashboard(norms, names)

st.markdown(
    """
    > **互动：** 切换上方四种模拟场景，观察仪表盘中每根柱子的高度、颜色和脉冲频率变化。
    > 绿色脉冲代表梯度在健康范围内；蓝色暗淡（几乎看不到柱子）代表梯度消失——参数几乎无法更新；
    > 红色高频闪烁代表梯度爆炸——每步更新幅度极大，训练不稳定。
    >
    > **工程经验：** 只看单个 batch 的梯度不够，至少观察最近 10~50 步的趋势。
    > loss 发散 + 梯度 max 暴涨 → 先降学习率；loss 不动 + 前层梯度接近 0 → 查初始化和激活函数。
    """,
)

# ── 实时训练梯度监控 ──
st.subheader("2. 实时训练梯度监控")

st.markdown(
    '<div class="callout">下面运行一个 4 层全连接网络的训练循环（100 步），'
    "实时记录每层梯度并生成诊断报告。训练完成后自动展示梯度健康仪表盘和诊断结果。</div>",
    unsafe_allow_html=True,
)

if st.button("▶ 运行训练并监控梯度", type="primary"):
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    X = torch.randn(200, 20)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    monitor = GradientMonitor(model)
    dynamics = TrainingDynamicsVisualizer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    progress = st.progress(0, text="训练中…")
    loss_chart = st.empty()

    loss_history: list[float] = []
    for step in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        acc = ((pred > 0.5) == y).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        monitor.record()
        optimizer.step()

        dynamics.log(loss=loss.item(), accuracy=acc)
        loss_history.append(loss.item())

        if step % 20 == 0:
            dynamics.snapshot_weights(step)

        progress.progress((step + 1) / 100, text=f"Step {step + 1}/100 — Loss: {loss.item():.4f}")

    progress.empty()

    # 训练曲线
    import plotly.graph_objects as go

    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            y=loss_history,
            mode="lines",
            name="Loss",
            line=dict(color=NEON_BLUE, width=2),
        )
    )
    fig_loss.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="训练步数",
        yaxis_title="Loss",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        yaxis_type="log",
    )
    loss_chart.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

    # 最终梯度仪表盘
    st.markdown("#### 训练结束后的梯度健康状态")
    layer_names, layer_norms = monitor.latest_norms()
    render_gradient_health_dashboard(layer_norms, layer_names)

    # 诊断报告
    monitor.report()

    # 权重演化
    if dynamics.weight_snapshots:
        st.markdown("#### 权重分布演化")
        all_layer_names = list(dynamics.weight_snapshots[list(dynamics.weight_snapshots.keys())[0]].keys())
        weight_layers = [n for n in all_layer_names if "weight" in n]
        if weight_layers:
            selected_weight_layer = st.selectbox("选择层", weight_layers)
            import plotly.graph_objects as go

            fig_w = go.Figure()
            for snap_step, snap_data in sorted(dynamics.weight_snapshots.items()):
                w = snap_data.get(selected_weight_layer, np.array([]))
                if len(w) > 0:
                    fig_w.add_trace(
                        go.Histogram(
                            x=w,
                            name=f"Step {snap_step}",
                            opacity=0.6,
                            nbinsx=40,
                        )
                    )
            fig_w.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=10, b=10),
                barmode="overlay",
                xaxis_title="权重值",
                yaxis_title="频次",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.15)",
            )
            st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    """
    > **互动：** 点击运行按钮后，观察训练过程中 loss 曲线是否平稳下降。
    > 训练结束后，梯度健康仪表盘会展示每层的最终梯度状态。
    > 如果某层显示"消失"或"爆炸"，参考诊断建议调整模型结构或超参数。
    >
    > **进阶思考：** 如果只有最后一层梯度很大，你会先查标签和 loss，还是先重写模型？
    > 如果只有前几层梯度消失，残差连接和归一化各解决什么问题？
    """,
)

# ── 学习导读 ──
st.subheader("3. 学习导读")

st.markdown(
    """
    <div class="note">
    <strong>梯度监控面板怎么看</strong><br>
    • 仪表盘中每根柱子代表一层的梯度范数，颜色和脉冲编码健康状态。<br>
    • 绿色脉冲 = 正常（1e-6 ~ 100），蓝色暗淡 = 消失（&lt; 1e-6），红色闪烁 = 爆炸（&gt; 100）。<br>
    • 只看单个 batch 不够，至少观察最近 10~50 步的趋势。

    <strong>和训练曲线怎么联合诊断</strong><br>
    • loss 发散 + 梯度 max 暴涨 → 学习率过高、标签异常或 loss 实现错误。<br>
    • loss 不动 + 前层 mean 接近 0 → 梯度消失，优先查初始化、激活函数、残差/归一化。<br>
    • loss 正常下降但某层长期异常 → 该层可能被冻结或没有接入计算图。

    <strong>工程坑案例</strong><br>
    在一个文本分类项目里只看验证准确率以为模型欠拟合，连续加大模型；
    后来梯度图显示 embedding 层长期接近 0，根因是 tokenizer 把大部分词都映射成了 UNK。
    先看梯度，可以少走很多弯路。
    </div>
    """,
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
