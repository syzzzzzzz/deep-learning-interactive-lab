"""Motion gallery page for checking and demonstrating the visual system."""

from __future__ import annotations

from components.visual_effects import render_advanced_conv_comparison
from components.visual_effects import render_attention_light_beams
from components.visual_effects import render_backprop_current_flow
from components.visual_effects import render_central_console_assembly
from components.visual_effects import render_cnn_layer_pipeline
from components.visual_effects import render_convolution_particle_flow
from components.visual_effects import render_gradient_descent_landscape
from components.visual_effects import render_gradient_monitor
from components.visual_effects import render_rnn_hidden_state_flow
from components.visual_effects import render_training_dashboard_gauges
from components.visual_effects import render_training_dynamics_panel
from components.visual_effects import render_transformer_attention_heatmap
from components.visual_primitives import render_beginner_hint
from components.visual_primitives import render_card
from components.visual_primitives import render_motion_note
from components.visual_primitives import render_neon_metric_card
from components.visual_primitives import render_responsive_motion_grid
from components.visual_primitives import render_shape_flow
from components.visual_primitives import render_status_badge
from components.visual_primitives import render_tooltip_label
from components.visual_runtime import _st
from components.visual_runtime import render_loading_bar
from components.visual_runtime import render_visual_system
from components.visual_tokens import NEON_BLUE
from components.visual_tokens import NEON_GREEN
from components.visual_tokens import NEON_PURPLE
def render_motion_gallery() -> None:
    """展示所有教学动效组件的画廊，含新增组件演示。"""
    st = _st()
    st.subheader("核心教学动效")
    render_loading_bar("页面动效服务于观察：每一种发光都对应一个可解释的学习信号")

    # ── 通用组件区 ──
    st.markdown("### 通用 UI 组件")
    st.markdown(
        render_tooltip_label("什么是教学动效？", "动效必须对应一个可解释的学习信号，例如方向、强度、权重或状态变化。"),
        unsafe_allow_html=True,
    )
    render_motion_note(
        "动效观察顺序",
        "先看数据从哪里来，再看它流向哪里；最后把颜色亮度对应到权重、梯度或激活强度。",
    )
    render_beginner_hint(
        "小白先抓三个词",
        "方向表示信息流，颜色表示强弱，停顿表示当前步骤的关键节点。",
        action="看到图时不要急着看公式，先用一句话说出“谁影响了谁”。",
    )
    c0a, c0b, c0c = st.columns(3)
    with c0a:
        render_neon_metric_card("Accuracy", "98.2%", delta="+0.5%", icon="fa-solid fa-bullseye", accent=NEON_GREEN, caption="越高不一定越好，还要看验证集。")
    with c0b:
        render_neon_metric_card("Loss", "0.042", delta="-0.008", icon="fa-solid fa-fire", accent=NEON_PURPLE, caption="下降表示优化正在找到更低误差。")
    with c0c:
        render_neon_metric_card("Epoch", "127/200", icon="fa-solid fa-rotate", accent=NEON_BLUE, caption="训练轮次只说明看过数据几遍。")
    render_responsive_motion_grid(
        [
            f'<div class="vs-card" style="padding:.9rem">{render_status_badge("运行中", status="running")}<p>用于标记正在刷新或正在训练的教学面板。</p></div>',
            f'<div class="vs-card" style="padding:.9rem">{render_status_badge("已通过", status="success")}<p>用于标记 smoke、质量门或实验完成状态。</p></div>',
            f'<div class="vs-card" style="padding:.9rem">{render_status_badge("需复查", status="warning")}<p>用于提醒参数极端、图像异常或理解薄弱点。</p></div>',
        ]
    )
    render_shape_flow(
        [
            ("输入图片", "B×1×28×28"),
            ("卷积特征", "B×16×24×24"),
            ("展平", "B×9216"),
            ("分类输出", "B×10"),
        ],
        title="CNN 张量形状流",
    )
    render_card(
        "快速入门",
        "<b>Step 1</b> 导入 PyTorch → <b>Step 2</b> 定义模型 → <b>Step 3</b> 训练循环。就这么简单。",
        icon="fa-solid fa-rocket",
        accent=NEON_GREEN,
        footer="适用于所有章节的最小可运行示例",
    )

    # ── CNN 管线 ──
    st.markdown("### CNN 层级管线")
    render_cnn_layer_pipeline()

    # ── 梯度监控 ──
    st.markdown("### 梯度监控仪表盘")
    render_gradient_monitor("all")

    # ── 训练动态面板 ──
    st.markdown("### 训练动态监控面板")
    render_training_dynamics_panel()

    # ── 高级卷积对比 ──
    st.markdown("### 高级卷积对比")
    render_advanced_conv_comparison()

    # ── RNN 隐藏状态传递 ──
    st.markdown("### RNN 隐藏状态传递")
    render_rnn_hidden_state_flow()

    # ── 中央控制台模型拼装 ──
    st.markdown("### 中央控制台模型拼装")
    render_central_console_assembly()

    # ── Transformer 注意力热力图 ──
    st.markdown("### Transformer 注意力热力图")
    render_transformer_attention_heatmap()

    # ── 原有动效 ──
    st.markdown("### 交互式教学动效")
    c1, c2 = st.columns(2)
    with c1:
        render_convolution_particle_flow()
        render_attention_light_beams(["I", "love", "deep", "learning", "because", "it", "works"], 3)
    with c2:
        render_gradient_descent_landscape()
        render_training_dashboard_gauges()
    render_backprop_current_flow()
