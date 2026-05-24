"""
CS interview page: deep learning.

Run:
    streamlit run part7_interview/deep_learning_interview.py
or:
    python main.py part7/deep_learning_interview
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


T = TypeVar("T")

st.set_page_config(page_title="深度学习面试训练", layout="wide", initial_sidebar_state="expanded")


def css() -> str:
    return """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1180px; }
    .stApp { background: #f7f8f4; color: #172026; }
    h1, h2, h3, p, li, label, span { letter-spacing: 0; }
    .hero { border-bottom: 1px solid #d8dee3; padding-bottom: 1rem; margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: clamp(2rem, 3vw, 3.1rem); line-height: 1.1; }
    .hero p { color: #596772; line-height: 1.7; max-width: 920px; }
    .note { border-left: 4px solid #0f8b8d; background: rgba(255,255,255,.78); border-radius: 0 8px 8px 0; padding: .74rem .9rem; line-height: 1.68; margin: .4rem 0 .9rem; }
    .step { background: rgba(255,255,255,.82); border: 1px solid #d8dee3; border-radius: 8px; padding: .72rem .82rem; min-height: 104px; line-height: 1.55; }
    .arrow { text-align: center; font-weight: 800; color: #0f8b8d; padding-top: 2.25rem; }
    .flow { background: #172026; color: #f7fbfc; border-radius: 8px; padding: .82rem 1rem; font-family: Consolas, "Courier New", monospace; line-height: 1.72; white-space: pre-wrap; }
    .small { color: #596772; font-size: .92rem; line-height: 1.58; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    .comparison-card { background: rgba(255,255,255,.82); border: 1px solid #d8dee3; border-radius: 8px; padding: .82rem; }
    .comparison-card h4 { margin-top: 0; color: #0f8b8d; }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("深度学习面试页面执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="dl-back-home", use_container_width=True):
        st.query_params.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Interactive components
# ---------------------------------------------------------------------------


def gradient_vanish_explode_demo() -> None:
    """Interactive demo: gradient magnitude across layers."""
    st.subheader("梯度消失与梯度爆炸可视化")
    st.markdown("调节激活函数和网络深度，观察反向传播时梯度如何逐层变化。")

    col1, col2 = st.columns(2)
    with col1:
        activation = st.selectbox(
            "激活函数",
            ["Sigmoid", "Tanh", "ReLU", "LeakyReLU"],
            key="dl-grad-activation",
        )
    with col2:
        depth = st.slider("网络层数", 4, 20, 8, key="dl-grad-depth")

    # Simulate gradient magnitude across layers
    import math

    layers = list(range(1, depth + 1))
    grads = []
    base = 1.0
    for i in range(depth):
        if activation == "Sigmoid":
            # sigmoid derivative max = 0.25, gradients shrink fast
            factor = 0.25
        elif activation == "Tanh":
            # tanh derivative max = 1.0, but typically ~0.7 in practice
            factor = 0.7
        elif activation == "ReLU":
            # ReLU: gradient is 0 or 1, assume 50% active
            factor = 0.85
        else:  # LeakyReLU
            factor = 0.92
        base *= factor
        grads.append(base)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=grads, mode="lines+markers",
        line=dict(color="#0f8b8d", width=3),
        marker=dict(size=7),
        name="梯度范数",
    ))
    fig.update_layout(
        xaxis_title="层数（从输出层反向）",
        yaxis_title="梯度相对大小",
        yaxis_type="log",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if activation in ("Sigmoid", "Tanh"):
        st.warning(f"**{activation}** 的导数最大值 {'≤0.25' if activation == 'Sigmoid' else '＜1'}，梯度逐层连乘后指数衰减，深层参数几乎无法更新。这就是**梯度消失**。")
    elif activation == "ReLU":
        st.info("**ReLU** 正区间梯度恒为 1，不会消失；但负区间梯度为 0，可能导致「神经元死亡」。搭配合适初始化和归一化是标准做法。")
    else:
        st.success("**LeakyReLU** 正区间梯度为 1，负区间也有小梯度（通常 0.01），兼顾不消失和不死亡。")


def normalization_comparison() -> None:
    """BatchNorm vs LayerNorm interactive comparison."""
    st.subheader("BatchNorm vs LayerNorm 对比")
    st.markdown("选择场景，观察两种归一化的统计维度差异。")

    scenario = st.selectbox(
        "选择场景",
        ["CV 图像分类 (batch=32, C=64, H=W=32)",
         "NLP 文本分类 (batch=16, seq_len=128, d=768)",
         "小 batch 训练 (batch=2, C=256)",
         "推理阶段 (batch=1)"],
        key="dl-norm-scenario",
    )

    if "CV" in scenario:
        st.success("**BatchNorm 更合适**：同一 batch 内同一通道的统计量稳定（32×32×32=32768 个样本点），训练和推理的 running mean/var 差异小。")
        st.code("BN: 统计维度 = (N, H, W) per channel\nLN: 统计维度 = (C, H, W) per sample", language="text")
    elif "NLP" in scenario:
        st.success("**LayerNorm 更合适**：序列长度和 padding 不固定，batch 维度统计不稳定；LN 对单样本的特征维度归一化，与 batch 大小无关。")
        st.code("BN: 统计维度 = (N, seq_len) per feature  ← 不稳定\nLN: 统计维度 = (d_model) per token    ← 稳定", language="text")
    elif "小 batch" in scenario:
        st.warning("**BatchNorm 在小 batch 下不稳定**：仅 2 个样本算均值方差，方差估计噪声大。应改用 GroupNorm 或 LayerNorm。")
    else:
        st.info("**推理时 batch=1**：BatchNorm 的 running mean/var 是固定的，单样本无问题；但如果你的模型需要在 batch=1 时行为一致，LayerNorm 更自然。")

    # Visual comparison table
    st.table(pd.DataFrame([
        ["统计维度", "跨 batch 的 (H,W) 或 (seq_len)", "单样本的特征维度"],
        ["依赖 batch", "是", "否"],
        ["适合场景", "CV、大 batch", "NLP、Transformer、小 batch"],
        ["推理行为", "用 running mean/var", "与训练一致"],
        ["代表模型", "ResNet, EfficientNet", "BERT, GPT, ViT"],
    ], columns=["维度", "BatchNorm", "LayerNorm"]))


def attention_complexity_demo() -> None:
    """Visualize O(n²) attention cost."""
    st.subheader("注意力复杂度 O(n²) 直觉")
    st.markdown("拖动序列长度，观察注意力矩阵大小和计算量如何增长。")

    seq_len = st.slider("序列长度 n", 16, 2048, 128, step=16, key="dl-attn-seq")

    # Attention matrix size
    matrix_elements = seq_len * seq_len
    # Rough FLOPs: 2 * n^2 * d (QK^T) + 2 * n^2 * d (score * V), assume d=64
    d = 64
    flops_qk = 2 * seq_len * seq_len * d
    flops_sv = 2 * seq_len * seq_len * d
    total_flops = flops_qk + flops_sv

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("注意力矩阵大小", f"{seq_len}×{seq_len} = {matrix_elements:,}")
    with col2:
        st.metric("QK^T 计算量", f"{flops_qk/1e6:.1f} MFLOPs")
    with col3:
        st.metric("总注意力计算量", f"{total_flops/1e6:.1f} MFLOPs")

    # Growth curve
    seqs = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    costs = [2 * s * s * d * 2 / 1e6 for s in seqs]
    colors = ["#0f8b8d" if s <= seq_len else "#d8dee3" for s in seqs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(s) for s in seqs], y=costs, marker_color=colors))
    fig.update_layout(
        xaxis_title="序列长度",
        yaxis_title="注意力计算量 (MFLOPs, d=64)",
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="note">
    <strong>为什么是 O(n²)?</strong> 自注意力需要让每个 token 与所有其他 token 计算相关性，
    形成 n×n 的注意力矩阵。当序列长度翻倍，矩阵元素变为 4 倍。
    这就是为什么长上下文（4K→32K→128K）是工程难题：不只显存翻倍，计算也翻倍。
    Flash Attention、稀疏注意力和线性注意力都是为了解决这个问题。
    </div>
    """, unsafe_allow_html=True)


def training_debug_checklist() -> None:
    """Interactive training debugging checklist."""
    st.subheader("训练 Loss 不下降排查清单")
    st.markdown("勾选你已经排查过的步骤，系统帮你追踪排查进度。")

    checklist = [
        ("数据与标签", "确认数据加载正确、标签对齐、无全零或乱码"),
        ("小数据集过拟合", "用极小数据集（几十条）验证模型能否 loss→0"),
        ("前向传播检查", "打印中间张量 shape 和值域，确认无 NaN/Inf"),
        ("损失函数", "确认损失函数与任务匹配（分类用 CE，回归用 MSE）"),
        ("学习率", "尝试不同学习率（1e-5 ~ 1e-1），观察 loss 曲线"),
        ("梯度检查", "打印梯度范数，确认不为 0、不为 NaN、不爆炸"),
        ("参数更新", "确认 optimizer.step() 被调用、梯度未被 detach"),
        ("初始化", "检查权重初始化是否合理（Xavier/He/Kaiming）"),
        ("归一化", "检查 BatchNorm/LayerNorm 是否正确使用"),
        ("硬件与精度", "确认 GPU 可用、混合精度下 loss scaling 正常"),
    ]

    checked = 0
    for i, (title, desc) in enumerate(checklist):
        if st.checkbox(f"**{title}**：{desc}", key=f"dl-debug-{i}"):
            checked += 1

    progress = checked / len(checklist)
    st.progress(progress, text=f"排查进度：{checked}/{len(checklist)}")
    if checked == len(checklist):
        st.success("🎉 所有排查项都已完成！如果问题仍未解决，考虑：降低学习率到极小值、换用更简单的模型架构、检查数据泄漏。")
    elif checked >= 7:
        st.info("已排查大部分关键项。重点关注「小数据集过拟合」——如果连小数据都 loss 不降，问题大概率在模型或代码。")
    elif checked >= 4:
        st.warning("继续排查。最常见的原因是学习率不合适或数据/标签有 bug。")


def loss_landscape_viz() -> None:
    """Visualize learning rate impact on training."""
    st.subheader("学习率对训练的影响")
    st.markdown("选择学习率范围，观察典型训练 loss 曲线形态。")

    lr = st.select_slider(
        "学习率",
        options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        value=1e-3,
        key="dl-lr-select",
    )

    import math

    epochs = list(range(1, 51))
    # Simulate loss curves for different LR regimes
    if lr <= 1e-4:
        # Too small: slow convergence
        losses = [2.3 * math.exp(-0.02 * lr * 1e4 * e) + 0.5 for e in epochs]
        label = "过小：收敛极慢，50 epoch 还没到好结果"
        color = "#e6a817"
    elif lr <= 1e-3:
        # Good range
        losses = [2.3 * math.exp(-0.15 * e) + 0.05 + 0.02 * math.sin(e * 0.3) for e in epochs]
        label = "合适：快速收敛到低 loss"
        color = "#0f8b8d"
    elif lr <= 1e-2:
        # Slightly large: oscillation
        losses = [2.3 * math.exp(-0.1 * e) + 0.15 + 0.12 * math.sin(e * 0.8) for e in epochs]
        label = "偏大：收敛但震荡明显"
        color = "#e67e22"
    else:
        # Too large: loss explodes
        losses = [2.3 + 0.5 * e * (1 + 0.3 * math.sin(e)) for e in epochs]
        label = "过大：loss 不降反升，训练发散"
        color = "#e74c3c"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=losses, mode="lines",
        line=dict(color=color, width=2.5),
        name=label,
    ))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=350,
        yaxis_range=[0, max(losses) * 1.1 + 0.1],
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**当前学习率 {lr:.0e}**：{label}")


def model_deployment_pipeline() -> None:
    """Show model deployment optimization techniques."""
    st.subheader("模型部署优化手段速查")

    techniques = [
        ("量化 (Quantization)", "FP32→INT8/INT4，显存↓4-8×，速度↑2-4×", "精度可能略降，需要校准数据"),
        ("知识蒸馏 (Distillation)", "大模型→小模型，保留泛化能力", "需要教师模型和训练资源"),
        ("剪枝 (Pruning)", "移除不重要的权重/通道", "非结构化剪枝需要硬件支持"),
        ("算子融合 (Fusion)", "合并相邻算子减少内存搬运", "需要编译器支持（TensorRT/ONNX）"),
        ("KV Cache", "缓存已计算的 K/V，避免重复计算", "增加显存占用，需要管理"),
        ("动态 Batching", "合并多个请求一起推理", "增加首请求延迟，需要请求队列"),
        ("Flash Attention", "IO-aware 注意力计算，减少 HBM 访问", "需要特定硬件和 CUDA 版本"),
        ("LoRA 热切换", "共享基座模型，按需加载小矩阵", "多 LoRA 并发需要显存管理"),
    ]

    df = pd.DataFrame(techniques, columns=["技术", "效果", "注意事项"])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# High-frequency Q&A
# ---------------------------------------------------------------------------


def faq_section() -> None:
    st.subheader("高频问答区")
    faqs = [
        ("过拟合怎么缓解？",
         "更多数据、数据增强、正则化（L1/L2）、Dropout、早停（Early Stopping）、降低模型复杂度、标签平滑。"
         "面试中要区分「过拟合」和「数据分布漂移」：过拟合是训练/验证 gap，漂移是线上数据与训练数据分布不同。"),
        ("BatchNorm 和 LayerNorm 区别？",
         "BatchNorm 跨 batch 统计（N,H,W）per channel，依赖 batch 大小；LayerNorm 单样本统计（C,H,W）或（d_model）per token，与 batch 无关。"
         "CV 用 BN 更稳（batch 大），NLP/Transformer 用 LN（序列不固定、batch 小）。"),
        ("Transformer 注意力为什么 O(n²)?",
         "自注意力需要每个 token 与所有 token 计算相关性，形成 n×n 矩阵。序列翻倍→矩阵 4 倍。"
         "Flash Attention 不改变复杂度但减少 HBM 访问；稀疏/线性注意力把复杂度降到 O(n√n) 或 O(n)。"),
        ("梯度消失和梯度爆炸？",
         "消失：反向传播连乘小梯度，前层几乎不更新（Sigmoid/Tanh）。爆炸：连乘大梯度，更新不稳定。"
         "解法：合适初始化（Xavier/He）、归一化、残差连接、门控（LSTM/GRU）、梯度裁剪、学习率调整。"),
        ("混合精度训练？",
         "FP16/BF16 降低显存和带宽，利用张量核心加速。风险：数值下溢/溢出。"
         "解法：loss scaling（FP16 需要，BF16 通常不需要）、保留 FP32 主权重、动态损失缩放。"),
        ("LoRA 微调原理？",
         "在预训练权重 W 旁插入低秩矩阵 A(d×r) 和 B(r×d)，ΔW=BA，只训练 A、B。"
         "r 远小于 d（通常 8-64），可训练参数量↓100×+。不同任务换不同 LoRA 矩阵，共享大模型权重。"),
    ]
    for question, answer in faqs:
        with st.expander(question):
            st.write(answer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>🧠 深度学习面试训练</h1>
          <p>覆盖过拟合、归一化、注意力机制、梯度问题、混合精度、LoRA 微调、模型部署等高频考点，
          每个概念配有交互可视化帮你建立直觉，而不是死记硬背。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gradient_vanish_explode_demo()
    st.divider()

    normalization_comparison()
    st.divider()

    attention_complexity_demo()
    st.divider()

    loss_landscape_viz()
    st.divider()

    training_debug_checklist()
    st.divider()

    model_deployment_pipeline()
    st.divider()

    faq_section()

    st.divider()
    st.markdown(
        """
        <div class="note">
        深度学习面试不只考概念，更考「能不能把原理和工程场景串起来」。
        上面每个可视化都对应面试追问：比如调激活函数时梯度怎么变？小 batch 用什么归一化？
        序列长度翻倍对显存和计算的影响？把直觉练成条件反射，面试就不慌。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("进入深度学习专项刷题", "/?module=part7%2Finterview_quiz", width="stretch")
    render_back_home()


if __name__ == "__main__":
    safe_run(main)
