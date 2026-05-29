"""
Model compression, deployment formats, and deep learning toolbox overview.

Run:
    streamlit run part5_toolbox/deployment_tools.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="模型压缩部署与工具框架",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    .stApp { background: #f8f9f6; color: #172027; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #d7ded8;
        border-radius: 8px;
        padding: 10px 12px;
    }
    .note {
        border-left: 4px solid #207f7a;
        background: rgba(255,255,255,0.76);
        border-radius: 0 8px 8px 0;
        padding: 0.76rem 0.92rem;
        line-height: 1.68;
        margin: 0.35rem 0 0.95rem 0;
    }
    .mini-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid #d7ded8;
        border-radius: 8px;
        padding: 0.82rem 0.9rem;
        min-height: 8.4rem;
        line-height: 1.58;
    }
    .small {
        color: #59656a;
        font-size: 0.92rem;
        line-height: 1.58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PALETTE = {
    "base": "#59656a",
    "prune": "#207f7a",
    "quant": "#c47f1f",
    "distill": "#4f65b0",
    "combo": "#b84d5a",
}


@dataclass(frozen=True)
class CompressionResult:
    method: str
    accuracy: float
    model_size_mb: float
    latency_ms: float
    active_params_m: float
    note: str


def render_note(title: str, body: str) -> None:
    st.markdown(
        f'<div class="note"><strong>{title}</strong> {body}</div>',
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def compression_results(
    pruning_ratio: float,
    quant_bits: int,
    student_width: float,
    distill_quality: float,
) -> list[CompressionResult]:
    base_acc = 92.0
    base_params_m = 12.8
    base_size_mb = base_params_m * 4.0
    base_latency_ms = 18.0

    prune_active = base_params_m * (1.0 - pruning_ratio)
    prune_size = base_size_mb * (1.0 - 0.84 * pruning_ratio)
    prune_latency = base_latency_ms * max(0.38, 1.0 - 0.58 * pruning_ratio)
    prune_acc = base_acc - 2.15 * pruning_ratio**1.35

    quant_size = base_size_mb * quant_bits / 32
    quant_latency_factor = {16: 0.78, 8: 0.56, 4: 0.39}[quant_bits]
    quant_drop = {16: 0.15, 8: 0.55, 4: 2.05}[quant_bits]
    quant_acc = base_acc - quant_drop

    student_scale = student_width**2
    student_params = base_params_m * student_scale
    student_size = base_size_mb * student_scale
    student_latency = base_latency_ms * max(0.23, 0.18 + 0.78 * student_scale)
    plain_student_drop = 6.4 * (1.0 - student_width) ** 1.25
    recovered = plain_student_drop * distill_quality
    distill_acc = base_acc - plain_student_drop + recovered - 0.25

    combo_params = student_params * (1.0 - pruning_ratio)
    combo_size = student_size * (quant_bits / 32) * (1.0 - 0.64 * pruning_ratio)
    combo_latency = student_latency * quant_latency_factor * max(0.5, 1.0 - 0.42 * pruning_ratio)
    combo_drop = (
        plain_student_drop * (1.0 - distill_quality)
        + quant_drop * 0.72
        + 1.55 * pruning_ratio**1.4
        + 0.35
    )
    combo_acc = base_acc - combo_drop

    return [
        CompressionResult(
            "基线 FP32",
            base_acc,
            base_size_mb,
            base_latency_ms,
            base_params_m,
            "未压缩模型，精度和兼容性最好。",
        ),
        CompressionResult(
            "剪枝",
            prune_acc,
            prune_size,
            prune_latency,
            prune_active,
            "删除低重要性权重或通道，结构化剪枝更容易带来真实加速。",
        ),
        CompressionResult(
            f"{quant_bits}-bit 量化",
            quant_acc,
            quant_size,
            base_latency_ms * quant_latency_factor,
            base_params_m,
            "用低比特整数或半精度表示权重和激活，常见部署收益来自 INT8。",
        ),
        CompressionResult(
            "知识蒸馏",
            distill_acc,
            student_size,
            student_latency,
            student_params,
            "用大教师模型的软标签训练小学生模型，保留暗知识和类别相似性。",
        ),
        CompressionResult(
            "组合方案",
            combo_acc,
            combo_size,
            combo_latency,
            combo_params,
            "先训练学生模型，再剪枝、量化、校准，通常是边缘部署的实用路线。",
        ),
    ]


def compression_dataframe(results: list[CompressionResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "方法": [r.method for r in results],
            "Top-1 精度": [format_percent(r.accuracy) for r in results],
            "模型大小": [f"{r.model_size_mb:.1f} MB" for r in results],
            "推理延迟": [f"{r.latency_ms:.1f} ms" for r in results],
            "有效参数": [f"{r.active_params_m:.2f} M" for r in results],
            "说明": [r.note for r in results],
        }
    )


def plot_compression(results: list[CompressionResult]) -> go.Figure:
    methods = [r.method for r in results]
    colors = [PALETTE["base"], PALETTE["prune"], PALETTE["quant"], PALETTE["distill"], PALETTE["combo"]]
    fig = go.Figure()
    fig.add_bar(
        x=methods,
        y=[r.accuracy for r in results],
        name="Top-1 精度 (%)",
        marker_color=colors,
        yaxis="y",
        text=[f"{r.accuracy:.1f}%" for r in results],
        textposition="outside",
    )
    fig.add_scatter(
        x=methods,
        y=[r.model_size_mb for r in results],
        name="模型大小 (MB)",
        mode="lines+markers",
        yaxis="y2",
        line={"color": "#172027", "width": 2},
        marker={"size": 9},
    )
    fig.update_layout(
        height=430,
        margin={"l": 24, "r": 24, "t": 34, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        legend={"orientation": "h", "y": 1.15, "x": 0},
        yaxis={"title": "精度 (%)", "range": [max(0, min(r.accuracy for r in results) - 4), 96]},
        yaxis2={
            "title": "模型大小 (MB)",
            "overlaying": "y",
            "side": "right",
            "range": [0, max(r.model_size_mb for r in results) * 1.22],
        },
        xaxis={"title": ""},
    )
    return fig


def render_compression() -> None:
    st.subheader("1. 模型压缩技术演示")
    render_note(
        "核心直觉：",
        "剪枝减少计算图里的冗余连接，量化降低每个数值的存储和计算成本，知识蒸馏用大模型指导小模型学习。真实项目里通常组合使用，而不是三选一。",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pruning_ratio = st.slider("剪枝比例", 0.0, 0.85, 0.45, 0.05)
    with c2:
        quant_bits = st.select_slider("量化位宽", options=[16, 8, 4], value=8)
    with c3:
        student_width = st.slider("学生模型宽度", 0.35, 1.0, 0.62, 0.01)
    with c4:
        distill_quality = st.slider("蒸馏恢复能力", 0.0, 0.9, 0.62, 0.02)

    results = compression_results(pruning_ratio, int(quant_bits), student_width, distill_quality)
    base = results[0]
    combo = results[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("组合方案大小", f"{combo.model_size_mb:.1f} MB", f"{combo.model_size_mb / base.model_size_mb:.0%} of FP32")
    m2.metric("组合方案延迟", f"{combo.latency_ms:.1f} ms", f"{combo.latency_ms / base.latency_ms:.0%} of FP32")
    m3.metric("精度变化", f"{combo.accuracy:.1f}%", f"{combo.accuracy - base.accuracy:.1f} pts")
    m4.metric("有效参数", f"{combo.active_params_m:.2f} M", f"{combo.active_params_m / base.active_params_m:.0%} of base")

    st.plotly_chart(plot_compression(results), width="stretch")
    st.dataframe(compression_dataframe(results), width="stretch", hide_index=True)

    with st.expander("三种压缩技术的工程要点", expanded=True):
        st.markdown(
            """
            - **剪枝**：非结构化剪枝能让权重矩阵更稀疏，但不一定自动加速；通道剪枝、层剪枝、注意力头剪枝更容易被普通推理引擎利用。
            - **量化**：训练后量化部署最快，量化感知训练更稳；INT8 常见，INT4 更依赖模型结构、校准数据和硬件支持。
            - **知识蒸馏**：教师模型输出的概率分布比硬标签更有信息量；常和小模型架构搜索、剪枝、量化配合使用。
            """
        )


def render_deployment_formats() -> None:
    st.subheader("2. 模型部署格式说明")
    render_note(
        "选择原则：",
        "先看目标运行时和硬件，再选导出格式。格式不是越底层越好，越底层通常意味着更强性能、更弱可移植性和更多构建约束。",
    )

    df = pd.DataFrame(
        [
            {
                "格式": "ONNX",
                "定位": "跨框架中间表示",
                "典型场景": "PyTorch/TensorFlow 训练后交给 ONNX Runtime、OpenVINO、TensorRT 等后端",
                "优势": "生态广、运行时多、便于模型交换和图优化",
                "限制": "动态 shape、控制流、自定义算子可能需要额外处理",
            },
            {
                "格式": "TensorRT",
                "定位": "NVIDIA GPU 高性能推理引擎",
                "典型场景": "服务端 GPU、Jetson、低延迟视觉和生成式模型推理",
                "优势": "层融合、kernel 选择、FP16/INT8 校准，吞吐和延迟通常最好",
                "限制": "强绑定 NVIDIA 硬件和 CUDA 版本，engine 往往不可跨机器通用",
            },
            {
                "格式": "TorchScript",
                "定位": "PyTorch 模型序列化和运行时表示",
                "典型场景": "保留 PyTorch 生态的服务端或 C++ 部署",
                "优势": "接近 PyTorch 语义，便于从训练代码过渡到部署代码",
                "限制": "跨框架能力弱，部分 Python 动态逻辑需要改写；新项目也会评估 torch.export",
            },
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**常见导出路线**")
        st.code(
            """
# PyTorch -> ONNX
torch.onnx.export(model, example_input, "model.onnx", opset_version=17)

# PyTorch -> TorchScript
scripted = torch.jit.trace(model, example_input)
scripted.save("model.pt")

# ONNX -> TensorRT
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16
            """.strip(),
            language="python",
        )
    with right:
        st.markdown("**选型提示**")
        st.markdown(
            """
            - 需要跨平台交付：优先 ONNX。
            - 已确定 NVIDIA GPU 并追求极限延迟：优先 TensorRT。
            - 服务端仍以 PyTorch 为主且模型包含 PyTorch 特有逻辑：考虑 TorchScript 或 PyTorch 2 导出链路。
            - 有自定义算子：尽早做最小模型导出验证，不要等训练结束才发现部署不可落地。
            """
        )


def render_edge_deployment() -> None:
    st.subheader("3. 边缘部署的概念和挑战")
    render_note(
        "定义：",
        "边缘部署是在手机、摄像头、工业网关、车载设备、Jetson、NPU 开发板等靠近数据源的位置运行模型，目标是减少云端依赖、降低延迟并保护数据。",
    )

    cols = st.columns(4)
    cards = [
        ("算力", "CPU、GPU、NPU、DSP 能力差异很大，同一个模型在不同芯片上瓶颈可能完全不同。"),
        ("内存", "权重、激活、输入缓存和运行时都占内存，batch size 通常很小，峰值内存比平均值更关键。"),
        ("功耗", "持续推理会带来发热和降频，移动端需要关注每帧能耗，而不只是毫秒延迟。"),
        ("维护", "离线设备更新慢，数据漂移、版本回滚、日志回传和安全补丁都要提前设计。"),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="mini-card"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)

    st.markdown("**边缘落地流程**")
    flow = pd.DataFrame(
        [
            ["1. 确定约束", "延迟、功耗、内存、输入分辨率、离线需求、目标芯片"],
            ["2. 训练可部署模型", "尽量使用部署引擎支持良好的算子和固定输入形状"],
            ["3. 压缩与校准", "剪枝、蒸馏、INT8/FP16，使用真实校准集"],
            ["4. 导出与编译", "ONNX/TFLite/TensorRT/CoreML/厂商 SDK"],
            ["5. 端侧验证", "测端到端延迟、峰值内存、温度、稳定性和异常输入"],
            ["6. 监控迭代", "采集匿名指标、灰度升级、保留回滚版本"],
        ],
        columns=["阶段", "关注点"],
    )
    st.dataframe(flow, width="stretch", hide_index=True)


def render_frameworks() -> None:
    st.subheader("4. 深度学习框架对比：PyTorch vs TensorFlow/Keras")
    render_note(
        "工程判断：",
        "PyTorch 常在研究、调试和自定义训练循环里更顺手；TensorFlow/Keras 在移动端、浏览器端和某些成熟生产链路里仍有优势。团队已有资产通常比框架口味更重要。",
    )

    df = pd.DataFrame(
        [
            ["开发体验", "Pythonic、动态图直观，调试接近普通 Python", "Keras 高层 API 简洁，TensorFlow 底层能力完整"],
            ["研究生态", "论文复现、开源模型、Hugging Face 生态非常活跃", "学术和工业都有积累，但新研究代码 PyTorch 占比更高"],
            ["生产部署", "TorchServe、ONNX、TorchScript、torch.compile/export 生态持续增强", "TensorFlow Serving、TFLite、TF.js 链路成熟"],
            ["移动与端侧", "可用 PyTorch Mobile、ExecuTorch、ONNX Runtime 等路线", "TFLite、NNAPI、Core ML 转换链路常见"],
            ["调试方式", "直接断点、打印 tensor、逐行定位更自然", "图模式性能好，但复杂图调试成本更高"],
            ["适合用户", "需要快速试验、自定义模型、紧跟开源模型的团队", "重视端侧标准链路、Keras 快速建模、已有 TF 生产资产的团队"],
        ],
        columns=["维度", "PyTorch", "TensorFlow / Keras"],
    )
    st.dataframe(df, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**更偏 PyTorch 的信号**")
        st.markdown(
            """
            - 主要任务是研究、实验、快速改网络结构。
            - 依赖 Transformers、Diffusers、PyTorch Geometric 等生态。
            - 需要复杂自定义 loss、训练循环或调试中间张量。
            """
        )
    with right:
        st.markdown("**更偏 TensorFlow/Keras 的信号**")
        st.markdown(
            """
            - 已经有 TensorFlow Serving、TFLite 或 TF.js 生产链路。
            - 端侧部署是第一优先级，模型结构相对标准。
            - 团队希望用 Keras 快速搭建常规模型。
            """
        )


def render_visualization_tools() -> None:
    st.subheader("5. 可视化工具介绍")
    render_note(
        "工具分工：",
        "TensorBoard 适合记录训练过程，Visdom 适合快速交互式实验看板，Netron 适合检查模型结构、输入输出和算子是否符合部署预期。",
    )

    df = pd.DataFrame(
        [
            {
                "工具": "TensorBoard",
                "看什么": "loss、accuracy、学习率、直方图、图像、embedding、计算图",
                "优势": "和 PyTorch/TensorFlow 都能配合，训练记录标准化",
                "典型命令": "tensorboard --logdir runs",
            },
            {
                "工具": "Visdom",
                "看什么": "实时曲线、图片、文本、实验面板",
                "优势": "轻量灵活，适合研究阶段快速搭面板",
                "典型命令": "python -m visdom.server",
            },
            {
                "工具": "Netron",
                "看什么": "ONNX、TorchScript、TensorFlow、TFLite 等模型图",
                "优势": "部署前检查 shape、算子、权重和图结构很方便",
                "典型命令": "netron model.onnx",
            },
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)

    tab1, tab2, tab3 = st.tabs(["TensorBoard", "Visdom", "Netron"])
    with tab1:
        st.code(
            """
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp01")
for step, (loss, acc) in enumerate(history):
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("val/acc", acc, step)
writer.close()
            """.strip(),
            language="python",
        )
    with tab2:
        st.code(
            """
import visdom

viz = visdom.Visdom()
viz.line(Y=[0.9], X=[0], win="loss", opts={"title": "training loss"})
viz.line(Y=[0.7], X=[1], win="loss", update="append")
            """.strip(),
            language="python",
        )
    with tab3:
        st.markdown(
            """
            Netron 不负责训练，也不负责推理。它最常见的价值是在部署前回答这些问题：

            - 输入输出 shape 是否符合运行时约定。
            - 模型里是否出现部署引擎不支持的算子。
            - 融合、量化或导出后图结构是否和预期一致。
            """
        )


def render_datasets() -> None:
    st.subheader("6. 常用数据集介绍")
    render_note(
        "使用提醒：",
        "经典数据集适合建立基线和教学，但不代表真实业务难度。选择数据集时要同时看任务类型、规模、标注质量、评价指标和许可证。",
    )

    df = pd.DataFrame(
        [
            ["MNIST", "手写数字分类", "70,000 张", "28x28 灰度图", "10 类", "入门和调试训练流程"],
            ["CIFAR-10", "小图像分类", "60,000 张", "32x32 RGB", "10 类", "CNN、数据增强、正则化教学"],
            ["CIFAR-100", "细粒度小图像分类", "60,000 张", "32x32 RGB", "100 类", "更难的分类基线和迁移学习练习"],
            ["ImageNet", "大规模图像分类", "约 1.28M 训练图像", "多尺寸 RGB", "1,000 类", "预训练骨干网络和迁移学习"],
            ["COCO", "检测、分割、关键点、字幕", "2017 train 约 118K，val 约 5K", "自然场景 RGB", "80 个检测类别", "目标检测、实例分割和多任务视觉"],
        ],
        columns=["数据集", "主要任务", "规模", "输入形式", "类别", "常见用途"],
    )
    st.dataframe(df, width="stretch", hide_index=True)

    selected = st.selectbox("查看数据集要点", df["数据集"].tolist(), index=1)
    details = {
        "MNIST": "太简单，模型很快接近饱和。它适合检查代码是否跑通，不适合证明模型真的强。",
        "CIFAR-10": "分辨率低但类别直观，适合演示卷积、数据增强、BatchNorm、Dropout 和学习率调度。",
        "CIFAR-100": "每类样本更少，类别更细，能更明显暴露过拟合和表示能力不足。",
        "ImageNet": "常用作视觉预训练基准。训练成本高，很多项目直接使用 ImageNet 预训练权重。",
        "COCO": "比分类数据集更接近真实场景，包含多目标、小目标、遮挡和复杂标注，评价指标也更复杂。",
    }
    st.info(details[selected])


def main() -> None:
    st.title("模型压缩部署与工具框架")
    st.markdown(
        "把模型从训练代码带到真实环境，需要同时理解压缩方法、导出格式、端侧约束、框架生态、可视化工具和常用数据集。"
    )

    with st.sidebar:
        st.header("主题导航")
        section = st.radio(
            "选择模块",
            [
                "模型压缩技术",
                "部署格式对比",
                "边缘部署",
                "框架对比",
                "可视化工具",
                "常用数据集",
            ],
        )
        st.caption("压缩页里的数字是趋势演示，不替代真实硬件 benchmark。")

    if section == "模型压缩技术":
        render_compression()
    elif section == "部署格式对比":
        render_deployment_formats()
    elif section == "边缘部署":
        render_edge_deployment()
    elif section == "框架对比":
        render_frameworks()
    elif section == "可视化工具":
        render_visualization_tools()
    else:
        render_datasets()


if __name__ == "__main__":
    main()


render = main


def compute(seed: int = 42) -> dict[str, object]:
    """Pure computation placeholder."""
    return {"status": "ok", "seed": seed}


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""
    return True
