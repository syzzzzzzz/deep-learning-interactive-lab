"""
Classic paper reading lab for deep learning learners.

Run:
    streamlit run part6_universal_framework/paper_reading_lab.py
or:
    python main.py part6/paper_reading_lab
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

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
class Paper:
    key: str
    title: str
    year: int
    family: str
    problem: str
    core_idea: str
    mechanism: tuple[str, ...]
    impact: str
    read_first: tuple[str, ...]
    implement: tuple[str, ...]
    common_trap: str
    prereq: tuple[str, ...]
    x: float
    y: float
    size: int


PAPERS: tuple[Paper, ...] = (
    Paper(
        "lenet",
        "LeNet-5",
        1998,
        "CNN",
        "手写数字识别需要摆脱手工特征。",
        "把局部连接、参数共享和池化组合成端到端视觉模型。",
        ("卷积核扫描局部区域", "池化降低位移敏感性", "分类头把空间特征变成类别"),
        "奠定了现代 CNN 的基本构件。",
        ("卷积与池化结构", "端到端训练方式", "为什么先局部再全局"),
        ("实现一个 2 层 CNN", "观察中间特征图", "比较有无池化"),
        "不要只记结构层数，重点是参数共享如何降低样本需求。",
        ("神经网络基础", "卷积直觉"),
        0.12,
        0.25,
        22,
    ),
    Paper(
        "alexnet",
        "AlexNet",
        2012,
        "CNN",
        "ImageNet 级别视觉识别需要更大模型和更强训练技巧。",
        "用深 CNN、GPU 训练、ReLU、Dropout 和数据增强扩大可训练模型规模。",
        ("ReLU 缓解饱和", "Dropout 抑制共适应", "数据增强提高泛化", "GPU 支撑更大批量计算"),
        "让深度学习成为主流视觉方法。",
        ("训练技巧清单", "错误率变化", "为什么 ReLU 比 sigmoid 更适合深层网络"),
        ("复现小型 AlexNet", "开关 dropout/增强", "记录训练与验证曲线"),
        "不要把成功只归因于网络更深，训练配方同样关键。",
        ("CNN 基础", "过拟合与正则化"),
        0.25,
        0.42,
        30,
    ),
    Paper(
        "resnet",
        "ResNet",
        2015,
        "CNN",
        "网络加深后训练误差反而变差，优化成为瓶颈。",
        "让层学习残差 F(x)，再与输入 x 相加，使深层网络更容易接近恒等映射。",
        ("跳连保留梯度通道", "残差块学习修正量", "批归一化稳定训练"),
        "让上百层视觉网络成为可训练标准架构。",
        ("退化问题", "残差块公式", "plain net 与 residual net 对照"),
        ("实现 BasicBlock", "画梯度范数", "比较 18 层 plain/residual toy net"),
        "残差连接不是简单堆层技巧，它改变了优化目标的形状。",
        ("反向传播", "CNN 架构"),
        0.38,
        0.54,
        36,
    ),
    Paper(
        "attention",
        "Attention Is All You Need",
        2017,
        "Transformer",
        "RNN 难以并行处理长序列，长距离依赖也不稳定。",
        "用自注意力直接建立任意位置之间的信息路由，再用多头机制学习不同关系。",
        ("Q/K/V 计算相似度与取值", "多头注意力并行看不同关系", "位置编码补充顺序信息", "残差与层归一化稳定堆叠"),
        "成为大语言模型、视觉 Transformer 和多模态模型的基础。",
        ("Scaled dot-product attention", "Multi-head attention", "Encoder/Decoder mask"),
        ("手写 attention 函数", "可视化注意力矩阵", "测试 mask 是否泄漏未来 token"),
        "注意力权重可以帮助观察，但不等于完整解释。",
        ("矩阵乘法", "序列建模", "Softmax"),
        0.55,
        0.75,
        42,
    ),
    Paper(
        "bert",
        "BERT",
        2018,
        "Transformer",
        "很多 NLP 任务缺少大规模标注，模型需要先学通用语言表示。",
        "用双向 Transformer 编码器和掩码语言建模预训练，再在下游任务微调。",
        ("Masked LM 构造自监督目标", "双向上下文编码", "预训练与微调分离"),
        "推动 NLP 进入预训练-微调范式。",
        ("预训练任务", "输入表示", "微调方式"),
        ("做一个 mask prediction toy task", "比较冻结与微调", "分析句向量变化"),
        "BERT 不是生成式解码器，它更擅长理解与编码。",
        ("Transformer 编码器", "语言模型基础"),
        0.68,
        0.63,
        34,
    ),
    Paper(
        "gan",
        "GAN",
        2014,
        "生成模型",
        "显式写出高维数据分布很难，但可以训练一个会造样本的模型。",
        "让生成器和判别器博弈：一个造得更真，一个分得更准。",
        ("生成器把噪声映射成样本", "判别器学习真假边界", "对抗训练逼近数据分布"),
        "打开了高质量生成建模的重要路线。",
        ("minimax 目标", "训练不稳定原因", "模式崩塌"),
        ("在 2D 高斯上训练 GAN", "观察判别器边界", "尝试 WGAN 思路"),
        "GAN 的难点主要在训练动力学，不是只改生成器结构。",
        ("概率分布", "优化", "神经网络基础"),
        0.48,
        0.33,
        30,
    ),
    Paper(
        "gcn",
        "GCN",
        2016,
        "GNN",
        "图数据没有规则网格，传统卷积不能直接套用。",
        "让节点从邻居聚合信息，并用归一化邻接矩阵稳定消息传递。",
        ("邻接矩阵定义信息流", "节点特征逐层混合", "归一化防止度数支配"),
        "成为图神经网络入门和半监督节点分类的经典基线。",
        ("消息传递公式", "邻居聚合", "层数过深的过平滑"),
        ("实现一层 GCN", "可视化节点嵌入", "比较 1/2/4 层效果"),
        "图上堆很多层常常会过平滑，不能照搬 CNN 的加深直觉。",
        ("矩阵乘法", "图结构数据"),
        0.62,
        0.38,
        26,
    ),
    Paper(
        "dqn",
        "DQN",
        2015,
        "RL",
        "强化学习样本相关性强，深度网络训练容易发散。",
        "用经验回放打散样本相关性，用目标网络稳定 bootstrap 目标。",
        ("Q 网络估计动作价值", "Replay buffer 复用经验", "Target network 降低目标漂移"),
        "把深度学习带入高维视觉控制任务。",
        ("Bellman 目标", "经验回放", "目标网络更新"),
        ("实现小网格 Q-learning", "加 replay buffer", "比较目标网络开关"),
        "分数上升不代表策略稳定，要看回报方差和探索策略。",
        ("强化学习入门", "神经网络基础"),
        0.78,
        0.48,
        28,
    ),
)


CONNECTIONS = (
    ("lenet", "alexnet", "扩大模型与训练配方"),
    ("alexnet", "resnet", "深层视觉网络可训练性"),
    ("attention", "bert", "预训练语言表示"),
    ("attention", "gcn", "结构化信息路由"),
    ("gan", "attention", "生成模型后来吸收注意力"),
    ("resnet", "attention", "残差与归一化成为 Transformer 标配"),
    ("attention", "dqn", "序列决策也可使用注意力表征"),
)


def e(value: str) -> str:
    return html.escape(value, quote=True)


def selected_paper(name: str) -> Paper:
    return next(paper for paper in PAPERS if paper.title == name)


def paper_by_key(key: str) -> Paper:
    return next(paper for paper in PAPERS if paper.key == key)


def page_style() -> None:
    st.set_page_config(page_title="经典论文解读实验室", layout="wide", initial_sidebar_state="expanded")
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
        .mini-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid #d8dee3;
            border-radius: 8px;
            padding: 0.82rem 0.9rem;
            line-height: 1.58;
            min-height: 7.5rem;
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


def metric_row(paper: Paper) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年份", str(paper.year))
    c2.metric("方向", paper.family)
    c3.metric("阅读优先级", "高" if paper.size >= 32 else "中")
    c4.metric("前置知识", f"{len(paper.prereq)} 项")


def render_cards(items: tuple[str, ...] | list[str]) -> None:
    cols = st.columns(min(3, len(items)))
    for index, item in enumerate(items):
        with cols[index % len(cols)]:
            st.markdown(f'<div class="mini-card">{e(item)}</div>', unsafe_allow_html=True)


def timeline_chart() -> go.Figure:
    df = pd.DataFrame(
        {
            "title": [p.title for p in PAPERS],
            "year": [p.year for p in PAPERS],
            "family": [p.family for p in PAPERS],
            "impact": [p.impact for p in PAPERS],
            "size": [p.size for p in PAPERS],
        }
    )
    color_map = {"CNN": TEAL, "Transformer": BLUE, "生成模型": ROSE, "GNN": GREEN, "RL": AMBER}
    fig = go.Figure()
    for family, group in df.groupby("family"):
        fig.add_trace(
            go.Scatter(
                x=group["year"],
                y=[family] * len(group),
                mode="markers+text",
                marker={"size": group["size"], "color": color_map.get(family, VIOLET), "line": {"color": "white", "width": 1}},
                text=group["title"],
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>%{x}<br>%{customdata}<extra></extra>",
                customdata=group["impact"],
                name=family,
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=PLOT_FONT,
        xaxis={"title": "", "dtick": 2, "gridcolor": "rgba(23,32,38,0.12)"},
        yaxis={"title": "", "gridcolor": "rgba(23,32,38,0.08)"},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def idea_graph(selected: Paper) -> go.Figure:
    node_index = {paper.key: i for i, paper in enumerate(PAPERS)}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for left, right, _ in CONNECTIONS:
        a, b = paper_by_key(left), paper_by_key(right)
        edge_x.extend([a.x, b.x, None])
        edge_y.extend([a.y, b.y, None])

    colors = [ROSE if paper.key == selected.key else BLUE if paper.family == "Transformer" else TEAL for paper in PAPERS]
    sizes = [paper.size + (10 if paper.key == selected.key else 0) for paper in PAPERS]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.4, "color": "rgba(89,103,114,0.45)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p.x for p in PAPERS],
            y=[p.y for p in PAPERS],
            mode="markers+text",
            marker={"size": sizes, "color": colors, "line": {"color": "white", "width": 1.5}},
            text=[p.title for p in PAPERS],
            textposition="top center",
            customdata=[p.core_idea for p in PAPERS],
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    for left, right, label in CONNECTIONS:
        a, b = paper_by_key(left), paper_by_key(right)
        fig.add_annotation(
            x=(a.x + b.x) / 2,
            y=(a.y + b.y) / 2,
            text=label,
            showarrow=False,
            font={"size": 10, "color": MUTED},
            bgcolor="rgba(248,249,246,0.72)",
            bordercolor="rgba(216,222,227,0.6)",
            borderpad=2,
        )
    fig.update_layout(
        height=430,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=PLOT_FONT,
        xaxis={"visible": False, "range": [0, 0.92]},
        yaxis={"visible": False, "range": [0.12, 0.88]},
    )
    return fig


def attention_cost(seq_len: int, d_model: int, heads: int, layers: int) -> tuple[float, float, float]:
    head_dim = d_model / heads
    attention_scores = layers * heads * seq_len * seq_len
    qkv_params = layers * 3 * d_model * d_model
    approx_ops = layers * (4 * seq_len * d_model * d_model + 2 * heads * seq_len * seq_len * head_dim)
    return attention_scores / 1_000_000, qkv_params / 1_000_000, approx_ops / 1_000_000_000


def cost_chart(seq_len: int, d_model: int, heads: int, layers: int) -> go.Figure:
    lengths = [64, 128, 256, 512, 1024, 2048]
    rows = []
    for length in lengths:
        attn_mb, _, ops_g = attention_cost(length, d_model, heads, layers)
        rows.append({"seq_len": length, "attention_cells_m": attn_mb, "ops_g": ops_g})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["seq_len"],
            y=df["attention_cells_m"],
            mode="lines+markers",
            name="注意力矩阵单元数 M",
            line={"color": ROSE, "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["seq_len"],
            y=df["ops_g"],
            mode="lines+markers",
            name="近似计算量 GOp",
            yaxis="y2",
            line={"color": BLUE, "width": 3},
        )
    )
    fig.add_vline(x=seq_len, line_dash="dash", line_color=AMBER)
    fig.update_layout(
        height=360,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=PLOT_FONT,
        xaxis={"title": "序列长度", "gridcolor": "rgba(23,32,38,0.12)"},
        yaxis={"title": "注意力矩阵单元数 M", "gridcolor": "rgba(23,32,38,0.12)"},
        yaxis2={"title": "近似计算量 GOp", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.22},
    )
    return fig


def pseudo_code(paper: Paper) -> str:
    snippets = {
        "resnet": """y = block(x)
out = activation(y + shortcut(x))""",
        "attention": """scores = (Q @ K.T) / sqrt(d_k)
weights = softmax(scores + mask)
output = weights @ V""",
        "gan": """for real_batch in loader:
    update(discriminator, real_batch, generator(noise))
    update(generator, noise, target="fool discriminator")""",
        "gcn": """h_next = activation(norm_adj @ h @ weight)
# 每个节点接收邻居的加权信息""",
        "dqn": """target = reward + gamma * max(Q_target(next_state))
loss = mse(Q_online(state, action), target)""",
    }
    return snippets.get(
        paper.key,
        """features = backbone(x)
logits = classifier(features)
loss = criterion(logits, y)
loss.backward()""",
    )


page_style()

st.title("经典论文解读实验室")
st.markdown(
    '<div class="note">把论文当成一个工程假设来读：它解决什么瓶颈，用什么机制改变了训练或表示，今天复现时最该验证哪一个现象。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("阅读设置")
    paper_name = st.selectbox("选择论文", [paper.title for paper in PAPERS], index=3)
    view = st.radio("阅读视角", ["核心思想", "机制拆解", "复现清单"], index=0)
    seq_len = st.slider("Transformer 序列长度", 64, 2048, 512, step=64)
    d_model = st.select_slider("隐藏维度", options=[128, 256, 512, 768, 1024], value=512)
    heads = st.select_slider("注意力头数", options=[2, 4, 8, 12, 16], value=8)
    layers = st.slider("层数", 1, 24, 6)
    st.divider()
    st.caption("图表是教学用结构化示意，不依赖外部服务。")

paper = selected_paper(paper_name)
metric_row(paper)

left, right = st.columns([0.52, 0.48])
with left:
    st.subheader(paper.title)
    st.markdown(f'<div class="note"><strong>核心思想：</strong>{e(paper.core_idea)}</div>', unsafe_allow_html=True)
    if view == "核心思想":
        rows = [
            ("要解决的问题", paper.problem),
            ("关键影响", paper.impact),
            ("常见误读", paper.common_trap),
        ]
        for title, body in rows:
            st.markdown(f"**{title}**")
            st.write(body)
    elif view == "机制拆解":
        st.markdown("**机制链条**")
        render_cards(list(paper.mechanism))
    else:
        st.markdown("**建议先读**")
        render_cards(list(paper.read_first))
        st.markdown("**最小复现任务**")
        render_cards(list(paper.implement))
with right:
    st.plotly_chart(idea_graph(paper), use_container_width=True, config=PLOT_CONFIG)

tabs = st.tabs(["论文时间线", "Transformer 成本实验", "阅读模板", "实现提示"])

with tabs[0]:
    st.subheader("经典论文时间线")
    st.plotly_chart(timeline_chart(), use_container_width=True, config=PLOT_CONFIG)
    st.dataframe(
        pd.DataFrame(
            {
                "论文": [p.title for p in PAPERS],
                "年份": [p.year for p in PAPERS],
                "方向": [p.family for p in PAPERS],
                "一句话": [p.core_idea for p in PAPERS],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with tabs[1]:
    st.subheader("为什么长上下文让注意力变贵")
    attn_m, qkv_m, ops_g = attention_cost(seq_len, d_model, heads, layers)
    m1, m2, m3 = st.columns(3)
    m1.metric("注意力矩阵单元", f"{attn_m:.1f}M")
    m2.metric("QKV 参数量", f"{qkv_m:.1f}M")
    m3.metric("近似计算量", f"{ops_g:.1f}GOp")
    st.plotly_chart(cost_chart(seq_len, d_model, heads, layers), use_container_width=True, config=PLOT_CONFIG)
    st.markdown(
        '<div class="note">序列长度翻倍时，注意力矩阵按平方增长。Flash Attention 等方法的价值，不只是算得快，更是减少中间矩阵读写造成的显存压力。</div>',
        unsafe_allow_html=True,
    )

with tabs[2]:
    st.subheader("三遍阅读模板")
    template = [
        ("第一遍：抓问题", "只回答：作者认为旧方法卡在哪里？这个瓶颈是表达能力、优化、数据、计算还是评估？"),
        ("第二遍：拆机制", "把核心公式或结构拆成输入、变换、输出、训练信号。每一步都问：如果删掉它会怎样？"),
        ("第三遍：做复现", "选一个最小实验复现论文声称的关键现象，而不是一开始追完整榜单。"),
    ]
    render_cards([f"{title}<br>{body}" for title, body in template])
    st.markdown("**当前论文前置知识**")
    st.markdown("".join(f'<span class="tag">{e(item)}</span>' for item in paper.prereq), unsafe_allow_html=True)

with tabs[3]:
    st.subheader("最小实现骨架")
    st.code(pseudo_code(paper), language="python")
    st.markdown("**复现检查点**")
    for item in paper.implement:
        st.checkbox(item, key=f"paper-{paper.key}-{item}")
    st.markdown(
        f'<div class="note"><strong>调试提醒：</strong>{e(paper.common_trap)}</div>',
        unsafe_allow_html=True,
    )
