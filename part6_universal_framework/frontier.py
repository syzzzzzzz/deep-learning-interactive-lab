"""
Frontier AI concepts teaching page.

Run:
    streamlit run part6_universal_framework/frontier.py
or:
    python main.py part6/frontier
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}

PALETTE = {
    "ink": "#172026",
    "muted": "#596772",
    "line": "#d8dee3",
    "teal": "#0f8b8d",
    "rose": "#bf3f5b",
    "amber": "#c4871f",
    "blue": "#3268a8",
    "green": "#3f7d58",
    "violet": "#7353ba",
}


@dataclass(frozen=True)
class FrontierTopic:
    name: str
    idea: str
    core_question: str
    examples: tuple[str, ...]
    risks: tuple[str, ...]


TOPICS = [
    FrontierTopic(
        "大语言模型与 AGI",
        "大语言模型先用海量文本学习语言中的统计结构，再通过指令微调、偏好学习和工具调用变成通用助手。AGI 则是更高层目标：系统能在广泛任务上迁移、规划、学习和自我纠错。",
        "语言建模能力能扩展到多广的通用智能？",
        ("GPT 系列", "Claude", "Gemini", "Llama"),
        ("幻觉", "能力评估困难", "工具误用", "过度自动化"),
    ),
    FrontierTopic(
        "多模态大模型",
        "多模态模型把图像、文本、音频或视频映射到可共同计算的表示空间，让模型既能看图，也能对视觉内容推理和生成说明。",
        "不同模态如何对齐成同一个语义空间？",
        ("CLIP", "BLIP", "GPT-4V", "Gemini Vision"),
        ("视觉错觉", "OCR 错误", "空间推理失败", "隐私泄露"),
    ),
    FrontierTopic(
        "自监督、小样本、零样本",
        "自监督学习从数据自身构造训练信号；小样本学习用极少标注快速适配；零样本学习直接把任务描述转成模型可执行的行为。",
        "少标注甚至无标注时，模型凭什么泛化？",
        ("Masked modeling", "Contrastive learning", "In-context learning", "Prompting"),
        ("数据偏见迁移", "提示词脆弱", "评估泄漏", "分布外失败"),
    ),
    FrontierTopic(
        "可解释性 AI",
        "XAI 试图回答模型为什么这样预测。方法包括特征归因、注意力可视化、概念探针、反事实样本和机制解释。",
        "解释是在解释模型本身，还是解释一个事后近似？",
        ("SHAP", "LIME", "Grad-CAM", "Mechanistic interpretability"),
        ("解释不稳定", "因果性不足", "过度信任", "展示偏差"),
    ),
    FrontierTopic(
        "AI 安全与对齐",
        "安全关注模型不会造成不可接受的伤害；对齐关注模型目标、行为和人类意图是否一致。它覆盖训练、评测、部署、监控和治理。",
        "怎样让能力更强的系统仍然可控、可靠、可问责？",
        ("RLHF/RLAIF", "Red teaming", "Constitutional AI", "Model evaluations"),
        ("越狱", "欺骗性行为", "滥用", "目标错配"),
    ),
    FrontierTopic(
        "AI 智能体与工具使用",
        "AI 智能体把大语言模型当作推理核心，通过规划、工具调用、记忆和环境反馈完成复杂任务。模型不再只回答问题，而是拆解目标、执行代码、操作浏览器、调用 API，形成感知—规划—行动的循环。",
        "语言模型的推理能力能否可靠地驱动多步决策？",
        ("Function Calling", "ReAct", "AutoGPT", "MCP 协议", "Claude Computer Use"),
        ("工具误调", "死循环", "权限越界", "幻觉驱动错误操作", "安全沙箱不足"),
    ),
    FrontierTopic(
        "推理模型与测试时计算",
        "推理模型在回答前进行长链思维（Chain-of-Thought），用更多测试时计算换取更高质量输出。这一范式表明：扩展不仅发生在训练阶段，推理阶段的计算投入同样能提升数学、编程和复杂推理表现。",
        "在推理时投入更多计算，能否像训练时 scaling law 一样可预测地提升能力？",
        ("o1/o3", "DeepSeek R1", "Chain-of-Thought", "Test-time Compute Scaling"),
        ("推理成本飙升", "延迟增大", "思维链不可审计", "过度推理"),
    ),
]


MULTIMODAL_ROWS = [
    ("CLIP", "图像-文本对比学习", "把图片和文字编码到同一向量空间", "检索、零样本分类、图文匹配"),
    ("BLIP", "视觉语言预训练", "结合图像编码器、文本编码器和生成式解码器", "图像描述、VQA、图文检索"),
    ("GPT-4V", "通用视觉语言模型", "把图像作为上下文输入大语言模型进行推理", "看图问答、文档理解、界面分析"),
]


st.set_page_config(
    page_title="前沿方向：LLM、AGI 与 AI 安全",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172026;
        --muted: #596772;
        --line: #d8dee3;
        --teal: #0f8b8d;
        --rose: #bf3f5b;
        --amber: #c4871f;
        --blue: #3268a8;
        --green: #3f7d58;
        --violet: #7353ba;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(239,246,243,0.96) 100%),
            #fbfaf6;
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
    }
    h1, h2, h3 { letter-spacing: 0; }
    section[data-testid="stSidebar"] {
        background: #eef4f2;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.9rem;
        margin-bottom: 0.9rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 1.08;
        margin: 0;
    }
    .hero p {
        color: var(--muted);
        max-width: 980px;
        line-height: 1.75;
        margin: 0.45rem 0 0 0;
        font-size: 1.02rem;
    }
    .topic-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.65rem 0 1rem 0;
    }
    .topic-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        min-height: 156px;
    }
    .topic-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .topic-card p {
        color: var(--muted);
        margin: 0;
        line-height: 1.62;
        font-size: 0.92rem;
    }
    .note {
        border-left: 4px solid var(--teal);
        background: rgba(255,255,255,0.74);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        color: #26343b;
        line-height: 1.7;
        margin: 0.35rem 0 0.85rem 0;
    }
    .mini-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.25rem 0 0.85rem 0;
        font-size: 0.93rem;
    }
    .mini-table td {
        border-bottom: 1px solid rgba(216,222,227,0.9);
        padding: 0.5rem 0.38rem;
        color: var(--muted);
        vertical-align: top;
    }
    .mini-table td:first-child {
        color: var(--ink);
        font-weight: 700;
        width: 24%;
    }
    .tag {
        display: inline-block;
        margin: 0.42rem 0.32rem 0 0;
        padding: 0.15rem 0.46rem;
        border: 1px solid rgba(15,139,141,0.26);
        border-radius: 999px;
        color: #25616a;
        font-size: 0.78rem;
        background: rgba(15,139,141,0.07);
    }
    @media (max-width: 1000px) {
        .topic-grid { grid-template-columns: 1fr; }
        .topic-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def e(text: str) -> str:
    return html.escape(text, quote=True)


def render_topic_cards() -> None:
    cards = []
    for topic in TOPICS:
        tags = "".join(f'<span class="tag">{e(item)}</span>' for item in topic.examples[:3])
        cards.append(
            '<div class="topic-card">'
            f"<strong>{e(topic.name)}</strong>"
            f"<p>{e(topic.core_question)}</p>"
            f"{tags}"
            "</div>"
        )
    st.markdown('<div class="topic-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_rows(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<tr><td>{e(left)}</td><td>{e(right)}</td></tr>" for left, right in rows)
    st.markdown(f'<table class="mini-table">{body}</table>', unsafe_allow_html=True)


def capability_radar(values: dict[str, int]) -> go.Figure:
    categories = list(values)
    scores = list(values.values())
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line={"color": PALETTE["teal"], "width": 3},
                fillcolor="rgba(15,139,141,0.22)",
                hovertemplate="%{theta}: %{r}/5<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="rgba(255,255,255,0)",
        polar={
            "radialaxis": {"visible": True, "range": [0, 5], "gridcolor": "rgba(89,103,114,0.18)"},
            "angularaxis": {"gridcolor": "rgba(89,103,114,0.18)"},
            "bgcolor": "rgba(255,255,255,0.65)",
        },
        font=PLOT_FONT,
        showlegend=False,
    )
    return fig


def plot_learning_modes() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(name="标注需求", x=["自监督", "小样本", "零样本"], y=[1, 2, 0], marker_color=PALETTE["blue"]),
            go.Bar(name="迁移依赖", x=["自监督", "小样本", "零样本"], y=[3, 4, 5], marker_color=PALETTE["amber"]),
            go.Bar(name="提示词敏感度", x=["自监督", "小样本", "零样本"], y=[1, 3, 5], marker_color=PALETTE["rose"]),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=410,
        margin={"l": 30, "r": 18, "t": 45, "b": 35},
        title="三种少标注学习范式的直观对比",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font=PLOT_FONT,
        yaxis={"range": [0, 5], "title": "相对强度"},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def plot_alignment_loop() -> go.Figure:
    labels = ["预训练", "指令微调", "偏好对齐", "红队评测", "部署监控", "数据回流"]
    parents = ["", "预训练", "指令微调", "偏好对齐", "偏好对齐", "部署监控"]
    values = [10, 7, 5, 3, 4, 2]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["green"], PALETTE["rose"], PALETTE["amber"], PALETTE["violet"]]
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values, marker={"colors": colors}))
    fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        title="从训练到部署的安全与对齐闭环",
        paper_bgcolor="rgba(255,255,255,0)",
        font=PLOT_FONT,
    )
    return fig


def selected_topic(name: str) -> FrontierTopic:
    return next(topic for topic in TOPICS if topic.name == name)


st.markdown(
    """
    <div class="hero">
      <h1>前沿方向：LLM、AGI 与可信 AI</h1>
      <p>
      本页把大语言模型、多模态、自监督与少样本学习、可解释性、安全对齐、AI 智能体和推理模型放到同一张地图里。
      重点不是追热点名词，而是看清每个方向在解决什么问题、依赖什么假设、有哪些风险边界。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_topic_cards()

with st.sidebar:
    st.header("探索设置")
    topic_name = st.radio("主题", [topic.name for topic in TOPICS], index=0)
    depth = st.select_slider("讲解深度", options=["概念", "机制", "风险"], value="机制")
    st.divider()
    st.caption("这个页面聚焦概念和框架，不调用外部模型。所有图表都是教学用的结构化示意。")

topic = selected_topic(topic_name)

st.subheader(topic.name)
st.markdown(f'<div class="note">{e(topic.idea)}</div>', unsafe_allow_html=True)

col_text, col_plot = st.columns([0.55, 0.45])
with col_text:
    render_rows(
        [
            ("核心问题", topic.core_question),
            ("代表系统/方法", "、".join(topic.examples)),
            ("主要风险", "、".join(topic.risks)),
        ]
    )
    if depth == "概念":
        st.write("先抓住问题定义：这个方向通常不是单一模型，而是一组训练目标、数据形态、评估方式和部署约束的组合。")
    elif depth == "机制":
        st.write("机制层要看三件事：输入如何表示，训练信号从哪里来，模型输出如何被评估或约束。")
    else:
        st.write("风险层要区分能力失败、对齐失败和治理失败。能力失败是做不到；对齐失败是会做但目标不对；治理失败是部署边界没有管住。")

with col_plot:
    radar_values = {
        "通用性": 5 if "大语言" in topic.name else 4,
        "数据规模": 5 if topic.name in {"大语言模型与 AGI", "多模态大模型"} else 3,
        "可解释性": 5 if "可解释性" in topic.name else 2,
        "部署风险": 5 if "安全" in topic.name or "大语言" in topic.name else 3,
        "工程复杂度": 5 if topic.name in {"多模态大模型", "AI 安全与对齐", "AI 智能体与工具使用"} else 4,
        "交互复杂度": 4 if "智能体" in topic.name else (5 if "推理模型" in topic.name else 2),
    }
    st.plotly_chart(capability_radar(radar_values), width="stretch", config=PLOT_CONFIG)

tabs = st.tabs(["多模态概览", "少标注学习", "XAI", "安全与对齐", "智能体与推理"])

with tabs[0]:
    st.subheader("CLIP、BLIP、GPT-4V 的位置")
    df = pd.DataFrame(MULTIMODAL_ROWS, columns=["模型", "训练/能力重点", "核心机制", "典型任务"])
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown(
        '<div class="note">CLIP 更像语义对齐底座，BLIP 更偏视觉语言预训练任务组合，GPT-4V 代表把视觉输入接入通用语言推理系统。</div>',
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.subheader("自监督、小样本、零样本")
    col_a, col_b = st.columns([0.48, 0.52])
    with col_a:
        render_rows(
            [
                ("自监督学习", "从数据自身构造标签，例如预测被遮住的词、对比同一图像的不同增强视图。"),
                ("小样本学习", "给模型少量示例，让它快速适配新类别、新格式或新任务。"),
                ("零样本学习", "不提供任务样本，只用自然语言描述、类别名或提示词完成迁移。"),
            ]
        )
    with col_b:
        st.plotly_chart(plot_learning_modes(), width="stretch", config=PLOT_CONFIG)

with tabs[2]:
    st.subheader("可解释性 AI 的几类问题")
    render_rows(
        [
            ("特征归因", "哪些输入特征对输出影响最大，例如 SHAP、LIME、Integrated Gradients。"),
            ("视觉解释", "图像模型关注哪里，例如 Grad-CAM 对分类依据区域做热力图。"),
            ("概念解释", "模型内部是否编码了可命名概念，例如颜色、形状、语法角色。"),
            ("机制解释", "直接研究神经元、电路和注意力头怎样组合出可观察行为。"),
        ]
    )
    st.markdown(
        '<div class="note">好的解释应当能帮助调试、审计或预测模型失败；如果解释只是在输出后生成一个看似合理的故事，它的工程价值有限。</div>',
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.subheader("安全与对齐不是最后一步")
    col_a, col_b = st.columns([0.5, 0.5])
    with col_a:
        render_rows(
            [
                ("训练前", "数据治理、危险能力过滤、基准任务设计。"),
                ("训练中", "指令微调、偏好学习、拒答边界和能力评估。"),
                ("发布前", "红队测试、越狱测试、风险分级和系统卡。"),
                ("发布后", "监控、事件响应、用户反馈、模型和策略更新。"),
            ]
        )
    with col_b:
        st.plotly_chart(plot_alignment_loop(), width="stretch", config=PLOT_CONFIG)

with tabs[4]:
    st.subheader("智能体架构与推理模型")
    col_a, col_b = st.columns([0.48, 0.52])
    with col_a:
        st.markdown("**AI 智能体的核心循环**")
        render_rows(
            [
                ("感知", "接收用户指令、环境观测或工具返回结果，形成当前上下文。"),
                ("规划", "把复杂任务分解成子目标，决定下一步调用哪个工具或直接生成答案。"),
                ("执行", "通过 Function Calling 调用代码解释器、搜索引擎、浏览器或外部 API。"),
                ("记忆", "维护短期工作记忆和长期经验，在多轮交互中保持一致性。"),
                ("反思", "检查执行结果是否符合预期，决定继续、重试还是换策略。"),
            ]
        )
        st.markdown(
            '<div class="note">智能体不是简单的 prompt 拼接。可靠运行需要工具描述精确、输出格式约束、权限沙箱隔离和失败恢复机制，否则幻觉会在多步执行中累积放大。</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("**推理模型的范式变化**")
        render_rows(
            [
                ("训练时 Scaling", "更多参数、更多数据、更多 GPU → 模型能力持续提升。"),
                ("测试时 Scaling", "推理阶段投入更多计算（更长思维链） → 同样提升困难任务表现。"),
                ("思维链", "模型在输出答案前先进行内部推理，步骤越完整、错误率越低。"),
                ("过程奖励", "奖励模型评估每个推理步骤是否正确，而非只看最终答案。"),
            ]
        )
        st.markdown(
            '<div class="note">推理模型的工程挑战：延迟可能从秒级膨胀到分钟级，token 消耗大幅增加，且长思维链目前难以完全审计和调试。这意味着部署时需要在质量、速度和成本之间做更精细的权衡。</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.subheader("智能体 vs 单次推理：何时该用哪个？")
    agent_df = pd.DataFrame(
        [
            ("单次问答", "简单查询、翻译、摘要", "低", "低", "低"),
            ("RAG 增强", "需要检索外部知识", "中", "低", "中"),
            ("工具调用", "需要计算、代码执行、API", "中", "中", "中"),
            ("多步智能体", "复杂研究、数据分析、自动化", "高", "高", "高"),
        ],
        columns=["模式", "适用场景", "推理成本", "延迟", "可靠性风险"],
    )
    st.dataframe(agent_df, width="stretch", hide_index=True)
