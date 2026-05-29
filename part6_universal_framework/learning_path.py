"""
Learning path recommendation and knowledge graph page.

Run:
    streamlit run part6_universal_framework/learning_path.py
or:
    python main.py part6/learning_path
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

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

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": INK}


@dataclass(frozen=True)
class LearningModule:
    module_id: str
    title: str
    part: str
    level: str
    duration: str
    summary: str
    prerequisites: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearningPath:
    name: str
    audience: str
    module_ids: tuple[str, ...]
    focus: str


MODULES: tuple[LearningModule, ...] = (
    LearningModule(
        "ml_basics",
        "机器学习基础",
        "基础",
        "入门",
        "1-2 天",
        "监督学习、损失函数、泛化、训练/验证划分。",
        tags=("数学", "机器学习"),
    ),
    LearningModule(
        "math_primer",
        "数学基础速查",
        "基础",
        "入门",
        "1-2 天",
        "线性代数、微积分、概率和梯度下降的最小必备概念。",
        tags=("数学", "优化"),
    ),
    LearningModule(
        "nn_basics",
        "神经网络基础",
        "基础",
        "入门",
        "2-3 天",
        "感知机、多层网络、激活函数、反向传播。",
        ("ml_basics", "math_primer"),
        ("神经网络", "优化"),
    ),
    LearningModule(
        "data_training",
        "数据与训练流程",
        "工具箱",
        "入门",
        "1-2 天",
        "数据管线、批次、指标、过拟合和调参的基本工作流。",
        ("nn_basics",),
        ("训练", "工程"),
    ),
    LearningModule(
        "cnn",
        "CNN 与视觉模型",
        "CNN",
        "进阶",
        "3-5 天",
        "卷积、特征图、池化、经典视觉网络和迁移学习。",
        ("nn_basics", "data_training"),
        ("视觉", "CNN"),
    ),
    LearningModule(
        "rnn",
        "RNN 与序列建模",
        "RNN",
        "进阶",
        "2-4 天",
        "隐藏状态、LSTM/GRU、序列分类、seq2seq 和注意力雏形。",
        ("nn_basics", "data_training"),
        ("序列", "NLP"),
    ),
    LearningModule(
        "transformer",
        "Transformer 架构",
        "Transformer",
        "核心",
        "4-6 天",
        "自注意力、多头注意力、位置编码、编码器/解码器和 GPT/BERT 思路。",
        ("rnn", "data_training"),
        ("Transformer", "NLP"),
    ),
    LearningModule(
        "generative",
        "生成模型入门",
        "前沿",
        "进阶",
        "2-4 天",
        "自编码器、GAN、潜空间和生成任务的基础直觉。",
        ("cnn", "transformer"),
        ("生成模型", "表征"),
    ),
    LearningModule(
        "gnn",
        "图神经网络",
        "前沿",
        "进阶",
        "2-3 天",
        "节点、边、消息传递和结构化数据建模。",
        ("nn_basics", "data_training"),
        ("GNN", "结构数据"),
    ),
    LearningModule(
        "deployment",
        "部署与工程工具",
        "工具箱",
        "工程",
        "2-4 天",
        "模型导出、服务化、推理优化和实验管理。",
        ("data_training", "transformer"),
        ("工程", "部署"),
    ),
    LearningModule(
        "streamlit_lab",
        "可视化实验台",
        "统一框架",
        "核心",
        "1-2 天",
        "用 Streamlit 交互式观察决策边界、卷积和注意力。",
        ("nn_basics",),
        ("实验", "可视化"),
    ),
    LearningModule(
        "rl",
        "强化学习入门",
        "前沿",
        "进阶",
        "2-4 天",
        "智能体、环境、奖励、探索利用、Q-Learning 和策略梯度。",
        ("nn_basics", "data_training"),
        ("RL", "智能体"),
    ),
    LearningModule(
        "framework",
        "统一接口与项目框架",
        "统一框架",
        "工程",
        "2-3 天",
        "把模型、数据、训练任务抽象成可扩展项目结构。",
        ("data_training", "deployment"),
        ("架构", "工程"),
    ),
    LearningModule(
        "frontier",
        "LLM 与前沿方向",
        "前沿",
        "前沿",
        "2-5 天",
        "LLM、多模态、自监督、XAI、安全与对齐的地图式总览。",
        ("transformer", "deployment"),
        ("LLM", "安全", "前沿"),
    ),
)

MODULE_BY_ID = {module.module_id: module for module in MODULES}

PATHS: tuple[LearningPath, ...] = (
    LearningPath(
        "零基础稳扎稳打",
        "适合刚开始系统学习深度学习的人。",
        ("ml_basics", "math_primer", "nn_basics", "data_training", "streamlit_lab", "cnn", "rnn", "transformer"),
        "先补齐概念和训练流程，再进入视觉、序列和 Transformer。",
    ),
    LearningPath(
        "有机器学习基础",
        "适合已经懂监督学习、想快速进入深度学习主线的人。",
        ("nn_basics", "data_training", "streamlit_lab", "cnn", "rnn", "transformer", "generative", "frontier"),
        "用实验台建立直觉，然后沿 CNN/RNN/Transformer 拓展。",
    ),
    LearningPath(
        "工程落地路线",
        "适合想把模型训练、部署和项目结构串起来的人。",
        ("nn_basics", "data_training", "transformer", "deployment", "framework", "streamlit_lab", "frontier"),
        "强调训练流程、部署工具和统一项目接口。",
    ),
    LearningPath(
        "大模型与前沿路线",
        "适合目标是理解 LLM、智能体和前沿研究方向的人。",
        ("nn_basics", "rnn", "transformer", "deployment", "frontier", "rl", "gnn", "generative"),
        "以 Transformer 为主干，补部署、前沿、安全和智能体概念。",
    ),
)

QUESTION_BANK = (
    {
        "question": "看到一个监督学习任务时，你能否解释训练集、验证集、测试集的区别？",
        "options": ("完全不熟", "知道大概", "能解释并应用"),
        "scores": (0, 1, 2),
    },
    {
        "question": "你对矩阵乘法、导数、梯度下降的直觉有多稳定？",
        "options": ("需要补课", "能跟着推导", "能自己排查问题"),
        "scores": (0, 1, 2),
    },
    {
        "question": "你能否说清反向传播为什么能训练多层神经网络？",
        "options": ("还说不清", "知道链式法则", "能结合代码解释"),
        "scores": (0, 1, 2),
    },
    {
        "question": "你是否实际训练过一个神经网络，并观察过 loss / metric 曲线？",
        "options": ("没有", "跟过教程", "独立做过"),
        "scores": (0, 1, 2),
    },
    {
        "question": "你对 CNN、RNN、Transformer 的适用场景是否有区分？",
        "options": ("基本混在一起", "知道典型任务", "能解释结构差异"),
        "scores": (0, 1, 2),
    },
    {
        "question": "你是否关心模型部署、推理服务、项目结构或实验管理？",
        "options": ("暂时不关心", "有一些需求", "这是主要目标"),
        "scores": (0, 1, 2),
    },
)


st.set_page_config(
    page_title="学习路径推荐与知识图谱",
    layout="wide",
    initial_sidebar_state="auto",
)


def e(value: str) -> str:
    return html.escape(value, quote=True)


def css() -> str:
    return """
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
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(239,246,243,0.97) 100%),
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
        padding: 0.72rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #172026;
        background: #172026;
        color: white;
        min-height: 2.45rem;
        font-weight: 700;
    }
    .stButton > button:hover {
        border-color: #0f8b8d;
        background: #0f8b8d;
        color: white;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.95rem;
        margin-bottom: 0.95rem;
    }
    .hero h1 {
        font-size: clamp(2.05rem, 3vw, 3.35rem);
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
    .card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.7rem 0 1rem 0;
    }
    .path-card, .module-card, .next-card {
        background: rgba(255,255,255,0.80);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
    }
    .path-card strong, .module-card strong, .next-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .path-card p, .module-card p, .next-card p {
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
    .tag {
        display: inline-block;
        margin: 0.44rem 0.32rem 0 0;
        padding: 0.14rem 0.44rem;
        border: 1px solid rgba(15,139,141,0.26);
        border-radius: 999px;
        color: #25616a;
        font-size: 0.78rem;
        background: rgba(15,139,141,0.07);
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }
    @media (max-width: 1000px) {
        .card-grid { grid-template-columns: 1fr; }
    }
    </style>
    """


def initialize_state() -> None:
    st.session_state.setdefault("learning_answers", [1] * len(QUESTION_BANK))
    st.session_state.setdefault("completed_modules", set())
    st.session_state.setdefault("selected_path", PATHS[1].name)


def normalize_completed() -> set[str]:
    raw = st.session_state.get("completed_modules", set())
    if isinstance(raw, set):
        completed = {item for item in raw if item in MODULE_BY_ID}
    else:
        completed = {item for item in list(raw) if item in MODULE_BY_ID}
    st.session_state["completed_modules"] = completed
    return completed


def classify_level(score: int) -> tuple[str, str]:
    if score <= 4:
        return "入门", "先建立机器学习、数学和神经网络的共同语言。"
    if score <= 8:
        return "进阶", "你已经能进入主要模型结构，但训练流程和关键直觉仍值得补齐。"
    return "高阶", "可以把重点放在 Transformer、工程化、前沿方向和项目整合。"


def recommended_path_name(score: int, interest: str) -> str:
    if interest == "工程落地":
        return "工程落地路线"
    if interest == "大模型与前沿":
        return "大模型与前沿路线"
    if score <= 4:
        return "零基础稳扎稳打"
    return "有机器学习基础"


def get_path(name: str) -> LearningPath:
    return next(path for path in PATHS if path.name == name)


def module_ready(module: LearningModule, completed: set[str]) -> bool:
    return all(prerequisite in completed for prerequisite in module.prerequisites)


def progress_for_path(path: LearningPath, completed: set[str]) -> tuple[int, int, float]:
    done = sum(module_id in completed for module_id in path.module_ids)
    total = len(path.module_ids)
    return done, total, done / total if total else 0.0


def recommend_next(path: LearningPath, completed: set[str]) -> LearningModule | None:
    for module_id in path.module_ids:
        module = MODULE_BY_ID[module_id]
        if module_id not in completed and module_ready(module, completed):
            return module
    for module_id in path.module_ids:
        if module_id not in completed:
            return MODULE_BY_ID[module_id]
    return None


def blocked_by(module: LearningModule, completed: set[str]) -> list[LearningModule]:
    return [MODULE_BY_ID[item] for item in module.prerequisites if item not in completed]


def render_note(text: str) -> None:
    st.markdown(f'<div class="note">{e(text)}</div>', unsafe_allow_html=True)


def render_module_card(module: LearningModule, completed: set[str]) -> str:
    status = "已完成" if module.module_id in completed else ("可学习" if module_ready(module, completed) else "需前置")
    tags = "".join(f'<span class="tag">{e(tag)}</span>' for tag in (status, module.level, module.duration, *module.tags[:2]))
    prereq = "、".join(MODULE_BY_ID[item].title for item in module.prerequisites) or "无"
    return (
        '<div class="module-card">'
        f"<strong>{e(module.title)}</strong>"
        f"<p>{e(module.summary)}</p>"
        f'<p class="small-muted">前置：{e(prereq)}</p>'
        f"{tags}"
        "</div>"
    )


def render_path_cards(completed: set[str]) -> None:
    cards = []
    for path in PATHS:
        done, total, ratio = progress_for_path(path, completed)
        cards.append(
            '<div class="path-card">'
            f"<strong>{e(path.name)}</strong>"
            f"<p>{e(path.audience)}</p>"
            f'<p class="small-muted">进度：{done}/{total}，{ratio:.0%}</p>'
            f'<span class="tag">{e(path.focus)}</span>'
            "</div>"
        )
    st.markdown('<div class="card-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def graph_layout() -> dict[str, tuple[float, float]]:
    columns = {
        "基础": 0.0,
        "工具箱": 1.0,
        "CNN": 2.0,
        "RNN": 2.0,
        "Transformer": 3.0,
        "统一框架": 4.0,
        "前沿": 5.0,
    }
    rows = {
        "ml_basics": 2.0,
        "math_primer": 0.8,
        "nn_basics": 1.45,
        "data_training": 1.45,
        "cnn": 2.35,
        "rnn": 1.35,
        "transformer": 1.35,
        "generative": 2.35,
        "gnn": 0.45,
        "deployment": 1.45,
        "streamlit_lab": 0.55,
        "rl": -0.45,
        "framework": 1.45,
        "frontier": 1.45,
    }
    return {module.module_id: (columns[module.part], rows[module.module_id]) for module in MODULES}


def graph_figure(path: LearningPath, completed: set[str], selected_id: str | None) -> go.Figure:
    layout = graph_layout()
    path_ids = set(path.module_ids)
    next_module = recommend_next(path, completed)
    next_id = next_module.module_id if next_module else None

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for module in MODULES:
        x1, y1 = layout[module.module_id]
        for prerequisite in module.prerequisites:
            x0, y0 = layout[prerequisite]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    sizes = []
    colors = []
    symbols = []
    labels = []
    hover = []

    for module in MODULES:
        x, y = layout[module.module_id]
        node_x.append(x)
        node_y.append(y)
        is_completed = module.module_id in completed
        is_next = module.module_id == next_id
        is_path = module.module_id in path_ids
        is_selected = module.module_id == selected_id
        color = GREEN if is_completed else (AMBER if is_next else (TEAL if is_path else "#b8c2c8"))
        size = 28 if is_selected else (25 if is_next else (22 if is_path else 16))
        symbol = "star" if is_next else ("circle" if is_path else "circle-open")
        labels.append(module.title)
        colors.append(color)
        sizes.append(size)
        symbols.append(symbol)
        prereq = "、".join(MODULE_BY_ID[item].title for item in module.prerequisites) or "无"
        state = "已完成" if is_completed else ("推荐下一步" if is_next else ("可学习" if module_ready(module, completed) else "等待前置"))
        hover.append(
            f"<b>{e(module.title)}</b><br>"
            f"状态：{e(state)}<br>"
            f"层级：{e(module.level)}<br>"
            f"预计：{e(module.duration)}<br>"
            f"前置：{e(prereq)}<br>"
            f"{e(module.summary)}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.6, "color": "rgba(89,103,114,0.36)"},
            hoverinfo="skip",
            name="依赖关系",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="bottom center",
            marker={
                "size": sizes,
                "color": colors,
                "symbol": symbols,
                "line": {"color": "#ffffff", "width": 2},
            },
            customdata=[module.module_id for module in MODULES],
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover,
            name="模块",
        )
    )

    for part, x in {"基础": 0, "工具箱": 1, "CNN/RNN": 2, "Transformer": 3, "统一框架": 4, "前沿": 5}.items():
        fig.add_annotation(
            x=x,
            y=3.05,
            text=part,
            showarrow=False,
            font={"size": 13, "color": INK},
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor=LINE,
            borderpad=4,
        )

    fig.update_layout(
        height=620,
        margin={"l": 20, "r": 20, "t": 34, "b": 20},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.70)",
        font=PLOT_FONT,
        showlegend=False,
        hoverlabel={"bgcolor": INK, "font_color": "#ffffff"},
    )
    fig.update_xaxes(visible=False, range=[-0.45, 5.45])
    fig.update_yaxes(visible=False, range=[-0.9, 3.35])
    return fig


def readiness_radar(answers: list[int]) -> go.Figure:
    categories = ["机器学习", "数学", "反向传播", "训练经验", "模型结构", "工程意识"]
    scores = [answer + 1 for answer in answers]
    fig = go.Figure(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line={"color": TEAL, "width": 3},
            fillcolor="rgba(15,139,141,0.20)",
            hovertemplate="%{theta}: %{r}/3<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 25, "b": 20},
        paper_bgcolor="rgba(255,255,255,0)",
        polar={
            "radialaxis": {"visible": True, "range": [0, 3], "gridcolor": "rgba(89,103,114,0.18)"},
            "angularaxis": {"gridcolor": "rgba(89,103,114,0.18)"},
            "bgcolor": "rgba(255,255,255,0.65)",
        },
        font=PLOT_FONT,
        showlegend=False,
    )
    return fig


def path_timeline(path: LearningPath, completed: set[str]) -> go.Figure:
    x_values = list(range(1, len(path.module_ids) + 1))
    modules = [MODULE_BY_ID[module_id] for module_id in path.module_ids]
    colors = [GREEN if module.module_id in completed else (AMBER if module_ready(module, completed) else "#b8c2c8") for module in modules]
    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=[1] * len(modules),
            marker_color=colors,
            text=[module.title for module in modules],
            textposition="inside",
            hovertemplate="第 %{x} 步<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=190,
        margin={"l": 16, "r": 16, "t": 24, "b": 24},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        font=PLOT_FONT,
        xaxis={"title": "学习顺序", "dtick": 1},
        yaxis={"visible": False},
        showlegend=False,
    )
    return fig


def reset_progress() -> None:
    st.session_state["completed_modules"] = set()


initialize_state()
completed_modules = normalize_completed()

st.markdown(css(), unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <h1>学习路径推荐与知识图谱</h1>
      <p>
      先用一个短测评估当前基础，再把课程模块放到依赖关系图里。
      页面会根据你的水平、目标和已完成模块，推荐路径、追踪进度，并给出下一步最适合学习的内容。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("进度追踪")
    selected_for_toggle = st.multiselect(
        "已完成模块",
        [module.module_id for module in MODULES],
        default=sorted(completed_modules),
        format_func=lambda module_id: MODULE_BY_ID[module_id].title,
    )
    st.session_state["completed_modules"] = set(selected_for_toggle)
    completed_modules = normalize_completed()
    if st.button("清空进度", width="stretch"):
        reset_progress()
        st.rerun()
    st.divider()
    st.caption("进度保存在 Streamlit session_state 中，当前浏览器会话内刷新页面仍会保留。")

tabs = st.tabs(["入门测试", "路径推荐", "知识图谱", "下一步"])

with tabs[0]:
    left, right = st.columns([0.56, 0.44])
    with left:
        st.subheader("入门测试")
        answers: list[int] = []
        for index, item in enumerate(QUESTION_BANK):
            selected = st.radio(
                item["question"],
                list(range(len(item["options"]))),
                format_func=lambda option_index, item=item: item["options"][option_index],
                horizontal=True,
                key=f"learning_question_{index}",
                index=int(st.session_state["learning_answers"][index]),
            )
            answers.append(int(selected))
        st.session_state["learning_answers"] = answers
        score = sum(QUESTION_BANK[index]["scores"][answer] for index, answer in enumerate(answers))
        level, level_note = classify_level(score)
        interest = st.segmented_control(
            "当前主要目标",
            ["系统入门", "工程落地", "大模型与前沿"],
            default="系统入门",
        )
        suggested_name = recommended_path_name(score, interest or "系统入门")
        st.session_state["selected_path"] = suggested_name
    with right:
        score = sum(QUESTION_BANK[index]["scores"][answer] for index, answer in enumerate(st.session_state["learning_answers"]))
        level, level_note = classify_level(score)
        suggested_path = get_path(st.session_state["selected_path"])
        c1, c2, c3 = st.columns(3)
        c1.metric("测评分", f"{score}/12")
        c2.metric("水平", level)
        c3.metric("推荐路径", suggested_path.name)
        st.plotly_chart(readiness_radar(st.session_state["learning_answers"]), width="stretch", config=PLOT_CONFIG)
        render_note(f"{level_note} 推荐先走「{suggested_path.name}」：{suggested_path.focus}")

with tabs[1]:
    st.subheader("推荐学习路径")
    render_path_cards(completed_modules)
    chosen_path_name = st.selectbox(
        "选择要追踪的路径",
        [path.name for path in PATHS],
        index=[path.name for path in PATHS].index(st.session_state["selected_path"]),
    )
    st.session_state["selected_path"] = chosen_path_name
    chosen_path = get_path(chosen_path_name)
    done, total, ratio = progress_for_path(chosen_path, completed_modules)
    st.progress(ratio, text=f"{chosen_path.name}：已完成 {done}/{total}")
    st.plotly_chart(path_timeline(chosen_path, completed_modules), width="stretch", config=PLOT_CONFIG)
    st.markdown(
        '<div class="card-grid">'
        + "".join(render_module_card(MODULE_BY_ID[module_id], completed_modules) for module_id in chosen_path.module_ids)
        + "</div>",
        unsafe_allow_html=True,
    )

with tabs[2]:
    chosen_path = get_path(st.session_state["selected_path"])
    st.subheader("知识图谱：模块依赖关系")
    graph_left, graph_right = st.columns([0.72, 0.28])
    with graph_right:
        selected_module_id = st.selectbox(
            "查看模块详情",
            [module.module_id for module in MODULES],
            format_func=lambda module_id: MODULE_BY_ID[module_id].title,
        )
        selected_module = MODULE_BY_ID[selected_module_id]
        st.markdown(f"**{selected_module.title}**")
        st.write(selected_module.summary)
        st.caption(f"层级：{selected_module.level} | 预计：{selected_module.duration}")
        missing = blocked_by(selected_module, completed_modules)
        if selected_module_id in completed_modules:
            render_note("这个模块已经标记为完成。")
        elif missing:
            render_note("还需要先完成：" + "、".join(module.title for module in missing))
        else:
            render_note("前置条件已经满足，可以开始学习。")
        if st.button("切换完成状态", width="stretch"):
            updated = set(completed_modules)
            if selected_module_id in updated:
                updated.remove(selected_module_id)
            else:
                updated.add(selected_module_id)
            st.session_state["completed_modules"] = updated
            st.rerun()
    with graph_left:
        st.plotly_chart(
            graph_figure(chosen_path, completed_modules, selected_module_id),
            width="stretch",
            config=PLOT_CONFIG,
        )
        st.caption("绿色为已完成，黄色星标为当前路径的下一步推荐，青色为所选路径中的待学模块。悬停节点可查看前置依赖和说明。")

with tabs[3]:
    chosen_path = get_path(st.session_state["selected_path"])
    next_module = recommend_next(chosen_path, completed_modules)
    done, total, ratio = progress_for_path(chosen_path, completed_modules)
    st.subheader("下一步学习建议")
    m1, m2, m3 = st.columns(3)
    m1.metric("当前路径", chosen_path.name)
    m2.metric("完成进度", f"{done}/{total}")
    m3.metric("完成比例", f"{ratio:.0%}")
    if next_module is None:
        render_note("当前路径已经完成。可以切换到前沿路线，或回到知识图谱选择新的方向继续扩展。")
    else:
        missing = blocked_by(next_module, completed_modules)
        if missing:
            headline = "先补前置模块"
            body = "、".join(module.title for module in missing)
            recommendation = missing[0]
        else:
            headline = "推荐下一步"
            body = next_module.summary
            recommendation = next_module
        st.markdown(
            '<div class="next-card">'
            f"<strong>{e(headline)}：{e(recommendation.title)}</strong>"
            f"<p>{e(body)}</p>"
            f'<span class="tag">{e(recommendation.level)}</span>'
            f'<span class="tag">{e(recommendation.duration)}</span>'
            f'<span class="tag">{e(recommendation.part)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            if st.button("标记推荐模块为完成", width="stretch"):
                updated = set(completed_modules)
                updated.add(recommendation.module_id)
                st.session_state["completed_modules"] = updated
                st.rerun()
        with c2:
            st.caption("如果你正在按路径学习，每完成一个模块后这里会自动跳到下一个依赖已满足的模块。")

    st.subheader("待办队列")
    ready = [
        MODULE_BY_ID[module_id]
        for module_id in chosen_path.module_ids
        if module_id not in completed_modules and module_ready(MODULE_BY_ID[module_id], completed_modules)
    ]
    blocked = [
        MODULE_BY_ID[module_id]
        for module_id in chosen_path.module_ids
        if module_id not in completed_modules and not module_ready(MODULE_BY_ID[module_id], completed_modules)
    ]
    queue_cols = st.columns(2)
    with queue_cols[0]:
        st.markdown("**现在可以学**")
        if ready:
            for module in ready:
                st.markdown(f"- {module.title} `{module.duration}`")
        else:
            st.caption("暂无可直接开始的模块。")
    with queue_cols[1]:
        st.markdown("**等待前置**")
        if blocked:
            for module in blocked:
                missing = "、".join(item.title for item in blocked_by(module, completed_modules))
                st.markdown(f"- {module.title}：先完成 {missing}")
        else:
            st.caption("没有被前置依赖卡住的模块。")
