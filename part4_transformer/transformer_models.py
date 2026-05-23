"""
Interactive Transformer teaching lab.

Run:
    streamlit run part4_transformer/transformer_models.py
or:
    python main.py part4_transformer/transformer_models
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plotly.subplots import make_subplots


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}

PALETTE = {
    "ink": "#172026",
    "muted": "#596772",
    "line": "#d8dee3",
    "blue": "#3268a8",
    "teal": "#0f8b8d",
    "rose": "#bf3f5b",
    "amber": "#c4871f",
    "green": "#3f7d58",
    "violet": "#7353ba",
    "paper": "#fbfaf6",
}


@dataclass(frozen=True)
class AttentionPack:
    tokens: list[str]
    embeddings: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    scores: np.ndarray
    scaled_scores: np.ndarray
    weights: np.ndarray
    output: np.ndarray


st.set_page_config(
    page_title="Transformer 架构教学",
    layout="wide",
    initial_sidebar_state="expanded",
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
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
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
    .callout {
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        color: #2b3941;
        line-height: 1.68;
        margin: 0.35rem 0 0.75rem 0;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 0.85rem 0;
    }
    .mini-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 112px;
    }
    .mini-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .mini-card p {
        color: var(--muted);
        margin: 0;
        line-height: 1.62;
        font-size: 0.92rem;
    }
    .formula {
        font-family: Consolas, Menlo, monospace;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.72rem 0.9rem;
        line-height: 1.72;
        color: #1e2b32;
        overflow-x: auto;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.58;
    }
    @media (max-width: 1000px) {
        .mini-grid { grid-template-columns: 1fr; }
        .mini-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def tokenize(text: str, max_tokens: int = 14) -> list[str]:
    text = text.strip()
    if not text:
        text = "The cat sat on the mat because it was tired"
    pieces = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u4e00-\u9fff]|[^\s]", text)
    return pieces[:max_tokens] if pieces else ["Transformer", "reads", "context"]


def token_seed(token: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def build_token_embeddings(tokens: list[str], d_model: int, seed: int) -> np.ndarray:
    rows = []
    for index, token in enumerate(tokens):
        rng = np.random.default_rng(token_seed(token, seed))
        base = rng.normal(0, 1, d_model)
        base[index % d_model] += 1.2
        if token.lower() in {"the", "a", "an", "的"}:
            base[0] += 1.4
        if token.lower() in {"because", "therefore", "so", "if", "when", "因为", "所以"}:
            base[1] += 1.8
        if token.lower() in {"it", "he", "she", "they", "它", "他", "她"}:
            base[2] += 1.6
        rows.append(base)
    return normalize_rows(np.array(rows))


def compute_attention(tokens: list[str], d_model: int = 16, seed: int = 7) -> AttentionPack:
    rng = np.random.default_rng(seed)
    x = build_token_embeddings(tokens, d_model, seed)
    w_q = rng.normal(0, 1 / math.sqrt(d_model), (d_model, d_model))
    w_k = rng.normal(0, 1 / math.sqrt(d_model), (d_model, d_model))
    w_v = rng.normal(0, 1 / math.sqrt(d_model), (d_model, d_model))
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    scores = q @ k.T
    heuristic = np.zeros_like(scores)
    lower = [t.lower() for t in tokens]
    for i, ti in enumerate(lower):
        for j, tj in enumerate(lower):
            distance = abs(i - j)
            heuristic[i, j] += 0.45 * math.exp(-distance / 2.6)
            if ti == tj:
                heuristic[i, j] += 0.75
            if ti in {"it", "he", "she", "they", "它", "他", "她"} and j < i:
                heuristic[i, j] += 0.9 * math.exp(-(i - j) / 4)
            if tj in {"because", "therefore", "so", "if", "when", "因为", "所以"}:
                heuristic[i, j] += 0.45
    scores = scores + heuristic
    scaled_scores = scores / math.sqrt(d_model)
    weights = softmax(scaled_scores, axis=-1)
    output = weights @ v
    return AttentionPack(tokens, x, q, k, v, scores, scaled_scores, weights, output)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates
    enc = np.zeros((seq_len, d_model))
    enc[:, 0::2] = np.sin(angles[:, 0::2])
    enc[:, 1::2] = np.cos(angles[:, 1::2])
    return enc


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


def feed_forward_like(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = x.shape[-1]
    w1 = rng.normal(0, 1 / math.sqrt(d), (d, d * 2))
    w2 = rng.normal(0, 1 / math.sqrt(d * 2), (d * 2, d))
    hidden = np.maximum(x @ w1, 0)
    return hidden @ w2


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Transformer 架构</h1>
            <p>
            从编码器-解码器总览开始，逐步拆开自注意力、多头注意力、位置编码、
            残差归一化以及 BERT/GPT 的结构差异。页面里的矩阵和热力图都可交互，
            适合把论文公式变成可观察的计算过程。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_concept_cards() -> None:
    cards = [
        ("Token 表示", "文本先被切成 token，再映射成向量；后续所有计算都发生在向量空间。"),
        ("Q/K/V", "Query 问我要看哪里，Key 回答我是否相关，Value 提供真正被汇总的信息。"),
        ("并行建模", "每个 token 同时看整句，训练时不必像 RNN 那样一步步串行传递状态。"),
        ("结构复用", "BERT、GPT、T5 等模型的主要差异来自 mask、目标函数和编码器/解码器组合方式。"),
    ]
    html = '<div class="mini-grid">'
    for title, body in cards:
        html += f'<div class="mini-card"><strong>{title}</strong><p>{body}</p></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_attention_textbook_intro(tokens: list[str]) -> None:
    token_preview = "、".join(tokens[:8])
    st.markdown(
        dedent(
            f"""
            #### 一、什么是注意力机制？（直觉与定义）

            读一段文章时，人不会把每个词平均用力。看到“它”时，我们会回头寻找“它”指代谁；看到“因为”时，我们会注意前后的因果关系。**注意力机制**做的事情很像这种“给证据划重点”：当前词先提出一个问题，再给句子里的每个词分配权重，最后把重要词的信息按权重汇总回来。

            严谨地说，注意力机制是一种**根据 Query 与 Key 的相似度，对 Value 进行加权汇聚的可微分信息选择机制**。在 Transformer 中，自注意力让每个 token 都能直接读取同一句话里的其他 token，是模型理解上下文、建立长距离依赖的核心部件。

            当前演示文本被切成了这些 token：**{token_preview}**。页面里的所有矩阵行列，都围绕这些 token 展开。

            > 互动：先在左侧栏修改“输入文本”，换成一句你熟悉的话；再回到本页观察 token 列表和热力图是否一起改变。思考：为什么同一个词放在不同句子里，应该关注的对象会不同？

            #### 二、直观理解：它到底在做什么？

            下方的可视化把一次自注意力计算拆成 5 步。左图展示当前步骤产生的向量或矩阵，右图展示这个步骤最关键的关系结构。读注意力热力图时要记住三件事：**行表示正在思考的 Query token，列表示被比较的 Key token，颜色越深表示权重越大**。

            从黑盒效果看，自注意力把一句话中的每个词都改写成“带上下文的新表示”。例如一个词本来只是孤立的“it”，经过注意力后，它的向量里会混入前文名词、因果连接词、相邻词等信息。模型后续做分类、翻译或生成时，拿到的就不再是孤立词，而是已经读过上下文的词。

            > 互动：拖动“计算步骤”滑块，从第 1 步慢慢拖到第 5 步，像播放动画一样观察矩阵如何变化。重点看第 4 步：每一行 softmax 后都会变成一组加起来为 1 的注意力权重。
            """
        )
    )


def render_attention_math_section(d_model: int) -> None:
    st.markdown(
        dedent(
            r"""
            #### 三、深入本质：数学原理详解

            自注意力先把输入 token 向量矩阵 \(X\) 投影成三种角色：Query、Key、Value。
            """
        )
    )
    st.latex(r"Q = XW_Q,\quad K = XW_K,\quad V = XW_V")
    st.markdown(
        dedent(
            fr"""
            这里的 \(X\) 是输入 token 的向量表；\(W_Q\)、\(W_K\)、\(W_V\) 是模型学习出来的线性变换；\(Q\) 表示“我想找什么信息”，\(K\) 表示“我拥有什么线索可供匹配”，\(V\) 表示“如果别人关注我，真正拿走的内容”。当前演示的向量维度是 **d_model = {d_model}**。

            接着，模型用 Query 和 Key 做点积，得到每个 token 对其他 token 的匹配分数：
            """
        )
    )
    st.latex(r"\text{score}_{ij}=q_i \cdot k_j")
    st.markdown(
        dedent(
            r"""
            分数越高，表示第 \(i\) 个 token 越认为第 \(j\) 个 token 与自己相关。为了避免维度变大后点积数值过大，Transformer 会除以 \(\sqrt{d_k}\)，再用 softmax 把分数变成概率分布：
            """
        )
    )
    st.latex(r"A=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)")
    st.latex(r"\text{Attention}(Q,K,V)=AV=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    st.markdown(
        dedent(
            r"""
            这个公式可以按图中的步骤读：第 2 步产生 Q/K/V；第 3 步计算 \(QK^T\)；第 4 步缩放并 softmax，得到注意力矩阵 \(A\)；第 5 步用 \(A\) 对 Value 加权求和，生成新的上下文向量。

            > 互动：把“注意力演示维度”从 8 调到 64，再重复观察第 3、4、5 步。你会发现矩阵形状和数值模式会变化，因为更高维的向量能表达更多方向，但这个教学演示的随机投影也会带来不同的可视化纹理。
            """
        )
    )


def render_attention_experiment_guide(step: int, d_model: int) -> None:
    step_notes = {
        1: "第 1 步显示输入嵌入 X，它是 token 进入 Transformer 后的原始向量表示。",
        2: "第 2 步显示 Q/K/V 投影，同一个 token 会被拆成三种计算角色。",
        3: "第 3 步显示未缩放分数 QK^T，颜色强弱对应 token 之间的匹配程度。",
        4: "第 4 步显示 softmax 后的注意力权重，每一行都表示一个 token 的信息分配方案。",
        5: "第 5 步显示加权求和后的输出向量，它已经混合了上下文信息。",
    }
    st.markdown(
        dedent(
            f"""
            #### 四、动手实验：通过调参理解参数的作用

            **当前计算步骤：{step_notes[step]}**

            “输入文本”决定 token 序列本身。改变词序、加入“because / 因为 / 所以”等连接词，热力图中的强连接通常会重新分布，因为每个 token 面对的上下文证据变了。

            > 互动：在左侧栏把“输入文本”改成 `The dog chased the ball because it was excited`，再选择第 4 步。观察 `it` 这一行更容易把注意力分给哪些前文 token，并思考代词为什么需要回看上下文。

            “计算步骤”决定你正在看公式中的哪一层中间结果。第 1 步是输入，第 2 步是角色投影，第 3 步是相似度分数，第 4 步是归一化权重，第 5 步是最终输出。

            > 互动：把“计算步骤”从第 1 步拖到第 5 步。观察右图标题从“token 相似度”变成“每行 softmax 后权重和为 1”，再变成“输出向量”。这说明注意力不是一张静态图片，而是一条完整的可微分计算链。

            “注意力演示维度”现在是 **{d_model}**。维度越高，每个 token 可用的特征方向越多；但维度不是越大越好，真实工程中还要考虑显存、速度和过拟合风险。

            > 互动：把“注意力演示维度”依次设为 8、16、32、64，观察注意力矩阵的纹理变化。思考：为什么表示空间更大时，模型更有表达能力，但计算成本也更高？

            “多头注意力锐度”在“多头”页签中控制各个 head 的分布尖锐程度。数值低时，每个 head 更平均地看许多词；数值高时，head 会更像聚光灯。

            > 互动：切到“多头”页签，把“多头注意力锐度”从 0.5 拖到 3.0。观察局部邻近、前文依赖、连接词锚点、语义重复这些 head 是否变得更集中，并思考为什么“更集中”不总等于“更聪明”。

            “选择一个 query token”在“文本热力图”页签中决定右侧柱状图解释哪一行注意力。换一个 query token，相当于换一个正在发问的词。

            > 互动：切到“文本热力图”页签，分别选择名词、代词、连接词作为 query token。观察前三个关注对象如何变化，思考不同词为什么需要不同的信息来源。
            """
        )
    )


def render_attention_misconceptions() -> None:
    st.markdown(
        dedent(
            """
            #### 五、常见误区与易错点

            **误区 1：注意力权重越大，就一定是模型真正的解释。**  
            正确理解：注意力权重能提示模型在一次计算中“取了多少信息”，但它不是完整因果解释。模型后面还有残差连接、前馈网络、层归一化等计算。

            **误区 2：Q/K/V 就是数据库里的查询、键和值。**  
            正确理解：这个类比有帮助，但不能照搬。Q/K/V 都是从 token 向量线性投影出来的连续向量，不是人工写死的字段。

            **误区 3：注意力越集中越好。**  
            正确理解：翻译代词时可能需要集中看先行词，但总结一段话时可能需要分散整合多个线索。过度尖锐会丢失上下文。

            **误区 4：自注意力天然知道词语顺序。**  
            正确理解：裸的自注意力只看 token 之间的内容相似度，不知道“第几个词”。Transformer 必须加入位置编码或其他位置信息。

            **误区 5：多头注意力只是重复算很多遍。**  
            正确理解：不同 head 在不同子空间里学习关系，有的看邻近词，有的看语义重复，有的看因果连接。多头的价值在于并行捕捉多种关系。
            """
        )
    )


def render_attention_engineering_history() -> None:
    st.markdown(
        dedent(
            r"""
            #### 六、工程意义与应用场景

            注意力机制在工程上主要解决一个问题：**让模型在处理当前位置时，能直接读取远处相关信息**。这让机器翻译可以对齐源语言和目标语言，让 BERT/GPT 能根据上下文理解或生成文本，也让检索、重排序、代码补全等任务能在长文本中寻找关键证据。

            经典应用包括：机器翻译中的词对齐，BERT 类模型中的双向文本理解，GPT 类模型中的自回归生成。它的优点是上下文建模能力强、并行度高；边界是注意力矩阵随序列长度按 \(O(n^2)\) 增长，长文本会带来显存和速度压力。

            > 互动：把输入文本加长到 12 个 token 左右，观察右上角“注意力矩阵”指标如何从小矩阵变成 \(n \times n\)。思考：当 n 从 1,000 变成 10,000 时，为什么普通注意力会变得很贵？

            #### 七、历史小注

            现代神经网络注意力最早在 2014/2015 年由 Bahdanau、Cho、Bengio 等人在机器翻译中系统提出，用来缓解传统 seq2seq 把整句压成单个向量的瓶颈。2017 年 Vaswani 等人的 **Attention Is All You Need** 进一步证明，模型可以主要依靠自注意力替代循环结构，从而得到更高并行度和更强的长距离建模能力。
            """
        )
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: int = 10,
) -> None:
    rect = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color,
        edgecolor="white",
        linewidth=2,
        alpha=0.94,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#52616b") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def plot_transformer_architecture(num_layers: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(3.2, 7.55, "Encoder stack", ha="center", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    ax.text(9.7, 7.55, "Decoder stack", ha="center", fontsize=14, fontweight="bold", color=PALETTE["ink"])

    add_box(ax, (0.8, 0.45), 4.8, 0.62, "Input tokens + positional encoding", PALETTE["blue"], 10)
    add_box(ax, (7.3, 0.45), 4.8, 0.62, "Shifted target tokens + positional encoding", PALETTE["rose"], 10)

    encoder_y = [1.55, 2.75, 3.95, 5.15]
    decoder_y = [1.55, 2.75, 3.95, 5.15, 6.35]
    encoder_labels = [
        "Multi-head\nself-attention",
        "Add & LayerNorm",
        "Feed-forward\nnetwork",
        "Add & LayerNorm",
    ]
    decoder_labels = [
        "Masked multi-head\nself-attention",
        "Add & LayerNorm",
        "Cross-attention\n(Q from decoder)",
        "Feed-forward\nnetwork",
        "Linear + Softmax",
    ]
    encoder_colors = [PALETTE["teal"], PALETTE["green"], PALETTE["amber"], PALETTE["green"]]
    decoder_colors = [PALETTE["rose"], PALETTE["green"], PALETTE["violet"], PALETTE["amber"], PALETTE["blue"]]

    for y, label, color in zip(encoder_y, encoder_labels, encoder_colors):
        add_box(ax, (1.25, y), 3.9, 0.74, label, color)
    for y, label, color in zip(decoder_y, decoder_labels, decoder_colors):
        add_box(ax, (7.75, y), 3.9, 0.74, label, color)

    for y0, y1 in zip([1.07, *[y + 0.74 for y in encoder_y[:-1]]], encoder_y):
        add_arrow(ax, (3.2, y0), (3.2, y1))
    for y0, y1 in zip([1.07, *[y + 0.74 for y in decoder_y[:-1]]], decoder_y):
        add_arrow(ax, (9.7, y0), (9.7, y1))

    add_arrow(ax, (5.2, 4.35), (7.75, 4.35), PALETTE["violet"])
    ax.text(6.5, 4.6, "encoder memory\nK,V", ha="center", fontsize=10, color=PALETTE["violet"])

    ax.text(0.72, 4.65, f"x {num_layers}", fontsize=18, fontweight="bold", color=PALETTE["muted"])
    ax.text(12.0, 4.65, f"x {num_layers}", fontsize=18, fontweight="bold", color=PALETTE["muted"])

    ax.text(
        6.5,
        0.1,
        "训练机器翻译时，编码器读完整输入；解码器一边看已生成的目标词，一边通过 cross-attention 读取编码器输出。",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )
    fig.tight_layout()
    return fig


def plot_attention_step(pack: AttentionPack, step: int) -> go.Figure:
    titles = [
        "1. 输入嵌入 X",
        "2. 线性投影 Q, K, V",
        "3. 分数 QK^T",
        "4. 缩放后 softmax",
        "5. 加权求和 Attention(Q,K,V)",
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(titles[step - 1], "当前步骤的矩阵视图"),
        column_widths=[0.42, 0.58],
    )

    n = len(pack.tokens)
    token_labels = [f"{i}:{tok}" for i, tok in enumerate(pack.tokens)]
    if step == 1:
        left = pack.embeddings
        right = pack.embeddings @ pack.embeddings.T
        right_title = "token 相似度"
    elif step == 2:
        left = np.concatenate([pack.q, pack.k, pack.v], axis=1)
        right = np.corrcoef(left)
        right_title = "Q/K/V 拼接后的相关性"
    elif step == 3:
        left = pack.scores
        right = pack.scores
        right_title = "未缩放注意力分数"
    elif step == 4:
        left = pack.scaled_scores
        right = pack.weights
        right_title = "每行 softmax 后权重和为 1"
    else:
        left = pack.output
        right = pack.weights @ pack.v
        right_title = "输出向量"

    fig.add_trace(
        go.Heatmap(z=left, colorscale="RdBu", zmid=0, colorbar={"title": "value", "len": 0.78}),
        row=1,
        col=1,
    )
    colorscale = "Blues" if step == 4 else "RdBu"
    zmid = None if step == 4 else 0
    fig.add_trace(
        go.Heatmap(z=right, colorscale=colorscale, zmid=zmid, showscale=False),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="feature / key", row=1, col=1)
    fig.update_yaxes(tickmode="array", tickvals=list(range(n)), ticktext=token_labels, row=1, col=1)
    fig.update_xaxes(tickmode="array", tickvals=list(range(n)), ticktext=pack.tokens, title_text=right_title, row=1, col=2)
    fig.update_yaxes(tickmode="array", tickvals=list(range(n)), ticktext=pack.tokens, row=1, col=2)
    fig.update_layout(height=470, margin=dict(l=20, r=20, t=55, b=35), font=PLOT_FONT)
    return fig


def make_head_patterns(tokens: list[str], sharpness: float) -> dict[str, np.ndarray]:
    n = len(tokens)
    positions = np.arange(n)
    local = np.zeros((n, n))
    previous = np.zeros((n, n))
    delimiter = np.zeros((n, n))
    semantic = np.zeros((n, n))
    lower = [t.lower() for t in tokens]
    anchor_words = {"because", "therefore", "so", "if", "when", "and", "but", "因为", "所以"}

    for i in range(n):
        local[i] = np.exp(-np.abs(positions - i) / 1.4)
        previous[i] = np.where(positions <= i, np.exp(-(i - positions) / 2.0), 0.02)
        anchors = [j for j, tok in enumerate(lower) if tok in anchor_words]
        if not anchors:
            anchors = [0]
        delimiter[i] = 0.2 * np.exp(-np.abs(positions - i) / 3)
        for j in anchors:
            delimiter[i, j] += 1.4
        for j, tok in enumerate(lower):
            semantic[i, j] = 0.35 * np.exp(-abs(i - j) / 4)
            if tok == lower[i]:
                semantic[i, j] += 1.3
            if tok[:3] == lower[i][:3] and len(tok) >= 3:
                semantic[i, j] += 0.55

    patterns = {
        "局部窗口": local,
        "前文依赖": previous,
        "连接词锚点": delimiter,
        "语义重复": semantic,
    }
    return {name: softmax(matrix * sharpness, axis=-1) for name, matrix in patterns.items()}


def plot_multihead(tokens: list[str], sharpness: float) -> go.Figure:
    patterns = make_head_patterns(tokens, sharpness)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(patterns.keys()),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    for index, (title, weights) in enumerate(patterns.items()):
        row = index // 2 + 1
        col = index % 2 + 1
        fig.add_trace(
            go.Heatmap(
                z=weights,
                x=tokens,
                y=tokens,
                colorscale="Blues",
                zmin=0,
                zmax=max(0.35, float(weights.max())),
                showscale=index == 0,
                colorbar={"title": "weight", "len": 0.72},
            ),
            row=row,
            col=col,
        )
        entropy = -(weights * np.log(weights + 1e-9)).sum(axis=-1).mean()
        fig.add_annotation(
            text=f"平均熵 {entropy:.2f}",
            xref=f"x{index + 1 if index else ''} domain",
            yref=f"y{index + 1 if index else ''} domain",
            x=0.98,
            y=1.08,
            showarrow=False,
            font=dict(size=11, color=PALETTE["muted"]),
        )
    fig.update_layout(height=680, margin=dict(l=20, r=20, t=70, b=30), font=PLOT_FONT)
    fig.update_xaxes(tickangle=-35)
    return fig


def plot_positional_encoding(seq_len: int, d_model: int, dims: list[int]) -> go.Figure:
    enc = positional_encoding(seq_len, d_model)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("正弦/余弦位置编码曲线", "位置编码热力图"),
        column_widths=[0.56, 0.44],
    )
    x = np.arange(seq_len)
    for dim in dims:
        fig.add_trace(go.Scatter(x=x, y=enc[:, dim], mode="lines", name=f"dim {dim}"), row=1, col=1)
    fig.add_trace(
        go.Heatmap(z=enc.T, x=list(range(seq_len)), y=list(range(d_model)), colorscale="RdBu", zmid=0),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="position", row=1, col=1)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_xaxes(title_text="position", row=1, col=2)
    fig.update_yaxes(title_text="dimension", row=1, col=2)
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=55, b=35), font=PLOT_FONT)
    return fig


def simulate_residual_norm(depth: int, d_model: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, (18, d_model))
    plain = [x0]
    residual = [x0]
    residual_norm = [x0]
    for _ in range(depth):
        plain.append(feed_forward_like(plain[-1], rng))
        residual.append(residual[-1] + 0.55 * feed_forward_like(residual[-1], rng))
        residual_norm.append(layer_norm(residual_norm[-1] + 0.55 * feed_forward_like(residual_norm[-1], rng)))
    return np.array(plain), np.array(residual), np.array(residual_norm)


def plot_residual_norm(depth: int, d_model: int, seed: int) -> go.Figure:
    plain, residual, residual_norm = simulate_residual_norm(depth, d_model, seed)
    layers = np.arange(depth + 1)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("激活标准差随深度变化", "最后一层激活分布"),
        column_widths=[0.48, 0.52],
    )
    series = [
        ("无残差/无归一化", plain, PALETTE["rose"]),
        ("只加残差", residual, PALETTE["amber"]),
        ("残差 + LayerNorm", residual_norm, PALETTE["teal"]),
    ]
    for name, values, color in series:
        std = values.std(axis=(1, 2))
        fig.add_trace(go.Scatter(x=layers, y=std, mode="lines+markers", name=name, line=dict(color=color)), row=1, col=1)
        fig.add_trace(
            go.Histogram(x=values[-1].ravel(), name=name, opacity=0.58, marker_color=color, nbinsx=44),
            row=1,
            col=2,
        )
    fig.update_layout(
        height=470,
        barmode="overlay",
        margin=dict(l=20, r=20, t=55, b=35),
        font=PLOT_FONT,
        legend=dict(orientation="h", y=-0.16),
    )
    fig.update_xaxes(title_text="layer", row=1, col=1)
    fig.update_yaxes(title_text="std", row=1, col=1)
    fig.update_xaxes(title_text="activation value", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)
    return fig


def plot_text_attention(pack: AttentionPack, query_index: int) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("自注意力权重热力图", f"'{pack.tokens[query_index]}' 关注了谁"),
        column_widths=[0.58, 0.42],
    )
    fig.add_trace(
        go.Heatmap(
            z=pack.weights,
            x=pack.tokens,
            y=pack.tokens,
            colorscale="Blues",
            zmin=0,
            zmax=max(0.35, float(pack.weights.max())),
            colorbar={"title": "weight", "len": 0.78},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=pack.tokens,
            y=pack.weights[query_index],
            marker_color=[PALETTE["rose"] if i == query_index else PALETTE["blue"] for i in range(len(pack.tokens))],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=55, b=35), font=PLOT_FONT)
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title_text="query token", row=1, col=1)
    fig.update_yaxes(title_text="weight", range=[0, max(0.42, float(pack.weights[query_index].max()) * 1.18)], row=1, col=2)
    return fig


def plot_bert_gpt_masks(n: int) -> go.Figure:
    bidirectional = np.ones((n, n))
    causal = np.tril(np.ones((n, n)))
    labels = [f"t{i}" for i in range(n)]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("BERT: 双向可见", "GPT: 因果 mask"))
    fig.add_trace(go.Heatmap(z=bidirectional, x=labels, y=labels, colorscale="Greens", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=causal, x=labels, y=labels, colorscale="Reds", showscale=False), row=1, col=2)
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=30), font=PLOT_FONT)
    fig.update_xaxes(title_text="key/value position")
    fig.update_yaxes(title_text="query position")
    return fig


def plot_bert_gpt_architecture() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.8, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(2.9, 4.55, "BERT", ha="center", fontsize=16, fontweight="bold", color=PALETTE["green"])
    ax.text(9.1, 4.55, "GPT", ha="center", fontsize=16, fontweight="bold", color=PALETTE["rose"])

    for i, label in enumerate(["Token + Segment\n+ Position", "Encoder block\nbidirectional", "Encoder block\nbidirectional", "MLM / classification"]):
        add_box(ax, (0.8 + i * 1.35, 2.3), 1.15, 1.0, label, [PALETTE["blue"], PALETTE["green"], PALETTE["green"], PALETTE["amber"]][i], 8)
        if i:
            add_arrow(ax, (0.8 + (i - 1) * 1.35 + 1.15, 2.8), (0.8 + i * 1.35, 2.8))

    for i, label in enumerate(["Token + Position", "Decoder block\ncausal mask", "Decoder block\ncausal mask", "Next-token\nprediction"]):
        add_box(ax, (7.0 + i * 1.35, 2.3), 1.15, 1.0, label, [PALETTE["blue"], PALETTE["rose"], PALETTE["rose"], PALETTE["amber"]][i], 8)
        if i:
            add_arrow(ax, (7.0 + (i - 1) * 1.35 + 1.15, 2.8), (7.0 + i * 1.35, 2.8))

    ax.text(2.9, 1.45, "读完整句，适合理解任务", ha="center", color=PALETTE["muted"], fontsize=10)
    ax.text(9.1, 1.45, "只看过去，适合生成任务", ha="center", color=PALETTE["muted"], fontsize=10)
    fig.tight_layout()
    return fig


def render_overview(num_layers: int) -> None:
    st.subheader("1. Transformer 整体架构图")
    st.markdown(
        '<div class="note">原始 Transformer 是编码器-解码器架构：编码器把输入句子压成一组上下文向量，解码器在生成每个目标 token 时同时看已生成前缀和编码器记忆。</div>',
        unsafe_allow_html=True,
    )
    st.pyplot(plot_transformer_architecture(num_layers), clear_figure=True)


def render_self_attention(text: str, d_model: int, seed: int) -> AttentionPack:
    st.subheader("2. 自注意力机制的数学原理和计算过程")
    tokens = tokenize(text)
    pack = compute_attention(tokens, d_model=d_model, seed=seed)
    step = st.slider("计算步骤", 1, 5, 4, format="第 %d 步")
    render_attention_textbook_intro(tokens)
    render_attention_math_section(d_model)
    st.plotly_chart(plot_attention_step(pack, step), use_container_width=True, config=PLOT_CONFIG)
    col1, col2, col3 = st.columns(3)
    col1.metric("tokens", len(tokens))
    col2.metric("d_model", d_model)
    col3.metric("注意力矩阵", f"{len(tokens)} x {len(tokens)}")
    render_attention_experiment_guide(step, d_model)
    render_attention_misconceptions()
    render_attention_engineering_history()
    return pack


def render_multihead(pack: AttentionPack, sharpness: float) -> None:
    st.subheader("3. 多头注意力的不同关注点")
    st.markdown(
        '<div class="note">多头注意力不是简单重复同一张热力图。不同头可以学习局部相邻、前文依赖、连接词锚点、语义重复等不同关系，最后再拼接回模型维度。把左侧“多头注意力锐度”调低时，每个头会更像广角镜头；调高时，会更像聚光灯。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        dedent(
            """
            > 互动：先把“多头注意力锐度”设为 0.5，观察每个 head 是否保留较多候选词；再拖到 3.0，观察颜色是否集中到少数格子。思考：为什么真实模型需要多个 head 同时看局部、前文、连接词和语义重复，而不是只保留一个最强关系？
            """
        )
    )
    st.plotly_chart(plot_multihead(pack.tokens, sharpness), use_container_width=True, config=PLOT_CONFIG)


def render_positional() -> None:
    st.subheader("4. 位置编码的正弦/余弦函数图像")
    col1, col2 = st.columns([0.28, 0.72])
    with col1:
        seq_len = st.slider("序列长度", 16, 128, 64, 8)
        d_model = st.select_slider("位置编码维度", options=[16, 32, 64, 128], value=32)
        dims = st.multiselect("显示哪些维度", list(range(min(d_model, 16))), default=[0, 1, 2, 3])
        dims = dims or [0, 1]
        st.markdown(
            '<div class="callout">低维曲线变化快，擅长区分近距离；高维曲线变化慢，给长距离位置提供平滑坐标。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.plotly_chart(plot_positional_encoding(seq_len, d_model, dims), use_container_width=True, config=PLOT_CONFIG)


def render_residual_norm(seed: int) -> None:
    st.subheader("5. 残差连接与层归一化的作用")
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        depth = st.slider("堆叠层数", 2, 24, 12)
        d_model = st.select_slider("隐藏维度", options=[16, 32, 64, 128], value=64)
        st.markdown(
            '<div class="callout">残差让信息和梯度有短路路径；LayerNorm 把每个 token 的特征分布拉回稳定尺度，减少深层堆叠时的数值漂移。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.plotly_chart(plot_residual_norm(depth, d_model, seed), use_container_width=True, config=PLOT_CONFIG)


def render_text_heatmap(pack: AttentionPack) -> None:
    st.subheader("6. 输入文本的自注意力权重热力图")
    query_index = st.selectbox(
        "选择一个 query token",
        list(range(len(pack.tokens))),
        format_func=lambda i: f"{i}: {pack.tokens[i]}",
        index=min(2, len(pack.tokens) - 1),
    )
    st.markdown(
        dedent(
            """
            这张图专门解释“某一个词到底看向哪里”。左侧热力图保留完整注意力矩阵，右侧柱状图只抽出当前 query token 的那一行，所以它更适合逐词阅读。

            > 互动：切换“选择一个 query token”，观察右侧柱状图如何变化。名词、代词、连接词的关注对象通常不同；这说明注意力不是全句共用的一套权重，而是每个 token 都有自己的分配方案。
            """
        )
    )
    st.plotly_chart(plot_text_attention(pack, query_index), use_container_width=True, config=PLOT_CONFIG)
    strongest = np.argsort(pack.weights[query_index])[::-1][:3]
    summary = "、".join(f"{pack.tokens[i]}({pack.weights[query_index, i]:.2f})" for i in strongest)
    st.markdown(
        f'<div class="note">当前 token <strong>{pack.tokens[query_index]}</strong> 的前三个关注对象：{summary}。</div>',
        unsafe_allow_html=True,
    )


def render_bert_gpt() -> None:
    st.subheader("7. BERT vs GPT 的架构对比")
    st.pyplot(plot_bert_gpt_architecture(), clear_figure=True)
    st.plotly_chart(plot_bert_gpt_masks(8), use_container_width=True, config=PLOT_CONFIG)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["维度", "BERT", "GPT"],
                    fill_color="#eef4f2",
                    align="left",
                    font=dict(color=PALETTE["ink"], size=13),
                ),
                cells=dict(
                    values=[
                        ["核心结构", "注意力方向", "训练目标", "典型能力", "常见任务"],
                        ["Encoder-only", "双向注意力，能看左右文", "Masked Language Modeling", "理解、分类、抽取", "文本分类、NER、检索表征"],
                        ["Decoder-only", "因果注意力，只看过去", "Next Token Prediction", "续写、对话、工具调用", "生成、代码、聊天、推理"],
                    ],
                    fill_color="white",
                    align="left",
                    height=30,
                    font=dict(color=PALETTE["ink"], size=12),
                ),
            )
        ]
    )
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10), font=PLOT_FONT)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def main() -> None:
    render_hero()
    render_concept_cards()

    st.sidebar.header("交互参数")
    text = st.sidebar.text_area(
        "输入文本",
        value="The cat sat on the mat because it was tired",
        height=92,
        help="支持英文按词切分，中文会按字和标点切分。",
    )
    seed = st.sidebar.slider("随机种子", 0, 99, 7)
    d_model = st.sidebar.select_slider("注意力演示维度", options=[8, 16, 32, 64], value=16)
    num_layers = st.sidebar.slider("架构图层数标注", 1, 12, 6)
    head_sharpness = st.sidebar.slider("多头注意力锐度", 0.5, 3.0, 1.4, 0.1)

    tab_names = ["总览", "自注意力", "多头", "位置编码", "残差归一化", "文本热力图", "BERT vs GPT"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_overview(num_layers)
    with tabs[1]:
        pack = render_self_attention(text, d_model, seed)
    with tabs[2]:
        pack = compute_attention(tokenize(text), d_model=d_model, seed=seed)
        render_multihead(pack, head_sharpness)
    with tabs[3]:
        render_positional()
    with tabs[4]:
        render_residual_norm(seed)
    with tabs[5]:
        pack = compute_attention(tokenize(text), d_model=d_model, seed=seed)
        render_text_heatmap(pack)
    with tabs[6]:
        render_bert_gpt()


if __name__ == "__main__":
    main()
