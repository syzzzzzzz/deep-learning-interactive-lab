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
            fr"""
            #### 一、什么是注意力机制？（直觉与定义）

            读一段文章时，人不会把每个词平均用力。看到“它”时，我们会回头寻找“它”指代谁；看到“because / 因为”时，我们会注意前后的因果关系；看到重复出现的关键词时，我们会把它们连成同一条线索。页面左侧的“输入文本”就是这个类比的实验入口：你输入的每个 token，都会在下方热力图中拥有一行自己的注意力分配。

            换三个角度看，它分别像三种工程动作。第一，它像阅读时的**划重点**：在“文本热力图”页签选择一个 query token，就能看到当前词把重点画给了谁。第二，它像搜索系统里的**检索排序**：Query 与 Key 的点积分数越高，第 4 步 softmax 后的颜色通常越深。第三，它像团队协作里的**信息汇总**：第 5 步不是只取一个词，而是按权重把多个 Value 混合成新的上下文向量。

            严谨地说，注意力机制至少有两种常见定义。表示学习视角会说：它是**根据 Query 与 Key 的相似度，对 Value 进行加权汇聚的可微分信息选择机制**；信息检索视角会说：它是**用当前查询在上下文记忆表中寻找相关证据，并把证据按置信度聚合的机制**。两者差别很细：前者强调所有操作都能反向传播，所以你会在“计算步骤”里看到 Q/K/V、softmax、输出向量连成一条计算链；后者强调“相关证据”的排序，所以你会在“文本热力图”中看到每一行都像一次检索结果。

            在知识体系中的位置可以这样读：

            ```text
            词向量 / 线性代数 -> Q/K/V 线性投影 -> 缩放点积注意力 -> 多头注意力 -> Transformer block -> BERT / GPT
            ```

            这张“位置图”正好对应当前页面的页签：左侧“输入文本”提供词向量入口，“自注意力”页签展示 Q/K/V 和 5 步计算，“多头”页签展示多个关系子空间，“位置编码”和“残差归一化”页签补上 Transformer block 的另外两根支柱。

            如果没有注意力机制，深度学习处理长文本时会更像早期 RNN：远处信息必须一站一站传递，越传越容易稀释；你在页面里把“输入文本”加长时，普通自注意力仍然能让最后一个 token 直接看第一个 token，这正是它改变序列建模的关键。

            当前演示文本被切成了这些 token：**{token_preview}**。页面里的所有矩阵行列，都围绕这些 token 展开。

            > 互动：先在左侧栏修改“输入文本”，换成一句你熟悉的话；再回到本页观察 token 列表和热力图是否一起改变。思考：为什么同一个词放在不同句子里，应该关注的对象会不同？
            >
            > 进阶思考：如果你把句子改成只有 3 个 token，注意力矩阵会变得很小；如果改成 12 个 token，右上角“注意力矩阵”指标会明显变大。这个变化暗示了什么计算成本？

            #### 二、直观理解：它到底在做什么？

            下方的可视化把一次自注意力计算拆成 5 步。左图展示当前步骤产生的向量或矩阵，右图展示这个步骤最关键的关系结构。读注意力热力图时要记住三件事：**行表示正在思考的 Query token，列表示被比较的 Key token，颜色越深表示权重越大**。第 4 步右图使用蓝色深浅表示 softmax 后的权重，深蓝格子代表“当前行的 token 更依赖这一列 token”；第 3 步的红蓝色则表示未归一化分数，颜色正负和大小都还没有变成概率。

            这张图可以逐帧读。第 1 步，“输入嵌入 X”显示每个 token 的初始向量，右图的 token 相似度只是原始表示之间的粗略关系。第 2 步，同一个 token 被投影成 Q/K/V 三种角色，左图会横向拼接三块向量。第 3 步，\(QK^T\) 产生一张 n x n 分数表，行列都对应当前 token 列表。第 4 步，分数被缩放并 softmax，每一行变成总和为 1 的权重分布。第 5 步，权重乘上 Value，得到每个 token 的上下文输出。

            从黑盒效果看，自注意力把一句话中的每个词都改写成“带上下文的新表示”。例如一个词本来只是孤立的“it”，经过注意力后，它的向量里会混入前文名词、因果连接词、相邻词等信息。模型后续做分类、翻译或生成时，拿到的就不再是孤立词，而是已经读过上下文的词。

            也要看懂这个教学图的局限性。当前页面为了让变化稳定可见，用随机投影加少量启发式关系生成演示矩阵，所以它不是某个真实大模型的内部注意力截图；真实模型的 Q/K/V 权重来自大量语料训练，并且每层、每个 head 都会出现不同模式。不过图中“行看列、softmax 归一化、按 Value 加权求和”的计算形状，与真实 Transformer 是一致的。

            > 互动：拖动“计算步骤”滑块，从第 1 步慢慢拖到第 5 步，像播放动画一样观察矩阵如何变化。重点看第 4 步：每一行 softmax 后都会变成一组加起来为 1 的注意力权重。
            >
            > 反例实验：把“计算步骤”停在第 3 步，只看未缩放分数；再切到第 4 步。你会发现第 3 步的数值还不能直接当作“分配比例”，因为它们既可能为负，也不会每行加起来等于 1。没有 softmax，第 5 步就失去了“按比例汇总 Value”的稳定含义。
            >
            > 进阶思考：在“文本热力图”页签里切换不同 query token，为什么同一列 token 对某些行很重要、对另一些行却不重要？
            """
        )
    )


def render_attention_math_section(d_model: int) -> None:
    st.markdown(
        dedent(
            r"""
            #### 三、深入本质：数学原理详解

            数学从最朴素的问题开始：当前 token 想从句子里找信息，应该怎么判断“谁和我相关”？如果只看 token 原始向量 \(X\)，每个词只有一种身份；但在页面第 2 步里，同一个词会被拆成 Query、Key、Value 三种角色。这不是故意复杂化，而是为了把“发问”“被匹配”“被取走的内容”分开学习。
            """
        )
    )
    st.latex(r"Q = XW_Q,\quad K = XW_K,\quad V = XW_V")
    st.markdown(
        dedent(
            fr"""
            这里的 \(X\) 是输入 token 的向量表；\(W_Q\)、\(W_K\)、\(W_V\) 是模型学习出来的线性变换；\(Q\) 表示“我想找什么信息”，\(K\) 表示“我拥有什么线索可供匹配”，\(V\) 表示“如果别人关注我，真正拿走的内容”。当前演示的向量维度是 **d_model = {d_model}**。

            为什么这里是矩阵乘法，而不是给每个词手写规则？因为矩阵乘法能把每个 token 投影到不同的特征方向，而且这些方向能通过训练自动调整。你把“注意力演示维度”从 8 调到 64 时，看到的纹理变化就来自这个表示空间变大：维度越高，Q/K/V 可以携带的方向越多，但后面的 \(QK^T\) 也会更容易出现较大的数值。

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

            为什么要除以 \(\sqrt{d_k}\)，而不是除以 \(d_k\) 或者不除？如果 Q 和 K 的每个维度方差大约为 1，那么 \(q_i \cdot k_j\) 是 \(d_k\) 个随机项相加，典型尺度会随 \(\sqrt{d_k}\) 增大。除以 \(\sqrt{d_k}\) 可以把第 3 步分数拉回稳定范围，让第 4 步 softmax 不至于过早变成“几乎只看一个词”。如果不缩放，你把“注意力演示维度”调大时，权重会更容易尖锐，训练中梯度也更容易不稳定；如果缩放过强，权重又会过平，模型难以聚焦关键证据。

            数值上可以手算一行。假设某个 query 和三个 key 的点积是 \([4, 2, 0]\)，并且 \(d_k=4\)，缩放后变成 \([2, 1, 0]\)。softmax 约为 \([0.665, 0.245, 0.090]\)。如果不除以 \(\sqrt{4}\)，softmax(\([4,2,0]\)) 约为 \([0.867, 0.117, 0.016]\)，第一个词会过度压倒其他词。你在第 4 步看到的蓝色权重，就是这种“分数差距被 softmax 翻译成比例”的结果。

            哪个项影响最大？短句里，输入 token 与演示启发式会让相邻词、重复词、连接词更容易变深；真实模型里，影响最大的是训练后学到的 \(W_Q,W_K,W_V\) 和上下文 token 本身。\(V\) 不决定“看谁”，但决定“看到了什么内容”；所以只看第 4 步权重还不够，第 5 步输出向量才是模型真正传给后续层的表示。

            > 互动：把“注意力演示维度”从 8 调到 64，再重复观察第 3、4、5 步。你会发现矩阵形状和数值模式会变化，因为更高维的向量能表达更多方向，但这个教学演示的随机投影也会带来不同的可视化纹理。
            >
            > 对比辨析：点积注意力和余弦相似度都在比较方向，但当前页面第 3 步使用的是点积，所以向量长度也会影响分数；softmax 和普通归一化都能把数值压到某个范围，但 softmax 会指数放大高分差距，因此更适合做“可微分选择”。
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
            fr"""
            #### 四、动手实验：通过调参理解参数的作用

            **当前计算步骤：{step_notes[step]}**

            “输入文本”决定 token 序列本身。这个控件没有固定上限，但教学上建议先输入 **6 到 14 个 token**：太短时热力图几乎看不出上下文关系，太长时矩阵会变密，新手很难判断哪一格重要。默认句子适合作为基准，因为它同时包含名词、动词和上下文依赖；换成带有“because / 因为 / 所以”的句子时，连接词附近的格子通常会更深，因为它们提供了因果或转折线索。

            可以把“输入文本”的效果分成四段看：3 个 token 以下主要是在观察矩阵形状；4 到 8 个 token 适合学习局部相邻关系；9 到 16 个 token 开始能看到远距离回看；超过 16 个 token 时，普通注意力的 \(n \times n\) 成本会更明显。工程里做文本实验时，90% 的早期排查都会先用短句，因为短句能快速看清 token 是否被正确切分、注意力是否出现异常尖峰。

            > 互动：在左侧栏把“输入文本”改成 `The dog chased the ball because it was excited`，再选择第 4 步。观察 `it` 这一行更容易把注意力分给哪些前文 token，并思考代词为什么需要回看上下文。
            >
            > 极端值测试：把“输入文本”改成只有 `dog runs` 两个词，再改成长到 12 个 token 左右的句子。观察右上角“注意力矩阵”指标和热力图密度如何变化。思考：为什么句子变长时，理解能力变强的同时，计算也会迅速变贵？

            “计算步骤”的完整范围是 **1 到 5**，默认建议停在 **第 4 步**，因为第 4 步最接近人们口中的“注意力权重”。第 1 步适合检查输入向量是否形成基本相似性；第 2 步适合理解 Q/K/V 的角色分工；第 3 步适合看未归一化分数，数值可能为负且行和不固定；第 4 步适合看 softmax 后的概率分配；第 5 步适合理解最终被送往后续网络的上下文输出。

            这五步之间不能跳着理解。很多白屏或“看不出变化”的挫败感，本质上来自把第 3 步分数当成第 4 步权重，或者只盯着热力图颜色却忘了第 5 步还要乘上 Value。实际项目里排查注意力异常时，我通常按 **第 2 步形状 -> 第 3 步数值范围 -> 第 4 步是否过尖 -> 第 5 步输出是否爆炸** 的顺序看，速度最快。

            > 互动：把“计算步骤”从第 1 步拖到第 5 步。观察右图标题从“token 相似度”变成“每行 softmax 后权重和为 1”，再变成“输出向量”。这说明注意力不是一张静态图片，而是一条完整的可微分计算链。
            >
            > 对比实验：停在第 3 步时找一个颜色很深的格子，再切到第 4 步看它是否仍然占主导。思考：softmax 为什么会改变“看起来很大”的分数之间的相对差距？

            “注意力演示维度”的完整可选范围是 **8、16、32、64**，当前值是 **{d_model}**，默认值是 **16**。8 维像一张很薄的草图，便于观察但表达力有限；16 维是本页推荐默认值，变化清楚且不至于太随机；32 维开始更接近真实模型的表示习惯，能容纳更多关系方向；64 维会让表示空间更丰富，但演示里的随机投影也更容易带来纹理变化，不能把每次颜色差异都解释成语义规律。

            维度与其他控件会互相影响。维度升高后，第 3 步 \(QK^T\) 的原始分数更容易拉开距离，所以要特别观察第 4 步是否过度尖锐；如果你同时把“多头注意力锐度”调高，热力图会更像只保留少数格子。工业界不会只因为“维度更大”就盲目加宽模型，通常先选一个能跑得动的基线，再根据验证集效果和显存占用上下浮动 20% 到 50%。

            > 互动：把“注意力演示维度”依次设为 8、16、32、64，观察注意力矩阵的纹理变化。思考：为什么表示空间更大时，模型更有表达能力，但计算成本也更高？
            >
            > 极端值测试：把维度设为 8 后观察第 4 步，再设为 64 后观察同一位置。不要只问“哪张更好看”，要问：颜色更复杂时，模型是否真的更稳定、更可解释？

            “多头注意力锐度”的范围是 **0.5 到 3.0**，默认值是 **1.4**。0.5 到 1.0 的区间像广角镜头，多个 token 都能保留一定权重，适合观察“整合上下文”；1.1 到 1.8 是教学上最平衡的区间，既能看出重点，又不会只剩一个格子；1.9 到 2.5 会明显聚焦，适合观察某个 head 是否学到局部邻近或连接词锚点；2.6 到 3.0 是极端聚焦区，视觉上很爽，但可能丢掉多证据任务需要的细节。

            锐度和 query token 的选择也会互相影响。同样的锐度下，代词可能需要强烈回看一个名词，而总结性词语可能需要分散关注多个上下文位置。真实项目中如果注意力分布长期接近 one-hot，我会先怀疑温度、初始化或 mask；如果长期接近平均分布，我会检查模型是否没有学到有效的 Q/K 匹配。

            > 互动：切到“多头”页签，把“多头注意力锐度”从 0.5 拖到 3.0。观察局部邻近、前文依赖、连接词锚点、语义重复这些 head 是否变得更集中，并思考为什么“更集中”不总等于“更聪明”。
            >
            > 极端值测试：先把锐度设为 0.5，看每个 head 是否像“平均听大家发言”；再设为 3.0，看它是否像“只听一个人”。思考：做摘要、问答、翻译时，哪类任务更害怕过度集中？

            “选择一个 query token”在“文本热力图”页签中决定右侧柱状图解释哪一行注意力。它的取值范围就是当前“输入文本”切出的所有 token，默认通常是靠前的 token。换一个 query token，相当于换一个正在发问的词：名词常看修饰语或重复词，动词常看主语和宾语，代词常看先行词，连接词常看前后两个片段。

            这个控件最适合训练“逐行读热力图”的习惯。不要只看整张图中最深的颜色，而要问：当前这一行是谁在发问？它为什么需要这些列的信息？工程调试时，如果所有 query token 的关注对象都几乎一样，往往说明模型退化成了全局模板；如果每个 token 都只看自己，可能说明上下文没有被有效使用。

            > 互动：切到“文本热力图”页签，分别选择名词、代词、连接词作为 query token。观察前三个关注对象如何变化，思考不同词为什么需要不同的信息来源。
            >
            > 进阶思考：当你把“输入文本”改短时，“选择一个 query token”的候选项会减少；当你把文本改长时，候选项会增加。这个现象为什么说明注意力矩阵的行列都来自同一组 token？
            """
        )
    )


def render_attention_misconceptions() -> None:
    st.markdown(
        dedent(
            r"""
            #### 五、常见误区与易错点

            **误区 1：注意力权重越大，就一定是模型真正的解释。**  
            正确理解：注意力权重能提示模型在一次计算中“取了多少信息”，但它不是完整因果解释。初学者容易犯这个错，是因为“文本热力图”的颜色非常直观，深色格子看起来像模型给出的理由；但在真实 Transformer 里，后面还有残差连接、前馈网络、层归一化和下一层注意力继续改写表示。具体症状是：你会发现某个 token 权重很高，但最终预测并没有按这个 token 的直觉变化。排查时先切到第 4 步看权重，再切到第 5 步看输出向量是否真的变化，最后不要忘记查看后续层或最终 logits。我曾在做文本分类解释页时踩过这个坑，只展示第 4 步热力图，结果用户以为深色词就是唯一证据，后来必须补上“权重只是线索，不是判决书”的说明。

            **误区 2：Q/K/V 就是数据库里的查询、键和值。**  
            正确理解：这个类比有帮助，但不能照搬。Q/K/V 都是从 token 向量线性投影出来的连续向量，不是人工写死的字段。初学者会误会，是因为 Query、Key、Value 这三个名字本来就来自检索系统；但页面第 2 步已经显示，同一个 token 会同时拥有三种角色。踩坑后的症状是：会试图给每个词手工指定 Key 或 Value，或者以为 Value 一定是原始词本身。排查步骤是回到第 2 步，确认 Q/K/V 的形状都来自同一个输入 \(X\)，再看第 5 步如何把 Value 混合成输出。我在做一个客服问答 demo 时见过类似问题：团队把 Q/K/V 当作三份不同表格维护，模型解释越写越乱；改成“同一 token 的三种投影角色”后，整个页面才讲得通。

            **误区 3：注意力越集中越好。**  
            正确理解：翻译代词时可能需要集中看先行词，但总结一段话时可能需要分散整合多个线索。这个误区来自视觉偏好：深色热力图看起来更“有决断力”，所以很多人会把“多头注意力锐度”一路拖到 3.0。具体症状是：模型在需要多证据的任务上变得武断，生成文本容易漏掉限定条件，问答时只抓住一个关键词。排查时把锐度从 3.0 降到 1.4，再观察多个 head 是否恢复分工；如果第 4 步每一行都接近 one-hot，就要怀疑温度或分数尺度过大。我曾在长文摘要实验里见过模型过度盯标题词，摘要看似流畅但漏掉正文限制条件，最后通过降低注意力温度并增加验证集检查才稳定下来。

            **误区 4：自注意力天然知道词语顺序。**  
            正确理解：裸的自注意力只看 token 之间的内容相似度，不知道“第几个词”。Transformer 必须加入位置编码或其他位置信息。初学者会误会，是因为热力图天然按词序排成方阵，看起来模型似乎已经知道左右顺序；但那只是页面布局，不是数学公式自带的顺序信息。具体症状是：去掉位置编码后，词序敏感任务会明显变差，例如“狗追人”和“人追狗”可能被混得更像。排查时切到“位置编码”页签，观察位置向量如何给每个 token 注入位置信号，再回到“自注意力”页签比较 token 顺序变化后的热力图。我做早期 Transformer 教学代码时故意关掉位置编码，训练曲线还能下降，但生成句子顺序混乱，这正是这个误区的典型表现。

            **误区 5：多头注意力只是重复算很多遍。**  
            正确理解：不同 head 在不同子空间里学习关系，有的看邻近词，有的看语义重复，有的看因果连接。多头的价值在于并行捕捉多种关系。这个误区来自“多头”页签里多个热力图长得都像方阵，于是看起来像复制粘贴；但你调节“多头注意力锐度”并逐张看 head，会发现它们的关注模式并不一样。踩坑症状是：为了省计算粗暴减少 head 数，模型在复杂句法、跨句指代或代码补全任务上掉点。排查时不要只看平均注意力，要分别查看每个 head 的局部邻近、前文依赖、连接词锚点和语义重复模式。一个常见工程案例是小模型蒸馏时把 head 裁得太狠，验证集总体损失变化不大，但长距离依赖题明显变差。

            > 互动：完成这一节后，请依次操作三个控件排查误区：先在“计算步骤”比较第 4 步和第 5 步，再在“多头”页签调节“多头注意力锐度”，最后在“位置编码”页签观察顺序信息。思考：你刚才看到的是模型计算的哪一部分，而不是哪一部分？
            """
        )
    )


def render_attention_engineering_history() -> None:
    st.markdown(
        dedent(
            r"""
            #### 六、工程意义与应用场景

            注意力机制在工程上主要解决一个问题：**让模型在处理当前位置时，能直接读取远处相关信息**。这让机器翻译可以对齐源语言和目标语言，让 BERT/GPT 能根据上下文理解或生成文本，也让检索、重排序、代码补全等任务能在长文本中寻找关键证据。

            在实际项目里，注意力通常按这条流程落地：先把“输入文本”切成 token，得到嵌入矩阵 \(X\)；再用第 2 步的线性层生成 Q/K/V；接着用第 3、4 步得到权重矩阵；最后用第 5 步输出上下文表示，并把它交给分类头、解码器或检索重排序模块。页面里的“计算步骤”就是这条工程流水线的剖面图，调试时按这个顺序查，最容易定位错误。

            经典应用包括：机器翻译中的词对齐，BERT 类模型中的双向文本理解，GPT 类模型中的自回归生成。它的优点是上下文建模能力强、并行度高；边界是注意力矩阵随序列长度按 \(O(n^2)\) 增长，长文本会带来显存和速度压力。具体地说，序列长度为 \(n\)、单头维度为 \(d_k\) 时，计算 \(QK^T\) 的时间复杂度约为 \(O(n^2 d_k)\)，保存注意力矩阵的空间复杂度约为 \(O(n^2)\)。所以你把“输入文本”从 8 个 token 加到 16 个 token，矩阵格子不是翻倍，而是从 64 个变成 256 个。

            > 互动：把输入文本加长到 12 个 token 左右，观察右上角“注意力矩阵”指标如何从小矩阵变成 \(n \times n\)。思考：当 n 从 1,000 变成 10,000 时，为什么普通注意力会变得很贵？

            一个最小可运行的 PyTorch 版本如下，它正好对应页面的第 2 到第 5 步：

            ```python
            import torch
            import torch.nn.functional as F

            x = torch.randn(2, 6, 16)       # batch=2, token=6, d_model=16
            w_q = torch.randn(16, 16)
            w_k = torch.randn(16, 16)
            w_v = torch.randn(16, 16)
            q = x @ w_q                    # 第 2 步：生成 Query
            k = x @ w_k                    # 第 2 步：生成 Key
            v = x @ w_v                    # 第 2 步：生成 Value
            scores = q @ k.transpose(-2, -1) / (16 ** 0.5)
            weights = F.softmax(scores, dim=-1)
            out = weights @ v              # 第 5 步：按权重汇总 Value
            print(out.shape)               # torch.Size([2, 6, 16])
            ```

            逐行看，`x` 对应页面中的输入 token 向量；`w_q/w_k/w_v` 对应三组可学习投影；`q/k/v` 对应 Q/K/V 表格；`scores` 对应第 3 步缩放后的匹配分数；`weights` 对应第 4 步蓝色热力图；`out` 对应第 5 步上下文输出。真实工程中这些权重不会用 `torch.randn` 手写，而是在训练中被优化器不断更新。

            与相近技术相比，RNN 按时间步传递状态，适合流式处理但长距离信息容易衰减；CNN 用固定窗口提取局部模式，速度快但全局依赖需要堆很多层；稀疏注意力只计算部分格子，能处理更长文本但设计 mask 更复杂；线性注意力或检索增强方法试图绕开 \(O(n^2)\)，但在精确对齐和通用性上常有取舍。工业界的经验是：中短文本优先用标准注意力，因为它稳定、好调、生态成熟；长文档或多轮对话才重点考虑稀疏、分块、缓存或检索增强。

            未来改进主要围绕三件事：降低长序列成本、让注意力解释更可靠、让模型更好地结合外部记忆。你在本页看到的“热力图颜色”“多头锐度”“query token 选择”，都是理解这些改进的起点：如果不知道普通注意力怎样花钱、怎样聚焦、怎样出错，就很难判断新结构到底改进了什么。

            > 进阶思考：当你把“多头注意力锐度”调高时，视觉上更清晰；当你把“输入文本”加长时，计算上更昂贵。真实产品里，为什么“看起来更可解释”和“系统更可靠”不一定是同一件事？

            #### 七、历史小注

            注意力出现之前，主流 seq2seq 翻译模型常把整句源语言压进一个固定长度向量，再让解码器从这个向量里生成目标句。这个做法在短句上还能工作，但句子一长，前半句信息就容易被压扁，像把整本书塞进一张便签。你在本页把“输入文本”加长时仍能看到任意两词直接形成格子，这正是注意力对旧瓶颈的反击：不要把所有信息挤进一个瓶口，而是让每个位置按需读取全局。

            现代神经网络注意力最早在 2014/2015 年由 Bahdanau、Cho、Bengio 等人在机器翻译中系统提出，用来缓解传统 seq2seq 把整句压成单个向量的瓶颈。早期争议主要来自两个问题：一是它比单向 RNN 多出对齐矩阵，计算和实现更复杂；二是很多人还不确定这种“软对齐”是否能稳定学出语言结构。后来注意力在翻译对齐上的可视化效果非常有说服力，因为热力图能显示源词和目标词之间的对应关系，这与本页“行看列”的读图方式一脉相承。

            2017 年 Vaswani 等人的 **Attention Is All You Need** 进一步证明，模型可以主要依靠自注意力替代循环结构，从而得到更高并行度和更强的长距离建模能力。关键里程碑可以这样读：2014/2015 年注意力解决 seq2seq 压缩瓶颈；2017 年 Transformer 把自注意力推到架构中心；2018 年 BERT 证明双向 Transformer 可以成为通用文本理解底座；2018 年之后 GPT 系列证明自回归 Transformer 可以随着数据和算力扩展出强大的生成能力。

            Bahdanau、Cho、Bengio 的贡献不只是提出一个公式，而是把“可微分对齐”带进神经机器翻译；Vaswani 等人的贡献不只是堆叠注意力层，而是把并行计算、位置编码、多头子空间和残差归一化组织成可扩展的工程结构。页面中的“自注意力”“多头”“位置编码”“残差归一化”四个页签，正好对应这条历史演化留下的核心部件。

            > 互动：最后请按历史顺序重看页面：先在“自注意力”页签理解单次按需读取，再到“多头”页签看多关系并行，接着到“位置编码”页签看顺序信息，最后到“残差归一化”页签看深层训练稳定性。思考：为什么 Transformer 不是只靠一个公式成功，而是靠一组互相配合的结构成功？
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
