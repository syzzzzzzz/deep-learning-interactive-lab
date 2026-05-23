"""
Interactive math primer for deep learning.

Run:
    streamlit run part1_foundations/math_primer.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from textwrap import dedent
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="深度学习数学基础速查",
    layout="wide",
    initial_sidebar_state="expanded",
)


INK = "#172026"
MUTED = "#58646d"
TEAL = "#0f8b8d"
ROSE = "#c73e5b"
AMBER = "#d99a22"
GREEN = "#477b44"
VIOLET = "#5e4ae3"
BLUE = "#2d6cdf"
PAPER = "#fbfaf6"
LINE = "#d7dde1"


st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(238,245,243,0.97) 100%),
            #fbfaf6;
        color: #172026;
    }
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2.6rem;
    }
    h1, h2, h3 { letter-spacing: 0; }
    section[data-testid="stSidebar"] {
        background: #eef4f1;
        border-right: 1px solid #d7dde1;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.72rem;
    }
    .hero {
        border-bottom: 1px solid #d7dde1;
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3rem);
        margin: 0;
    }
    .hero p {
        color: #58646d;
        font-size: 1rem;
        line-height: 1.72;
        max-width: 1080px;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.74);
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 0.9rem;
        color: #26343b;
        line-height: 1.65;
        margin: 0.4rem 0 0.95rem 0;
    }
    .formula {
        background: rgba(255,255,255,0.78);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.8rem 0.95rem;
        color: #172026;
        line-height: 1.72;
        margin: 0.35rem 0 0.9rem 0;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.4rem 0 0.95rem 0;
    }
    .mini-cell {
        background: rgba(255,255,255,0.74);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.68rem 0.76rem;
        min-height: 84px;
    }
    .mini-cell strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #172026;
    }
    .mini-cell span {
        color: #58646d;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    @media (max-width: 900px) {
        .mini-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def formula(text: str) -> None:
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


def concept_cards(cards: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="mini-cell"><strong>{title}</strong><span>{text}</span></div>'
        for title, text in cards
    )
    st.markdown(f'<div class="mini-grid">{body}</div>', unsafe_allow_html=True)


def render_math_learning_map() -> None:
    st.markdown(
        dedent(
            r"""
            #### 这张速查表应该怎么用？

            深度学习里的数学不是为了做题，而是为了回答四个工程问题：**数据怎么表示、误差怎么变化、预测有多不确定、参数怎么更新**。本页把这些问题拆成线性代数、微积分、概率论和梯度下降四块；每一块都配了可调图形，用来把公式变成看得见的几何运动。

            把它放进一次训练循环里，可以这样读：

            ```text
            输入向量 x
              -> 线性代数：z = Wx + b，把特征混合成新的表示
              -> 概率/激活：把 z 变成概率、采样或不确定性
              -> 损失函数：把预测错误压成一个标量 L
              -> 微积分：用链式法则求 dL/dW
              -> 梯度下降：W <- W - eta * grad，让下一次预测更好
            ```

            所以本页不是公式仓库，而是一张**训练循环地图**。你在“线性代数基础”里拖动向量和矩阵，本质是在观察前向传播如何搬运信息；你在“微积分基础”里调观察点和割线步长，本质是在观察反向传播怎样获得局部斜率；你在“概率论基础”里调先验、分布和采样数量，本质是在观察模型如何表达不确定性；你在“梯度下降几何直觉”里调学习率和动量，本质是在观察优化器怎样真正改参数。

            > 互动：建议先从“线性代数基础”开始，拖动向量和矩阵；再切到“微积分基础”观察斜率；最后看“概率论基础”和“梯度下降几何直觉”。每切一个知识点，都先问自己：这个公式在神经网络训练中负责哪一步？
            >
            > 进阶思考：如果只会背公式，却不知道它出现在训练循环的哪一环，调参时会发生什么？请带着这个问题看每个滑块的变化。
            """
        )
    )
    concept_cards(
        [
            ("线性代数", "回答“样本和参数如何表示”：向量是特征，矩阵是特征混合与空间变换。"),
            ("微积分", "回答“误差如何改变”：导数和梯度告诉参数往哪个方向动。"),
            ("概率与优化", "回答“预测有多可信、模型如何变好”：分布描述不确定性，梯度下降执行更新。"),
        ]
    )


def render_linear_algebra_overview() -> None:
    st.markdown(
        dedent(
            r"""
            **直觉与位置：**向量可以看成一支箭头，也可以看成一条样本的特征列表；矩阵可以看成一台机器，把整张坐标纸一起拉伸、旋转、翻转。神经网络中的线性层 \(y=Wx+b\)，本质上就是用矩阵 \(W\) 重新组织特征，再用偏置 \(b\) 平移。它位于训练循环的第一步：**把原始输入变成模型能继续处理的表示**。

            本页的两个图刚好对应线性代数的两种读法。上半部分的 `u_x/u_y/v_x/v_y/缩放系数 alpha` 用箭头解释“一个样本或一个权重方向”如何相加、缩放、对齐；下半部分的 `a,b,c,d/输入向量 x/y` 用网格解释“一个线性层”如何把整个特征空间一起变形。前者帮助理解点积相似度，后者帮助理解矩阵乘法和线性层。

            > 互动：先只拖动 `u_x` 和 `u_y`，观察向量 `u` 的方向和长度；再拖动 `v_x`、`v_y`，观察 `u + v` 如何按“首尾相接”形成。最后看 `u · v` 和 `cos(theta)`，思考为什么点积能衡量两个方向是否一致。
            >
            > 极端值测试：把 `缩放系数 alpha` 调成负数，观察 `alpha u` 是否反向；再把矩阵里的 `a,d` 调大，观察网格是否整体拉伸。思考：为什么神经网络权重过大时，后续激活和梯度可能变得不稳定？
            """
        )
    )


def render_vector_reading(dot: float, cosine: float, alpha: float) -> None:
    relation = "方向大体一致" if cosine > 0.25 else "方向接近垂直" if abs(cosine) <= 0.25 else "方向相反"
    st.markdown(
        dedent(
            f"""
            **读图重点：**`u` 和 `v` 是两条原始向量，`u + v` 是把两条箭头首尾相接后的合成结果，`alpha u` 是把 `u` 按当前缩放系数 **{alpha:.1f}** 拉长、缩短或反向。当前点积是 **{dot:.2f}**，cos(theta) 是 **{cosine:.3f}**，说明两条向量 **{relation}**。

            **为什么必须这样看？** 深度学习里的一个神经元常做的事，就是把输入向量和权重向量做点积：方向越一致，输出越大；方向相反，输出越小；接近垂直，说明这个权重几乎没有捕捉到输入的当前模式。`u · v` 同时受长度和方向影响，`cos(theta)` 更强调方向本身，所以本页同时给出两个指标。

            **参数实验。** `u_x/u_y/v_x/v_y` 的范围都是 **-4.0 到 4.0**，默认向量被设置成不平行，方便观察点积变化。`缩放系数 alpha` 的范围是 **-2.0 到 3.0**，默认值是 **1.4**：0 会把 `alpha u` 压成零向量，负数会反向，2 以上会明显拉长。工程上，向量尺度过大常会导致 logits 或注意力分数过尖，所以模型里经常要做归一化、初始化控制或缩放。

            **常见误区：**点积不是单纯比较长度，而是同时看长度和方向。两个很长的向量如果几乎垂直，点积仍然可能接近 0；这也是注意力机制和相似度检索常用点积或余弦相似度的原因。

            > 进阶思考：请把 `u` 和 `v` 调到几乎垂直，再把它们同时拉长。为什么点积可能变大，但 cos(theta) 仍接近 0？这对“相似度”和“激活强度”的区别有什么启发？
            """
        )
    )


def render_matrix_reading(det: float, transformed: np.ndarray) -> None:
    orientation = "发生了方向翻转" if det < 0 else "保持方向" if det > 0 else "被压扁到低维"
    st.markdown(
        dedent(
            fr"""
            **读图重点：**浅灰网格是原坐标纸，青色网格是矩阵 \(A\) 作用后的坐标纸。`A 的第 1 列` 和 `A 的第 2 列` 分别告诉你原来的两个基向量会被送到哪里。当前 det(A) = **{det:.2f}**，表示面积缩放倍数，并且空间 **{orientation}**。输入向量被映射到 **({transformed[0]:.2f}, {transformed[1]:.2f})**。

            **训练循环中的含义：**矩阵 \(A\) 就像一个极简线性层。`输入向量 x/y` 是样本特征，`a,b,c,d` 是权重矩阵的四个参数，`A @ x` 是前向传播输出。你拖动 `a,b,c,d` 时，看到的不是单个点移动，而是整张特征空间被重新组织；这正是神经网络每一层都在做的事情。

            **参数实验。** `a,b,c,d` 的范围都是 **-2.0 到 2.0**，默认矩阵是单位矩阵；`输入向量 x/y` 的范围是 **-2.0 到 2.0**。当 det(A) 接近 0，网格会被压扁，说明不同输入可能被映射到几乎同一个输出，特征信息被丢掉；当 det(A) 为负，方块方向翻转，表示空间方向发生了反射。

            > 互动：把 `a,d` 设大，观察网格拉伸；把 `b,c` 调成非零，观察坐标轴倾斜；让 det(A) 接近 0，观察方块如何被压扁。思考：为什么神经网络权重矩阵如果退化，会丢失特征信息？
            >
            > 对比辨析：向量点积回答“两个方向是否对齐”，矩阵乘法回答“整个空间如何被变形”。请同时观察上半部分的 `u · v` 和下半部分的 `det(A)`，它们分别在训练循环里承担什么角色？
            """
        )
    )


def render_calculus_overview() -> None:
    st.markdown(
        dedent(
            r"""
            **直觉与位置：**导数不是一个神秘符号，它就是“当前位置的坡度”。坡度为正，往右走函数值上升；坡度为负，往右走函数值下降；坡度接近 0，当前位置可能比较平。训练神经网络时，损失函数的梯度就是告诉每个参数：你现在让误差变大还是变小。它位于训练循环的反向传播阶段：**把一个标量损失拆回每个参数的更新方向**。

            本页三块图对应三层难度。`选择函数/观察点 x0/割线步长 h` 解释单变量导数；`x^2 系数/y^2 系数/xy 耦合系数/当前 x/当前 y` 解释多变量梯度；`输入 x/权重 w/偏置 b/目标 y` 解释链式法则如何把输出误差传回参数。

            > 互动：先选择“二次函数”，把 `观察点 x0` 从左拖到右，观察切线斜率由负变正；再把 `割线步长 h` 调小，观察割线如何贴近切线。
            >
            > 进阶思考：为什么训练神经网络时，我们不需要人工推导整张大网络的一条巨型公式，而只需要每个模块知道自己的局部导数？
            """
        )
    )


def render_derivative_reading(fname: str, x0: float, h: float, exact: float, approx: float) -> None:
    error = abs(exact - approx)
    st.markdown(
        dedent(
            f"""
            **读图重点：**当前函数是 **{fname}**，观察点是 **x0 = {x0:.2f}**，割线步长是 **h = {h:.2f}**。精确导数为 **{exact:.4f}**，割线近似为 **{approx:.4f}**，二者差距约 **{error:.4f}**。

            **为什么必须区分切线和割线？** 训练时，优化器需要的是当前位置的局部斜率，而不是远处一大片区域的平均趋势。`割线步长 h` 太大时，你看到的是“从 x0 到 x0+h 的平均变化”；h 变小时，割线才逐渐逼近切线。数值梯度检查也是这个思想：用一个很小的扰动验证自动求导是否合理。

            **参数实验。** `选择函数` 可在二次函数、正弦函数、Sigmoid 之间切换。`观察点 x0` 的范围随函数变化，保证你能看到主要曲线区域；`割线步长 h` 的范围是 **0.05 到 3.00**，默认值是 **0.80**。h 过大误差明显，h 过小在真实浮点计算中又可能遇到数值精度问题，工程里通常选一个足够小但不极端的扰动做检查。

            **常见误区：**导数不是整条曲线的平均趋势，而是某一个点附近的局部变化率。请尤其观察 Sigmoid：在两端它的导数接近 0，这就是深层网络早期常见“梯度变小”的直观来源之一。

            > 极端值测试：选择 Sigmoid，把 `观察点 x0` 拖到 -8 或 8 附近，再看精确导数。为什么模型输出已经接近 0 或 1 时，继续纠正它会变难？
            """
        )
    )


def render_gradient_reading(grad: np.ndarray) -> None:
    norm = float(np.linalg.norm(grad))
    st.markdown(
        dedent(
            f"""
            **读图重点：**三维曲面表示一个二元函数，红色箭头表示当前点的梯度方向。梯度指向函数值上升最快的方向，所以做最小化时要沿着**负梯度方向**走。当前梯度长度约为 **{norm:.3f}**，长度越大，说明这个位置的坡越陡。

            **训练循环中的含义：**`当前 x` 和 `当前 y` 可以看成两个待训练参数，曲面高度就是损失。`x^2 系数`、`y^2 系数` 改变两个方向的弯曲程度，`xy 耦合系数` 改变两个参数之间是否互相牵连。真实神经网络里，参数通常不是独立工作的，一个参数的变化会改变另一个参数的最佳方向，这就是耦合项的意义。

            **参数实验。** `x^2 系数` 和 `y^2 系数` 的范围是 **0.1 到 2.0**，数值越大，该方向曲面越陡；`xy 耦合系数` 的范围是 **-1.0 到 1.0**，绝对值越大，等高线越倾斜；`当前 x/y` 的范围是 **-2.5 到 2.5**，拖动它们可以观察梯度方向如何随位置改变。工程上，如果某些方向特别陡、某些方向特别平，普通梯度下降会走得摇摆，归一化和自适应优化器就是为了解这个痛点。

            > 互动：拖动“当前 x / 当前 y”，观察红色箭头如何改变；再调大 `xy 耦合系数`，看曲面是否发生旋转和倾斜。思考：为什么多个参数之间有耦合时，单独看某一个参数会不够？
            >
            > 进阶思考：如果梯度长度很大，学习率应该更谨慎还是更激进？请先在本图观察坡度，再到“梯度下降几何直觉”里验证你的判断。
            """
        )
    )


def render_chain_reading(chain_x: float, chain_w: float, chain_b: float, target: float, dloss_dw: float) -> None:
    direction = "减小 w" if dloss_dw > 0 else "增大 w" if dloss_dw < 0 else "暂时不改变 w"
    st.markdown(
        dedent(
            fr"""
            **读图重点：**这个小计算图是 `x -> z=wx+b -> a=sigmoid(z) -> L`。反向传播会把三个局部斜率乘起来，得到 \(dL/dw\)。当前 x = **{chain_x:.2f}**，w = **{chain_w:.2f}**，b = **{chain_b:.2f}**，目标 y = **{target:.2f}**，所以 \(dL/dw = {dloss_dw:.4f}\)，梯度下降会倾向于 **{direction}**。

            **为什么这是反向传播的核心？** 前向传播从 `输入 x` 走到 `L`，反向传播从 `L` 走回 `w`。每一段只需要知道自己的局部斜率：损失对激活的斜率 `dL/da`，Sigmoid 对 z 的斜率 `da/dz`，线性层对 w 的斜率 `dz/dw`。页面里的动画把这三个条形依次放出来，最后一个条形就是它们相乘后的总梯度。

            **参数实验。** `输入 x` 和 `权重 w` 的范围都是 **-3.0 到 3.0**，`偏置 b` 的范围是 **-2.0 到 2.0**，`目标 y` 的范围是 **0.0 到 1.0**。把 `w` 或 `b` 调得很大时，Sigmoid 可能进入饱和区，`da/dz` 会变小，最终 `dL/dw` 也可能变小；这就是“局部斜率相乘”带来的工程后果。

            > 互动：点击图上的“播放”，观察 `dL/da`、`da/dz`、`dz/dw` 如何一步步合成 `dL/dw`。思考：为什么深层网络只要每个局部模块可导，就能把误差传回很早的参数？
            >
            > 对比辨析：梯度的正负不是“模型好坏”，而是“参数该往哪边改”。请把 `目标 y` 从 0.2 调到 0.8，观察 `dL/dw` 的符号是否可能改变。
            """
        )
    )


def render_probability_overview() -> None:
    st.markdown(
        dedent(
            """
            **直觉与位置：**概率论给模型一套描述“不确定”的语言。分类模型输出的不是绝对真理，而是每个类别的概率；生成模型不是背答案，而是从分布中采样；评估模型时，我们也要区分先验、证据和后验。在训练循环里，概率常出现在三处：模型输出概率、损失函数里的似然或交叉熵、采样和评估中的不确定性。

            本页三块图各有分工：贝叶斯柱状图解释“看到证据后信念怎样更新”；分布图解释“模型或数据可能遵循什么随机规律”；采样直方图解释“有限样本为什么会抖，样本变多为什么更稳定”。这些都直接对应分类、生成、异常检测和 mini-batch 训练。

            > 互动：先看贝叶斯定理，把“先验 P(H)”调低；再观察即使检测很灵敏，阳性结果为什么仍可能包含很多假阳性。这个现象对医学检测、风控报警、异常检测都很重要。
            >
            > 进阶思考：如果一个异常检测模型在罕见事件上假阳性率很小，但先验更小，报警结果仍可能不可靠。请用本页的三个贝叶斯滑块验证这句话。
            """
        )
    )


def render_bayes_reading(prior: float, sensitivity: float, false_positive: float, posterior: float) -> None:
    st.markdown(
        dedent(
            f"""
            **读图重点：**柱状图把概率翻译成 10000 人中的人数。当前先验 P(H) = **{prior:.2f}**，敏感度 P(E|H) = **{sensitivity:.2f}**，假阳性率 P(E|not H) = **{false_positive:.3f}**，得到后验 P(H|E) = **{posterior:.3f}**。

            **为什么柱状图比公式更重要？** 先验低时，没病人群或正常样本的基数很大，即使假阳性率只有一点点，也会制造大量假阳性。柱状图里的“真阳性”和“假阳性”就是后验概率的分子竞争：阳性结果里到底有多少是真，多少是假。

            **参数实验。** `先验 P(H)` 的范围是 **0.01 到 0.99**，默认值是 **0.12**；`敏感度 P(E|H)` 的范围是 **0.01 到 0.99**，默认值是 **0.90**；`假阳性 P(E|not H)` 的范围是 **0.001 到 0.50**，默认值是 **0.08**。在工业异常检测里，先验往往很低，所以假阳性率经常比模型宣传的“准确率”更关键。

            **常见误区：**P(E|H) 和 P(H|E) 不是一回事。前者是“真的有 H 时看到证据 E 的概率”，后者是“看到证据 E 后真的有 H 的概率”。贝叶斯定理就是把证据倒过来用时必须经过的换算。

            > 极端值测试：把 `先验 P(H)` 调到 0.01，把 `敏感度` 保持 0.90，再逐步增大 `假阳性`。观察后验为什么会迅速下降。思考：这对风控报警阈值有什么启发？
            """
        )
    )


def render_distribution_reading(dist: str) -> None:
    st.markdown(
        dedent(
            f"""
            **读图重点：**当前分布是 **{dist}**。正态分布适合表示许多小扰动叠加后的连续噪声；伯努利分布表示一次是/否事件；二项分布表示多次独立是/否事件的成功次数；泊松分布常用于固定时间或空间窗口里的稀疏计数。

            **训练循环中的含义：**分类模型的一个标签可以看成伯努利或类别分布，mini-batch 里的正确个数可以用二项分布直觉理解，输入噪声和初始化误差常近似正态，单位时间内的点击、故障、请求数常接近泊松建模。你切换 `选择分布` 时，其实是在切换“随机变量是什么”和“横轴代表什么”。

            **参数实验。** 正态分布的 `均值 mu` 范围是 **-3.0 到 3.0**，控制曲线中心；`标准差 sigma` 范围是 **0.2 到 3.0**，控制不确定性宽度。伯努利和二项分布的 `成功概率 p` 范围是 **0.01 到 0.99**；二项分布的 `试验次数 n` 范围是 **1 到 80**；泊松分布的 `事件率 lambda` 范围是 **0.2 到 15.0**。工程上，先问“我的输出是什么随机变量”，再选择分布，别反过来硬套曲线。

            > 互动：切换不同分布，观察横轴和纵轴含义如何变化。连续分布看密度曲线，离散分布看每个整数点的概率质量；不要把 PDF 的高度直接误读成某一点的概率。
            >
            > 对比辨析：伯努利看“一次成不成功”，二项看“多次成功了几次”，泊松看“一个窗口里来了几个事件”。请分别调 `p`、`n`、`lambda`，观察图形变宽或右移的方式是否相同。
            """
        )
    )


def render_sampling_reading(sample_n: int) -> None:
    st.markdown(
        dedent(
            f"""
            **读图重点：**直方图是从标准正态分布中抽出的经验样本，红线是真实 PDF。当前采样数量是 **{sample_n}**。样本少时，直方图会抖；样本多时，它会逐渐贴近真实分布。

            **训练循环中的含义：**深度学习中的 mini-batch 训练也是在用有限样本估计总体梯度。batch 太小会噪声大，batch 变大估计更稳，但计算和显存成本也会上升。本图的红线像“总体真实规律”，直方图像“你这一次 batch 看到的估计”。

            **参数实验。** `采样数量` 的范围是 **20 到 5000**，默认值是 **500**。20 到 100 时直方图波动很明显，适合观察小 batch 的随机性；500 到 1000 通常能看出总体形状；3000 以上更平滑，但计算量更大。工业训练里，batch size 不是越大越好，常常要在吞吐、显存、泛化和梯度噪声之间折中。

            > 极端值测试：把 `采样数量` 设为 20，再设为 5000。观察直方图和红线的贴合程度。思考：为什么小 batch 有噪声，但有时反而能帮助模型跳出很尖的局部区域？
            """
        )
    )


def render_gradient_descent_overview() -> None:
    st.markdown(
        dedent(
            """
            **直觉与位置：**梯度下降像在雾里下山：看不见整座山，只能看脚下哪里最陡，然后朝相反方向迈一步。学习率决定步子大小，动量决定是否沿着过去的方向保留惯性。它位于训练循环的最后一步：**已经知道梯度以后，真正改变参数**。

            本页的 `损失曲面` 是训练难度的缩影：碗形凸函数像最理想的课堂例子，峡谷函数像尺度不均的真实损失地形，非凸波浪函数像深度网络常见的多低谷地形。`起点 x/y` 是初始参数，`学习率` 是每次更新幅度，`动量` 是历史方向的惯性，`迭代步数` 是训练轮数的简化版。

            > 互动：先选择“碗形凸函数”，把学习率调到 0.05 和 0.35 对比；再切到“峡谷函数”，打开动量，观察路径是否更容易穿过狭长地形。
            >
            > 进阶思考：为什么同一个学习率在碗形凸函数上很舒服，在峡谷函数或非凸波浪函数上却可能震荡？请结合等高线密度回答。
            """
        )
    )


def render_gradient_descent_reading(kind: str, start_x: float, start_y: float, lr: float, momentum: float, steps: int, final_loss: float) -> None:
    st.markdown(
        dedent(
            f"""
            **读图重点：**等高线越密，坡度越陡；红线是参数更新轨迹；黄色小箭头是负梯度方向。当前损失曲面是 **{kind}**，起点是 **({start_x:.2f}, {start_y:.2f})**，学习率是 **{lr:.2f}**，动量是 **{momentum:.2f}**，迭代 **{steps}** 步后 loss 为 **{final_loss:.4f}**。

            **为什么这些参数会互相影响？** 学习率决定红线每一步走多远；动量会把过去几步的方向累积起来，让路径在峡谷里不那么左右横跳，但动量过大也可能冲过低谷。迭代步数决定你给优化器多少机会，起点决定它从哪片地形开始看局部斜率。

            **参数实验。** `损失曲面` 可选 **碗形凸函数、峡谷函数、非凸波浪函数**。`起点 x/y` 的范围都是 **-4.0 到 4.0**；`学习率` 的范围是 **0.01 到 0.45**，默认值是 **0.12**；`动量` 的范围是 **0.0 到 0.9**，默认值是 **0.0**；`迭代步数` 的范围是 **5 到 80**，默认值是 **32**。工程上，学习率通常先保守设置，再用验证集和 loss 曲线判断是否升高；动量常在峡谷形地形中有帮助，但不是所有问题都越大越好。

            **常见误区：**梯度下降不是每一步都直奔全局最优，它只看局部斜率。非凸曲面可能有局部低谷，学习率过大可能震荡，学习率过小又会慢得像没动。

            > 极端值测试：在“峡谷函数”里把 `学习率` 调到 0.45，再把 `动量` 调到 0.9，观察红线是否可能冲出稳定路线。再把学习率降到 0.03，观察是否过慢。思考：为什么真实训练常需要学习率调度？
            """
        )
    )


def render_cheatsheet_guide() -> None:
    st.markdown(
        dedent(
            r"""
            **使用方法：**把这一页当成训练循环的最小词典。前向传播用线性代数组织数据，激活函数加入非线性，损失函数把错误变成标量，反向传播用链式法则算梯度，优化器按梯度更新参数。下方公式不是用来孤立背诵的，而是把前面四个互动页面压缩成一条可执行流水线。

            读速查表时请对应回图形：`z = Wx + b` 对应“矩阵把整张坐标纸一起变形”；`a = sigma(z)` 对应 Sigmoid 曲线和概率输出；`L = ell(a, y)` 对应损失曲面高度；`dL/dW` 对应链式法则动画里的局部斜率相乘；`W <- W - eta grad` 对应梯度下降红色轨迹。

            > 互动：回到前四个知识点，各找一个公式和这里对应起来：`z = Wx + b` 对应矩阵变换，`dL/dw` 对应链式法则，`W <- W - eta grad` 对应梯度下降轨迹，概率分布对应模型输出和采样。
            >
            > 复盘问题：如果训练 loss 不下降，你会先检查哪一块？输入矩阵是否退化、Sigmoid 是否饱和、梯度是否过小、学习率是否过大，还是 batch 估计是否太噪？请把答案映射到前面对应的图。
            """
        )
    )


def render_matplotlib(fig: plt.Figure) -> None:
    try:
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def plotly_layout(
    fig: go.Figure,
    title: str | None = None,
    height: int = 520,
    equal_axes: bool = False,
) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(color=INK),
        margin=dict(l=20, r=20, t=52 if title else 20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        zeroline=True,
        zerolinecolor="#9aa7ad",
        gridcolor="#e7ecef",
        linecolor=LINE,
    )
    fig.update_yaxes(
        zeroline=True,
        zerolinecolor="#9aa7ad",
        gridcolor="#e7ecef",
        linecolor=LINE,
        scaleanchor="x" if equal_axes else None,
        scaleratio=1 if equal_axes else None,
    )
    return fig


def add_vector(
    fig: go.Figure,
    vec: np.ndarray,
    name: str,
    color: str,
    origin: tuple[float, float] = (0.0, 0.0),
    width: int = 4,
) -> None:
    x0, y0 = origin
    x1, y1 = x0 + float(vec[0]), y0 + float(vec[1])
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=width),
            marker=dict(size=[5, 9], color=color),
            hovertemplate=f"{name}: ({vec[0]:.2f}, {vec[1]:.2f})<extra></extra>",
        )
    )
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=width,
        arrowcolor=color,
        text="",
    )


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


@dataclass(frozen=True)
class ScalarFunction:
    label: str
    f: Callable[[np.ndarray], np.ndarray]
    df: Callable[[np.ndarray], np.ndarray]
    latex: str
    x_min: float
    x_max: float


FUNCTIONS: dict[str, ScalarFunction] = {
    "二次函数": ScalarFunction(
        "二次函数",
        lambda x: 0.35 * (x - 1.0) ** 2 + 0.25,
        lambda x: 0.7 * (x - 1.0),
        "f(x)=0.35(x-1)^2+0.25",
        -4.0,
        6.0,
    ),
    "正弦函数": ScalarFunction(
        "正弦函数",
        lambda x: np.sin(x) + 0.2 * x,
        lambda x: np.cos(x) + 0.2,
        "f(x)=sin(x)+0.2x",
        -2.0 * math.pi,
        2.0 * math.pi,
    ),
    "Sigmoid": ScalarFunction(
        "Sigmoid",
        lambda x: sigmoid(x),
        lambda x: sigmoid(x) * (1.0 - sigmoid(x)),
        "f(x)=1/(1+e^{-x})",
        -8.0,
        8.0,
    ),
}


def matrix_from_controls(prefix: str) -> np.ndarray:
    col1, col2 = st.columns(2)
    with col1:
        a = st.slider("a: x 基向量的 x 分量", -2.0, 2.0, 1.0, 0.1, key=f"{prefix}_a")
        c = st.slider("c: x 基向量的 y 分量", -2.0, 2.0, 0.0, 0.1, key=f"{prefix}_c")
    with col2:
        b = st.slider("b: y 基向量的 x 分量", -2.0, 2.0, 0.0, 0.1, key=f"{prefix}_b")
        d = st.slider("d: y 基向量的 y 分量", -2.0, 2.0, 1.0, 0.1, key=f"{prefix}_d")
    return np.array([[a, b], [c, d]], dtype=float)


def line_trace(points: np.ndarray, name: str, color: str, width: int = 2) -> go.Scatter:
    return go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        hoverinfo="skip",
    )


def render_linear_algebra() -> None:
    st.header("1. 线性代数基础")
    note("向量表示方向和长度，矩阵表示空间变换。深度学习里的线性层 y = Wx + b，本质就是先用矩阵重排、拉伸、旋转特征空间，再平移。")
    render_linear_algebra_overview()

    concept_cards(
        [
            ("向量", "一个点、一个方向，也可以是一条样本的特征列表。"),
            ("矩阵", "一组列向量；每一列说明原来的基向量会被送到哪里。"),
            ("矩阵乘法", "B 先变换向量，A 再变换结果，所以 ABx = A(Bx)。"),
        ]
    )

    st.subheader("向量运算：加法、缩放、点积")
    controls, chart = st.columns([0.34, 0.66])
    with controls:
        ux = st.slider("u_x", -4.0, 4.0, 2.0, 0.1)
        uy = st.slider("u_y", -4.0, 4.0, 1.0, 0.1)
        vx = st.slider("v_x", -4.0, 4.0, -1.0, 0.1)
        vy = st.slider("v_y", -4.0, 4.0, 2.5, 0.1)
        alpha = st.slider("缩放系数 alpha", -2.0, 3.0, 1.4, 0.1)
        u = np.array([ux, uy])
        v = np.array([vx, vy])
        dot = float(u @ v)
        norm_product = float(np.linalg.norm(u) * np.linalg.norm(v))
        cosine = dot / norm_product if norm_product > 1e-9 else 0.0
        st.metric("u · v", f"{dot:.2f}")
        st.metric("cos(theta)", f"{cosine:.3f}")
    with chart:
        fig = go.Figure()
        add_vector(fig, u, "u", TEAL)
        add_vector(fig, v, "v", ROSE)
        add_vector(fig, u + v, "u + v", AMBER)
        add_vector(fig, alpha * u, "alpha u", BLUE)
        add_vector(fig, v, "平移后的 v", ROSE, origin=(float(u[0]), float(u[1])), width=2)
        lim = max(5.0, float(np.max(np.abs([u, v, u + v, alpha * u]))) + 1.0)
        fig.update_xaxes(range=[-lim, lim])
        fig.update_yaxes(range=[-lim, lim])
        st.plotly_chart(plotly_layout(fig, "向量加法与缩放", equal_axes=True), width="stretch")

    formula(
        "点积 u · v = ||u|| ||v|| cos(theta)。在神经网络中，它常被用来衡量输入特征和权重方向的对齐程度。"
    )
    render_vector_reading(dot, cosine, alpha)

    st.subheader("矩阵乘法的几何意义")
    left, right = st.columns([0.34, 0.66])
    with left:
        mat = matrix_from_controls("matrix")
        x = st.slider("输入向量 x", -2.0, 2.0, 1.0, 0.1)
        y = st.slider("输入向量 y", -2.0, 2.0, 1.0, 0.1)
        vector = np.array([x, y])
        transformed = mat @ vector
        det = float(np.linalg.det(mat))
        st.metric("det(A)", f"{det:.2f}", help="面积缩放倍数；负数表示翻转方向。")
        st.metric("A @ x", f"({transformed[0]:.2f}, {transformed[1]:.2f})")
    with right:
        grid = np.linspace(-3.0, 3.0, 13)
        fig = go.Figure()
        for g in grid:
            vertical = np.column_stack([np.full_like(grid, g), grid])
            horizontal = np.column_stack([grid, np.full_like(grid, g)])
            fig.add_trace(line_trace(vertical, "原网格" if g == grid[0] else "", "#d3dadd", 1))
            fig.add_trace(line_trace(horizontal, "", "#d3dadd", 1))
            fig.add_trace(line_trace((mat @ vertical.T).T, "变换后网格" if g == grid[0] else "", "#97b7b8", 1))
            fig.add_trace(line_trace((mat @ horizontal.T).T, "", "#97b7b8", 1))

        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        fig.add_trace(line_trace(square, "单位方块", INK, 3))
        fig.add_trace(line_trace((mat @ square.T).T, "A 变换后的方块", AMBER, 4))
        add_vector(fig, np.array([1.0, 0.0]), "原 e1", "#9aa7ad", width=2)
        add_vector(fig, np.array([0.0, 1.0]), "原 e2", "#9aa7ad", width=2)
        add_vector(fig, mat[:, 0], "A 的第 1 列", TEAL)
        add_vector(fig, mat[:, 1], "A 的第 2 列", ROSE)
        add_vector(fig, vector, "输入 x", VIOLET)
        add_vector(fig, transformed, "A @ x", BLUE)
        fig.update_xaxes(range=[-5, 5])
        fig.update_yaxes(range=[-5, 5])
        st.plotly_chart(plotly_layout(fig, "矩阵把整张坐标纸一起变形", equal_axes=True), width="stretch")

    formula(
        "若 A = [[a, b], [c, d]]，则 A[x, y]^T = x[a, c]^T + y[b, d]^T。也就是说，输出是 A 的两列按输入坐标加权相加。"
    )
    render_matrix_reading(det, transformed)


def render_calculus() -> None:
    st.header("2. 微积分基础")
    note("导数告诉我们沿某个方向走一点点，函数值会变快还是变慢。反向传播不是魔法，它就是把许多局部导数用链式法则乘起来。")
    render_calculus_overview()

    st.subheader("导数的几何含义：切线斜率")
    controls, chart = st.columns([0.32, 0.68])
    with controls:
        fname = st.selectbox("选择函数", list(FUNCTIONS), index=0)
        fn = FUNCTIONS[fname]
        x0 = st.slider("观察点 x0", fn.x_min, fn.x_max, 1.0, 0.05)
        h = st.slider("割线步长 h", 0.05, 3.0, 0.8, 0.05)
        exact = float(fn.df(np.array([x0]))[0])
        approx = float((fn.f(np.array([x0 + h]))[0] - fn.f(np.array([x0]))[0]) / h)
        st.metric("精确导数 f'(x0)", f"{exact:.4f}")
        st.metric("割线近似", f"{approx:.4f}")
    with chart:
        xs = np.linspace(fn.x_min, fn.x_max, 500)
        ys = fn.f(xs)
        y0 = float(fn.f(np.array([x0]))[0])
        y1 = float(fn.f(np.array([x0 + h]))[0])
        tangent = y0 + exact * (xs - x0)
        secant = y0 + approx * (xs - x0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=fn.latex, line=dict(color=TEAL, width=4)))
        fig.add_trace(go.Scatter(x=xs, y=tangent, mode="lines", name="切线", line=dict(color=ROSE, width=3)))
        fig.add_trace(go.Scatter(x=xs, y=secant, mode="lines", name="割线", line=dict(color=AMBER, width=2, dash="dash")))
        fig.add_trace(
            go.Scatter(
                x=[x0, x0 + h],
                y=[y0, y1],
                mode="markers+lines",
                name="两点差商",
                marker=dict(size=9, color=BLUE),
                line=dict(color=BLUE, width=2),
            )
        )
        st.plotly_chart(plotly_layout(fig, "h 越小，割线越接近切线"), width="stretch")
    render_derivative_reading(fname, x0, h, exact, approx)

    st.subheader("偏导数与梯度：多变量函数的方向感")
    left, right = st.columns([0.32, 0.68])
    with left:
        ax2 = st.slider("x^2 系数", 0.1, 2.0, 0.8, 0.1)
        by2 = st.slider("y^2 系数", 0.1, 2.0, 0.5, 0.1)
        cxy = st.slider("xy 耦合系数", -1.0, 1.0, 0.25, 0.05)
        px = st.slider("当前 x", -2.5, 2.5, 1.0, 0.1)
        py = st.slider("当前 y", -2.5, 2.5, -1.0, 0.1)
        grad = np.array([2 * ax2 * px + cxy * py, 2 * by2 * py + cxy * px])
        st.metric("partial f / partial x", f"{grad[0]:.3f}")
        st.metric("partial f / partial y", f"{grad[1]:.3f}")
    with right:
        xg = np.linspace(-3, 3, 80)
        yg = np.linspace(-3, 3, 80)
        xx, yy = np.meshgrid(xg, yg)
        zz = ax2 * xx**2 + by2 * yy**2 + cxy * xx * yy
        z0 = ax2 * px**2 + by2 * py**2 + cxy * px * py
        fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                colorscale="Viridis",
                opacity=0.84,
                showscale=False,
                name="损失曲面",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[px, px + 0.55 * grad[0]],
                y=[py, py + 0.55 * grad[1]],
                z=[z0, z0 + 0.55 * float(grad @ grad)],
                mode="lines+markers",
                name="梯度方向",
                line=dict(color=ROSE, width=7),
                marker=dict(size=5, color=ROSE),
            )
        )
        fig.update_layout(
            title="梯度指向函数上升最快的方向，负梯度用于下降",
            height=540,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK),
            margin=dict(l=0, r=0, t=45, b=0),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="f(x,y)",
                xaxis=dict(backgroundcolor="white", gridcolor="#e7ecef"),
                yaxis=dict(backgroundcolor="white", gridcolor="#e7ecef"),
                zaxis=dict(backgroundcolor="white", gridcolor="#e7ecef"),
            ),
        )
        st.plotly_chart(fig, width="stretch")
    render_gradient_reading(grad)

    st.subheader("链式法则动画：局部斜率如何相乘")
    controls, chart = st.columns([0.32, 0.68])
    with controls:
        chain_x = st.slider("输入 x", -3.0, 3.0, 1.2, 0.1, key="chain_x")
        chain_w = st.slider("权重 w", -3.0, 3.0, 1.4, 0.1, key="chain_w")
        chain_b = st.slider("偏置 b", -2.0, 2.0, -0.3, 0.1, key="chain_b")
        target = st.slider("目标 y", 0.0, 1.0, 0.8, 0.05, key="chain_target")
        z = chain_w * chain_x + chain_b
        a = float(sigmoid(z))
        loss = (a - target) ** 2
        dloss_da = 2 * (a - target)
        da_dz = a * (1 - a)
        dz_dw = chain_x
        dloss_dw = dloss_da * da_dz * dz_dw
        st.metric("L", f"{loss:.4f}")
        st.metric("dL/dw", f"{dloss_dw:.4f}")
    with chart:
        labels = ["dL/da", "da/dz", "dz/dw", "dL/dw"]
        values = [dloss_da, da_dz, dz_dw, dloss_dw]
        frames = []
        for idx in range(1, len(labels) + 1):
            frames.append(
                go.Frame(
                    data=[
                        go.Bar(
                            x=labels[:idx],
                            y=values[:idx],
                            marker_color=[ROSE, TEAL, AMBER, BLUE][:idx],
                        )
                    ],
                    name=str(idx),
                )
            )
        fig = go.Figure(
            data=[go.Bar(x=[labels[0]], y=[values[0]], marker_color=[ROSE])],
            frames=frames,
        )
        fig.update_layout(
            title="dL/dw = dL/da * da/dz * dz/dw",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(color=INK),
            margin=dict(l=20, r=20, t=52, b=20),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0,
                    y=1.16,
                    buttons=[
                        dict(
                            label="播放",
                            method="animate",
                            args=[None, {"frame": {"duration": 650, "redraw": True}, "fromcurrent": True}],
                        )
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(method="animate", args=[[str(i)], {"frame": {"duration": 0, "redraw": True}}], label=str(i))
                        for i in range(1, len(labels) + 1)
                    ],
                    x=0.1,
                    y=-0.05,
                )
            ],
        )
        fig.update_yaxes(gridcolor="#e7ecef", zerolinecolor="#9aa7ad")
        st.plotly_chart(fig, width="stretch")
        formula(
            f"z = wx+b = {z:.3f}，a = sigmoid(z) = {a:.3f}，L = (a-y)^2 = {loss:.4f}。"
        )
    render_chain_reading(chain_x, chain_w, chain_b, target, dloss_dw)


def normal_pdf(xs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    return np.array([math.exp(-lam) * lam**int(i) / math.factorial(int(i)) for i in k])


def render_probability() -> None:
    st.header("3. 概率论基础")
    note("概率不是只用来算硬币。分类模型的 softmax、生成模型的采样、贝叶斯更新和不确定性评估，都需要同一套语言。")
    render_probability_overview()

    st.subheader("贝叶斯定理：证据如何更新信念")
    left, right = st.columns([0.34, 0.66])
    with left:
        prior = st.slider("先验 P(H)：样本真实为正的比例", 0.01, 0.99, 0.12, 0.01)
        sensitivity = st.slider("敏感度 P(E|H)：有病时检测阳性", 0.01, 0.99, 0.90, 0.01)
        false_positive = st.slider("假阳性 P(E|not H)：没病时检测阳性", 0.001, 0.50, 0.08, 0.001)
        evidence = sensitivity * prior + false_positive * (1 - prior)
        posterior = sensitivity * prior / evidence
        st.metric("证据概率 P(E)", f"{evidence:.3f}")
        st.metric("后验 P(H|E)", f"{posterior:.3f}")
    with right:
        population = 10000
        true_pos = prior * population * sensitivity
        false_pos = (1 - prior) * population * false_positive
        true_neg = (1 - prior) * population * (1 - false_positive)
        false_neg = prior * population * (1 - sensitivity)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["真阳性", "假阳性", "真阴性", "假阴性"],
                y=[true_pos, false_pos, true_neg, false_neg],
                marker_color=[TEAL, ROSE, GREEN, AMBER],
                text=[f"{v:.0f}" for v in [true_pos, false_pos, true_neg, false_neg]],
                textposition="auto",
            )
        )
        st.plotly_chart(plotly_layout(fig, "把概率想象成 10000 人中的计数", height=430), width="stretch")
    formula("P(H|E) = P(E|H)P(H) / P(E)。当先验很低时，即使检测很灵敏，阳性结果里也可能混入大量假阳性。")
    render_bayes_reading(prior, sensitivity, false_positive, posterior)

    st.subheader("常见概率分布可视化")
    controls, chart = st.columns([0.32, 0.68])
    with controls:
        dist = st.selectbox("选择分布", ["Normal 正态分布", "Bernoulli 伯努利分布", "Binomial 二项分布", "Poisson 泊松分布"])
        if dist.startswith("Normal"):
            mu = st.slider("均值 mu", -3.0, 3.0, 0.0, 0.1)
            sigma = st.slider("标准差 sigma", 0.2, 3.0, 1.0, 0.1)
        elif dist.startswith("Bernoulli"):
            p = st.slider("成功概率 p", 0.01, 0.99, 0.5, 0.01, key="bern_p")
        elif dist.startswith("Binomial"):
            n = st.slider("试验次数 n", 1, 80, 20, 1)
            p = st.slider("成功概率 p", 0.01, 0.99, 0.35, 0.01, key="binom_p")
        else:
            lam = st.slider("事件率 lambda", 0.2, 15.0, 4.0, 0.1)
    with chart:
        fig = go.Figure()
        if dist.startswith("Normal"):
            xs = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 500)
            ys = normal_pdf(xs, mu, sigma)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="PDF", line=dict(color=TEAL, width=4)))
            fig.add_vline(x=mu, line_color=ROSE, line_width=2, annotation_text="均值")
            title = "正态分布：很多小扰动相加后的形状"
        elif dist.startswith("Bernoulli"):
            xs = np.array([0, 1])
            ys = np.array([1 - p, p])
            fig.add_trace(go.Bar(x=["0 失败", "1 成功"], y=ys, marker_color=[ROSE, TEAL], name="PMF"))
            title = "伯努利分布：一次是/否事件"
        elif dist.startswith("Binomial"):
            xs = np.arange(n + 1)
            coeffs = np.array([math.comb(n, int(k)) for k in xs], dtype=float)
            ys = coeffs * (p**xs) * ((1 - p) ** (n - xs))
            fig.add_trace(go.Bar(x=xs, y=ys, marker_color=BLUE, name="PMF"))
            title = "二项分布：n 次独立伯努利事件的成功次数"
        else:
            xs = np.arange(0, max(25, int(lam * 4) + 1))
            ys = poisson_pmf(xs, lam)
            fig.add_trace(go.Bar(x=xs, y=ys, marker_color=AMBER, name="PMF"))
            title = "泊松分布：固定窗口中的稀疏事件数"
        st.plotly_chart(plotly_layout(fig, title, height=470), width="stretch")
    render_distribution_reading(dist)

    st.subheader("采样直觉：样本越多，经验分布越稳定")
    sample_n = st.slider("采样数量", 20, 5000, 500, 20)
    rng = np.random.default_rng(7)
    sample = rng.normal(0, 1, sample_n)
    fig_mpl, ax = plt.subplots(figsize=(8, 3.6))
    fig_mpl.patch.set_facecolor(PAPER)
    ax.set_facecolor("white")
    ax.hist(sample, bins=32, density=True, color="#8fbfc0", edgecolor="white", alpha=0.88)
    xs = np.linspace(-4, 4, 300)
    ax.plot(xs, normal_pdf(xs, 0, 1), color=ROSE, linewidth=2.5, label="标准正态 PDF")
    ax.set_title("Matplotlib 采样直方图")
    ax.set_xlabel("x")
    ax.set_ylabel("密度")
    ax.grid(True, alpha=0.24)
    ax.legend()
    render_matplotlib(fig_mpl)
    render_sampling_reading(sample_n)


def loss_and_grad(kind: str, x: float, y: float) -> tuple[float, np.ndarray]:
    if kind == "碗形凸函数":
        value = 0.55 * (x - 1.5) ** 2 + 0.9 * (y + 0.8) ** 2
        grad = np.array([1.1 * (x - 1.5), 1.8 * (y + 0.8)])
    elif kind == "峡谷函数":
        value = 0.12 * (x + y) ** 2 + 1.8 * (x - y) ** 2
        grad = np.array([0.24 * (x + y) + 3.6 * (x - y), 0.24 * (x + y) - 3.6 * (x - y)])
    else:
        value = 0.15 * (x**2 + y**2) + math.sin(1.4 * x) * math.cos(1.1 * y) + 1.2
        grad = np.array(
            [
                0.3 * x + 1.4 * math.cos(1.4 * x) * math.cos(1.1 * y),
                0.3 * y - 1.1 * math.sin(1.4 * x) * math.sin(1.1 * y),
            ]
        )
    return value, grad


def gradient_descent_path(
    kind: str,
    start: tuple[float, float],
    learning_rate: float,
    momentum: float,
    steps: int,
) -> np.ndarray:
    point = np.array(start, dtype=float)
    velocity = np.zeros(2, dtype=float)
    rows = []
    for step in range(steps + 1):
        value, grad = loss_and_grad(kind, float(point[0]), float(point[1]))
        rows.append([step, point[0], point[1], value, grad[0], grad[1]])
        velocity = momentum * velocity + grad
        point = point - learning_rate * velocity
        if np.linalg.norm(point) > 12:
            point = np.clip(point, -12, 12)
    return np.array(rows)


def render_gradient_descent() -> None:
    st.header("4. 梯度下降的几何直觉")
    note("梯度下降每一步都问同一个问题：当前位置哪里上升最快？然后朝相反方向走。学习率决定步子大小，动量决定是否带着过去方向的惯性。")
    render_gradient_descent_overview()

    controls, chart = st.columns([0.31, 0.69])
    with controls:
        kind = st.selectbox("损失曲面", ["碗形凸函数", "峡谷函数", "非凸波浪函数"])
        start_x = st.slider("起点 x", -4.0, 4.0, -3.0, 0.1)
        start_y = st.slider("起点 y", -4.0, 4.0, 3.0, 0.1)
        lr = st.slider("学习率", 0.01, 0.45, 0.12, 0.01)
        momentum = st.slider("动量", 0.0, 0.9, 0.0, 0.05)
        steps = st.slider("迭代步数", 5, 80, 32, 1)
        path = gradient_descent_path(kind, (start_x, start_y), lr, momentum, steps)
        final = path[-1]
        st.metric("最终 loss", f"{final[3]:.4f}")
        st.metric("最终位置", f"({final[1]:.2f}, {final[2]:.2f})")
    with chart:
        grid = np.linspace(-5, 5, 120)
        xx, yy = np.meshgrid(grid, grid)
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j], _ = loss_and_grad(kind, float(xx[i, j]), float(yy[i, j]))

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=grid,
                y=grid,
                z=zz,
                colorscale="Viridis",
                contours=dict(showlabels=False),
                showscale=False,
                opacity=0.9,
                name="等高线",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path[:, 1],
                y=path[:, 2],
                mode="lines+markers",
                name="下降轨迹",
                line=dict(color=ROSE, width=4),
                marker=dict(size=7, color=path[:, 0], colorscale="Reds", showscale=False),
            )
        )
        step_stride = max(1, len(path) // 12)
        for row in path[::step_stride]:
            grad = np.array([row[4], row[5]])
            norm = np.linalg.norm(grad)
            if norm > 1e-9:
                direction = -grad / norm * 0.45
                add_vector(fig, direction, "负梯度方向" if row[0] == 0 else "", AMBER, origin=(row[1], row[2]), width=2)
        fig.update_xaxes(range=[-5, 5])
        fig.update_yaxes(range=[-5, 5])
        st.plotly_chart(plotly_layout(fig, "2D 等高线上的梯度下降", height=540, equal_axes=True), width="stretch")
    render_gradient_descent_reading(kind, start_x, start_y, lr, momentum, steps, float(final[3]))

    st.subheader("3D 损失曲面动画")
    frame_count = min(steps + 1, 45)
    indices = np.unique(np.linspace(0, len(path) - 1, frame_count).astype(int))
    surface = go.Surface(x=xx, y=yy, z=zz, colorscale="Viridis", opacity=0.72, showscale=False)
    initial = path[indices[:1]]
    frames = []
    for idx in indices:
        segment = path[: idx + 1]
        frames.append(
            go.Frame(
                data=[
                    surface,
                    go.Scatter3d(
                        x=segment[:, 1],
                        y=segment[:, 2],
                        z=segment[:, 3] + 0.08,
                        mode="lines+markers",
                        line=dict(color=ROSE, width=7),
                        marker=dict(size=4, color=AMBER),
                        name="下降轨迹",
                    ),
                ],
                name=str(int(idx)),
            )
        )
    fig3d = go.Figure(
        data=[
            surface,
            go.Scatter3d(
                x=initial[:, 1],
                y=initial[:, 2],
                z=initial[:, 3] + 0.08,
                mode="lines+markers",
                line=dict(color=ROSE, width=7),
                marker=dict(size=5, color=AMBER),
                name="下降轨迹",
            ),
        ],
        frames=frames,
    )
    fig3d.update_layout(
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="loss",
            xaxis=dict(backgroundcolor="white", gridcolor="#e7ecef"),
            yaxis=dict(backgroundcolor="white", gridcolor="#e7ecef"),
            zaxis=dict(backgroundcolor="white", gridcolor="#e7ecef"),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0,
                y=1.05,
                buttons=[
                    dict(
                        label="播放",
                        method="animate",
                        args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[str(int(i))], {"frame": {"duration": 0, "redraw": True}}], label=str(int(i)))
                    for i in indices
                ],
                x=0.1,
                y=-0.02,
            )
        ],
    )
    st.plotly_chart(fig3d, width="stretch")

    formula("学习率太小会慢，太大会震荡甚至发散；峡谷函数展示了为什么特征缩放、归一化和自适应优化器很重要。")


def render_cheatsheet() -> None:
    st.header("5. 一页速查")
    note("这部分把前面四类工具压缩成深度学习里最常用的形式。调参时先找对应的数学对象，再回到图上看几何含义。")
    render_cheatsheet_guide()
    concept_cards(
        [
            ("线性层", "y = Wx + b。W 负责旋转、缩放、混合特征；b 负责平移。"),
            ("激活函数", "给线性变换加非线性，否则多层线性层仍然等价于一层线性层。"),
            ("损失函数", "把模型输出变成一个可优化的标量地形。"),
            ("梯度", "每个参数该往哪里动、动多少的一阶局部信号。"),
            ("链式法则", "把输出误差沿计算图逐层传回每个参数。"),
            ("概率分布", "描述数据、噪声、预测不确定性和采样机制。"),
        ]
    )
    st.latex(r"""
    \begin{aligned}
    \text{Linear: }& z = Wx + b\\
    \text{Activation: }& a = \sigma(z)\\
    \text{Loss: }& L = \ell(a, y)\\
    \text{Backprop: }& \frac{\partial L}{\partial W}
      = \frac{\partial L}{\partial a}
        \frac{\partial a}{\partial z}
        \frac{\partial z}{\partial W}\\
    \text{Update: }& W \leftarrow W - \eta \nabla_W L
    \end{aligned}
    """)


st.markdown(
    """
    <div class="hero">
      <h1>深度学习数学基础速查</h1>
      <p>把线性代数、微积分、概率论和梯度下降放到同一块交互画布上。每个知识点都可以拖动参数，直接观察公式背后的几何变化。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_math_learning_map()

topic = st.sidebar.radio(
    "选择知识点",
    [
        "线性代数基础",
        "微积分基础",
        "概率论基础",
        "梯度下降几何直觉",
        "一页速查",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("建议顺序：先看矩阵如何变形空间，再看导数如何给出局部方向，最后用概率和梯度下降连接到训练过程。")

if topic == "线性代数基础":
    render_linear_algebra()
elif topic == "微积分基础":
    render_calculus()
elif topic == "概率论基础":
    render_probability()
elif topic == "梯度下降几何直觉":
    render_gradient_descent()
else:
    render_cheatsheet()
