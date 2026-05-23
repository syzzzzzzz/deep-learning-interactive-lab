"""
Interactive quiz system for the deep learning learning path.

Run:
    streamlit run part5_toolbox/quiz_system.py
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


QuestionType = Literal["choice", "true_false", "blank"]


st.set_page_config(
    page_title="练习题与测验系统",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .question-card {
        background: rgba(255,255,255,0.82);
        border: 1px solid #d7ded8;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem 0;
    }
    .question-title {
        font-weight: 700;
        line-height: 1.55;
        margin-bottom: 0.62rem;
    }
    .pill {
        display: inline-block;
        border: 1px solid #cbd5d4;
        border-radius: 999px;
        padding: 0.12rem 0.52rem;
        margin-right: 0.35rem;
        color: #455258;
        background: rgba(255,255,255,0.72);
        font-size: 0.84rem;
    }
    .answer-box {
        border-left: 4px solid #c47f1f;
        background: rgba(255,255,255,0.72);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        line-height: 1.65;
        margin-top: 0.6rem;
    }
    .ok { color: #207f7a; font-weight: 700; }
    .bad { color: #b84d5a; font-weight: 700; }
    .small {
        color: #59656a;
        font-size: 0.92rem;
        line-height: 1.58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class Question:
    id: str
    module: str
    topic: str
    qtype: QuestionType
    prompt: str
    answer: str | bool
    options: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    explanation: str = ""
    link: str = ""
    difficulty: str = "基础"


MODULE_LINKS = {
    "机器学习基础": "part1_foundations/machine_learning_basics.py",
    "CNN": "part2_cnn/cnn_architectures.py",
    "RNN": "part3_rnn/sequence_models.py",
    "Transformer": "part4_transformer/transformer_models.py",
    "GAN": "part4_transformer/gan_ae.py",
}


QUESTIONS: tuple[Question, ...] = (
    Question(
        "ml-01",
        "机器学习基础",
        "监督学习",
        "choice",
        "下面哪一项最符合监督学习的定义？",
        "使用带标签样本学习输入到目标的映射",
        (
            "只让模型压缩输入数据",
            "使用带标签样本学习输入到目标的映射",
            "不使用任何损失函数",
            "只通过环境奖励更新策略",
        ),
        explanation="监督学习的核心是已知训练样本的目标值或类别，模型通过损失函数学习从输入到目标的映射。",
        link=MODULE_LINKS["机器学习基础"],
        difficulty="基础",
    ),
    Question(
        "ml-02",
        "机器学习基础",
        "泛化",
        "true_false",
        "训练集准确率很高，就一定说明模型在未见数据上泛化很好。",
        False,
        explanation="训练集表现高可能只是记住了训练样本。判断泛化要看验证集、测试集或交叉验证表现。",
        link=MODULE_LINKS["机器学习基础"],
        difficulty="基础",
    ),
    Question(
        "ml-03",
        "机器学习基础",
        "损失函数",
        "blank",
        "二分类中常用的概率型损失函数是 ______。",
        "交叉熵",
        aliases=("binary cross entropy", "bce", "cross entropy", "对数损失", "log loss"),
        explanation="二分类通常把模型输出解释成概率，再用二元交叉熵惩罚错误且自信的预测。",
        link=MODULE_LINKS["机器学习基础"],
        difficulty="基础",
    ),
    Question(
        "ml-04",
        "机器学习基础",
        "过拟合",
        "choice",
        "当训练误差持续下降而验证误差开始上升时，最可能发生了什么？",
        "过拟合",
        ("欠拟合", "过拟合", "学习率过小且没有训练", "标签编码一定错误"),
        explanation="训练误差下降说明模型越来越适配训练集；验证误差上升说明这种适配没有转化为未见数据能力。",
        link=MODULE_LINKS["机器学习基础"],
        difficulty="基础",
    ),
    Question(
        "ml-05",
        "机器学习基础",
        "优化",
        "true_false",
        "学习率过大可能导致损失震荡甚至发散。",
        True,
        explanation="学习率控制每次参数更新步长。步长过大时会越过低损失区域，导致训练不稳定。",
        link=MODULE_LINKS["机器学习基础"],
        difficulty="基础",
    ),
    Question(
        "ml-06",
        "机器学习基础",
        "正则化",
        "blank",
        "L2 正则化常被称为权重 ______。",
        "衰减",
        aliases=("weight decay", "权重衰减"),
        explanation="L2 正则会惩罚较大的权重，在优化器实现中常对应 weight decay。",
        link=MODULE_LINKS["机器学习基础"],
        difficulty="进阶",
    ),
    Question(
        "cnn-01",
        "CNN",
        "卷积核",
        "choice",
        "卷积层中一个卷积核在整张图上共享参数，主要带来什么好处？",
        "减少参数并捕捉平移相关的局部模式",
        ("让模型只能处理灰度图", "减少参数并捕捉平移相关的局部模式", "完全消除过拟合", "让输出尺寸永远不变"),
        explanation="参数共享让同一种局部特征可以在不同位置被检测到，同时显著降低参数数量。",
        link=MODULE_LINKS["CNN"],
        difficulty="基础",
    ),
    Question(
        "cnn-02",
        "CNN",
        "感受野",
        "blank",
        "网络越深，后层神经元在原图上的有效 ______ 通常越大。",
        "感受野",
        aliases=("receptive field",),
        explanation="多层卷积叠加会让后层单元聚合更大范围的输入区域，因此能表达更高层语义。",
        link=MODULE_LINKS["CNN"],
        difficulty="基础",
    ),
    Question(
        "cnn-03",
        "CNN",
        "池化",
        "true_false",
        "最大池化可以降低空间分辨率，并增强一定程度的位置鲁棒性。",
        True,
        explanation="池化把局部区域汇聚为较少的输出，减少计算，并让小范围位移不那么影响特征响应。",
        link=MODULE_LINKS["CNN"],
        difficulty="基础",
    ),
    Question(
        "cnn-04",
        "CNN",
        "填充",
        "choice",
        "在 stride=1 的 3x3 卷积中，如果希望输入输出宽高保持不变，通常使用多大的 padding？",
        "1",
        ("0", "1", "2", "3"),
        explanation="3x3 卷积每个方向会少 2 个像素，padding=1 可以补回边界，使宽高保持不变。",
        link=MODULE_LINKS["CNN"],
        difficulty="基础",
    ),
    Question(
        "cnn-05",
        "CNN",
        "残差连接",
        "true_false",
        "ResNet 的残差连接有助于缓解深层网络训练中的梯度传播问题。",
        True,
        explanation="跳连提供了更直接的梯度路径，使深层网络更容易优化，也鼓励学习相对于输入的残差映射。",
        link=MODULE_LINKS["CNN"],
        difficulty="进阶",
    ),
    Question(
        "cnn-06",
        "CNN",
        "迁移学习",
        "blank",
        "使用 ImageNet 预训练模型再适配小数据任务，通常称为 ______ 学习。",
        "迁移",
        aliases=("迁移学习", "transfer learning"),
        explanation="迁移学习复用大规模数据上学到的通用视觉特征，再针对目标任务微调或训练分类头。",
        link=MODULE_LINKS["CNN"],
        difficulty="基础",
    ),
    Question(
        "rnn-01",
        "RNN",
        "循环状态",
        "choice",
        "RNN 适合处理序列数据的关键机制是？",
        "隐藏状态在时间步之间传递信息",
        ("只使用 1x1 卷积", "隐藏状态在时间步之间传递信息", "完全不共享参数", "只能一次读取全部未来标签"),
        explanation="RNN 在每个时间步更新隐藏状态，隐藏状态携带此前输入的信息，因此适合序列建模。",
        link=MODULE_LINKS["RNN"],
        difficulty="基础",
    ),
    Question(
        "rnn-02",
        "RNN",
        "梯度问题",
        "true_false",
        "普通 RNN 在长序列训练中可能出现梯度消失或梯度爆炸。",
        True,
        explanation="循环计算相当于反复乘以类似的雅可比矩阵，长链式求导容易导致梯度指数级变小或变大。",
        link=MODULE_LINKS["RNN"],
        difficulty="基础",
    ),
    Question(
        "rnn-03",
        "RNN",
        "LSTM",
        "blank",
        "LSTM 中控制旧记忆保留程度的门通常叫 ______ 门。",
        "遗忘",
        aliases=("forget", "forget gate", "遗忘门"),
        explanation="遗忘门决定上一时刻细胞状态中哪些信息继续保留，哪些信息被削弱。",
        link=MODULE_LINKS["RNN"],
        difficulty="基础",
    ),
    Question(
        "rnn-04",
        "RNN",
        "GRU",
        "choice",
        "与 LSTM 相比，GRU 的典型特点是？",
        "门结构更简化，参数通常更少",
        ("必须使用卷积核", "门结构更简化，参数通常更少", "无法处理文本", "没有隐藏状态"),
        explanation="GRU 用更新门、重置门等较简化结构建模记忆，通常参数少于 LSTM。",
        link=MODULE_LINKS["RNN"],
        difficulty="基础",
    ),
    Question(
        "rnn-05",
        "RNN",
        "双向模型",
        "true_false",
        "双向 RNN 可以同时利用当前位置左侧和右侧的上下文，因此常用于离线序列标注。",
        True,
        explanation="双向结构包含正向和反向两个序列编码器，适合不要求实时生成且能看到完整序列的任务。",
        link=MODULE_LINKS["RNN"],
        difficulty="进阶",
    ),
    Question(
        "rnn-06",
        "RNN",
        "序列到序列",
        "blank",
        "Seq2Seq 模型中把源序列压成表示的部分通常叫 ______。",
        "编码器",
        aliases=("encoder",),
        explanation="编码器读取源序列并形成上下文表示，解码器再基于该表示逐步生成目标序列。",
        link=MODULE_LINKS["RNN"],
        difficulty="基础",
    ),
    Question(
        "tf-01",
        "Transformer",
        "自注意力",
        "choice",
        "自注意力中 Q、K、V 分别常被称为？",
        "查询、键、值",
        ("查询、键、值", "卷积、池化、归一化", "输入、标签、损失", "均值、方差、标准差"),
        explanation="注意力通过 Query 与 Key 的相似度决定权重，再对 Value 做加权汇聚。",
        link=MODULE_LINKS["Transformer"],
        difficulty="基础",
    ),
    Question(
        "tf-02",
        "Transformer",
        "位置编码",
        "blank",
        "Transformer 为了表达词序，通常需要加入 ______ 编码。",
        "位置",
        aliases=("positional", "位置编码", "positional encoding"),
        explanation="纯注意力本身对输入排列不含顺序偏置，因此需要位置编码或相对位置机制注入序列顺序。",
        link=MODULE_LINKS["Transformer"],
        difficulty="基础",
    ),
    Question(
        "tf-03",
        "Transformer",
        "多头注意力",
        "true_false",
        "多头注意力可以让不同头在不同子空间中关注不同关系。",
        True,
        explanation="每个头有独立投影，能学习语法依赖、局部邻近、长程引用等不同模式。",
        link=MODULE_LINKS["Transformer"],
        difficulty="基础",
    ),
    Question(
        "tf-04",
        "Transformer",
        "复杂度",
        "choice",
        "标准自注意力对序列长度 n 的主要计算和显存复杂度通常接近？",
        "O(n^2)",
        ("O(log n)", "O(n)", "O(n^2)", "O(1)"),
        explanation="注意力要计算每个 token 与每个 token 的相似度，因此注意力矩阵大小是 n x n。",
        link=MODULE_LINKS["Transformer"],
        difficulty="进阶",
    ),
    Question(
        "tf-05",
        "Transformer",
        "架构差异",
        "true_false",
        "GPT 类模型通常使用因果掩码，避免当前位置看到未来 token。",
        True,
        explanation="自回归语言模型按从左到右生成，下一个词预测不能泄露未来词，因此需要因果 mask。",
        link=MODULE_LINKS["Transformer"],
        difficulty="基础",
    ),
    Question(
        "tf-06",
        "Transformer",
        "归一化",
        "blank",
        "Transformer 块中常与残差连接搭配使用的归一化层是 Layer ______。",
        "Norm",
        aliases=("normalization", "layernorm", "层归一化", "归一化"),
        explanation="LayerNorm 稳定每个样本内部的特征分布，配合残差连接改善深层堆叠的训练稳定性。",
        link=MODULE_LINKS["Transformer"],
        difficulty="基础",
    ),
    Question(
        "gan-01",
        "GAN",
        "博弈结构",
        "choice",
        "GAN 训练中生成器的直接目标是？",
        "生成能骗过判别器的样本",
        ("压缩输入图像", "生成能骗过判别器的样本", "给真实样本打标签", "计算注意力位置编码"),
        explanation="生成器从噪声或条件输入产生样本，目标是让判别器难以区分真假。",
        link=MODULE_LINKS["GAN"],
        difficulty="基础",
    ),
    Question(
        "gan-02",
        "GAN",
        "判别器",
        "true_false",
        "GAN 的判别器负责判断输入样本是真实数据还是生成样本。",
        True,
        explanation="判别器是二分类器，它提供生成器改进所需的训练信号。",
        link=MODULE_LINKS["GAN"],
        difficulty="基础",
    ),
    Question(
        "gan-03",
        "GAN",
        "潜变量",
        "blank",
        "GAN 生成器通常从随机噪声或潜在向量 z 所在的 ______ 空间开始生成。",
        "潜在",
        aliases=("latent", "潜空间", "潜在空间", "latent space"),
        explanation="潜在空间提供可采样、可插值的低维表示，生成器把它映射到数据空间。",
        link=MODULE_LINKS["GAN"],
        difficulty="基础",
    ),
    Question(
        "gan-04",
        "GAN",
        "训练难点",
        "choice",
        "GAN 常见的训练问题之一是？",
        "模式崩塌",
        ("模式崩塌", "卷积核只能为奇数", "无法使用 GPU", "标签一定需要 one-hot"),
        explanation="模式崩塌指生成器只覆盖少数样本模式，生成结果缺乏多样性。",
        link=MODULE_LINKS["GAN"],
        difficulty="进阶",
    ),
    Question(
        "gan-05",
        "GAN",
        "自编码器",
        "true_false",
        "自编码器通常由编码器和解码器组成，目标是重构输入。",
        True,
        explanation="编码器把输入压缩为表示，解码器从表示恢复输入，重构误差提供训练信号。",
        link=MODULE_LINKS["GAN"],
        difficulty="基础",
    ),
    Question(
        "gan-06",
        "GAN",
        "VAE",
        "blank",
        "VAE 的损失通常包含重构项和 ______ 散度项。",
        "KL",
        aliases=("kl divergence", "kullback-leibler", "kl散度", "KL散度"),
        explanation="KL 散度约束潜变量分布接近先验分布，使模型能从先验中采样生成。",
        link=MODULE_LINKS["GAN"],
        difficulty="进阶",
    ),
)


def render_note(title: str, body: str) -> None:
    st.markdown(f'<div class="note"><strong>{escape(title)}</strong> {escape(body)}</div>', unsafe_allow_html=True)


def stable_seed(*parts: str) -> int:
    text = "::".join(parts).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


def normalize_blank(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("，", ",")
        .replace("。", ".")
    )


def is_correct(question: Question, user_answer: Any) -> bool:
    if question.qtype == "true_false":
        return bool(user_answer) is bool(question.answer)

    if question.qtype == "choice":
        return str(user_answer) == str(question.answer)

    normalized = normalize_blank(str(user_answer))
    accepted = (str(question.answer), *question.aliases)
    return any(normalized == normalize_blank(item) for item in accepted)


def answer_text(question: Question) -> str:
    if question.qtype == "true_false":
        return "正确" if question.answer else "错误"
    return str(question.answer)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "quiz_history": [],
        "quiz_submitted": False,
        "quiz_result": None,
        "quiz_active_ids": [],
        "quiz_run_id": 0,
        "quiz_config": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def questions_by_module() -> dict[str, list[Question]]:
    grouped: dict[str, list[Question]] = {}
    for question in QUESTIONS:
        grouped.setdefault(question.module, []).append(question)
    return grouped


def select_questions(module_names: list[str], count: int, mode: str, run_id: int) -> list[Question]:
    grouped = questions_by_module()
    pool = [question for module in module_names for question in grouped[module]]
    if mode == "混合题型":
        ordered = pool[:]
    else:
        qtype = {"只做选择题": "choice", "只做判断题": "true_false", "只做填空题": "blank"}[mode]
        ordered = [question for question in pool if question.qtype == qtype]
        if len(ordered) < count:
            ordered = pool[:]

    rng = random.Random(stable_seed(",".join(module_names), mode, str(run_id)))
    rng.shuffle(ordered)
    return ordered[: min(count, len(ordered))]


def clear_current_submission() -> None:
    st.session_state.quiz_submitted = False
    st.session_state.quiz_result = None


def start_new_quiz(module_names: list[str], count: int, mode: str) -> None:
    st.session_state.quiz_run_id += 1
    selected = select_questions(module_names, count, mode, st.session_state.quiz_run_id)
    st.session_state.quiz_active_ids = [question.id for question in selected]
    st.session_state.quiz_config = (tuple(module_names), count, mode)
    clear_current_submission()


def active_questions(module_names: list[str], count: int, mode: str) -> list[Question]:
    by_id = {question.id: question for question in QUESTIONS}
    config = (tuple(module_names), count, mode)
    if st.session_state.quiz_config != config:
        start_new_quiz(module_names, count, mode)

    active_ids = st.session_state.quiz_active_ids
    selected = [by_id[qid] for qid in active_ids if qid in by_id]
    if selected:
        return selected

    selected = select_questions(module_names, count, mode, st.session_state.quiz_run_id)
    st.session_state.quiz_active_ids = [question.id for question in selected]
    return selected


def collect_answer(question: Question, index: int, submitted: bool) -> Any:
    key = f"quiz_answer_{st.session_state.quiz_run_id}_{question.id}"
    disabled = submitted
    if question.qtype == "choice":
        return st.radio(
            "选择一个答案",
            question.options,
            key=key,
            disabled=disabled,
            label_visibility="collapsed",
        )
    if question.qtype == "true_false":
        label_map = {"正确": True, "错误": False}
        selected = st.radio(
            "判断正误",
            list(label_map),
            key=key,
            horizontal=True,
            disabled=disabled,
            label_visibility="collapsed",
        )
        return label_map[selected]

    return st.text_input(
        "填写答案",
        key=key,
        disabled=disabled,
        placeholder="输入关键词即可，例如：交叉熵",
        label_visibility="collapsed",
    )


def render_question(question: Question, index: int, submitted: bool, result_map: dict[str, dict[str, Any]]) -> Any:
    type_label = {"choice": "选择题", "true_false": "判断题", "blank": "填空题"}[question.qtype]
    st.markdown(
        f"""
        <div class="question-card">
          <div class="question-title">{index}. {escape(question.prompt)}</div>
          <span class="pill">{escape(question.module)}</span>
          <span class="pill">{escape(type_label)}</span>
          <span class="pill">{escape(question.topic)}</span>
          <span class="pill">{escape(question.difficulty)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_answer = collect_answer(question, index, submitted)
    if submitted and question.id in result_map:
        item = result_map[question.id]
        status = '<span class="ok">答对</span>' if item["correct"] else '<span class="bad">答错</span>'
        shown_user = "正确" if item["user_answer"] is True else "错误" if item["user_answer"] is False else str(item["user_answer"])
        st.markdown(
            f"""
            <div class="answer-box">
            {status}<br>
            你的答案：{escape(shown_user or "未填写")}<br>
            标准答案：{escape(answer_text(question))}<br>
            解析：{escape(question.explanation)}<br>
            知识点链接：<code>{escape(question.link)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return user_answer


def submit_quiz(questions: list[Question], answers: dict[str, Any]) -> None:
    details = []
    correct_count = 0
    for question in questions:
        user_answer = answers.get(question.id, "")
        correct = is_correct(question, user_answer)
        correct_count += int(correct)
        details.append(
            {
                "question_id": question.id,
                "module": question.module,
                "topic": question.topic,
                "qtype": question.qtype,
                "difficulty": question.difficulty,
                "prompt": question.prompt,
                "user_answer": user_answer,
                "answer": answer_text(question),
                "correct": correct,
                "link": question.link,
            }
        )

    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": correct_count,
        "total": len(questions),
        "accuracy": correct_count / max(1, len(questions)),
        "details": details,
    }
    st.session_state.quiz_history.append(record)
    st.session_state.quiz_result = record
    st.session_state.quiz_submitted = True


def history_dataframe() -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(st.session_state.quiz_history, 1):
        rows.append(
            {
                "轮次": idx,
                "时间": record["time"],
                "正确数": record["score"],
                "题数": record["total"],
                "正确率": record["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def detail_dataframe() -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(st.session_state.quiz_history, 1):
        for item in record["details"]:
            rows.append(
                {
                    "轮次": idx,
                    "模块": item["module"],
                    "知识点": item["topic"],
                    "题型": item["qtype"],
                    "难度": item["difficulty"],
                    "是否正确": item["correct"],
                    "链接": item["link"],
                }
            )
    return pd.DataFrame(rows)


def module_stats() -> pd.DataFrame:
    details = detail_dataframe()
    if details.empty:
        return pd.DataFrame(columns=["模块", "答题数", "正确数", "正确率"])
    grouped = (
        details.groupby("模块", as_index=False)
        .agg(答题数=("是否正确", "size"), 正确数=("是否正确", "sum"))
        .sort_values(["正确数", "答题数"], ascending=[True, False])
    )
    grouped["正确率"] = grouped["正确数"] / grouped["答题数"]
    return grouped.sort_values("正确率", ascending=True)


def plot_progress(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["轮次"],
                y=df["正确率"] * 100,
                mode="lines+markers",
                name="总体正确率",
                line=dict(color="#207f7a", width=3),
                marker=dict(size=8),
            )
        )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=28, b=20),
        yaxis=dict(title="正确率", ticksuffix="%", range=[0, 100]),
        xaxis=dict(title="测验轮次", dtick=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        font=dict(color="#172027"),
    )
    return fig


def plot_module_accuracy(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not stats.empty:
        fig.add_trace(
            go.Bar(
                x=stats["模块"],
                y=stats["正确率"] * 100,
                marker_color=["#b84d5a" if value < 0.7 else "#c47f1f" if value < 0.85 else "#207f7a" for value in stats["正确率"]],
                text=[f"{value:.0%}" for value in stats["正确率"]],
                textposition="outside",
            )
        )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=28, b=20),
        yaxis=dict(title="正确率", ticksuffix="%", range=[0, 105]),
        xaxis=dict(title="知识模块"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        font=dict(color="#172027"),
    )
    return fig


def weak_recommendations(stats: pd.DataFrame) -> list[str]:
    if stats.empty:
        return ["先完成一轮测验，系统会按模块正确率生成复习建议。"]

    weak = stats[stats["正确率"] < 0.75].copy()
    if weak.empty:
        weakest = stats.head(1)
        return [
            f"整体表现稳定。下一步可以挑战 {weakest.iloc[0]['模块']} 的进阶题，并回看 `{MODULE_LINKS[weakest.iloc[0]['模块']]}`。"
        ]

    recs = []
    for _, row in weak.iterrows():
        module = row["模块"]
        recs.append(
            f"{module}：当前正确率 {row['正确率']:.0%}，建议复习 `{MODULE_LINKS[module]}`，优先补齐错题涉及的核心概念。"
        )
    return recs


def render_summary() -> None:
    history = history_dataframe()
    details = detail_dataframe()
    stats = module_stats()

    total_rounds = len(history)
    total_answered = int(history["题数"].sum()) if not history.empty else 0
    total_correct = int(history["正确数"].sum()) if not history.empty else 0
    overall_acc = total_correct / total_answered if total_answered else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("测验轮次", total_rounds)
    c2.metric("累计答题", total_answered)
    c3.metric("累计正确", total_correct)
    c4.metric("总体正确率", f"{overall_acc:.0%}" if total_answered else "-")

    left, right = st.columns([0.58, 0.42])
    with left:
        st.plotly_chart(plot_progress(history), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.plotly_chart(plot_module_accuracy(stats), use_container_width=True, config={"displayModeBar": False})

    st.subheader("复习推荐")
    for item in weak_recommendations(stats):
        st.markdown(f"- {item}")

    if not details.empty:
        with st.expander("答题历史明细", expanded=False):
            visible = details.copy()
            visible["是否正确"] = visible["是否正确"].map({True: "正确", False: "错误"})
            st.dataframe(visible, use_container_width=True, hide_index=True)


def main() -> None:
    init_state()
    grouped = questions_by_module()

    with st.sidebar:
        st.header("测验设置")
        module_names = st.multiselect(
            "知识模块",
            list(grouped),
            default=list(grouped),
        )
        if not module_names:
            st.warning("至少选择一个模块。")
            module_names = list(grouped)

        max_count = sum(len(grouped[module]) for module in module_names)
        question_count = st.slider("题目数量", 5, max(5, max_count), min(10, max_count), 1)
        mode = st.segmented_control(
            "题型范围",
            ["混合题型", "只做选择题", "只做判断题", "只做填空题"],
            default="混合题型",
        )
        st.divider()
        if st.button("重新抽题", use_container_width=True):
            start_new_quiz(module_names, question_count, mode)
            st.rerun()
        if st.button("清空历史", use_container_width=True):
            st.session_state.quiz_history = []
            clear_current_submission()
            st.rerun()
        st.caption("答题历史保存在当前 Streamlit session_state 中，刷新会保留，同一浏览器会话关闭后清空。")

    st.title("练习题与测验系统")
    render_note(
        "学习方式",
        "先做题，再看解析。系统会记录每轮正确率，并根据模块表现推荐复习入口。",
    )

    questions = active_questions(module_names, question_count, mode)
    result = st.session_state.quiz_result if st.session_state.quiz_submitted else None
    result_map = {item["question_id"]: item for item in result["details"]} if result else {}

    tab_quiz, tab_dashboard, tab_bank = st.tabs(["开始测验", "统计与复习", "题库概览"])

    with tab_quiz:
        st.subheader("本轮题目")
        answers: dict[str, Any] = {}
        for index, question in enumerate(questions, 1):
            answers[question.id] = render_question(question, index, st.session_state.quiz_submitted, result_map)

        left, right = st.columns([0.25, 0.75])
        with left:
            if st.button("提交答案", disabled=st.session_state.quiz_submitted, use_container_width=True):
                submit_quiz(questions, answers)
                st.rerun()
        with right:
            if result:
                st.success(f"本轮得分：{result['score']}/{result['total']}，正确率 {result['accuracy']:.0%}")

    with tab_dashboard:
        render_summary()

    with tab_bank:
        st.subheader("题库覆盖")
        rows = [
            {
                "模块": question.module,
                "知识点": question.topic,
                "题型": {"choice": "选择题", "true_false": "判断题", "blank": "填空题"}[question.qtype],
                "难度": question.difficulty,
                "知识点链接": question.link,
            }
            for question in QUESTIONS
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
