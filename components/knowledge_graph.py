"""深度学习知识图谱元数据与导航组件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class KnowledgeNode:
    name: str
    title: str
    description: str
    prerequisites: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    mastery_criteria: str = ""
    practice_target: str = ""
    tags: list[str] = field(default_factory=list)
    difficulty: str = "入门"


KNOWLEDGE_GRAPH: dict[str, KnowledgeNode] = {
    "math_primer": KnowledgeNode(
        name="math_primer",
        title="数学基础速查",
        description="补齐线性代数、微积分、概率和梯度下降的最低必要知识。",
        prerequisites=[],
        related=["tensors_gradients", "classical_ml", "training_dynamics"],
        next_steps=["tensors_gradients", "classical_ml"],
        mastery_criteria="能解释向量、矩阵、导数、概率和梯度下降各自解决什么问题。",
        practice_target="手写一次一元函数梯度下降，并画出损失变化。",
        tags=["数学", "基础", "梯度"],
        difficulty="入门",
    ),
    "tensors_gradients": KnowledgeNode(
        name="tensors_gradients",
        title="张量与梯度",
        description="理解张量形状、自动求导和反向传播的基本链路。",
        prerequisites=["math_primer"],
        related=["activation_functions", "training_dynamics", "gradient_monitor"],
        next_steps=["activation_functions", "neural_network_basics"],
        mastery_criteria="能判断张量形状是否匹配，并说清梯度从损失回传到参数的路径。",
        practice_target="构造一个两层线性模型，打印每个参数的梯度范数。",
        tags=["张量", "梯度", "自动求导"],
        difficulty="入门",
    ),
    "activation_functions": KnowledgeNode(
        name="activation_functions",
        title="激活函数",
        description="比较 Sigmoid、Tanh、ReLU 等非线性函数对表达能力和梯度的影响。",
        prerequisites=["tensors_gradients"],
        related=["neural_network_basics", "training_dynamics", "gradient_monitor"],
        next_steps=["neural_network_basics", "training_dynamics"],
        mastery_criteria="能说明为什么神经网络需要非线性，以及饱和区为什么会削弱梯度。",
        practice_target="对比 ReLU 与 Tanh 在同一玩具任务上的训练曲线。",
        tags=["激活函数", "非线性", "梯度"],
        difficulty="入门",
    ),
    "classical_ml": KnowledgeNode(
        name="classical_ml",
        title="经典机器学习",
        description="用传统模型建立监督学习、损失、泛化和基线意识。",
        prerequisites=["math_primer"],
        related=["neural_network_basics", "training_dynamics"],
        next_steps=["neural_network_basics", "training_dynamics"],
        mastery_criteria="能为一个任务选择简单基线，并用验证集判断是否值得上深度模型。",
        practice_target="在同一数据集上比较逻辑回归、SVM 和小型 MLP。",
        tags=["机器学习", "基线", "评估"],
        difficulty="入门",
    ),
    "neural_network_basics": KnowledgeNode(
        name="neural_network_basics",
        title="神经网络基础",
        description="从感知机到多层网络，理解层、参数、损失和反向传播。",
        prerequisites=["tensors_gradients", "activation_functions", "classical_ml"],
        related=["training_dynamics", "convolution_visual", "sequence_models"],
        next_steps=["convolution_visual", "rnn_intuition", "training_dynamics"],
        mastery_criteria="能解释一个 MLP 的前向计算、损失计算和参数更新。",
        practice_target="训练一个小型 MLP 解决 XOR 或螺旋分类任务。",
        tags=["神经网络", "反向传播", "MLP"],
        difficulty="入门",
    ),
    "convolution_visual": KnowledgeNode(
        name="convolution_visual",
        title="卷积直觉",
        description="通过滑窗、卷积核和特征图理解局部模式提取。",
        prerequisites=["neural_network_basics"],
        related=["cnn_architectures", "gradient_monitor"],
        next_steps=["cnn_architectures", "training_dynamics"],
        mastery_criteria="能说明卷积核、步幅、填充和感受野如何影响特征图。",
        practice_target="用几个手写卷积核观察边缘、模糊和锐化效果。",
        tags=["CNN", "卷积", "视觉"],
        difficulty="进阶",
    ),
    "cnn_architectures": KnowledgeNode(
        name="cnn_architectures",
        title="CNN 架构",
        description="理解 LeNet、VGG、ResNet 等视觉网络如何组织卷积、池化和残差。",
        prerequisites=["convolution_visual"],
        related=["training_dynamics", "gradient_monitor"],
        next_steps=["training_dynamics", "gradient_monitor"],
        mastery_criteria="能说清经典 CNN 架构的主要差异和残差连接的训练价值。",
        practice_target="对比一个浅层 CNN 和带残差连接的小网络。",
        tags=["CNN", "架构", "残差"],
        difficulty="进阶",
    ),
    "rnn_intuition": KnowledgeNode(
        name="rnn_intuition",
        title="RNN 直觉",
        description="从循环状态理解序列信息如何随时间流动。",
        prerequisites=["neural_network_basics"],
        related=["sequence_models", "attention_mechanism", "training_dynamics"],
        next_steps=["sequence_models", "attention_mechanism"],
        mastery_criteria="能解释隐藏状态如何携带历史信息，以及长序列为什么难训练。",
        practice_target="训练一个字符级序列预测模型并观察隐藏状态变化。",
        tags=["RNN", "序列", "隐藏状态"],
        difficulty="进阶",
    ),
    "sequence_models": KnowledgeNode(
        name="sequence_models",
        title="序列模型",
        description="理解 RNN、LSTM、GRU 与序列任务的基本范式。",
        prerequisites=["rnn_intuition"],
        related=["attention_mechanism", "training_dynamics", "gradient_monitor"],
        next_steps=["attention_mechanism", "training_dynamics"],
        mastery_criteria="能区分普通 RNN、LSTM、GRU 的记忆机制和适用场景。",
        practice_target="在同一序列任务上比较 RNN、LSTM 和 GRU。",
        tags=["序列", "LSTM", "GRU"],
        difficulty="进阶",
    ),
    "attention_mechanism": KnowledgeNode(
        name="attention_mechanism",
        title="注意力机制",
        description="从查询、键、值和权重矩阵理解信息检索过程。",
        prerequisites=["sequence_models"],
        related=["transformer_models", "rnn_intuition"],
        next_steps=["transformer_models", "training_dynamics"],
        mastery_criteria="能用查询、键、值解释注意力权重表示什么。",
        practice_target="手算一个三词序列的缩放点积注意力。",
        tags=["注意力", "NLP", "权重"],
        difficulty="核心",
    ),
    "transformer_models": KnowledgeNode(
        name="transformer_models",
        title="Transformer 模型",
        description="拆解自注意力、多头注意力、位置编码、残差和前馈网络。",
        prerequisites=["attention_mechanism"],
        related=["training_dynamics", "gradient_monitor"],
        next_steps=["training_dynamics", "gradient_monitor"],
        mastery_criteria="能画出一个 Transformer Block 的数据流，并说明每个子层的作用。",
        practice_target="实现一个最小 Transformer Block 并检查张量形状。",
        tags=["Transformer", "多头注意力", "位置编码"],
        difficulty="核心",
    ),
    "training_dynamics": KnowledgeNode(
        name="training_dynamics",
        title="训练动态",
        description="用损失曲线、指标曲线和学习率变化观察模型如何学习。",
        prerequisites=["tensors_gradients", "neural_network_basics"],
        related=["gradient_monitor", "cnn_architectures", "transformer_models"],
        next_steps=["gradient_monitor"],
        mastery_criteria="能根据训练和验证曲线判断欠拟合、过拟合、震荡或发散。",
        practice_target="记录一次训练过程，并写出下一轮调参决策。",
        tags=["训练", "曲线", "诊断"],
        difficulty="工程",
    ),
    "gradient_monitor": KnowledgeNode(
        name="gradient_monitor",
        title="梯度监控",
        description="监控梯度范数、爆炸、消失和更新比例，定位训练稳定性问题。",
        prerequisites=["tensors_gradients", "training_dynamics"],
        related=["activation_functions", "cnn_architectures", "transformer_models"],
        next_steps=["training_dynamics"],
        mastery_criteria="能通过梯度范数和更新比例判断训练是否健康。",
        practice_target="为一个模型添加梯度监控，并定位一次异常更新。",
        tags=["梯度", "监控", "调试"],
        difficulty="工程",
    ),
}


def get_node(key: str) -> KnowledgeNode | None:
    return KNOWLEDGE_GRAPH.get(key)


def _nodes(keys: Iterable[str]) -> list[KnowledgeNode]:
    return [node for key in keys if (node := get_node(key)) is not None]


def get_prerequisites(key: str) -> list[KnowledgeNode]:
    node = get_node(key)
    return _nodes(node.prerequisites) if node else []


def get_next_steps(key: str) -> list[KnowledgeNode]:
    node = get_node(key)
    return _nodes(node.next_steps) if node else []


def _module_url(key: str) -> str:
    route_map = {
        "tensors_gradients": "part1_foundations/01_tensors_gradients",
        "activation_functions": "part1_foundations/02_activations_normalization",
        "math_primer": "part1_foundations/math_primer",
        "classical_ml": "part1_foundations/classical_ml",
        "neural_network_basics": "part1_foundations/neural_network_basics",
        "convolution_visual": "part2_cnn/01_convolution_visual",
        "cnn_architectures": "part2_cnn/cnn_architectures",
        "rnn_intuition": "part3_rnn/01_rnn_intuition",
        "sequence_models": "part3_rnn/sequence_models",
        "attention_mechanism": "part4_transformer/01_attention_mechanism",
        "transformer_models": "part4_transformer/transformer_models",
        "training_dynamics": "part5_toolbox/03_training_dynamics",
        "gradient_monitor": "part5_toolbox/02_gradient_monitor",
    }
    from urllib.parse import quote

    return f"/?module={quote(route_map.get(key, key), safe='')}"


def _render_node_link(node: KnowledgeNode) -> None:
    st = __import__("streamlit")
    st.markdown(f"**{node.title}**")
    st.caption(f"{node.description}｜难度：{node.difficulty}")
    st.link_button("打开学习", _module_url(node.name), width="stretch")


def render知识图谱导航(current_key: str) -> None:
    """在 Streamlit 页面底部渲染前置知识和下一步推荐。"""
    try:
        st = __import__("streamlit")
        node = get_node(current_key)
        if node is None:
            st.info("暂时没有找到当前知识点的图谱信息。")
            return

        st.subheader("知识图谱导航")
        st.markdown(f"**当前位置：{node.title}**")
        st.caption(node.description)

        left, right = st.columns(2)
        with left:
            st.markdown("**前置知识**")
            prerequisites = get_prerequisites(current_key)
            if prerequisites:
                for item in prerequisites:
                    _render_node_link(item)
            else:
                st.info("这是推荐起点之一，可以直接开始。")

        with right:
            st.markdown("**下一步推荐**")
            next_steps = get_next_steps(current_key)
            if next_steps:
                for item in next_steps:
                    _render_node_link(item)
            else:
                st.info("已经到达当前路径末端，建议进入实战项目。")
    except Exception as error:
        st = __import__("streamlit")
        st.warning("知识图谱导航暂时无法显示，请继续阅读正文。")
        with st.expander("查看组件错误详情", expanded=False):
            st.code(str(error), language="text")

