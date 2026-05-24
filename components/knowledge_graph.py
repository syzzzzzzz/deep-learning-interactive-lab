"""深度学习交互式网站的知识图谱元数据与导航组件。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import quote


@dataclass(frozen=True)
class KnowledgeNode:
    name: str
    title: str
    description: str
    route: str
    prerequisites: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    mastery_criteria: str = ""
    practice_target: str = ""
    practice_route: str = ""
    tags: list[str] = field(default_factory=list)
    difficulty: str = "入门"


@dataclass(frozen=True)
class ModuleSeed:
    key: str
    route: str
    title: str
    description: str
    difficulty: str
    tags: tuple[str, ...]


# 顺序对齐主站 MODULES，也覆盖旧 Markdown 教材的章节骨架。
MODULE_SEEDS: tuple[ModuleSeed, ...] = (
    ModuleSeed("part1/01_tensors_gradients", "part1/01_tensors_gradients", "张量与梯度", "用可视化理解张量、自动求导和梯度传播。", "入门", ("基础", "张量", "梯度")),
    ModuleSeed("part1/02_activations_normalization", "part1/02_activations_normalization", "激活与归一化", "比较常见激活函数、归一化方法和训练稳定性。", "入门", ("基础", "激活函数", "归一化")),
    ModuleSeed("part1/03_datasets_optimizers", "part1/03_datasets_optimizers", "数据集与优化器", "理解数据划分、批训练、SGD、Adam 和优化曲线。", "入门", ("基础", "数据", "优化")),
    ModuleSeed("part1/math_primer", "part1/math_primer", "数学基础速查", "线性代数、微积分、概率论和梯度下降的交互式速查。", "入门", ("基础", "数学", "可视化")),
    ModuleSeed("part1/machine_learning_basics", "part1/machine_learning_basics", "机器学习基础", "监督学习、损失函数、泛化、评估和模型选择。", "入门", ("基础", "机器学习", "评估")),
    ModuleSeed("part1/neural_network_basics", "part1/neural_network_basics", "神经网络基础", "从感知机到多层网络，理解反向传播和非线性表达。", "入门", ("基础", "神经网络", "反向传播")),
    ModuleSeed("part1/classical_ml", "part1/classical_ml", "经典机器学习", "用传统模型建立深度学习前的基线意识。", "入门", ("基础", "机器学习", "模型", "基线")),
    ModuleSeed("part2/01_convolution_visual", "part2/01_convolution_visual", "卷积直觉", "用滑窗、卷积核和边缘检测建立 CNN 直觉。", "进阶", ("视觉", "CNN", "卷积")),
    ModuleSeed("part2/02_feature_maps", "part2/02_feature_maps", "特征图可视化", "观察卷积层如何从局部纹理逐步形成抽象特征。", "进阶", ("视觉", "CNN", "可视化")),
    ModuleSeed("part2/03_classic_architectures", "part2/03_classic_architectures", "经典 CNN 架构", "梳理 LeNet、AlexNet、VGG、GoogLeNet 和 ResNet。", "进阶", ("视觉", "CNN", "架构")),
    ModuleSeed("part2/04_debug_panel", "part2/04_debug_panel", "CNN 调试面板", "定位卷积模型训练中的过拟合、梯度和数据问题。", "工程", ("视觉", "调试", "训练")),
    ModuleSeed("part2/05_mnist_toy", "part2/05_mnist_toy", "MNIST 玩具实验", "用小型手写数字实验串起数据、模型、训练和评估。", "实验", ("视觉", "CNN", "实验")),
    ModuleSeed("part2/06_modern_architectures", "part2/06_modern_architectures", "现代 CNN 架构", "理解残差、深度可分离卷积和高效视觉网络。", "进阶", ("视觉", "CNN", "架构", "残差")),
    ModuleSeed("part2/07_advanced_convolution", "part2/07_advanced_convolution", "高级卷积技术", "扩张卷积、转置卷积、分组卷积和感受野分析。", "进阶", ("视觉", "CNN", "卷积")),
    ModuleSeed("part2/08_visualization_gradcam", "part2/08_visualization_gradcam", "Grad-CAM 可视化", "用热力图解释 CNN 决策关注区域。", "实验", ("视觉", "解释性", "可视化")),
    ModuleSeed("part2/09_transfer_learning", "part2/09_transfer_learning", "迁移学习", "复用预训练模型完成小数据任务。", "工程", ("视觉", "迁移学习", "工程")),
    ModuleSeed("part2/cnn_architectures", "part2/cnn_architectures", "CNN 架构实验", "对比经典卷积网络的结构与特征提取方式。", "进阶", ("视觉", "CNN", "架构")),
    ModuleSeed("part2/advanced_cnn", "part2/advanced_cnn", "高级 CNN", "现代卷积技巧、残差思想和视觉模型设计。", "进阶", ("视觉", "CNN", "残差")),
    ModuleSeed("part3/01_rnn_intuition", "part3/01_rnn_intuition", "RNN 直觉", "从循环状态理解序列信息如何流动。", "进阶", ("序列", "RNN", "隐藏状态")),
    ModuleSeed("part3/02_hidden_states", "part3/02_hidden_states", "隐藏状态", "观察隐藏状态、门控结构和长期依赖。", "进阶", ("序列", "RNN", "可视化")),
    ModuleSeed("part3/03_sequence_toys", "part3/03_sequence_toys", "序列玩具任务", "用可控任务理解记忆、预测和序列泛化。", "实验", ("序列", "RNN", "实验")),
    ModuleSeed("part3/04_hyperparam_rnn", "part3/04_hyperparam_rnn", "RNN 超参实验", "比较学习率、隐藏维度、层数和截断反传。", "实验", ("序列", "训练", "超参数")),
    ModuleSeed("part3/05_seq2seq_attention", "part3/05_seq2seq_attention", "Seq2Seq 与注意力", "理解编码器解码器和注意力对齐。", "核心", ("序列", "注意力", "NLP")),
    ModuleSeed("part3/06_text_classification", "part3/06_text_classification", "文本分类", "用序列模型完成文本表示和分类。", "实验", ("序列", "NLP", "分类")),
    ModuleSeed("part3/07_advanced_training", "part3/07_advanced_training", "高级训练技巧", "处理梯度裁剪、Teacher Forcing、正则化和训练稳定性。", "工程", ("序列", "训练", "调试")),
    ModuleSeed("part3/08_debug_problems", "part3/08_debug_problems", "RNN 调试问题", "定位序列模型中的梯度、数据和评估问题。", "工程", ("序列", "调试", "梯度")),
    ModuleSeed("part3/sequence_models", "part3/sequence_models", "序列模型", "RNN、LSTM、GRU 与序列任务的基本范式。", "进阶", ("序列", "NLP", "LSTM", "GRU")),
    ModuleSeed("part4/01_attention_mechanism", "part4/01_attention_mechanism", "注意力机制", "从查询、键、值理解注意力权重。", "核心", ("Transformer", "注意力", "NLP")),
    ModuleSeed("part4/02_multihead_visual", "part4/02_multihead_visual", "多头注意力可视化", "观察不同注意力头如何捕获互补关系。", "核心", ("Transformer", "注意力", "可视化")),
    ModuleSeed("part4/03_encoder_decoder", "part4/03_encoder_decoder", "编码器与解码器", "拆解 Transformer 编码器、解码器和掩码机制。", "核心", ("Transformer", "NLP", "掩码")),
    ModuleSeed("part4/04_minimal_transformer", "part4/04_minimal_transformer", "最小 Transformer", "用精简实现串起嵌入、注意力、MLP 和残差。", "核心", ("Transformer", "实现", "残差")),
    ModuleSeed("part4/05_flash_attention", "part4/05_flash_attention", "Flash Attention", "理解高效注意力的内存访问与计算优化。", "前沿", ("Transformer", "性能", "注意力")),
    ModuleSeed("part4/06_debug_problems", "part4/06_debug_problems", "Transformer 调试", "分析大模型训练中的掩码、位置编码和梯度问题。", "工程", ("Transformer", "调试", "训练")),
    ModuleSeed("part4/transformer_models", "part4/transformer_models", "Transformer 架构", "可视化拆解自注意力、多头、位置编码和 BERT/GPT。", "核心", ("Transformer", "NLP", "架构")),
    ModuleSeed("part4/gan_ae", "part4/gan_ae", "GAN 与自编码器", "理解生成模型、潜空间和重构学习。", "进阶", ("生成模型", "表征", "重构")),
    ModuleSeed("part4/gnn_intro", "part4/gnn_intro", "图神经网络", "从节点、边和消息传递理解图学习。", "进阶", ("GNN", "结构数据", "消息传递")),
    ModuleSeed("part5/01_feature_visualization", "part5/01_feature_visualization", "特征可视化", "观察特征、激活、嵌入和决策边界。", "工程", ("工具", "可视化", "解释性")),
    ModuleSeed("part5/02_gradient_monitor", "part5/02_gradient_monitor", "梯度监控", "监控梯度范数、爆炸、消失和训练健康度。", "工程", ("工具", "梯度", "调试")),
    ModuleSeed("part5/03_training_dynamics", "part5/03_training_dynamics", "训练动态", "用曲线和指标追踪模型如何学习。", "工程", ("训练", "监控", "曲线")),
    ModuleSeed("part5/04_hyperparam_search", "part5/04_hyperparam_search", "超参搜索", "比较网格搜索、随机搜索和实验记录。", "工程", ("超参数", "工具", "实验")),
    ModuleSeed("part5/05_dataset_toys", "part5/05_dataset_toys", "玩具数据集", "用小数据集快速验证模型直觉。", "实验", ("数据", "实验", "基线")),
    ModuleSeed("part5/data_training", "part5/data_training", "数据与训练", "数据管线、训练循环、指标与调试。", "工程", ("训练", "数据", "指标")),
    ModuleSeed("part5/case_studies", "part5/case_studies", "案例研究", "用完整案例串联建模、调参和诊断流程。", "工程", ("案例", "实践", "诊断")),
    ModuleSeed("part5/deployment_tools", "part5/deployment_tools", "部署工具", "模型导出、服务化、推理和工程落地。", "工程", ("部署", "工程", "推理")),
    ModuleSeed("part5/quiz_system", "part5/quiz_system", "练习题与测验", "覆盖机器学习基础、CNN、RNN、Transformer 和 GAN 的交互式测验。", "复习", ("测验", "复习", "练习")),
    ModuleSeed("part5/tuning_challenge", "part5/tuning_challenge", "调参实战挑战", "在真实约束下练习学习率、正则、模型规模和实验记录决策。", "实验", ("调参", "实验", "诊断")),
    ModuleSeed("part6/01_unified_interface", "part6/01_unified_interface", "统一接口", "把模型、数据和任务抽象成统一可扩展接口。", "工程", ("框架", "架构", "接口")),
    ModuleSeed("part6/02_modular_structure", "part6/02_modular_structure", "模块化结构", "拆分配置、数据、模型、训练和评估边界。", "工程", ("框架", "模块化", "架构")),
    ModuleSeed("part6/03_full_project", "part6/03_full_project", "完整项目骨架", "组织可复用的深度学习项目目录和执行流程。", "工程", ("项目", "架构", "工程")),
    ModuleSeed("part6/04_plugin_system", "part6/04_plugin_system", "插件系统", "用注册表和插件扩展任务、模型与工具。", "工程", ("框架", "插件", "扩展")),
    ModuleSeed("part6/05_one_click_training", "part6/05_one_click_training", "一键训练", "从配置到训练、评估和产物保存的一键流程。", "工程", ("训练", "自动化", "工程")),
    ModuleSeed("part6/06_streamlit_demo", "part6/06_streamlit_demo", "可视化实验台", "用 Streamlit 交互观察边界、卷积和注意力。", "核心", ("实验", "可视化", "Streamlit")),
    ModuleSeed("part6/neural_network_playground", "part6/neural_network_playground", "神经网络乐高工厂", "用表单构建神经网络、形状推导、代码生成、示例模型加载。", "核心", ("构建器", "Playground", "实战")),
    ModuleSeed("part6/training_demo", "part6/training_demo", "训练过程可视化", "用轻量数据集演示训练循环，实时展示损失、准确率、梯度范数。", "核心", ("训练", "可视化", "演示")),
    ModuleSeed("part6/07_project_template", "part6/07_project_template", "项目模板", "训练脚本、评估脚本、K-Fold 和集成预测模板。", "工程", ("项目", "模板", "评估")),
    ModuleSeed("part6/reinforcement_learning", "part6/reinforcement_learning", "强化学习入门", "强化学习概念、多臂老虎机、Q-Learning 和纯 Python 环境 demo。", "核心", ("RL", "强化学习", "实验")),
    ModuleSeed("part6/learning_path", "part6/learning_path", "学习路径推荐", "入门测评、个性化路径、知识图谱、进度追踪和下一步推荐。", "核心", ("路径", "知识图谱", "测评")),
    ModuleSeed("part6/glossary", "part6/glossary", "深度学习术语表", "集中检索常见概念、缩写和相关模块。", "复习", ("术语", "搜索", "复习")),
    ModuleSeed("part6/frontier", "part6/frontier", "前沿方向", "LLM、AGI、多模态、自监督、XAI、安全与对齐。", "前沿", ("LLM", "AGI", "安全")),
    ModuleSeed("part6/paper_reading_lab", "part6/paper_reading_lab", "经典论文解读实验室", "用时间线、机制图和最小复现清单读懂经典深度学习论文。", "进阶", ("论文", "可视化", "复现")),
    ModuleSeed("part7/networking", "part7/networking", "计算机网络", "TCP 握手挥手、HTTP/HTTPS、DNS 解析、高频面试题与交互练习。", "核心", ("网络", "TCP", "HTTP", "面试")),
    ModuleSeed("part7/database_sql", "part7/database_sql", "数据库 SQL", "SELECT 执行流程、B+ 树索引、慢查询排查、高频面试题与交互练习。", "核心", ("数据库", "SQL", "索引", "面试")),
    ModuleSeed("part7/data_structures", "part7/data_structures", "数据结构与算法", "数组链表可视化、排序算法动画、BFS/DFS、高频面试题。", "核心", ("数据结构", "算法", "排序", "面试")),
    ModuleSeed("part7/operating_system", "part7/operating_system", "操作系统", "进程线程、调度算法、虚拟内存、死锁、高频面试题。", "核心", ("操作系统", "进程", "内存", "面试")),
    ModuleSeed("part7/interview_quiz", "part7/interview_quiz", "面试刷题模式", "随机出题、按方向难度筛选、错题本、面试官追问。", "核心", ("刷题", "面试", "错题本")),
)


ALIASES: dict[str, str] = {
    "math_primer": "part1/math_primer",
    "tensors_gradients": "part1/01_tensors_gradients",
    "activation_functions": "part1/02_activations_normalization",
    "classical_ml": "part1/classical_ml",
    "ml_basics": "part1/machine_learning_basics",
    "neural_network_basics": "part1/neural_network_basics",
    "convolution_visual": "part2/01_convolution_visual",
    "feature_maps": "part2/02_feature_maps",
    "cnn_architectures": "part2/cnn_architectures",
    "advanced_cnn": "part2/advanced_cnn",
    "rnn_intuition": "part3/01_rnn_intuition",
    "hidden_states": "part3/02_hidden_states",
    "sequence_models": "part3/sequence_models",
    "attention_mechanism": "part4/01_attention_mechanism",
    "multihead_attention": "part4/02_multihead_visual",
    "transformer_models": "part4/transformer_models",
    "training_dynamics": "part5/03_training_dynamics",
    "gradient_monitor": "part5/02_gradient_monitor",
    "hyperparam_search": "part5/04_hyperparam_search",
    "data_training": "part5/data_training",
    "streamlit_lab": "part6/06_streamlit_demo",
    "neural_network_playground": "part6/neural_network_playground",
    "training_demo": "part6/training_demo",
    "learning_path": "part6/learning_path",
}


PREREQUISITES: dict[str, list[str]] = {
    "part1/math_primer": [],
    "part1/01_tensors_gradients": ["part1/math_primer"],
    "part1/02_activations_normalization": ["part1/01_tensors_gradients"],
    "part1/03_datasets_optimizers": ["part1/01_tensors_gradients", "part1/math_primer"],
    "part1/machine_learning_basics": ["part1/math_primer"],
    "part1/neural_network_basics": ["part1/01_tensors_gradients", "part1/02_activations_normalization", "part1/machine_learning_basics"],
    "part1/classical_ml": ["part1/machine_learning_basics", "part1/math_primer"],
    "part2/01_convolution_visual": ["part1/neural_network_basics"],
    "part2/02_feature_maps": ["part2/01_convolution_visual"],
    "part2/03_classic_architectures": ["part2/01_convolution_visual", "part2/02_feature_maps"],
    "part2/04_debug_panel": ["part2/03_classic_architectures", "part5/02_gradient_monitor"],
    "part2/05_mnist_toy": ["part2/01_convolution_visual", "part1/03_datasets_optimizers"],
    "part2/06_modern_architectures": ["part2/03_classic_architectures"],
    "part2/07_advanced_convolution": ["part2/01_convolution_visual", "part2/03_classic_architectures"],
    "part2/08_visualization_gradcam": ["part2/02_feature_maps", "part5/01_feature_visualization"],
    "part2/09_transfer_learning": ["part2/03_classic_architectures", "part5/data_training"],
    "part2/cnn_architectures": ["part2/03_classic_architectures"],
    "part2/advanced_cnn": ["part2/06_modern_architectures", "part2/07_advanced_convolution"],
    "part3/01_rnn_intuition": ["part1/neural_network_basics"],
    "part3/02_hidden_states": ["part3/01_rnn_intuition"],
    "part3/03_sequence_toys": ["part3/01_rnn_intuition", "part1/03_datasets_optimizers"],
    "part3/04_hyperparam_rnn": ["part3/03_sequence_toys", "part5/04_hyperparam_search"],
    "part3/05_seq2seq_attention": ["part3/sequence_models", "part4/01_attention_mechanism"],
    "part3/06_text_classification": ["part3/sequence_models", "part5/data_training"],
    "part3/07_advanced_training": ["part3/sequence_models", "part5/03_training_dynamics"],
    "part3/08_debug_problems": ["part3/07_advanced_training", "part5/02_gradient_monitor"],
    "part3/sequence_models": ["part3/01_rnn_intuition", "part3/02_hidden_states"],
    "part4/01_attention_mechanism": ["part3/sequence_models", "part1/math_primer"],
    "part4/02_multihead_visual": ["part4/01_attention_mechanism"],
    "part4/03_encoder_decoder": ["part4/02_multihead_visual"],
    "part4/04_minimal_transformer": ["part4/03_encoder_decoder", "part1/01_tensors_gradients"],
    "part4/05_flash_attention": ["part4/01_attention_mechanism", "part5/03_training_dynamics"],
    "part4/06_debug_problems": ["part4/transformer_models", "part5/02_gradient_monitor"],
    "part4/transformer_models": ["part4/01_attention_mechanism", "part4/02_multihead_visual"],
    "part4/gan_ae": ["part1/neural_network_basics", "part5/03_training_dynamics"],
    "part4/gnn_intro": ["part1/neural_network_basics", "part1/math_primer"],
    "part5/01_feature_visualization": ["part2/02_feature_maps", "part1/neural_network_basics"],
    "part5/02_gradient_monitor": ["part1/01_tensors_gradients", "part5/03_training_dynamics"],
    "part5/03_training_dynamics": ["part1/03_datasets_optimizers", "part1/neural_network_basics"],
    "part5/04_hyperparam_search": ["part5/03_training_dynamics"],
    "part5/05_dataset_toys": ["part1/machine_learning_basics"],
    "part5/data_training": ["part1/03_datasets_optimizers", "part5/03_training_dynamics"],
    "part5/case_studies": ["part5/data_training", "part5/04_hyperparam_search"],
    "part5/deployment_tools": ["part6/03_full_project", "part6/05_one_click_training"],
    "part5/quiz_system": ["part1/math_primer"],
    "part5/tuning_challenge": ["part5/03_training_dynamics", "part5/04_hyperparam_search"],
    "part6/01_unified_interface": ["part5/data_training"],
    "part6/02_modular_structure": ["part6/01_unified_interface"],
    "part6/03_full_project": ["part6/02_modular_structure"],
    "part6/04_plugin_system": ["part6/03_full_project"],
    "part6/05_one_click_training": ["part6/03_full_project", "part5/data_training"],
    "part6/06_streamlit_demo": ["part5/01_feature_visualization", "part4/01_attention_mechanism"],
    "part6/neural_network_playground": ["part1/neural_network_basics", "part6/06_streamlit_demo"],
    "part6/training_demo": ["part5/03_training_dynamics", "part5/02_gradient_monitor"],
    "part6/07_project_template": ["part6/03_full_project"],
    "part6/reinforcement_learning": ["part1/math_primer", "part5/05_dataset_toys"],
    "part6/learning_path": ["part1/math_primer"],
    "part6/glossary": ["part1/math_primer"],
    "part6/frontier": ["part4/transformer_models", "part6/paper_reading_lab"],
    "part6/paper_reading_lab": ["part1/neural_network_basics", "part5/05_dataset_toys"],
    "part7/networking": ["part6/learning_path"],
    "part7/database_sql": ["part6/learning_path"],
    "part7/data_structures": ["part1/math_primer"],
    "part7/operating_system": ["part7/data_structures"],
    "part7/interview_quiz": ["part7/networking", "part7/database_sql", "part7/data_structures", "part7/operating_system"],
}


THEORY_TO_PRACTICE: dict[str, str] = {
    "part1/01_tensors_gradients": "part5/02_gradient_monitor",
    "part1/02_activations_normalization": "part6/06_streamlit_demo",
    "part1/03_datasets_optimizers": "part6/training_demo",
    "part1/math_primer": "part1/01_tensors_gradients",
    "part1/machine_learning_basics": "part1/classical_ml",
    "part1/neural_network_basics": "part6/neural_network_playground",
    "part1/classical_ml": "part5/05_dataset_toys",
    "part2/01_convolution_visual": "part2/02_feature_maps",
    "part2/02_feature_maps": "part2/05_mnist_toy",
    "part2/03_classic_architectures": "part2/cnn_architectures",
    "part2/04_debug_panel": "part5/02_gradient_monitor",
    "part2/05_mnist_toy": "part6/training_demo",
    "part2/06_modern_architectures": "part2/advanced_cnn",
    "part2/07_advanced_convolution": "part2/advanced_cnn",
    "part2/08_visualization_gradcam": "part5/01_feature_visualization",
    "part2/09_transfer_learning": "part5/case_studies",
    "part2/cnn_architectures": "part6/neural_network_playground",
    "part2/advanced_cnn": "part6/neural_network_playground",
    "part3/01_rnn_intuition": "part3/03_sequence_toys",
    "part3/02_hidden_states": "part3/03_sequence_toys",
    "part3/03_sequence_toys": "part6/training_demo",
    "part3/04_hyperparam_rnn": "part5/04_hyperparam_search",
    "part3/05_seq2seq_attention": "part4/01_attention_mechanism",
    "part3/06_text_classification": "part5/data_training",
    "part3/07_advanced_training": "part5/02_gradient_monitor",
    "part3/08_debug_problems": "part5/02_gradient_monitor",
    "part3/sequence_models": "part3/03_sequence_toys",
    "part4/01_attention_mechanism": "part6/06_streamlit_demo",
    "part4/02_multihead_visual": "part4/transformer_models",
    "part4/03_encoder_decoder": "part4/04_minimal_transformer",
    "part4/04_minimal_transformer": "part6/neural_network_playground",
    "part4/05_flash_attention": "part5/03_training_dynamics",
    "part4/06_debug_problems": "part5/02_gradient_monitor",
    "part4/transformer_models": "part6/neural_network_playground",
    "part4/gan_ae": "part5/01_feature_visualization",
    "part4/gnn_intro": "part5/05_dataset_toys",
    "part5/01_feature_visualization": "part5/case_studies",
    "part5/02_gradient_monitor": "part6/training_demo",
    "part5/03_training_dynamics": "part6/training_demo",
    "part5/04_hyperparam_search": "part5/tuning_challenge",
    "part5/05_dataset_toys": "part5/data_training",
    "part5/data_training": "part5/case_studies",
    "part5/case_studies": "part6/03_full_project",
    "part5/deployment_tools": "part6/07_project_template",
    "part5/quiz_system": "part7/interview_quiz",
    "part5/tuning_challenge": "part5/04_hyperparam_search",
    "part6/01_unified_interface": "part6/03_full_project",
    "part6/02_modular_structure": "part6/03_full_project",
    "part6/03_full_project": "part6/05_one_click_training",
    "part6/04_plugin_system": "part6/07_project_template",
    "part6/05_one_click_training": "part6/training_demo",
    "part6/06_streamlit_demo": "part6/neural_network_playground",
    "part6/neural_network_playground": "part6/training_demo",
    "part6/training_demo": "part5/02_gradient_monitor",
    "part6/07_project_template": "part5/deployment_tools",
    "part6/reinforcement_learning": "part5/05_dataset_toys",
    "part6/learning_path": "part6/neural_network_playground",
    "part6/glossary": "part5/quiz_system",
    "part6/frontier": "part6/paper_reading_lab",
    "part6/paper_reading_lab": "part6/neural_network_playground",
    "part7/networking": "part7/interview_quiz",
    "part7/database_sql": "part7/interview_quiz",
    "part7/data_structures": "part7/interview_quiz",
    "part7/operating_system": "part7/interview_quiz",
    "part7/interview_quiz": "part7/interview_quiz",
}


NEXT_STEPS: dict[str, list[str]] = {
    "part1/math_primer": ["part1/01_tensors_gradients", "part1/machine_learning_basics"],
    "part1/neural_network_basics": ["part2/01_convolution_visual", "part3/01_rnn_intuition", "part6/neural_network_playground"],
    "part2/advanced_cnn": ["part2/08_visualization_gradcam", "part2/09_transfer_learning", "part5/01_feature_visualization"],
    "part3/sequence_models": ["part4/01_attention_mechanism", "part3/06_text_classification", "part3/07_advanced_training"],
    "part4/transformer_models": ["part4/05_flash_attention", "part4/06_debug_problems", "part6/neural_network_playground"],
    "part5/03_training_dynamics": ["part5/02_gradient_monitor", "part5/04_hyperparam_search", "part6/training_demo"],
    "part6/neural_network_playground": ["part6/training_demo", "part6/03_full_project", "part5/tuning_challenge"],
    "part7/interview_quiz": ["part6/learning_path", "part5/quiz_system"],
}


def _canonical(key: str) -> str:
    return ALIASES.get(key, key)


def canonical_node_keys() -> list[str]:
    return [seed.key for seed in MODULE_SEEDS]


def _default_prerequisites(index: int) -> list[str]:
    if index == 0:
        return []
    return [MODULE_SEEDS[index - 1].key]


def _default_next_steps(index: int) -> list[str]:
    return [seed.key for seed in MODULE_SEEDS[index + 1 : index + 3]]


def _mastery(seed: ModuleSeed) -> str:
    tag_text = "、".join(seed.tags[:3])
    return f"能说清“{seed.title}”解决的问题、核心机制和适用边界；能在页面可视化或调参结果中解释至少 2 个关键变化，并把它们和 {tag_text} 联系起来。"


def _practice_target(seed: ModuleSeed, practice_title: str) -> str:
    return f"完成“{practice_title}”中的一次动手实验：记录输入、关键参数、输出变化和一句结论，再回到“{seed.title}”解释为什么会这样。"


def _related_from_tags(seed: ModuleSeed, index: int) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    tag_set = set(seed.tags)
    for other_index, other in enumerate(MODULE_SEEDS):
        if other.key == seed.key:
            continue
        overlap = len(tag_set & set(other.tags))
        same_part = seed.key.split("/", 1)[0] == other.key.split("/", 1)[0]
        score = overlap * 10 + (3 if same_part else 0) - abs(index - other_index)
        if score > 0:
            scored.append((score, -abs(index - other_index), other.key))
    scored.sort(reverse=True)
    return [key for _, _, key in scored[:4]]


def _build_graph() -> dict[str, KnowledgeNode]:
    by_key = {seed.key: seed for seed in MODULE_SEEDS}
    nodes: dict[str, KnowledgeNode] = {}
    for index, seed in enumerate(MODULE_SEEDS):
        practice_route = _canonical(THEORY_TO_PRACTICE.get(seed.key, seed.key))
        practice_seed = by_key.get(practice_route, seed)
        related = _related_from_tags(seed, index)
        prerequisites = [_canonical(key) for key in PREREQUISITES.get(seed.key, _default_prerequisites(index))]
        next_steps = [_canonical(key) for key in NEXT_STEPS.get(seed.key, _default_next_steps(index))]
        nodes[seed.key] = KnowledgeNode(
            name=seed.key,
            title=seed.title,
            description=seed.description,
            route=seed.route,
            prerequisites=[key for key in prerequisites if key in by_key],
            related=[key for key in related if key in by_key],
            next_steps=[key for key in next_steps if key in by_key],
            mastery_criteria=_mastery(seed),
            practice_target=_practice_target(seed, practice_seed.title),
            practice_route=practice_route,
            tags=list(seed.tags),
            difficulty=seed.difficulty,
        )

    for alias, canonical in ALIASES.items():
        if canonical in nodes:
            nodes[alias] = nodes[canonical]
    return nodes


KNOWLEDGE_GRAPH: dict[str, KnowledgeNode] = _build_graph()


def get_node(key: str) -> KnowledgeNode | None:
    return KNOWLEDGE_GRAPH.get(key) or KNOWLEDGE_GRAPH.get(_canonical(key))


def _nodes(keys: Iterable[str]) -> list[KnowledgeNode]:
    return [node for key in keys if (node := get_node(key)) is not None]


def get_prerequisites(key: str) -> list[KnowledgeNode]:
    node = get_node(key)
    return _nodes(node.prerequisites) if node else []


def get_related(key: str) -> list[KnowledgeNode]:
    node = get_node(key)
    return _nodes(node.related) if node else []


def get_next_steps(key: str) -> list[KnowledgeNode]:
    node = get_node(key)
    return _nodes(node.next_steps) if node else []


def _module_url(key: str) -> str:
    node = get_node(key)
    route = node.route if node else _canonical(key)
    return f"/?module={quote(route, safe='')}"


def practice_url(key: str) -> str:
    node = get_node(key)
    if node is None:
        return _module_url(key)
    return _module_url(node.practice_route or node.name)


def graph_summary() -> dict[str, int]:
    counts = Counter(get_node(key).difficulty for key in canonical_node_keys() if get_node(key) is not None)
    counts["总节点"] = len(canonical_node_keys())
    return dict(counts)


def _render_node_link(node: KnowledgeNode) -> None:
    st = __import__("streamlit")
    st.markdown(f"**{node.title}**")
    st.caption(f"{node.description}｜难度：{node.difficulty}")
    cols = st.columns(2)
    cols[0].link_button("打开理论", _module_url(node.name), width="stretch")
    cols[1].link_button("去实战", practice_url(node.name), width="stretch")


def render知识图谱导航(current_key: str) -> None:
    """在 Streamlit 页面底部渲染前置知识、相关知识、下一步推荐和实战入口。"""

    try:
        st = __import__("streamlit")
        node = get_node(current_key)
        if node is None:
            st.info("暂时没有找到当前知识点的图谱信息。")
            return

        st.divider()
        st.subheader("知识图谱导航")
        st.markdown(f"**当前位置：{node.title}**")
        st.caption(node.description)

        info_cols = st.columns([0.34, 0.33, 0.33])
        info_cols[0].markdown(f"**掌握标准**  \n{node.mastery_criteria}")
        info_cols[1].markdown(f"**去实战目标**  \n{node.practice_target}")
        info_cols[2].link_button("进入对应实战页", practice_url(node.name), width="stretch")

        prereq_col, related_col, next_col = st.columns(3)
        with prereq_col:
            st.markdown("**前置知识**")
            prerequisites = get_prerequisites(node.name)
            if prerequisites:
                for item in prerequisites[:4]:
                    _render_node_link(item)
            else:
                st.info("这是推荐起点之一，可以直接开始。")

        with related_col:
            st.markdown("**相关知识**")
            related = get_related(node.name)
            if related:
                for item in related[:4]:
                    _render_node_link(item)
            else:
                st.info("暂时没有强相关节点，可以先完成当前页。")

        with next_col:
            st.markdown("**后续推荐**")
            next_steps = get_next_steps(node.name)
            if next_steps:
                for item in next_steps[:4]:
                    _render_node_link(item)
            else:
                st.info("已经到达当前路径末端，建议进入项目或面试复盘。")
    except Exception as error:
        st = __import__("streamlit")
        st.warning("知识图谱导航暂时无法显示，请继续阅读正文。")
        with st.expander("查看组件错误详情", expanded=False):
            st.code(str(error), language="text")
