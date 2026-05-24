"""
Deep learning glossary and search page.

Run:
    streamlit run part6_universal_framework/glossary.py
or:
    python main.py part6/glossary
"""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="深度学习术语词典",
    layout="wide",
    initial_sidebar_state="expanded",
)


INK = "#172026"
MUTED = "#596772"
LINE = "#d8dee3"
TEAL = "#0f8b8d"
ROSE = "#bf3f5b"
AMBER = "#c4871f"
BLUE = "#3268a8"
GREEN = "#3f7d58"
VIOLET = "#7353ba"

CATEGORY_COLORS = {
    "基础概念": TEAL,
    "数学基础": BLUE,
    "优化训练": AMBER,
    "神经网络": GREEN,
    "数据与评估": ROSE,
    "CNN视觉": "#3f6f9f",
    "序列模型": "#8b5a2b",
    "Transformer与NLP": VIOLET,
    "生成模型": "#a64d79",
    "图学习与强化学习": "#4f7c45",
    "工程部署": "#607d8b",
    "前沿与安全": "#7a4c9a",
}

MODULES = {
    "part1/machine_learning_basics": "Part 1 机器学习基础",
    "part1/math_primer": "Part 1 数学基础速查",
    "part1/neural_network_basics": "Part 1 神经网络基础",
    "part1/classical_ml": "Part 1 经典机器学习",
    "part1/03_datasets_optimizers": "Part 1 数据集与优化器",
    "part2/01_convolution_visual": "Part 2 卷积可视化",
    "part2/02_feature_maps": "Part 2 特征图",
    "part2/03_classic_architectures": "Part 2 经典 CNN 架构",
    "part2/06_modern_architectures": "Part 2 现代 CNN 架构",
    "part2/08_visualization_gradcam": "Part 2 可视化与 Grad-CAM",
    "part2/09_transfer_learning": "Part 2 迁移学习",
    "part3/01_rnn_intuition": "Part 3 RNN 直觉",
    "part3/02_hidden_states": "Part 3 隐状态",
    "part3/05_seq2seq_attention": "Part 3 Seq2Seq 与注意力",
    "part3/06_text_classification": "Part 3 文本分类",
    "part3/07_advanced_training": "Part 3 序列高级训练",
    "part4/01_attention_mechanism": "Part 4 注意力机制",
    "part4/02_multihead_visual": "Part 4 多头注意力",
    "part4/03_encoder_decoder": "Part 4 编码器-解码器",
    "part4/04_minimal_transformer": "Part 4 最小 Transformer",
    "part4/05_flash_attention": "Part 4 Flash Attention",
    "part4/gan_ae": "Part 4 GAN 与自编码器",
    "part4/gnn_intro": "Part 4 图神经网络",
    "part4/transformer_models": "Part 4 Transformer 架构",
    "part5/data_training": "Part 5 数据与训练",
    "part5/03_training_dynamics": "Part 5 训练动态",
    "part5/04_hyperparam_search": "Part 5 超参数搜索",
    "part5/deployment_tools": "Part 5 部署工具",
    "part5/quiz_system": "Part 5 练习题与测验",
    "part6/06_streamlit_demo": "Part 6 可视化实验台",
    "part6/learning_path": "Part 6 学习路径推荐",
    "part6/frontier": "Part 6 前沿方向",
    "part6/reinforcement_learning": "Part 6 强化学习入门",
}

CATEGORY_GUIDES = {
    "基础概念": "它通常决定问题如何被形式化、模型如何接收信号以及结果如何被解释。",
    "数学基础": "它提供了深度学习中表示、求导、概率推断或数值计算的语言。",
    "优化训练": "它直接影响参数更新、收敛速度、训练稳定性和最终泛化能力。",
    "神经网络": "它描述了网络组件、信息流或参数化函数的基本构造方式。",
    "数据与评估": "它帮助判断数据是否可靠、指标是否合适以及模型是否真的学到了可泛化规律。",
    "CNN视觉": "它服务于图像、视频或空间局部结构建模，是视觉网络设计的核心积木。",
    "序列模型": "它用于处理文本、语音、时间序列等有顺序依赖的数据。",
    "Transformer与NLP": "它围绕注意力、上下文建模、语言表示和大模型架构展开。",
    "生成模型": "它关注从分布中采样、重构、去噪或生成新样本的机制。",
    "图学习与强化学习": "它用于结构化关系建模、决策过程、奖励优化或智能体学习。",
    "工程部署": "它把实验模型变成可复现、可监控、可扩展、可上线的系统。",
    "前沿与安全": "它连接当前研究趋势、模型能力边界、风险治理和可信 AI。",
}


@dataclass(frozen=True)
class GlossaryTerm:
    english: str
    chinese: str
    category: str
    definition: str
    detail: str
    related: tuple[str, ...]
    modules: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return normalize_key(self.english)

    @property
    def initial(self) -> str:
        first = self.english[:1].upper()
        return first if first.isalpha() else "#"

    @property
    def module_labels(self) -> tuple[str, ...]:
        return tuple(MODULES.get(module, module) for module in self.modules)

    @property
    def search_text(self) -> str:
        fields = (
            self.english,
            self.chinese,
            self.category,
            self.definition,
            self.detail,
            " ".join(self.aliases),
            " ".join(self.related),
            " ".join(self.module_labels),
        )
        return " ".join(fields).lower()


RawTerm = tuple[str, str, str, str, str, tuple[str, ...], tuple[str, ...]]


RAW_TERMS: tuple[RawTerm, ...] = (
    ("Accuracy", "准确率", "数据与评估", "预测正确样本数占总样本数的比例。", "part5/data_training", ("Precision", "Recall", "Confusion Matrix"), ("acc",)),
    ("Activation Function", "激活函数", "神经网络", "为神经元输出引入非线性的函数。", "part1/neural_network_basics", ("ReLU", "Sigmoid", "Tanh"), ("activation",)),
    ("Adam", "Adam 优化器", "优化训练", "结合动量和自适应学习率的一阶优化算法。", "part1/03_datasets_optimizers", ("Momentum", "Learning Rate", "Weight Decay"), ("Adaptive Moment Estimation",)),
    ("Adagrad", "Adagrad 优化器", "优化训练", "按历史梯度平方和缩放每个参数学习率的优化器。", "part1/03_datasets_optimizers", ("Learning Rate", "RMSProp", "Adam"), ()),
    ("Attention", "注意力", "Transformer与NLP", "根据相关性为不同位置或特征分配权重的机制。", "part4/01_attention_mechanism", ("Self-Attention", "Query", "Key"), ("attention mechanism",)),
    ("Autoencoder", "自编码器", "生成模型", "通过编码再解码来学习压缩表示和重构输入的网络。", "part4/gan_ae", ("Latent Space", "Decoder", "Variational Autoencoder"), ("AE",)),
    ("Autograd", "自动求导", "优化训练", "自动记录计算图并用链式法则计算梯度的机制。", "part1/01_tensors_gradients", ("Computational Graph", "Backpropagation", "Gradient"), ("automatic differentiation",)),
    ("AUC", "曲线下面积", "数据与评估", "ROC 或 PR 曲线下方的面积，用于衡量排序质量。", "part5/data_training", ("ROC Curve", "Precision", "Recall"), ("Area Under Curve",)),
    ("Average Pooling", "平均池化", "CNN视觉", "用局部窗口平均值降低特征图尺寸。", "part2/02_feature_maps", ("Pooling", "Max Pooling", "Feature Map"), ()),
    ("Backbone", "主干网络", "CNN视觉", "用于提取通用特征的核心网络部分。", "part2/06_modern_architectures", ("Feature Extractor", "Transfer Learning", "Fine-tuning"), ()),
    ("Backpropagation", "反向传播", "优化训练", "用链式法则从损失反向计算各层梯度的算法。", "part1/neural_network_basics", ("Gradient", "Chain Rule", "Autograd"), ("BP",)),
    ("Bagging", "袋装法", "基础概念", "在多个自助采样数据集上训练模型并聚合结果的集成方法。", "part1/classical_ml", ("Bootstrap", "Ensemble", "Random Forest"), ()),
    ("Batch", "批次", "数据与评估", "一次前向和反向传播共同处理的一组样本。", "part5/data_training", ("Mini-batch", "Batch Size", "Epoch"), ()),
    ("Batch Normalization", "批归一化", "神经网络", "在小批次维度标准化中间激活以稳定训练的层。", "part1/02_activations_normalization", ("Layer Normalization", "Internal Covariate Shift", "Regularization"), ("BatchNorm",)),
    ("Batch Size", "批大小", "优化训练", "每次参数更新使用的样本数量。", "part5/data_training", ("Batch", "Mini-batch", "Gradient Accumulation"), ()),
    ("Beam Search", "束搜索", "Transformer与NLP", "在生成序列时保留若干高分候选路径的解码策略。", "part3/05_seq2seq_attention", ("Greedy Decoding", "Sampling", "Decoder"), ()),
    ("Bias", "偏置", "神经网络", "在线性变换中独立于输入的可学习平移参数。", "part1/neural_network_basics", ("Weight", "Linear Layer", "Affine Transform"), ()),
    ("Bias-Variance Tradeoff", "偏差-方差权衡", "基础概念", "模型欠拟合与对训练数据波动敏感之间的平衡。", "part1/machine_learning_basics", ("Underfitting", "Overfitting", "Regularization"), ()),
    ("Bilinear Interpolation", "双线性插值", "CNN视觉", "根据周围四个像素线性组合估计新位置取值的方法。", "part2/07_advanced_convolution", ("Upsampling", "Transposed Convolution", "Feature Map"), ()),
    ("Binary Cross Entropy", "二元交叉熵", "优化训练", "二分类任务中衡量预测概率和真实标签差异的损失。", "part1/machine_learning_basics", ("Cross Entropy", "Log Loss", "Sigmoid"), ("BCE",)),
    ("Bottleneck", "瓶颈层", "CNN视觉", "通过压缩再扩展通道降低计算量或控制信息流的结构。", "part2/06_modern_architectures", ("Residual Connection", "Inverted Bottleneck", "1x1 Convolution"), ()),
    ("Bounding Box", "边界框", "CNN视觉", "用矩形坐标表示目标在图像中的位置。", "part2/09_transfer_learning", ("Object Detection", "IoU", "Anchor Box"), ("bbox",)),
    ("BPE", "字节对编码", "Transformer与NLP", "迭代合并高频符号对以构造子词词表的分词方法。", "part4/transformer_models", ("Tokenization", "Vocabulary", "Subword"), ("Byte Pair Encoding",)),
    ("Broadcasting", "广播机制", "数学基础", "在张量运算中自动扩展兼容维度的规则。", "part1/01_tensors_gradients", ("Tensor", "Shape", "Vectorization"), ()),
    ("Calibration", "校准", "数据与评估", "预测置信度与真实正确率的一致程度。", "part5/data_training", ("Expected Calibration Error", "Reliability Diagram", "Temperature Scaling"), ()),
    ("Catastrophic Forgetting", "灾难性遗忘", "前沿与安全", "模型学习新任务时快速丢失旧任务能力的现象。", "part6/frontier", ("Continual Learning", "Fine-tuning", "Regularization"), ()),
    ("Causal Mask", "因果掩码", "Transformer与NLP", "阻止当前位置看到未来 token 的注意力掩码。", "part4/04_minimal_transformer", ("Mask", "Decoder-only", "Autoregressive Model"), ()),
    ("Chain Rule", "链式法则", "数学基础", "复合函数求导时逐层相乘导数的规则。", "part1/math_primer", ("Backpropagation", "Gradient", "Jacobian"), ()),
    ("Checkpoint", "检查点", "工程部署", "保存模型权重、优化器状态和训练进度的文件。", "part5/data_training", ("State Dict", "Resume Training", "Early Stopping"), ()),
    ("Classification", "分类", "基础概念", "将输入样本映射到离散类别的监督学习任务。", "part1/machine_learning_basics", ("Regression", "Softmax", "Cross Entropy"), ()),
    ("CLIP", "图文对比预训练", "前沿与安全", "把图像和文本对齐到同一嵌入空间的对比学习模型。", "part6/frontier", ("Contrastive Learning", "Embedding", "Multimodal Model"), ()),
    ("Clipping", "裁剪", "优化训练", "限制梯度或数值范围以避免不稳定更新。", "part3/07_advanced_training", ("Gradient Clipping", "Exploding Gradient", "Training Stability"), ()),
    ("CNN", "卷积神经网络", "CNN视觉", "通过卷积、非线性和池化处理网格数据的神经网络。", "part2/01_convolution_visual", ("Convolution", "Pooling", "Feature Map"), ("ConvNet",)),
    ("Computational Graph", "计算图", "优化训练", "把张量运算表示为节点和边的有向图。", "part1/01_tensors_gradients", ("Autograd", "Backpropagation", "Tensor"), ()),
    ("Confusion Matrix", "混淆矩阵", "数据与评估", "统计真实类别和预测类别组合次数的表格。", "part5/data_training", ("Accuracy", "Precision", "Recall"), ()),
    ("Contrastive Learning", "对比学习", "前沿与安全", "拉近正样本表示并推远负样本表示的自监督方法。", "part6/frontier", ("CLIP", "Embedding", "Self-supervised Learning"), ()),
    ("Convolution", "卷积", "CNN视觉", "用可学习卷积核在局部窗口上滑动提取特征。", "part2/01_convolution_visual", ("Kernel", "Stride", "Padding"), ("conv",)),
    ("Convolution Kernel", "卷积核", "CNN视觉", "在输入局部区域上共享使用的小型权重矩阵。", "part2/01_convolution_visual", ("Filter", "Convolution", "Receptive Field"), ("filter",)),
    ("Cosine Similarity", "余弦相似度", "数学基础", "用向量夹角余弦衡量方向相似性的指标。", "part1/math_primer", ("Embedding", "Dot Product", "Vector"), ()),
    ("Cross Entropy", "交叉熵", "优化训练", "衡量两个概率分布差异的常用分类损失。", "part1/machine_learning_basics", ("Softmax", "Negative Log Likelihood", "KL Divergence"), ()),
    ("Cross Validation", "交叉验证", "数据与评估", "多次划分训练和验证集以估计泛化表现的方法。", "part5/data_training", ("Validation Set", "K-fold", "Generalization"), ("CV",)),
    ("CUDA", "CUDA", "工程部署", "NVIDIA GPU 上的并行计算平台和编程接口。", "part5/deployment_tools", ("GPU", "Tensor Core", "Mixed Precision"), ()),
    ("Curriculum Learning", "课程学习", "优化训练", "按由易到难的顺序组织训练样本或任务。", "part5/data_training", ("Sampling Strategy", "Training Dynamics", "Fine-tuning"), ()),
    ("Data Augmentation", "数据增强", "数据与评估", "通过变换样本扩充训练数据并提升泛化的技术。", "part5/data_training", ("Regularization", "Transform", "Overfitting"), ()),
    ("Data Leakage", "数据泄漏", "数据与评估", "训练过程意外使用测试或未来信息导致评估虚高。", "part5/data_training", ("Train Test Split", "Validation Set", "Generalization"), ()),
    ("Data Loader", "数据加载器", "工程部署", "按批次读取、打乱、并行预处理数据的组件。", "part5/data_training", ("Dataset", "Batch", "Shuffle"), ()),
    ("Dataset", "数据集", "数据与评估", "用于训练、验证或测试的一组样本和标签。", "part5/data_training", ("Data Loader", "Train Set", "Test Set"), ()),
    ("Decoder", "解码器", "Transformer与NLP", "把潜在表示或上下文转换为目标输出的模块。", "part4/03_encoder_decoder", ("Encoder", "Seq2Seq", "Autoregressive Model"), ()),
    ("Decoder-only Transformer", "仅解码器 Transformer", "Transformer与NLP", "只使用因果自注意力解码器堆叠的自回归架构。", "part4/transformer_models", ("GPT", "Causal Mask", "Autoregressive Model"), ()),
    ("Decision Boundary", "决策边界", "基础概念", "模型在特征空间中划分类别的分界面。", "part1/machine_learning_basics", ("Classifier", "Margin", "Overfitting"), ()),
    ("Deep Learning", "深度学习", "基础概念", "使用多层可学习表示从数据中学习复杂函数的机器学习方法。", "part1/neural_network_basics", ("Neural Network", "Representation Learning", "Backpropagation"), ()),
    ("Depthwise Separable Convolution", "深度可分离卷积", "CNN视觉", "先逐通道卷积再逐点混合通道以减少计算量的卷积。", "part2/07_advanced_convolution", ("Depthwise Convolution", "Pointwise Convolution", "MobileNet"), ()),
    ("Determinism", "确定性", "工程部署", "相同输入和配置下得到可重复结果的性质。", "part5/data_training", ("Random Seed", "Reproducibility", "Checkpoint"), ()),
    ("Diffusion Model", "扩散模型", "生成模型", "学习从噪声逐步反向去噪生成样本的模型。", "part6/frontier", ("Denoising", "Score Matching", "Generative Model"), ()),
    ("Dilation", "空洞率", "CNN视觉", "在卷积核元素之间插入间隔以扩大感受野的参数。", "part2/07_advanced_convolution", ("Dilated Convolution", "Receptive Field", "Kernel"), ()),
    ("Dilated Convolution", "空洞卷积", "CNN视觉", "使用空洞率扩大视野但不显著增加参数的卷积。", "part2/07_advanced_convolution", ("Dilation", "Convolution", "Receptive Field"), ()),
    ("Dimensionality Reduction", "降维", "数学基础", "把高维数据映射到低维空间并尽量保留重要结构。", "part1/classical_ml", ("PCA", "Embedding", "Manifold"), ()),
    ("Distributed Training", "分布式训练", "工程部署", "用多块 GPU 或多台机器协同训练模型。", "part5/deployment_tools", ("Data Parallelism", "Model Parallelism", "Gradient Synchronization"), ()),
    ("Domain Adaptation", "领域适配", "前沿与安全", "让模型从源领域迁移到分布不同的目标领域。", "part2/09_transfer_learning", ("Transfer Learning", "Fine-tuning", "Domain Shift"), ()),
    ("Domain Shift", "领域偏移", "数据与评估", "训练数据和部署数据分布不一致的现象。", "part5/data_training", ("Out-of-distribution", "Generalization", "Data Drift"), ()),
    ("Dot Product", "点积", "数学基础", "把两个向量对应元素相乘再求和的运算。", "part1/math_primer", ("Vector", "Cosine Similarity", "Attention"), ()),
    ("Dropout", "随机失活", "神经网络", "训练时随机置零部分激活以降低过拟合的正则化方法。", "part1/02_activations_normalization", ("Regularization", "Ensemble", "Overfitting"), ()),
    ("Early Stopping", "早停", "优化训练", "当验证集性能不再提升时提前结束训练。", "part5/data_training", ("Validation Loss", "Overfitting", "Checkpoint"), ()),
    ("Embedding", "嵌入", "Transformer与NLP", "把离散对象映射为连续向量表示。", "part4/transformer_models", ("Token Embedding", "Representation Learning", "Vector"), ()),
    ("Encoder", "编码器", "Transformer与NLP", "把输入转换为上下文表示或潜在表示的模块。", "part4/03_encoder_decoder", ("Decoder", "Seq2Seq", "Representation"), ()),
    ("Encoder-only Transformer", "仅编码器 Transformer", "Transformer与NLP", "只使用双向自注意力编码器堆叠的表征模型。", "part4/transformer_models", ("BERT", "Masked Language Modeling", "Encoder"), ()),
    ("Ensemble", "集成学习", "基础概念", "组合多个模型以获得更稳健预测的方法。", "part1/classical_ml", ("Bagging", "Boosting", "Voting"), ()),
    ("Epoch", "轮次", "优化训练", "模型完整遍历一次训练集的过程。", "part5/data_training", ("Batch", "Iteration", "Training Loop"), ()),
    ("Evaluation Metric", "评估指标", "数据与评估", "用来衡量模型性能的量化标准。", "part5/data_training", ("Accuracy", "F1 Score", "AUC"), ()),
    ("Exploding Gradient", "梯度爆炸", "优化训练", "反向传播中梯度变得过大导致训练不稳定的现象。", "part3/07_advanced_training", ("Gradient Clipping", "Vanishing Gradient", "RNN"), ()),
    ("F1 Score", "F1 值", "数据与评估", "精确率和召回率的调和平均。", "part5/data_training", ("Precision", "Recall", "Confusion Matrix"), ()),
    ("False Negative", "假阴性", "数据与评估", "真实为正类但模型预测为负类的样本。", "part5/data_training", ("Recall", "False Positive", "Confusion Matrix"), ("FN",)),
    ("False Positive", "假阳性", "数据与评估", "真实为负类但模型预测为正类的样本。", "part5/data_training", ("Precision", "False Negative", "Confusion Matrix"), ("FP",)),
    ("Feature", "特征", "基础概念", "输入数据中供模型学习和决策使用的信息维度。", "part1/machine_learning_basics", ("Feature Engineering", "Representation", "Embedding"), ()),
    ("Feature Engineering", "特征工程", "基础概念", "人工构造、选择和转换输入特征以改善模型表现的过程。", "part1/classical_ml", ("Feature", "Normalization", "Classical Machine Learning"), ()),
    ("Feature Map", "特征图", "CNN视觉", "卷积层输出的空间激活图。", "part2/02_feature_maps", ("Convolution", "Channel", "Activation"), ()),
    ("Few-shot Learning", "少样本学习", "前沿与安全", "模型用极少示例适应新任务的能力或方法。", "part6/frontier", ("Zero-shot Learning", "In-context Learning", "Transfer Learning"), ()),
    ("Fine-tuning", "微调", "优化训练", "在预训练模型基础上用目标任务数据继续训练。", "part2/09_transfer_learning", ("Pretraining", "Transfer Learning", "Learning Rate"), ()),
    ("Flash Attention", "Flash Attention", "Transformer与NLP", "通过重排注意力计算和内存访问提升效率的算法。", "part4/05_flash_attention", ("Attention", "Memory Bandwidth", "Scaled Dot-Product Attention"), ()),
    ("FLOPs", "浮点运算量", "工程部署", "衡量模型计算成本的浮点运算次数。", "part5/deployment_tools", ("Latency", "Throughput", "Model Compression"), ()),
    ("Foundation Model", "基础模型", "前沿与安全", "在大规模数据上训练并可迁移到多种任务的大模型。", "part6/frontier", ("Pretraining", "Fine-tuning", "LLM"), ()),
    ("Fully Connected Layer", "全连接层", "神经网络", "每个输出单元连接所有输入单元的线性层。", "part1/neural_network_basics", ("Linear Layer", "Weight", "Bias"), ("Dense Layer",)),
    ("GAN", "生成对抗网络", "生成模型", "由生成器和判别器相互博弈学习数据分布的模型。", "part4/gan_ae", ("Generator", "Discriminator", "Adversarial Loss"), ("Generative Adversarial Network",)),
    ("Gated Recurrent Unit", "门控循环单元", "序列模型", "用更新门和重置门缓解长期依赖问题的 RNN 变体。", "part3/01_rnn_intuition", ("RNN", "LSTM", "Hidden State"), ("GRU",)),
    ("GELU", "GELU 激活", "神经网络", "常用于 Transformer 的平滑非线性激活函数。", "part4/04_minimal_transformer", ("ReLU", "Activation Function", "Feed Forward Network"), ()),
    ("Generalization", "泛化", "基础概念", "模型在未见数据上保持有效表现的能力。", "part1/machine_learning_basics", ("Overfitting", "Validation Set", "Regularization"), ()),
    ("Generative Model", "生成模型", "生成模型", "学习数据分布并能产生新样本的模型类别。", "part4/gan_ae", ("GAN", "VAE", "Diffusion Model"), ()),
    ("Generator", "生成器", "生成模型", "在 GAN 中把噪声或条件输入映射为生成样本的网络。", "part4/gan_ae", ("GAN", "Discriminator", "Latent Vector"), ()),
    ("Global Average Pooling", "全局平均池化", "CNN视觉", "对每个通道的整张特征图取平均形成向量。", "part2/06_modern_architectures", ("Average Pooling", "Feature Map", "Classifier Head"), ("GAP",)),
    ("Gradient", "梯度", "数学基础", "函数对参数的偏导数组成的向量，指示最陡上升方向。", "part1/math_primer", ("Derivative", "Backpropagation", "Gradient Descent"), ()),
    ("Gradient Accumulation", "梯度累积", "优化训练", "跨多个小批次累加梯度后再更新参数。", "part5/data_training", ("Batch Size", "Distributed Training", "Optimizer Step"), ()),
    ("Gradient Clipping", "梯度裁剪", "优化训练", "把梯度范数限制在阈值内以避免爆炸。", "part3/07_advanced_training", ("Exploding Gradient", "RNN", "Training Stability"), ()),
    ("Gradient Descent", "梯度下降", "优化训练", "沿负梯度方向迭代更新参数以降低损失的方法。", "part1/math_primer", ("Learning Rate", "SGD", "Loss Function"), ()),
    ("Gradient Flow", "梯度流", "优化训练", "梯度在网络各层传播时的大小和稳定性。", "part5/03_training_dynamics", ("Vanishing Gradient", "Exploding Gradient", "Residual Connection"), ()),
    ("Graph Neural Network", "图神经网络", "图学习与强化学习", "在节点和边组成的图结构上进行消息传递学习的网络。", "part4/gnn_intro", ("Message Passing", "Node Embedding", "Edge"), ("GNN",)),
    ("Greedy Decoding", "贪心解码", "Transformer与NLP", "每一步选择当前概率最高 token 的生成策略。", "part3/05_seq2seq_attention", ("Beam Search", "Sampling", "Decoder"), ()),
    ("Grid Search", "网格搜索", "优化训练", "在预设超参数组合上逐一训练和评估的方法。", "part5/04_hyperparam_search", ("Hyperparameter", "Random Search", "Validation Set"), ()),
    ("Ground Truth", "真实标签", "数据与评估", "用于监督训练或评估的参考答案。", "part5/data_training", ("Label", "Annotation", "Loss Function"), ()),
    ("Group Normalization", "组归一化", "神经网络", "按通道分组归一化激活，减少对批大小的依赖。", "part1/02_activations_normalization", ("Batch Normalization", "Layer Normalization", "Normalization"), ("GroupNorm",)),
    ("Hallucination", "幻觉", "前沿与安全", "生成模型输出看似合理但事实错误或无依据内容的现象。", "part6/frontier", ("LLM", "Alignment", "Evaluation"), ()),
    ("Hidden Layer", "隐藏层", "神经网络", "位于输入层和输出层之间的可学习变换层。", "part1/neural_network_basics", ("Neural Network", "Activation Function", "Representation"), ()),
    ("Hidden State", "隐状态", "序列模型", "循环模型中携带历史上下文的状态向量。", "part3/02_hidden_states", ("RNN", "LSTM", "GRU"), ()),
    ("Huber Loss", "Huber 损失", "优化训练", "结合平方误差和绝对误差、对异常值更稳健的损失。", "part1/machine_learning_basics", ("MSE", "MAE", "Regression"), ()),
    ("Hyperparameter", "超参数", "优化训练", "训练前设定、通常不由梯度直接学习的配置。", "part5/04_hyperparam_search", ("Learning Rate", "Batch Size", "Grid Search"), ()),
    ("Image Classification", "图像分类", "CNN视觉", "为整张图像预测类别标签的视觉任务。", "part2/05_mnist_toy", ("CNN", "Softmax", "Transfer Learning"), ()),
    ("Image Segmentation", "图像分割", "CNN视觉", "为图像中的每个像素或区域预测类别的任务。", "part2/09_transfer_learning", ("Semantic Segmentation", "U-Net", "IoU"), ()),
    ("Imbalanced Dataset", "类别不平衡数据集", "数据与评估", "不同类别样本数量差异很大的数据集。", "part5/data_training", ("Class Weight", "F1 Score", "Sampling Strategy"), ()),
    ("In-context Learning", "上下文学习", "前沿与安全", "模型根据提示中的示例临时适应任务的能力。", "part6/frontier", ("Few-shot Learning", "Prompt", "LLM"), ("ICL",)),
    ("Inference", "推理", "工程部署", "用训练好的模型对新输入生成预测的过程。", "part5/deployment_tools", ("Latency", "Throughput", "Model Serving"), ()),
    ("Instance Normalization", "实例归一化", "神经网络", "对单个样本的每个通道独立归一化的层。", "part1/02_activations_normalization", ("Batch Normalization", "Layer Normalization", "Style Transfer"), ("InstanceNorm",)),
    ("Intersection over Union", "交并比", "数据与评估", "预测区域和真实区域交集面积与并集面积之比。", "part2/09_transfer_learning", ("Bounding Box", "Object Detection", "Segmentation"), ("IoU",)),
    ("Iteration", "迭代", "优化训练", "一次批次计算和参数更新步骤。", "part5/data_training", ("Batch", "Epoch", "Optimizer Step"), ()),
    ("Jacobian", "雅可比矩阵", "数学基础", "向量函数对向量输入的一阶偏导矩阵。", "part1/math_primer", ("Gradient", "Chain Rule", "Hessian"), ()),
    ("Key", "键向量", "Transformer与NLP", "注意力中与查询比较以计算权重的向量。", "part4/01_attention_mechanism", ("Query", "Value", "Attention"), ()),
    ("KL Divergence", "KL 散度", "数学基础", "衡量一个概率分布相对另一个分布的信息差异。", "part1/math_primer", ("Cross Entropy", "Probability Distribution", "VAE"), ("Kullback-Leibler Divergence",)),
    ("Label", "标签", "数据与评估", "监督学习中样本对应的目标答案。", "part5/data_training", ("Ground Truth", "Annotation", "Classification"), ()),
    ("Label Smoothing", "标签平滑", "优化训练", "把硬标签替换为带少量不确定性的软标签以改善泛化。", "part5/data_training", ("Cross Entropy", "Regularization", "Calibration"), ()),
    ("Latent Space", "潜在空间", "生成模型", "模型学习到的低维或抽象表示空间。", "part4/gan_ae", ("Embedding", "Autoencoder", "VAE"), ()),
    ("Latent Vector", "潜向量", "生成模型", "潜在空间中的一个向量，常作为生成过程输入。", "part4/gan_ae", ("Latent Space", "Generator", "Decoder"), ()),
    ("Layer Normalization", "层归一化", "神经网络", "在单个样本的特征维度上归一化激活。", "part4/04_minimal_transformer", ("Batch Normalization", "Transformer", "Residual Connection"), ("LayerNorm",)),
    ("Learning Rate", "学习率", "优化训练", "控制每次参数更新步长的超参数。", "part1/03_datasets_optimizers", ("Gradient Descent", "Scheduler", "Adam"), ("lr",)),
    ("Learning Rate Scheduler", "学习率调度器", "优化训练", "按训练进度动态调整学习率的策略。", "part5/data_training", ("Learning Rate", "Warmup", "Cosine Annealing"), ("scheduler",)),
    ("Linear Layer", "线性层", "神经网络", "执行矩阵乘法加偏置的可学习变换层。", "part1/neural_network_basics", ("Fully Connected Layer", "Weight", "Bias"), ()),
    ("Log Loss", "对数损失", "优化训练", "对错误且高置信预测施加较大惩罚的概率损失。", "part1/machine_learning_basics", ("Cross Entropy", "Binary Cross Entropy", "Softmax"), ()),
    ("Logit", "Logit", "神经网络", "进入 sigmoid 或 softmax 前的未归一化分数。", "part1/neural_network_basics", ("Softmax", "Sigmoid", "Cross Entropy"), ()),
    ("Long Short-Term Memory", "长短期记忆网络", "序列模型", "用门控和记忆单元建模长距离依赖的 RNN 变体。", "part3/01_rnn_intuition", ("RNN", "GRU", "Hidden State"), ("LSTM",)),
    ("Loss Function", "损失函数", "优化训练", "衡量模型预测和目标之间差异的可优化目标。", "part1/machine_learning_basics", ("Cross Entropy", "MSE", "Gradient Descent"), ()),
    ("Machine Learning", "机器学习", "基础概念", "让系统从数据中学习规律并对新样本做预测的方法。", "part1/machine_learning_basics", ("Deep Learning", "Supervised Learning", "Generalization"), ("ML",)),
    ("Masked Language Modeling", "掩码语言建模", "Transformer与NLP", "遮盖部分 token 并训练模型恢复它们的预训练任务。", "part4/transformer_models", ("BERT", "Pretraining", "Encoder-only Transformer"), ("MLM",)),
    ("Mask", "掩码", "Transformer与NLP", "在计算中屏蔽无效位置或未来信息的二值结构。", "part4/04_minimal_transformer", ("Causal Mask", "Padding Mask", "Attention"), ()),
    ("Max Pooling", "最大池化", "CNN视觉", "取局部窗口最大值以降低尺寸并保留强响应。", "part2/02_feature_maps", ("Pooling", "Average Pooling", "Feature Map"), ()),
    ("Mean Absolute Error", "平均绝对误差", "数据与评估", "预测值与真实值绝对差的平均。", "part1/machine_learning_basics", ("MSE", "Huber Loss", "Regression"), ("MAE",)),
    ("Mean Squared Error", "均方误差", "数据与评估", "预测值与真实值差的平方平均。", "part1/machine_learning_basics", ("MAE", "Huber Loss", "Regression"), ("MSE",)),
    ("Memory Bandwidth", "内存带宽", "工程部署", "单位时间内硬件可读写数据的速度上限。", "part4/05_flash_attention", ("Flash Attention", "GPU", "Throughput"), ()),
    ("Message Passing", "消息传递", "图学习与强化学习", "图神经网络中节点从邻居聚合信息并更新表示的过程。", "part4/gnn_intro", ("Graph Neural Network", "Node Embedding", "Aggregation"), ()),
    ("Mini-batch", "小批量", "优化训练", "介于单样本和全量数据之间的一小组训练样本。", "part5/data_training", ("Batch", "Batch Size", "SGD"), ()),
    ("Mixed Precision", "混合精度", "工程部署", "结合低精度和高精度计算以加速训练或推理的方法。", "part5/deployment_tools", ("FP16", "BF16", "Loss Scaling"), ()),
    ("MLP", "多层感知机", "神经网络", "由线性层和非线性激活堆叠形成的前馈神经网络。", "part1/neural_network_basics", ("Fully Connected Layer", "Activation Function", "Backpropagation"), ("Multilayer Perceptron",)),
    ("Model Compression", "模型压缩", "工程部署", "通过剪枝、量化或蒸馏降低模型大小和计算成本。", "part5/deployment_tools", ("Pruning", "Quantization", "Knowledge Distillation"), ()),
    ("Model Parallelism", "模型并行", "工程部署", "把同一个大模型拆分到多个设备上计算。", "part5/deployment_tools", ("Distributed Training", "Pipeline Parallelism", "Tensor Parallelism"), ()),
    ("Model Serving", "模型服务", "工程部署", "把模型封装为可被业务系统调用的在线服务。", "part5/deployment_tools", ("Inference", "Latency", "Throughput"), ()),
    ("Momentum", "动量", "优化训练", "把历史梯度方向加入当前更新以加速和稳定优化。", "part1/03_datasets_optimizers", ("SGD", "Adam", "Optimizer"), ()),
    ("Multi-head Attention", "多头注意力", "Transformer与NLP", "并行使用多个注意力头从不同子空间建模关系。", "part4/02_multihead_visual", ("Attention", "Head", "Transformer"), ("MHA",)),
    ("Multimodal Model", "多模态模型", "前沿与安全", "同时处理文本、图像、音频或视频等多种模态的模型。", "part6/frontier", ("CLIP", "Vision-Language Model", "Embedding"), ()),
    ("Negative Log Likelihood", "负对数似然", "优化训练", "最大似然训练中最小化的目标形式。", "part1/machine_learning_basics", ("Cross Entropy", "Log Loss", "Probability"), ("NLL",)),
    ("Neural Network", "神经网络", "神经网络", "由可学习层和非线性函数组成的参数化模型。", "part1/neural_network_basics", ("Layer", "Activation Function", "Backpropagation"), ("NN",)),
    ("Node Embedding", "节点嵌入", "图学习与强化学习", "图中节点的连续向量表示。", "part4/gnn_intro", ("Graph Neural Network", "Message Passing", "Embedding"), ()),
    ("Normalization", "归一化", "数据与评估", "把数据或激活调整到稳定尺度的处理方法。", "part1/02_activations_normalization", ("Standardization", "Batch Normalization", "Layer Normalization"), ()),
    ("Object Detection", "目标检测", "CNN视觉", "同时定位并分类图像中目标的视觉任务。", "part2/09_transfer_learning", ("Bounding Box", "IoU", "Anchor Box"), ()),
    ("One-hot Encoding", "独热编码", "数据与评估", "用只有一个位置为 1 的向量表示离散类别。", "part1/machine_learning_basics", ("Label", "Categorical Variable", "Softmax"), ()),
    ("Optimizer", "优化器", "优化训练", "根据梯度和状态规则更新模型参数的算法。", "part1/03_datasets_optimizers", ("SGD", "Adam", "Learning Rate"), ()),
    ("Optimizer Step", "优化器步进", "优化训练", "优化器根据当前梯度实际更新参数的一次操作。", "part5/data_training", ("Gradient", "Learning Rate", "Training Loop"), ()),
    ("Out-of-distribution", "分布外", "数据与评估", "输入来自训练分布之外的情况。", "part6/frontier", ("Domain Shift", "Robustness", "Uncertainty"), ("OOD",)),
    ("Output Layer", "输出层", "神经网络", "把内部表示转换为任务预测的最后一层。", "part1/neural_network_basics", ("Logit", "Softmax", "Classifier Head"), ()),
    ("Overfitting", "过拟合", "基础概念", "模型过度记住训练数据而在新数据上表现变差。", "part1/machine_learning_basics", ("Generalization", "Regularization", "Early Stopping"), ()),
    ("Padding", "填充", "CNN视觉", "在输入边界补值以控制卷积输出尺寸的操作。", "part2/01_convolution_visual", ("Convolution", "Stride", "Kernel"), ()),
    ("Padding Mask", "填充掩码", "Transformer与NLP", "屏蔽序列中补齐位置以避免参与注意力计算的掩码。", "part4/04_minimal_transformer", ("Mask", "Attention", "Tokenization"), ()),
    ("Parameter", "参数", "神经网络", "模型中通过训练学习得到的权重或偏置。", "part1/neural_network_basics", ("Weight", "Bias", "Optimizer"), ()),
    ("PCA", "主成分分析", "数学基础", "寻找最大方差方向并投影数据的线性降维方法。", "part1/classical_ml", ("Dimensionality Reduction", "Eigenvector", "Variance"), ("Principal Component Analysis",)),
    ("Perceptron", "感知机", "神经网络", "最早的线性二分类神经元模型。", "part1/neural_network_basics", ("Linear Classifier", "Activation Function", "MLP"), ()),
    ("Perplexity", "困惑度", "Transformer与NLP", "语言模型平均预测不确定性的指数化指标。", "part4/transformer_models", ("Language Model", "Cross Entropy", "Token"), ("PPL",)),
    ("Pipeline", "流水线", "工程部署", "把数据处理、训练、评估和部署组织成可重复流程。", "part5/deployment_tools", ("Data Loader", "Training Loop", "Model Serving"), ()),
    ("Pooling", "池化", "CNN视觉", "在局部区域聚合响应以降低空间尺寸的操作。", "part2/02_feature_maps", ("Max Pooling", "Average Pooling", "Feature Map"), ()),
    ("Positional Encoding", "位置编码", "Transformer与NLP", "向模型提供 token 顺序或位置的信息。", "part4/04_minimal_transformer", ("Transformer", "Embedding", "Self-Attention"), ()),
    ("Precision", "精确率", "数据与评估", "预测为正的样本中真正为正的比例。", "part5/data_training", ("Recall", "F1 Score", "False Positive"), ()),
    ("Pretraining", "预训练", "优化训练", "先在大规模通用数据上训练模型以获得可迁移能力。", "part6/frontier", ("Fine-tuning", "Foundation Model", "Self-supervised Learning"), ()),
    ("Prompt", "提示词", "前沿与安全", "给生成模型的任务说明、上下文和约束输入。", "part6/frontier", ("In-context Learning", "LLM", "Instruction Tuning"), ()),
    ("Pruning", "剪枝", "工程部署", "移除不重要权重、通道或结构以压缩模型的方法。", "part5/deployment_tools", ("Model Compression", "Sparsity", "Quantization"), ()),
    ("Q-Learning", "Q 学习", "图学习与强化学习", "学习状态-动作价值函数的无模型强化学习算法。", "part6/reinforcement_learning", ("Q-value", "Reward", "Policy"), ()),
    ("Q-value", "动作价值", "图学习与强化学习", "在某状态执行某动作后期望获得的长期回报。", "part6/reinforcement_learning", ("Q-Learning", "Value Function", "Reward"), ()),
    ("Quantization", "量化", "工程部署", "用更低比特数表示权重或激活以降低成本的方法。", "part5/deployment_tools", ("Model Compression", "INT8", "Mixed Precision"), ()),
    ("Query", "查询向量", "Transformer与NLP", "注意力中用于检索相关键值信息的向量。", "part4/01_attention_mechanism", ("Key", "Value", "Attention"), ()),
    ("Random Forest", "随机森林", "基础概念", "由多个随机决策树组成的袋装集成模型。", "part1/classical_ml", ("Bagging", "Decision Tree", "Ensemble"), ()),
    ("Random Search", "随机搜索", "优化训练", "从超参数空间随机采样组合进行评估的方法。", "part5/04_hyperparam_search", ("Grid Search", "Hyperparameter", "Validation Set"), ()),
    ("Random Seed", "随机种子", "工程部署", "控制伪随机过程初始状态以提高可复现性。", "part5/data_training", ("Determinism", "Reproducibility", "Experiment Tracking"), ()),
    ("Recall", "召回率", "数据与评估", "真实正类中被模型正确找出的比例。", "part5/data_training", ("Precision", "F1 Score", "False Negative"), ()),
    ("Receptive Field", "感受野", "CNN视觉", "输出单元在输入空间中能够看到的区域范围。", "part2/02_feature_maps", ("Convolution", "Pooling", "Dilation"), ()),
    ("Recurrent Neural Network", "循环神经网络", "序列模型", "通过循环连接在序列时间步间共享状态的神经网络。", "part3/01_rnn_intuition", ("Hidden State", "LSTM", "GRU"), ("RNN",)),
    ("Regression", "回归", "基础概念", "预测连续数值目标的监督学习任务。", "part1/machine_learning_basics", ("Classification", "MSE", "MAE"), ()),
    ("Regularization", "正则化", "优化训练", "约束模型复杂度以减少过拟合的方法集合。", "part1/machine_learning_basics", ("Dropout", "Weight Decay", "Data Augmentation"), ()),
    ("Reinforcement Learning", "强化学习", "图学习与强化学习", "智能体通过与环境交互并最大化回报来学习策略。", "part6/reinforcement_learning", ("Agent", "Reward", "Policy"), ("RL",)),
    ("ReLU", "线性整流单元", "神经网络", "输出 max(0, x) 的简单高效非线性激活函数。", "part1/02_activations_normalization", ("Activation Function", "Leaky ReLU", "GELU"), ()),
    ("Representation Learning", "表示学习", "基础概念", "让模型自动学习适合任务的特征表示。", "part1/neural_network_basics", ("Embedding", "Feature", "Deep Learning"), ()),
    ("Residual Connection", "残差连接", "神经网络", "把输入直接加到后续层输出以改善梯度传播的连接。", "part2/06_modern_architectures", ("ResNet", "Gradient Flow", "Skip Connection"), ()),
    ("ResNet", "残差网络", "CNN视觉", "大量使用残差连接构建的深层卷积网络。", "part2/03_classic_architectures", ("Residual Connection", "CNN", "Batch Normalization"), ()),
    ("Reward", "奖励", "图学习与强化学习", "强化学习环境给智能体行为反馈的标量信号。", "part6/reinforcement_learning", ("Policy", "Q-value", "Value Function"), ()),
    ("RLHF", "人类反馈强化学习", "前沿与安全", "用人类偏好训练奖励模型并优化生成模型行为的方法。", "part6/frontier", ("Alignment", "Reward Model", "Policy Optimization"), ("Reinforcement Learning from Human Feedback",)),
    ("RMSProp", "RMSProp 优化器", "优化训练", "用梯度平方的移动平均调整学习率的优化器。", "part1/03_datasets_optimizers", ("Adagrad", "Adam", "Learning Rate"), ()),
    ("ROC Curve", "ROC 曲线", "数据与评估", "展示不同阈值下真正率和假正率关系的曲线。", "part5/data_training", ("AUC", "Threshold", "Binary Classification"), ()),
    ("Sampling", "采样", "生成模型", "从概率分布中抽取样本或生成 token 的过程。", "part4/transformer_models", ("Temperature", "Top-k Sampling", "Top-p Sampling"), ()),
    ("Scaled Dot-Product Attention", "缩放点积注意力", "Transformer与NLP", "用查询和键的点积缩放后计算权重并加权值向量。", "part4/01_attention_mechanism", ("Query", "Key", "Value"), ()),
    ("Scheduler", "调度器", "优化训练", "按预设规则改变学习率或训练策略的组件。", "part5/data_training", ("Learning Rate Scheduler", "Warmup", "Cosine Annealing"), ()),
    ("Self-Attention", "自注意力", "Transformer与NLP", "同一序列内部各位置相互计算注意力的机制。", "part4/01_attention_mechanism", ("Attention", "Transformer", "Multi-head Attention"), ()),
    ("Self-supervised Learning", "自监督学习", "前沿与安全", "从数据自身构造监督信号进行训练的方法。", "part6/frontier", ("Pretraining", "Contrastive Learning", "Masked Language Modeling"), ()),
    ("Semantic Segmentation", "语义分割", "CNN视觉", "给图像中每个像素分配语义类别的任务。", "part2/09_transfer_learning", ("Image Segmentation", "U-Net", "IoU"), ()),
    ("Seq2Seq", "序列到序列", "序列模型", "把一个输入序列映射为一个输出序列的建模框架。", "part3/05_seq2seq_attention", ("Encoder", "Decoder", "Attention"), ("Sequence-to-Sequence",)),
    ("SGD", "随机梯度下降", "优化训练", "用随机小批量梯度近似全量梯度进行更新的方法。", "part1/03_datasets_optimizers", ("Gradient Descent", "Mini-batch", "Momentum"), ()),
    ("Shape", "形状", "数学基础", "张量每个维度的大小。", "part1/01_tensors_gradients", ("Tensor", "Broadcasting", "Dimension"), ()),
    ("Shuffle", "打乱", "数据与评估", "随机重排样本顺序以降低训练顺序偏差。", "part5/data_training", ("Data Loader", "Batch", "Random Seed"), ()),
    ("Sigmoid", "Sigmoid 函数", "神经网络", "把实数映射到 0 到 1 区间的 S 形函数。", "part1/02_activations_normalization", ("Activation Function", "Binary Cross Entropy", "Logit"), ()),
    ("Skip Connection", "跳跃连接", "神经网络", "让信息跨过若干层直接传递的连接方式。", "part2/06_modern_architectures", ("Residual Connection", "U-Net", "Gradient Flow"), ()),
    ("Softmax", "Softmax 函数", "神经网络", "把一组分数归一化为概率分布的函数。", "part1/neural_network_basics", ("Logit", "Cross Entropy", "Classification"), ()),
    ("Sparsity", "稀疏性", "工程部署", "大量元素为零或可忽略的结构特征。", "part5/deployment_tools", ("Pruning", "Regularization", "Model Compression"), ()),
    ("Standardization", "标准化", "数据与评估", "把特征转换为零均值和单位方差的预处理方法。", "part1/machine_learning_basics", ("Normalization", "Feature Scaling", "Data Preprocessing"), ()),
    ("State Dict", "状态字典", "工程部署", "保存模型参数和缓冲区名称到张量映射的结构。", "part5/data_training", ("Checkpoint", "Parameter", "PyTorch"), ()),
    ("Stride", "步幅", "CNN视觉", "卷积核或池化窗口每次滑动的间隔。", "part2/01_convolution_visual", ("Convolution", "Padding", "Feature Map"), ()),
    ("Subword", "子词", "Transformer与NLP", "介于字符和完整词之间的分词单位。", "part4/transformer_models", ("BPE", "Tokenization", "Vocabulary"), ()),
    ("Supervised Learning", "监督学习", "基础概念", "用带标签数据学习输入到目标映射的机器学习范式。", "part1/machine_learning_basics", ("Label", "Classification", "Regression"), ()),
    ("Tanh", "双曲正切", "神经网络", "把输入映射到 -1 到 1 区间的 S 形激活函数。", "part1/02_activations_normalization", ("Activation Function", "Sigmoid", "RNN"), ()),
    ("Temperature", "温度", "生成模型", "控制生成分布平滑程度和随机性的参数。", "part4/transformer_models", ("Sampling", "Softmax", "Top-p Sampling"), ()),
    ("Tensor", "张量", "数学基础", "可表示标量、向量、矩阵和高维数组的基本数据结构。", "part1/01_tensors_gradients", ("Shape", "Broadcasting", "Autograd"), ()),
    ("Tensor Core", "张量核心", "工程部署", "GPU 中加速矩阵乘法和低精度计算的专用单元。", "part5/deployment_tools", ("CUDA", "Mixed Precision", "Throughput"), ()),
    ("Test Set", "测试集", "数据与评估", "只在最终评估时使用的数据子集。", "part5/data_training", ("Train Set", "Validation Set", "Generalization"), ()),
    ("Text Classification", "文本分类", "序列模型", "为文本片段预测类别标签的 NLP 任务。", "part3/06_text_classification", ("Tokenization", "Embedding", "Classifier"), ()),
    ("Threshold", "阈值", "数据与评估", "把连续分数转换为离散决策的切分值。", "part5/data_training", ("Precision", "Recall", "ROC Curve"), ()),
    ("Throughput", "吞吐量", "工程部署", "单位时间内系统可处理的请求数或样本数。", "part5/deployment_tools", ("Latency", "Batching", "Model Serving"), ()),
    ("Token", "词元", "Transformer与NLP", "模型处理文本时的最小离散输入单位。", "part4/transformer_models", ("Tokenization", "BPE", "Embedding"), ()),
    ("Token Embedding", "词元嵌入", "Transformer与NLP", "把 token ID 映射为连续向量的嵌入表。", "part4/transformer_models", ("Token", "Embedding", "Vocabulary"), ()),
    ("Tokenization", "分词", "Transformer与NLP", "把文本切分为模型词表中 token 序列的过程。", "part4/transformer_models", ("Token", "BPE", "Vocabulary"), ()),
    ("Top-k Sampling", "Top-k 采样", "生成模型", "每步只从概率最高的 k 个 token 中采样。", "part4/transformer_models", ("Sampling", "Temperature", "Top-p Sampling"), ()),
    ("Top-p Sampling", "核采样", "生成模型", "从累计概率达到 p 的最小候选集合中采样。", "part4/transformer_models", ("Sampling", "Temperature", "Top-k Sampling"), ("Nucleus Sampling",)),
    ("Train Set", "训练集", "数据与评估", "用于拟合模型参数的数据子集。", "part5/data_training", ("Validation Set", "Test Set", "Dataset"), ()),
    ("Training Loop", "训练循环", "优化训练", "重复执行前向传播、损失计算、反向传播和参数更新的流程。", "part5/data_training", ("Forward Pass", "Backpropagation", "Optimizer Step"), ()),
    ("Transfer Learning", "迁移学习", "优化训练", "把一个任务或领域学到的知识用于另一个任务或领域。", "part2/09_transfer_learning", ("Pretraining", "Fine-tuning", "Domain Adaptation"), ()),
    ("Transformer", "Transformer", "Transformer与NLP", "基于自注意力和前馈网络堆叠的序列建模架构。", "part4/04_minimal_transformer", ("Self-Attention", "Multi-head Attention", "Positional Encoding"), ()),
    ("Transposed Convolution", "转置卷积", "CNN视觉", "常用于上采样特征图的可学习卷积变换。", "part2/07_advanced_convolution", ("Upsampling", "Bilinear Interpolation", "U-Net"), ("deconvolution",)),
    ("True Negative", "真阴性", "数据与评估", "真实为负类且模型预测为负类的样本。", "part5/data_training", ("Confusion Matrix", "False Positive", "Accuracy"), ("TN",)),
    ("True Positive", "真阳性", "数据与评估", "真实为正类且模型预测为正类的样本。", "part5/data_training", ("Confusion Matrix", "False Negative", "Recall"), ("TP",)),
    ("U-Net", "U-Net", "CNN视觉", "带编码器、解码器和跳跃连接的图像分割网络。", "part2/09_transfer_learning", ("Semantic Segmentation", "Skip Connection", "Transposed Convolution"), ()),
    ("Underfitting", "欠拟合", "基础概念", "模型过于简单或训练不足而无法捕捉训练数据规律。", "part1/machine_learning_basics", ("Overfitting", "Bias-Variance Tradeoff", "Model Capacity"), ()),
    ("Unsupervised Learning", "无监督学习", "基础概念", "在没有显式标签的情况下发现数据结构的学习范式。", "part1/machine_learning_basics", ("Clustering", "Dimensionality Reduction", "Self-supervised Learning"), ()),
    ("Upsampling", "上采样", "CNN视觉", "把低分辨率特征图放大到更高分辨率的操作。", "part2/07_advanced_convolution", ("Transposed Convolution", "Bilinear Interpolation", "U-Net"), ()),
    ("Validation Loss", "验证损失", "数据与评估", "模型在验证集上的损失值，用于监控泛化趋势。", "part5/data_training", ("Validation Set", "Early Stopping", "Overfitting"), ()),
    ("Validation Set", "验证集", "数据与评估", "训练期间用于调参和模型选择的数据子集。", "part5/data_training", ("Train Set", "Test Set", "Cross Validation"), ()),
    ("Value", "值向量", "Transformer与NLP", "注意力中被权重加权汇总的信息向量。", "part4/01_attention_mechanism", ("Query", "Key", "Attention"), ()),
    ("Value Function", "价值函数", "图学习与强化学习", "估计状态或状态-动作长期回报的函数。", "part6/reinforcement_learning", ("Q-value", "Reward", "Policy"), ()),
    ("Vanishing Gradient", "梯度消失", "优化训练", "梯度在反向传播中逐层变小导致早期层难以学习的现象。", "part3/07_advanced_training", ("Exploding Gradient", "Gradient Flow", "LSTM"), ()),
    ("Variational Autoencoder", "变分自编码器", "生成模型", "用概率潜变量和重参数化技巧学习生成分布的自编码器。", "part4/gan_ae", ("Autoencoder", "Latent Space", "KL Divergence"), ("VAE",)),
    ("Vector", "向量", "数学基础", "具有方向和大小的一维数值数组。", "part1/math_primer", ("Tensor", "Dot Product", "Embedding"), ()),
    ("Vectorization", "向量化", "工程部署", "用批量张量运算替代显式循环以提升效率。", "part1/01_tensors_gradients", ("Tensor", "Broadcasting", "GPU"), ()),
    ("Vision Transformer", "视觉 Transformer", "CNN视觉", "把图像切成 patch 并用 Transformer 建模的视觉架构。", "part2/06_modern_architectures", ("Transformer", "Patch Embedding", "Self-Attention"), ("ViT",)),
    ("Vocabulary", "词表", "Transformer与NLP", "模型可识别 token 到 ID 的映射集合。", "part4/transformer_models", ("Tokenization", "BPE", "Token"), ()),
    ("Warmup", "预热", "优化训练", "训练初期从较小学习率逐步升高到目标学习率的策略。", "part5/data_training", ("Learning Rate Scheduler", "Transformer", "Training Stability"), ()),
    ("Weight", "权重", "神经网络", "模型中通过训练学习的乘性参数。", "part1/neural_network_basics", ("Bias", "Parameter", "Linear Layer"), ()),
    ("Weight Decay", "权重衰减", "优化训练", "通过惩罚较大权重实现正则化的优化技巧。", "part1/03_datasets_optimizers", ("Regularization", "AdamW", "Overfitting"), ()),
    ("Word Embedding", "词嵌入", "Transformer与NLP", "把词或子词表示为连续向量的技术。", "part4/transformer_models", ("Embedding", "Token Embedding", "Word2Vec"), ()),
    ("Zero-shot Learning", "零样本学习", "前沿与安全", "模型在没有目标任务训练样本时直接完成任务的能力。", "part6/frontier", ("Few-shot Learning", "Prompt", "Foundation Model"), ()),
    ("Agent", "智能体", "图学习与强化学习", "在环境中观察、行动并接收奖励的决策实体。", "part6/reinforcement_learning", ("Environment", "Policy", "Reward"), ()),
    ("Alignment", "对齐", "前沿与安全", "让模型行为、目标和人类意图保持一致的研究方向。", "part6/frontier", ("RLHF", "Safety", "Reward Model"), ()),
    ("Anchor Box", "锚框", "CNN视觉", "目标检测中预设形状和尺度的候选边界框。", "part2/09_transfer_learning", ("Bounding Box", "Object Detection", "IoU"), ()),
    ("Annotation", "标注", "数据与评估", "为样本添加标签、边界框或文本说明的过程。", "part5/data_training", ("Label", "Ground Truth", "Dataset"), ()),
    ("Autoregressive Model", "自回归模型", "Transformer与NLP", "按顺序用已生成内容预测下一个元素的模型。", "part4/transformer_models", ("Decoder-only Transformer", "Causal Mask", "Language Model"), ()),
    ("BF16", "Brain Float 16", "工程部署", "保留较大指数范围的 16 位浮点格式。", "part5/deployment_tools", ("Mixed Precision", "FP16", "Tensor Core"), ()),
    ("Boosting", "提升法", "基础概念", "按顺序训练多个弱模型并逐步纠正前序错误的集成方法。", "part1/classical_ml", ("Ensemble", "Bagging", "Gradient Boosting"), ()),
    ("Channel", "通道", "CNN视觉", "特征图中表示不同滤波响应或颜色分量的维度。", "part2/02_feature_maps", ("Feature Map", "Convolution", "Kernel"), ()),
    ("Classifier Head", "分类头", "神经网络", "接在主干网络后用于输出类别预测的小型模块。", "part2/09_transfer_learning", ("Backbone", "Fine-tuning", "Output Layer"), ()),
    ("Clustering", "聚类", "基础概念", "把相似样本自动分组的无监督学习任务。", "part1/classical_ml", ("Unsupervised Learning", "Embedding", "K-means"), ()),
    ("Concept Drift", "概念漂移", "数据与评估", "输入与目标之间的关系随时间变化的现象。", "part5/deployment_tools", ("Data Drift", "Monitoring", "Domain Shift"), ()),
    ("Cosine Annealing", "余弦退火", "优化训练", "按余弦曲线逐步降低学习率的调度策略。", "part5/data_training", ("Learning Rate Scheduler", "Warmup", "SGD"), ()),
    ("Data Drift", "数据漂移", "工程部署", "线上输入数据分布随时间偏离训练分布的现象。", "part5/deployment_tools", ("Monitoring", "Domain Shift", "Concept Drift"), ()),
    ("Data Parallelism", "数据并行", "工程部署", "把不同批次分配到多个设备并同步梯度的训练方式。", "part5/deployment_tools", ("Distributed Training", "Gradient Synchronization", "Batch Size"), ()),
    ("Denoising", "去噪", "生成模型", "从被噪声污染的数据恢复干净信号的学习任务。", "part6/frontier", ("Diffusion Model", "Autoencoder", "Score Matching"), ()),
    ("Derivative", "导数", "数学基础", "函数输出对输入微小变化的局部变化率。", "part1/math_primer", ("Gradient", "Chain Rule", "Backpropagation"), ()),
    ("Discriminator", "判别器", "生成模型", "在 GAN 中区分真实样本和生成样本的网络。", "part4/gan_ae", ("GAN", "Generator", "Adversarial Loss"), ()),
    ("Drop Path", "随机路径丢弃", "神经网络", "训练时随机跳过整个残差分支以正则化深层网络。", "part2/06_modern_architectures", ("Dropout", "Residual Connection", "Regularization"), ()),
    ("Early Fusion", "早期融合", "前沿与安全", "在较低层或输入附近合并不同模态信息的方法。", "part6/frontier", ("Multimodal Model", "Late Fusion", "Embedding"), ()),
    ("Edge", "边", "图学习与强化学习", "图中连接两个节点并表示关系的结构。", "part4/gnn_intro", ("Node", "Graph Neural Network", "Message Passing"), ()),
    ("Entropy", "熵", "数学基础", "衡量概率分布不确定性的信息量。", "part1/math_primer", ("Cross Entropy", "KL Divergence", "Probability Distribution"), ()),
    ("Environment", "环境", "图学习与强化学习", "强化学习中接收动作并返回状态和奖励的外部系统。", "part6/reinforcement_learning", ("Agent", "Reward", "State"), ()),
    ("Experiment Tracking", "实验追踪", "工程部署", "记录配置、指标、模型和产物以便比较实验的方法。", "part5/deployment_tools", ("Checkpoint", "Metric", "Reproducibility"), ()),
    ("Feature Scaling", "特征缩放", "数据与评估", "把不同量纲特征调整到相近数值范围的预处理。", "part1/machine_learning_basics", ("Standardization", "Normalization", "Gradient Descent"), ()),
    ("Forward Pass", "前向传播", "神经网络", "输入经过模型各层计算得到输出和损失的过程。", "part1/neural_network_basics", ("Backpropagation", "Loss Function", "Computational Graph"), ()),
    ("FP16", "半精度浮点", "工程部署", "使用 16 位表示浮点数以降低显存和提升速度的格式。", "part5/deployment_tools", ("Mixed Precision", "BF16", "Loss Scaling"), ()),
    ("Gradient Synchronization", "梯度同步", "工程部署", "分布式训练中聚合多个设备梯度的操作。", "part5/deployment_tools", ("Distributed Training", "Data Parallelism", "AllReduce"), ()),
    ("Head", "注意力头", "Transformer与NLP", "多头注意力中独立计算一组查询、键、值投影的分支。", "part4/02_multihead_visual", ("Multi-head Attention", "Query", "Key"), ()),
    ("Hessian", "海森矩阵", "数学基础", "标量函数二阶偏导数组成的矩阵。", "part1/math_primer", ("Gradient", "Jacobian", "Optimization"), ()),
    ("Instruction Tuning", "指令微调", "前沿与安全", "用指令-回答数据训练模型更好遵循人类任务描述。", "part6/frontier", ("Prompt", "Fine-tuning", "Alignment"), ()),
    ("INT8", "8 位整数量化", "工程部署", "用 8 位整数表示权重或激活的量化格式。", "part5/deployment_tools", ("Quantization", "Model Compression", "Inference"), ()),
    ("K-fold", "K 折验证", "数据与评估", "把数据分成 K 份轮流做验证集的交叉验证方法。", "part5/data_training", ("Cross Validation", "Validation Set", "Generalization"), ()),
    ("K-means", "K 均值", "基础概念", "把样本分配到 K 个簇并迭代更新簇中心的聚类算法。", "part1/classical_ml", ("Clustering", "Unsupervised Learning", "Centroid"), ()),
    ("Language Model", "语言模型", "Transformer与NLP", "估计文本序列概率或预测下一个 token 的模型。", "part4/transformer_models", ("Autoregressive Model", "Perplexity", "Token"), ("LM",)),
    ("Late Fusion", "后期融合", "前沿与安全", "先分别处理不同模态再合并高层表示或决策的方法。", "part6/frontier", ("Multimodal Model", "Early Fusion", "Ensemble"), ()),
    ("Latency", "延迟", "工程部署", "一次请求从输入到得到输出所需的时间。", "part5/deployment_tools", ("Inference", "Throughput", "Model Serving"), ()),
    ("Leaky ReLU", "带泄漏 ReLU", "神经网络", "在负半轴保留小斜率以缓解神经元死亡的 ReLU 变体。", "part1/02_activations_normalization", ("ReLU", "Activation Function", "Gradient Flow"), ()),
    ("LLM", "大语言模型", "前沿与安全", "在海量文本上训练、具备强语言理解和生成能力的大模型。", "part6/frontier", ("Foundation Model", "Prompt", "Alignment"), ("Large Language Model",)),
    ("Loss Scaling", "损失缩放", "工程部署", "混合精度训练中放大损失以避免梯度下溢的方法。", "part5/deployment_tools", ("Mixed Precision", "FP16", "Gradient"), ()),
    ("Margin", "间隔", "基础概念", "分类边界到样本之间的距离或置信余量。", "part1/classical_ml", ("Decision Boundary", "SVM", "Generalization"), ()),
    ("Mean", "均值", "数学基础", "一组数值的平均水平。", "part1/math_primer", ("Variance", "Standardization", "Batch Normalization"), ()),
    ("Metric", "指标", "数据与评估", "用于跟踪训练或评估表现的数值。", "part5/data_training", ("Evaluation Metric", "Validation Loss", "Experiment Tracking"), ()),
    ("Model Capacity", "模型容量", "基础概念", "模型可表示函数复杂度的能力。", "part1/machine_learning_basics", ("Overfitting", "Underfitting", "Parameter"), ()),
    ("Monitoring", "监控", "工程部署", "持续观察线上指标、数据分布和模型行为的过程。", "part5/deployment_tools", ("Data Drift", "Latency", "Experiment Tracking"), ()),
    ("Node", "节点", "图学习与强化学习", "图结构中的实体或顶点。", "part4/gnn_intro", ("Edge", "Graph Neural Network", "Node Embedding"), ()),
    ("Patch Embedding", "图像块嵌入", "CNN视觉", "把图像切块并投影为 Transformer token 的表示。", "part2/06_modern_architectures", ("Vision Transformer", "Token Embedding", "Positional Encoding"), ()),
    ("Pipeline Parallelism", "流水线并行", "工程部署", "把模型层切分到不同设备并流水执行微批次的并行方式。", "part5/deployment_tools", ("Model Parallelism", "Distributed Training", "Throughput"), ()),
    ("Pointwise Convolution", "逐点卷积", "CNN视觉", "使用 1x1 卷积混合通道信息的操作。", "part2/07_advanced_convolution", ("Depthwise Separable Convolution", "Channel", "Bottleneck"), ()),
    ("Policy", "策略", "图学习与强化学习", "强化学习中从状态到动作选择的规则或概率分布。", "part6/reinforcement_learning", ("Agent", "Reward", "Value Function"), ()),
    ("Policy Optimization", "策略优化", "图学习与强化学习", "直接调整策略参数以提高期望回报的方法。", "part6/reinforcement_learning", ("Policy", "Reward", "RLHF"), ()),
    ("Probability Distribution", "概率分布", "数学基础", "描述随机变量各取值可能性的函数或表。", "part1/math_primer", ("Softmax", "Entropy", "KL Divergence"), ()),
    ("Prompt Engineering", "提示词工程", "前沿与安全", "设计提示输入以引导模型稳定完成任务的方法。", "part6/frontier", ("Prompt", "In-context Learning", "LLM"), ()),
    ("Reliability Diagram", "可靠性图", "数据与评估", "展示预测置信度和实际准确率偏差的校准图。", "part5/data_training", ("Calibration", "Expected Calibration Error", "Metric"), ()),
    ("Replay Buffer", "经验回放池", "图学习与强化学习", "存储历史交互样本供强化学习反复采样训练的缓存。", "part6/reinforcement_learning", ("Q-Learning", "Off-policy Learning", "Sampling"), ()),
    ("Reward Model", "奖励模型", "前沿与安全", "根据人类偏好或规则评估输出质量的模型。", "part6/frontier", ("RLHF", "Alignment", "Policy Optimization"), ()),
    ("Robustness", "鲁棒性", "数据与评估", "模型面对噪声、扰动或分布变化时保持性能的能力。", "part6/frontier", ("Out-of-distribution", "Domain Shift", "Adversarial Example"), ()),
    ("Safety", "安全性", "前沿与安全", "模型避免造成不可接受伤害并可被约束和审计的性质。", "part6/frontier", ("Alignment", "Hallucination", "Robustness"), ()),
    ("Score Matching", "分数匹配", "生成模型", "学习数据分布对输入梯度的生成建模方法。", "part6/frontier", ("Diffusion Model", "Denoising", "Generative Model"), ()),
    ("State", "状态", "图学习与强化学习", "强化学习中环境在某一时刻提供给智能体的信息。", "part6/reinforcement_learning", ("Agent", "Action", "Policy"), ()),
    ("Tensor Parallelism", "张量并行", "工程部署", "把大矩阵或张量运算切分到多个设备上执行。", "part5/deployment_tools", ("Model Parallelism", "Distributed Training", "GPU"), ()),
    ("Training Stability", "训练稳定性", "优化训练", "训练过程中损失、梯度和指标保持可控变化的状态。", "part5/03_training_dynamics", ("Gradient Flow", "Learning Rate", "Normalization"), ()),
    ("Variance", "方差", "数学基础", "衡量数据围绕均值分散程度的统计量。", "part1/math_primer", ("Mean", "Standardization", "Bias-Variance Tradeoff"), ()),
    ("Vision-Language Model", "视觉语言模型", "前沿与安全", "同时理解图像和文本并能跨模态推理的模型。", "part6/frontier", ("CLIP", "Multimodal Model", "Embedding"), ("VLM",)),
    ("Voting", "投票法", "基础概念", "集成模型中通过多数票或加权票合并预测的方法。", "part1/classical_ml", ("Ensemble", "Bagging", "Classifier"), ()),
    ("Adapter", "适配器模块", "优化训练", "插入预训练模型层间的小型可训练模块，冻结主干只训练适配器实现高效微调。", "part6/frontier", ("Parameter-Efficient Fine-Tuning", "LoRA", "Fine-tuning"), ()),
    ("Agentic AI", "智能体式 AI", "前沿与安全", "具备自主规划、工具调用、记忆和多步推理能力的 AI 系统范式。", "part6/frontier", ("Agent", "Tool Use", "LLM"), ()),
    ("Chain of Thought", "思维链", "前沿与安全", "让模型在回答前逐步推理中间步骤以提升复杂任务表现的提示或训练策略。", "part6/frontier", ("Prompt", "Reasoning", "In-context Learning"), ("CoT",)),
    ("Direct Preference Optimization", "直接偏好优化", "前沿与安全", "绕过奖励模型、直接用偏好数据对比优化策略的对齐方法。", "part6/frontier", ("RLHF", "Alignment", "Reward Model"), ("DPO",)),
    ("Knowledge Distillation", "知识蒸馏", "工程部署", "用大模型的软标签或中间表示训练小模型以压缩知识和推理成本。", "part5/deployment_tools", ("Model Compression", "Transfer Learning", "Teacher-Student"), ()),
    ("KV Cache", "键值缓存", "工程部署", "自回归生成时缓存已计算的键值张量以避免重复计算、加速逐 token 推理。", "part4/05_flash_attention", ("Attention", "Inference", "Memory Bandwidth"), ()),
    ("LoRA", "低秩适配", "前沿与安全", "在权重矩阵旁注入低秩分解矩阵，只训练少量参数实现大模型高效微调。", "part6/frontier", ("Parameter-Efficient Fine-Tuning", "Adapter", "Fine-tuning"), ("Low-Rank Adaptation",)),
    ("Mixture of Experts", "混合专家", "前沿与安全", "用门控机制选择性激活部分专家子网络以扩大模型容量而不等比增加计算。", "part6/frontier", ("Transformer", "Gating Mechanism", "Sparse Model"), ("MoE",)),
    ("Parameter-Efficient Fine-Tuning", "参数高效微调", "前沿与安全", "只更新模型少部分参数以降低微调显存和数据需求的方法类别。", "part6/frontier", ("LoRA", "Adapter", "Fine-tuning"), ("PEFT",)),
    ("Red Teaming", "红队测试", "前沿与安全", "用对抗性测试、越狱尝试或恶意提示探测模型安全边界和失败模式。", "part6/frontier", ("Safety", "Alignment", "Robustness"), ()),
    ("Retrieval Augmented Generation", "检索增强生成", "前沿与安全", "在生成前检索外部知识库并注入上下文以减少幻觉和扩展知识边界。", "part6/frontier", ("LLM", "Embedding", "Hallucination"), ("RAG",)),
    ("Reward Hacking", "奖励攻击", "前沿与安全", "策略找到奖励模型漏洞并获得高分但实际输出不符合人类偏好的现象。", "part6/frontier", ("RLHF", "Alignment", "Reward Model"), ()),
    ("Test-time Compute", "推理时计算", "前沿与安全", "在推理阶段投入更多计算资源（如多次采样、长链推理）以提升输出质量。", "part6/frontier", ("Chain of Thought", "Inference", "Scaling"), ()),
)


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def e(text: str) -> str:
    return html.escape(text, quote=True)


def build_detail(english: str, chinese: str, category: str, definition: str, module: str, related: tuple[str, ...]) -> str:
    module_label = MODULES.get(module, module)
    relation = "、".join(related[:3]) if related else "上下游概念"
    return (
        f"{definition} 在深度学习实践中，{chinese}通常不是孤立概念："
        f"{CATEGORY_GUIDES[category]} 本书在「{module_label}」中讲解或使用它，"
        f"学习时可以把它和 {relation} 放在一起比较，重点观察它改变了数据表示、"
        f"梯度传播、模型容量、评估方式还是工程成本。"
    )


def load_terms() -> tuple[GlossaryTerm, ...]:
    terms = []
    for english, chinese, category, definition, module, related, aliases in RAW_TERMS:
        terms.append(
            GlossaryTerm(
                english=english,
                chinese=chinese,
                category=category,
                definition=definition,
                detail=build_detail(english, chinese, category, definition, module, related),
                related=related,
                modules=(module,),
                aliases=aliases,
            )
        )
    return tuple(sorted(terms, key=lambda term: (term.category, term.english.lower())))


TERMS = load_terms()
TERM_BY_KEY = {term.key: term for term in TERMS}
TERM_BY_RELATED = {normalize_key(term.english): term for term in TERMS}
TERM_BY_RELATED.update({normalize_key(alias): term for term in TERMS for alias in term.aliases})


def fuzzy_score(term: GlossaryTerm, query: str) -> float:
    query = query.strip().lower()
    if not query:
        return 1.0

    compact_query = normalize_key(query)
    weighted_fields = [
        (term.english.lower(), 1.0),
        (term.chinese.lower(), 1.0),
        *[(alias.lower(), 0.96) for alias in term.aliases],
        (term.definition.lower(), 0.86),
        (term.category.lower(), 0.80),
        (" ".join(term.related).lower(), 0.74),
        (" ".join(term.module_labels).lower(), 0.68),
        (term.detail.lower(), 0.28),
    ]
    best = 0.0
    for field, weight in weighted_fields:
        compact_field = normalize_key(field)
        if query in field or (compact_query and compact_query in compact_field):
            best = max(best, weight)
        best = max(best, SequenceMatcher(None, query, field).ratio() * weight)
        if compact_query:
            best = max(best, SequenceMatcher(None, compact_query, compact_field).ratio() * weight)
    return best


def search_terms(
    query: str,
    categories: list[str],
    modules: list[str],
    initials: list[str],
    minimum_score: float,
) -> list[tuple[GlossaryTerm, float]]:
    selected = []
    for term in TERMS:
        if categories and term.category not in categories:
            continue
        if modules and not any(label in modules for label in term.module_labels):
            continue
        if initials and term.initial not in initials:
            continue
        score = fuzzy_score(term, query)
        if score >= minimum_score:
            selected.append((term, score))
    return sorted(selected, key=lambda item: (-item[1], item[0].english.lower()))


def term_dataframe(terms: list[GlossaryTerm]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "English": term.english,
                "中文": term.chinese,
                "分类": term.category,
                "一句话定义": term.definition,
                "讲解模块": "；".join(term.module_labels),
                "相关术语": "，".join(term.related),
            }
            for term in terms
        ]
    )


def render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172026;
            --muted: #596772;
            --line: #d8dee3;
            --paper: #fbfaf6;
            --panel: #ffffff;
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
        .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
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
            font-size: clamp(2rem, 3vw, 3.1rem);
            line-height: 1.08;
            margin: 0;
        }
        .hero p {
            color: var(--muted);
            max-width: 1040px;
            line-height: 1.75;
            margin: 0.45rem 0 0 0;
            font-size: 1.02rem;
        }
        .term-card {
            background: rgba(255,255,255,0.80);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.92rem 1rem;
            margin: 0 0 0.72rem 0;
        }
        .term-card h3 {
            margin: 0;
            font-size: 1.15rem;
            line-height: 1.35;
        }
        .term-card p {
            color: var(--muted);
            line-height: 1.7;
            margin: 0.38rem 0 0 0;
        }
        .term-meta {
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
            margin-top: 0.55rem;
        }
        .tag {
            display: inline-block;
            padding: 0.15rem 0.46rem;
            border: 1px solid rgba(15,139,141,0.24);
            border-radius: 999px;
            color: #25616a;
            font-size: 0.78rem;
            background: rgba(15,139,141,0.07);
        }
        .tag.module {
            border-color: rgba(50,104,168,0.28);
            background: rgba(50,104,168,0.08);
            color: #315b86;
        }
        .tag.related {
            border-color: rgba(191,63,91,0.25);
            background: rgba(191,63,91,0.07);
            color: #864050;
        }
        .alpha-row {
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
            margin: 0.35rem 0 0.75rem 0;
        }
        .alpha-chip {
            min-width: 2rem;
            text-align: center;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.22rem 0.38rem;
            background: rgba(255,255,255,0.72);
            color: #26343b;
            font-size: 0.85rem;
        }
        .category-band {
            border-left: 4px solid var(--teal);
            background: rgba(255,255,255,0.72);
            border-radius: 0 8px 8px 0;
            padding: 0.7rem 0.85rem;
            margin: 0.4rem 0 0.8rem 0;
        }
        .category-band strong { display: block; margin-bottom: 0.25rem; }
        .category-band span { color: var(--muted); line-height: 1.65; }
        @media (max-width: 760px) {
            .term-card { padding: 0.82rem; }
            .hero h1 { font-size: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>深度学习术语词典</h1>
            <p>
                一个可搜索、可按模块回溯的核心概念索引。每个术语包含中英文名称、
                一句话定义、详细解释、相关术语和本项目中的讲解位置。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alpha_index(counts: Counter[str]) -> None:
    chips = []
    for initial in sorted(counts):
        chips.append(f'<span class="alpha-chip">{e(initial)} · {counts[initial]}</span>')
    st.markdown('<div class="alpha-row">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def render_category_browser() -> None:
    grouped: dict[str, list[GlossaryTerm]] = defaultdict(list)
    for term in TERMS:
        grouped[term.category].append(term)

    cols = st.columns(3)
    for index, (category, terms) in enumerate(sorted(grouped.items())):
        color = CATEGORY_COLORS.get(category, TEAL)
        preview = "、".join(term.chinese for term in sorted(terms, key=lambda item: item.english)[:7])
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="category-band" style="border-left-color:{color}">
                    <strong>{e(category)} · {len(terms)} 个</strong>
                    <span>{e(preview)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_term_card(term: GlossaryTerm, score: float | None = None) -> None:
    score_text = f'<span class="tag">匹配 {score:.2f}</span>' if score is not None and score < 0.999 else ""
    related_tags = "".join(f'<span class="tag related">{e(item)}</span>' for item in term.related)
    module_tags = "".join(f'<span class="tag module">{e(item)}</span>' for item in term.module_labels)
    aliases = f" · 别名：{e('，'.join(term.aliases))}" if term.aliases else ""
    st.markdown(
        f"""
        <div class="term-card" id="{e(term.key)}">
            <h3>{e(term.english)} <span style="color:{MUTED};font-weight:500;">{e(term.chinese)}</span></h3>
            <p><strong>一句话：</strong>{e(term.definition)}{aliases}</p>
            <p><strong>详细解释：</strong>{e(term.detail)}</p>
            <div class="term-meta">
                <span class="tag" style="border-color:{CATEGORY_COLORS.get(term.category, TEAL)}55;background:{CATEGORY_COLORS.get(term.category, TEAL)}12;color:{CATEGORY_COLORS.get(term.category, TEAL)};">{e(term.category)}</span>
                {score_text}
                {module_tags}
                {related_tags}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_related_explorer(selected_name: str) -> None:
    term = TERM_BY_KEY.get(normalize_key(selected_name))
    if not term:
        return

    st.subheader("相关术语链")
    cols = st.columns([1, 2])
    with cols[0]:
        render_term_card(term)
    with cols[1]:
        related_terms = []
        for related in term.related:
            match = TERM_BY_RELATED.get(normalize_key(related))
            if match:
                related_terms.append(match)
        if not related_terms:
            st.info("该术语的相关项还没有独立词条。")
            return
        for related_term in related_terms[:6]:
            render_term_card(related_term)


def main() -> None:
    render_css()
    render_hero()

    all_categories = sorted({term.category for term in TERMS})
    all_modules = sorted({label for term in TERMS for label in term.module_labels})
    all_initials = sorted({term.initial for term in TERMS})
    initial_counts = Counter(term.initial for term in TERMS)
    category_counts = Counter(term.category for term in TERMS)

    with st.sidebar:
        st.header("搜索与过滤")
        query = st.text_input("模糊搜索", placeholder="例如 attention、梯度、微调、CNN")
        categories = st.multiselect("分类", all_categories)
        modules = st.multiselect("讲解模块", all_modules)
        initials = st.multiselect("首字母", all_initials)
        minimum_score = st.slider("模糊匹配阈值", 0.10, 1.00, 0.34, 0.02)
        view_mode = st.radio("显示方式", ("卡片", "表格", "术语链"), horizontal=True)

    filtered = search_terms(query, categories, modules, initials, minimum_score)
    filtered_terms = [term for term, _ in filtered]

    metric_cols = st.columns(4)
    metric_cols[0].metric("词条总数", len(TERMS))
    metric_cols[1].metric("当前结果", len(filtered_terms))
    metric_cols[2].metric("分类数", len(all_categories))
    metric_cols[3].metric("覆盖模块", len(all_modules))

    tab_search, tab_categories, tab_alpha, tab_modules = st.tabs(["搜索结果", "分类浏览", "首字母索引", "模块索引"])

    with tab_search:
        if view_mode == "表格":
            st.dataframe(term_dataframe(filtered_terms), use_container_width=True, hide_index=True)
        elif view_mode == "术语链":
            options = [f"{term.english} · {term.chinese}" for term in filtered_terms] or [
                f"{term.english} · {term.chinese}" for term in TERMS
            ]
            selected = st.selectbox("选择一个术语查看关联概念", options)
            render_related_explorer(selected.split(" · ", 1)[0])
        else:
            if not filtered:
                st.warning("没有匹配的术语。可以降低阈值，或清空部分过滤条件。")
            for term, score in filtered[:80]:
                render_term_card(term, score if query else None)
            if len(filtered) > 80:
                st.caption(f"已显示前 80 个结果，共 {len(filtered)} 个。可继续增加过滤条件缩小范围。")

    with tab_categories:
        render_category_browser()
        selected_category = st.selectbox(
            "展开分类",
            all_categories,
            format_func=lambda item: f"{item} ({category_counts[item]})",
        )
        for term in sorted((item for item in TERMS if item.category == selected_category), key=lambda item: item.english):
            render_term_card(term)

    with tab_alpha:
        render_alpha_index(initial_counts)
        selected_initial = st.selectbox(
            "展开首字母",
            all_initials,
            format_func=lambda item: f"{item} ({initial_counts[item]})",
        )
        st.dataframe(
            term_dataframe(sorted((term for term in TERMS if term.initial == selected_initial), key=lambda item: item.english)),
            use_container_width=True,
            hide_index=True,
        )

    with tab_modules:
        module_choice = st.selectbox("选择项目模块", all_modules)
        module_terms = sorted(
            (term for term in TERMS if module_choice in term.module_labels),
            key=lambda item: (item.category, item.english),
        )
        st.caption(f"{module_choice} 中讲解或使用了 {len(module_terms)} 个术语。")
        st.dataframe(term_dataframe(module_terms), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
