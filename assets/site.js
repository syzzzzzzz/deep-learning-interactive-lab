const PARTS = [
  { key: "part1", roman: "I", title: "第一部分 基础", short: "基础", description: "张量、梯度、经典机器学习、数学基础和神经网络入门。" },
  { key: "part2", roman: "II", title: "第二部分 CNN", short: "CNN", description: "卷积、特征图、现代视觉架构、调试、迁移学习和可视化。" },
  { key: "part3", roman: "III", title: "第三部分 RNN", short: "RNN", description: "序列建模、隐藏状态、注意力、文本分类和高级训练技巧。" },
  { key: "part4", roman: "IV", title: "第四部分 Transformer", short: "Transformer", description: "自注意力、多头注意力、编码器解码器、生成模型和图神经网络。" },
  { key: "part5", roman: "V", title: "第五部分 工具箱", short: "工具箱", description: "数据训练、特征可视化、训练监控、超参搜索、部署和测验系统。" },
  { key: "part6", roman: "VI", title: "第六部分 统一框架与前沿", short: "框架与前沿", description: "统一接口、模块化结构、项目模板、学习路径、术语表、AI 智能体、推理模型和前沿方向。" },
  { key: "part7", roman: "VII", title: "第七部分 CS面试八股", short: "面试八股", description: "计算机网络、数据库、数据结构、操作系统、系统设计面试训练。" },
];

const MODULES = [
  ["part1", "part1_foundations", "张量与梯度", "01_tensors_gradients", "用可视化理解张量、自动求导和梯度传播。", "入门", ["基础", "张量", "梯度"]],
  ["part1", "part1_foundations", "激活与归一化", "02_activations_normalization", "比较常见激活函数、归一化方法和训练稳定性。", "入门", ["基础", "激活函数", "归一化"]],
  ["part1", "part1_foundations", "数据集与优化器", "03_datasets_optimizers", "理解数据划分、批训练、SGD、Adam 和优化曲线。", "入门", ["基础", "数据", "优化"]],
  ["part1", "part1_foundations", "数学基础速查", "math_primer", "线性代数、微积分、概率论和梯度下降的交互式速查。", "入门", ["基础", "数学", "可视化"]],
  ["part1", "part1_foundations", "机器学习基础", "machine_learning_basics", "监督学习、损失函数、泛化、评估和模型选择。", "入门", ["基础", "机器学习"]],
  ["part1", "part1_foundations", "神经网络基础", "neural_network_basics", "从感知机到多层网络，理解反向传播和非线性表达。", "入门", ["基础", "神经网络"]],
  ["part1", "part1_foundations", "经典机器学习", "classical_ml", "用传统模型建立深度学习前的基线意识。", "入门", ["基础", "模型", "基线"]],
  ["part2", "part2_cnn", "卷积直觉", "01_convolution_visual", "用滑窗、卷积核和边缘检测建立 CNN 直觉。", "进阶", ["视觉", "CNN", "卷积"]],
  ["part2", "part2_cnn", "特征图可视化", "02_feature_maps", "观察卷积层如何从局部纹理逐步形成抽象特征。", "进阶", ["视觉", "CNN", "可视化"]],
  ["part2", "part2_cnn", "经典 CNN 架构", "03_classic_architectures", "梳理 LeNet、AlexNet、VGG、GoogLeNet 和 ResNet。", "进阶", ["视觉", "CNN", "架构"]],
  ["part2", "part2_cnn", "CNN 调试面板", "04_debug_panel", "定位卷积模型训练中的过拟合、梯度和数据问题。", "工程", ["视觉", "调试", "训练"]],
  ["part2", "part2_cnn", "MNIST 玩具实验", "05_mnist_toy", "用小型手写数字实验串起数据、模型、训练和评估。", "实验", ["视觉", "CNN", "实验"]],
  ["part2", "part2_cnn", "现代 CNN 架构", "06_modern_architectures", "理解残差、深度可分离卷积和高效视觉网络。", "进阶", ["视觉", "CNN", "架构"]],
  ["part2", "part2_cnn", "高级卷积技术", "07_advanced_convolution", "扩张卷积、转置卷积、分组卷积和感受野分析。", "进阶", ["视觉", "CNN", "卷积"]],
  ["part2", "part2_cnn", "Grad-CAM 可视化", "08_visualization_gradcam", "用热力图解释 CNN 决策关注区域。", "实验", ["视觉", "解释性", "可视化"]],
  ["part2", "part2_cnn", "迁移学习", "09_transfer_learning", "复用预训练模型完成小数据任务。", "工程", ["视觉", "迁移学习"]],
  ["part2", "part2_cnn", "CNN 架构实验", "cnn_architectures", "对比经典卷积网络的结构与特征提取方式。", "进阶", ["视觉", "CNN"]],
  ["part2", "part2_cnn", "高级 CNN", "advanced_cnn", "现代卷积技巧、残差思想和视觉模型设计。", "进阶", ["视觉", "CNN"]],
  ["part3", "part3_rnn", "RNN 直觉", "01_rnn_intuition", "从循环状态理解序列信息如何流动。", "进阶", ["序列", "RNN"]],
  ["part3", "part3_rnn", "隐藏状态", "02_hidden_states", "观察隐藏状态、门控结构和长期依赖。", "进阶", ["序列", "RNN", "可视化"]],
  ["part3", "part3_rnn", "序列玩具任务", "03_sequence_toys", "用可控任务理解记忆、预测和序列泛化。", "实验", ["序列", "RNN", "实验"]],
  ["part3", "part3_rnn", "RNN 超参实验", "04_hyperparam_rnn", "比较学习率、隐藏维度、层数和截断反传。", "实验", ["序列", "训练", "超参数"]],
  ["part3", "part3_rnn", "Seq2Seq 与注意力", "05_seq2seq_attention", "理解编码器解码器和注意力对齐。", "核心", ["序列", "注意力", "NLP"]],
  ["part3", "part3_rnn", "文本分类", "06_text_classification", "用序列模型完成文本表示和分类。", "实验", ["序列", "NLP", "分类"]],
  ["part3", "part3_rnn", "高级训练技巧", "07_advanced_training", "处理梯度裁剪、Teacher Forcing、正则化和训练稳定性。", "工程", ["序列", "训练", "调试"]],
  ["part3", "part3_rnn", "RNN 调试问题", "08_debug_problems", "定位序列模型中的梯度、数据和评估问题。", "工程", ["序列", "调试"]],
  ["part3", "part3_rnn", "序列模型", "sequence_models", "RNN、LSTM、GRU 与序列任务的基本范式。", "进阶", ["序列", "NLP"]],
  ["part4", "part4_transformer", "注意力机制", "01_attention_mechanism", "从查询、键、值理解注意力权重。", "核心", ["Transformer", "注意力"]],
  ["part4", "part4_transformer", "多头注意力可视化", "02_multihead_visual", "观察不同注意力头如何捕获互补关系。", "核心", ["Transformer", "注意力", "可视化"]],
  ["part4", "part4_transformer", "编码器与解码器", "03_encoder_decoder", "拆解 Transformer 编码器、解码器和掩码机制。", "核心", ["Transformer", "NLP"]],
  ["part4", "part4_transformer", "最小 Transformer", "04_minimal_transformer", "用精简实现串起嵌入、注意力、MLP 和残差。", "核心", ["Transformer", "实现"]],
  ["part4", "part4_transformer", "Flash Attention", "05_flash_attention", "理解高效注意力的内存访问与计算优化。", "前沿", ["Transformer", "性能", "注意力"]],
  ["part4", "part4_transformer", "Transformer 调试", "06_debug_problems", "分析大模型训练中的掩码、位置编码和梯度问题。", "工程", ["Transformer", "调试"]],
  ["part4", "part4_transformer", "Transformer 架构", "transformer_models", "可视化拆解自注意力、多头、位置编码和 BERT/GPT。", "核心", ["Transformer", "NLP"]],
  ["part4", "part4_transformer", "GAN 与自编码器", "gan_ae", "理解生成模型、潜空间和重构学习。", "进阶", ["生成模型", "表征"]],
  ["part4", "part4_transformer", "图神经网络", "gnn_intro", "从节点、边和消息传递理解图学习。", "进阶", ["GNN", "结构数据"]],
  ["part5", "part5_toolbox", "特征可视化", "01_feature_visualization", "观察特征、激活、嵌入和决策边界。", "工程", ["工具", "可视化", "解释性"]],
  ["part5", "part5_toolbox", "梯度监控", "02_gradient_monitor", "监控梯度范数、爆炸、消失和训练健康度。", "工程", ["工具", "梯度", "调试"]],
  ["part5", "part5_toolbox", "训练动态", "03_training_dynamics", "用曲线和指标追踪模型如何学习。", "工程", ["训练", "监控"]],
  ["part5", "part5_toolbox", "超参搜索", "04_hyperparam_search", "比较网格搜索、随机搜索和实验记录。", "工程", ["超参数", "工具"]],
  ["part5", "part5_toolbox", "玩具数据集", "05_dataset_toys", "用小数据集快速验证模型直觉。", "实验", ["数据", "实验"]],
  ["part5", "part5_toolbox", "数据与训练", "data_training", "数据管线、训练循环、指标与调试。", "工程", ["训练", "数据"]],
  ["part5", "part5_toolbox", "案例研究", "case_studies", "用完整案例串联建模、调参和诊断流程。", "工程", ["案例", "实践"]],
  ["part5", "part5_toolbox", "部署工具", "deployment_tools", "模型导出、服务化、推理和工程落地。", "工程", ["部署", "工具"]],
  ["part5", "part5_toolbox", "练习题与测验", "quiz_system", "覆盖机器学习基础、CNN、RNN、Transformer 和 GAN 的交互式测验。", "复习", ["测验", "复习"]],
  ["part5", "part5_toolbox", "调参实战挑战", "tuning_challenge", "在真实约束下练习学习率、正则、模型规模和实验记录决策。", "实验", ["调参", "实验", "诊断"]],
  ["part6", "part6_universal_framework", "统一接口", "01_unified_interface", "把模型、数据和任务抽象成统一可扩展接口。", "工程", ["框架", "架构"]],
  ["part6", "part6_universal_framework", "模块化结构", "02_modular_structure", "拆分配置、数据、模型、训练和评估边界。", "工程", ["框架", "模块化"]],
  ["part6", "part6_universal_framework", "完整项目骨架", "03_full_project", "组织可复用的深度学习项目目录和执行流程。", "工程", ["项目", "架构"]],
  ["part6", "part6_universal_framework", "插件系统", "04_plugin_system", "用注册表和插件扩展任务、模型与工具。", "工程", ["框架", "插件"]],
  ["part6", "part6_universal_framework", "一键训练", "05_one_click_training", "从配置到训练、评估和产物保存的一键流程。", "工程", ["训练", "自动化"]],
  ["part6", "part6_universal_framework", "可视化实验台", "06_streamlit_demo", "原生 HTML/JS 实验入口，Python 源码仅作为实现对照。", "核心", ["实验", "可视化", "HTML"]],
  ["part6", "part6_universal_framework", "神经网络乐高工厂", "neural_network_playground", "用表单构建神经网络、形状推导、代码生成、示例模型加载。", "核心", ["构建器", "Playground", "实战"]],
  ["part6", "part6_universal_framework", "训练过程可视化", "training_demo", "用轻量数据集演示训练循环，实时展示损失、准确率、梯度范数。", "核心", ["训练", "可视化", "演示"]],
  ["part6", "part6_universal_framework", "项目模板", "07_project_template", "训练脚本、评估脚本、K-Fold 和集成预测模板。", "工程", ["项目", "模板"]],
  ["part6", "part6_universal_framework", "强化学习入门", "reinforcement_learning", "强化学习概念、多臂老虎机、Q-Learning 和纯 Python 环境 demo。", "核心", ["RL", "强化学习", "实验"]],
  ["part6", "part6_universal_framework", "学习路径推荐", "learning_path", "入门测评、个性化路径、知识图谱、进度追踪和下一步推荐。", "核心", ["路径", "知识图谱", "测评"]],
  ["part6", "part6_universal_framework", "深度学习术语表", "glossary", "集中检索常见概念、缩写和相关模块。", "复习", ["术语", "搜索", "复习"]],
  ["part6", "part6_universal_framework", "前沿方向", "frontier", "LLM、AGI、多模态、AI 智能体、推理模型、自监督、XAI、安全与对齐。", "前沿", ["LLM", "AGI", "Agents", "安全"]],
  ["part6", "part6_universal_framework", "经典论文解读实验室", "paper_reading_lab", "用时间线、机制图和最小复现清单读懂经典深度学习论文。", "进阶", ["论文", "可视化", "复现"]],
  ["part7", "part7_interview", "计算机网络", "networking", "TCP握手挥手、HTTP/HTTPS、DNS解析、高频面试题与交互练习。", "核心", ["网络", "TCP", "HTTP", "面试"]],
  ["part7", "part7_interview", "数据库SQL", "database_sql", "SELECT执行流程、B+树索引、慢查询排查、高频面试题与交互练习。", "核心", ["数据库", "SQL", "索引", "面试"]],
  ["part7", "part7_interview", "数据结构与算法", "data_structures", "数组链表可视化、排序算法动画、BFS/DFS、高频面试题。", "核心", ["数据结构", "算法", "排序", "面试"]],
  ["part7", "part7_interview", "操作系统", "operating_system", "进程线程、调度算法、虚拟内存、死锁、高频面试题。", "核心", ["操作系统", "进程", "内存", "面试"]],
  ["part7", "part7_interview", "系统设计", "system_design", "CAP定理、缓存三兄弟、消息队列、推荐架构、分布式训练、推理平台设计与交互练习。", "核心", ["系统设计", "架构", "面试", "分布式"]],
  ["part7", "part7_interview", "深度学习", "deep_learning_interview", "梯度消失可视化、BatchNorm vs LayerNorm、注意力复杂度、训练排查清单、LoRA微调、模型部署。", "核心", ["深度学习", "梯度", "归一化", "面试"]],
  ["part7", "part7_interview", "自测刷题模式", "interview_quiz", "随机出题、按方向难度筛选、错题本、延伸追问。", "核心", ["刷题", "自测", "错题本"]],
].map(([partKey, partDir, title, module, summary, level, tags], index) => {
  const part = PARTS.find((item) => item.key === partKey);
  return {
    id: `${partKey}/${module}`,
    partKey,
    partDir,
    partTitle: part.title,
    partShort: part.short,
    title,
    module,
    summary,
    level,
    tags,
    index,
    sourcePath: `${partDir}/${module}.py`,
  };
});

const app = document.querySelector("#app");
const drawer = document.querySelector("[data-drawer]");
const scrim = document.querySelector(".drawer-scrim");
const drawerSearch = document.querySelector("#drawer-search");
let activePart = "all";
let activeGoal = "整体理解";
let motionObserver = null;
const LEARNING_PROGRESS_KEY = "deep-learning-book-progress-v1";
const LEARNING_MODE_KEY = "deep-learning-book-mode-v1";

const SOURCE_LIBRARY = {
  DLBOOK: {
    title: "Deep Learning",
    authors: "Goodfellow, Bengio, Courville",
    url: "https://www.deeplearningbook.org/",
  },
  D2L: {
    title: "Dive into Deep Learning",
    authors: "Zhang, Lipton, Li, Smola",
    url: "https://d2l.ai/",
  },
  CS231N: {
    title: "CS231n: Convolutional Neural Networks for Visual Recognition",
    authors: "Stanford",
    url: "https://cs231n.github.io/",
  },
  CS224N: {
    title: "CS224n: Natural Language Processing with Deep Learning",
    authors: "Stanford",
    url: "https://web.stanford.edu/class/cs224n/",
  },
  PYTORCH_SDPA: {
    title: "torch.nn.functional.scaled_dot_product_attention",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
  },
  PYTORCH_AUTOGRAD: {
    title: "Autograd mechanics",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/notes/autograd.html",
  },
  PYTORCH_NN: {
    title: "torch.nn API",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/nn.html",
  },
  PYTORCH_TRANSFORMER: {
    title: "torch.nn.Transformer",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html",
  },
  PYTORCH_RNN: {
    title: "Recurrent layers: RNN / LSTM / GRU",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/nn.html#recurrent-layers",
  },
  PYTORCH_ONNX: {
    title: "torch.onnx",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/onnx.html",
  },
  PYTORCH_QUANT: {
    title: "Quantization",
    authors: "PyTorch Docs",
    url: "https://pytorch.org/docs/stable/quantization.html",
  },
  SKLEARN: {
    title: "scikit-learn User Guide",
    authors: "scikit-learn",
    url: "https://scikit-learn.org/stable/user_guide.html",
  },
  MLCC: {
    title: "Machine Learning Crash Course",
    authors: "Google Developers",
    url: "https://developers.google.com/machine-learning/crash-course",
  },
  VAS2017: {
    title: "Attention Is All You Need",
    authors: "Vaswani et al., 2017",
    url: "https://arxiv.org/abs/1706.03762",
  },
  BAH2015: {
    title: "Neural Machine Translation by Jointly Learning to Align and Translate",
    authors: "Bahdanau, Cho, Bengio, 2015",
    url: "https://arxiv.org/abs/1409.0473",
  },
  DAO2022: {
    title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    authors: "Dao et al., 2022",
    url: "https://arxiv.org/abs/2205.14135",
  },
  LENET1998: {
    title: "Gradient-Based Learning Applied to Document Recognition",
    authors: "LeCun et al., 1998",
    url: "http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf",
  },
  ALEX2012: {
    title: "ImageNet Classification with Deep Convolutional Neural Networks",
    authors: "Krizhevsky, Sutskever, Hinton, 2012",
    url: "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks",
  },
  VGG2014: {
    title: "Very Deep Convolutional Networks for Large-Scale Image Recognition",
    authors: "Simonyan, Zisserman, 2014",
    url: "https://arxiv.org/abs/1409.1556",
  },
  HE2015: {
    title: "Deep Residual Learning for Image Recognition",
    authors: "He et al., 2015",
    url: "https://arxiv.org/abs/1512.03385",
  },
  MOBILENET2017: {
    title: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
    authors: "Howard et al., 2017",
    url: "https://arxiv.org/abs/1704.04861",
  },
  EFFICIENTNET2019: {
    title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    authors: "Tan, Le, 2019",
    url: "https://arxiv.org/abs/1905.11946",
  },
  GRADCAM2017: {
    title: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
    authors: "Selvaraju et al., 2017",
    url: "https://arxiv.org/abs/1610.02391",
  },
  BN2015: {
    title: "Batch Normalization",
    authors: "Ioffe, Szegedy, 2015",
    url: "https://arxiv.org/abs/1502.03167",
  },
  DROPOUT2014: {
    title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    authors: "Srivastava et al., 2014",
    url: "https://www.jmlr.org/papers/v15/srivastava14a.html",
  },
  ADAM2014: {
    title: "Adam: A Method for Stochastic Optimization",
    authors: "Kingma, Ba, 2014",
    url: "https://arxiv.org/abs/1412.6980",
  },
  LSTM1997: {
    title: "Long Short-Term Memory",
    authors: "Hochreiter, Schmidhuber, 1997",
    url: "https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory",
  },
  CHO2014: {
    title: "Learning Phrase Representations using RNN Encoder-Decoder",
    authors: "Cho et al., 2014",
    url: "https://arxiv.org/abs/1406.1078",
  },
  SUTS2014: {
    title: "Sequence to Sequence Learning with Neural Networks",
    authors: "Sutskever, Vinyals, Le, 2014",
    url: "https://arxiv.org/abs/1409.3215",
  },
  MDN_HTTP: {
    title: "HTTP",
    authors: "MDN Web Docs",
    url: "https://developer.mozilla.org/en-US/docs/Web/HTTP",
  },
  POSTGRES: {
    title: "PostgreSQL Documentation",
    authors: "PostgreSQL",
    url: "https://www.postgresql.org/docs/",
  },
  OSTEP: {
    title: "Operating Systems: Three Easy Pieces",
    authors: "Remzi H. Arpaci-Dusseau, Andrea C. Arpaci-Dusseau",
    url: "https://pages.cs.wisc.edu/~remzi/OSTEP/",
  },
  CP_ALGORITHMS: {
    title: "Algorithms for Competitive Programming",
    authors: "cp-algorithms",
    url: "https://cp-algorithms.com/",
  },
};

const CONTENT_CREDIBILITY = {
  "part4/01_attention_mechanism": {
    level: "A",
    label: "已校对样板",
    summary: "公式、历史脉络和 PyTorch API 已对照公开来源。页面中的热力图仍是教学演示，不等价于完整因果解释。",
    boundaries: [
      "Q/K/V 与缩放点积公式以 Transformer 原论文和 PyTorch scaled_dot_product_attention 文档为基准。",
      "页面中的随机张量和热力图用于说明形状、归一化和 mask，不代表真实训练好的模型权重。",
      "注意力权重只能提供线索；残差、MLP、LayerNorm 和后续层也会继续改写表示。",
    ],
    sources: ["VAS2017", "BAH2015", "PYTORCH_SDPA", "D2L"],
  },
  "part4/transformer_models": {
    level: "B",
    label: "教学简化",
    summary: "页面覆盖自注意力、多头、位置编码和 BERT/GPT 对比，但部分热力图和多头模式是教学化生成。",
    boundaries: [
      "适合建立机制直觉，不适合作为真实模型解释结论。",
      "BERT/GPT 对比需要继续补论文和官方实现来源。",
    ],
    sources: ["VAS2017", "PYTORCH_TRANSFORMER", "D2L"],
  },
  default: {
    level: "C",
    label: "待复核",
    summary: "这一页已完成教学结构整理，但还没有逐条对照公开来源校订。",
    boundaries: [
      "请把页面内容当作学习笔记和交互演示，关键结论建议再对照公开教材、论文或官方文档。",
      "旧脚本迁移内容可能包含简化实现、模拟数据或早期讲义口吻。",
    ],
    sources: ["D2L", "DLBOOK"],
  },
};

const CREDIBILITY_PROFILES = {
  foundation: {
    level: "B",
    label: "教学简化",
    summary: "基础章节主要对照开放教材、PyTorch 文档和机器学习公开课程；公式和演示会保留教学化简化。",
    boundaries: [
      "数学图形用于建立方向、尺度、概率和梯度直觉，不替代严格数学教材。",
      "自动求导、优化器和归一化的 API 细节以 PyTorch 官方文档为准。",
      "经典机器学习与数据划分以 scikit-learn 文档和开放课程为主要校对来源。",
    ],
    sources: ["D2L", "DLBOOK", "PYTORCH_AUTOGRAD", "PYTORCH_NN", "SKLEARN", "MLCC"],
  },
  cnn: {
    level: "B",
    label: "教学简化",
    summary: "CNN 章节对照 CS231n、经典 CNN 论文和 PyTorch API；可视化图主要用于解释局部连接、特征图和结构取舍。",
    boundaries: [
      "卷积、池化和特征图演示使用小图或合成数据，不能直接代表真实视觉模型的全部行为。",
      "架构年份、结构名称和核心思想会参考原论文；简化实现不等价于论文完整模型。",
      "Grad-CAM 与特征可视化只能作为解释线索，不是完整因果证明。",
    ],
    sources: ["CS231N", "PYTORCH_NN", "LENET1998", "ALEX2012", "VGG2014", "HE2015", "GRADCAM2017"],
  },
  rnn: {
    level: "B",
    label: "教学简化",
    summary: "RNN 章节对照 D2L、PyTorch 循环层文档和 LSTM/GRU/Seq2Seq 经典论文；动画强调记忆流和梯度直觉。",
    boundaries: [
      "隐藏状态和门控动画是低维解释图，真实模型内部表示更高维、更难直接解释。",
      "玩具序列任务用于暴露记忆、截断和梯度问题，不代表真实 NLP 数据集效果。",
      "Teacher Forcing、采样和注意力对齐需要结合训练/推理口径区分。",
    ],
    sources: ["D2L", "PYTORCH_RNN", "LSTM1997", "CHO2014", "SUTS2014", "BAH2015"],
  },
  transformer: {
    level: "B",
    label: "教学简化",
    summary: "Transformer 章节对照原论文、PyTorch 文档、D2L 和 FlashAttention 论文；部分热力图是教学化可视化。",
    boundaries: [
      "标准注意力公式和 mask 机制以论文与 PyTorch API 为准。",
      "多头注意力图、位置编码图和 BERT/GPT 对比有教学化抽象，不等价于真实大模型内部解释。",
      "长上下文、FlashAttention 和性能结论需要结合硬件、kernel、序列长度和实现版本判断。",
    ],
    sources: ["VAS2017", "BAH2015", "DAO2022", "PYTORCH_SDPA", "PYTORCH_TRANSFORMER", "D2L"],
  },
  toolbox: {
    level: "B",
    label: "教学简化",
    summary: "工具箱章节对照 PyTorch、scikit-learn、解释性论文和部署文档；实验结果多为教学模拟或轻量数据。",
    boundaries: [
      "训练曲线、调参挑战和案例研究多为可控演示，不应当作真实数据集基准。",
      "部署、量化、ONNX 和解释性内容必须结合目标硬件、模型结构和数据分布复核。",
      "解释性图像只提供证据，不是模型因果决策的完整证明。",
    ],
    sources: ["PYTORCH_NN", "PYTORCH_ONNX", "PYTORCH_QUANT", "SKLEARN", "GRADCAM2017", "ADAM2014"],
  },
  framework: {
    level: "B",
    label: "工程实践整理",
    summary: "统一框架与中央控制台章节主要是本站项目经验和 PyTorch 工程模式整理，不是通用工业标准。",
    boundaries: [
      "模型构建器和训练联动是教学项目实现，不能等同于成熟深度学习框架。",
      "项目结构、插件系统和一键训练是本站当前选择，真实项目要按团队、数据和部署约束调整。",
      "生成代码和 shape 诊断用于学习和原型验证，上线前仍需要独立测试。",
    ],
    sources: ["PYTORCH_NN", "PYTORCH_AUTOGRAD", "PYTORCH_ONNX", "D2L", "MLCC"],
  },
  cs: {
    level: "C",
    label: "待复核",
    summary: "CS 基础训练营是复习和自测入口，目前优先保证交互路径，细节还需要继续对照官方文档和教材校订。",
    boundaries: [
      "网络、数据库、操作系统和算法解释以入门复习为目标，不替代系统教材。",
      "题目答案需要继续补官方文档、RFC 或经典教材依据。",
      "系统设计题没有唯一标准答案，应按约束、取舍和故障模式讨论。",
    ],
    sources: ["MDN_HTTP", "POSTGRES", "OSTEP", "CP_ALGORITHMS", "SKLEARN"],
  },
};

const images = [
  "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=900&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=900&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=900&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=900&auto=format&fit=crop",
];

const LLM_COOKBOOK_TRACKS = [
  {
    title: "Prompt Engineering",
    route: "part4/transformer_models",
    stage: "表达任务",
    summary: "把任务、上下文、约束、输出格式和少样本示例写清楚，让模型先知道要解决什么问题。",
    observe: "看输出是否稳定遵循格式，是否遗漏边界条件，是否把示例当成要复制的答案。",
    practice: "把同一个问题写成零样本、少样本、结构化输出三版 prompt，对比答案可控性。",
    workflow: ["先写任务边界：目标、输入、禁止事项、输出格式。", "再加一到两个正例，覆盖最容易误解的格式。", "最后做反例测试：缺字段、歧义输入、超长输入和恶意指令。"],
    failure: "最常见的失败不是“模型不聪明”，而是 prompt 把角色、任务、证据和格式混在一起，模型只能猜你的优先级。",
    acceptance: "同一批测试题连续运行时，字段完整、语气稳定、边界拒答一致，且修改一个约束不会破坏其他约束。",
    drill: "在 Transformer 页观察 attention，再把“解释注意力机制”写成教学版、面试版、JSON 版三种 prompt。",
    concept: "Prompt 是把人类意图压缩成模型可执行的任务接口；它决定模型先看什么、忽略什么、以什么格式交付。",
    checklist: ["是否写清楚输入来自哪里、用户能改什么、模型不能做什么。", "是否给出可自动检查的输出结构，而不是只写“回答得专业一点”。", "是否用边界样例测试了歧义、越权、缺字段和长文本。"],
  },
  {
    title: "Chat System",
    route: "part7/system_design",
    stage: "组织对话",
    summary: "把系统指令、用户消息、历史状态、工具结果和最终回答拆开管理，而不是把所有文本直接拼成一段。",
    observe: "检查上下文窗口、历史截断、角色边界、拒答策略和多轮状态是否互相污染。",
    practice: "画出一次对话请求链路：入口、消息构造、模型调用、后处理、日志和异常兜底。",
    workflow: ["把 system、developer、user、tool result 分层保存。", "为历史消息做摘要或滑动窗口，不把全部聊天记录无脑塞回模型。", "对最终输出做格式校验、引用校验和安全边界检查。"],
    failure: "多轮系统容易在历史里积累旧指令、脏工具结果和用户越权请求；如果没有状态边界，越聊越不可控。",
    acceptance: "用户改口、补充资料、撤销需求时，系统能解释当前采用哪条上下文，并能在日志里还原一次回答的来源。",
    drill: "在系统设计页把一次问答拆成入口、上下文构造、模型调用、后处理、观测日志五个节点。",
    concept: "Chat System 是把一次模型调用放进真实学习链路：身份、状态、历史、工具、审核和结果呈现都要分层管理。",
    checklist: ["是否能区分系统规则、用户意图、检索证据和工具返回。", "是否有历史截断策略，避免旧上下文污染新任务。", "是否记录每次回答使用了哪些消息、工具结果和后处理规则。"],
  },
  {
    title: "RAG 问答",
    route: "part6/frontier",
    stage: "接入私有知识",
    summary: "用检索把外部文档变成可引用上下文，再让模型基于证据回答，降低只靠参数记忆带来的幻觉。",
    observe: "重点看切块粒度、召回质量、上下文排序、引用证据和答案是否忠于原文。",
    practice: "把一个课程章节切成 chunk，写出 query -> top-k 文档 -> answer -> citation 的最小流程。",
    workflow: ["先清洗文档，保留标题层级、来源、更新时间和段落边界。", "按语义而不是固定字数切块，再为每块写可追踪 metadata。", "回答时要求先找证据、再合成答案、最后列出引用和不确定点。"],
    failure: "RAG 的失败经常发生在检索阶段：切块太碎会丢上下文，切块太大会稀释命中，top-k 太高会把噪声送进模型。",
    acceptance: "答案中的关键结论都能回到原文片段；问到资料没有覆盖的问题时，系统会说明缺证据，而不是编一个像真的答案。",
    drill: "把本站任意一页知识点索引当作文档库，设计三条问题，手工标注每条问题应该召回哪些段落。",
    concept: "RAG 的核心不是“把文档塞给模型”，而是先把证据找准，再让模型在证据约束下组织答案。",
    checklist: ["每个 chunk 是否带标题、来源、章节、更新时间和可回溯 ID。", "top-k 命中是否真的覆盖问题需要的事实，而不是只有关键词相似。", "答案是否明确引用证据，并在证据不足时停止编造。"],
  },
  {
    title: "Embedding & Search",
    route: "part5/01_feature_visualization",
    stage: "语义检索",
    summary: "用向量表示文本含义，通过相似度找相关片段；它是 RAG、推荐和去重的底层能力。",
    observe: "观察查询向量是否召回语义相关而非只匹配字面词，检查误召回和漏召回样例。",
    practice: "拿两个相近问题和一个干扰问题，比较它们在向量检索结果中的排名变化。",
    workflow: ["先准备查询集：同义问法、反向问法、干扰问法都要有。", "比较关键词、向量和混合检索三种召回结果。", "对错误召回做标签：词面相似、语义漂移、粒度不匹配或领域词缺失。"],
    failure: "向量检索不是魔法。短查询、专有名词、否定句和数值条件都可能让相似度看起来很高但语义其实错了。",
    acceptance: "核心问题的 top-3 命中稳定，干扰问题不会压过正确文档，召回结果能解释为什么相关。",
    drill: "在特征可视化页把“图像特征”类比成“文本向量”，观察相似表示如何服务后续分类或检索。",
    concept: "Embedding 把文本放进语义空间，检索就是在这个空间里找近邻；好坏取决于表示、切块和排序共同作用。",
    checklist: ["是否准备了同义、反义、干扰、长查询和短查询测试集。", "是否比较关键词检索、向量检索和混合检索的差异。", "是否把误召回样例沉淀成规则、重排或文档切块调整。"],
  },
  {
    title: "Evaluation & Debugging",
    route: "part5/03_training_dynamics",
    stage: "评估调试",
    summary: "LLM 应用不能只看一次回答，要建立测试集、失败样例、人工评分标准和回归检查。",
    observe: "看准确性、相关性、事实性、格式遵循、安全边界和延迟成本是否同时被记录。",
    practice: "为一个 RAG 问答写 5 条测试问题，记录标准答案、引用片段、失败类型和修复动作。",
    workflow: ["先收集真实失败样例，不只写理想问题。", "把评估拆成事实正确、引用充分、格式合规、拒答边界、延迟成本。", "每次改 prompt、检索或模型后跑同一套回归题。"],
    failure: "只看平均分会掩盖致命样例；一个医疗、金融或代码生成边界错误，可能比十个普通问题答得好更重要。",
    acceptance: "每个失败都有类别、复现输入、期望输出和修复归因；修复后能证明不是靠硬编码通过单题。",
    drill: "在训练动态页把 loss/validation gap 的思维迁移到 LLM：不仅看回答好不好，还看泛化、回归和失败分布。",
    concept: "评估是 LLM 应用的仪表盘：它把主观“感觉不错”变成可复现的样例、指标、失败类型和修复记录。",
    checklist: ["是否有固定测试集、真实失败集和上线前回归集。", "是否分开记录事实错误、引用错误、格式错误、安全错误和成本延迟。", "是否能说明每次改动改善了哪类失败、是否引入新退化。"],
  },
  {
    title: "Fine-tuning / LoRA",
    route: "part7/deep_learning_interview",
    stage: "定制能力",
    summary: "当 prompt 和 RAG 仍不能稳定改变行为时，再考虑监督微调、LoRA 或偏好数据。",
    observe: "先判断问题是知识缺口、格式不稳、风格不一致还是能力不足，避免把所有问题都交给微调。",
    practice: "写一张决策表：Prompt、RAG、LoRA、全量微调分别适合什么数据量和失败类型。",
    workflow: ["先做失败归因：知识问题优先 RAG，格式问题优先 prompt，稳定风格才考虑微调。", "准备高质量输入输出对，清理重复、冲突和低可信样本。", "用保留集检查过拟合、灾难性遗忘和边界拒答是否变差。"],
    failure: "用脏数据微调会把坏答案固化进模型；数据量小、标签不一致时，LoRA 也可能只是更稳定地犯错。",
    acceptance: "微调后目标风格或任务通过率提升，同时通用能力、安全边界和未见题表现没有明显退化。",
    drill: "在深度学习面试页把微调问题拆成数据、参数、目标函数、验证集和部署成本五个面试追问。",
    concept: "微调是改变模型行为分布，不是知识库替代品；它适合稳定风格、格式、策略和特定任务模式。",
    checklist: ["是否先证明 prompt 与 RAG 已经不能解决主要失败。", "训练样本是否一致、去重、可追溯，并覆盖负例和边界例。", "是否用保留集检查目标任务提升和通用能力退化。"],
  },
  {
    title: "Agent & Tools",
    route: "part6/frontier",
    stage: "工具调用",
    summary: "让模型选择工具、读取结果、继续规划，但必须给工具权限、输入格式、错误恢复和执行边界。",
    observe: "检查工具描述是否精确，参数是否可验证，失败是否能重试，日志是否能还原每一步决策。",
    practice: "设计一个“查资料 -> 生成答案 -> 自检引用”的三步 Agent，并写出每步失败兜底。",
    workflow: ["先定义工具 schema：名称、参数、返回值、权限和副作用。", "让模型每一步先说明意图，再调用工具，再根据结果决定是否继续。", "为超时、空结果、冲突结果和权限不足写停止条件。"],
    failure: "Agent 最危险的不是不会调用工具，而是把工具结果当成绝对真理，或在失败后无限重试、越权执行。",
    acceptance: "一次任务的计划、工具输入、工具输出、下一步决策都能审计；失败时能停在可解释状态。",
    drill: "在前沿方向页把 Agent 看成一个有控制流的系统：规划、执行、观察、修正，每步都要有边界。",
    concept: "Agent 是带工具和状态的控制循环；模型负责选择下一步，但系统必须负责权限、审计和停止条件。",
    checklist: ["每个工具是否有清晰 schema、权限边界、副作用说明和错误码。", "是否限制最大步数、重试次数、外部写操作和敏感数据访问。", "是否能回放完整轨迹：为什么调用、传了什么、得到什么、下一步为何如此。"],
  },
  {
    title: "Gradio / App Delivery",
    route: "part5/deployment_tools",
    stage: "应用交付",
    summary: "把模型能力包装成可交互应用，关注输入控件、输出解释、加载状态、成本提示和错误信息。",
    observe: "看用户是否知道等待什么、失败时怎么办、哪些输入会产生不可靠结果。",
    practice: "把一个课程实验改成最小 Web Demo：输入、运行、输出、错误提示、示例用例五件套。",
    workflow: ["先固定最小闭环：输入、运行、输出、解释、错误提示。", "再补样例、加载态、取消/重试、成本或 token 提示。", "最后用真实用户路径检查首屏、移动端、长文本和失败状态。"],
    failure: "Demo 只要缺少加载态和错误解释，用户就会把模型延迟误判成页面卡死，把模型不确定误判成系统坏了。",
    acceptance: "用户不用读说明也能完成一次任务；失败时知道原因、可重试动作和哪些输入需要调整。",
    drill: "在部署工具页把任意一个实验包装成产品化界面，列出学生第一次使用时会卡住的三个点。",
    concept: "应用交付是把模型能力变成用户能完成任务的界面；控件、反馈、失败状态和成本提示都属于模型体验。",
    checklist: ["是否有默认样例、加载状态、取消/重试入口和清晰错误提示。", "是否解释输出可信度、引用来源、限制条件和下一步动作。", "是否在移动端、长文本、慢响应和失败请求下都能保持可用。"],
  },
];

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function kickRouteMotion() {
  document.body.classList.add("is-motion-ready");
  app.classList.remove("is-route-entering");
  void app.offsetWidth;
  app.classList.add("is-route-entering");
  window.setTimeout(() => app.classList.remove("is-route-entering"), 260);
}

function isElementInRevealWindow(element) {
  const rect = element.getBoundingClientRect();
  return rect.top < window.innerHeight * 0.94 && rect.bottom > 0;
}

function ensureMotionObserver() {
  if (motionObserver) return motionObserver;
  motionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      motionObserver.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  return motionObserver;
}

function applyMotionReveal(root = app, options = {}) {
  if (!root) return;

  const reset = options.reset ?? root === app;
  if (reset && motionObserver) {
    motionObserver.disconnect();
    motionObserver = null;
  }

  const selectors = [
    ".hero .kicker",
    ".hero h1",
    ".hero p",
    ".hero-actions",
    ".hero-stats",
    ".hero-media",
    ".starter-card",
    ".onboarding-note",
    ".section-head",
    ".course-card",
    ".path-panel",
    ".search-band",
    ".catalog-tabs",
    ".module-card",
    ".stat-card",
    ".course-topline",
    ".course-article > .eyebrow",
    ".course-article > h1",
    ".course-article > .summary",
    ".course-article > .tag-row",
    ".student-route-card",
    ".course-progress-list",
    ".course-next-actions",
    ".course-console-cta",
    ".mode-switcher",
    ".reading-section",
    ".lesson-card",
    ".zero-basics-card",
    ".concept-demo",
    ".knowledge-section",
    ".practice-callout",
    ".note-item",
    ".interactive-lab",
    ".course-aside",
    ".console-hero",
    ".console-task-strip",
    ".console-purpose-strip",
    ".console-context",
    ".console-workbench",
    ".console-panel",
    ".console-chip-list span",
    ".console-steps li",
    ".code-toolbar",
    ".code-window",
  ];
  const elements = [...root.querySelectorAll(selectors.join(","))]
    .filter((element) => !element.closest(".side-drawer"));

  elements.forEach((element, index) => {
    element.classList.add("motion-reveal");
    element.style.setProperty("--motion-delay", `${Math.min(index * 42, 260)}ms`);
    if (reset || !element.classList.contains("is-visible")) {
      element.classList.remove("is-visible");
    }
  });

  if (prefersReducedMotion()) {
    elements.forEach((element) => element.classList.add("is-visible"));
    return;
  }

  const observer = ensureMotionObserver();
  elements.forEach((element) => {
    if (isElementInRevealWindow(element)) {
      element.classList.add("is-visible");
    } else {
      observer.observe(element);
    }
  });
}

function pulseLabReadout(lab) {
  const readout = lab.querySelector("[data-lab-readout]");
  if (!readout || prefersReducedMotion()) return;
  readout.classList.remove("is-updating");
  void readout.offsetWidth;
  readout.classList.add("is-updating");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function moduleHref(module) {
  return `#course/${encodeURIComponent(module.id)}`;
}

function consoleHref(module) {
  return `#console/${encodeURIComponent(module.id)}`;
}

function byId(id) {
  return MODULES.find((module) => module.id === id);
}

function readLearningProgress() {
  try {
    const parsed = JSON.parse(localStorage.getItem(LEARNING_PROGRESS_KEY) || "{}");
    return {
      understood: Array.isArray(parsed.understood) ? parsed.understood : [],
      review: Array.isArray(parsed.review) ? parsed.review : [],
    };
  } catch (error) {
    return { understood: [], review: [] };
  }
}

function writeLearningProgress(progress) {
  try {
    localStorage.setItem(LEARNING_PROGRESS_KEY, JSON.stringify({
      understood: [...new Set(progress.understood || [])],
      review: [...new Set(progress.review || [])],
      updatedAt: new Date().toISOString(),
    }));
  } catch (error) {
    // localStorage may be unavailable in strict browser modes; the page should still teach.
  }
}

function markLearningProgress(moduleId, kind) {
  const progress = readLearningProgress();
  const list = kind === "review" ? progress.review : progress.understood;
  if (!list.includes(moduleId)) list.push(moduleId);
  writeLearningProgress(progress);
  return progress;
}

function progressSummary() {
  const progress = readLearningProgress();
  const understood = new Set(progress.understood);
  const review = new Set(progress.review);
  const next = MODULES.find((module) => !understood.has(module.id)) || MODULES[0];
  return {
    understood: understood.size,
    review: review.size,
    next,
    percent: Math.round((understood.size / MODULES.length) * 100),
  };
}

function adjacentModules(module) {
  const samePart = MODULES.filter((item) => item.partKey === module.partKey);
  const index = samePart.findIndex((item) => item.id === module.id);
  return {
    previous: index > 0 ? samePart[index - 1] : null,
    next: index >= 0 && index < samePart.length - 1 ? samePart[index + 1] : null,
  };
}

function readLearningMode() {
  try {
    return localStorage.getItem(LEARNING_MODE_KEY) === "advanced" ? "advanced" : "beginner";
  } catch (error) {
    return "beginner";
  }
}

function writeLearningMode(mode) {
  const next = mode === "advanced" ? "advanced" : "beginner";
  document.body.dataset.learningMode = next;
  try {
    localStorage.setItem(LEARNING_MODE_KEY, next);
  } catch (error) {
    // Mode is a comfort setting; keep the page usable if storage is blocked.
  }
  return next;
}

function applyLearningMode(mode = readLearningMode()) {
  document.body.dataset.learningMode = mode;
  document.querySelectorAll("[data-learning-mode]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.learningMode === mode);
  });
}

function tagHtml(tags, options = {}) {
  const interactive = options.interactive ?? false;
  return `<div class="tag-row">${tags.map((tag) => interactive
    ? `<button class="tag tag-button" type="button" data-tag-filter="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`
    : `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>`;
}

function moduleLearningPlan(module) {
  const profile = getLessonProfile(module);
  const levelMinutes = {
    入门: "20-30 分钟",
    核心: "35-50 分钟",
    进阶: "40-60 分钟",
    实验: "30-45 分钟",
    工程: "35-55 分钟",
    复习: "15-25 分钟",
    前沿: "45-70 分钟",
  };
  const prereqByDomain = {
    foundation: "四则运算、函数图像、能看懂表格",
    cnn: "张量形状、矩阵乘法、训练/验证的区别",
    sequence: "张量与梯度、文本 token、基础神经网络",
    transformer: "向量相似度、softmax、序列 token",
    training: "损失函数、学习率、训练/验证曲线",
    systems: "函数接口、模块拆分、一次请求的输入输出",
    interview: "对应方向的基本术语和一两个真实例子",
  };
  return {
    duration: levelMinutes[module.level] || "30-45 分钟",
    prereq: prereqByDomain[profile.domain] || "本章前一节 + 页面里的默认实验",
    completion: `能解释 ${profile.signal}，并写出“我调了什么 -> 画面哪里变 -> 为什么”。`,
    review: `隔天回到动画区，只改一个参数复现 ${profile.focus} 的变化。`,
  };
}

function scrollToHashTarget(hash, delay = 60) {
  const selector = hash && hash.startsWith("#") ? hash : `#${hash}`;
  window.setTimeout(() => {
    try {
      document.querySelector(selector)?.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    } catch (error) {
      // Invalid hashes should not break routing.
    }
  }, delay);
}

const DOMAIN_PROFILES = {
  foundation: {
    demoKind: "gradient",
    label: "基础建模",
    thesis: "先把对象、形状、参数和损失放进同一条训练链路里，后面的模型才不会变成黑箱。",
    mechanism: "核心机制是：输入被表示成向量或张量，参数把它变换成预测，损失把预测错误压成标量，梯度再把误差信号传回参数。",
    steps: ["确认输入和输出的形状", "写出前向计算", "观察损失如何变化", "解释参数为什么这样更新"],
    pitfalls: ["只背公式，不检查 shape 是否能相乘。", "把梯度当成结果，而不是局部变化方向。", "忽略尺度，导致激活、logits 或梯度过大。"],
  },
  cnn: {
    demoKind: "convolution",
    label: "视觉表征",
    thesis: "CNN 的重点不是把图片缩小，而是让局部模式在空间上被反复检测、组合和压缩。",
    mechanism: "卷积核像一组可学习的探测器：浅层响应边缘和纹理，中层组合局部形状，深层把空间线索压成更抽象的类别证据。",
    steps: ["看卷积核在局部窗口内读到了什么", "看输出特征图哪里变亮", "比较 stride、padding、通道数对形状的影响", "判断亮区是否真的对应任务证据"],
    pitfalls: ["把特征图当成原图复制品。", "只看准确率，不看输入归一化、padding 和通道响应。", "误以为更深一定更好，忽略感受野和数据规模。"],
  },
  sequence: {
    demoKind: "sequence",
    label: "序列记忆",
    thesis: "序列模型的关键是让历史信息以可控方式进入当前状态，而不是把每个 token 独立处理。",
    mechanism: "RNN/LSTM/GRU 会反复更新隐藏状态；门控决定保留多少旧信息、写入多少新信息，注意力则允许模型直接回看关键位置。",
    steps: ["明确当前 token 看到哪些历史", "观察隐藏状态如何累积或遗忘", "检查长序列里梯度和记忆是否衰减", "用注意力或门控解释预测来源"],
    pitfalls: ["只看最后输出，不看中间状态。", "把序列长度调大却不检查梯度和截断反传。", "用平均池化掩盖关键 token 的位置关系。"],
  },
  transformer: {
    demoKind: "attention",
    label: "注意力机制",
    thesis: "Transformer 的主线是让每个 token 主动检索上下文，而不是按时间一步一步传递记忆。",
    mechanism: "Query 负责提出问题，Key 负责被匹配，Value 负责被汇总；多头注意力让不同子空间同时关注语义、位置和结构线索。",
    steps: ["选定一个 query token", "看它对所有 key 的打分", "观察 softmax 后权重是否过尖或过散", "解释 value 汇总后改变了什么表示"],
    pitfalls: ["把注意力权重直接等同于解释。", "忽略 mask、位置编码和尺度缩放。", "只看单头结果，不比较多头分工。"],
  },
  training: {
    demoKind: "training",
    label: "训练诊断",
    thesis: "训练页的价值是建立诊断顺序：先看数据和目标，再看损失曲线、梯度、学习率和验证集。",
    mechanism: "损失下降说明优化在推进，验证集走势说明泛化是否稳定；梯度范数、更新幅度和学习率共同决定模型是在学习、震荡还是停滞。",
    steps: ["先确认数据切分和标签质量", "观察训练/验证曲线是否分叉", "检查梯度是否爆炸或消失", "一次只改一个超参数并记录结果"],
    pitfalls: ["看到 loss 降就以为模型一定变好。", "同时改学习率、正则和模型大小，无法归因。", "没有实验记录，复现不了最佳结果。"],
  },
  architecture: {
    demoKind: "architecture",
    label: "工程架构",
    thesis: "框架与项目页要回答边界问题：哪些东西应该稳定，哪些东西应该被配置、插件或脚本替换。",
    mechanism: "好的工程结构会把数据、模型、训练、评估、产物和实验记录拆成清晰接口，让新任务可以替换局部而不是复制整套代码。",
    steps: ["划清数据、模型、训练和评估边界", "把可变项放进配置", "用注册表或插件管理扩展点", "保留可复现实验产物"],
    pitfalls: ["目录很多但边界不清。", "配置和代码互相泄漏。", "只追求一键运行，却没有日志、版本和产物管理。"],
  },
  systems: {
    demoKind: "systems",
    label: "面试与系统",
    thesis: "面试知识点不是背答案，而是把约束、取舍、流程和失败模式讲清楚。",
    mechanism: "网络、数据库、操作系统和系统设计题都可以拆成：请求从哪里来、经过哪些层、瓶颈在哪里、失败后如何恢复。",
    steps: ["画出流程而不是直接背定义", "指出关键数据结构或协议状态", "比较至少两个方案的代价", "给出排查顺序和边界条件"],
    pitfalls: ["只背结论，不解释触发条件。", "忽略一致性、延迟、吞吐和资源隔离的取舍。", "没有把答案连接到深度学习训练或推理场景。"],
  },
};

const DOMAIN_BLUEPRINTS = {
  foundation: {
    variable: "张量形状、参数尺度、损失值和梯度方向",
    signal: "shape 是否闭合、loss 是否下降、梯度是否有合理量级",
    transfer: "写 PyTorch 代码前，先用纸面检查输入维度、输出维度和每一步反向传播的来源。",
    lab: "调学习率、初始点和迭代次数，看优化路径怎样收敛、震荡或走偏。",
  },
  cnn: {
    variable: "卷积核、感受野、通道响应、stride、padding 和特征图亮区",
    signal: "输出特征图是否突出任务相关区域，而不是只复制原图纹理",
    transfer: "做视觉项目时，先检查输入归一化和浅层特征，再谈网络深度或迁移学习。",
    lab: "换输入图案和卷积核，看局部窗口怎样变成边缘、纹理或类别证据。",
  },
  sequence: {
    variable: "序列长度、隐藏状态、门控保留率、截断反传和关键 token 位置",
    signal: "模型是否记住了真正需要跨步保留的信息，而不是只依赖最后几个 token",
    transfer: "处理文本或时间序列时，把问题拆成：什么必须记住、什么应该遗忘、预测来自哪一步。",
    lab: "调序列长度、记忆保留率和输入噪声，看隐藏状态何时稳定、衰减或被噪声冲掉。",
  },
  transformer: {
    variable: "query/key/value、注意力锐度、mask、位置编码和多头分工",
    signal: "权重是否集中到有用上下文，value 汇总后是否改变了当前 token 表示",
    transfer: "读大模型结构时，始终追问：当前 token 在问谁、拿回什么、哪些位置被 mask 掉。",
    lab: "调 query、锐度和噪声，看注意力从平均浏览变成集中检索。",
  },
  training: {
    variable: "学习率、正则、数据噪声、训练/验证曲线、梯度范数和实验记录",
    signal: "训练集和验证集是否同步改善，还是出现震荡、过拟合或梯度异常",
    transfer: "调参时一次只改一个因素，并把失败样本当作诊断证据，而不是只保留最好结果。",
    lab: "调学习率、正则强度和数据噪声，观察训练曲线、验证曲线和梯度健康度如何变化。",
  },
  architecture: {
    variable: "配置、数据接口、模型注册表、训练器、产物、日志和复现实验边界",
    signal: "新增任务时是否只替换局部组件，而不需要复制整套项目",
    transfer: "做工程落地时，优先稳定接口和产物约定，再追求一键运行和漂亮封装。",
    lab: "调模块耦合度、插件化程度和记录完整度，看项目从脚本堆变成可复用系统。",
  },
  systems: {
    variable: "请求路径、协议状态、缓存命中、队列积压、失败模式和资源隔离",
    signal: "答案是否能解释瓶颈在哪里、失败如何传播、该先排查哪一层",
    transfer: "面试或部署时，把抽象概念落到一次模型推理请求：入口、网络、服务、模型、存储。",
    lab: "调请求负载、缓存命中率和故障位置，看端到端延迟与排查顺序怎样变化。",
  },
};

const BEGINNER_BLUEPRINTS = {
  foundation: {
    analogy: "像整理书桌：先知道每本书的大小和位置，再决定怎么移动它们，最后看桌面是否更整齐。",
    intuition: "深度学习先把对象变成数字，再用损失和梯度告诉参数往哪边改。",
    definition: "基础建模关注张量表示、前向计算、损失函数与反向传播之间的闭环关系。",
    elements: "点表示当前参数或数据状态，曲线表示损失变化路径，坐标轴表示不同变量或参数方向。",
    visualEncoding: "颜色越深通常代表响应或损失越强，方向表示更新方向，速度表示参数改变得快慢。",
    knobs: "优先调学习率、起点、迭代步数和输入形状相关参数。",
    observe: "看 loss 是否下降、轨迹是否震荡、shape 是否能对齐、梯度是否过大或过小。",
    why: "梯度给出局部最快变化方向；学习率决定每次沿这个方向走多远。",
    misuse: "把公式背下来却不检查 shape，或者看到数值变化就以为模型一定学对了。",
    engineering: "写训练脚本、排查 NaN、选择优化器和读模型日志时都离不开这套基础链路。",
    consoleTask: "在中央控制台里搭一个最小网络，改学习率和层宽，观察损失曲线与梯度读数。",
  },
  cnn: {
    analogy: "像拿不同滤镜看照片：有的滤镜找边缘，有的滤镜找纹理，有的滤镜压掉无关细节。",
    intuition: "卷积不是直接看整张图，而是一小块一小块找局部证据。",
    definition: "CNN 用共享卷积核在空间上滑动，把局部像素模式映射成特征图，再逐层组合成高层语义。",
    elements: "输入网格表示图像局部区域，卷积核表示探测器，特征图表示每个位置的响应强弱。",
    visualEncoding: "亮区代表响应强，暗区代表响应弱；扫描方向代表卷积核移动；边界变化代表 padding 或 stride 的影响。",
    knobs: "优先调输入图案、卷积核类型、输入强度、stride/padding 或通道数。",
    observe: "看哪些位置变亮，输出尺寸是否改变，边缘/纹理/形状是否被正确突出。",
    why: "卷积核与局部窗口越匹配，乘加结果越大，对应位置在特征图中就越亮。",
    misuse: "把特征图当原图复印件，或者只看准确率不看输入归一化和局部响应。",
    engineering: "图像分类、检测、迁移学习、Grad-CAM 解释和视觉模型调试都依赖这套读图方式。",
    consoleTask: "在中央控制台里选择 CNN/视觉预设，调卷积层数和通道数，看特征响应与训练曲线怎么变。",
  },
  sequence: {
    analogy: "像读一句长句子：你不能只记最后一个词，要保留前文真正影响当前含义的信息。",
    intuition: "序列模型的核心是决定哪些历史要记住，哪些历史可以忘掉。",
    definition: "RNN/LSTM/GRU 通过隐藏状态或门控机制把时间步信息递推到当前预测。",
    elements: "token 表示时间步，记忆柱表示隐藏状态强度，状态卡表示最终记忆、长程保留和噪声风险。",
    visualEncoding: "柱子越高代表状态影响越强，变淡代表记忆衰减，流动方向代表时间顺序。",
    knobs: "优先调序列长度、记忆保留率、输入噪声、隐藏维度和截断反传长度。",
    observe: "看前面 token 的影响是否还能保留，后面噪声是否覆盖了真正关键的信息。",
    why: "隐藏状态会反复乘上保留系数，长序列中信息容易衰减；门控和注意力能缓解这个问题。",
    misuse: "只看最后输出，不看中间状态，或者把序列长度调大却不检查梯度和记忆衰减。",
    engineering: "文本分类、时间序列预测、语音、日志序列和 Seq2Seq 任务都要先处理记忆路径。",
    consoleTask: "在中央控制台里切到序列/训练实验，调隐藏维度和序列长度，观察记忆柱和损失曲线。",
  },
  transformer: {
    analogy: "像开卷考试：每个词都可以带着问题去全文里找最相关的词，而不是只听前一个词传话。",
    intuition: "注意力让当前 token 主动检索上下文。",
    definition: "Transformer 用 Query-Key 匹配得到注意力权重，再对 Value 加权汇总，形成新的 token 表示。",
    elements: "Query 是提问者，Key 是可匹配线索，Value 是被取回的信息，权重条表示当前 token 看向谁。",
    visualEncoding: "条形越长代表权重越大，颜色越醒目代表关注越强，锐度越高代表分布越集中。",
    knobs: "优先调 query token、注意力锐度、上下文噪声、mask 和多头数量。",
    observe: "看权重是集中到有用上下文，还是平均分散或被噪声带偏。",
    why: "softmax 会把相似度分数变成概率分布；锐度越高，高分位置越容易主导汇总结果。",
    misuse: "把注意力权重直接当最终解释，或者忽略 mask、位置编码和多头分工。",
    engineering: "读 BERT/GPT、调长上下文、排查 mask 泄露和理解 Flash Attention 都要用这套视角。",
    consoleTask: "在中央控制台里选择注意力/Transformer 预设，调 query 和头数，看权重分布与输出变化。",
  },
  training: {
    analogy: "像练习投篮：不只看进了几个，还要看动作是否稳定、训练和比赛表现是否同时变好。",
    intuition: "训练诊断不是看 loss 降没降，而是看模型为什么这样学。",
    definition: "训练诊断用训练/验证曲线、梯度范数、学习率、正则和数据噪声判断优化与泛化状态。",
    elements: "黑线表示训练损失，金线表示验证损失，指标卡表示间隙、梯度健康和末端表现。",
    visualEncoding: "曲线下降代表优化推进，曲线分叉代表泛化风险，点的抖动代表噪声或不稳定。",
    knobs: "优先调学习率、正则强度、数据噪声、batch size 和训练轮数。",
    observe: "看训练线和验证线是否同步下降，梯度健康度是否变差，曲线是否震荡或停滞。",
    why: "学习率控制更新幅度，正则限制模型复杂度，数据噪声会改变可学习信号和泛化难度。",
    misuse: "同时改多个超参数导致无法归因，或者只保存最好结果不记录失败实验。",
    engineering: "模型调参、实验复现、过拟合排查、部署前验证和训练平台监控都依赖训练诊断。",
    consoleTask: "在中央控制台里运行训练可视化，调学习率和正则，记录训练/验证间隙的变化。",
  },
  architecture: {
    analogy: "像搭积木城市：道路、水电和接口先规划好，后面换楼、扩街区才不会拆掉全城。",
    intuition: "工程架构的核心是边界清楚：什么稳定，什么可替换。",
    definition: "工程架构通过配置、接口、注册表、训练器、日志和产物约定组织可复用的深度学习项目。",
    elements: "节点表示配置、数据、模型、训练器和产物；状态卡表示替换成本、可复现性和扩展能力。",
    visualEncoding: "激活节点代表边界健康，风险节点代表耦合过高；指标越高代表对应能力越强。",
    knobs: "优先调模块耦合度、插件化程度、记录完整度、配置项和产物保存策略。",
    observe: "看新增任务是否只需要替换局部组件，日志和产物是否足够复现实验。",
    why: "低耦合减少连锁修改，插件化提供扩展入口，完整记录让实验可以回放和比较。",
    misuse: "目录很多但边界不清，或者只有一键运行，没有配置、日志、版本和产物约定。",
    engineering: "课程项目、比赛模板、训练平台、模型服务和团队协作都需要这套结构。",
    consoleTask: "在中央控制台里从空白配置组装一个模型流程，检查数据、模型、训练、产物边界是否清楚。",
  },
  systems: {
    analogy: "像查快递：先看包裹经过哪些站点，再判断卡在哪一站，而不是直接背公司规则。",
    intuition: "系统题要先画链路，再谈结论。",
    definition: "系统与面试知识把请求路径、协议状态、存储、缓存、资源和失败模式组织成可排查流程。",
    elements: "节点表示 client、gateway、cache、database、model 等层级；指标卡表示延迟、瓶颈和第一检查项。",
    visualEncoding: "高亮节点表示当前链路，风险节点表示故障位置，延迟数值越高代表体验越差。",
    knobs: "优先调请求负载、缓存命中率、故障位置、并发量和资源限制。",
    observe: "看瓶颈转移到哪一层，端到端延迟如何变化，第一排查步骤是否随故障改变。",
    why: "负载会放大慢节点，缓存能减少后端压力，单点故障会沿调用链传播。",
    misuse: "只背答案不讲触发条件，或者忽略一致性、延迟、吞吐和隔离之间的取舍。",
    engineering: "推理服务、训练平台、数据库调优、网络排障和系统设计面试都需要链路思维。",
    consoleTask: "在中央控制台里把模型推理当成一次系统请求，调负载和缓存，写出瓶颈与排查顺序。",
  },
};

const LAB_CONTROL_GUIDES = {
  foundation: {
    controls: "起点 x、起点 y、学习率、迭代步数",
    changes: "轨迹是否靠近谷底、loss 是否下降、路径是否震荡或绕远",
  },
  cnn: {
    controls: "输入图案、卷积核、输入强度",
    changes: "特征图哪些格子变亮、边缘/纹理是否被突出、输出响应最大值是否变化",
  },
  sequence: {
    controls: "序列长度、记忆保留率、输入噪声",
    changes: "记忆柱是否衰减、关键 token 是否还能影响末尾状态、噪声风险是否升高",
  },
  transformer: {
    controls: "Query token、注意力锐度、上下文噪声",
    changes: "权重条是否集中到相关 token、注意力是否过尖、噪声是否把关注带偏",
  },
  training: {
    controls: "学习率、正则强度、数据噪声",
    changes: "训练线和验证线是否分叉、梯度健康度是否下降、曲线是否震荡",
  },
  architecture: {
    controls: "模块耦合度、插件化程度、记录完整度",
    changes: "替换成本、可复现性、扩展能力三张状态卡怎样变化",
  },
  systems: {
    controls: "请求负载、缓存命中率、故障位置",
    changes: "端到端延迟、瓶颈层、第一排查步骤是否发生切换",
  },
};

const MODULE_TEACHING_NOTES = {
  "part1/01_tensors_gradients": {
    what: "这页讲的是张量如何承载数据、参数如何参与计算、梯度如何把错误从损失函数一路传回去。",
    analogy: "像在仓库里按货架编号找货，再根据差错单回溯是哪一层货架摆错了。",
    intuition: "张量负责装数据，梯度负责指路，优化器负责真的迈出那一步。",
    variable: "张量 shape、requires_grad、loss 标量、参数 grad 和学习率",
    elements: "矩阵格子代表张量里的数，箭头代表计算依赖，损失点代表当前错误，梯度箭头代表参数该改的方向。",
    controls: "张量维度、初始权重、学习率、迭代步数和是否开启梯度记录",
    observe: "shape 是否能闭合，loss 是否下降，梯度方向是否把参数推向更小错误，是否出现 NaN 或梯度为 0",
    why: "自动求导把每个操作的局部导数按链式法则乘起来，所以后面的错误能分配回前面的参数。",
    misconception: "梯度不是最终答案，也不是越大越好；它只是当前点附近最敏感的变化方向。",
    engineering: "排查 shape mismatch、loss 不降、NaN、参数没更新时，第一步都要看张量和梯度链路。",
    consoleTask: "在中央控制台搭一个最小多层网络，先只改学习率，再看 loss、梯度范数和输出形状是否一起变化。",
  },
  "part1/02_activations_normalization": {
    what: "这页讲激活函数怎样给网络加入非线性，归一化怎样让不同层的数值尺度更稳定。",
    analogy: "像调音台：激活决定哪些声音放大或压低，归一化先把音量拉回可控范围。",
    intuition: "激活让网络能弯曲决策边界，归一化让训练不被过大或过小的数值拖垮。",
    variable: "激活输出范围、均值方差、梯度斜率、BatchNorm/LayerNorm 的统计量",
    elements: "函数曲线代表输入到输出的映射，平坦区代表梯度容易变小，统计卡代表归一化前后的均值和方差。",
    controls: "激活类型、输入尺度、batch size、归一化方式和是否使用仿射参数",
    observe: "输出是否饱和，梯度是否消失，归一化后分布是否回到稳定区间，训练曲线是否更平滑",
    why: "非线性改变局部斜率，归一化改变数据尺度；尺度稳定时，优化器更容易持续沿有用方向更新。",
    misconception: "ReLU 不是永远安全，输入长期为负会死掉；归一化也不是越多越好，位置放错会破坏表示。",
    engineering: "模型训练震荡、梯度消失、不同 batch 表现差异大时，经常要回到激活和归一化检查。",
    consoleTask: "在中央控制台切换 ReLU、Sigmoid、Tanh 和归一化开关，对比损失曲线、梯度健康度和输出分布。",
  },
  "part1/03_datasets_optimizers": {
    what: "这页讲数据如何被切成训练/验证/测试，批训练如何喂给模型，优化器如何根据梯度更新参数。",
    analogy: "像备考刷题：训练集是练习题，验证集是模拟考，测试集是最终考试，优化器是调整复习策略的人。",
    intuition: "数据决定模型看见什么，优化器决定每次看到错误后怎么改。",
    variable: "数据划分比例、batch size、学习率、动量、Adam 一阶二阶矩和验证指标",
    elements: "数据块代表不同 split，曲线代表训练过程，更新箭头代表 SGD 或 Adam 对参数的移动。",
    controls: "训练验证比例、batch size、学习率、优化器类型、shuffle 和训练轮数",
    observe: "训练 loss 是否下降，验证 loss 是否同步改善，小 batch 是否抖动，Adam 是否更快但可能泛化不同",
    why: "不同数据切分改变评估信号，不同优化器改变更新方向的平滑方式和步长自适应方式。",
    misconception: "验证集不能反复当测试集用；Adam 收敛快不等于最终一定更泛化。",
    engineering: "训练脚本、实验复现、模型选择和线上前验收，都依赖正确的数据划分和优化器记录。",
    consoleTask: "在中央控制台固定模型，只改 batch size、学习率和优化器，对比训练/验证间隙。",
  },
  "part1/math_primer": {
    what: "这页把线性代数、微积分、概率和梯度下降压成深度学习真正会用到的数学工具箱。",
    analogy: "像出门前看地图、指南针和天气：不必背完整地理学，但要知道每个工具什么时候能指路。",
    intuition: "矩阵管形状和变换，导数管变化方向，概率管不确定性，梯度下降管怎么变好。",
    variable: "向量维度、矩阵乘法、导数符号、概率分布、损失曲面和梯度方向",
    elements: "向量箭头代表方向和长度，矩阵网格代表线性变换，曲面代表损失，轨迹点代表优化过程。",
    controls: "矩阵尺寸、向量方向、函数斜率、学习率、起点和迭代次数",
    observe: "矩阵能否相乘，导数正负是否对应上升下降，概率是否归一化，优化轨迹是否接近低点",
    why: "深度学习把大量计算都写成矩阵组合，再用导数衡量局部变化，因此数学工具会直接决定代码是否能跑通。",
    misconception: "数学基础不是刷公式数量，而是能把公式翻译成 shape、方向、尺度和不确定性。",
    engineering: "读论文公式、排查维度错误、理解优化曲线和解释模型输出概率时都会用到这一页。",
    consoleTask: "在中央控制台先用最小网络做一次形状推导，再把学习率调到过小和过大，观察梯度下降轨迹。",
  },
  "part1/machine_learning_basics": {
    what: "这页讲监督学习的基本闭环：数据给输入和标签，模型给预测，损失衡量错误，评估判断泛化。",
    analogy: "像请家教批作业：学生先答题，老师指出错在哪，再用新题检查是不是只背了答案。",
    intuition: "机器学习不是记住训练集，而是从训练样本里学到能迁移到新样本的规则。",
    variable: "特征、标签、模型假设、损失函数、训练/验证指标和泛化间隙",
    elements: "散点代表样本，边界代表模型决策，曲线代表误差变化，指标卡代表准确率、召回率或损失。",
    controls: "模型复杂度、训练集大小、正则强度、评估指标和数据噪声",
    observe: "训练指标和验证指标是否一起变好，边界是否过度贴合噪声，错误样本是否有共同模式",
    why: "模型容量越大越容易记住训练细节，正则和验证集帮助你判断学到的是规律还是噪声。",
    misconception: "训练准确率高不等于模型好；指标也不能脱离业务代价单独看。",
    engineering: "建模前的基线、指标选择、过拟合判断和模型上线验收，都来自这套基本框架。",
    consoleTask: "在中央控制台固定数据，改变模型层宽和正则，记录训练准确率与验证准确率是否分叉。",
  },
  "part1/neural_network_basics": {
    what: "这页讲神经网络如何把多个线性变换和非线性激活叠起来，形成比单层模型更强的表达能力。",
    analogy: "像流水线加工：每一层只做一小步变换，多层叠加后能加工出复杂形状。",
    intuition: "一层负责简单变换，多层负责逐步组合特征，反向传播负责告诉每层该怎么改。",
    variable: "层数、隐藏单元、权重矩阵、激活函数、loss 和反向传播梯度",
    elements: "节点代表神经元，连线代表权重，层代表表示阶段，输出条代表模型对类别或数值的预测。",
    controls: "隐藏层数量、每层宽度、激活函数、学习率和训练轮数",
    observe: "输出是否从随机变得稳定，隐藏层是否有非线性表达，梯度是否能传回前面层",
    why: "线性层只会做线性组合，加入激活后，多层网络能拼出弯曲的函数或复杂决策边界。",
    misconception: "神经元不是大脑细胞的真实模拟；层数更多也不自动意味着更好。",
    engineering: "设计 MLP、读模型结构、调 hidden size、排查反向传播断开时都要先理解这一页。",
    consoleTask: "在中央控制台从一层网络开始，逐步加宽和加深，看参数量、loss 和决策能力如何变化。",
  },
  "part1/classical_ml": {
    what: "这页用线性模型、树模型、KNN、SVM 等传统方法建立深度学习之前的基线意识。",
    analogy: "像先用直尺、剪刀和模板解决问题，再决定是否真的需要复杂机器。",
    intuition: "经典模型是判断任务难度和数据质量的参照物，不是过时知识。",
    variable: "特征工程、模型假设、决策边界、正则项、距离度量和交叉验证结果",
    elements: "样本点代表数据，分割线或树节点代表规则，指标表代表不同模型的基线表现。",
    controls: "特征缩放、模型类型、正则强度、树深度、邻居数和交叉验证折数",
    observe: "简单模型是否已经足够，错误集中在哪类样本，复杂模型是否只是提升训练分数",
    why: "很多任务的主要瓶颈在特征和数据，而不是模型深度；强基线能暴露深度模型是否真的带来增益。",
    misconception: "上来就用深度模型可能掩盖数据泄漏、特征错误和评估设计问题。",
    engineering: "竞赛、业务建模和论文复现都需要先跑传统基线，作为后续深度模型的对照。",
    consoleTask: "在中央控制台用同一数据先跑小模型，再加深网络，对比提升是否超过基线误差波动。",
  },
  "part2/01_convolution_visual": {
    what: "这页讲卷积核怎样在图像上滑动，用局部窗口的乘加结果检测边缘、纹理和简单形状。",
    analogy: "像拿一个小印章在图片上逐格比对，哪里图案吻合，哪里就留下更深的印记。",
    intuition: "卷积是在整张图上重复使用同一个局部探测器。",
    variable: "输入窗口、卷积核权重、stride、padding、输出特征图和响应最大值",
    elements: "左侧网格是输入图像，中间小矩阵是卷积核，右侧亮格是对应位置的响应。",
    controls: "输入图案、卷积核类型、输入强度、stride 和 padding",
    observe: "卷积核移动到哪里响应最高，边缘是否被突出，输出尺寸是否随 stride/padding 改变",
    why: "局部窗口与卷积核越相似，逐项相乘再求和的值越大，特征图对应位置就越亮。",
    misconception: "卷积核不是裁剪图片，也不是固定只能找边缘；训练中它会学习最有用的局部模式。",
    engineering: "读 CNN 第一层、设计输入分辨率、解释浅层特征和排查 padding 错误都离不开卷积直觉。",
    consoleTask: "在中央控制台选择视觉预设，只改卷积核和 stride，记录响应图和输出形状怎样变化。",
  },
  "part2/02_feature_maps": {
    what: "这页讲特征图如何把一张图中不同位置的局部证据展开，让你看见网络每层关注什么。",
    analogy: "像多位侦探同时看同一张照片：有人找边缘，有人找纹理，有人找局部形状。",
    intuition: "特征图越亮，说明该位置越像当前通道正在寻找的模式。",
    variable: "通道数、激活强度、层深、感受野、特征图亮区和类别证据",
    elements: "每张小图是一个通道，亮区是强响应，层级切换代表从低级纹理走向高级语义。",
    controls: "层级、通道、输入图案、激活阈值和可视化归一化方式",
    observe: "浅层是否看纹理，中层是否组合形状，深层是否更贴近任务相关区域",
    why: "卷积层把局部模式逐层组合，越深的层感受野越大，也越容易形成抽象特征。",
    misconception: "特征图不是原图副本，亮也不一定代表人眼认为重要，必须结合任务和类别判断。",
    engineering: "诊断 CNN 是否学到背景捷径、解释错误分类和选择迁移学习冻结层时会用到特征图。",
    consoleTask: "在中央控制台对比浅层和深层特征响应，写下哪个通道更像边缘、纹理或类别线索。",
  },
  "part2/03_classic_architectures": {
    what: "这页梳理 LeNet、AlexNet、VGG、GoogLeNet、ResNet 如何一步步改变 CNN 的深度、宽度和连接方式。",
    analogy: "像看建筑史：同样是楼，结构材料和承重方式变了，能盖的高度也变了。",
    intuition: "经典架构的核心差异，是如何在更深网络里保持可训练和高效表达。",
    variable: "卷积层堆叠、池化、Inception 分支、残差连接、参数量和计算量",
    elements: "模块块状图代表网络阶段，分支代表并行特征提取，跳连代表信息绕过若干层直接相加。",
    controls: "架构类型、层数、卷积核大小、是否使用残差和通道宽度",
    observe: "参数量如何变化，特征分辨率在哪些阶段下降，残差连接是否缩短梯度路径",
    why: "更深网络有更强表达力，但梯度传播更难；残差和多分支结构是在优化难度和特征多样性之间取平衡。",
    misconception: "背架构年份没有意义，必须能说出每个结构解决了什么训练或表达问题。",
    engineering: "选 backbone、读视觉论文、做模型压缩和迁移学习时，经典架构是重要参照系。",
    consoleTask: "在中央控制台切换不同视觉骨架，比较参数量、层级结构和训练曲线的差异。",
  },
  "part2/04_debug_panel": {
    what: "这页讲如何从数据、特征、梯度、过拟合和错误样本几个入口调试 CNN。",
    analogy: "像修相机：画面糊了不一定是镜头坏，也可能是光线、对焦、传感器或后期参数出问题。",
    intuition: "CNN 调试先看输入和错误样本，再看曲线和梯度，最后才盲目换结构。",
    variable: "输入归一化、训练/验证间隙、梯度范数、类别混淆、错误热区和学习率",
    elements: "曲线是训练状态，样本缩略图是数据证据，指标卡是故障线索，热区是模型关注位置。",
    controls: "学习率、数据增强强度、正则、batch size、冻结层和错误类别筛选",
    observe: "是否过拟合，错误样本是否来自同一类噪声，梯度是否异常，模型是否看错区域",
    why: "视觉模型很容易学到背景、尺度和数据偏差，调试必须把数值曲线和图像证据放在一起看。",
    misconception: "准确率低不一定是模型小，可能是标签、归一化、增强或 train/eval 模式错了。",
    engineering: "真实视觉项目上线前，必须用这套面板排查数据质量、训练稳定性和解释可靠性。",
    consoleTask: "在中央控制台保持架构不变，只改增强或学习率，观察验证间隙和错误样本是否变化。",
  },
  "part2/05_mnist_toy": {
    what: "这页用 MNIST 手写数字把数据读取、CNN 前向、训练循环、评估和错误分析串成最小闭环。",
    analogy: "像用儿童积木搭一座小桥：规模小，但桥墩、梁和受力关系都能看清。",
    intuition: "MNIST 的价值不是难度，而是让完整训练流程足够小、足够可观察。",
    variable: "输入尺寸、卷积层、学习率、epoch、准确率、混淆类别和错误样本",
    elements: "手写数字是输入，卷积层是特征提取器，曲线是学习过程，混淆矩阵显示哪些数字容易混。",
    controls: "网络层数、通道数、学习率、batch size、epoch 和是否加入数据增强",
    observe: "loss 是否快速下降，准确率是否接近稳定，哪些数字对最容易混淆，错误是否可解释",
    why: "数字任务结构清晰、噪声较少，所以模型能快速学到笔画和形状特征，也适合验证训练代码是否正确。",
    misconception: "MNIST 跑通不代表真实视觉任务也简单，它只能证明训练管线基本可用。",
    engineering: "新框架、新 GPU、新训练脚本上线前，常用小数据集先验证端到端链路。",
    consoleTask: "在中央控制台用小型 CNN 训练几轮，记录学习率变化对准确率和混淆样本的影响。",
  },
  "part2/06_modern_architectures": {
    what: "这页讲残差、深度可分离卷积、高效网络和现代 CNN 如何在精度、速度、参数量之间取舍。",
    analogy: "像设计高铁线路：不只是跑得快，还要省能耗、少换乘、维修方便。",
    intuition: "现代 CNN 的重点是让信息走得更顺、计算花得更值。",
    variable: "残差路径、瓶颈层、深度可分离卷积、通道扩展、FLOPs 和参数量",
    elements: "主干块代表特征变换，跳连代表保留原信息，窄宽结构代表压缩和扩展通道。",
    controls: "模型宽度、深度、是否启用残差、是否使用深度可分离卷积和输入分辨率",
    observe: "参数量是否下降，训练是否更稳，特征分辨率如何变化，速度和精度是否出现取舍",
    why: "残差缓解深层优化困难，深度可分离卷积分开空间和通道计算，减少不必要的乘加。",
    misconception: "轻量模型不是简单砍层；如果通道和分辨率分配不当，速度快但表示会崩。",
    engineering: "移动端视觉、实时检测、边缘部署和 backbone 选择都需要理解现代 CNN 取舍。",
    consoleTask: "在中央控制台比较普通卷积和轻量卷积预设，看参数量、训练速度和验证表现如何变。",
  },
  "part2/07_advanced_convolution": {
    what: "这页讲扩张卷积、转置卷积、分组卷积等高级卷积如何改变感受野、上采样和计算分工。",
    analogy: "像换不同网眼的筛子：有的看得更远，有的把图放大，有的把任务分给不同小组。",
    intuition: "高级卷积不是新魔法，而是在控制看多大、怎么算、输出多密。",
    variable: "dilation、groups、kernel size、stride、输出尺寸、感受野和棋盘伪影",
    elements: "空洞点表示跳采样位置，分组块表示通道拆分，转置卷积箭头表示从低分辨率回到高分辨率。",
    controls: "dilation rate、group 数、kernel size、stride、padding 和上采样方式",
    observe: "感受野是否变大，输出尺寸是否符合预期，是否出现棋盘格，通道之间是否信息不足",
    why: "扩张卷积用间隔采样扩大感受野，分组卷积减少连接，转置卷积通过可学习权重生成更大特征图。",
    misconception: "转置卷积不是严格意义的反卷积；扩张率过大也会漏掉局部连续细节。",
    engineering: "语义分割、超分辨率、轻量模型和检测网络里的 FPN 都会用到高级卷积。",
    consoleTask: "在中央控制台调 dilation、groups 或上采样方式，观察输出形状和响应连续性。",
  },
  "part2/08_visualization_gradcam": {
    what: "这页讲 Grad-CAM 如何用梯度和特征图生成热力图，解释 CNN 做分类时关注了图像哪里。",
    analogy: "像问阅卷老师这道题给分主要看哪几处步骤，而不是只拿到最后分数。",
    intuition: "热力图越亮，说明该区域对当前类别分数贡献越大。",
    variable: "目标类别、最后卷积层特征图、类别梯度、通道权重和热力图叠加",
    elements: "原图是输入证据，热区是类别相关区域，类别分数是解释目标，叠加图显示模型关注位置。",
    controls: "目标类别、解释层、热力图透明度、阈值和输入样本",
    observe: "热区是否落在目标物体上，是否被背景带偏，不同类别的热图是否不同",
    why: "Grad-CAM 用类别分数对特征图的梯度衡量通道重要性，再把重要通道加权汇总成空间热图。",
    misconception: "热图不是因果证明，只是当前模型当前层对某个类别的局部解释线索。",
    engineering: "排查背景偏置、做模型可解释报告、审核医疗/工业视觉模型时会用到 Grad-CAM。",
    consoleTask: "在中央控制台切换目标类别或输入样本，比较热区是否随类别目标合理移动。",
  },
  "part2/09_transfer_learning": {
    what: "这页讲如何复用预训练 CNN 的通用视觉特征，在小数据任务上只训练部分层或新分类头。",
    analogy: "像请一个已经会画画的人学新题材，只需要补新风格，而不是从握笔开始教。",
    intuition: "浅层视觉特征常常通用，最后几层更贴近具体任务。",
    variable: "预训练权重、冻结层数、新分类头、学习率、数据规模和域差异",
    elements: "冻结块代表不更新的特征提取器，新头代表任务输出，曲线代表微调是否稳定。",
    controls: "冻结比例、分类头大小、微调学习率、数据增强强度和预训练模型选择",
    observe: "小数据下是否更快收敛，冻结太多是否欠拟合，解冻太多是否过拟合或破坏旧特征",
    why: "预训练模型已经学到边缘、纹理和形状等通用特征，小数据任务只需调整任务相关部分。",
    misconception: "迁移学习不是无脑冻结；数据域差异大时，必须重新评估解冻策略和学习率。",
    engineering: "工业小样本分类、医学影像、缺陷检测和快速原型开发常靠迁移学习起步。",
    consoleTask: "在中央控制台加载视觉预设，比较只训分类头和解冻部分层的训练/验证曲线。",
  },
  "part2/cnn_architectures": {
    what: "这页把多个 CNN 架构放在同一张对照表里，帮助你比较特征提取路径和结构取舍。",
    analogy: "像把不同相机拆开摆在桌上，看镜头、传感器和处理芯片各自怎么配合。",
    intuition: "看架构时先看信息如何流动，再看参数量和计算量。",
    variable: "卷积块、池化阶段、残差连接、通道变化、参数量和输出特征层",
    elements: "每个结构块代表一段特征变换，箭头代表信息流，宽度代表通道规模，台阶代表分辨率下降。",
    controls: "架构选择、输入分辨率、通道倍率、残差开关和分类头大小",
    observe: "哪一层降采样，哪里保留高分辨率，参数量集中在哪些模块，训练曲线是否稳定",
    why: "不同架构用不同方式平衡表达能力、梯度传播、内存占用和推理速度。",
    misconception: "架构名不是答案，必须能解释它为什么适合当前数据规模和部署约束。",
    engineering: "选 backbone、写模型报告、做消融实验和部署前估算资源时都要做架构对照。",
    consoleTask: "在中央控制台切换 CNN 架构预设，记录参数量、输出形状和验证指标。",
  },
  "part2/advanced_cnn": {
    what: "这页综合残差、注意力式通道调整、轻量卷积和多尺度特征，讲现代 CNN 的设计思路。",
    analogy: "像升级一条生产线：不是只加机器，而是重新安排旁路、质检和分工。",
    intuition: "高级 CNN 设计的目标是让有用特征走得稳、算得省、融合得好。",
    variable: "残差块、通道注意、深度可分离卷积、多尺度融合、参数量和延迟",
    elements: "旁路表示信息保留，融合节点表示多尺度汇合，指标卡表示精度、速度和模型大小。",
    controls: "残差深度、通道倍率、注意力模块开关、输入尺度和轻量化策略",
    observe: "精度是否提升，延迟是否可接受，深层是否仍能训练，低层细节是否被保留",
    why: "现代 CNN 通常通过更短梯度路径、更高效卷积和多尺度融合来提升可训练性和部署效率。",
    misconception: "堆新模块不等于高级；每个模块都要说明它改善的是梯度、感受野、通道选择还是效率。",
    engineering: "视觉竞赛方案、移动端模型、检测分割骨干和工业部署都会遇到这些取舍。",
    consoleTask: "在中央控制台打开轻量化或残差选项，比较 loss、参数量和推理成本的变化。",
  },
  "part3/01_rnn_intuition": {
    what: "这页讲 RNN 如何把上一时刻的隐藏状态带到下一时刻，让模型拥有最基本的序列记忆。",
    analogy: "像边听故事边做笔记，每听一句就更新一次当前理解。",
    intuition: "RNN 的当前输出由当前输入和过去状态共同决定。",
    variable: "当前 token、隐藏状态、循环权重、序列长度和最终输出",
    elements: "横向节点代表时间步，循环箭头代表记忆传递，状态柱代表当前隐藏状态强度。",
    controls: "序列长度、隐藏维度、记忆保留率、输入噪声和初始状态",
    observe: "早期 token 的影响能保留多久，状态是否被后续输入覆盖，长序列是否更难稳定",
    why: "隐藏状态每一步都会被更新，历史信息要经过多次变换才能到达末尾，因此容易衰减或混杂。",
    misconception: "RNN 不是自动记住全部历史，它只能把历史压缩进有限维隐藏状态。",
    engineering: "日志序列、传感器数据、短文本建模和理解 LSTM/GRU 前都要先掌握 RNN 直觉。",
    consoleTask: "在中央控制台调序列长度和记忆保留率，观察前面 token 对末尾预测的影响是否消失。",
  },
  "part3/02_hidden_states": {
    what: "这页专门看隐藏状态如何存储、衰减和更新，是理解 RNN 记忆能力的核心窗口。",
    analogy: "像白板上的会议纪要：每轮讨论都会擦掉一部分、补上一部分，最后留下摘要。",
    intuition: "隐藏状态是序列模型对过去信息的压缩记忆。",
    variable: "隐藏向量、门控值、记忆保留率、输入写入量和长程依赖",
    elements: "状态条代表不同维度的记忆强度，门控开关代表保留或写入，时间箭头代表递推顺序。",
    controls: "保留门、更新门、隐藏维度、序列长度和噪声强度",
    observe: "哪些维度持续保留，哪些维度快速归零，关键输入是否能跨多个时间步留下痕迹",
    why: "门控通过接近 0 或 1 的系数控制遗忘与写入，从而减缓普通 RNN 的长期依赖问题。",
    misconception: "隐藏状态维度越大不一定越好，过大可能记住噪声并增加训练难度。",
    engineering: "调 LSTM/GRU、解释序列分类错误、处理长上下文衰减时需要读懂隐藏状态。",
    consoleTask: "在中央控制台提高噪声或降低保留率，看隐藏状态柱是否更快失去早期线索。",
  },
  "part3/03_sequence_toys": {
    what: "这页用复制、延迟预测、括号匹配等玩具任务，让序列记忆和泛化问题变得可控。",
    analogy: "像用简化迷宫训练方向感，地图小但能暴露记忆和规划问题。",
    intuition: "玩具任务的价值是把一个序列能力单独拎出来测试。",
    variable: "任务规则、序列长度、关键 token 位置、噪声比例和预测正确率",
    elements: "彩色 token 表示输入类别，目标标记表示需要记住的位置，结果卡表示是否成功泛化。",
    controls: "任务类型、序列长度、噪声、隐藏维度和训练样本数",
    observe: "模型是否只记最近输入，长度变长后是否失败，训练过的模式能否迁移到新长度",
    why: "玩具任务把干扰因素降到最少，所以失败通常能直接指向记忆、优化或数据分布问题。",
    misconception: "玩具任务分数高不等于真实任务成功，但玩具任务失败通常说明基础机制还没学稳。",
    engineering: "设计新序列模型或调试训练循环时，先用玩具任务验证记忆和泛化能力。",
    consoleTask: "在中央控制台选择一个序列玩具任务，拉长序列长度，记录准确率从哪一步开始下降。",
  },
  "part3/04_hyperparam_rnn": {
    what: "这页讲 RNN 的学习率、隐藏维度、层数、截断反传长度如何影响训练稳定性和记忆能力。",
    analogy: "像调收音机：音量、频段和天线长度都影响信号，但一次调太多就不知道是谁起作用。",
    intuition: "RNN 超参本质是在优化稳定、记忆容量和计算成本之间取平衡。",
    variable: "学习率、hidden size、层数、BPTT 长度、梯度裁剪阈值和验证损失",
    elements: "控制条是超参，曲线是训练反馈，梯度卡显示爆炸或消失风险。",
    controls: "学习率、隐藏维度、层数、截断长度、梯度裁剪和 dropout",
    observe: "loss 是否震荡，梯度是否爆炸，长依赖是否改善，参数量是否明显增加",
    why: "序列反传会跨时间累积梯度，超参稍微不合适就可能放大震荡或截断有用信号。",
    misconception: "hidden size 加大不是万能，可能让训练更慢、更过拟合，也更难稳定。",
    engineering: "训练文本模型、时间序列模型和语音模型时，需要用这页的方法做可复现实验记录。",
    consoleTask: "在中央控制台一次只改一个 RNN 超参，保存 loss、梯度范数和验证表现三列记录。",
  },
  "part3/05_seq2seq_attention": {
    what: "这页讲编码器如何压缩输入序列，解码器如何逐步输出，注意力如何在每一步回看源序列。",
    analogy: "像翻译长句时边写译文边回看原文相关片段，而不是只靠脑子里的一句摘要。",
    intuition: "注意力让解码器不用把全部源信息都塞进一个固定向量。",
    variable: "编码器状态、解码器状态、对齐权重、上下文向量和输出 token",
    elements: "输入 token 是源句，输出 token 是目标句，连线或热格表示当前输出看向哪个输入位置。",
    controls: "解码步、注意力温度、输入长度、隐藏维度和 teacher forcing 比例",
    observe: "对齐是否落在合理源词，长句是否比无注意力更稳，错误输出来自错看还是不会生成",
    why: "每个解码步会用当前状态查询所有编码器状态，权重越高的位置贡献越大。",
    misconception: "注意力对齐不是总等于人类翻译对齐，尤其在多词短语和重排语言里要谨慎解释。",
    engineering: "机器翻译、摘要、语音识别和早期 encoder-decoder 系统都依赖这套思路。",
    consoleTask: "在中央控制台选一个 query 步，调注意力锐度，看对齐热格和输出 token 是否同步变化。",
  },
  "part3/06_text_classification": {
    what: "这页讲如何把一句文本编码成向量，再用分类头判断情感、主题或标签。",
    analogy: "像读完一段评论后写一句摘要，再根据摘要判断它是好评还是差评。",
    intuition: "文本分类的关键是把关键 token 的信息压进最终表示。",
    variable: "分词、嵌入、序列编码、池化方式、分类 logits 和标签概率",
    elements: "token 条表示输入词，嵌入点表示词向量，池化节点表示汇总，输出条表示各类别概率。",
    controls: "最大长度、嵌入维度、池化方式、隐藏维度、dropout 和分类阈值",
    observe: "关键词是否影响最终类别，截断是否丢掉重要信息，概率是否过度自信",
    why: "分类头只看编码后的表示，如果编码器没有保留关键 token 或位置关系，后面再强也难补救。",
    misconception: "词频高不一定重要，短文本也可能因为否定词、反讽或截断导致误判。",
    engineering: "舆情分类、工单路由、垃圾文本识别和日志告警都需要理解文本表示与分类边界。",
    consoleTask: "在中央控制台改变最大长度或池化方式，观察分类概率和错误样本解释。",
  },
  "part3/07_advanced_training": {
    what: "这页讲梯度裁剪、Teacher Forcing、dropout、学习率策略等序列模型训练技巧。",
    analogy: "像训练合唱队：要控制音量爆掉，也要决定老师带唱多久、什么时候让学生独立唱。",
    intuition: "高级训练技巧是在稳定优化和真实推理之间找平衡。",
    variable: "梯度范数、裁剪阈值、teacher forcing 比例、dropout、学习率计划和验证指标",
    elements: "梯度条显示爆炸风险，比例控制条显示教师信号强度，曲线显示训练和验证差距。",
    controls: "裁剪阈值、teacher forcing 比例、dropout、学习率衰减和 batch size",
    observe: "梯度是否被控制，训练是否不再发散，验证表现是否因过度依赖教师信号而下降",
    why: "序列训练常因长链反传和自回归误差累积不稳定，这些技巧分别控制梯度、噪声和训练推理偏差。",
    misconception: "Teacher Forcing 比例越高不一定越好，训练时太依赖真实前缀会导致推理时崩。",
    engineering: "训练翻译、生成、语音和时间序列模型时，这些技巧决定实验能不能稳定复现。",
    consoleTask: "在中央控制台先打开梯度裁剪，再逐步降低 teacher forcing，比较训练稳定性和验证指标。",
  },
  "part3/08_debug_problems": {
    what: "这页讲序列模型常见故障：梯度爆炸、记忆衰减、数据错位、mask 错误和评估口径不一致。",
    analogy: "像查一条流水账：总额不对时，要看每一天记录、结转和最后统计公式。",
    intuition: "RNN 调试要沿时间轴查，不要只看最后一个输出。",
    variable: "序列长度、mask、梯度范数、隐藏状态、标签对齐和训练/推理差异",
    elements: "时间轴展示输入输出对齐，告警卡显示梯度或 mask 风险，错误样本显示失败模式。",
    controls: "截断长度、mask 开关、学习率、梯度裁剪、padding 策略和评估窗口",
    observe: "padding 是否被模型看到，标签是否错位，长序列性能是否突然下降，训练推理是否不一致",
    why: "序列模型对位置、长度和状态传递敏感，一个小的对齐或 mask 错误会沿时间步持续放大。",
    misconception: "序列模型预测差不一定是结构弱，很多时候是 padding、截断或 teacher forcing 口径错。",
    engineering: "线上文本、日志和语音模型排障时，要用这页的 checklist 快速定位问题层。",
    consoleTask: "在中央控制台人为增加噪声或改变序列长度，写出第一排查项和下一步验证方法。",
  },
  "part3/sequence_models": {
    what: "这页总览 RNN、LSTM、GRU 和注意力前的序列建模范式，帮助你选择合适的记忆结构。",
    analogy: "像选择记事工具：便签、笔记本和索引卡都能记录，但适合的任务不同。",
    intuition: "序列模型的差异主要在于历史信息怎样保存、遗忘和读取。",
    variable: "隐藏状态、门控结构、序列长度、输入噪声、任务目标和计算成本",
    elements: "模型卡代表不同结构，时间轴代表输入顺序，状态条代表历史信息保留程度。",
    controls: "模型类型、隐藏维度、序列长度、门控强度和训练轮数",
    observe: "哪种结构更能保留长依赖，哪种训练更快，错误是否来自记不住还是读错位置",
    why: "普通 RNN 压缩历史最简单但易衰减，LSTM/GRU 用门控改善记忆，注意力进一步提供直接读取。",
    misconception: "不是所有序列任务都必须用 Transformer，小数据或短序列下简单模型可能更稳。",
    engineering: "文本分类、预测、异常检测和资源受限场景都需要按任务选择序列模型。",
    consoleTask: "在中央控制台用同一序列任务切换 RNN/LSTM/GRU 预设，对比长序列表现。",
  },
  "part4/01_attention_mechanism": {
    what: "这页讲 Query、Key、Value 如何通过相似度打分和 softmax 得到注意力权重。",
    analogy: "像带着问题查资料：问题是 Query，目录线索是 Key，真正摘抄回来的内容是 Value。",
    intuition: "注意力就是当前 token 按相关性从上下文取回信息。",
    variable: "Query 向量、Key 向量、Value 向量、相似度分数、softmax 权重和上下文向量",
    elements: "行表示当前 query，列表示可看的 key，热格或条形代表权重，汇总节点代表加权后的 value。",
    controls: "query token、注意力锐度、上下文噪声、mask 和缩放因子",
    observe: "权重是否集中到相关 token，噪声是否抢走关注，mask 后未来位置是否不可见",
    why: "点积衡量 query 与 key 的匹配程度，softmax 把分数变成权重，再对 value 做加权平均。",
    misconception: "注意力权重有解释价值，但不能直接等同模型最终因果解释。",
    engineering: "读 Transformer、排查长上下文错误、设计检索增强和理解大模型上下文都要掌握注意力。",
    consoleTask: "在中央控制台固定一句话，切换 query token，观察注意力权重如何重新分布。",
  },
  "part4/02_multihead_visual": {
    what: "这页讲多头注意力如何让不同头在不同子空间同时捕捉位置、语义、句法或局部关系。",
    analogy: "像一个评审团同时看文本：有人看主语谓语，有人看指代，有人看相邻词。",
    intuition: "多头不是重复看同一件事，而是并行看不同关系。",
    variable: "头数、每头维度、各头权重图、拼接输出和投影矩阵",
    elements: "每张热图是一个注意力头，颜色表示权重，拼接块表示多头结果合并。",
    controls: "头数、选中头、query token、注意力锐度和噪声",
    observe: "不同头是否有不同关注模式，是否所有头都塌缩到同一区域，合并后输出是否更稳定",
    why: "每个头有独立投影矩阵，能在不同表示子空间计算匹配关系，最后再融合。",
    misconception: "头数越多不一定越好；如果维度和数据不足，多头可能冗余或难训练。",
    engineering: "调 Transformer 大小、分析 attention pattern、做模型压缩和剪枝时会检查多头分工。",
    consoleTask: "在中央控制台逐个查看注意力头，记录每个头更像在看局部、长距还是特殊 token。",
  },
  "part4/03_encoder_decoder": {
    what: "这页拆解 Transformer 编码器和解码器，重点是自注意力、交叉注意力和因果 mask 的分工。",
    analogy: "像翻译团队：编码器先通读原文，解码器边写译文边回看原文并遮住未来答案。",
    intuition: "编码器负责理解输入，解码器负责在约束下逐步生成输出。",
    variable: "编码器层、解码器层、自注意力、交叉注意力、因果 mask 和位置编码",
    elements: "上游块代表编码，右侧块代表解码，mask 阴影代表不可看的未来位置，跨层箭头代表读源序列。",
    controls: "编码层数、解码层数、mask 开关、解码步和注意力头数",
    observe: "解码器是否只能看已生成 token，交叉注意力是否回到输入关键位置，输出是否逐步形成",
    why: "生成任务必须防止未来信息泄漏，同时又要从编码器结果读取源输入，因此需要两类注意力。",
    misconception: "编码器和解码器不是简单上下游复制，解码器多了因果约束和交叉读取。",
    engineering: "机器翻译、摘要、语音识别、图像字幕和 seq2seq 大模型都基于这套结构。",
    consoleTask: "在中央控制台打开 mask 对比实验，观察能看未来和不能看未来时权重图的差别。",
  },
  "part4/04_minimal_transformer": {
    what: "这页用最小实现串起 token embedding、位置编码、注意力、MLP、残差和 LayerNorm。",
    analogy: "像拆一台简化发动机：零件少，但每个零件的连接顺序必须准确。",
    intuition: "Transformer block 的骨架是注意力负责混合 token，MLP 负责逐位置变换，残差负责稳定传递。",
    variable: "embedding 维度、位置编码、QKV 投影、残差连接、LayerNorm 和 MLP 隐藏层",
    elements: "代码块对应结构块，箭头表示张量流动，残差旁路表示输入与变换结果相加。",
    controls: "模型维度、头数、层数、上下文长度、MLP 比例和 dropout",
    observe: "每一步 shape 是否不变或按预期变化，残差相加是否维度一致，mask 是否正确广播",
    why: "最小实现把复杂模型压成可检查的张量变换，任何 shape 或 mask 错误都会在这里暴露。",
    misconception: "能跑通最小实现不代表高性能，但它是理解大模型结构的最好骨架。",
    engineering: "读开源 LLM、改模型结构、写自定义 attention 或排查训练错误都需要最小实现能力。",
    consoleTask: "在中央控制台按层组装 Transformer block，逐步检查 QKV、attention 输出和残差 shape。",
  },
  "part4/05_flash_attention": {
    what: "这页讲 Flash Attention 如何通过分块计算减少显存读写，让注意力在长上下文下更高效。",
    analogy: "像做大表格统计时分批搬运纸张，不把整张巨表摊满桌子。",
    intuition: "Flash Attention 优化的主要不是数学公式，而是内存访问路径。",
    variable: "序列长度、block size、显存读写次数、softmax 分块统计和吞吐",
    elements: "块状矩阵代表分片计算，内存箭头代表读写，指标卡代表显存占用和速度。",
    controls: "序列长度、块大小、头数、精度类型和是否使用高效 kernel",
    observe: "显存是否下降，长序列吞吐是否提升，输出是否与普通注意力数值一致",
    why: "注意力矩阵很大，反复读写会拖慢训练；分块在线 softmax 可以避免完整存下全部权重矩阵。",
    misconception: "Flash Attention 不改变注意力语义，它是更高效的精确计算，不是近似注意力。",
    engineering: "训练长上下文 LLM、节省显存、提高吞吐和选择推理 kernel 时必须理解这一页。",
    consoleTask: "在中央控制台增加上下文长度，对比普通注意力和高效注意力的显存/速度指标。",
  },
  "part4/06_debug_problems": {
    what: "这页讲 Transformer 常见故障：mask 泄露、位置编码错位、梯度异常、注意力塌缩和训练不稳定。",
    analogy: "像检查考试作弊：分数异常时要看是否偷看未来答案、座位号是否错、评分表是否坏。",
    intuition: "Transformer 调试先查 mask 和 shape，再查训练曲线和数据。",
    variable: "attention mask、position ids、QKV shape、loss 曲线、梯度范数和 token 对齐",
    elements: "mask 矩阵显示可见范围，位置条显示 token 编号，告警卡显示异常曲线或梯度。",
    controls: "mask 类型、上下文长度、学习率、warmup、位置编码方式和 batch size",
    observe: "未来 token 是否被遮住，位置编号是否连续，loss 是否突然异常，注意力是否过尖或全平",
    why: "Transformer 高度依赖并行矩阵和位置约束，任何广播或 mask 错误都会让模型学到错误捷径。",
    misconception: "loss 很低可能是数据泄漏或 mask 错，不一定是模型学得好。",
    engineering: "训练 LLM、改 tokenizer、扩上下文、微调和部署前验证都需要这套调试清单。",
    consoleTask: "在中央控制台切换 mask 设置，观察权重图是否出现未来信息泄露。",
  },
  "part4/transformer_models": {
    what: "这页总览自注意力、多头、位置编码、BERT/GPT 差异和 Transformer 架构选择。",
    analogy: "像看不同读书方式：BERT 双向理解全文，GPT 按顺序续写下一个词。",
    intuition: "Transformer 家族的差异，主要来自可见上下文、训练目标和输出方式。",
    variable: "双向 mask、因果 mask、位置编码、层数、头数、预训练目标和输出头",
    elements: "结构卡说明模型类型，mask 图显示能看哪些 token，输出区呈现分类或生成结果。",
    controls: "模型类型、上下文长度、头数、层数、mask 方式和任务头",
    observe: "BERT 是否能看双向上下文，GPT 是否只能看左侧，位置编码改变后长文本是否稳定",
    why: "相同 block 在不同 mask 和训练目标下会形成不同能力：理解、生成或编码检索。",
    misconception: "Transformer 不等于 GPT；编码器、解码器和编码器-解码器适合的任务不同。",
    engineering: "选择预训练模型、设计微调任务、理解 LLM 架构和排查上下文问题都要用这页总览。",
    consoleTask: "在中央控制台分别选择编码器和生成式预设，对比可见 token 范围和输出方式。",
  },
  "part4/gan_ae": {
    what: "这页讲自编码器如何学习压缩重构，GAN 如何让生成器和判别器对抗学习数据分布。",
    analogy: "自编码器像压缩照片再还原，GAN 像造假者和鉴定师互相较劲。",
    intuition: "自编码器学的是可重构表征，GAN 学的是让生成样本越来越像真实数据。",
    variable: "潜空间、重构误差、生成器、判别器、对抗损失和样本质量",
    elements: "瓶颈向量代表潜变量，重构图代表还原结果，两条损失曲线代表生成器和判别器博弈。",
    controls: "潜变量维度、重构损失权重、生成器学习率、判别器学习率和噪声输入",
    observe: "重构是否保留关键信息，潜空间是否平滑，GAN 是否模式崩塌或判别器过强",
    why: "自编码器通过压缩迫使模型保留主要因素，GAN 通过对抗反馈逼近真实样本分布。",
    misconception: "GAN 不是只看生成图漂亮，还要警惕模式崩塌和训练不稳定。",
    engineering: "异常检测、表征学习、数据生成、图像修复和生成模型入门都需要这两个框架。",
    consoleTask: "在中央控制台调潜变量维度或生成器学习率，观察重构质量和对抗损失是否稳定。",
  },
  "part4/gnn_intro": {
    what: "这页讲图神经网络如何让节点通过边交换消息，把邻居信息聚合成新的节点表示。",
    analogy: "像朋友圈打听消息：一个人不仅看自己，也会听邻居怎么说，再更新判断。",
    intuition: "GNN 的核心是沿边传递、聚合、更新。",
    variable: "节点特征、边关系、邻居集合、消息函数、聚合函数和层数",
    elements: "圆点代表节点，连线代表边，箭头代表消息流，颜色变化代表节点表示更新。",
    controls: "聚合方式、GNN 层数、邻居采样数、边权重和隐藏维度",
    observe: "节点表示是否受邻居影响，层数太多是否过平滑，关键边是否改变预测",
    why: "每一层 GNN 让节点吸收一跳邻居信息，多层后信息来自更远邻域，但也可能变得过于相似。",
    misconception: "图上堆很多层不一定更好，过平滑会让不同节点表示失去区分度。",
    engineering: "社交网络、推荐系统、分子图、知识图谱和依赖关系建模都会用到 GNN。",
    consoleTask: "在中央控制台调整层数和邻居采样，观察节点颜色是否从局部差异变成过度相似。",
  },
  "part5/01_feature_visualization": {
    what: "这页讲如何可视化特征、激活、嵌入和决策边界，让抽象表示变成可观察证据。",
    analogy: "像给模型做体检片：不只看最终诊断，还看内部器官有没有异常信号。",
    intuition: "特征可视化是在问模型到底把什么当成有用信息。",
    variable: "激活值、嵌入距离、通道响应、决策边界和类别分离度",
    elements: "点云代表样本嵌入，颜色代表标签或预测，边界代表分类区域，热区代表强激活。",
    controls: "层级、通道、降维方式、样本筛选、阈值和类别",
    observe: "同类样本是否聚在一起，错误样本是否靠近边界，强激活是否对应合理特征",
    why: "网络内部表示会把输入压到任务相关空间，可视化能暴露聚类、混淆和捷径特征。",
    misconception: "降维图只是一种投影，不能把二维距离当成原始高维距离的绝对真相。",
    engineering: "模型解释、错误分析、数据清洗、特征漂移监控和汇报模型行为都需要可视化。",
    consoleTask: "在中央控制台选择一层特征，切换类别筛选，看错误样本是否聚在边界附近。",
  },
  "part5/02_gradient_monitor": {
    what: "这页讲如何监控梯度范数、爆炸、消失和更新比例，判断训练是否健康。",
    analogy: "像看心电图：曲线不是治疗本身，但能告诉你系统是否危险。",
    intuition: "梯度监控回答的是参数有没有收到合适强度的学习信号。",
    variable: "全局梯度范数、分层梯度、更新/参数比例、裁剪阈值和 NaN 告警",
    elements: "柱状图代表各层梯度，警戒线代表阈值，曲线代表训练过程中梯度强弱变化。",
    controls: "学习率、裁剪阈值、初始化方式、归一化开关和 batch size",
    observe: "梯度是否长期接近 0，是否突然爆炸，前层和后层是否差距过大，裁剪是否频繁触发",
    why: "反向传播链路过长或尺度不合适会让梯度被反复放大或缩小，导致训练发散或停滞。",
    misconception: "loss 下降不代表梯度一定健康，局部层可能已经失去学习信号。",
    engineering: "大模型训练、RNN/Transformer 调试、混合精度和分布式训练都需要梯度监控。",
    consoleTask: "在中央控制台把学习率调大到震荡区，观察梯度范数和 loss 如何同时异常。",
  },
  "part5/03_training_dynamics": {
    what: "这页讲用训练曲线、验证曲线、梯度和指标追踪模型从不会到会的动态过程。",
    analogy: "像看运动员训练日志：单场成绩不够，要看长期趋势、伤病信号和比赛表现。",
    intuition: "训练动态关注的是模型学习路径，而不是单个最终分数。",
    variable: "训练 loss、验证 loss、准确率、泛化间隙、梯度健康度和学习率变化",
    elements: "黑线是训练，金线是验证，间距代表泛化风险，指标卡显示当前诊断结论。",
    controls: "学习率、正则强度、数据噪声、epoch、batch size 和 early stopping",
    observe: "曲线是否同步下降，是否过拟合，是否震荡，是否进入平台期或欠拟合",
    why: "优化、泛化和噪声会在曲线上留下不同形状，读曲线能决定下一步该加数据、调参还是改模型。",
    misconception: "只截取最后一个 epoch 的指标很危险，趋势比单点数字更能说明问题。",
    engineering: "实验管理、模型调参、故障复盘和训练平台 dashboard 都围绕训练动态展开。",
    consoleTask: "在中央控制台运行训练可视化，分别制造欠拟合、过拟合和学习率过大的曲线。",
  },
  "part5/04_hyperparam_search": {
    what: "这页讲网格搜索、随机搜索和实验记录如何系统比较超参数，而不是凭感觉调参。",
    analogy: "像试菜谱：一次记录火候、盐量和时间，才知道哪次好吃是因为什么。",
    intuition: "超参搜索的核心是可比较、可复现、可归因。",
    variable: "搜索空间、试验编号、学习率、正则、模型大小、验证指标和资源预算",
    elements: "表格代表实验记录，散点代表不同试验，颜色代表指标好坏，边界代表预算约束。",
    controls: "搜索方法、学习率范围、正则范围、试验次数、随机种子和早停规则",
    observe: "好结果是否集中在某个范围，随机搜索是否比网格更快找到有效区域，失败实验是否有模式",
    why: "高维空间里网格很容易浪费在不重要维度，随机和记录能更快发现敏感超参。",
    misconception: "只保存最佳参数会丢掉最有价值的失败信息，也无法解释为什么好。",
    engineering: "比赛调参、业务模型迭代、AutoML 和论文消融都需要严格实验记录。",
    consoleTask: "在中央控制台设定小搜索空间，运行多组学习率和正则组合，按验证指标排序并总结规律。",
  },
  "part5/05_dataset_toys": {
    what: "这页用小型可控数据集验证模型直觉，比如线性可分、环形分布、噪声标签和类别不平衡。",
    analogy: "像风洞实验：先在小环境里测清楚现象，再上真实飞机。",
    intuition: "玩具数据能让你把数据形状和模型能力直接对应起来。",
    variable: "数据分布、噪声比例、类别间隔、样本数量、决策边界和验证准确率",
    elements: "散点代表样本，颜色代表类别，边界代表模型预测区域，错分点暴露失败位置。",
    controls: "数据形状、样本数、噪声、类别不平衡、模型复杂度和正则",
    observe: "线性模型能否分开，非线性边界是否过拟合噪声，少数类是否被忽略",
    why: "数据分布决定任务难度，模型假设如果不匹配分布，再多训练也只能学到有限边界。",
    misconception: "真实数据复杂不代表不能用玩具数据；玩具数据是验证机制，不是替代真实评估。",
    engineering: "新模型、新 loss、新采样策略和课程演示都适合先在玩具数据上排除基本问题。",
    consoleTask: "在中央控制台切换线性和环形数据，观察同一个模型的决策边界如何失败或成功。",
  },
  "part5/data_training": {
    what: "这页总览数据管线、训练循环、指标、调试和实验记录，强调训练是一个可诊断系统。",
    analogy: "像厨房出餐流程：食材、切配、烹饪、试味、留样，每一步出错都会影响成品。",
    intuition: "训练质量来自数据、代码、优化和评估的整条链路。",
    variable: "数据加载、增强、batch、前向、损失、反向、优化器、指标和日志",
    elements: "流程节点代表训练循环阶段，曲线代表结果反馈，日志卡代表可复现实验信息。",
    controls: "batch size、学习率、增强强度、评估频率、保存策略和随机种子",
    observe: "数据是否稳定喂入，训练是否可复现，指标是否和任务目标一致，日志是否足够回放",
    why: "训练循环把数据和模型反复闭环，任何一段不稳定都会通过曲线、指标或产物暴露出来。",
    misconception: "训练脚本跑完不等于训练正确，必须能解释数据、指标和产物如何对应。",
    engineering: "从课程练习到生产训练平台，都需要把训练过程写成可追踪、可回放、可排查的系统。",
    consoleTask: "在中央控制台从数据设置开始，逐项检查训练循环的输入、输出、指标和保存结果。",
  },
  "part5/case_studies": {
    what: "这页用完整案例把问题定义、数据检查、建模、调参、诊断和复盘串起来。",
    analogy: "像看医生完整病历：症状、检查、诊断、用药和复查都要连起来看。",
    intuition: "案例研究训练的是从现象到决策的完整路径。",
    variable: "业务目标、数据质量、基线模型、实验变量、错误样本和最终取舍",
    elements: "时间线代表实验推进，决策卡代表关键选择，错误面板代表下一轮改进依据。",
    controls: "案例场景、基线模型、实验变量、诊断维度和复盘视角",
    observe: "每个决策是否有证据，失败实验是否改变下一步，最终方案是否匹配约束",
    why: "真实项目不是单点技术题，数据、指标、资源、解释和部署约束会共同决定方案。",
    misconception: "案例不是故事会，必须能指出每一步证据如何支持下一步决策。",
    engineering: "写项目报告、面试讲项目、做模型复盘和团队知识沉淀时都需要案例化表达。",
    consoleTask: "在中央控制台选择一个训练失败场景，按案例方式写出观察、假设、操作和结论。",
  },
  "part5/deployment_tools": {
    what: "这页讲模型导出、推理服务、版本管理、延迟监控和工程部署的基本工具链。",
    analogy: "像把实验室样机变成售卖产品：不仅要能用，还要稳定、可维护、可追踪。",
    intuition: "部署关注的是模型在真实请求中是否稳定、快速、可回滚。",
    variable: "模型格式、输入输出 schema、推理延迟、吞吐、版本、监控指标和回滚策略",
    elements: "服务节点代表推理链路，版本标签代表模型产物，指标卡代表延迟、错误率和资源使用。",
    controls: "batch 推理、量化开关、并发数、模型版本、缓存和超时阈值",
    observe: "延迟是否超标，输入 schema 是否一致，模型版本是否可追踪，错误率是否随流量上升",
    why: "训练好的权重只有接入真实系统后才面对延迟、资源、兼容性和数据漂移问题。",
    misconception: "部署不是把模型文件复制到服务器，接口契约、监控和回滚同样是模型的一部分。",
    engineering: "API 推理、边缘部署、A/B 测试、模型监控和线上事故排查都属于部署工具链。",
    consoleTask: "在中央控制台把一次推理请求画成链路，调并发或缓存，观察延迟和瓶颈位置。",
  },
  "part5/quiz_system": {
    what: "这页用交互式测验检查机器学习、CNN、RNN、Transformer 和生成模型的理解程度。",
    analogy: "像做错题本：答案本身不够，关键是知道自己错在哪类概念。",
    intuition: "测验的价值是把模糊的会变成明确的掌握或未掌握。",
    variable: "题目类别、难度、选项、解释、错题记录和知识点覆盖",
    elements: "题卡代表当前问题，选项代表判断路径，解析区代表纠错证据，进度条代表覆盖情况。",
    controls: "题目方向、难度、随机模式、错题筛选和是否显示解析",
    observe: "错题是否集中在某个主题，解析能否纠正误解，复做后是否真的减少同类错误",
    why: "主动检索会暴露记忆缺口，错题分类能把学习重点从感觉转成证据。",
    misconception: "刷题不是追求点完所有选项，而是把错误解释修正成正确因果链。",
    engineering: "课程复习、面试准备、团队培训和知识点验收都需要测验系统。",
    consoleTask: "在中央控制台选一个自己不稳的模块，做题后把错因映射回对应实验页面。",
  },
  "part5/tuning_challenge": {
    what: "这页把调参放进真实约束：预算有限、指标冲突、实验要记录、失败要能解释。",
    analogy: "像赛车调校：轮胎、悬挂、油耗和速度互相牵制，不能只看一圈最快。",
    intuition: "调参不是碰运气，而是在约束下做有记录的决策。",
    variable: "学习率、正则、模型规模、数据增强、预算、验证指标和实验日志",
    elements: "挑战面板显示约束，实验表记录尝试，曲线和指标卡显示每次选择的后果。",
    controls: "学习率、正则、层宽、增强、训练轮数和早停策略",
    observe: "哪一个超参最敏感，是否牺牲稳定换来短期指标，失败实验是否能缩小下一轮搜索范围",
    why: "超参改变优化路径和模型容量，真实约束会迫使你在速度、精度和泛化之间取舍。",
    misconception: "调参不等于把所有滑块调到最大，也不是只靠最后最高分决定方案。",
    engineering: "真实项目、比赛和团队实验管理都需要调参挑战里的记录和取舍习惯。",
    consoleTask: "在中央控制台设定一个预算，只允许三次实验，写出每次选择和下一步依据。",
  },
  "part6/01_unified_interface": {
    what: "这页讲把模型、数据集、任务和评估抽象成统一接口，让不同实验可以复用同一套流程。",
    analogy: "像统一充电接口：设备不同，但接入方式一致，系统就容易扩展。",
    intuition: "统一接口让变化发生在局部，而不是每加一个任务就重写全项目。",
    variable: "Dataset、Model、Trainer、Evaluator、Config 和输入输出契约",
    elements: "接口块代表稳定边界，插槽代表可替换实现，箭头代表数据和控制流。",
    controls: "接口粒度、配置字段、返回格式、任务类型和评估方法",
    observe: "新增任务是否只需要实现接口，旧训练流程是否不用改，错误是否能定位到边界内",
    why: "接口把调用方和实现方解耦，稳定契约越清晰，扩展成本越低。",
    misconception: "统一接口不是把所有东西写成一个巨类，而是定义最小稳定契约。",
    engineering: "训练框架、课程项目、比赛模板和团队协作都需要统一接口。",
    consoleTask: "在中央控制台从一个空接口开始，分别接入数据、模型和评估模块，检查输入输出契约。",
  },
  "part6/02_modular_structure": {
    what: "这页讲把配置、数据、模型、训练、评估和工具拆成模块，形成可维护项目结构。",
    analogy: "像整理工具箱：锤子、螺丝刀和零件分格放，维修时不用翻完整个房间。",
    intuition: "模块化的核心是边界清楚、依赖单向、替换成本低。",
    variable: "模块职责、依赖方向、配置入口、共享工具、日志和产物目录",
    elements: "文件夹节点代表模块，连线代表依赖，风险标记代表循环引用或职责混杂。",
    controls: "模块拆分粒度、配置集中度、公共工具范围和产物保存位置",
    observe: "改模型是否影响数据代码，换数据是否影响训练器，日志和配置是否能独立追踪",
    why: "职责越混杂，修改越容易连锁；模块化通过清晰边界把变更限制在局部。",
    misconception: "目录分得多不等于模块化，关键是职责和依赖是否真的清楚。",
    engineering: "长期课程项目、多人协作、实验复现和上线维护都依赖模块化结构。",
    consoleTask: "在中央控制台检查一个项目流程，把数据、模型、训练、评估分别映射到独立模块。",
  },
  "part6/03_full_project": {
    what: "这页讲一个完整深度学习项目应该如何组织目录、配置、脚本、日志、产物和文档。",
    analogy: "像装修一套房：水电、墙面、家具和验收单都要有固定位置。",
    intuition: "完整项目骨架让实验从临时代码变成可交付系统。",
    variable: "目录结构、训练入口、配置文件、实验日志、checkpoint、评估报告和 README",
    elements: "目录树说明职责分布，流程图说明运行顺序，产物卡记录每次训练留下什么。",
    controls: "项目模板类型、配置数量、日志级别、checkpoint 策略和报告格式",
    observe: "新人能否从 README 跑通，失败后能否找到日志，模型结果能否回到具体配置",
    why: "深度学习项目结果高度依赖数据和配置，骨架把这些依赖显式保存下来。",
    misconception: "能一键运行不等于项目完整，没有记录和产物就无法复现。",
    engineering: "课程作业、论文复现、比赛和公司项目都需要完整项目骨架。",
    consoleTask: "在中央控制台生成一个训练流程清单，确认每一步都有输入、输出和保存位置。",
  },
  "part6/04_plugin_system": {
    what: "这页讲用注册表和插件机制扩展任务、模型、数据处理或评估，而不修改核心流程。",
    analogy: "像浏览器插件：浏览器本体稳定，新功能通过插件接入。",
    intuition: "插件系统把新增能力变成注册新组件，而不是改核心代码。",
    variable: "注册表、插件入口、组件名称、配置参数、加载顺序和接口校验",
    elements: "核心框架是主干，插件卡是可插拔模块，注册箭头表示被发现和调用。",
    controls: "插件类型、注册名称、启用开关、配置参数和加载优先级",
    observe: "新增插件是否不影响旧功能，接口不匹配是否能早报错，配置是否能控制启停",
    why: "注册表把字符串配置映射到具体实现，让核心流程只依赖接口而不是具体类。",
    misconception: "插件不是随意动态执行代码，必须有接口约束和错误隔离。",
    engineering: "多任务训练平台、模型库、评估工具和课程实验扩展都适合插件系统。",
    consoleTask: "在中央控制台添加一个新模型插件，检查它是否只通过注册表接入训练流程。",
  },
  "part6/05_one_click_training": {
    what: "这页讲从配置读取到训练、评估、保存和报告生成的一键训练流程。",
    analogy: "像咖啡机一键出杯：背后仍然有研磨、萃取、温控和清洗流程。",
    intuition: "一键训练的价值是自动化重复流程，同时保留每一步可追踪证据。",
    variable: "配置文件、随机种子、训练入口、评估脚本、checkpoint、日志和报告",
    elements: "流水线节点代表执行阶段，状态灯代表成功失败，产物区代表保存结果。",
    controls: "配置路径、训练模式、评估频率、保存间隔、随机种子和输出目录",
    observe: "一键运行是否可复现，失败是否能定位阶段，产物是否包含配置和指标",
    why: "自动化把人为步骤压缩成脚本，但只有记录完整才能避免一键黑箱。",
    misconception: "一键按钮不能替代诊断能力；跑通后仍要看数据、曲线、日志和产物。",
    engineering: "训练平台、比赛批量实验、课程作业验收和团队标准流程都需要一键训练。",
    consoleTask: "在中央控制台执行一次完整训练流程，检查配置、日志、模型文件和评估结果是否齐全。",
  },
  "part6/06_streamlit_demo": {
    what: "这页保留原实验台思路，但现在重点是原生 HTML/JS 的交互入口和 Python 源码对照。",
    analogy: "像把课堂演示从投影软件搬到网页白板，学生可以直接拖控件观察结果。",
    intuition: "实验台应该服务理解：控件改变变量，画面显示后果，源码解释实现。",
    variable: "交互控件、演示状态、读数区、源码片段和课程模块路由",
    elements: "控件区负责输入，动画区负责展示机制，读数区负责解释，源码区负责对照实现。",
    controls: "演示类型、参数滑块、观察重点、代码展开和课程跳转",
    observe: "拖动控件后动画是否即时更新，读数是否解释变化，源码是否能对应到画面行为",
    why: "学习网站的交互必须让变量和结果成对出现，否则动画只是装饰。",
    misconception: "实验台不是炫技页面，动画如果不能解释概念，就应该简化或重做。",
    engineering: "教学产品、可视化调试工具和模型演示页面都需要这种控件-结果-源码闭环。",
    consoleTask: "在中央控制台任选一个实验，把控件、画面、读数和源码四部分逐一对应起来。",
  },
  "part6/neural_network_playground": {
    what: "这页是中央控制台：用表单搭建神经网络，实时推导形状、参数量、代码和训练预期。",
    analogy: "像乐高工厂：先选积木类型和尺寸，再看拼出来的结构能否稳固。",
    intuition: "Playground 的价值是把抽象网络结构变成可编辑、可检查的工程对象。",
    variable: "层类型、输入形状、输出形状、参数量、激活函数和生成代码",
    elements: "层卡代表网络模块，形状标签代表张量流，参数表代表可训练规模，代码区代表实现落点。",
    controls: "输入尺寸、层类型、通道数、隐藏单元、激活函数、是否加入归一化或 dropout",
    observe: "每层输出形状是否合理，参数量是否暴涨，生成代码是否和结构图一致",
    why: "网络本质是一串张量变换；只要形状和参数闭合，才能进入训练和诊断。",
    misconception: "能拼出结构不代表能训练好，还要检查数据、loss、优化器和正则。",
    engineering: "原型设计、教学演示、模型结构复盘和面试讲解都可以先在 Playground 里搭骨架。",
    consoleTask: "就在本页先搭一个 MLP，再搭一个 CNN，比较形状流和参数量差异。",
  },
  "part6/training_demo": {
    what: "这页用轻量数据集演示训练循环，实时展示损失、准确率和梯度范数。",
    analogy: "像透明跑步机：你能看到速度、心率和步态，而不是只看到终点成绩。",
    intuition: "训练过程可视化让每一次参数更新都变成可观察事件。",
    variable: "学习率、epoch、loss、accuracy、梯度范数、训练/验证间隙",
    elements: "曲线显示指标随时间变化，读数卡显示当前状态，控制条决定训练条件。",
    controls: "学习率、正则、数据噪声、训练轮数、模型大小和随机种子",
    observe: "loss 是否下降，准确率是否提升，梯度是否异常，验证曲线是否与训练曲线分叉",
    why: "每个 batch 都会用梯度更新参数，参数变化积累起来才形成曲线趋势。",
    misconception: "单次训练曲线有随机性，必须结合多次运行和验证指标判断。",
    engineering: "训练脚本调试、调参教学、实验复盘和故障定位都需要训练过程可视化。",
    consoleTask: "在本页把学习率调到过小、合适、过大三档，记录曲线形状差异。",
  },
  "part6/07_project_template": {
    what: "这页讲训练脚本、评估脚本、K-Fold、集成预测和项目模板如何组合成可复用方案。",
    analogy: "像标准化表格：每次项目内容不同，但填写位置和验收流程一致。",
    intuition: "模板的价值是减少重复劳动，同时强制保留关键工程证据。",
    variable: "train.py、evaluate.py、配置文件、fold 划分、预测产物和提交/报告格式",
    elements: "模板目录代表文件职责，流程箭头代表执行顺序，产物卡代表每一步输出。",
    controls: "模板类型、fold 数、集成方式、保存策略、评估指标和报告字段",
    observe: "换数据后模板是否仍可运行，fold 是否无泄漏，集成是否真的提升验证指标",
    why: "标准模板把常见流程固化下来，减少因临时代码导致的评估错、产物丢和不可复现。",
    misconception: "模板不是复制粘贴越多越好，应该保留稳定骨架，把任务差异放进配置。",
    engineering: "比赛、课程项目、团队初始化和论文复现都适合从项目模板启动。",
    consoleTask: "在中央控制台选一个模板，把数据、训练、评估、预测四个入口逐一检查。",
  },
  "part6/reinforcement_learning": {
    what: "这页讲强化学习中的智能体、环境、状态、动作、奖励，以及多臂老虎机和 Q-Learning。",
    analogy: "像训练宠物走迷宫：不是直接给标准答案，而是通过奖励告诉它哪种行为更好。",
    intuition: "强化学习是在试错中学习策略，让长期奖励最大化。",
    variable: "state、action、reward、policy、Q 值、探索率和折扣因子",
    elements: "状态节点代表当前位置，动作箭头代表选择，奖励数字代表反馈，Q 表代表经验价值。",
    controls: "探索率 epsilon、学习率、折扣因子、回合数、奖励设置和环境大小",
    observe: "智能体是否从随机探索转向稳定策略，Q 值是否收敛，短期奖励是否伤害长期收益",
    why: "智能体通过奖励更新动作价值，探索负责发现新路径，利用负责选择当前最优动作。",
    misconception: "强化学习不是有奖励就能学好，奖励设计、探索和样本效率是核心难点。",
    engineering: "推荐策略、游戏 AI、机器人控制、资源调度和自动决策都需要 RL 思维。",
    consoleTask: "在中央控制台调探索率和奖励，观察策略从随机到稳定的过程。",
  },
  "part6/learning_path": {
    what: "这页讲入门测评、知识图谱、学习进度和下一步推荐如何组成个性化学习路径。",
    analogy: "像私人教练排课：先测基础，再安排顺序，练完还要根据表现调整。",
    intuition: "学习路径的核心是根据当前掌握程度选择下一步，而不是线性硬刷。",
    variable: "知识点节点、先修关系、掌握度、学习目标、测评结果和推荐规则",
    elements: "图节点代表知识点，边代表先修关系，进度条代表掌握度，推荐卡代表下一步任务。",
    controls: "目标方向、当前水平、每日时间、测评答案、跳过/复习和推荐策略",
    observe: "推荐是否避开缺失先修，复习是否集中在薄弱点，进度是否真实反映掌握而非点击",
    why: "知识点之间有依赖，个性化推荐要先补关键短板，再进入更高阶模块。",
    misconception: "进度百分比不等于掌握度，做过页面不代表能解释和迁移。",
    engineering: "在线课程、企业培训、刷题系统和自学规划都需要学习路径推荐。",
    consoleTask: "在中央控制台选一个目标，把当前薄弱模块映射到下一步学习计划。",
  },
  "part6/glossary": {
    what: "这页是深度学习术语表，用来快速检索概念、缩写、相关模块和最小解释。",
    analogy: "像专业词典：遇到陌生词先查准含义，再回到文章继续读。",
    intuition: "术语表解决的是读不懂词导致的理解中断。",
    variable: "术语名称、英文缩写、定义、相关概念、所在章节和使用场景",
    elements: "搜索框是入口，词条卡是解释，关联标签显示它应该跳回哪些课程页。",
    controls: "关键词、分类筛选、难度筛选、相关模块跳转和收藏/复习状态",
    observe: "同义词是否能搜到，定义是否能连接到页面例子，相关模块是否指向可实操内容",
    why: "概念学习需要反复建立词与机制的连接，术语表能把散落页面串成网络。",
    misconception: "查到定义不代表理解，必须回到图、代码或实验里验证这个词怎么工作。",
    engineering: "读论文、看源码、面试复习和团队统一语言都需要术语表。",
    consoleTask: "在中央控制台搜索一个不熟的术语，再跳回对应实验页，用控件验证它的含义。",
  },
  "part6/frontier": {
    what: "这页总览 LLM、AGI、多模态、智能体、推理模型、自监督、XAI、安全与对齐等前沿方向。",
    analogy: "像看科技地图：先知道哪些城市在哪，再决定自己要深入哪条路线。",
    intuition: "前沿方向不是名词堆砌，要看它解决的新约束和带来的新风险。",
    variable: "模型规模、数据来源、模态、推理能力、工具调用、安全约束和评估指标",
    elements: "方向卡代表研究主题，关系线代表技术依赖，风险标签代表安全、成本或评估问题。",
    controls: "方向筛选、成熟度、应用场景、风险维度和推荐阅读路径",
    observe: "每个方向解决什么问题，依赖哪些基础模块，工程落地卡在哪个瓶颈",
    why: "前沿技术通常是在数据、算力、架构、交互和安全约束变化后出现的新解法。",
    misconception: "前沿不等于离基础很远，很多热点仍然回到训练、注意力、数据和系统工程。",
    engineering: "选研究方向、做技术路线图、读论文和规划项目时需要用前沿地图定位。",
    consoleTask: "在中央控制台选一个前沿主题，把它拆回数据、模型、训练、评估和部署五个基础问题。",
  },
  "part6/paper_reading_lab": {
    what: "这页讲如何用时间线、机制图、最小复现清单和关键实验读懂经典深度学习论文。",
    analogy: "像拆一篇侦探案卷：先看问题背景，再看关键证据，最后复现核心推理。",
    intuition: "读论文不是逐字翻译，而是提取问题、方法、证据和可复现动作。",
    variable: "论文问题、核心机制、实验设置、消融结果、贡献边界和复现步骤",
    elements: "时间线展示研究脉络，机制图展示方法，实验表展示证据，清单展示最小复现。",
    controls: "论文选择、阅读深度、机制视图、实验筛选和复现任务",
    observe: "作者解决了什么旧问题，关键改动在哪里，实验是否支持结论，哪些条件不能外推",
    why: "论文价值来自问题和证据的匹配，不是抽象口号；机制图和复现清单能防止只读摘要。",
    misconception: "引用量高不等于每个结论都适用于你的任务，必须看数据、设置和边界。",
    engineering: "论文复现、技术选型、组会汇报和模型改造都需要结构化阅读。",
    consoleTask: "在中央控制台选一篇经典论文，把贡献拆成机制、实验和最小复现三张卡。",
  },
  "part7/networking": {
    what: "这页讲 TCP、HTTP/HTTPS、DNS 和常见网络面试题，重点是请求如何从客户端到服务端。",
    analogy: "像寄快递：先查地址，再建立运输通道，途中每一站都可能影响时效。",
    intuition: "网络题先画请求链路，再解释协议细节。",
    variable: "DNS 解析、TCP 连接、TLS 握手、HTTP 请求响应、延迟和重传",
    elements: "节点代表客户端、DNS、网关和服务器，箭头代表报文流，状态卡代表连接阶段。",
    controls: "请求负载、DNS 缓存、连接复用、丢包率、TLS 开关和超时设置",
    observe: "慢在哪里，是解析、建连、传输还是服务端处理；失败是否能按层定位",
    why: "一次请求由多层协议共同完成，每一层都有状态和超时，任何一层异常都会影响端到端体验。",
    misconception: "背 TCP 三次握手不够，要能解释为什么需要状态同步和如何排查连接问题。",
    engineering: "模型 API、在线推理、数据下载和分布式训练都离不开网络链路诊断。",
    consoleTask: "在中央控制台把一次模型推理请求画成 DNS、TCP、TLS、HTTP、服务端处理的链路。",
  },
  "part7/database_sql": {
    what: "这页讲 SQL 执行流程、索引、B+树、慢查询排查和数据库高频面试题。",
    analogy: "像图书馆检索：有索引能直达书架，没索引就只能全馆一排排找。",
    intuition: "数据库性能问题先看查询路径和数据访问量。",
    variable: "SQL 语句、执行计划、索引、B+树层级、扫描行数、锁和事务",
    elements: "树节点代表索引层级，执行流程块代表解析优化执行，指标卡代表扫描量和耗时。",
    controls: "WHERE 条件、索引选择、排序字段、表数据量、事务隔离级别和缓存命中",
    observe: "是否走索引，扫描行数是否过大，排序或 join 是否成为瓶颈，慢查询是否可复现",
    why: "数据库优化器会根据统计信息选择访问路径，索引能减少扫描，但也带来写入和维护成本。",
    misconception: "加索引不是万能，低选择性字段、函数包裹字段或错误联合索引可能仍然很慢。",
    engineering: "特征存储、实验记录、用户数据查询和在线服务都需要 SQL 与索引诊断。",
    consoleTask: "在中央控制台把一次慢查询拆成过滤、索引、排序、返回四步，写出第一优化动作。",
  },
  "part7/data_structures": {
    what: "这页讲数组、链表、栈、队列、树、图、排序、BFS/DFS 等数据结构与算法基础。",
    analogy: "像选择收纳方式：抽屉、书架、队列和地图适合取放不同东西。",
    intuition: "数据结构决定操作成本，算法是在这些结构上安排步骤。",
    variable: "时间复杂度、空间复杂度、指针关系、遍历顺序、排序比较和队列/栈状态",
    elements: "节点代表元素，连线代表引用或边，指针代表当前位置，动画步骤代表算法推进。",
    controls: "数据结构类型、输入规模、排序算法、遍历方式和目标节点",
    observe: "每一步访问了谁，额外空间用了多少，规模变大后耗时如何增长",
    why: "不同结构对查找、插入、删除和遍历的成本不同，算法选择必须匹配操作模式。",
    misconception: "只背复杂度不够，要能说出为什么这个结构导致这种复杂度。",
    engineering: "模型服务缓存、任务队列、图学习、检索系统和面试算法题都依赖数据结构。",
    consoleTask: "在中央控制台选一个遍历动画，逐步写出队列或栈在每一轮的变化。",
  },
  "part7/operating_system": {
    what: "这页讲进程线程、调度、虚拟内存、死锁和操作系统面试高频问题。",
    analogy: "像酒店管理：房间、住客、排班、电梯和安全规则都要协调。",
    intuition: "操作系统是在有限资源上隔离、调度和保护程序。",
    variable: "进程、线程、CPU 时间片、内存页、锁、等待图和上下文切换",
    elements: "队列代表调度等待，内存块代表页表映射，锁图代表资源占用和等待关系。",
    controls: "进程数量、线程数量、调度算法、页面大小、锁顺序和资源数量",
    observe: "谁在等待 CPU，谁占用锁，缺页是否增加，死锁是否形成环",
    why: "并发程序共享资源，系统必须在效率和隔离之间取舍；资源顺序不当会造成互相等待。",
    misconception: "进程和线程不是只背定义，要能解释隔离范围、共享资源和切换成本。",
    engineering: "训练进程、多线程数据加载、GPU 作业调度和线上服务稳定性都需要 OS 知识。",
    consoleTask: "在中央控制台模拟多个训练任务竞争资源，写出调度、内存和锁的风险点。",
  },
  "part7/system_design": {
    what: "这页讲 CAP、缓存、消息队列、推荐系统、分布式训练和推理平台的系统设计思路。",
    analogy: "像设计城市交通：道路、车站、仓库和应急方案要一起规划。",
    intuition: "系统设计先定目标和约束，再画链路、找瓶颈、做取舍。",
    variable: "QPS、延迟、吞吐、缓存命中、队列积压、一致性、可用性和扩展方式",
    elements: "服务节点代表组件，箭头代表数据流，指标卡代表容量，风险标签代表单点或瓶颈。",
    controls: "流量规模、缓存策略、队列容量、副本数、一致性要求和故障位置",
    observe: "瓶颈在哪里，故障如何传播，缓存是否降低压力，队列是否引入延迟",
    why: "系统性能由最慢或最拥堵的链路限制，设计要在一致性、延迟、成本和复杂度之间取舍。",
    misconception: "画很多组件不等于好设计，必须说明每个组件解决什么约束。",
    engineering: "推荐系统、模型推理平台、训练平台、特征服务和面试系统设计都用这套框架。",
    consoleTask: "在中央控制台设计一次模型推理平台请求，标出缓存、队列、模型服务和监控点。",
  },
  "part7/deep_learning_interview": {
    what: "这页把深度学习面试中的梯度、归一化、注意力复杂度、训练排查、LoRA 和部署串起来。",
    analogy: "像面试前整理工具包：每个工具都要知道用途、限制和真实项目里怎么用。",
    intuition: "深度学习面试不是背名词，而是解释机制、故障和取舍。",
    variable: "梯度流、BatchNorm/LayerNorm、注意力复杂度、训练诊断、LoRA rank 和部署延迟",
    elements: "题卡代表问题，机制图代表解释路径，追问区代表学习时应该继续追清楚的点。",
    controls: "题目方向、难度、追问类型、是否显示答案和实战场景",
    observe: "回答是否包含定义、直觉、公式/机制、工程用途和常见坑，是否能应对追问",
    why: "面试考的是能否把理论连接到工程问题，例如为什么训练不稳、为什么部署变慢。",
    misconception: "背标准答案很容易被追问打穿，必须能从变量变化解释现象。",
    engineering: "准备算法岗、模型岗、应用岗和项目答辩时，都需要用机制化语言表达深度学习。",
    consoleTask: "在中央控制台抽一道深度学习题，按这十二问补全自己的口头答案。",
  },
  "part7/interview_quiz": {
    what: "这页是面试刷题模式，按方向、难度和错题记录训练高频问题与追问。",
    analogy: "像模拟面试间：不仅出题，还会根据你的回答继续追问薄弱处。",
    intuition: "刷题的核心是暴露薄弱点并把回答练成结构化表达。",
    variable: "题库方向、难度、回答结构、错题本、追问和复习间隔",
    elements: "题目卡显示当前问题，选项或输入区记录回答，追问卡推动更深解释。",
    controls: "方向筛选、难度、随机出题、错题模式、显示提示和复习计划",
    observe: "错题是否反复出现在同类知识点，回答是否能从定义走到工程案例",
    why: "重复检索和追问能逼出真正不会的部分，比被动看答案更能形成可迁移表达。",
    misconception: "刷题不是记答案数量，而是让每道题都能讲出机制、边界和排查步骤。",
    engineering: "面试准备、团队内训、课程验收和自测复盘都适合用刷题模式。",
    consoleTask: "在中央控制台选择一个错题方向，按定义、直觉、工程用途、误区四步重答。",
  },
};

function lessonDomain(module) {
  const text = [module.id, module.title, module.summary, module.level, ...module.tags].join(" ");
  if (module.partKey === "part1") return "foundation";
  if (module.partKey === "part2" || /CNN|卷积|视觉|Grad-CAM|特征图/.test(text)) return "cnn";
  if (module.partKey === "part3" || /RNN|序列|LSTM|GRU|Seq2Seq/.test(text)) return "sequence";
  if (module.partKey === "part4" || /Transformer|注意力|Flash|GNN|GAN|自编码器/.test(text)) return "transformer";
  if (module.partKey === "part5" || /训练|梯度|超参|数据|部署|工具|测验|调参/.test(text)) return "training";
  if (module.partKey === "part6" || /框架|项目|插件|路径|术语|论文|强化学习|前沿|Playground/.test(text)) return "architecture";
  if (module.partKey === "part7" || /面试|网络|数据库|系统|算法|操作系统/.test(text)) return "systems";
  return "foundation";
}

function getLessonProfile(module) {
  const domain = lessonDomain(module);
  const base = DOMAIN_PROFILES[domain];
  const blueprint = DOMAIN_BLUEPRINTS[domain];
  const beginner = BEGINNER_BLUEPRINTS[domain];
  const focus = module.tags.slice(0, 3).join(" / ") || module.partShort;
  const title = module.title;
  const shortSummary = module.summary.replace(/[。.]$/, "");
  return {
    ...base,
    ...blueprint,
    beginner,
    domain,
    focus,
    title,
    core: [
      {
        title: "核心问题",
        body: `${title} 不是孤立名词，本节要回答：${shortSummary} 时，输入、关键变量和输出之间怎样发生因果变化。`,
      },
      {
        title: "机制链路",
        body: `${base.mechanism} 读的时候把“${focus}”放进这条链路，确认每一步对应的对象和变化。`,
      },
      {
        title: "观察指标",
        body: `重点观察 ${blueprint.variable}。如果 ${blueprint.signal}，说明你抓到了本节真正的可视化信号。`,
      },
      {
        title: "工程落点",
        body: blueprint.transfer,
      },
    ],
    practice: `${blueprint.lab} 写下“我改了什么、画面哪里变了、这说明 ${title} 的哪个机制在起作用”。`,
  };
}

function centralConsoleModule(module) {
  const consoleModule = byId("part6/neural_network_playground");
  const trainingModule = byId("part6/training_demo");
  if (module.id === "part6/neural_network_playground") return trainingModule || consoleModule || module;
  return consoleModule || trainingModule || module;
}

function renderLessonBrief(module) {
  const profile = getLessonProfile(module);
  return `
    <section class="reading-section lesson-brief">
      <div class="section-kicker">${escapeHtml(profile.label)} · ${escapeHtml(profile.focus)}</div>
      <h2>这一节真正要掌握什么</h2>
      <div class="lesson-grid">
        ${profile.core.map((item) => `<article class="lesson-card"><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.body)}</p></article>`).join("")}
      </div>
    </section>
  `;
}

function renderZeroBasics(module) {
  const profile = getLessonProfile(module);
  const beginner = profile.beginner;
  const labGuide = LAB_CONTROL_GUIDES[profile.domain];
  const note = MODULE_TEACHING_NOTES[module.id] || {};
  const tags = module.tags.join("、") || module.partShort;
  const shortSummary = module.summary.replace(/[。.]$/, "");
  const what = note.what || `${module.title} 是 ${module.partShort} 里的一个 ${module.level} 学习单元，专门解释“${shortSummary}”。`;
  const analogy = note.analogy || beginner.analogy;
  const intuition = note.intuition || beginner.intuition;
  const variable = note.variable || profile.variable;
  const elements = note.elements || beginner.elements;
  const controls = note.controls || labGuide.controls;
  const observe = note.observe || labGuide.changes;
  const why = note.why || `${beginner.why} ${profile.mechanism}`;
  const misconception = note.misconception || `${beginner.misuse} ${profile.pitfalls[0]}`;
  const engineering = note.engineering || beginner.engineering;
  const consoleTask = note.consoleTask || beginner.consoleTask;
  const concreteCase = `${module.title} 的最小案例：把“${module.summary.replace(/[。.]$/, "")}”放进一个真实任务，先观察 ${observe}，再判断 ${profile.signal} 是否支持你的解释。`;
  const errorCase = `${module.title} 的典型错误：${profile.pitfalls[0]} 出现时，先回到实验区只改 ${controls.split(/[、,，/]/)[0] || "第一个控件"}，确认问题是变量导致还是理解跳步。`;
  const items = [
    {
      title: "这是什么？",
      body: `${what} 先把它当成一个可验证问题：给定 ${tags} 相关输入，模型或系统经过什么机制，最后输出什么可观察结果。`,
      task: `自测：不用术语复述一遍“我为什么要学 ${module.title}”，并指出它解决的是输入、变换、输出还是诊断中的哪一段。`,
    },
    {
      title: "生活类比",
      body: `${analogy} 放到本节，就是把“${module.title}”看成一个可操作场景：你改变一个条件，观察结果怎样连锁变化。`,
      task: "读法：先用类比建立画面感，再回到页面里的变量、控件和图形元素，不要把类比当成最终答案。",
    },
    {
      title: "一句话直觉",
      body: `${intuition} 对这页来说，直觉不是背概念，而是先找到“${variable}”里的关键量，再看它怎样改变图像、曲线或状态卡。`,
      task: `自测：把直觉压缩成一句“${profile.focus} 改变时，画面里的某个信号会跟着改变”的因果句。`,
    },
    {
      title: "严谨定义",
      body: `${module.title} 可以严谨地看作：围绕 ${variable} 建立输入、状态变换、输出和诊断指标之间的对应关系。${beginner.definition}`,
      task: "读法：先看定义里的对象，再看对象之间的箭头；能画出这条链路，才算不是只背名词。",
    },
    {
      title: "图中每个元素代表什么",
      body: `${elements} 本页额外要把标题、坐标轴、节点、曲线、柱状条和指标卡都当成证据，不要只看最醒目的图形。`,
      task: "操作：鼠标停在图形区域前，先逐个点名每个元素代表的对象；说不清的元素，就是下一步要补的知识点。",
    },
    {
      title: "颜色/亮度/方向/速度代表什么",
      body: `${beginner.visualEncoding} 在 ${module.title} 的演示里，颜色和亮度通常表示强弱，方向表示信息、参数或请求的流动，动画速度表示变化发生的节奏。观察时要把这些视觉信号和“${variable}”对应起来。`,
      task: "观察：只盯一个视觉编码，例如亮度或方向，看它是否随着控件变化稳定改变，避免同时追太多信号。",
    },
    {
      title: "用户应该调哪个参数",
      body: `这页先调 ${controls}。如果你只调一个，优先选择第一个能明显改变图像或指标的控件，再把变化写成一句因果关系。`,
      task: "操作：一次只调一个参数，先试默认值，再试一个极小值和一个极大值；不要同时拖多个控件，否则很难归因。",
    },
    {
      title: "观察什么变化",
      body: `重点观察：${observe}。同时对照读数区，看文字解释是否和画面变化一致；如果不一致，先怀疑自己没把控件、变量和结果连成同一条链。`,
      task: "记录：写下“调参前是什么样、调参后哪里变了、变化方向是否符合预期”三句话，训练真正的观察能力。",
    },
    {
      title: "为什么会这样",
      body: `${why} 具体到本节，控件变化会先改动“${variable}”中的某个量，再传导到图形、指标或源码行为。`,
      task: "追问：不要停在“它变了”，继续问变化从哪个变量开始，经过哪一步传导，最后落到哪个可见结果。",
    },
    {
      title: "常见误区",
      body: `${misconception} 本节还要避免：${profile.pitfalls[0]} 如果说不出“我调了什么，为什么变”，就说明还停留在看热闹。`,
      task: "检查：看完页面后故意说一个错误解释，再用图形、读数或定义把它纠正回来，这比直接背正确答案更稳。",
    },
    {
      title: "工程用途",
      body: `${engineering} 学完 ${module.title} 后，至少要能说明它在真实项目里能帮你做什么判断：调参、排错、选结构、解释结果或设计链路。`,
      task: "迁移：把这一节连接到一个真实任务，例如训练不收敛、结果解释不清、系统变慢或模型结构难选，并说出第一步该查什么。",
    },
    {
      title: "去中央控制台实战",
      body: `${consoleTask} 当前页建议从“${module.title}”的关键变量开始，把 ${controls} 中至少一个参数迁移到中央控制台里复现实验。`,
      task: "操作：先在本页写下一个假设，再打开中央控制台复现；如果控制台里的曲线、形状或指标变化能验证这个假设，才算把这一节学成了自己的经验。",
      actionHref: consoleHref(module),
      actionLabel: "去控制台验证这个假设",
    },
  ];

  return `
    <section class="reading-section zero-basics" data-zero-basics>
      <div class="section-kicker">零基础导读</div>
      <h2>把刚才的动画讲明白</h2>
      <p class="summary">先看上面的动画和实验，再按这 12 个问题复述。每一项都必须回到刚才看到的颜色、方向、控件或读数。</p>
      <div class="zero-basics-case-strip">
        <article><span>本节案例</span><p>${escapeHtml(concreteCase)}</p></article>
        <article><span>错误样本</span><p>${escapeHtml(errorCase)}</p></article>
        <article><span>工程场景</span><p>${escapeHtml(engineering)}</p></article>
      </div>
      <div class="zero-basics-grid">
        ${items.map((item, index) => `
          <article class="zero-basics-card">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <h3>${escapeHtml(item.title)}</h3>
            <p>${escapeHtml(item.body)}</p>
            <small>${escapeHtml(item.task)}</small>
            ${item.actionHref ? `<a class="zero-basics-card-action action" href="${item.actionHref}">${escapeHtml(item.actionLabel)}</a>` : ""}
          </article>
        `).join("")}
      </div>
      <div class="zero-basics-action">
        <div>
          <span>练习</span>
          <p>${escapeHtml(`为什么现在去控制台：你已经看过动画和实验，下一步要把“${module.title}”的关键变量迁移到统一实验台，验证自己能不能预测画面和指标变化。`)}</p>
        </div>
        <a class="action" href="${consoleHref(module)}">去控制台验证这个假设</a>
      </div>
    </section>
  `;
}

function renderDryGoods(module) {
  const profile = getLessonProfile(module);
  const note = MODULE_TEACHING_NOTES[module.id] || {};
  const labGuide = LAB_CONTROL_GUIDES[profile.domain];
  const controls = note.controls || labGuide.controls;
  const observe = note.observe || labGuide.changes;
  const variable = note.variable || profile.variable;
  const why = note.why || profile.mechanism;
  const misconception = note.misconception || profile.pitfalls[0];
  const engineering = note.engineering || profile.transfer;
  const consoleTask = note.consoleTask || profile.practice;
  const sourceName = module.sourcePath.split("/").pop();
  const cards = [
    {
      title: "机制骨架",
      claim: note.what || `${module.title} 要回答 ${module.summary}`,
      bullets: [
        `输入：先定位 ${profile.focus} 相关的数据、状态或请求。`,
        `变换：追踪 ${variable} 怎样被函数、层、协议或训练循环改写。`,
        `输出：用图像、曲线、指标或读数验证变化是否真的发生。`,
        `最小闭环：能说清“输入对象 -> 中间变量 -> 可见结果”，才算读懂这一节。`,
        `反向追踪：看到异常结果时，从输出倒推到变量和输入，不要只改默认参数。`,
      ],
    },
    {
      title: "必看变量",
      claim: `本节不要泛泛看图，先盯住 ${variable}。`,
      bullets: [
        `先调：${controls}。`,
        `再看：${observe}。`,
        `只改一个变量，写出“变量变了 -> 中间机制变了 -> 结果变了”的因果句。`,
        `记录默认值、极小值和极大值三组结果，确认现象不是偶然动画效果。`,
        `如果读数没有变化，先检查控件是否真的影响了模型、图形或源码中的变量。`,
      ],
    },
    {
      title: "为什么会这样",
      claim: why,
      bullets: [
        `把页面读数和动画变化对上，不接受“看起来差不多”的解释。`,
        `如果极端参数下现象不符合预期，优先检查 shape、尺度、mask、数据切分或缓存命中这类边界条件。`,
        `能反向解释失败现象，才说明这节不是只看懂了默认演示。`,
        `把机制写成一句可检验预测：当某个变量升高或降低时，哪个指标应该随之改变。`,
        `再用源码对照预测是否成立，找到负责这个变化的函数、数组、层或状态字段。`,
      ],
    },
    {
      title: "判错清单",
      claim: misconception,
      bullets: [
        profile.pitfalls[0],
        profile.pitfalls[1] || `只记住 ${module.title} 的结论，没有说出触发条件。`,
        `看完后必须能指出一个“什么时候这招会失效”的场景。`,
        `不要把可视化当成结论本身；图只是证据，结论必须能回到变量、公式或代码。`,
        `不要跳过边界条件：batch、维度、归一化、mask、缓存、并发和随机种子都可能改变现象。`,
      ],
    },
    {
      title: "工程落地",
      claim: engineering,
      bullets: [
        `第一排查项：${profile.signal}。`,
        `真实项目里，把它用于调参、排错、解释结果、选结构或设计链路中的至少一个判断。`,
        `不要只写“提升效果”，要写清楚提升的是收敛、泛化、可解释性、吞吐、延迟还是可维护性。`,
        `形成操作顺序：先定位现象，再锁定变量，再做单因素实验，最后把结论写进实验记录。`,
        `把本节迁移到一个具体场景：训练失败、结果异常、结构难选、系统变慢或面试追问。`,
      ],
    },
    {
      title: "源码抓手",
      claim: `源码入口是 ${sourceName}，不要只看页面，至少对照一个 compute / render / smoke 或核心函数。`,
      bullets: [
        `先找数据从哪里来，再找参数在哪里进函数，最后找输出在哪里被渲染。`,
        `看不懂整段代码时，先抄出输入 shape、输出 shape 和关键超参数。`,
        consoleTask,
        `如果是旧教材迁移章节，对照讲义 Markdown 和 Python 源码，确认文字解释、公式和演示变量没有脱节。`,
        `最后跑一次中央控制台，把同一个变量换到实验台里复现，避免只在静态页面上“读懂”。`,
      ],
    },
  ];

  return `
    <section class="reading-section dry-goods" data-dry-goods>
      <div class="section-kicker">Hard Notes</div>
      <h2>知识点硬核笔记</h2>
      <p class="summary">这部分专门回答“到底有什么用、该看什么、哪里会错”。先读这里，再去拖动画和看旧讲义。</p>
      <div class="dry-goods-grid">
        ${cards.map((card, index) => `
          <article class="dry-goods-card">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <h3>${escapeHtml(card.title)}</h3>
            <p class="dry-goods-claim">${escapeHtml(card.claim)}</p>
            <ul class="dry-goods-list">
              ${card.bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>
          </article>
        `).join("")}
      </div>
    </section>
  `;
}

function renderConceptAnimation(module) {
  const profile = getLessonProfile(module);
  const isAttention = profile.demoKind === "attention";
  const attentionTokens = ["深度", "学习", "需要", "注意力", "机制"];
  const focusOptions = isAttention ? attentionTokens : profile.steps;
  return `
    <section class="reading-section concept-demo" data-demo="${profile.demoKind}">
      <div class="lab-title-row">
        <div>
          <div class="eyebrow">Concept Animation</div>
          <h2>${escapeHtml(profile.label)}演示</h2>
        </div>
        <span class="lab-badge">${escapeHtml(profile.demoKind)}</span>
      </div>
      <p class="summary">${escapeHtml(profile.mechanism)}</p>
      <div class="demo-grid">
        <div class="lab-controls">
          <label class="lab-control">${isAttention ? "注意力锐度" : "强度"} <input type="range" min="1" max="10" step="1" value="6" data-demo-control="intensity"></label>
          <label class="lab-control">${isAttention ? "上下文噪声" : "复杂度"} <input type="range" min="1" max="8" step="1" value="4" data-demo-control="complexity"></label>
          <label class="lab-control">${isAttention ? "Query token" : "观察重点"}
            <select data-demo-control="focus">
              ${focusOptions.map((step, index) => `<option value="${index}">${escapeHtml(step)}</option>`).join("")}
            </select>
          </label>
        </div>
        <div class="demo-stage" data-demo-stage></div>
      </div>
      <p class="lab-readout" data-demo-readout></p>
    </section>
  `;
}

function renderKnowledgeSections(module) {
  const profile = getLessonProfile(module);
  return `
    <section class="reading-section knowledge-section">
      <h2>读图顺序与工程判断</h2>
      <div class="knowledge-columns">
        <article>
          <div class="section-kicker">Reading Order</div>
          <ol class="knowledge-list">
            ${profile.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
            <li>${escapeHtml(`把 ${module.title} 的结论重新说成一个输入到输出的过程。`)}</li>
            <li>${escapeHtml(`指出 ${profile.focus} 中至少一个变量变大或变小时，图像或指标会怎样改变。`)}</li>
          </ol>
        </article>
        <article>
          <div class="section-kicker">Common Traps</div>
          <ul class="knowledge-list">
            ${profile.pitfalls.map((pitfall) => `<li>${escapeHtml(pitfall)}</li>`).join("")}
            <li>${escapeHtml(`只记住“${module.title}”这个标题，却说不出它在真实项目里改变了哪个决策。`)}</li>
            <li>${escapeHtml(`把演示图当成装饰图，没有把控制项和结果变化连成因果句。`)}</li>
          </ul>
        </article>
      </div>
      <div class="practice-callout">
        <strong>本节练习</strong>
        <p>${escapeHtml(profile.practice)}</p>
      </div>
    </section>
  `;
}

function isLLMCookbookRelevant(module) {
  const keywords = [
    "Transformer",
    "NLP",
    "LLM",
    "Agents",
    "安全",
    "部署",
    "训练",
    "调试",
    "系统设计",
    "深度学习",
    "论文",
    "路径",
    "术语",
  ];
  return module.partKey === "part4"
    || module.id === "part6/frontier"
    || module.id === "part6/glossary"
    || module.id === "part6/learning_path"
    || module.id === "part6/paper_reading_lab"
    || module.id === "part5/deployment_tools"
    || module.id === "part5/data_training"
    || module.id === "part7/networking"
    || module.id === "part7/system_design"
    || module.id === "part7/deep_learning_interview"
    || module.tags.some((tag) => keywords.includes(tag));
}

const LLM_TRACKS_BY_MODULE = {
  "part5/deployment_tools": ["Gradio / App Delivery", "Evaluation & Debugging", "Chat System", "RAG 问答", "Agent & Tools", "Embedding & Search"],
  "part5/data_training": ["Evaluation & Debugging", "Embedding & Search", "RAG 问答", "Fine-tuning / LoRA", "Gradio / App Delivery", "Agent & Tools"],
  "part6/frontier": ["RAG 问答", "Agent & Tools", "Evaluation & Debugging", "Fine-tuning / LoRA", "Embedding & Search", "Gradio / App Delivery"],
  "part6/glossary": ["Prompt Engineering", "RAG 问答", "Embedding & Search", "Fine-tuning / LoRA", "Agent & Tools", "Gradio / App Delivery"],
  "part6/learning_path": ["Prompt Engineering", "Chat System", "RAG 问答", "Embedding & Search", "Evaluation & Debugging", "Agent & Tools"],
  "part6/paper_reading_lab": ["Evaluation & Debugging", "RAG 问答", "Embedding & Search", "Fine-tuning / LoRA", "Agent & Tools", "Prompt Engineering"],
  "part7/networking": ["Chat System", "RAG 问答", "Embedding & Search", "Evaluation & Debugging", "Agent & Tools", "Gradio / App Delivery"],
  "part7/system_design": ["Chat System", "RAG 问答", "Agent & Tools", "Evaluation & Debugging", "Gradio / App Delivery", "Embedding & Search"],
  "part7/deep_learning_interview": ["Fine-tuning / LoRA", "Evaluation & Debugging", "Prompt Engineering", "RAG 问答", "Embedding & Search", "Agent & Tools"],
};

const LLM_STUDY_ORDER = [
  {
    title: "必修 1：Prompt 与基础任务",
    body: "先掌握总结、抽取、改写、分类、结构化输出这些基础任务。产出物不是一堆 prompt，而是一组可复用的任务模板和边界测试题。",
  },
  {
    title: "必修 2：对话系统与状态管理",
    body: "再学习如何组织 system/user/tool/history，理解上下文窗口、历史压缩、后处理和安全边界。产出物是一张完整请求链路图。",
  },
  {
    title: "必修 3：LangChain 式应用组织",
    body: "把模型调用、提示模板、解析器、检索器和工具封装成可组合模块。产出物是一条可替换模型、可替换数据源的最小应用流水线。",
  },
  {
    title: "必修 4：私有数据问答",
    body: "进入 RAG：文档加载、切块、向量化、召回、重排、引用和答案合成。产出物是一套能解释证据来源的课程问答原型。",
  },
  {
    title: "选修推进：评估、微调、Agent 与交付",
    body: "当原型能跑后，再做评估调试、语义检索优化、LoRA/微调决策、工具调用 Agent 和 Web Demo 交付。产出物要能回归、能复盘、能解释每一次取舍。",
  },
];

function getLLMCookbookTracks(module) {
  if (!module) return LLM_COOKBOOK_TRACKS;
  const preferred = LLM_TRACKS_BY_MODULE[module.id];
  if (preferred) {
    return preferred
      .map((title) => LLM_COOKBOOK_TRACKS.find((track) => track.title === title))
      .filter(Boolean);
  }
  return LLM_COOKBOOK_TRACKS.filter((track) => (
    track.route === module.id
    || module.partKey === "part4"
    || module.tags.some((tag) => ["部署", "训练", "系统设计", "深度学习"].includes(tag))
  )).slice(0, 6);
}

function renderLLMStudyOrder() {
  return `
    <div class="llm-roadmap-panel">
      ${LLM_STUDY_ORDER.map((item, index) => `
        <article>
          <span>${String(index + 1).padStart(2, "0")}</span>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.body)}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function renderLLMDetailList(items) {
  return `
    <ul class="llm-detail-list">
      ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderLLMCookbookBridge(module = null) {
  const tracks = getLLMCookbookTracks(module);
  const intro = module
    ? `把 ${module.title} 接到 LLM 应用开发：先判断它属于提示、检索、评估、微调、工具调用还是交付链路。`
    : "参考 Datawhale LLM Cookbook 的学习路线重新组织成本项目的工程入口：从 prompt 到 RAG、评估、微调、Agent 和应用交付。建议按顺序完成每一段的产出物，再回到相关课程页做对照实验，最终形成可复盘的项目记录。";

  return `
    <section ${module ? "" : 'id="llm-cookbook"'} class="${module ? "reading-section" : "view section"} llm-cookbook-section" data-llm-cookbook>
      <div class="section-head">
        <div>
          <div class="eyebrow">LLM Cookbook Track</div>
          <h2>${module ? "LLM 应用开发接线图" : "LLM 应用开发路线"}</h2>
        </div>
        <p>${escapeHtml(intro)}</p>
      </div>
      ${module ? "" : renderLLMStudyOrder()}
      <div class="llm-track-grid">
        ${tracks.map((track) => `
          <article class="llm-track-card">
            <span>${escapeHtml(track.stage)}</span>
            <h3>${escapeHtml(track.title)}</h3>
            <p>${escapeHtml(track.summary)}</p>
            <dl>
              <div><dt>观察</dt><dd>${escapeHtml(track.observe)}</dd></div>
              <div><dt>实战</dt><dd>${escapeHtml(track.practice)}</dd></div>
              <div><dt>核心直觉</dt><dd>${escapeHtml(track.concept)}</dd></div>
              <div><dt>工作流</dt><dd>${renderLLMDetailList(track.workflow)}</dd></div>
              <div><dt>常见失败</dt><dd>${escapeHtml(track.failure)}</dd></div>
              <div><dt>验收标准</dt><dd>${escapeHtml(track.acceptance)}</dd></div>
              <div><dt>落地检查</dt><dd>${renderLLMDetailList(track.checklist)}</dd></div>
              <div><dt>接到本站</dt><dd>${escapeHtml(track.drill)}</dd></div>
            </dl>
            <a class="ghost-action" href="#course/${encodeURIComponent(track.route)}">连接课程</a>
          </article>
        `).join("")}
      </div>
      <p class="source-note">知识体系参考 <a href="https://github.com/datawhalechina/llm-cookbook" target="_blank" rel="noreferrer">datawhalechina/llm-cookbook</a>，本页为面向本站课程结构的自写整理。</p>
    </section>
  `;
}

const LEGACY_NOTE_MODULES = {
  part1_foundations: new Set(["01_tensors_gradients", "02_activations_normalization", "03_datasets_optimizers"]),
  part2_cnn: new Set([
    "01_convolution_visual",
    "02_feature_maps",
    "03_classic_architectures",
    "04_debug_panel",
    "05_mnist_toy",
    "06_modern_architectures",
    "07_advanced_convolution",
    "08_visualization_gradcam",
    "09_transfer_learning",
  ]),
  part3_rnn: new Set([
    "01_rnn_intuition",
    "02_hidden_states",
    "03_sequence_toys",
    "04_hyperparam_rnn",
    "05_seq2seq_attention",
    "06_text_classification",
    "07_advanced_training",
    "08_debug_problems",
  ]),
  part4_transformer: new Set([
    "01_attention_mechanism",
    "02_multihead_visual",
    "03_encoder_decoder",
    "04_minimal_transformer",
    "05_flash_attention",
    "06_debug_problems",
  ]),
  part5_toolbox: new Set([
    "01_feature_visualization",
    "02_gradient_monitor",
    "03_training_dynamics",
    "04_hyperparam_search",
    "05_dataset_toys",
  ]),
  part6_universal_framework: new Set([
    "01_unified_interface",
    "02_modular_structure",
    "03_full_project",
    "04_plugin_system",
    "05_one_click_training",
    "06_streamlit_demo",
    "07_project_template",
  ]),
};

const LEGACY_NOTE_ALIASES = {
  part1_foundations: {
    math_primer: "part1_foundations/01_tensors_gradients",
    machine_learning_basics: "part1_foundations/03_datasets_optimizers",
    neural_network_basics: "part1_foundations/02_activations_normalization",
    classical_ml: "part1_foundations/03_datasets_optimizers",
  },
  part2_cnn: {
    cnn_architectures: "part2_cnn/03_classic_architectures",
    advanced_cnn: "part2_cnn/07_advanced_convolution",
  },
  part3_rnn: {
    sequence_models: "part3_rnn/03_sequence_toys",
  },
  part4_transformer: {
    transformer_models: "part4_transformer/04_minimal_transformer",
  },
  part5_toolbox: {
    data_training: "part5_toolbox/03_training_dynamics",
    case_studies: "part5_toolbox/05_dataset_toys",
    deployment_tools: "part6_universal_framework/05_one_click_training",
    quiz_system: "part4_transformer/06_debug_problems",
    tuning_challenge: "part5_toolbox/04_hyperparam_search",
  },
  part6_universal_framework: {
    neural_network_playground: "part6_universal_framework/03_full_project",
    training_demo: "part6_universal_framework/05_one_click_training",
    reinforcement_learning: "part5_toolbox/05_dataset_toys",
    learning_path: "part6_universal_framework/07_project_template",
    glossary: "part6_universal_framework/01_unified_interface",
    frontier: "part4_transformer/05_flash_attention",
    paper_reading_lab: "part4_transformer/01_attention_mechanism",
  },
};

function legacyMarkdownKey(module) {
  if (LEGACY_NOTE_MODULES[module.partDir]?.has(module.module)) {
    return `${module.partDir}/${module.module}`;
  }
  return LEGACY_NOTE_ALIASES[module.partDir]?.[module.module] || "";
}

function legacyMarkdownCandidates(module) {
  const key = legacyMarkdownKey(module);
  if (!key) return [];
  return [`deep_learning_book/${key}.md`, `docs/legacy_book/${key}.md`];
}

function legacyMarkdownPath(module) {
  return legacyMarkdownCandidates(module)[0] || "";
}

function hasLegacyMarkdown(module) {
  return legacyMarkdownCandidates(module).length > 0;
}

function moduleCredibility(module) {
  if (CONTENT_CREDIBILITY[module.id]) return CONTENT_CREDIBILITY[module.id];
  return credibilityProfileForModule(module);
}

function credibilityProfileForModule(module) {
  if (module.partKey === "part1") return CREDIBILITY_PROFILES.foundation;
  if (module.partKey === "part2") return CREDIBILITY_PROFILES.cnn;
  if (module.partKey === "part3") return CREDIBILITY_PROFILES.rnn;
  if (module.partKey === "part4") return CREDIBILITY_PROFILES.transformer;
  if (module.partKey === "part5") return CREDIBILITY_PROFILES.toolbox;
  if (module.partKey === "part6") return CREDIBILITY_PROFILES.framework;
  if (module.partKey === "part7") return CREDIBILITY_PROFILES.cs;
  return CONTENT_CREDIBILITY.default;
}

function renderSourceReferences(sourceIds) {
  return sourceIds
    .map((id) => {
      const source = SOURCE_LIBRARY[id];
      if (!source) return "";
      return `
        <li>
          <a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(id)} · ${escapeHtml(source.title)}</a>
          <span>${escapeHtml(source.authors)}</span>
        </li>
      `;
    })
    .join("");
}

function renderCredibilitySection(module) {
  const credibility = moduleCredibility(module);
  const levelClass = `level-${credibility.level.toLowerCase()}`;
  return `
    <section id="course-credibility" class="reading-section credibility-section course-anchor-section" data-content-credibility>
      <div class="credibility-head">
        <div>
          <div class="section-kicker">内容可信度</div>
          <h2>这页内容可信到什么程度？</h2>
        </div>
        <span class="credibility-badge ${levelClass}">${escapeHtml(credibility.level)} · ${escapeHtml(credibility.label)}</span>
      </div>
      <p class="summary">${escapeHtml(credibility.summary)}</p>
      <div class="credibility-grid">
        <article>
          <strong>边界说明</strong>
          <ul>
            ${credibility.boundaries.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
          </ul>
        </article>
        <article>
          <strong>本页参考来源</strong>
          <ol class="reference-list">
            ${renderSourceReferences(credibility.sources)}
          </ol>
          <p class="source-note">来源用于校对定义、公式、历史和 API 边界；本站文字为重新整理的学习讲义，不复制外部书籍正文。</p>
        </article>
      </div>
    </section>
  `;
}

function renderLessonDeepDiveShell(module) {
  return `
    <section class="reading-section lesson-deep-dive" data-lesson-notes data-legacy-path="${legacyMarkdownPath(module)}">
      <div class="section-kicker">章节讲义</div>
      <h2>先看 3 分钟版，再看完整精读</h2>
      <p class="summary">正在恢复讲义。加载后先读“知识点全量索引”，确认自己该看什么，再展开完整卡片。</p>
    </section>
  `;
}

function renderImmediateThreeMinuteBrief(module) {
  return `
    <div data-testid="lesson-three-minute">
      ${renderThreeMinuteBrief(buildSyntheticLessonSections(module), module)}
    </div>
  `;
}

function cleanMarkdownText(value) {
  return String(value)
    .replace(/!\[[^\]]*]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/^#+\s*/, "")
    .replace(/^\s*[-*+]\s+/, "")
    .replace(/^\s*\d+[.)]\s+/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function trimText(value, maxLength = 260) {
  const text = cleanMarkdownText(value);
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}...` : text;
}

function isCurrentSiteLessonText(value) {
  const text = cleanMarkdownText(value);
  if (!text || text.length < 8) return false;
  if (/(<style|<\/|unsafe_allow_html|st\.|streamlit|python main\.py|page_title|set_page_config|layout=|padding|border|background|rgba|linear-gradient|font-size|@keyframes|animation:|display:flex|box-shadow|\.css|\.html|localhost:8501)/i.test(text)) {
    return false;
  }
  return true;
}

function trimCodeBlock(value) {
  return String(value).trim();
}

function buildSyntheticLessonSections(module) {
  const profile = getLessonProfile(module);
  const tagLine = module.tags.length ? module.tags.join(" / ") : module.partShort;
  return [
    {
      title: "本节问题",
      paragraphs: [
        `${module.title} 要解决的不是一个孤立概念，而是：${module.summary}。学习时先把它翻译成“输入是什么、经过什么机制、输出为什么改变”。`,
        profile.thesis,
      ],
      bullets: [
        `把关键词 ${tagLine} 放回真实任务，而不是只背定义。`,
        `能用一句因果话解释：当关键变量变化时，结果会怎样变化。`,
      ],
      codeBlocks: [],
    },
    {
      title: "机制拆解",
      paragraphs: [
        profile.mechanism,
        `本页的演示控制项对应 ${profile.variable}，拖动它们时要观察画面、指标和读数是否一起改变。`,
      ],
      bullets: profile.steps.slice(0, 4),
      codeBlocks: [],
    },
    {
      title: "观察指标",
      paragraphs: [
        `判断是否真正理解这一节，关键看 ${profile.signal}。如果只看标题或最终结论，很容易错过中间机制。`,
        `建议先看动画，再读源码对照：动画给因果直觉，源码告诉你这个因果关系在函数、参数和数据结构里落在哪里。`,
      ],
      bullets: profile.pitfalls.slice(0, 3),
      codeBlocks: [],
    },
    {
      title: "迁移练习",
      paragraphs: [
        profile.transfer,
        `完成后写下三句话：我调整了哪个变量，画面或指标如何变化，这说明 ${module.title} 的哪个机制在起作用。`,
      ],
      bullets: [
        "先复述问题，再复述机制，最后复述工程判断。",
        "把极端参数也试一遍，确认自己知道什么时候会失败。",
      ],
      codeBlocks: [],
    },
  ];
}

function lessonSectionDrills(section, module) {
  const profile = getLessonProfile(module);
  const sectionName = section.title || module.title;
  return [
    `先定位变量：把“${sectionName}”落到 ${profile.variable} 中的一个具体量，不要只停在标题。`,
    `再做单因素实验：只改一个控件或源码参数，观察 ${profile.signal} 是否按预期变化。`,
    `最后回到中央控制台或源码：写下“输入 -> 机制 -> 输出 -> 失败条件”四段式解释。`,
  ];
}

function renderDeepDiveCards(sections, module) {
  return sections.map((section, index) => `
    <article id="deep-dive-${index}" class="deep-dive-card">
      <h3>${escapeHtml(section.title)}</h3>
      ${section.paragraphs.slice(0, 3).map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("")}
      ${section.bullets.length ? `<ul>${section.bullets.slice(0, 6).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
      <div class="deep-dive-drill">
        <strong>这一节怎么学</strong>
        <ol>
          ${lessonSectionDrills(section, module).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ol>
      </div>
      ${section.codeBlocks.length ? `<pre class="lesson-code"><code>${escapeHtml(section.codeBlocks[0])}</code></pre>` : ""}
    </article>
  `).join("");
}

function knowledgePointText(section, module) {
  const firstParagraph = section.paragraphs.find(Boolean) || "";
  const firstBullet = section.bullets.find(Boolean) || "";
  const detail = firstParagraph || firstBullet || `${module.title} 的一个核心检查点，需要能说清输入、变量、输出和失败条件。`;
  return trimText(detail, 240);
}

function renderKnowledgePointIndex(sections, module) {
  const profile = getLessonProfile(module);
  const seen = new Set();
  const points = [];
  const add = (title, detail, source, target = "course-reading", action = "看讲义") => {
    const cleanTitle = trimText(title, 64);
    const cleanDetail = trimText(detail, 260);
    const key = `${cleanTitle}|${cleanDetail}`;
    if (!cleanTitle || seen.has(key)) return;
    seen.add(key);
    points.push({ title: cleanTitle, detail: cleanDetail, source, target, action });
  };

  sections.forEach((section, index) => {
    add(section.title, knowledgePointText(section, module), "讲义", `deep-dive-${index}`, "展开精读");
    section.bullets.forEach((bullet) => add(
      bullet,
      `把这一条落到 ${profile.variable}：它改变了哪个变量、影响了哪个中间状态、最后让图形或指标出现什么变化。`,
      "要点",
      "course-reading",
      "看 3 分钟版",
    ));
  });
  profile.steps.forEach((step) => add(
    step,
    `把这一步落到本页动画、读数或源码中的一个具体对象，并写出“我看见了什么、它为什么变”的观察句。`,
    "读图",
    "course-animation",
    "跳到动画",
  ));
  profile.pitfalls.forEach((pitfall) => add(
    pitfall,
    "这是本节必须能识别并纠正的误区；纠正时要引用页面读数、源码变量或讲义中的机制说明。",
    "误区",
    "course-lab",
    "做实验验证",
  ));

  return `
    <div class="knowledge-point-index" data-knowledge-points>
      <div>
        <span class="section-kicker">可点击学习目录</span>
        <h3>知识点全量索引</h3>
        <p>下面每一项都能跳到对应的动画、实验或精读段落。先点“跳到动画”看现象，再点“展开精读”补原理。</p>
      </div>
      <ol>
        ${points.map((point) => `
          <li>
            <span>${escapeHtml(point.source)}</span>
            <strong>${escapeHtml(point.title)}</strong>
            <p>${escapeHtml(point.detail)}</p>
            <button type="button" data-course-scroll="${escapeHtml(point.target)}">${escapeHtml(point.action)}</button>
          </li>
        `).join("")}
      </ol>
    </div>
  `;
}

function renderThreeMinuteBrief(sections, module) {
  const profile = getLessonProfile(module);
  const plan = moduleLearningPlan(module);
  const first = sections[0];
  const firstPoint = first ? knowledgePointText(first, module) : profile.thesis;
  return `
    <div class="three-minute-brief">
      <article>
        <span>01 先看现象</span>
        <strong>${escapeHtml(firstPoint)}</strong>
        <p>${escapeHtml(`回到概念动画，先盯住 ${profile.signal}，不要同时追所有图形。`)}</p>
        <button type="button" data-course-scroll="course-animation">跳到动画</button>
      </article>
      <article>
        <span>02 再做单因素实验</span>
        <strong>${escapeHtml(`本节关键变量：${profile.variable}`)}</strong>
        <p>${escapeHtml(`用 ${plan.duration} 完成一轮默认值、极小值、极大值对比。`)}</p>
        <button type="button" data-course-scroll="course-lab">跳到实验</button>
      </article>
      <article>
        <span>03 最后说清完成标准</span>
        <strong>${escapeHtml(plan.completion)}</strong>
        <p>${escapeHtml(`隔天复习动作：${plan.review}`)}</p>
        <button type="button" data-course-scroll="course-source">看源码对照</button>
      </article>
    </div>
  `;
}

function parseLegacyMarkdown(markdown, module) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const parsed = {
    title: module.title,
    outline: [],
    sections: [],
  };
  let current = null;
  let paragraph = [];
  let inCode = false;
  let codeLines = [];

  const ensureSection = () => {
    if (!current) current = { title: "核心导读", paragraphs: [], bullets: [], codeBlocks: [] };
    return current;
  };
  const flushParagraph = () => {
    if (!paragraph.length) return;
    const text = trimText(paragraph.join(" "), 360);
    if (isCurrentSiteLessonText(text)) ensureSection().paragraphs.push(text);
    paragraph = [];
  };
  const pushSection = () => {
    flushParagraph();
    if (!current) return;
    const hasContent = current.paragraphs.length || current.bullets.length || current.codeBlocks.length;
    if (hasContent) parsed.sections.push(current);
    current = null;
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);

    if (trimmed.startsWith("```")) {
      if (!inCode) {
        flushParagraph();
        inCode = true;
        codeLines = [];
      } else {
        const code = codeLines.join("\n").trim();
        if (code && ensureSection().codeBlocks.length < 1) ensureSection().codeBlocks.push(trimCodeBlock(code));
        inCode = false;
        codeLines = [];
      }
      return;
    }

    if (inCode) {
      codeLines.push(line);
      return;
    }

    if (heading) {
      const level = heading[1].length;
      const title = cleanMarkdownText(heading[2]);
      if (level === 1) {
        parsed.title = title || parsed.title;
        return;
      }
      if (level === 2 || level === 3) {
        pushSection();
        current = { title, paragraphs: [], bullets: [], codeBlocks: [] };
        parsed.outline.push(title);
        return;
      }
    }

    if (!trimmed || trimmed === "---" || /^\|?\s*:?-{3,}/.test(trimmed)) {
      flushParagraph();
      return;
    }

    if (/^\s*[-*+]\s+/.test(line) || /^\s*\d+[.)]\s+/.test(line)) {
      flushParagraph();
      const bullet = trimText(line, 180);
      if (isCurrentSiteLessonText(bullet)) ensureSection().bullets.push(bullet);
      return;
    }

    paragraph.push(line);
  });

  pushSection();
  return parsed;
}

function renderParsedLessonNotes(parsed, module) {
  const profile = getLessonProfile(module);
  const extractedSections = parsed.sections
    .filter((section) => section.paragraphs.length || section.bullets.length || section.codeBlocks.length);
  const sections = extractedSections.length ? extractedSections : buildSyntheticLessonSections(module);
  const outline = parsed.outline.length ? parsed.outline : profile.steps;

  if (!sections.length) return renderFallbackLessonNotes(module);

  return `
    <div class="section-kicker">章节讲义 · ${escapeHtml(module.partShort)}</div>
    <h2>先看 3 分钟版，再看完整精读</h2>
    <p class="summary">以下内容直接从原讲义提炼。先用索引找本节要点，再展开完整卡片；如果刚看完动画，优先寻找和控件、读数、颜色变化相对应的句子。</p>
    <div class="lesson-outline">
      ${outline.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>
    ${renderThreeMinuteBrief(sections, module)}
    ${renderKnowledgePointIndex(sections, module)}
    <details class="deep-dive-details">
      <summary>展开完整精读卡片</summary>
      <div class="deep-dive-grid">
        ${renderDeepDiveCards(sections, module)}
      </div>
    </details>
  `;
}

function renderFallbackLessonNotes(module) {
  const profile = getLessonProfile(module);
  const sections = buildSyntheticLessonSections(module);

  return `
    <div class="section-kicker">章节讲义 · ${escapeHtml(module.partShort)}</div>
    <h2>先看 3 分钟版，再看完整精读</h2>
    <p class="summary">这一页没有独立旧讲义 Markdown，因此用模块目标、领域机制、演示变量和源码入口补成可学习的精读路径。</p>
    <div class="lesson-outline">
      ${[...profile.steps, profile.focus].map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>
    ${renderThreeMinuteBrief(sections, module)}
    ${renderKnowledgePointIndex(sections, module)}
    <details class="deep-dive-details">
      <summary>展开完整精读卡片</summary>
      <div class="deep-dive-grid">
        ${renderDeepDiveCards(sections, module)}
      </div>
    </details>
  `;
}

function isMeaningfulSourceText(value) {
  const text = cleanMarkdownText(value).replace(/\{[^}]+}/g, "").trim();
  if (text.length < 18 || text.length > 420) return false;
  if (!/[\u4e00-\u9fff]/.test(text)) return false;
  if (/(<style|<\/|unsafe_allow_html|streamlit run|python main\.py|page_title|layout=|padding|border|background|rgba|linear-gradient|font-size|@keyframes|animation:|display:flex|box-shadow|color:)/i.test(text)) return false;
  return true;
}

function extractSourceTexts(source) {
  const values = [];
  const seen = new Set();
  const add = (value) => {
    const text = trimText(String(value).replace(/<[^>]+>/g, " "), 320);
    if (!isMeaningfulSourceText(text) || seen.has(text)) return;
    seen.add(text);
    values.push(text);
  };

  const tripleRe = /(?:f|r|fr|rf)?("""[\s\S]*?"""|'''[\s\S]*?''')/g;
  for (const match of source.matchAll(tripleRe)) {
    const block = match[1].slice(3, -3);
    block.split(/\n\s*\n/).forEach(add);
  }

  const quotedRe = /(?:f|r|fr|rf)?["']([^"'\n]{12,})["']/g;
  for (const match of source.matchAll(quotedRe)) add(match[1]);

  return values;
}

function extractSourceHeadings(source, module) {
  const headings = [];
  const seen = new Set();
  const headingRe = /st\.(?:title|header|subheader)\(\s*(?:f|r|fr|rf)?["']([^"'\n]+)["']/g;
  for (const match of source.matchAll(headingRe)) {
    const heading = trimText(match[1], 70);
    if (!heading || seen.has(heading)) continue;
    seen.add(heading);
    headings.push(heading);
  }

  if (!headings.length) headings.push(module.title);
  return headings;
}

function extractSourceSnippet(source) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const bannedNames = new Set(["css", "safe_run", "render_back_home"]);
  const defs = lines
    .map((line, index) => {
      const match = line.trim().match(/^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)/);
      return match ? { index, kind: match[1], name: match[2] } : null;
    })
    .filter(Boolean)
    .filter((item) => !bannedNames.has(item.name));
  const preferred = defs.find((item) => /^(compute_|plot_|render_|demo_|build_|make_|handshake|quiz_|explain_)/.test(item.name));
  const target = preferred || defs[0];
  const start = target
    ? target.index
    : lines.findIndex((line) => /^[A-Z][A-Z0-9_]+\s*=/.test(line.trim()));
  if (start < 0) return "";
  const nextDef = lines.findIndex((line, index) => index > start + 6 && /^(def|class)\s+\w+/.test(line.trim()));
  const end = nextDef > start ? Math.min(nextDef, start + 120) : start + 120;
  return trimCodeBlock(lines.slice(start, end).join("\n"));
}

function extractNamedSnippet(source, patterns, fallbackStart = 0) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const defs = lines
    .map((line, index) => {
      const match = line.trim().match(/^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)/);
      return match ? { index, kind: match[1], name: match[2] } : null;
    })
    .filter(Boolean);
  const target = defs.find((item) => patterns.some((pattern) => pattern.test(item.name)))
    || defs[fallbackStart]
    || defs[0];
  if (!target) return "";
  const nextDef = defs.find((item) => item.index > target.index + 5);
  const end = nextDef ? Math.min(nextDef.index, target.index + 44) : target.index + 44;
  return trimCodeBlock(lines.slice(target.index, end).join("\n"));
}

function renderTeachingSourceGuide(source, module) {
  const profile = getLessonProfile(module);
  const snippets = [
    {
      label: "动画对应",
      title: "上面的概念动画从这里取机制",
      body: `对应页面里的“概念动画”。读代码时只找 ${profile.signal} 怎样被计算出来，先不要管 Streamlit 或样式细节。`,
      snippet: extractNamedSnippet(source, [/render.*stage/i, /concept/i, /plot/i, /visual/i, /demo/i], 0),
    },
    {
      label: "控件对应",
      title: "滑块/选择器会改这里的变量",
      body: `对应“动手实验”。重点找 ${profile.variable} 中哪个量被控件改写，再看它怎样传到图形或指标。`,
      snippet: extractNamedSnippet(source, [/compute/i, /lab/i, /update/i, /make/i, /build/i], 1),
    },
    {
      label: "读数对应",
      title: "读数区和错误判断从这里来",
      body: `对应页面里的读数、诊断或解释文字。源码不是让小白背 API，而是用来确认“为什么画面会这样变”。`,
      snippet: extractNamedSnippet(source, [/metric/i, /diagn/i, /explain/i, /score/i, /loss/i, /accuracy/i], 2),
    },
  ].filter((item) => item.snippet);

  if (!snippets.length) {
    return `
      <div class="source-guide">
        <article>
          <span>源码对照</span>
          <strong>这份脚本暂时没有提取到稳定片段</strong>
          <p>${escapeHtml(`先回到动画和实验，确认 ${profile.signal}；完整源码仍可展开给开发者检查。`)}</p>
        </article>
      </div>
    `;
  }

  return `
    <div class="source-guide">
      ${snippets.map((item) => `
        <article>
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.title)}</strong>
          <p>${escapeHtml(item.body)}</p>
          <pre class="source-snippet"><code>${escapeHtml(item.snippet)}</code></pre>
        </article>
      `).join("")}
    </div>
  `;
}

function parseSourceLessonNotes(source, module) {
  const headings = extractSourceHeadings(source, module);
  const texts = extractSourceTexts(source);
  const snippet = extractSourceSnippet(source);
  const sections = headings.map((heading, index) => {
    const offset = index * 2;
    return {
      title: heading,
      paragraphs: texts.slice(offset, offset + 2),
      bullets: texts.slice(offset + 2, offset + 4),
      codeBlocks: index === 0 && snippet ? [snippet] : [],
    };
  }).filter((section) => section.paragraphs.length || section.bullets.length || section.codeBlocks.length);

  return {
    title: module.title,
    outline: headings,
    sections,
  };
}

function renderSourceLessonNotes(parsed, module) {
  const profile = getLessonProfile(module);
  const sections = parsed.sections.length ? parsed.sections : buildSyntheticLessonSections(module);
  const outline = parsed.outline.length ? parsed.outline : profile.steps;
  return `
    <div class="section-kicker">章节讲义 · ${escapeHtml(module.partShort)}</div>
    <h2>先看 3 分钟版，再看完整精读</h2>
    <p class="summary">这一节没有独立旧讲义 Markdown，下面从原 Python 教学脚本中提炼真实讲解，并补上本节应观察的机制、变量和迁移练习。</p>
    <div class="lesson-outline">
      ${outline.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>
    ${renderThreeMinuteBrief(sections, module)}
    ${renderKnowledgePointIndex(sections, module)}
    <details class="deep-dive-details">
      <summary>展开完整精读卡片</summary>
      <div class="deep-dive-grid">
        ${renderDeepDiveCards(sections, module)}
      </div>
    </details>
  `;
}

async function renderSourceNotesInto(target, module) {
  const sourceResponse = await fetch(module.sourcePath);
  if (!sourceResponse.ok) throw new Error(`HTTP ${sourceResponse.status}`);
  const source = await sourceResponse.text();
  const parsed = parseSourceLessonNotes(source, module);
  target.innerHTML = renderSourceLessonNotes(parsed, module);
  target.dataset.fallback = "source";
}

async function loadLessonNotes(module) {
  const target = document.querySelector("[data-lesson-notes]");
  if (!target) return;

  try {
    const candidates = legacyMarkdownCandidates(module);
    for (const path of candidates) {
      try {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const markdown = await response.text();
        const parsed = parseLegacyMarkdown(markdown, module);
        target.innerHTML = renderParsedLessonNotes(parsed, module);
        target.dataset.fallback = "markdown";
        target.dataset.legacyPath = path;
        return;
      } catch (error) {
        // Try the next candidate before falling back to the Python source.
      }
    }
    await renderSourceNotesInto(target, module);
  } catch (error) {
    try {
      await renderSourceNotesInto(target, module);
    } catch (sourceError) {
      target.innerHTML = renderFallbackLessonNotes(module);
      target.dataset.fallback = "generic";
    }
  } finally {
    target.dataset.loaded = "true";
    requestAnimationFrame(() => applyMotionReveal(target, { reset: false }));
  }
}

function demoNumber(scope, name) {
  return Number(scope.querySelector(`[data-demo-control="${name}"]`)?.value || 0);
}

function demoValue(scope, name) {
  return scope.querySelector(`[data-demo-control="${name}"]`)?.value || "";
}

function flowNode(label, detail, tone = "") {
  return `<div class="flow-node ${tone}"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(detail)}</span></div>`;
}

function renderConceptStage(demo, module) {
  const profile = getLessonProfile(module);
  const kind = demo.dataset.demo;
  const intensity = demoNumber(demo, "intensity");
  const complexity = demoNumber(demo, "complexity");
  const focusIndex = Number(demoValue(demo, "focus"));
  const focus = profile.steps[focusIndex] || profile.steps[0];
  const stage = demo.querySelector("[data-demo-stage]");
  const readout = demo.querySelector("[data-demo-readout]");
  const scale = Math.max(1, intensity);
  const width = Math.min(86, 34 + scale * 5);
  const delay = Math.max(90, 620 - complexity * 52);
  const attentionTokens = ["深度", "学习", "需要", "注意力", "机制"];
  const normalizedFocus = Number.isFinite(focusIndex) ? Math.max(0, focusIndex) : 0;
  const readoutFocus = kind === "attention"
    ? attentionTokens[normalizedFocus % attentionTokens.length]
    : focus;
  let readoutText = `当前观察：${readoutFocus}。强度 ${intensity}、复杂度 ${complexity} 会改变动画中的响应幅度和路径密度。`;

  if (kind === "gradient") {
    stage.innerHTML = `
      <svg class="concept-svg" viewBox="0 0 620 300" role="img" aria-label="训练链路和梯度回传">
        <path class="demo-grid-line" d="M70 232 C170 168 244 156 322 184 S474 236 552 86"></path>
        <path class="demo-flow-line" style="--flow-speed:${delay}ms" d="M86 222 C170 168 244 156 322 184 S474 236 548 92"></path>
        <g class="demo-point-train">
          <circle cx="86" cy="222" r="8"></circle>
          <circle cx="216" cy="164" r="7"></circle>
          <circle cx="338" cy="188" r="7"></circle>
          <circle cx="452" cy="178" r="7"></circle>
          <circle cx="548" cy="92" r="8"></circle>
        </g>
        <g class="demo-labels">
          <text x="66" y="262">input</text>
          <text x="192" y="136">forward</text>
          <text x="320" y="222">loss</text>
          <text x="438" y="150">gradient</text>
          <text x="508" y="70">update</text>
        </g>
      </svg>
      <div class="flow-strip">
        ${flowNode("x", "输入形状")}
        ${flowNode("Wx+b", "前向计算")}
        ${flowNode("L", "损失标量", "is-active")}
        ${flowNode("dL/dW", "反向传播")}
      </div>
    `;
  } else if (kind === "convolution") {
    const cells = Array.from({ length: 25 }, (_, index) => {
      const row = Math.floor(index / 5);
      const col = index % 5;
      const active = row >= 1 && row <= 3 && col >= 1 && col <= 3;
      const level = active ? 24 + ((row + col + intensity) % 4) * 14 : 8 + ((row * col + complexity) % 3) * 9;
      return `<span style="--cell-fill:${level}%"></span>`;
    }).join("");
    stage.innerHTML = `
      <div class="scan-demo">
        <div class="scan-grid">${cells}<i></i></div>
        <div class="feature-stack">
          ${["边缘", "纹理", "局部形状", "类别证据"].map((label, index) => `<b style="--bar-width:${Math.min(96, width + index * 5)}%; --bar-delay:${index * 70}ms">${escapeHtml(label)}</b>`).join("")}
        </div>
      </div>
      <div class="flow-strip">
        ${flowNode("kernel", "局部探测")}
        ${flowNode("feature map", "空间响应", "is-active")}
        ${flowNode("pool", "压缩保留")}
      </div>
    `;
  } else if (kind === "sequence") {
    const tokens = ["x1", "x2", "x3", "x4", "x5", "x6"].slice(0, Math.min(6, complexity + 2));
    stage.innerHTML = `
      <div class="sequence-demo">
        <div class="token-lane">
          ${tokens.map((token, index) => `<span style="--token-delay:${index * 90}ms">${token}</span>`).join("")}
        </div>
        <div class="memory-lane">
          ${tokens.map((_, index) => `<i style="--memory-height:${Math.max(18, width - index * 6)}%; --token-delay:${index * 90}ms"></i>`).join("")}
        </div>
      </div>
      <div class="flow-strip">
        ${flowNode("hidden", "历史压缩")}
        ${flowNode("gate", "保留/遗忘", "is-active")}
        ${flowNode("output", "当前预测")}
      </div>
    `;
  } else if (kind === "attention") {
    const tokens = attentionTokens;
    const queryIndex = normalizedFocus % tokens.length;
    const valueLabels = ["深度表征", "学习目标", "需求关系", "注意力线索", "机制解释"];
    const similarity = [
      [1, 0.86, 0.28, 0.52, 0.44],
      [0.86, 1, 0.36, 0.58, 0.48],
      [0.28, 0.42, 1, 0.62, 0.55],
      [0.48, 0.56, 0.68, 1, 0.92],
      [0.42, 0.46, 0.58, 0.9, 1],
    ];
    const sharpness = 0.85 + intensity * 0.18;
    const noise = Math.max(0, complexity - 4) * 0.045;
    const scored = tokens.map((token, index) => {
      const offset = (index % 2 === 0 ? -noise : noise) + (index === queryIndex ? 0.06 : 0);
      const score = similarity[queryIndex][index] * sharpness + offset;
      return { token, value: valueLabels[index], index, score };
    });
    const exps = scored.map((item) => Math.exp(item.score * 1.6));
    const total = exps.reduce((sum, value) => sum + value, 0);
    const weighted = scored
      .map((item, index) => ({ ...item, weight: exps[index] / total }))
      .sort((a, b) => b.weight - a.weight);
    const top = weighted[0];
    const runnerUp = weighted[1];
    const rows = scored.map((item) => {
      const weight = exps[item.index] / total;
      const scoreWidth = Math.max(18, Math.min(96, item.score * 36));
      const weightWidth = Math.max(8, Math.round(weight * 100));
      return `
        <div class="attention-score-row ${item.index === queryIndex ? "is-query" : ""}" style="--score-width:${scoreWidth}%; --weight-width:${weightWidth}%; --bar-delay:${item.index * 70}ms">
          <div>
            <strong>K: ${escapeHtml(item.token)}</strong>
            <small>Q·K = ${item.score.toFixed(2)}</small>
          </div>
          <i aria-hidden="true"></i>
          <b>${weightWidth}%</b>
          <em>V: ${escapeHtml(item.value)}</em>
        </div>
      `;
    }).join("");
    stage.innerHTML = `
      <div class="attention-demo attention-chain" data-attention-mechanism>
        <div class="attention-query-card" data-attention-query>
          <span>Query</span>
          <strong>${escapeHtml(tokens[queryIndex])}</strong>
          <small>当前 token 正在提出“我应该看上下文里的谁？”</small>
        </div>
        <div class="attention-score-table" data-attention-scores>
          <div class="attention-stage-label">
            <span>Q·K 匹配分</span>
            <span>softmax 权重</span>
            <span>Value 贡献</span>
          </div>
          ${rows}
        </div>
        <div class="attention-output-card" data-attention-output>
          <span>Weighted Value</span>
          <strong>输出更偏向「${escapeHtml(top.token)}」</strong>
          <p>softmax 把 Q·K 分数变成权重后，${escapeHtml(top.value)} 贡献最大，${escapeHtml(runnerUp.value)} 次之；最终表示不是复制某个词，而是按权重混合 Value。</p>
        </div>
      </div>
      <div class="flow-strip">
        ${flowNode("Q", `当前提问：${tokens[queryIndex]}`)}
        ${flowNode("Q·K", "逐个 key 打分")}
        ${flowNode("softmax", "分数转权重", "is-active")}
        ${flowNode("Σ wV", "加权汇总 Value")}
      </div>
    `;
    readoutText = `当前 Query 是「${tokens[queryIndex]}」。注意力锐度 ${intensity} 让 softmax ${intensity >= 7 ? "更集中" : "更平缓"}，上下文噪声 ${complexity} 会扰动 Q·K 分数；最高权重落在「${top.token}」(${Math.round(top.weight * 100)}%)，所以输出主要汇总它的 Value。`;
  } else if (kind === "training") {
    const trainPath = `M40 ${210 - intensity * 4} C150 ${190 - complexity * 5}, 240 ${120 - intensity * 3}, 360 ${106 - complexity * 4} S500 ${92 - intensity * 2}, 580 ${72}`;
    const valPath = `M40 220 C150 168, 240 ${136 - intensity * 2}, 360 ${128 + complexity * 2} S500 ${120 + intensity}, 580 ${112 + complexity * 3}`;
    stage.innerHTML = `
      <svg class="concept-svg" viewBox="0 0 620 280" role="img" aria-label="训练曲线诊断">
        <path class="demo-axis" d="M40 235 H590 M40 34 V235"></path>
        <path class="demo-flow-line" style="--flow-speed:${delay}ms" d="${trainPath}"></path>
        <path class="demo-compare-line" d="${valPath}"></path>
        <text x="64" y="52">loss</text>
        <text x="518" y="252">epoch</text>
      </svg>
      <div class="flow-strip">
        ${flowNode("train", "优化推进", "is-active")}
        ${flowNode("valid", "泛化检查")}
        ${flowNode("grad", "健康信号")}
      </div>
    `;
  } else if (kind === "architecture") {
    stage.innerHTML = `
      <div class="architecture-demo">
        ${["config", "data", "model", "trainer", "artifact"].map((label, index) => `<div class="arch-node ${index === focusIndex % 5 ? "is-active" : ""}" style="--node-delay:${index * 80}ms">${label}</div>`).join("")}
      </div>
      <div class="flow-strip">
        ${flowNode("interface", "稳定边界")}
        ${flowNode("registry", "扩展入口", "is-active")}
        ${flowNode("run log", "可复现")}
      </div>
    `;
  } else {
    stage.innerHTML = `
      <div class="systems-demo">
        ${["client", "gateway", "cache", "database", "model"].map((label, index) => `<div class="system-node ${index === focusIndex % 5 ? "is-active" : ""}" style="--node-delay:${index * 80}ms">${label}</div>`).join("")}
      </div>
      <div class="flow-strip">
        ${flowNode("flow", "请求路径")}
        ${flowNode("tradeoff", "约束取舍", "is-active")}
        ${flowNode("failure", "排查顺序")}
      </div>
    `;
  }

  readout.textContent = readoutText;
  pulseDemoReadout(demo);
}

function pulseDemoReadout(demo) {
  const readout = demo.querySelector("[data-demo-readout]");
  if (!readout || prefersReducedMotion()) return;
  readout.classList.remove("is-updating");
  void readout.offsetWidth;
  readout.classList.add("is-updating");
}

function wireConceptDemos(module) {
  document.querySelectorAll("[data-demo]").forEach((demo) => {
    const render = () => renderConceptStage(demo, module);
    demo.querySelectorAll("[data-demo-control]").forEach((control) => {
      control.addEventListener("input", render);
      control.addEventListener("change", render);
    });
    render();
  });
}

function courseCard(module, meta, image) {
  const plan = moduleLearningPlan(module);
  return `
    <a class="course-card" href="${moduleHref(module)}">
      <div class="course-image"><img src="${image}" alt="${escapeHtml(module.title)}"></div>
      <div class="course-body">
        <div class="course-meta">${escapeHtml(meta)}</div>
        <h3>${escapeHtml(module.title)}</h3>
        <p>${escapeHtml(module.summary)}</p>
        <dl class="learning-plan-mini">
          <div><dt>预计</dt><dd>${escapeHtml(plan.duration)}</dd></div>
          <div><dt>完成</dt><dd>${escapeHtml(plan.completion)}</dd></div>
        </dl>
        <div class="tag-row">
          <span class="tag">${escapeHtml(module.level)}</span>
          <span class="tag">${escapeHtml(module.id)}</span>
        </div>
      </div>
    </a>
  `;
}

function moduleCard(module, index = "") {
  const plan = moduleLearningPlan(module);
  return `
    <a class="module-card" href="${moduleHref(module)}">
      <span class="path-tag">${escapeHtml(module.partShort)} / ${escapeHtml(module.level)}</span>
      <h3>${index}${escapeHtml(module.title)}</h3>
      <p>${escapeHtml(module.summary)}</p>
      <dl class="learning-plan-mini">
        <div><dt>预计</dt><dd>${escapeHtml(plan.duration)}</dd></div>
        <div><dt>先修</dt><dd>${escapeHtml(plan.prereq)}</dd></div>
        <div><dt>完成</dt><dd>${escapeHtml(plan.completion)}</dd></div>
        <div><dt>复习</dt><dd>${escapeHtml(plan.review)}</dd></div>
      </dl>
      ${tagHtml(module.tags)}
    </a>
  `;
}

function renderPortfolioSection() {
  const codeWall = [
    ["静态课程站", "原生 HTML/CSS/JS 路由、动画、阅读页与控制台"],
    ["教学脚本库", "Python 课程源码、旧书稿 Markdown 与可复现实验"],
    ["质量门禁", "内容覆盖、UX 浏览器检查、CI 自动验证"],
    ["交互实验", "注意力、CNN 特征图、训练诊断、系统流动演示"],
  ];
  return `
    <section id="portfolio" class="view section portfolio-panel">
      <div class="section-head">
        <div>
          <div class="eyebrow">学习成果档案</div>
          <h2>这套网站怎样陪你学习</h2>
        </div>
        <p>这里记录网站的课程结构、交互实验、源码对照和质量检查。它不是让人匆忙翻亮点，而是帮助你知道自己学到了哪里、还能怎样继续改进。</p>
      </div>
      <div class="portfolio-layout">
        <article class="profile-card">
          <div class="profile-mark">DL</div>
          <div>
            <span class="section-kicker">Course Builder</span>
            <h3>深度学习交互书库作者</h3>
            <p>围绕“学生能不能看懂、能不能动手、能不能复现”来组织课程内容，兼顾课程讲义、交互演示、工程脚本与学习成果整理。</p>
          </div>
        </article>
        <div class="portfolio-detail">
          <div class="tech-stack" aria-label="技术栈">
            ${["HTML", "CSS", "JavaScript", "Python", "GitHub Actions", "Streamlit Legacy"].map((item) => `<span>${item}</span>`).join("")}
          </div>
          <div class="portfolio-links">
            <a class="ghost-action" href="https://github.com/syzzzzzzz" target="_blank" rel="noreferrer">GitHub</a>
            <a class="ghost-action" href="#hardcore-labs">深度实验</a>
            <a class="ghost-action" href="#notes">代码墙</a>
            <a class="ghost-action" href="#home" aria-label="返回学习起点">返回学习起点</a>
          </div>
        </div>
      </div>
      <div class="code-wall" aria-label="代码展示墙">
        ${codeWall.map(([title, detail]) => `
          <article>
            <strong>${escapeHtml(title)}</strong>
            <p>${escapeHtml(detail)}</p>
          </article>
        `).join("")}
      </div>
    </section>
  `;
}

function renderHardcoreLabsSection() {
  const labs = [
    {
      id: "xai",
      title: "模型可解释性实验室",
      detail: "调目标层、热力图阈值和分类置信度，观察证据区域是集中在目标上，还是泄漏到背景纹理。",
      ids: ["part2/08_visualization_gradcam", "part5/01_feature_visualization", "part4/02_multihead_visual"],
      controls: `
        <label class="hardcore-control">目标层
          <select data-hardcore-control="layer">
            <option value="shallow">浅层纹理</option>
            <option value="middle">中层部件</option>
            <option value="deep">深层语义</option>
          </select>
        </label>
        <label class="hardcore-control">热力图阈值 <input type="range" min="0.15" max="0.75" step="0.05" value="0.35" data-hardcore-control="threshold"></label>
        <label class="hardcore-control">分类置信度 <input type="range" min="0.35" max="0.95" step="0.05" value="0.7" data-hardcore-control="confidence"></label>
      `,
      stage: `<div class="xai-board" data-xai-board></div>`,
    },
    {
      id: "adversarial",
      title: "对抗样本演示",
      detail: "调扰动强度、攻击方向和防御策略，看一张几乎没变的输入如何让模型置信度翻转。",
      ids: ["part5/05_dataset_toys", "part5/02_gradient_monitor", "part6/frontier"],
      controls: `
        <label class="hardcore-control">扰动强度 ε <input type="range" min="0" max="0.35" step="0.01" value="0.12" data-hardcore-control="epsilon"></label>
        <label class="hardcore-control">攻击方向
          <select data-hardcore-control="direction">
            <option value="edge">边缘方向</option>
            <option value="texture">纹理方向</option>
            <option value="background">背景方向</option>
          </select>
        </label>
        <label class="hardcore-control">防御策略
          <select data-hardcore-control="defense">
            <option value="none">无防御</option>
            <option value="smooth">平滑输入</option>
            <option value="augment">增强训练</option>
          </select>
        </label>
      `,
      stage: `<div class="adversarial-board" data-adversarial-board></div>`,
    },
    {
      id: "challenge",
      title: "小型深度学习挑战",
      detail: "在有限预算下调学习率、正则和训练轮次，目标不是单点高分，而是稳定、泛化、可复盘。",
      ids: ["part5/tuning_challenge", "part5/03_training_dynamics", "part6/training_demo"],
      controls: `
        <label class="hardcore-control">学习率 <input type="range" min="0.02" max="0.8" step="0.02" value="0.22" data-hardcore-control="lr"></label>
        <label class="hardcore-control">正则强度 <input type="range" min="0" max="1" step="0.05" value="0.35" data-hardcore-control="regularization"></label>
        <label class="hardcore-control">预算上限 <input type="range" min="20" max="100" step="5" value="60" data-hardcore-control="budget"></label>
      `,
      stage: `<div class="challenge-board" data-challenge-board></div>`,
    },
    {
      id: "case",
      title: "端到端案例",
      detail: "从数据质量、模型容量、监控强度和部署形态判断项目是否能从 notebook 走到可交付系统。",
      ids: ["part5/case_studies", "part5/data_training", "part6/03_full_project"],
      controls: `
        <label class="hardcore-control">数据质量 <input type="range" min="30" max="100" step="5" value="70" data-hardcore-control="quality"></label>
        <label class="hardcore-control">模型容量
          <select data-hardcore-control="capacity">
            <option value="small">小模型</option>
            <option value="balanced">均衡模型</option>
            <option value="large">大模型</option>
          </select>
        </label>
        <label class="hardcore-control">监控强度 <input type="range" min="0" max="100" step="5" value="55" data-hardcore-control="monitoring"></label>
        <label class="hardcore-control">部署形态
          <select data-hardcore-control="deploy">
            <option value="batch">离线批处理</option>
            <option value="api">在线 API</option>
            <option value="edge">端侧部署</option>
          </select>
        </label>
      `,
      stage: `<div class="case-board" data-case-board></div>`,
    },
  ];
  return `
    <section id="hardcore-labs" class="view section hardcore-labs">
      <div class="section-head">
        <div>
          <div class="eyebrow">深度实验室</div>
          <h2>把模型现象拆开看</h2>
        </div>
        <p>这里不只放概念介绍，而是把解释性、对抗扰动、训练挑战和端到端案例做成可调、可观察、可复盘的实验区。</p>
      </div>
      <div class="hardcore-workbench-grid">
        ${labs.map((lab) => `
          <article class="hardcore-lab-card" data-hardcore-lab="${escapeHtml(lab.id)}">
            <div class="hardcore-lab-head">
              <span class="section-kicker">${escapeHtml(lab.title)}</span>
              <p>${escapeHtml(lab.detail)}</p>
            </div>
            <div class="hardcore-control-grid">
              ${lab.controls}
            </div>
            <div class="hardcore-stage" data-hardcore-stage>
              ${lab.stage}
              <div class="hardcore-metrics" data-hardcore-metrics></div>
              <p class="hardcore-readout" data-hardcore-readout></p>
            </div>
            <div class="hardcore-links">
              ${lab.ids.map((id) => {
                const module = byId(id);
                return module ? `<a href="${moduleHref(module)}">${escapeHtml(module.title)}</a>` : "";
              }).join("")}
            </div>
          </article>
        `).join("")}
      </div>
    </section>
  `;
}

function hardcoreControlValue(lab, name) {
  return lab.querySelector(`[data-hardcore-control="${name}"]`)?.value;
}

function hardcoreControlNumber(lab, name) {
  return Number(hardcoreControlValue(lab, name));
}

function hardcoreMetric(label, value, detail = "") {
  return `
    <div class="hardcore-meter">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(detail)}</small>
    </div>
  `;
}

function renderHardcoreXaiLab(lab) {
  const layer = hardcoreControlValue(lab, "layer");
  const threshold = hardcoreControlNumber(lab, "threshold");
  const confidence = hardcoreControlNumber(lab, "confidence");
  const centers = {
    shallow: { row: 1.3, col: 1.4, spread: 2.2, label: "浅层更像在看边缘和纹理" },
    middle: { row: 2.7, col: 2.2, spread: 1.55, label: "中层开始聚焦局部部件" },
    deep: { row: 3.6, col: 3.8, spread: 1.12, label: "深层更接近语义证据" },
  };
  const profile = centers[layer] || centers.middle;
  const values = Array.from({ length: 36 }, (_, index) => {
    const row = Math.floor(index / 6);
    const col = index % 6;
    const dist = Math.hypot(row - profile.row, col - profile.col);
    const wave = Math.sin((row + 1) * 1.7 + col * 0.8) * 0.1;
    return clampNumber(confidence * Math.exp(-dist / profile.spread) + wave + 0.08, 0, 1);
  });
  const active = values.filter((value) => value >= threshold).length;
  const coverage = Math.round(active / values.length * 100);
  const focusScore = clampNumber(Math.round(confidence * 78 + threshold * 32 - Math.abs(coverage - 34) * 0.55), 0, 100);
  const leakage = clampNumber(Math.round(Math.max(0, coverage - 42) * 1.8 + (1 - threshold) * 18), 0, 100);
  lab.querySelector("[data-xai-board]").innerHTML = values.map((value, index) => `
    <span class="xai-cell ${value >= threshold ? "is-hot" : ""}" style="--heat:${Math.round(value * 100)}%; --cell-delay:${index * 8}ms"></span>
  `).join("");
  lab.querySelector("[data-hardcore-metrics]").innerHTML = [
    hardcoreMetric("证据覆盖", `${coverage}%`, "高于阈值的热力格比例"),
    hardcoreMetric("聚焦评分", `${focusScore}`, "越高越像可解释证据"),
    hardcoreMetric("背景泄漏", `${leakage}%`, leakage > 42 ? "需要检查数据偏差" : "仍在可解释范围"),
  ].join("");
  lab.querySelector("[data-hardcore-readout]").textContent = `${profile.label}。学生要观察热力是否压在目标区域，而不是只看模型置信度 ${Math.round(confidence * 100)}%。`;
}

function renderSampleGrid(kind, epsilon = 0) {
  return Array.from({ length: 36 }, (_, index) => {
    const row = Math.floor(index / 6);
    const col = index % 6;
    const base = (row === col || row + col === 5 || (kind === "perturbed" && (row + col) % 4 === 0)) ? 62 : 16;
    const noise = kind === "perturbed" ? Math.round(Math.sin(index * 2.4) * epsilon * 120) : 0;
    return `<span class="sample-cell" style="--cell-tone:${clampNumber(base + noise, 4, 92)}%; --cell-delay:${index * 7}ms"></span>`;
  }).join("");
}

function renderHardcoreAdversarialLab(lab) {
  const epsilon = hardcoreControlNumber(lab, "epsilon");
  const direction = hardcoreControlValue(lab, "direction");
  const defense = hardcoreControlValue(lab, "defense");
  const directionBias = { edge: 0.04, texture: 0.08, background: 0.13 }[direction] || 0.04;
  const defenseFactor = { none: 0, smooth: 0.34, augment: 0.52 }[defense] || 0;
  const original = 0.88;
  const targetConfidence = clampNumber(original - epsilon * (1.9 - defenseFactor) - directionBias, 0.08, 0.92);
  const wrongConfidence = clampNumber(0.1 + epsilon * (2.15 - defenseFactor) + directionBias, 0.04, 0.9);
  const margin = Math.round((targetConfidence - wrongConfidence) * 100);
  const flipped = wrongConfidence > targetConfidence;
  lab.querySelector("[data-adversarial-board]").innerHTML = `
    <div class="adversarial-compare">
      <div><strong>干净输入</strong><div class="sample-grid">${renderSampleGrid("clean", 0)}</div></div>
      <div><strong>扰动后</strong><div class="sample-grid">${renderSampleGrid("perturbed", epsilon)}</div></div>
    </div>
    <div class="confidence-bars">
      <div class="confidence-row"><span>原类别</span><i style="width:${Math.round(targetConfidence * 100)}%"></i><strong>${Math.round(targetConfidence * 100)}%</strong></div>
      <div class="confidence-row is-risk"><span>错误类</span><i style="width:${Math.round(wrongConfidence * 100)}%"></i><strong>${Math.round(wrongConfidence * 100)}%</strong></div>
    </div>
  `;
  lab.querySelector("[data-hardcore-metrics]").innerHTML = [
    hardcoreMetric("分类间隔", `${margin}`, margin < 0 ? "已经被翻转" : "仍保持原分类"),
    hardcoreMetric("扰动 ε", epsilon.toFixed(2), "肉眼变化很小也可能有效"),
    hardcoreMetric("防御削弱", `${Math.round(defenseFactor * 100)}%`, defense === "none" ? "当前无防御" : "扰动影响被压低"),
  ].join("");
  lab.querySelector("[data-hardcore-readout]").textContent = flipped
    ? "模型已经被对抗扰动带偏。学生应观察：图像几乎没变，但置信度边界被推过了。"
    : "当前还未翻转。继续增大 ε 或改攻击方向，观察错误类置信度如何逼近原类别。";
}

function renderHardcoreChallengeLab(lab) {
  const lr = hardcoreControlNumber(lab, "lr");
  const regularization = hardcoreControlNumber(lab, "regularization");
  const budget = hardcoreControlNumber(lab, "budget");
  const stability = clampNumber(Math.round(100 - Math.abs(lr - 0.22) * 115 - Math.max(0, lr - 0.46) * 95), 0, 100);
  const generalization = clampNumber(Math.round(58 + regularization * 34 - Math.abs(regularization - 0.42) * 18 - Math.abs(lr - 0.2) * 26), 0, 100);
  const efficiency = clampNumber(Math.round(budget - lr * 18 + (regularization > 0.75 ? -8 : 8)), 0, 100);
  const score = clampNumber(Math.round(generalization * 0.45 + stability * 0.35 + efficiency * 0.2), 0, 100);
  const checks = [
    ["学习率不过冲", stability >= 65],
    ["验证间隙受控", generalization >= 70],
    ["预算没有爆掉", efficiency >= 58],
    ["实验可复盘", regularization >= 0.2 && budget >= 45],
  ];
  lab.querySelector("[data-challenge-board]").innerHTML = `
    <div class="challenge-gauge" style="--score:${score}%"><strong>${score}</strong><span>综合分</span></div>
    <div class="challenge-checklist">
      ${checks.map(([label, ok]) => `<span class="${ok ? "is-pass" : "is-warn"}">${ok ? "通过" : "待调"} · ${escapeHtml(label)}</span>`).join("")}
    </div>
  `;
  lab.querySelector("[data-hardcore-metrics]").innerHTML = [
    hardcoreMetric("稳定性", `${stability}`, "曲线是否震荡"),
    hardcoreMetric("泛化", `${generalization}`, "验证集是否掉队"),
    hardcoreMetric("效率", `${efficiency}`, "预算内能否收敛"),
  ].join("");
  lab.querySelector("[data-hardcore-readout]").textContent = `挑战目标：把综合分推到 80 以上，同时能解释每次调参为什么影响稳定性、泛化和预算。`;
}

function renderHardcoreCaseLab(lab) {
  const quality = hardcoreControlNumber(lab, "quality");
  const capacity = hardcoreControlValue(lab, "capacity");
  const monitoring = hardcoreControlNumber(lab, "monitoring");
  const deploy = hardcoreControlValue(lab, "deploy");
  const capacityScore = { small: 62, balanced: 84, large: 76 }[capacity] || 72;
  const deployScore = { batch: 82, api: 74, edge: 66 }[deploy] || 74;
  const readiness = clampNumber(Math.round(quality * 0.35 + capacityScore * 0.25 + monitoring * 0.3 + deployScore * 0.1), 0, 100);
  const risk = clampNumber(Math.round((100 - quality) * 0.36 + (100 - monitoring) * 0.36 + (capacity === "large" ? 12 : 0) + (deploy === "edge" ? 10 : 0)), 0, 100);
  const pipeline = [
    ["Data", quality >= 65, "样本/标签/切分"],
    ["Train", capacity !== "large" || quality >= 75, "容量与正则"],
    ["Evaluate", readiness >= 68, "指标与误差分析"],
    ["Deploy", monitoring >= 55, "服务与监控"],
  ];
  lab.querySelector("[data-case-board]").innerHTML = `
    <div class="case-pipeline">
      ${pipeline.map(([label, ok, detail], index) => `
        <div class="case-node ${ok ? "is-pass" : "is-warn"}" style="--node-delay:${index * 70}ms">
          <strong>${label}</strong><small>${detail}</small>
        </div>
      `).join("")}
    </div>
    <div class="artifact-checklist">
      ${["dataset_card.md", "train_config.yaml", "metrics.json", "model_card.md", "monitoring.md"].map((item, index) => `<span class="${index < Math.ceil(readiness / 22) ? "is-pass" : ""}">${item}</span>`).join("")}
    </div>
  `;
  lab.querySelector("[data-hardcore-metrics]").innerHTML = [
    hardcoreMetric("交付准备度", `${readiness}`, readiness >= 76 ? "可以整理成学习成果" : "还需要补证据"),
    hardcoreMetric("项目风险", `${risk}%`, risk > 42 ? "优先补数据和监控" : "风险可控"),
    hardcoreMetric("部署形态", deploy.toUpperCase(), capacity === "large" && deploy === "edge" ? "容量和端侧约束冲突" : "约束基本匹配"),
  ].join("");
  lab.querySelector("[data-hardcore-readout]").textContent = `端到端案例不是只展示最终准确率，而是展示数据证据、训练配置、评估记录、模型卡和上线监控。`;
}

function updateHardcoreLab(lab) {
  if (lab.dataset.hardcoreLab === "xai") renderHardcoreXaiLab(lab);
  else if (lab.dataset.hardcoreLab === "adversarial") renderHardcoreAdversarialLab(lab);
  else if (lab.dataset.hardcoreLab === "challenge") renderHardcoreChallengeLab(lab);
  else if (lab.dataset.hardcoreLab === "case") renderHardcoreCaseLab(lab);
}

function wireHardcoreLabs() {
  document.querySelectorAll("[data-hardcore-lab]").forEach((lab) => {
    const render = () => updateHardcoreLab(lab);
    lab.querySelectorAll("[data-hardcore-control]").forEach((control) => {
      control.addEventListener("input", render);
      control.addEventListener("change", render);
    });
    render();
  });
}

function renderHome() {
  const featured = [
    byId("part1/math_primer"),
    byId("part2/02_feature_maps"),
    byId("part4/01_attention_mechanism"),
  ].filter(Boolean);
  const starter = featured[0];
  const progress = progressSummary();
  const starterSteps = [
    ["01", "数学基础速查", "为什么先学它", "线性代数看形状，微积分看变化，概率看不确定性。没有这三件事，后面的图会像魔法。", starter ? moduleHref(starter) : "#courses", "进入第 1 课", byId("part1/math_primer")],
    ["02", "张量与梯度", "学完算什么", "能说出输入 shape、loss、gradient 和参数更新之间的关系。", "#course/part1%2F01_tensors_gradients", "看梯度动画", byId("part1/01_tensors_gradients")],
    ["03", "神经网络基础", "先修要求", "只要会四则运算和函数图像即可；不要求先会 PyTorch。", "#course/part1%2Fneural_network_basics", "看网络如何学习", byId("part1/neural_network_basics")],
    ["04", "卷积直觉", "第一个模型族", "用图片里的局部窗口理解特征提取，先看动画，再看公式。", "#course/part2%2F01_convolution_visual", "进入视觉模块", byId("part2/01_convolution_visual")],
    ["05", "注意力机制", "现代模型核心", "理解 Query、Key、Value 如何决定一个词该参考谁。", "#course/part4%2F01_attention_mechanism", "进入注意力", byId("part4/01_attention_mechanism")],
  ];
  app.innerHTML = `
    <section id="home" class="view hero">
      <div>
        <div class="kicker">零基础深度学习路径</div>
        <h1>先知道下一步，<br>再开始学<em>模型</em></h1>
        <p>这不是宣传首页，而是一条学习路线：先补数学直觉，再看神经网络、卷积、序列、注意力，最后进实验台验证。</p>
        <div class="hero-actions">
          <a class="action" href="#starter">从第 1 步开始</a>
          <a class="ghost-action" href="#path">先看完整路径</a>
        </div>
        <div class="hero-stats">
          <div><strong>01</strong><span>数学直觉</span></div>
          <div><strong>02</strong><span>张量与梯度</span></div>
          <div><strong>03</strong><span>神经网络</span></div>
          <div><strong>04</strong><span>卷积视觉</span></div>
          <div><strong>05</strong><span>注意力</span></div>
        </div>
      </div>
      <div class="hero-media">
        <img src="https://images.unsplash.com/photo-1518770660439-4636190af475?w=1600&auto=format&fit=crop" alt="深度学习计算电路与知识结构">
      </div>
    </section>

    <section id="starter" class="view section starter-section">
      <div class="section-head">
        <div>
          <div class="eyebrow">新手引导</div>
          <h2>第一次打开，按这 5 步走</h2>
        </div>
        <p>新手不需要先看完整目录。先完成这条短路径，能解释“数据怎样变成预测、错误怎样变成参数更新”，再进入更复杂模型。</p>
      </div>
      <div class="starter-grid">
        ${starterSteps.map(([step, title, label, body, href, action, module]) => {
          const plan = module ? moduleLearningPlan(module) : { duration: "20-30 分钟", prereq: "无", completion: "能说出输入、变化和输出。", review: "隔天复现一次动画。" };
          return `
          <a class="starter-card" href="${href}">
            <span>${step}</span>
            <strong>${title}</strong>
            <em>${label}</em>
            <p>${body}</p>
            <dl>
              <div><dt>时长</dt><dd>${escapeHtml(plan.duration)}</dd></div>
              <div><dt>先修</dt><dd>${escapeHtml(plan.prereq)}</dd></div>
              <div><dt>完成</dt><dd>${escapeHtml(plan.completion)}</dd></div>
            </dl>
            <small>${action}</small>
          </a>
        `;
        }).join("")}
      </div>
      <div class="onboarding-note">
        <strong>完成标准：</strong>每一课结束时，能用一句话说出“输入是什么、变化了什么、输出怎么看、错了怎么调”。说不出来，就先别急着看下一章。
      </div>
    </section>

    <section id="path" class="view section">
      <div class="section-head">
        <div>
          <div class="eyebrow">学习路径</div>
          <h2>按目标选择下一步</h2>
        </div>
        <p>这里是学生主路径。名片、LLM Cookbook 和硬核实验都放到后面，先把课程顺序跑通。</p>
      </div>
      <div class="path-panel">
        <div class="filters">
          ${["整体理解", "计算机视觉", "自然语言", "LLM应用", "工程落地", "面试准备"].map((goal) => `<button class="filter-chip ${goal === activeGoal ? "is-active" : ""}" data-goal="${goal}" type="button">${goal}</button>`).join("")}
        </div>
        <div class="module-list" id="recommendations"></div>
      </div>
    </section>

    <section id="courses" class="view section">
      <div class="section-head">
        <div>
          <div class="eyebrow">核心课程</div>
          <h2>先学这三门核心课</h2>
        </div>
        <p>先抓住最稳定的三条线：数学基础、视觉表征、注意力机制。每个入口都直接进入真实课程详情。</p>
      </div>
      <div class="course-grid">
        ${featured.map((module, index) => courseCard(module, ["FOUNDATIONS", "VISION", "ATTENTION"][index], images[index])).join("")}
      </div>
    </section>

    <section id="notes" class="view section">
      <div class="section-head">
        <div>
          <div class="eyebrow">完整目录</div>
          <h2>完整课程目录</h2>
        </div>
        <p>这是全量索引，适合已经知道要找什么的学生。零基础建议先按上面的 5 步路径走。</p>
      </div>
      <div class="search-band">
        <input id="catalog-search" type="search" placeholder="搜索课程、标签、路径，例如 Transformer / 梯度 / 部署" autocomplete="off">
        <button class="ghost-action" type="button" data-clear-search>清空</button>
      </div>
      <div class="catalog-tabs">
        <button class="filter-chip ${activePart === "all" ? "is-active" : ""}" data-part="all" type="button">全部</button>
        ${PARTS.map((part) => `<button class="filter-chip ${activePart === part.key ? "is-active" : ""}" data-part="${part.key}" type="button">${part.roman} ${part.short}</button>`).join("")}
      </div>
      <div id="catalog-grid" class="catalog-grid"></div>
    </section>

    <section id="progress" class="view section">
      <div class="section-head">
        <div>
          <div class="eyebrow">复习与进度</div>
          <h2>今天该继续哪里</h2>
        </div>
        <p>课程页的“我已理解”和“加入复习”会保存在本机浏览器里。它不是考试分数，而是提醒你下一次从哪里继续、哪里需要回看。</p>
      </div>
      <div class="progress-dashboard">
        <article class="stat-card progress-focus-card">
          <strong>${progress.percent}%</strong>
          <p>已标记理解 ${progress.understood} / ${MODULES.length} 节。先求“能讲清楚”，不要追求点完所有页面。</p>
        </article>
        <article class="stat-card progress-focus-card">
          <strong>${progress.review}</strong>
          <p>节加入稍后复习。复习时优先回到动画和实验，不要只重读长文字。</p>
        </article>
        <a class="stat-card progress-focus-card" href="${progress.next ? moduleHref(progress.next) : "#starter"}">
          <strong>下一步</strong>
          <p>${progress.next ? `${escapeHtml(progress.next.title)} · ${escapeHtml(progress.next.summary)}` : "回到新手路径重新选择第一课。"}</p>
        </a>
      </div>
      <div class="stats-grid progress-part-grid">
        ${PARTS.map((part) => {
          const count = MODULES.filter((module) => module.partKey === part.key).length;
          return `<a class="stat-card" href="#notes" data-jump-part="${part.key}"><strong>${part.roman} ${part.short}</strong><p>${count} 个页面 · ${escapeHtml(part.description)}</p></a>`;
        }).join("")}
      </div>
    </section>

    ${renderHardcoreLabsSection()}

    ${renderLLMCookbookBridge()}

    ${renderPortfolioSection()}
  `;
  wireHome();
  kickRouteMotion();
  requestAnimationFrame(() => applyMotionReveal());
  app.focus({ preventScroll: true });
}

function wireHome() {
  const catalogSearch = document.querySelector("#catalog-search");
  const renderCatalog = () => {
    const query = (catalogSearch?.value || "").trim().toLowerCase();
    const list = MODULES.filter((module) => {
      if (activePart !== "all" && module.partKey !== activePart) return false;
      if (!query) return true;
      return [module.title, module.summary, module.level, module.id, module.sourcePath, ...module.tags]
        .join(" ")
        .toLowerCase()
        .includes(query);
    });
    document.querySelector("#catalog-grid").innerHTML = list.length
      ? list.map((module) => moduleCard(module)).join("")
      : `<div class="empty-state">没有找到匹配课程。可以换一个关键词，或从左侧目录进入完整列表。</div>`;
    requestAnimationFrame(() => applyMotionReveal(document.querySelector("#catalog-grid"), { reset: false }));
  };
  const renderRecommendations = () => {
    const map = {
      整体理解: ["part1", "part2", "part4", "part6"],
      计算机视觉: ["part2", "part5"],
      自然语言: ["part3", "part4", "part6"],
      LLM应用: ["part4", "part5", "part6", "part7"],
      工程落地: ["part5", "part6"],
      面试准备: ["part7", "part5"],
    };
    const tags = {
      整体理解: ["基础", "CNN", "Transformer", "路径"],
      计算机视觉: ["视觉", "CNN", "可视化"],
      自然语言: ["序列", "NLP", "Transformer", "注意力"],
      LLM应用: ["LLM", "Transformer", "部署", "训练", "系统设计", "深度学习", "Agents"],
      工程落地: ["工程", "训练", "部署", "框架"],
      面试准备: ["面试", "系统设计", "数据库", "深度学习"],
    };
    const list = MODULES.filter((module) => map[activeGoal].includes(module.partKey))
      .map((module) => {
        const score = module.tags.filter((tag) => tags[activeGoal].includes(tag)).length + (module.level === "核心" ? 1 : 0);
        return { module, score };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score || a.module.index - b.module.index)
      .slice(0, 6)
      .map((item, index) => moduleCard(item.module, `${index + 1}. `))
      .join("");
    document.querySelector("#recommendations").innerHTML = list;
    requestAnimationFrame(() => applyMotionReveal(document.querySelector("#recommendations"), { reset: false }));
  };
  document.querySelectorAll("[data-part]").forEach((button) => {
    button.addEventListener("click", () => {
      activePart = button.dataset.part;
      renderHome();
      scrollToHashTarget("#notes", 80);
    });
  });
  document.querySelectorAll("[data-goal]").forEach((button) => {
    button.addEventListener("click", () => {
      activeGoal = button.dataset.goal;
      renderHome();
      scrollToHashTarget("#path", 80);
    });
  });
  document.querySelectorAll("[data-jump-part]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      activePart = link.dataset.jumpPart;
      renderHome();
      scrollToHashTarget("#notes", 80);
    });
  });
  document.querySelector("[data-clear-search]")?.addEventListener("click", () => {
    catalogSearch.value = "";
    renderCatalog();
  });
  catalogSearch?.addEventListener("input", renderCatalog);
  wireHardcoreLabs();
  renderRecommendations();
  renderCatalog();
}

function renderInteractiveLab(module) {
  const domain = lessonDomain(module);
  if (domain === "foundation") {
    return `
      <section class="reading-section interactive-lab" data-lab="math-gradient">
        <div class="lab-title-row">
          <div>
            <div class="eyebrow">原生实验</div>
            <h2>梯度下降轨迹</h2>
          </div>
          <span class="lab-badge">HTML / JS</span>
        </div>
        <p class="summary">调起点、学习率和步数，观察点如何沿着损失曲面走向最低处。</p>
        <div class="lab-grid">
          <div class="lab-controls">
            <label class="lab-control">起点 x <input type="range" min="-4" max="4" step="0.5" value="3" data-control="x"></label>
            <label class="lab-control">起点 y <input type="range" min="-4" max="4" step="0.5" value="-3" data-control="y"></label>
            <label class="lab-control">学习率 <input type="range" min="0.05" max="0.55" step="0.05" value="0.2" data-control="lr"></label>
            <label class="lab-control">迭代步数 <input type="range" min="2" max="18" step="1" value="9" data-control="steps"></label>
          </div>
          <div class="lab-stage">
            <svg class="lab-svg" viewBox="0 0 420 280" role="img" aria-label="梯度下降轨迹" data-gradient-plot></svg>
          </div>
        </div>
        <p class="lab-readout" data-lab-readout></p>
      </section>
    `;
  }

  if (domain === "cnn") {
    return `
      <section class="reading-section interactive-lab" data-lab="cnn-feature">
        <div class="lab-title-row">
          <div>
            <div class="eyebrow">原生实验</div>
            <h2>卷积特征图</h2>
          </div>
          <span class="lab-badge">HTML / JS</span>
        </div>
        <p class="summary">换输入图案和卷积核，观察局部模式如何被压缩成特征响应。</p>
        <div class="lab-grid">
          <div class="lab-controls">
            <label class="lab-control">输入图案
              <select data-control="pattern">
                <option value="edge">垂直边缘</option>
                <option value="spot">中心亮斑</option>
                <option value="checker">棋盘纹理</option>
              </select>
            </label>
            <label class="lab-control">卷积核
              <select data-control="kernel">
                <option value="edge">边缘检测</option>
                <option value="blur">平滑平均</option>
                <option value="sharpen">中心强化</option>
              </select>
            </label>
            <label class="lab-control">输入强度 <input type="range" min="0.4" max="1.6" step="0.1" value="1" data-control="strength"></label>
          </div>
          <div class="lab-stage lab-matrix-stage">
            <div>
              <strong>输入</strong>
              <div class="matrix-grid" data-input-grid></div>
            </div>
            <div>
              <strong>卷积核</strong>
              <div class="matrix-grid small" data-kernel-grid></div>
            </div>
            <div>
              <strong>输出特征</strong>
              <div class="matrix-grid" data-output-grid></div>
            </div>
          </div>
        </div>
        <p class="lab-readout" data-lab-readout></p>
      </section>
    `;
  }

  if (domain === "sequence") {
    return `
      <section class="reading-section interactive-lab" data-lab="sequence-memory">
        <div class="lab-title-row">
          <div>
            <div class="eyebrow">原生实验</div>
            <h2>序列记忆轨迹</h2>
          </div>
          <span class="lab-badge">HTML / JS</span>
        </div>
        <p class="summary">调序列长度、记忆保留率和输入噪声，观察隐藏状态怎样积累、遗忘或被干扰。</p>
        <div class="lab-grid">
          <div class="lab-controls">
            <label class="lab-control">序列长度 <input type="range" min="4" max="9" step="1" value="6" data-control="length"></label>
            <label class="lab-control">记忆保留率 <input type="range" min="0.35" max="0.95" step="0.05" value="0.7" data-control="retention"></label>
            <label class="lab-control">输入噪声 <input type="range" min="0" max="0.8" step="0.1" value="0.2" data-control="noise"></label>
          </div>
          <div class="lab-stage sequence-lab-stage">
            <div class="token-lane" data-sequence-tokens></div>
            <div class="memory-lane" data-memory-bars></div>
            <div class="state-card-grid" data-sequence-states></div>
          </div>
        </div>
        <p class="lab-readout" data-lab-readout></p>
      </section>
    `;
  }

  if (domain === "transformer") {
    return `
      <section class="reading-section interactive-lab" data-lab="attention">
        <div class="lab-title-row">
          <div>
            <div class="eyebrow">原生实验</div>
            <h2>注意力权重</h2>
          </div>
          <span class="lab-badge">HTML / JS</span>
        </div>
        <p class="summary">选择 query token 并调整锐度，观察注意力从平均浏览变成集中检索。</p>
        <div class="lab-grid">
          <div class="lab-controls">
            <label class="lab-control">Query token
              <select data-control="query">
                <option value="0">深度</option>
                <option value="1">学习</option>
                <option value="2">需要</option>
                <option value="3">注意力</option>
                <option value="4">机制</option>
              </select>
            </label>
            <label class="lab-control">注意力锐度 <input type="range" min="0.4" max="2.8" step="0.2" value="1.4" data-control="sharpness"></label>
            <label class="lab-control">上下文噪声 <input type="range" min="0" max="0.6" step="0.1" value="0.2" data-control="noise"></label>
          </div>
          <div class="lab-stage">
            <div class="token-row" data-token-row></div>
            <div class="attention-bars" data-attention-bars></div>
          </div>
        </div>
        <p class="lab-readout" data-lab-readout></p>
      </section>
    `;
  }

  if (domain === "training") {
    return `
      <section class="reading-section interactive-lab" data-lab="training-diagnostics">
        <div class="lab-title-row">
          <div>
            <div class="eyebrow">原生实验</div>
            <h2>训练诊断曲线</h2>
          </div>
          <span class="lab-badge">HTML / JS</span>
        </div>
        <p class="summary">调学习率、正则强度和数据噪声，观察训练曲线、验证曲线和梯度健康度怎样联动。</p>
        <div class="lab-grid">
          <div class="lab-controls">
            <label class="lab-control">学习率 <input type="range" min="0.02" max="0.8" step="0.02" value="0.22" data-control="lr"></label>
            <label class="lab-control">正则强度 <input type="range" min="0" max="1" step="0.05" value="0.35" data-control="regularization"></label>
            <label class="lab-control">数据噪声 <input type="range" min="0" max="1" step="0.05" value="0.25" data-control="noise"></label>
          </div>
          <div class="lab-stage training-lab-stage">
            <svg class="lab-svg" viewBox="0 0 460 280" role="img" aria-label="训练和验证曲线" data-training-plot></svg>
            <div class="diagnostic-metrics" data-training-metrics></div>
          </div>
        </div>
        <p class="lab-readout" data-lab-readout></p>
      </section>
    `;
  }

  if (domain === "architecture") {
    return `
      <section class="reading-section interactive-lab" data-lab="architecture-flow">
        <div class="lab-title-row">
          <div>
            <div class="eyebrow">原生实验</div>
            <h2>工程边界模拟</h2>
          </div>
          <span class="lab-badge">HTML / JS</span>
        </div>
        <p class="summary">调模块耦合度、插件化程度和记录完整度，观察一个项目从脚本堆变成可复用系统。</p>
        <div class="lab-grid">
          <div class="lab-controls">
            <label class="lab-control">模块耦合度 <input type="range" min="0" max="100" step="5" value="45" data-control="coupling"></label>
            <label class="lab-control">插件化程度 <input type="range" min="0" max="100" step="5" value="65" data-control="plugins"></label>
            <label class="lab-control">记录完整度 <input type="range" min="0" max="100" step="5" value="55" data-control="logging"></label>
          </div>
          <div class="lab-stage architecture-lab-stage">
            <div class="architecture-demo" data-architecture-flow></div>
            <div class="state-card-grid" data-architecture-metrics></div>
          </div>
        </div>
        <p class="lab-readout" data-lab-readout></p>
      </section>
    `;
  }

  return `
    <section class="reading-section interactive-lab" data-lab="systems-flow">
      <div class="lab-title-row">
        <div>
          <div class="eyebrow">原生实验</div>
          <h2>系统链路排查</h2>
        </div>
        <span class="lab-badge">HTML / JS</span>
      </div>
      <p class="summary">调请求负载、缓存命中率和故障位置，把面试知识点放回一次真实推理请求链路里。</p>
      <div class="lab-grid">
        <div class="lab-controls">
          <label class="lab-control">请求负载 <input type="range" min="1" max="10" step="1" value="5" data-control="load"></label>
          <label class="lab-control">缓存命中率 <input type="range" min="0" max="0.95" step="0.05" value="0.45" data-control="cache"></label>
          <label class="lab-control">故障位置
            <select data-control="failure">
              <option value="none">无故障</option>
              <option value="network">网络</option>
              <option value="cache">缓存</option>
              <option value="database">数据库</option>
              <option value="model">模型服务</option>
            </select>
          </label>
        </div>
        <div class="lab-stage systems-lab-stage">
          <div class="systems-demo" data-systems-flow></div>
          <div class="diagnostic-metrics" data-systems-metrics></div>
        </div>
      </div>
      <p class="lab-readout" data-lab-readout></p>
    </section>
  `;
}

function controlValue(scope, name) {
  return scope.querySelector(`[data-control="${name}"]`)?.value;
}

function controlNumber(scope, name) {
  return Number(controlValue(scope, name));
}

function matrixCell(value, maxAbs = 1, index = 0) {
  const normalized = Math.max(-1, Math.min(1, value / maxAbs));
  const intensity = Math.round(Math.abs(normalized) * 100);
  const background = normalized >= 0
    ? `color-mix(in srgb, var(--accent) ${18 + intensity * 0.55}%, #fffdf8)`
    : `color-mix(in srgb, #2a2118 ${12 + intensity * 0.45}%, #fffdf8)`;
  return `<span class="matrix-cell" style="background: ${background}; --cell-delay: ${Math.min(index * 12, 180)}ms">${Number(value).toFixed(1)}</span>`;
}

function renderMathGradientLab(lab) {
  const startX = controlNumber(lab, "x");
  const startY = controlNumber(lab, "y");
  const lr = controlNumber(lab, "lr");
  const steps = controlNumber(lab, "steps");
  const points = [{ x: startX, y: startY }];
  let x = startX;
  let y = startY;

  for (let i = 0; i < steps; i += 1) {
    const dx = 1.08 * x + 0.16 * y;
    const dy = 0.52 * y + 0.16 * x;
    x -= lr * dx;
    y -= lr * dy;
    points.push({ x, y });
  }

  const toSvg = (point) => ({
    x: 210 + point.x * 42,
    y: 140 - point.y * 28,
  });
  const path = points.map(toSvg).map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const finalLoss = 0.54 * x * x + 0.26 * y * y + 0.16 * x * y;
  const plot = lab.querySelector("[data-gradient-plot]");

  plot.innerHTML = `
    <defs>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#2a2118"></path>
      </marker>
    </defs>
    ${[1, 2, 3, 4].map((scale) => `<ellipse cx="210" cy="140" rx="${scale * 43}" ry="${scale * 27}" fill="none" stroke="#e6ded2" stroke-width="1"></ellipse>`).join("")}
    <line x1="24" y1="140" x2="396" y2="140" stroke="#e6ded2"></line>
    <line x1="210" y1="24" x2="210" y2="256" stroke="#e6ded2"></line>
    <path class="motion-trace" d="${path}" pathLength="1" fill="none" stroke="#2a2118" stroke-width="2.5" marker-end="url(#arrow)"></path>
    ${points.map((point, index) => {
      const svgPoint = toSvg(point);
      return `<circle class="motion-point" style="--point-delay: ${Math.min(index * 34, 360)}ms" cx="${svgPoint.x.toFixed(1)}" cy="${svgPoint.y.toFixed(1)}" r="${index === 0 ? 6 : 4}" fill="${index === points.length - 1 ? "#b08a4f" : "#2a2118"}"></circle>`;
    }).join("")}
  `;
  lab.querySelector("[data-lab-readout]").textContent = `最终位置 (${x.toFixed(2)}, ${y.toFixed(2)})，损失约 ${finalLoss.toFixed(3)}。学习率过大时会跨过谷底，过小时会走得很慢。`;
  pulseLabReadout(lab);
}

function patternMatrix(pattern, strength) {
  return Array.from({ length: 5 }, (_, row) => Array.from({ length: 5 }, (_, col) => {
    if (pattern === "edge") return (col >= 3 ? 1 : 0.12) * strength;
    if (pattern === "spot") {
      const dist = Math.abs(row - 2) + Math.abs(col - 2);
      return Math.max(0.08, 1 - dist * 0.28) * strength;
    }
    return ((row + col) % 2 === 0 ? 1 : 0.2) * strength;
  }));
}

function kernelMatrix(type) {
  if (type === "blur") return [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]];
  if (type === "sharpen") return [[0, -1, 0], [-1, 5, -1], [0, -1, 0]];
  return [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
}

function convolve2d(input, kernel) {
  return Array.from({ length: 3 }, (_, row) => Array.from({ length: 3 }, (_, col) => {
    let sum = 0;
    for (let kr = 0; kr < 3; kr += 1) {
      for (let kc = 0; kc < 3; kc += 1) {
        sum += input[row + kr][col + kc] * kernel[kr][kc];
      }
    }
    return sum;
  }));
}

function renderMatrix(target, matrix, maxAbs) {
  target.innerHTML = matrix
    .flatMap((row, rowIndex) => row.map((value, colIndex) => matrixCell(value, maxAbs, rowIndex * row.length + colIndex)))
    .join("");
}

function renderCnnFeatureLab(lab) {
  const pattern = controlValue(lab, "pattern");
  const kernelType = controlValue(lab, "kernel");
  const strength = controlNumber(lab, "strength");
  const input = patternMatrix(pattern, strength);
  const kernel = kernelMatrix(kernelType);
  const output = convolve2d(input, kernel);
  const outputMax = Math.max(0.1, ...output.flat().map(Math.abs));

  renderMatrix(lab.querySelector("[data-input-grid]"), input, Math.max(1, strength));
  renderMatrix(lab.querySelector("[data-kernel-grid]"), kernel, 5);
  renderMatrix(lab.querySelector("[data-output-grid]"), output, outputMax);
  lab.querySelector("[data-lab-readout]").textContent = `当前输出最大响应 ${outputMax.toFixed(2)}。边缘核会突出突变，平滑核会保留整体亮度，中心强化会放大局部差异。`;
  pulseLabReadout(lab);
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function renderAttentionLab(lab) {
  const tokens = ["深度", "学习", "需要", "注意力", "机制"];
  const baseScores = [
    [1.2, 1.1, 0.4, 0.8, 0.5],
    [1.0, 1.3, 0.5, 0.8, 0.6],
    [0.3, 0.5, 1.0, 0.7, 0.4],
    [0.6, 0.7, 0.8, 1.5, 1.3],
    [0.4, 0.5, 0.4, 1.2, 1.4],
  ];
  const query = controlNumber(lab, "query");
  const sharpness = controlNumber(lab, "sharpness");
  const noise = controlNumber(lab, "noise");
  const scores = baseScores[query].map((score, index) => (score + noise * Math.sin((query + 1) * (index + 2))) * sharpness);
  const weights = softmax(scores);
  const topIndex = weights.indexOf(Math.max(...weights));

  lab.querySelector("[data-token-row]").innerHTML = tokens
    .map((token, index) => `<span class="token-pill ${index === query ? "is-query" : ""}">${token}</span>`)
    .join("");
  lab.querySelector("[data-attention-bars]").innerHTML = tokens
    .map((token, index) => `
      <div class="attention-row">
        <span>${token}</span>
        <div><i style="width: ${(weights[index] * 100).toFixed(1)}%; animation-delay: ${index * 44}ms"></i></div>
        <strong>${(weights[index] * 100).toFixed(1)}%</strong>
      </div>
    `)
    .join("");
  lab.querySelector("[data-lab-readout]").textContent = `当前 query 是“${tokens[query]}”，最关注“${tokens[topIndex]}”。锐度越高，注意力越接近单点检索。`;
  pulseLabReadout(lab);
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function metricCard(label, value, detail, tone = "") {
  return `
    <article class="metric-card ${tone}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(detail)}</small>
    </article>
  `;
}

function stateCard(label, value, detail, tone = "") {
  return `
    <article class="state-card ${tone}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(detail)}</small>
    </article>
  `;
}

function renderSequenceMemoryLab(lab) {
  const length = controlNumber(lab, "length");
  const retention = controlNumber(lab, "retention");
  const noise = controlNumber(lab, "noise");
  const states = [];
  let state = 0;

  for (let index = 0; index < length; index += 1) {
    const anchor = index === 0 ? 0.9 : 0;
    const lateSignal = index === length - 1 ? 0.45 : 0;
    const signal = anchor + lateSignal + Math.sin((index + 1) * 1.42) * 0.42;
    const perturb = Math.cos((index + 2) * 1.73) * noise * 0.22;
    state = state * retention + signal * (1 - noise * 0.55) + perturb;
    states.push({ signal, value: state });
  }

  const maxAbs = Math.max(0.2, ...states.map((item) => Math.abs(item.value)));
  const finalState = states[states.length - 1]?.value || 0;
  const retainedMemory = Math.pow(retention, Math.max(0, length - 1));
  const noiseRisk = clampNumber(Math.round((noise * 70 + (1 - retention) * 35 + Math.max(0, length - 6) * 7)), 0, 100);

  lab.querySelector("[data-sequence-tokens]").innerHTML = states
    .map((item, index) => `<span class="${Math.abs(item.signal) > 0.72 ? "is-key" : ""}" style="--token-delay:${index * 70}ms">t${index + 1}</span>`)
    .join("");
  lab.querySelector("[data-memory-bars]").innerHTML = states
    .map((item, index) => {
      const height = 16 + Math.abs(item.value) / maxAbs * 78;
      return `<i class="${Math.abs(item.value) < maxAbs * 0.28 ? "is-faded" : ""}" style="--memory-height:${height.toFixed(1)}%; --token-delay:${index * 70}ms"></i>`;
    })
    .join("");
  lab.querySelector("[data-sequence-states]").innerHTML = [
    stateCard("最后隐藏状态", finalState.toFixed(2), "当前输出主要依赖的压缩记忆"),
    stateCard("长程保留", `${Math.round(retainedMemory * 100)}%`, "第一个 token 到末尾还能留下多少影响"),
    stateCard("噪声风险", `${noiseRisk}%`, noiseRisk > 58 ? "需要门控、注意力或更短截断" : "状态仍能保留主要线索", noiseRisk > 58 ? "is-alert" : ""),
  ].join("");
  lab.querySelector("[data-lab-readout]").textContent = `序列长度 ${length}、保留率 ${retention.toFixed(2)}、噪声 ${noise.toFixed(1)}。如果后几根记忆柱压过前面线索，模型就更像在“看最近”，而不是稳定记住长期依赖。`;
  pulseLabReadout(lab);
}

function renderTrainingDiagnosticsLab(lab) {
  const lr = controlNumber(lab, "lr");
  const regularization = controlNumber(lab, "regularization");
  const noise = controlNumber(lab, "noise");
  const instability = clampNumber((lr - 0.48) / 0.32, 0, 1);
  const tooSlow = clampNumber((0.1 - lr) / 0.08, 0, 1);
  const overfit = clampNumber((0.58 - regularization) * 1.4 + (0.42 - noise) * 0.35, 0, 1);
  const epochs = Array.from({ length: 12 }, (_, index) => index);
  const train = epochs.map((epoch) => {
    const decay = Math.exp(-epoch * (0.22 + lr * 1.18));
    return 0.12 + 1.05 * decay + noise * 0.13 + instability * epoch * 0.035 + tooSlow * 0.16;
  });
  const valid = epochs.map((epoch, index) => {
    const gap = overfit * Math.pow(index / (epochs.length - 1), 1.5) * 0.42;
    const wobble = Math.sin(index * 1.27) * noise * 0.045;
    return train[index] + 0.06 + gap + wobble + instability * index * 0.025;
  });
  const allValues = [...train, ...valid];
  const min = Math.min(...allValues) - 0.04;
  const max = Math.max(...allValues) + 0.04;
  const xFor = (index) => 42 + index * 34;
  const yFor = (value) => 238 - ((value - min) / Math.max(0.01, max - min)) * 178;
  const pathFor = (values) => values.map((value, index) => `${index === 0 ? "M" : "L"} ${xFor(index).toFixed(1)} ${yFor(value).toFixed(1)}`).join(" ");
  const gradHealth = clampNumber(Math.round(100 - instability * 62 - tooSlow * 24 - noise * 18 + regularization * 8), 0, 100);
  const gap = valid[valid.length - 1] - train[train.length - 1];
  const status = instability > 0.45 ? "学习率偏大，曲线开始震荡" : gap > 0.28 ? "验证集掉队，过拟合风险上升" : tooSlow > 0.5 ? "学习率偏小，下降太慢" : "训练和验证仍较同步";
  const plot = lab.querySelector("[data-training-plot]");

  plot.innerHTML = `
    <path class="demo-axis" d="M42 238 H424 M42 34 V238"></path>
    <text x="48" y="30">loss</text>
    <text x="374" y="262">epoch</text>
    <path class="motion-trace" d="${pathFor(train)}" pathLength="1" fill="none" stroke="#2a2118" stroke-width="2.6"></path>
    <path class="demo-compare-line" d="${pathFor(valid)}" pathLength="1" fill="none"></path>
    ${train.map((value, index) => `<circle class="motion-point" style="--point-delay:${index * 32}ms" cx="${xFor(index).toFixed(1)}" cy="${yFor(value).toFixed(1)}" r="3.7" fill="#2a2118"></circle>`).join("")}
    ${valid.map((value, index) => `<circle class="motion-point" style="--point-delay:${index * 32 + 120}ms" cx="${xFor(index).toFixed(1)}" cy="${yFor(value).toFixed(1)}" r="3.2" fill="#b08a4f"></circle>`).join("")}
  `;
  lab.querySelector("[data-training-metrics]").innerHTML = [
    metricCard("训练末端", train[train.length - 1].toFixed(2), "黑线，越低代表优化越充分"),
    metricCard("验证间隙", gap.toFixed(2), gap > 0.28 ? "泛化风险明显" : "泛化仍可接受", gap > 0.28 ? "is-alert" : ""),
    metricCard("梯度健康", `${gradHealth}%`, gradHealth < 55 ? "需要降学习率或查数据" : "更新幅度较稳定", gradHealth < 55 ? "is-alert" : ""),
  ].join("");
  lab.querySelector("[data-lab-readout]").textContent = `${status}。读训练页时不要只看 loss 降没降，要同时看验证间隙和梯度健康度。`;
  pulseLabReadout(lab);
}

function renderArchitectureFlowLab(lab) {
  const coupling = controlNumber(lab, "coupling");
  const plugins = controlNumber(lab, "plugins");
  const logging = controlNumber(lab, "logging");
  const replaceCost = clampNumber(Math.round(coupling * 0.62 + (100 - plugins) * 0.28 + (100 - logging) * 0.1), 0, 100);
  const reproducibility = clampNumber(Math.round((100 - coupling) * 0.25 + plugins * 0.25 + logging * 0.5), 0, 100);
  const extensionScore = clampNumber(Math.round((100 - coupling) * 0.35 + plugins * 0.5 + logging * 0.15), 0, 100);
  const nodes = [
    ["config", logging >= 45, "实验入口"],
    ["data", coupling <= 68, "数据接口"],
    ["model", plugins >= 45, "模型注册"],
    ["trainer", coupling <= 55 && plugins >= 50, "训练循环"],
    ["artifact", logging >= 60, "产物记录"],
  ];

  lab.querySelector("[data-architecture-flow]").innerHTML = nodes
    .map(([label, healthy, detail], index) => `
      <div class="arch-node ${healthy ? "is-active" : "is-risk"}" style="--node-delay:${index * 78}ms">
        <strong>${label}</strong>
        <small>${detail}</small>
      </div>
    `)
    .join("");
  lab.querySelector("[data-architecture-metrics]").innerHTML = [
    stateCard("替换成本", `${replaceCost}%`, replaceCost > 58 ? "改一个模型会牵动太多文件" : "边界比较清楚", replaceCost > 58 ? "is-alert" : ""),
    stateCard("可复现性", `${reproducibility}%`, reproducibility < 55 ? "缺日志和产物约定" : "能追踪一次实验"),
    stateCard("扩展能力", `${extensionScore}%`, extensionScore < 55 ? "插件入口还不够稳" : "适合新增任务或模型"),
  ].join("");
  lab.querySelector("[data-lab-readout]").textContent = `耦合度 ${coupling}%、插件化 ${plugins}%、记录完整度 ${logging}%。好的框架页应该让你看见边界：哪些稳定、哪些可换、哪里留下可复现实验。`;
  pulseLabReadout(lab);
}

function renderSystemsFlowLab(lab) {
  const load = controlNumber(lab, "load");
  const cache = controlNumber(lab, "cache");
  const failure = controlValue(lab, "failure");
  const failurePenalty = { none: 0, network: 140, cache: 85, database: 190, model: 220 }[failure] || 0;
  const latency = Math.round(72 + load * 28 * (1 - cache * 0.48) + failurePenalty);
  const bottleneck = failure !== "none"
    ? { network: "网关/网络", cache: "缓存层", database: "数据库", model: "模型服务" }[failure]
    : load >= 8 && cache < 0.45
      ? "数据库"
      : load >= 7
        ? "模型服务"
        : "链路正常";
  const firstCheck = failure === "none"
    ? (cache < 0.35 ? "先看缓存命中和慢查询" : "先看端到端耗时分布")
    : `先隔离${bottleneck}`;
  const nodes = [
    ["client", "入口", "none"],
    ["gateway", "路由/限流", "network"],
    ["cache", `${Math.round(cache * 100)}% hit`, "cache"],
    ["database", "索引/事务", "database"],
    ["model", "推理服务", "model"],
  ];

  lab.querySelector("[data-systems-flow]").innerHTML = nodes
    .map(([label, detail, key], index) => {
      const isFailure = failure === key;
      const isActive = !isFailure && (failure === "none" || index <= nodes.findIndex((node) => node[2] === failure));
      return `
        <div class="system-node ${isFailure ? "is-risk" : isActive ? "is-active" : ""}" style="--node-delay:${index * 78}ms">
          <strong>${label}</strong>
          <small>${detail}</small>
        </div>
      `;
    })
    .join("");
  lab.querySelector("[data-systems-metrics]").innerHTML = [
    metricCard("端到端延迟", `${latency}ms`, latency > 360 ? "体验会明显变慢" : "仍在可解释范围", latency > 360 ? "is-alert" : ""),
    metricCard("当前瓶颈", bottleneck, "不要先背答案，先定位层级"),
    metricCard("第一检查", firstCheck, "面试和线上排障都按这个顺序展开"),
  ].join("");
  lab.querySelector("[data-lab-readout]").textContent = `负载 ${load}/10、缓存命中 ${Math.round(cache * 100)}%、故障位置：${failure === "none" ? "无" : bottleneck}。系统题的干货是链路、瓶颈和取舍，而不是孤立定义。`;
  pulseLabReadout(lab);
}

function wireInteractiveLab(module) {
  const lab = document.querySelector("[data-lab]");
  if (!lab) return;
  const render = () => {
    if (lab.dataset.lab === "math-gradient") renderMathGradientLab(lab);
    else if (lab.dataset.lab === "cnn-feature") renderCnnFeatureLab(lab);
    else if (lab.dataset.lab === "attention") renderAttentionLab(lab);
    else if (lab.dataset.lab === "sequence-memory") renderSequenceMemoryLab(lab);
    else if (lab.dataset.lab === "training-diagnostics") renderTrainingDiagnosticsLab(lab);
    else if (lab.dataset.lab === "architecture-flow") renderArchitectureFlowLab(lab);
    else if (lab.dataset.lab === "systems-flow") renderSystemsFlowLab(lab);
  };
  lab.querySelectorAll("input, select").forEach((control) => {
    control.addEventListener("input", render);
    control.addEventListener("change", render);
  });
  render();
}

function consoleNoteFor(module) {
  const profile = getLessonProfile(module);
  const note = MODULE_TEACHING_NOTES[module.id] || {};
  const labGuide = LAB_CONTROL_GUIDES[profile.domain];
  return {
    profile,
    what: note.what || module.summary,
    variable: note.variable || profile.variable,
    controls: note.controls || labGuide.controls,
    observe: note.observe || labGuide.changes,
    why: note.why || profile.mechanism,
    consoleTask: note.consoleTask || profile.beginner.consoleTask,
  };
}

function splitConsoleControls(value) {
  return String(value)
    .split(/[、,，]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 6);
}

function consoleCanvasNodes(module, consoleNote) {
  const domainLabels = {
    foundation: ["Data", "Formula", "Gradient", "Step", "Readout"],
    cnn: ["Image", "Kernel", "Feature", "Classifier", "Heatmap"],
    sequence: ["Tokens", "State", "Memory", "Decoder", "Metric"],
    transformer: ["Query", "Key", "Softmax", "Value", "Output"],
    training: ["Dataset", "Model", "Loss", "Optimizer", "Monitor"],
    architecture: ["Config", "Data API", "Model Registry", "Trainer", "Artifact"],
    systems: ["Client", "Gateway", "Cache", "Model", "SLO"],
  };
  const labels = domainLabels[consoleNote.profile.domain] || domainLabels.training;
  const details = splitConsoleControls(consoleNote.controls);
  return labels.map((label, index) => ({
    id: `node-${index}`,
    label,
    title: index === 0 ? module.partShort : label,
    detail: details[index] || consoleNote.variable,
    x: [6, 28, 50, 70, 82][index],
    y: [18, 48, 20, 56, 30][index],
  }));
}

function renderNodeCanvas(module, consoleNote) {
  const nodes = consoleCanvasNodes(module, consoleNote);
  return `
    <section class="console-workbench console-node-section">
      <div class="section-head">
        <div>
          <div class="eyebrow">Node Canvas</div>
          <h2>拖拽式节点画布</h2>
        </div>
        <p>把当前知识点拆成数据、模型、损失、优化和观察节点。拖动节点，连线和读数会即时更新，用来训练“结构先于参数”的建模习惯。</p>
      </div>
      <div class="node-workspace">
        <div class="node-canvas" data-node-canvas aria-label="可拖拽模型节点画布">
          <svg class="canvas-edge-layer" data-canvas-edges aria-hidden="true">
            ${nodes.slice(0, -1).map((node, index) => `<line data-edge-from="${node.id}" data-edge-to="${nodes[index + 1].id}"></line>`).join("")}
          </svg>
          ${nodes.map((node, index) => `
            <button class="canvas-node" type="button" data-node="${node.id}" style="left:${node.x}%; top:${node.y}%; --node-delay:${index * 70}ms">
              <span>${escapeHtml(node.label)}</span>
              <strong>${escapeHtml(node.title)}</strong>
              <small>${escapeHtml(node.detail)}</small>
            </button>
          `).join("")}
        </div>
        <aside class="canvas-readout" data-node-readout>
          <span>Canvas Readout</span>
          <strong>拖动任意节点</strong>
          <p>观察拓扑距离如何改变：节点越分散，越适合解释流程；节点越靠近，越适合排查局部耦合。</p>
        </aside>
      </div>
    </section>
  `;
}

const MODEL_LAYER_LIBRARY = [
  { type: "Embedding", label: "Embedding", factory: () => ({ type: "Embedding", name: "Token Embedding", vocabSize: 8000, dim: 64 }) },
  { type: "MultiHeadAttention", label: "MultiHead Attention", factory: () => ({ type: "MultiHeadAttention", name: "MultiHeadAttention", heads: 4 }) },
  { type: "LayerNorm", label: "LayerNorm", factory: () => ({ type: "LayerNorm", name: "LayerNorm" }) },
  { type: "ResidualBlock", label: "Residual Block", factory: () => ({ type: "ResidualBlock", name: "ResidualBlock", expectedInput: [32, 64] }) },
  { type: "TransformerEncoder", label: "Transformer Encoder", factory: () => ({ type: "TransformerEncoder", name: "TransformerEncoder", heads: 4, ff: 256 }) },
  { type: "Flatten", label: "Flatten", factory: () => ({ type: "Flatten", name: "Flatten" }) },
  { type: "Dropout", label: "Dropout", factory: () => ({ type: "Dropout", name: "Dropout", p: 0.1 }) },
  { type: "Linear", label: "Linear", factory: () => ({ type: "Linear", name: "Linear", outFeatures: 10 }) },
];

const MODEL_PRESETS = {
  mlp: {
    id: "mlp",
    name: "MLP baseline",
    inputShape: [784],
    layers: [
      { type: "Linear", name: "Linear 784 -> 128", inFeatures: 784, outFeatures: 128 },
      { type: "Dropout", name: "Dropout", p: 0.2 },
      { type: "Linear", name: "Linear 128 -> 10", inFeatures: 128, outFeatures: 10 },
    ],
  },
  cnn: {
    id: "cnn",
    name: "CNN classifier",
    inputShape: [1, 28, 28],
    layers: [
      { type: "Conv2D", name: "Conv2D 1 -> 16", outShape: [16, 26, 26] },
      { type: "ResidualBlock", name: "ResidualBlock", expectedInput: [16, 26, 26] },
      { type: "Flatten", name: "Flatten" },
      { type: "Linear", name: "Linear 10816 -> 10", inFeatures: 10816, outFeatures: 10 },
    ],
  },
  transformer_mini: {
    id: "transformer_mini",
    name: "Transformer mini",
    inputShape: [32],
    layers: [
      { type: "Embedding", name: "Token Embedding", vocabSize: 8000, dim: 64, outShape: [32, 64] },
      { type: "MultiHeadAttention", name: "MultiHeadAttention", heads: 4, expectedInput: [32, 64] },
      { type: "ResidualBlock", name: "ResidualBlock", expectedInput: [32, 64] },
      { type: "LayerNorm", name: "LayerNorm", expectedInput: [32, 64] },
      { type: "TransformerEncoder", name: "TransformerEncoder", heads: 4, ff: 256, expectedInput: [32, 64] },
      { type: "Linear", name: "Classifier", inFeatures: 64, outFeatures: 10 },
    ],
  },
  shape_error: {
    id: "shape_error",
    name: "Shape error demo",
    inputShape: [32, 48],
    layers: [
      { type: "MultiHeadAttention", name: "MultiHeadAttention", heads: 4, expectedInput: [32, 64] },
      { type: "ResidualBlock", name: "ResidualBlock", expectedInput: [32, 64] },
      { type: "LayerNorm", name: "LayerNorm", expectedInput: [32, 64] },
    ],
  },
};

function cloneModelGraph(graph) {
  return JSON.parse(JSON.stringify(graph));
}

function shapeToText(shape) {
  return Array.isArray(shape) ? `[${shape.join(", ")}]` : "unknown";
}

function shapeSize(shape) {
  return Array.isArray(shape) ? shape.reduce((product, value) => product * Number(value || 1), 1) : 0;
}

function sameShape(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Number(value) === Number(right[index]));
}

function inferLayerShape(layer, currentShape) {
  if (layer.expectedInput && currentShape && !sameShape(currentShape, layer.expectedInput)) {
    return {
      input: currentShape,
      output: currentShape,
      error: `shape mismatch at ${layer.name}: expected ${shapeToText(layer.expectedInput)}, got ${shapeToText(currentShape)}. Insert Linear/projection layer before ${layer.name}.`,
    };
  }
  if (layer.type === "Embedding") {
    return { input: currentShape, output: layer.outShape || [currentShape?.[0] || 32, layer.dim || 64] };
  }
  if (layer.type === "Conv2D") {
    return { input: currentShape, output: layer.outShape || currentShape };
  }
  if (layer.type === "Flatten") {
    return { input: currentShape, output: [shapeSize(currentShape)] };
  }
  if (layer.type === "Linear") {
    const last = Array.isArray(currentShape) ? currentShape[currentShape.length - 1] : null;
    if (layer.inFeatures && last && Number(layer.inFeatures) !== Number(last)) {
      return {
        input: currentShape,
        output: [layer.outFeatures || 10],
        error: `shape mismatch at ${layer.name}: Linear expects last dim ${layer.inFeatures}, got ${last}. Insert Linear/projection layer or change in_features.`,
      };
    }
    if (Array.isArray(currentShape) && currentShape.length > 1) {
      return { input: currentShape, output: [...currentShape.slice(0, -1), layer.outFeatures || 10] };
    }
    return { input: currentShape, output: [layer.outFeatures || 10] };
  }
  return { input: currentShape, output: currentShape };
}

function inferModelGraph(graph) {
  let currentShape = graph.inputShape || [];
  const rows = [];
  const issues = [];
  (graph.layers || []).forEach((layer, index) => {
    const inferred = inferLayerShape(layer, currentShape);
    rows.push({ ...layer, index, input: inferred.input, output: inferred.output, error: inferred.error || "" });
    if (inferred.error) issues.push(inferred.error);
    currentShape = inferred.output;
  });
  return { rows, issues, outputShape: currentShape };
}

function generatePyTorchCode(graph) {
  const layers = graph.layers || [];
  const hasTransformer = layers.some((layer) => ["MultiHeadAttention", "TransformerEncoder", "LayerNorm", "Embedding"].includes(layer.type));
  const initLines = [];
  const forwardLines = [];
  layers.forEach((layer, index) => {
    const name = `layer_${index}`;
    if (layer.type === "Embedding") {
      initLines.push(`        self.${name} = nn.Embedding(${layer.vocabSize || 8000}, ${layer.dim || 64})`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "MultiHeadAttention") {
      initLines.push(`        self.${name} = nn.MultiheadAttention(embed_dim=64, num_heads=${layer.heads || 4}, batch_first=True)`);
      forwardLines.push(`        x, _ = self.${name}(x, x, x)`);
    } else if (layer.type === "TransformerEncoder") {
      initLines.push(`        self.${name} = nn.TransformerEncoderLayer(d_model=64, nhead=${layer.heads || 4}, dim_feedforward=${layer.ff || 256}, batch_first=True)`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "LayerNorm") {
      initLines.push(`        self.${name} = nn.LayerNorm(64)`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "Dropout") {
      initLines.push(`        self.${name} = nn.Dropout(p=${layer.p ?? 0.1})`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "Flatten") {
      initLines.push(`        self.${name} = nn.Flatten()`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "Linear") {
      initLines.push(`        self.${name} = nn.Linear(${layer.inFeatures || 64}, ${layer.outFeatures || 10})`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "Conv2D") {
      initLines.push(`        self.${name} = nn.Conv2d(1, 16, kernel_size=3)`);
      forwardLines.push(`        x = self.${name}(x)`);
    } else if (layer.type === "ResidualBlock") {
      initLines.push(`        self.${name} = nn.Identity()`);
      forwardLines.push(`        x = x + self.${name}(x)`);
    }
  });
  return [
    "import torch",
    "from torch import nn",
    "",
    "",
    "class VisualModel(nn.Module):",
    "    def __init__(self):",
    "        super().__init__()",
    initLines.length ? initLines.join("\n") : "        self.identity = nn.Identity()",
    "",
    "    def forward(self, x):",
    forwardLines.length ? forwardLines.join("\n") : "        x = self.identity(x)",
    hasTransformer ? "        # For classification, pool sequence output before the final head in production." : "        # Keep this generated code as a teaching scaffold.",
    "        return x",
  ].join("\n");
}

function renderModelBuilder(module) {
  const options = Object.values(MODEL_PRESETS)
    .map((preset) => `<option value="${preset.id}">${escapeHtml(preset.name)}</option>`)
    .join("");
  return `
    <section class="console-workbench console-builder" data-model-builder data-testid="console-builder">
      <div class="section-head">
        <div>
          <div class="eyebrow">Model Builder</div>
          <h2>神经网络乐高工厂</h2>
        </div>
        <p>${escapeHtml(`把 ${module.title} 放进一个真实模型骨架里：加载预设、检查 shape、导出 PyTorch、保存 JSON，再故意制造错误看诊断。`)}</p>
      </div>
      <div class="model-builder-grid">
        <div class="model-builder-panel">
          <label class="model-control">预设模型
            <select data-model-preset data-testid="model-preset">${options}</select>
          </label>
          <div class="model-action-row">
            <button class="action" type="button" data-load-preset data-testid="load-preset">加载预设</button>
            <button class="ghost-action" type="button" data-clear-graph data-testid="clear-graph">清空结构</button>
          </div>
          <div class="layer-palette" aria-label="可添加层">
            ${MODEL_LAYER_LIBRARY.map((layer) => `<button type="button" data-add-layer="${layer.type}">${escapeHtml(layer.label)}</button>`).join("")}
          </div>
        </div>
        <div class="model-node-list" data-model-node-list data-testid="model-node-list"></div>
        <div class="shape-diagnostics" data-shape-diagnostics data-testid="shape-diagnostics"></div>
      </div>
      <div class="model-io-grid">
        <div class="model-builder-panel">
          <div class="model-action-row">
            <button class="action" type="button" data-export-code data-testid="export-code">导出 PyTorch</button>
            <button class="ghost-action" type="button" data-save-graph data-testid="save-graph">保存 JSON</button>
            <button class="ghost-action" type="button" data-load-graph data-testid="load-graph">加载 JSON</button>
          </div>
          <textarea class="graph-json" data-graph-json data-testid="graph-json" rows="8" spellcheck="false" placeholder="保存后的模型 JSON 会出现在这里，也可以粘贴回来恢复。"></textarea>
        </div>
        <pre class="code-window code-export is-expanded"><code data-code-export data-testid="code-export">点击“导出 PyTorch”生成可读代码。</code></pre>
      </div>
    </section>
  `;
}

function paintModelBuilder(root) {
  const builder = root.querySelector("[data-model-builder]");
  if (!builder) return;
  const graph = builder._graph || cloneModelGraph(MODEL_PRESETS.mlp);
  builder._graph = graph;
  const inferred = inferModelGraph(graph);
  const list = builder.querySelector("[data-model-node-list]");
  const diagnostics = builder.querySelector("[data-shape-diagnostics]");
  list.innerHTML = graph.layers?.length
    ? inferred.rows.map((layer) => `
      <article class="model-node ${layer.error ? "has-error" : ""}">
        <span>${layer.index + 1}</span>
        <div>
          <strong>${escapeHtml(layer.name || layer.type)}</strong>
          <p>${escapeHtml(layer.type)} · ${shapeToText(layer.input)} -> ${shapeToText(layer.output)}</p>
        </div>
      </article>
    `).join("")
    : `<article class="model-node is-empty"><strong>No layers</strong><p>Load a preset or add a layer from the palette.</p></article>`;
  diagnostics.innerHTML = inferred.issues.length
    ? inferred.issues.map((issue) => `<article class="diagnostic-error"><strong>shape mismatch</strong><p>${escapeHtml(issue)}</p><small>Fix: Insert Linear/projection layer, or change the previous layer output dimension.</small></article>`).join("")
    : `<article class="diagnostic-ok"><strong>OK</strong><p>All layer shapes are connected. Output shape: ${shapeToText(inferred.outputShape)}</p></article>`;
}

function wireModelBuilder(root) {
  const builder = root.querySelector("[data-model-builder]");
  if (!builder) return;
  const jsonArea = builder.querySelector("[data-graph-json]");
  const code = builder.querySelector("[data-code-export]");
  const setGraph = (graph) => {
    builder._graph = cloneModelGraph(graph);
    paintModelBuilder(root);
  };
  setGraph(MODEL_PRESETS.mlp);
  builder.querySelector("[data-load-preset]")?.addEventListener("click", () => {
    const id = builder.querySelector("[data-model-preset]")?.value || "mlp";
    setGraph(MODEL_PRESETS[id] || MODEL_PRESETS.mlp);
  });
  builder.querySelector("[data-clear-graph]")?.addEventListener("click", () => {
    setGraph({ id: "custom", name: "Custom graph", inputShape: [64], layers: [] });
  });
  builder.querySelectorAll("[data-add-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = MODEL_LAYER_LIBRARY.find((layer) => layer.type === button.dataset.addLayer);
      if (!item) return;
      const graph = builder._graph || cloneModelGraph(MODEL_PRESETS.mlp);
      graph.id = "custom";
      graph.name = "Custom graph";
      graph.layers = [...(graph.layers || []), item.factory()];
      setGraph(graph);
    });
  });
  builder.querySelector("[data-save-graph]")?.addEventListener("click", () => {
    const graph = builder._graph || cloneModelGraph(MODEL_PRESETS.mlp);
    const value = JSON.stringify(graph, null, 2);
    jsonArea.value = value;
    try {
      localStorage.setItem("deep-learning-book-console-graph-v1", value);
    } catch (error) {
      // Saving to textarea is enough when storage is blocked.
    }
  });
  builder.querySelector("[data-load-graph]")?.addEventListener("click", () => {
    try {
      const value = jsonArea.value.trim() || localStorage.getItem("deep-learning-book-console-graph-v1") || "";
      if (!value) return;
      const graph = JSON.parse(value);
      if (!Array.isArray(graph.layers)) throw new Error("graph.layers must be an array");
      setGraph(graph);
    } catch (error) {
      builder.querySelector("[data-shape-diagnostics]").innerHTML = `<article class="diagnostic-error"><strong>JSON error</strong><p>${escapeHtml(error.message)}</p></article>`;
    }
  });
  builder.querySelector("[data-export-code]")?.addEventListener("click", () => {
    code.textContent = generatePyTorchCode(builder._graph || MODEL_PRESETS.mlp);
  });
}

function renderTrainingEventBus(module, consoleNote) {
  return `
    <section class="console-workbench training-event-bus" data-training-bus>
      <div class="section-head">
        <div>
          <div class="eyebrow">Training Event Bus</div>
          <h2>训练事件总线</h2>
        </div>
        <p>每次调参都会发布同一个训练事件，损失曲线、梯度监控、特征/注意力观察和实验笔记从同一份事件里更新。</p>
      </div>
      <div class="event-bus-layout">
        <div class="event-log" data-event-log aria-live="polite"></div>
        <div class="event-subscriber-grid">
          <article class="event-subscriber" data-event-subscriber="loss">
            <span>Loss Curve</span>
            <strong data-bus-loss>--</strong>
            <small data-bus-loss-detail>${escapeHtml(module.title)}</small>
          </article>
          <article class="event-subscriber" data-event-subscriber="gradient">
            <span>Gradient Monitor</span>
            <strong data-bus-gradient>--</strong>
            <small data-bus-gradient-detail>等待训练事件</small>
          </article>
          <article class="event-subscriber" data-event-subscriber="feature">
            <span>Feature / Attention</span>
            <strong data-bus-feature>--</strong>
            <small data-bus-feature-detail>${escapeHtml(consoleNote.profile.label)}</small>
          </article>
          <article class="event-subscriber" data-event-subscriber="note">
            <span>Experiment Note</span>
            <strong data-bus-note>--</strong>
            <small data-bus-note-detail>记录同一次调参的结论</small>
          </article>
        </div>
      </div>
    </section>
  `;
}

function buildTrainingEvent(module, consoleNote, lab, controls, step) {
  const controlPairs = controls.map((control) => {
    const label = control.closest("label")?.innerText.replace(/\s+/g, " ").trim() || control.dataset.control || "参数";
    const value = control.tagName === "SELECT" ? control.value : Number(control.value);
    return { key: control.dataset.control || label, label, value };
  });
  const numericValues = controlPairs
    .map((item) => Number(item.value))
    .filter((value) => Number.isFinite(value));
  const numericMean = numericValues.length
    ? numericValues.reduce((sum, value) => sum + Math.abs(value), 0) / numericValues.length
    : step + 1;
  const domain = consoleNote.profile.domain;
  const domainBias = ["foundation", "cnn", "sequence", "transformer", "training", "architecture", "systems"].indexOf(domain) + 1;
  const loss = clampNumber(1.18 - step * 0.075 + Math.sin(numericMean + domainBias) * 0.09 + domainBias * 0.018, 0.08, 1.34);
  const gradient = clampNumber(0.22 + numericMean * 0.08 + Math.abs(Math.cos(step + domainBias)) * 0.38, 0.05, 1.85);
  const parameterShift = clampNumber(Math.round((gradient * 28 + step * 4 + domainBias * 3)), 4, 100);
  const attentionSignal = domain === "transformer"
    ? `注意力峰值 ${(100 - loss * 34).toFixed(0)}%`
    : domain === "cnn"
      ? `特征响应 ${(parameterShift * 0.9).toFixed(0)}%`
      : domain === "sequence"
        ? `记忆保留 ${(100 - parameterShift * 0.42).toFixed(0)}%`
        : `状态变化 ${parameterShift}%`;
  return {
    step,
    module: module.title,
    moduleId: module.id,
    domain,
    controls: controlPairs.map((item) => `${item.label}=${item.value}`).join("；"),
    readout: lab.querySelector("[data-lab-readout]")?.innerText || "等待读数",
    loss,
    gradient,
    parameterShift,
    attentionSignal,
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
  };
}

function publishTrainingEvent(root, event) {
  const bus = root.querySelector("[data-training-bus]");
  if (!bus) return;
  bus._events = [event, ...(bus._events || [])].slice(0, 6);
  bus.querySelector("[data-event-log]").innerHTML = bus._events.map((item) => `
    <article class="event-log-item">
      <span>#${item.step} · ${escapeHtml(item.time)}</span>
      <strong>${escapeHtml(item.module)}</strong>
      <p>${escapeHtml(item.controls)}</p>
    </article>
  `).join("");
  bus.querySelector("[data-bus-loss]").textContent = event.loss.toFixed(3);
  bus.querySelector("[data-bus-loss-detail]").textContent = event.loss > 0.82 ? "loss 仍偏高，继续观察下降趋势" : "loss 已进入可用区间";
  bus.querySelector("[data-bus-gradient]").textContent = event.gradient.toFixed(2);
  bus.querySelector("[data-bus-gradient-detail]").textContent = event.gradient > 1.15 ? "梯度偏大，优先查学习率和归一化" : "梯度幅度较稳定";
  bus.querySelector("[data-bus-feature]").textContent = event.attentionSignal;
  bus.querySelector("[data-bus-feature-detail]").textContent = event.readout.slice(0, 68);
  bus.querySelector("[data-bus-note]").textContent = `Step ${event.step}`;
  bus.querySelector("[data-bus-note-detail]").textContent = `参数位移 ${event.parameterShift}% · ${event.moduleId}`;
}

function updateCanvasEdges(canvas) {
  const canvasRect = canvas.getBoundingClientRect();
  canvas.querySelectorAll("[data-edge-from]").forEach((line) => {
    const from = canvas.querySelector(`[data-node="${line.dataset.edgeFrom}"]`);
    const to = canvas.querySelector(`[data-node="${line.dataset.edgeTo}"]`);
    if (!from || !to) return;
    const fromRect = from.getBoundingClientRect();
    const toRect = to.getBoundingClientRect();
    line.setAttribute("x1", fromRect.left + fromRect.width / 2 - canvasRect.left);
    line.setAttribute("y1", fromRect.top + fromRect.height / 2 - canvasRect.top);
    line.setAttribute("x2", toRect.left + toRect.width / 2 - canvasRect.left);
    line.setAttribute("y2", toRect.top + toRect.height / 2 - canvasRect.top);
  });
}

function nodeDistanceSummary(canvas) {
  const nodes = [...canvas.querySelectorAll("[data-node]")];
  if (nodes.length < 2) return { average: 0, widest: 0 };
  const centers = nodes.map((node) => {
    const rect = node.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    return {
      x: rect.left + rect.width / 2 - canvasRect.left,
      y: rect.top + rect.height / 2 - canvasRect.top,
    };
  });
  const distances = centers.slice(0, -1).map((point, index) => {
    const next = centers[index + 1];
    return Math.hypot(next.x - point.x, next.y - point.y);
  });
  return {
    average: Math.round(distances.reduce((sum, value) => sum + value, 0) / distances.length),
    widest: Math.round(Math.max(...distances)),
  };
}

function wireNodeCanvas(root) {
  const canvas = root.querySelector("[data-node-canvas]");
  const readout = root.querySelector("[data-node-readout]");
  if (!canvas || !readout) return;

  const updateReadout = (label = "节点拓扑") => {
    const summary = nodeDistanceSummary(canvas);
    readout.innerHTML = `
      <span>Canvas Readout</span>
      <strong>${escapeHtml(label)}</strong>
      <p>平均相邻距离 ${summary.average}px，最大跨度 ${summary.widest}px。距离变大适合讲流程，距离变小通常意味着需要检查局部耦合。</p>
    `;
  };

  canvas.querySelectorAll("[data-node]").forEach((node) => {
    let drag = null;
    node.addEventListener("pointerdown", (event) => {
      const canvasRect = canvas.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      drag = {
        pointerId: event.pointerId,
        offsetX: event.clientX - nodeRect.left,
        offsetY: event.clientY - nodeRect.top,
        canvasRect,
      };
      try {
        node.setPointerCapture(event.pointerId);
      } catch (error) {
        // Synthetic browser tests may not register an active pointer capture.
      }
      node.classList.add("is-dragging");
      event.preventDefault();
    });
    node.addEventListener("pointermove", (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const canvasRect = canvas.getBoundingClientRect();
      const width = node.offsetWidth;
      const height = node.offsetHeight;
      const left = clampNumber(event.clientX - canvasRect.left - drag.offsetX, 6, canvasRect.width - width - 6);
      const top = clampNumber(event.clientY - canvasRect.top - drag.offsetY, 6, canvasRect.height - height - 6);
      node.style.left = `${left + width / 2}px`;
      node.style.top = `${top + height / 2}px`;
      updateCanvasEdges(canvas);
      updateReadout(node.querySelector("span")?.innerText || "节点移动中");
    });
    const finish = (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      node.classList.remove("is-dragging");
      try {
        node.releasePointerCapture(event.pointerId);
      } catch (error) {
        // See pointer capture note in pointerdown.
      }
      drag = null;
      updateCanvasEdges(canvas);
      updateReadout(node.querySelector("span")?.innerText || "节点已移动");
    };
    node.addEventListener("pointerup", finish);
    node.addEventListener("pointercancel", finish);
  });

  requestAnimationFrame(() => {
    updateCanvasEdges(canvas);
    updateReadout();
  });
  window.addEventListener("resize", () => updateCanvasEdges(canvas), { passive: true });
}

function renderCentralConsole(id) {
  const module = byId(id);
  if (!module) {
    app.innerHTML = `<section class="view section"><div class="empty-state">没有找到可实战的知识点：${escapeHtml(id)}。<br><a href="#home">返回首页</a></div></section>`;
    kickRouteMotion();
    requestAnimationFrame(() => applyMotionReveal());
    return;
  }

  const consoleNote = consoleNoteFor(module);
  const controlItems = splitConsoleControls(consoleNote.controls);
  const adjacent = adjacentModules(module);
  app.innerHTML = `
    <section class="view central-console" data-central-console data-module-id="${escapeHtml(module.id)}">
      <div class="console-topline">
        <a class="ghost-action" href="${moduleHref(module)}">返回课程</a>
        ${adjacent.previous ? `<a class="ghost-action" href="${moduleHref(adjacent.previous)}">上一节</a>` : ""}
        ${adjacent.next ? `<a class="ghost-action" href="${moduleHref(adjacent.next)}">下一节</a>` : ""}
        <a class="ghost-action" href="#home">返回首页</a>
      </div>

      <header class="console-hero">
        <div>
          <div class="eyebrow">第 1 步：先动一个参数</div>
          <h1>中央控制台实战</h1>
          <p>${escapeHtml(`从“${module.title}”进入。先不要读完所有说明，直接在下面实验台只改一个控件，观察画面和读数是否按你的预期变化。`)}</p>
        </div>
        <div class="console-context">
          <span>${escapeHtml(module.partShort)}</span>
          <strong>${escapeHtml(module.title)}</strong>
          <small>${escapeHtml(module.id)}</small>
        </div>
      </header>

      <section class="console-task-strip">
        <div>
          <span>现在做什么</span>
          <strong>${escapeHtml(controlItems[0] || "第一个参数")}</strong>
          <p>${escapeHtml(`先调整“${controlItems[0] || "第一个参数"}”，看读数区如何变化。一次只改一个变量，才知道因果关系。`)}</p>
        </div>
        <a class="action" href="#console-workbench" data-scroll-console="console-workbench">跳到实验台</a>
      </section>

      <section class="console-purpose-strip">
        <strong>为什么从这节来控制台？</strong>
        <p>${escapeHtml(consoleNote.consoleTask)}</p>
        <ol>
          <li>先写下一个预测：调大或调小这个参数，图会怎样变。</li>
          <li>只改一个控件，观察读数区是否支持你的预测。</li>
          <li>回到课程页点“我已理解”或“加入复习”。</li>
        </ol>
      </section>

      <section id="console-workbench" class="console-workbench">
        <div class="section-head">
          <div>
            <div class="eyebrow">实验台</div>
            <h2>统一实验台</h2>
          </div>
          <p>${escapeHtml(`当前实验类型：${consoleNote.profile.label}。先改一个控件，再看画面、指标和观察记录。`)}</p>
        </div>
        ${renderInteractiveLab(module)}
      </section>

      <div class="console-grid">
        <aside class="console-panel">
          <div class="section-kicker">当前知识</div>
          <h2>从当前知识点带入</h2>
          <p>${escapeHtml(consoleNote.what)}</p>
          <dl class="console-dl">
            <div><dt>关键变量</dt><dd>${escapeHtml(consoleNote.variable)}</dd></div>
            <div><dt>观察变化</dt><dd>${escapeHtml(consoleNote.observe)}</dd></div>
            <div><dt>为什么</dt><dd>${escapeHtml(consoleNote.why)}</dd></div>
          </dl>
        </aside>

        <aside class="console-panel">
          <div class="section-kicker">参数迁移</div>
          <h2>参数迁移清单</h2>
          <p>${escapeHtml(consoleNote.consoleTask)}</p>
          <div class="console-chip-list">
            ${controlItems.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
          </div>
          <ol class="console-steps">
            <li>先写下你预计会改变的图形或指标。</li>
            <li>一次只拖动一个控件，保留其它条件不变。</li>
            <li>用读数区判断结果是否验证了假设。</li>
          </ol>
        </aside>
      </div>

      ${renderModelBuilder(module)}

      ${renderNodeCanvas(module, consoleNote)}

      ${renderTrainingEventBus(module, consoleNote)}

      <section class="console-grid">
        <div class="console-panel">
          <div class="section-kicker">实时读数</div>
          <h2>实战读数</h2>
          <div class="console-metrics" data-console-metrics></div>
          <p class="console-result" data-console-result></p>
        </div>
        <div class="console-panel">
          <div class="section-kicker">观察记录</div>
          <h2>观察记录</h2>
          <textarea data-console-note rows="6" spellcheck="false">我调了：
画面/指标变化：
这说明：
下一步：</textarea>
          <button class="action" type="button" data-console-fill>生成观察结论</button>
        </div>
      </section>
    </section>
  `;

  kickRouteMotion();
  requestAnimationFrame(() => applyMotionReveal());
  app.focus({ preventScroll: true });
  wireInteractiveLab(module);
  wireCentralConsole(module);
}

function wireCentralConsole(module) {
  const root = document.querySelector("[data-central-console]");
  const lab = root?.querySelector("[data-lab]");
  if (!root || !lab) return;

  const consoleNote = consoleNoteFor(module);
  const result = root.querySelector("[data-console-result]");
  const metrics = root.querySelector("[data-console-metrics]");
  const noteArea = root.querySelector("[data-console-note]");
  const fillButton = root.querySelector("[data-console-fill]");
  const controls = [...lab.querySelectorAll("input, select")];
  let eventStep = 0;

  const controlLabel = (control) => control.closest("label")?.innerText.replace(/\s+/g, " ").trim() || control.dataset.control || "参数";
  const currentControlState = () => controls
    .map((control) => `${controlLabel(control)}=${control.tagName === "SELECT" ? control.selectedOptions[0]?.textContent : control.value}`)
    .join("；");

  const update = () => {
    const readout = lab.querySelector("[data-lab-readout]")?.innerText || "";
    metrics.innerHTML = [
      metricCard("当前知识点", module.title, module.partShort),
      metricCard("迁移控件", `${controls.length} 个`, controls.map(controlLabel).slice(0, 3).join(" / ")),
      metricCard("实验类型", consoleNote.profile.label, consoleNote.profile.domain),
    ].join("");
    result.textContent = `当前控件：${currentControlState()}。实时读数：${readout || "等待实验台完成首次渲染。"}`;
    eventStep += 1;
    publishTrainingEvent(root, buildTrainingEvent(module, consoleNote, lab, controls, eventStep));
  };

  controls.forEach((control) => {
    control.addEventListener("input", () => requestAnimationFrame(update));
    control.addEventListener("change", () => requestAnimationFrame(update));
  });
  fillButton?.addEventListener("click", () => {
    const readout = lab.querySelector("[data-lab-readout]")?.innerText || "";
    noteArea.value = [
      `我调了：${currentControlState()}`,
      `画面/指标变化：${readout}`,
      `这说明：${module.title} 的关键变量会通过“${consoleNote.variable}”传导到可见结果。`,
      `下一步：回到课程页核对“为什么会这样”和“常见误区”，再换一个极端参数复现实验。`,
    ].join("\n");
  });
  root.querySelectorAll("[data-scroll-console]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      scrollToHashTarget(`#${button.dataset.scrollConsole}`, 0);
    });
  });
  wireModelBuilder(root);
  wireNodeCanvas(root);
  update();
}

async function renderCourse(id) {
  const module = byId(id);
  if (!module) {
    app.innerHTML = `<section class="view section"><div class="empty-state">没有找到课程：${escapeHtml(id)}。<br><a href="#home">返回首页</a></div></section>`;
    kickRouteMotion();
    requestAnimationFrame(() => applyMotionReveal());
    return;
  }
  const adjacent = adjacentModules(module);
  const currentProgress = readLearningProgress();
  const isUnderstood = currentProgress.understood.includes(module.id);
  const isReview = currentProgress.review.includes(module.id);
  const mode = readLearningMode();
  const courseSections = [
    ["course-start", "学习目标"],
    ["course-animation", "概念动画"],
    ["course-lab", "动手实验"],
    ["course-reading", "章节精读"],
    ["course-credibility", "可信度与来源"],
    ...(mode === "advanced" ? [["course-source", "开发者信息"]] : []),
  ];
  app.innerHTML = `
    <section class="view course-layout" data-testid="course-layout">
      <article class="course-article">
        <nav class="course-topline" aria-label="课程上下文导航">
          <a class="ghost-action" href="#home">返回首页</a>
          ${adjacent.previous ? `<a class="ghost-action" href="${moduleHref(adjacent.previous)}">上一节</a>` : `<a class="ghost-action" href="#starter">本路径起点</a>`}
          ${adjacent.next ? `<a class="action" data-testid="next-lesson-top" href="${moduleHref(adjacent.next)}">下一节</a>` : `<a class="action" data-testid="next-lesson-top" href="#path">回到学习路径</a>`}
        </nav>
        <div class="eyebrow" style="margin-top: 34px;">${escapeHtml(module.partTitle)}</div>
        <h1>${escapeHtml(module.title)}</h1>
        <p class="summary">${escapeHtml(module.summary)}</p>
        ${tagHtml(module.tags, { interactive: true })}

        <section id="course-start" class="course-anchor-section">
          <div class="student-route-card" data-testid="lesson-roadmap">
            <strong>本课主线</strong>
            <ol>
              <li>先读“这一节掌握什么”，明确完成标准。</li>
              <li>看概念动画，只盯颜色、方向和读数。</li>
              <li>做一次单因素实验，只改一个控件。</li>
              <li>再读精读讲义和源码对照。</li>
            </ol>
          </div>
          ${renderImmediateThreeMinuteBrief(module)}
          ${renderLessonBrief(module)}
        </section>
        <section id="course-animation" class="course-anchor-section">
          ${renderConceptAnimation(module)}
        </section>
        <section id="course-lab" class="course-anchor-section">
          ${renderInteractiveLab(module)}
        </section>
        ${renderZeroBasics(module)}
        <div class="advanced-only">
          ${renderDryGoods(module)}
        </div>
        <section id="course-reading" class="course-anchor-section">
          ${renderLessonDeepDiveShell(module)}
        </section>
        ${renderCredibilitySection(module)}
        ${renderKnowledgeSections(module)}
        ${isLLMCookbookRelevant(module) ? `<div class="advanced-only">${renderLLMCookbookBridge(module)}</div>` : ""}

        <section id="course-source" class="reading-section course-anchor-section developer-source advanced-only" data-testid="developer-source">
          <h2>源码对照：只看和动画有关的几段</h2>
          <p class="summary">这里不是要求小白阅读整份 Python/Streamlit 文件。先看下面三张“动画对应 / 控件对应 / 读数对应”卡片；完整源码只给想继续深挖的同学展开。</p>
          <div data-source-guide class="source-guide-loading">正在把源码拆成教学对照卡片...</div>
          <div class="code-toolbar">
            <strong>开发者完整源码：${escapeHtml(module.sourcePath)}</strong>
            <button class="ghost-action" type="button" data-toggle-code>展开代码</button>
          </div>
          <pre class="code-window"><code data-source>正在读取源码...</code></pre>
        </section>
      </article>
      <aside class="course-aside" data-testid="course-toc">
        <div class="eyebrow">学习导航</div>
        <h2>当前课怎么学</h2>
        <p class="aside-summary">${escapeHtml(module.summary)}</p>
        <div class="course-progress-list">
          ${courseSections.map(([target, label], index) => `
            <button type="button" data-course-scroll="${target}" data-testid="toc-${target.replace("course-", "")}">
              <span>${index + 1}</span>
              ${label}
            </button>
          `).join("")}
        </div>
        <details class="developer-meta advanced-only">
          <summary>开发者信息</summary>
          <dl>
            <div><dt>章节</dt><dd>${escapeHtml(module.partShort)}</dd></div>
            <div><dt>层级</dt><dd>${escapeHtml(module.level)}</dd></div>
            <div><dt>源码路径</dt><dd>${escapeHtml(module.sourcePath)}</dd></div>
          </dl>
        </details>
        <div class="course-next-actions">
          ${adjacent.previous ? `<a class="ghost-action" href="${moduleHref(adjacent.previous)}">上一节</a>` : `<a class="ghost-action" href="#starter">新手路径</a>`}
          ${adjacent.next ? `<a class="action" data-testid="next-lesson" href="${moduleHref(adjacent.next)}">下一节</a>` : `<a class="action" data-testid="next-lesson" href="#path">回到路径</a>`}
          <a class="ghost-action" href="#path">返回本路径</a>
          <a class="ghost-action" href="#courses">全站课程目录</a>
        </div>
        <div class="course-console-cta">
          <strong>什么时候去控制台？</strong>
          <p>${escapeHtml(`完成动画和动手实验后，再去控制台复现“${module.title}”的关键变量。目标不是换页面，而是验证你能预测变化。`)}</p>
          <a class="action" href="${consoleHref(module)}">去控制台完成 1 个验证</a>
        </div>
        <div class="mode-switcher" aria-label="学习模式切换">
          <span>显示模式</span>
          <button type="button" data-learning-mode="beginner" data-testid="mode-beginner" class="${mode === "beginner" ? "is-active" : ""}">新手模式</button>
          <button type="button" data-learning-mode="advanced" data-testid="mode-advanced" class="${mode === "advanced" ? "is-active" : ""}">进阶模式</button>
          <p>新手模式隐藏硬核笔记、LLM 接线图和源码；进阶模式完整展开。</p>
        </div>
        <div class="course-learning-actions" data-learning-actions data-testid="learning-actions">
          <button class="action" type="button" data-mark-understood="${escapeHtml(module.id)}" data-testid="mark-understood">${isUnderstood ? "已标记理解" : "我已理解"}</button>
          <button class="ghost-action" type="button" data-mark-review="${escapeHtml(module.id)}" data-testid="mark-review">${isReview ? "已加入复习" : "加入复习"}</button>
          <p data-learning-status data-testid="learning-status">${isUnderstood ? "这节已进入你的本地学习记录。" : "学完动画、实验和 3 分钟版讲义后，再点“我已理解”。"}</p>
        </div>
      </aside>
    </section>
  `;
  kickRouteMotion();
  applyLearningMode(mode);
  requestAnimationFrame(() => applyMotionReveal());
  app.focus({ preventScroll: true });
  wireConceptDemos(module);
  wireInteractiveLab(module);
  await loadLessonNotes(module);
  await loadSource(module);
  document.querySelector("[data-toggle-code]")?.addEventListener("click", (event) => {
    const code = document.querySelector(".code-window");
    code.classList.toggle("is-expanded");
    event.currentTarget.textContent = code.classList.contains("is-expanded") ? "收起代码" : "展开代码";
  });
  document.querySelectorAll("[data-course-scroll]").forEach((button) => {
    button.addEventListener("click", () => {
      scrollCourseTarget(button.dataset.courseScroll);
    });
  });
  wireLearningActions(module);
  wireCourseModeSwitcher();
  wireTagFilters();
}

function scrollCourseTarget(targetId) {
  const target = document.querySelector(`#${targetId}`);
  if (!target) return;
  const details = target.closest("details");
  if (details) details.open = true;
  scrollToHashTarget(`#${targetId}`, 0);
}

function wireCourseModeSwitcher() {
  document.querySelectorAll("[data-learning-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      const mode = writeLearningMode(button.dataset.learningMode);
      applyLearningMode(mode);
    });
  });
}

function wireTagFilters() {
  document.querySelectorAll("[data-tag-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      const tag = button.dataset.tagFilter || "";
      activePart = "all";
      activeGoal = tag === "CNN" || tag === "视觉" ? "计算机视觉" : activeGoal;
      renderHome();
      const input = document.querySelector("#catalog-search");
      if (input) input.value = tag;
      const event = new Event("input", { bubbles: true });
      input?.dispatchEvent(event);
      scrollToHashTarget("#notes", 90);
    });
  });
}

function wireLearningActions(module) {
  const status = document.querySelector("[data-learning-status]");
  document.querySelector("[data-mark-understood]")?.addEventListener("click", (event) => {
    const progress = markLearningProgress(module.id, "understood");
    event.currentTarget.textContent = "已标记理解";
    if (status) status.textContent = `已保存到本机：理解 ${new Set(progress.understood).size} 节。下一次从“复习与进度”继续。`;
  });
  document.querySelector("[data-mark-review]")?.addEventListener("click", (event) => {
    const progress = markLearningProgress(module.id, "review");
    event.currentTarget.textContent = "已加入复习";
    if (status) status.textContent = `已加入稍后复习：共 ${new Set(progress.review).size} 节。复习时先看动画和实验。`;
  });
}

async function loadSource(module) {
  const target = document.querySelector("[data-source]");
  const guide = document.querySelector("[data-source-guide]");
  try {
    const response = await fetch(module.sourcePath);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    target.textContent = text;
    if (guide) guide.innerHTML = renderTeachingSourceGuide(text, module);
  } catch (error) {
    target.textContent = `暂时无法读取 ${module.sourcePath}\n请确认通过本地 HTTP 服务打开站点。\n错误：${error.message}`;
    if (guide) {
      guide.innerHTML = `
        <div class="source-guide">
          <article>
            <span>读取失败</span>
            <strong>源码卡片暂时不可用</strong>
            <p>${escapeHtml(`请确认通过本地 HTTP 服务打开站点。错误：${error.message}`)}</p>
          </article>
        </div>
      `;
    }
  }
}

function renderDrawer(query = "") {
  const lower = query.trim().toLowerCase();
  const groups = PARTS.map((part) => {
    const modules = MODULES.filter((module) => module.partKey === part.key).filter((module) => {
      if (!lower) return true;
      return [module.title, module.summary, module.level, module.id, module.sourcePath, ...module.tags].join(" ").toLowerCase().includes(lower);
    });
    if (!modules.length) return "";
    return `
      <section class="drawer-group">
        <h3>${part.roman} ${escapeHtml(part.title)}</h3>
        ${modules.map((module) => `<a class="drawer-link" href="${moduleHref(module)}">${escapeHtml(module.title)}<small>${escapeHtml(module.id)}</small></a>`).join("")}
      </section>
    `;
  }).join("");
  document.querySelector("#drawer-list").innerHTML = `
    <section class="drawer-onboarding">
      <span>如果你是第一次来</span>
      <strong>先别搜索，按新手路径走</strong>
      <p>搜索适合已经知道关键词的人。零基础建议先完成数学、张量、神经网络、卷积、注意力这 5 步。</p>
      <a class="action" href="#starter">进入新手路径</a>
    </section>
    ${groups || `<div class="empty-state">没有找到匹配课程。可以先回到新手路径，或换一个更短的关键词。</div>`}
  `;
}

function openDrawer() {
  drawer.classList.add("is-open");
  scrim.classList.add("is-open");
  drawerSearch.focus();
}

function closeDrawer() {
  drawer.classList.remove("is-open");
  scrim.classList.remove("is-open");
}

function route() {
  const hash = decodeURIComponent(location.hash || "#home");
  if (hash.startsWith("#console/")) {
    renderCentralConsole(hash.replace("#console/", ""));
    closeDrawer();
    return;
  }
  if (hash.startsWith("#course/")) {
    renderCourse(hash.replace("#course/", ""));
    closeDrawer();
    return;
  }
  renderHome();
  if (hash !== "#home") {
    scrollToHashTarget(hash, 90);
  }
}

document.querySelectorAll("[data-open-menu]").forEach((button) => button.addEventListener("click", openDrawer));
document.querySelectorAll("[data-close-menu]").forEach((button) => button.addEventListener("click", closeDrawer));
drawerSearch.addEventListener("input", () => renderDrawer(drawerSearch.value));
drawer.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeDrawer();
});
window.addEventListener("hashchange", route);

renderDrawer();
applyLearningMode();
route();
