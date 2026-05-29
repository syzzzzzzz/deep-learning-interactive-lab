# 深度学习完全手册：从零到研究者

> 一本可以玩一辈子的深度学习教科书 + 万能可视化玩具项目

---

## 目录结构

```
deep_learning_book/
├── README.md                          ← 本文件
│
├── part1_foundations/                 ← 第一部分：基础与数学直觉
│   ├── 01_tensors_gradients.md        ← 张量、梯度、反向传播
│   ├── 02_activations_normalization.md← 激活函数、初始化、归一化
│   └── 03_datasets_optimizers.md      ← 数据集、优化器、损失函数
│
├── part2_cnn/                         ← 第二部分：CNN 可视化玩具
│   ├── 01_convolution_visual.md       ← 卷积、池化、感受野可视化
│   ├── 02_feature_maps.md             ← 特征图实时查看
│   ├── 03_classic_networks.md         ← LeNet/AlexNet/VGG/ResNet
│   ├── 04_debug_panel.md              ← 可交互调试面板
│   └── 05_mnist_toy.md                ← 手写数字分类玩具
│
├── part3_rnn/                         ← 第三部分：RNN/LSTM/GRU
│   ├── 01_rnn_intuition.md            ← 循环结构、梯度消失
│   ├── 02_hidden_state_visual.md      ← 隐藏状态可视化
│   └── 03_sequence_toys.md            ← 名字生成、文本预测
│
├── part4_transformer/                 ← 第四部分：Attention & Transformer
│   ├── 01_attention_mechanism.md      ← Scaled Dot-Product Attention
│   ├── 02_multihead_visual.md         ← Multi-Head 权重可视化
│   ├── 03_positional_encoding.md      ← 位置编码直观理解
│   └── 04_minimal_transformer.md      ← 极简可运行 Transformer
│
├── part5_toolbox/                     ← 第五部分：调试工具箱
│   ├── 01_feature_visualization.md    ← 特征可视化工具
│   ├── 02_gradient_monitor.md         ← 梯度监控工具
│   ├── 03_training_dynamics.md        ← 训练动态绘图
│   └── 04_hyperparam_search.md        ← 超参搜索框架
│
└── part6_universal_framework/         ← 第六部分：万能框架
    ├── 01_unified_interface.md        ← 统一模型接口
    ├── 02_modular_design.md           ← 模块化设计
    └── 03_full_project.md             ← 完整可运行项目
```

## 快速开始

```bash
pip install torch torchvision matplotlib numpy seaborn ipywidgets jupyter
```

## 学习路径

| 阶段 | 目标 | 章节 |
|------|------|------|
| 入门 | 理解基本概念 | Part1 全部 |
| 进阶 | 掌握 CNN/RNN | Part2 + Part3 |
| 核心 | 理解 Transformer | Part4 全部 |
| 工程 | 调试与优化 | Part5 + Part6 |
