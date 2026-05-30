# 全站逐章内容校对矩阵

说明：

- **A 已校对**：已对照公开来源检查公式、定义、历史和 API 边界。
- **B 教学简化**：核心方向已挂公开来源，但图、数据或训练过程仍有教学化简化。
- **C 待复核**：已有学习结构和来源组，但还需要逐条校订。

当前策略：先让每页都有来源与边界，再逐章从 B/C 提升到 A。

## 第一部分：基础

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| 数学基础速查 | A | D2L / Deep Learning Book / MLCC / PyTorch Autograd | 已对照线性代数、导数、概率直觉与梯度下降边界；低维图形明确标注为教学演示。 |
| 张量与梯度 | A | PyTorch Autograd / D2L / Deep Learning Book / PyTorch NN | 已校对 autograd、计算图、链式法则和梯度传播图；补充梯度诊断边界。 |
| 激活与归一化 | B | PyTorch / BatchNorm / Dropout / LayerNorm | 区分 BatchNorm、LayerNorm、Dropout 的训练/推理边界。 |
| 数据集与优化器 | B | scikit-learn / Adam / MLCC | 校对数据划分、验证集、SGD/Adam 表述。 |
| 机器学习基础 | B | scikit-learn / MLCC | 校对监督学习、泛化、评估指标。 |
| 神经网络基础 | B | D2L / PyTorch | 校对反向传播和损失函数示例。 |
| 经典机器学习 | B | scikit-learn | 校对线性模型、树模型、SVM、聚类的适用边界。 |

## 第二部分：CNN

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| 卷积直觉 | A | CS231n / PyTorch Conv2d / D2L / LeNet | 已校对 padding、stride、kernel 与输出形状公式；补充固定滤波器和训练卷积核的边界。 |
| 特征图可视化 | B | CS231n / PyTorch | 强化“特征图不是因果解释”的边界说明。 |
| 经典 CNN 架构 | B | LeNet / AlexNet / VGG / ResNet | 校对架构年份、核心创新和简化实现差异。 |
| CNN 调试面板 | B | CS231n / PyTorch | 校对过拟合、数据增强、train/eval 模式排查。 |
| MNIST 玩具实验 | B | PyTorch / CS231n | 明确玩具实验与真实视觉任务差异。 |
| 现代 CNN 架构 | B | ResNet / MobileNet / EfficientNet | 校对残差、深度可分离卷积、模型缩放。 |
| 高级卷积技术 | B | CS231n / PyTorch | 校对扩张卷积、转置卷积、分组卷积。 |
| Grad-CAM 可视化 | B | Grad-CAM paper | 强化“热力图是线索，不是证明”。 |
| 迁移学习 | B | PyTorch / CS231n | 校对冻结、解冻、预处理分布和学习率策略。 |
| CNN 架构实验 | B | CS231n / CNN papers | 校对架构对比和参数量解释。 |
| 高级 CNN | B | CS231n / CNN papers | 校对 BN、Dropout、残差和现代卷积技巧。 |

## 第三部分：RNN

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| RNN 直觉 | A | D2L / PyTorch RNN / LSTM / GRU | 已校对隐藏状态、递推、梯度消失/爆炸和门控表述；补充低维可视化边界。 |
| 隐藏状态 | B | LSTM / GRU / PyTorch | 校对门控含义和长期依赖。 |
| 序列玩具任务 | B | D2L / PyTorch | 明确玩具任务与真实 NLP 任务差异。 |
| RNN 超参实验 | B | D2L / PyTorch | 校对 hidden size、截断 BPTT、dropout。 |
| Seq2Seq 与注意力 | B | Sutskever 2014 / Bahdanau 2015 | 校对 encoder-decoder、teacher forcing、对齐。 |
| 文本分类 | B | D2L / PyTorch | 校对 embedding、padding、截断、分类指标。 |
| 高级训练技巧 | B | D2L / PyTorch | 校对梯度裁剪、scheduled sampling 类表述。 |
| RNN 调试问题 | B | PyTorch / D2L | 校对 mask、采样退化、train/eval 排查。 |
| 序列模型 | B | LSTM / GRU / D2L | 校对 RNN/LSTM/GRU 对比。 |

## 第四部分：Transformer

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| 注意力机制 | A | Vaswani 2017 / Bahdanau 2015 / PyTorch SDPA / D2L | 已完成第一版来源尾注；后续可继续精修讲义连贯性。 |
| 多头注意力可视化 | A | Vaswani 2017 / PyTorch SDPA / PyTorch Transformer / D2L | 已校对 head 数、维度整除、拼接投影；补充 head 专门化不是必然解释的边界。 |
| 编码器与解码器 | B | Vaswani 2017 / D2L | 校对 causal mask、cross-attention、位置编码。 |
| 最小 Transformer | B | Vaswani 2017 / PyTorch Transformer | 校对最小实现、残差、LayerNorm、FFN。 |
| Flash Attention | B | Dao 2022 / PyTorch SDPA | 校对 exact attention、IO-aware、适用硬件边界。 |
| Transformer 调试 | B | PyTorch / D2L | 校对 mask、位置编码、loss 异常和数据泄漏。 |
| Transformer 架构 | B | Vaswani 2017 / PyTorch / D2L | 细化 BERT/GPT 对比来源，减少教学化热力图误解。 |
| GAN 与自编码器 | C | D2L / Deep Learning Book | 需要补 GAN/VAE/AE 原始来源。 |
| 图神经网络 | C | D2L / 公开论文待补 | 需要补 GCN/GAT/Message Passing 来源。 |

## 第五部分：工具箱

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| 特征可视化 | B | PyTorch / Grad-CAM | 校对降维图、激活图和解释性边界。 |
| 梯度监控 | B | PyTorch Autograd / D2L | 校对梯度范数、消失/爆炸阈值表述。 |
| 训练动态 | B | MLCC / D2L / PyTorch | 校对 loss、验证曲线、过拟合和欠拟合。 |
| 超参搜索 | B | scikit-learn / MLCC | 校对 grid/random search 与验证集使用。 |
| 玩具数据集 | B | scikit-learn / D2L | 明确玩具数据诊断价值和外推限制。 |
| 数据与训练 | B | PyTorch / scikit-learn | 校对 DataLoader、标准化、增强、指标。 |
| 案例研究 | B | PyTorch / scikit-learn | 标注合成数据和真实项目差异。 |
| 部署工具 | B | PyTorch ONNX / Quantization | 校对导出、量化、剪枝和推理边界。 |
| 练习题与测验 | C | 对应章节来源 | 需要逐题校对答案。 |
| 调参实战挑战 | B | MLCC / D2L / Adam | 明确模拟器规则不代表真实指标。 |

## 第六部分：统一框架与前沿

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| 统一接口 | B | PyTorch / 本站工程实践 | 校对接口边界，避免包装成通用标准。 |
| 模块化结构 | B | PyTorch / 本站工程实践 | 校对 config、dataset、model、trainer 职责。 |
| 完整项目骨架 | B | PyTorch / 本站工程实践 | 补实验可复现和产物管理来源。 |
| 插件系统 | B | 本站工程实践 | 标注这是教学实现，不是成熟插件框架。 |
| 一键训练 | B | PyTorch / MLCC | 校对 checkpoint、early stopping、指标保存。 |
| 可视化实验台 | B | 本站工程实践 | 继续补交互控件和图表说明。 |
| 神经网络乐高工厂 | B | PyTorch / 本站工程实践 | 强化 shape 诊断和代码生成测试。 |
| 训练过程可视化 | B | PyTorch / MLCC | 区分真实训练和教学模拟。 |
| 项目模板 | B | PyTorch / 本站工程实践 | 校对 K-Fold、日志、配置模板。 |
| 强化学习入门 | C | 待补 RL 来源 | 需要补 Sutton & Barto 或官方 RL 教材来源。 |
| 学习路径推荐 | B | 本站工程实践 | 标注推荐逻辑为启发式。 |
| 深度学习术语表 | C | 对应章节来源 | 逐条术语需要来源回链。 |
| 前沿方向 | C | 论文/官方资料待补 | LLM、Agent、安全内容需更严格来源。 |
| 经典论文解读实验室 | B | 原论文 | 每篇论文需要独立来源卡。 |

## 第七部分：CS 基础训练营

| 页面 | 状态 | 来源组 | 下一步 |
|---|---|---|---|
| 计算机网络 | C | MDN / RFC 待补 | 校对 TCP、HTTP、DNS、TLS 细节。 |
| 数据库 SQL | C | PostgreSQL / 官方文档 | 校对事务、索引、查询计划。 |
| 数据结构与算法 | C | cp-algorithms / 教材待补 | 校对复杂度、边界条件和动画。 |
| 操作系统 | C | OSTEP | 校对进程、线程、调度、虚拟内存。 |
| 系统设计 | C | 官方文档/公开资料待补 | 按约束和取舍重写，不给唯一答案。 |
| 深度学习自测 | B | 对应深度学习章节来源 | 逐题校对答案和追问。 |
| 自测刷题模式 | C | 对应题库来源 | 建立错题来源和答案校对流程。 |
