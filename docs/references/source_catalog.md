# 公开来源清单

这个项目的讲义会优先参考合法公开来源。这里记录“可以用来校对概念和公式”的来源，而不是把外部内容搬进仓库。

## 使用原则

- 优先使用一手论文、官方文档、开放教材和大学公开课程笔记。
- 不复制受版权保护书籍或非授权资源站正文。
- 章节文字必须用自己的话重写；来源只用于校对定义、公式、历史脉络和工程边界。
- 如果某段内容只是教学模拟，需要明确写出“简化演示，不等价于真实训练”。
- 如果一个工程经验没有明确来源，需要写成“通常可以先尝试”而不是“工业界一定如此”。

## 核心公开来源

| 编号 | 来源 | 类型 | 适合校对的内容 |
|---|---|---|---|
| DLBOOK | https://www.deeplearningbook.org/ | 开放教材 | 深度学习基础、优化、正则化、CNN、RNN |
| D2L | https://d2l.ai/ | 开放教材 | Dive into Deep Learning；张量、优化、CNN、RNN、注意力、Transformer、代码实践 |
| CS231N | https://cs231n.github.io/ | 公开课程笔记 | CNN、优化、反向传播、可视化、训练技巧 |
| CS224N | https://web.stanford.edu/class/cs224n/ | 公开课程资料 | NLP、RNN、注意力、Transformer |
| PYTORCH | https://pytorch.org/docs/stable/ | 官方文档 | PyTorch API、autograd、nn.Module、`torch.nn.functional.scaled_dot_product_attention`、Transformer |
| PYTORCH_TENSOR | https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html | 官方教程 | 张量 shape、dtype、device、基础运算 |
| PYTORCH_CONV2D | https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html | 官方文档 | Conv2d 输入输出形状、stride、padding、dilation、groups |
| PYTORCH_RNN | https://pytorch.org/docs/stable/nn.html#recurrent-layers | 官方文档 | RNN、LSTM、GRU 输入输出形状、层参数、batch 维度 |
| PYTORCH_SDPA | https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | 官方文档 | scaled dot-product attention、mask、dropout、scale、GQA |
| PYTORCH_TRANSFORMER | https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html | 官方文档 | Transformer 模块参数、nhead、batch_first、mask 接口 |
| SKLEARN | https://scikit-learn.org/stable/user_guide.html | 官方文档 | 经典机器学习、评估、数据预处理、模型选择 |
| MLCC | https://developers.google.com/machine-learning/crash-course | 官方开放课程 | 机器学习直觉、梯度下降、泛化、数据拆分 |
| MDN_HTTP | https://developer.mozilla.org/en-US/docs/Web/HTTP | 官方文档 | HTTP、缓存、状态码、Web 协议基础 |
| POSTGRES | https://www.postgresql.org/docs/ | 官方文档 | PostgreSQL Documentation；SQL、事务、索引、查询执行 |
| OSTEP | https://pages.cs.wisc.edu/~remzi/OSTEP/ | 开放教材 | Operating Systems: Three Easy Pieces；操作系统、进程、线程、虚拟内存、并发 |
| CP_ALGORITHMS | https://cp-algorithms.com/ | 开放算法资料 | 常见算法与数据结构复习 |

## 论文来源

| 编号 | 论文 | 链接 | 适合校对的内容 |
|---|---|---|---|
| BAH2015 | Bahdanau, Cho, Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. | https://arxiv.org/abs/1409.0473 | 早期可微注意力、seq2seq 对齐 |
| VAS2017 | Vaswani et al. Attention Is All You Need. | https://arxiv.org/abs/1706.03762 | Scaled dot-product attention、多头注意力、Transformer |
| DAO2022 | Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. | https://arxiv.org/abs/2205.14135 | FlashAttention、IO-aware attention、长序列效率 |
| BA2016 | Ba, Kiros, Hinton. Layer Normalization. | https://arxiv.org/abs/1607.06450 | LayerNorm、序列模型归一化 |
| BN2015 | Ioffe, Szegedy. Batch Normalization. | https://arxiv.org/abs/1502.03167 | BatchNorm、训练稳定性 |
| DROPOUT2014 | Srivastava et al. Dropout. | https://www.jmlr.org/papers/v15/srivastava14a.html | Dropout、正则化 |
| ADAM2014 | Kingma, Ba. Adam. | https://arxiv.org/abs/1412.6980 | Adam 优化器 |
| LENET1998 | LeCun et al. Gradient-Based Learning Applied to Document Recognition. | http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf | 早期 CNN、LeNet |
| ALEX2012 | Krizhevsky, Sutskever, Hinton. ImageNet Classification with Deep Convolutional Neural Networks. | https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks | AlexNet、ImageNet 分类 |
| VGG2014 | Simonyan, Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. | https://arxiv.org/abs/1409.1556 | VGG、深层 3x3 卷积 |
| HE2015 | He et al. Deep Residual Learning for Image Recognition. | https://arxiv.org/abs/1512.03385 | ResNet、残差连接 |
| MOBILENET2017 | Howard et al. MobileNets. | https://arxiv.org/abs/1704.04861 | 深度可分离卷积、移动端 CNN |
| EFFICIENTNET2019 | Tan, Le. EfficientNet. | https://arxiv.org/abs/1905.11946 | CNN 模型缩放 |
| SELVARAJU2016 | Selvaraju et al. Grad-CAM. | https://arxiv.org/abs/1610.02391 | Grad-CAM、可解释性边界 |
| LSTM1997 | Hochreiter, Schmidhuber. Long Short-Term Memory. | https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory | LSTM、长期依赖 |
| CHO2014 | Cho et al. Learning Phrase Representations using RNN Encoder-Decoder. | https://arxiv.org/abs/1406.1078 | GRU、Encoder-Decoder |
| SUTS2014 | Sutskever, Vinyals, Le. Sequence to Sequence Learning with Neural Networks. | https://arxiv.org/abs/1409.3215 | Seq2Seq、机器翻译 |

## 章节优先级

第一批先校对：

1. `part4_transformer/01_attention_mechanism`
2. `part2_cnn/01_convolution_visual`
3. `part1_foundations/math_primer`
4. `part1_foundations/01_tensors_gradients`
5. `part3_rnn/01_rnn_intuition`

每个章节校对后需要补：

- 可信度标签：已校对 / 教学简化 / 待复核
- 来源尾注：至少 2 个相关公开来源
- 边界说明：哪些图是模拟，哪些结论依赖具体条件
- 强断言清理：少用“必然、一定、90%、工业界都”

## 全站第一轮覆盖策略

第一轮不把所有页面都标成“已校对”。更诚实的做法是：

- `part1_foundations`：默认 B，来源组为 D2L、Deep Learning Book、PyTorch、scikit-learn、Google MLCC。
- `part2_cnn`：默认 B，来源组为 CS231n、PyTorch、LeNet/AlexNet/VGG/ResNet/Grad-CAM 等论文。
- `part3_rnn`：默认 B，来源组为 D2L、PyTorch RNN 文档、LSTM/GRU/Seq2Seq/Attention 论文。
- `part4_transformer`：默认 B，来源组为 Attention/Transformer/FlashAttention 论文和 PyTorch 文档；`01_attention_mechanism` 已先标 A。
- `part5_toolbox`：默认 B，来源组为 PyTorch、scikit-learn、Grad-CAM、Adam、部署文档。
- `part6_universal_framework`：默认 B，来源组为 PyTorch、D2L、MLCC 和本站工程实践。
- `part7_interview`：默认 C，来源组为 MDN、PostgreSQL、OSTEP、cp-algorithms 等；后续需要逐题校对。
