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
| SKLEARN | https://scikit-learn.org/stable/user_guide.html | 官方文档 | 经典机器学习、评估、数据预处理、模型选择 |
| MLCC | https://developers.google.com/machine-learning/crash-course | 官方开放课程 | 机器学习直觉、梯度下降、泛化、数据拆分 |

## 论文来源

| 编号 | 论文 | 链接 | 适合校对的内容 |
|---|---|---|---|
| BAH2015 | Bahdanau, Cho, Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. | https://arxiv.org/abs/1409.0473 | 早期可微注意力、seq2seq 对齐 |
| VAS2017 | Vaswani et al. Attention Is All You Need. | https://arxiv.org/abs/1706.03762 | Scaled dot-product attention、多头注意力、Transformer |
| DAO2022 | Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. | https://arxiv.org/abs/2205.14135 | FlashAttention、IO-aware attention、长序列效率 |
| BA2016 | Ba, Kiros, Hinton. Layer Normalization. | https://arxiv.org/abs/1607.06450 | LayerNorm、序列模型归一化 |
| HE2015 | He et al. Deep Residual Learning for Image Recognition. | https://arxiv.org/abs/1512.03385 | ResNet、残差连接 |
| SELVARAJU2016 | Selvaraju et al. Grad-CAM. | https://arxiv.org/abs/1610.02391 | Grad-CAM、可解释性边界 |

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
