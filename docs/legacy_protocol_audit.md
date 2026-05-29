# 旧教材脚本 render/compute 协议审计

本页专门收口 GitHub #10 中“约 37 个老脚本 render/compute 分离”的模糊口径。当前仓库实际导入的旧教材来源是 `docs/legacy_book/manifest.json`，共 38 个 Markdown 章节。审计口径以这 38 个旧教材章节对应的 Python 脚本为准，不把后续新增的静态站页面、面试页、个人名片区或 HTML 原生实验区混进这个数字。

```text
legacy_lessons: 38
strict_protocolized: 38
deferred_legacy: 0
```

## 验收口径

严格协议化表示脚本已经暴露稳定的 `MODULE_TITLE`、`MODULE_SUMMARY`、`MODULE_TAGS`、`MODULE_RELATED_TOPICS`、`PRACTICE_TARGET`、`compute*()`、`render()`、`smoke()`，并被 `scripts/quality_check.py` 的严格协议检查与严格 smoke 纳入质量门。

待拆分表示该脚本仍属于旧教材迁移范围，但还没有形成完整稳定的 render/compute 合约，或仍存在顶层执行演示、顶层 Streamlit 页面逻辑、缺少协议元数据等问题。当前待拆分清单为空。

## 已严格协议化

| 文件 | 旧教材章节 | 状态 | 验收说明 |
| --- | --- | --- | --- |
| `part1_foundations/01_tensors_gradients.py` | 第一章：张量、梯度、反向传播直观理解 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part1_foundations/02_activations_normalization.py` | 第二章：激活函数、初始化、归一化、过拟合 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part1_foundations/03_datasets_optimizers.py` | 第三章：数据集、批次、优化器、损失函数直观实验 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/01_convolution_visual.py` | 第一章：卷积、池化、填充、步幅——逐步数字计算与可视化 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/02_feature_maps.py` | 第五章：特征图实时查看与滤波器调试 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/03_classic_architectures.py` | 第六章：经典 CNN 架构——LeNet / AlexNet / VGG / ResNet | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/04_debug_panel.py` | 第七章：可交互调试面板——改卷积核、看每层输出 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/05_mnist_toy.py` | 第五章：经典 CNN 网络 + 可交互调试面板 + 手写数字玩具 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/06_modern_architectures.py` | 现代 CNN 架构：MobileNet / DenseNet / EfficientNet | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/07_advanced_convolution.py` | 高级卷积技术：空洞卷积 / 转置卷积 / 分组卷积 / 可变形卷积 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/08_visualization_gradcam.py` | CNN 可视化技术：GradCAM / 显著图 / 特征反转 / DeepDream | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part2_cnn/09_transfer_learning.py` | 迁移学习：预训练模型 / 微调策略 / 渐进式训练 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/01_rnn_intuition.py` | 第八章：循环结构、梯度消失与门机制直观解释 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/02_hidden_states.py` | 第九章：隐藏状态变化与序列预测可视化 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/03_sequence_toys.py` | 第六章：RNN / LSTM / GRU 序列模型完整玩具 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/04_hyperparam_rnn.py` | 第十章：RNN 超参数调试实验台 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/05_seq2seq_attention.py` | Seq2Seq 与注意力机制：从编码器-解码器到 Bahdanau/Luong 注意力 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/06_text_classification.py` | RNN 文本分类：情感分析完整项目 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/07_advanced_training.py` | RNN 高级训练技术：Teacher Forcing / 预定采样 / BPTT / 梯度策略 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part3_rnn/08_debug_problems.py` | RNN/LSTM 调试问题集：15个常见错误与解决方案 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part4_transformer/01_attention_mechanism.py` | 第一章：自注意力机制——完整数学推导与逐行张量分析 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part4_transformer/02_multihead_visual.py` | 第二章：Multi-Head Attention 完整实现与权重可视化 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part4_transformer/03_encoder_decoder.py` | 第三章：位置编码——数学原理、完整代码与可视化 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part4_transformer/04_minimal_transformer.py` | 第八章：极简可运行 Transformer 实现 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part4_transformer/05_flash_attention.py` | FlashAttention：高效注意力机制 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part4_transformer/06_debug_problems.py` | Transformer 调试问题集：15个常见错误与解决方案 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part5_toolbox/01_feature_visualization.py` | 第九章补充：特征可视化工具 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part5_toolbox/02_gradient_monitor.py` | 第九章：调试工具箱 — 梯度监控 + 训练动态 + 超参搜索 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part5_toolbox/03_training_dynamics.py` | 第九章补充：训练动态深度分析 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part5_toolbox/04_hyperparam_search.py` | 第九章补充：超参搜索进阶 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part5_toolbox/05_dataset_toys.py` | 第十三章：数据集玩具——自己造数据、看模型表现 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/01_unified_interface.py` | 第十章：万能训练框架 — 统一接口设计 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/02_modular_structure.py` | 第十四章：模块化结构——切换模型只需改一行 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/03_full_project.py` | 第十章：万能可视化框架 — 一个项目玩遍所有模型 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/04_plugin_system.py` | 可扩展插件系统：添加新模型、新数据集、新任务 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/05_one_click_training.py` | 一键训练与评估：自动保存、日志、曲线绘制 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/06_streamlit_demo.py` | Streamlit 交互式演示：一键搭建模型演示界面 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |
| `part6_universal_framework/07_project_template.py` | 项目模板：比赛 / 作业 / 个人项目完整结构 | 严格协议化 | 纳入 `STRICT_LEGACY_PROTOCOL_FILES`，严格 smoke 覆盖 |

## 待拆分旧脚本

当前无待拆分旧脚本。

## 维护约束

`scripts/quality_check.py` 会检查：

- 38 个旧教材章节必须全部落入“严格协议化”或“待拆分旧脚本”之一。
- 审计文档里的三行计数必须与代码名单一致。
- 每个严格协议化文件必须存在 `compute*()`、`render()`、`smoke()`，并通过严格 smoke。

因此当前结论是：老教材脚本已完成 38/38 的严格 render/compute 分离，剩余 0/38。后续如果新增旧教材章节，必须同步更新 manifest、严格名单和本审计页。
