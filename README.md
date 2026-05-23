# 深度学习交互式学习网站

这是一个面向初学者的深度学习交互式学习网站。项目使用 Streamlit、PyTorch、Matplotlib、Plotly 和 scikit-learn 构建，把深度学习中的数学基础、经典机器学习、卷积神经网络、循环神经网络、Transformer 和工程工具箱做成可视化、可调参、可逐步观察的网页实验室。

## 项目特色

- **中文教材式讲解**：重点章节补充了从直觉、定义、数学原理、动手实验、常见误区到工程应用的完整说明。
- **交互式可视化**：通过滑块、选择框、热力图、损失曲线、分类边界和动画观察模型行为。
- **适合零基础学习**：每个图都配有读图提示，告诉你应该调什么参数、观察什么变化、思考什么问题。
- **覆盖完整学习路径**：从张量、梯度、数学基础开始，逐步进入 CNN、RNN、Transformer 和项目化训练流程。

## 章节目录

- `part1_foundations`：张量、梯度、数学基础、经典机器学习、神经网络入门
- `part2_cnn`：卷积可视化、特征图、经典 CNN、MNIST、小型调试面板、迁移学习
- `part3_rnn`：RNN 直觉、隐藏状态、序列任务、注意力与文本分类
- `part4_transformer`：注意力机制、多头注意力、位置编码、Transformer、BERT 与 GPT 对比
- `part5_toolbox`：特征可视化、梯度监控、训练动态、超参数搜索、数据集玩具实验
- `part6_universal_framework`：统一接口、模块化结构、插件系统、一键训练、Streamlit 演示

## 环境要求

建议使用 Python 3.10 或更高版本。当前开发环境使用 Python 3.12。

## 安装依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

如果你的环境中同时存在多个 Python 版本，请使用实际的 Python 路径执行安装，例如：

```powershell
C:\Users\你的用户名\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

## 启动网站

推荐使用主入口启动：

```powershell
python -m streamlit run main.py --server.address 127.0.0.1 --server.port 8501
```

启动成功后，在浏览器打开：

```text
http://127.0.0.1:8501
```

也可以直接双击 Windows 批处理文件：

```text
start_lab.bat
```

## 打开指定章节

主站支持通过 URL 参数直接进入章节，例如：

```text
http://127.0.0.1:8501/?module=part1_foundations%2Fmath_primer
http://127.0.0.1:8501/?module=part1_foundations%2Fclassical_ml
http://127.0.0.1:8501/?module=part4_transformer%2Ftransformer_models
```

## 命令行运行器

如果只想查看模块列表，可以执行：

```powershell
python main.py
```

运行单个模块：

```powershell
python main.py part2_cnn/01_convolution_visual
```

## 学习建议

1. 先看 `数学基础速查`，理解向量、矩阵、导数、概率和梯度下降。
2. 再看 `经典机器学习`，建立决策边界、过拟合、正则化和距离度量的直觉。
3. 然后进入 CNN、RNN、Transformer，观察深度学习模型如何把这些基础概念组合成更强的结构。
4. 每个页面都建议先调极端参数，再回到默认值，这样最容易看出参数真正控制了什么。

## 维护备注

- 本项目会生成运行时图片、日志和 Streamlit 临时输出，这些文件已通过 `.gitignore` 排除。
- 根目录下的 `.png`、`.log`、`.pt` 通常是教学脚本运行产物，不建议提交到仓库。
- 如果新增章节，请优先保持“图文绑定、可调参数、误区解释、工程意义”的教学风格。

## 内容质量检查

提交前建议运行：

```powershell
python scripts/quality_check.py
```

这条命令会检查：

- Python 文件能否编译，并把 `SyntaxWarning` 当作失败处理。
- 教材内容是否残留模板占位符。
- 首批重点页面中，互动文案引用的控件是否真实存在。
- 主站重点路由是否已注册。
- 注意力机制、经典机器学习、数学基础速查的关键渲染分支是否能跑通。

如果只想快速做静态检查，可以执行：

```powershell
python scripts/quality_check.py --skip-smoke
```
