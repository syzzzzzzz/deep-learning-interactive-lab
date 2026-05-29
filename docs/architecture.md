# 架构说明

本项目是一个面向零基础学习者的深度学习交互式学习网站。当前主体验已经从 Streamlit 页面迁移到原生 HTML/CSS/JavaScript 静态站，Python 脚本保留为课程源码、旧实验运行器和质量检查入口。

## 总体分层

```text
index.html
assets/
  site.js              # 静态站路由、课程页、实验、中央控制台
  site.css             # 统一视觉系统与响应式样式
components/
  course_manifest.py   # 课程目录唯一事实源
  knowledge_graph.py   # 知识图谱与相关章节推荐
  legacy_runtime.py    # 旧脚本统一运行协议
  artifact_runtime.py  # 运行产物生命周期
  visual_*.py          # Streamlit legacy 视觉系统分层
part*_*/               # 课程脚本和源码对照
scripts/
  quality_check.py     # Python 内容质量门总入口
  quality_checks/      # 按领域拆分的质量门
tests/e2e/             # Playwright 真实浏览器验收
```

## 静态站壳层

静态站由 `index.html` 加载 `assets/site.css` 和 `assets/site.js`。页面通过 hash 路由切换，例如：

- `#home`
- `#course/part4%2Ftransformer_models`
- `#console/part4%2Ftransformer_models`

这种结构避免了旧 Streamlit 页面每次交互都重新运行 Python 的问题，课程页的动画、实验、源码对照和中央控制台都可以在浏览器中即时更新。

## 课程目录 Source of Truth

课程事实集中在 `components/course_manifest.py`。主站、知识图谱、质量门都从这里派生课程清单，避免新增章节时同时修改多份路由表。

关键收益：

- 新增章节只需要维护一份课程事实。
- 知识图谱节点顺序和主站目录保持一致。
- 质量门可以检查课程目录、知识图谱和路由是否偏离。

## 旧脚本运行协议

旧教学脚本已经统一成 `compute / render / smoke` 协议，并由 `components/legacy_runtime.py` 提供统一运行能力。`components/lesson_runtime.py` 只保留兼容导出，避免历史脚本全部改 import。

运行产物通过 `components/artifact_runtime.py` 管理：

- 创建 run 目录
- 写入 stdout / stderr / status
- 收集图片
- 生成产物解释上下文
- 防止根目录被 `*.png`、`*.csv`、`*.pt`、`*.log` 污染

## 中央控制台

中央控制台是项目的作品级亮点。它包含三层能力：

1. 统一实验台：从课程页跳转后，可以把当前知识点的参数迁移到实战环境。
2. 模型构建器：支持 MLP、CNN、Transformer mini、shape error demo 等预设。
3. 训练事件总线：一次调参事件同时更新 loss、梯度、特征/注意力和实验笔记。

当前模型构建器支持：

- Embedding
- MultiHeadAttention
- LayerNorm
- ResidualBlock
- TransformerEncoder
- Flatten
- Dropout
- Linear
- Shape 诊断
- JSON 保存 / 加载
- PyTorch 代码导出

## 视觉系统

静态站的视觉系统在 `assets/site.css` 中实现；Streamlit legacy 页面通过 `components/visual_system.py` 门面重导出，内部拆分为：

- `visual_tokens.py`
- `visual_runtime.py`
- `visual_primitives.py`
- `visual_effects.py`
- `visual_gallery.py`

这样既保留统一风格，又避免所有视觉逻辑继续堆在单个巨型文件里。

## 测试与质量门

项目有两类质量保障：

- Python 质量门：`python -X utf8 scripts/quality_check.py`
- 浏览器 E2E：`npm run test:e2e`

Python 质量门检查：

- 课程目录和知识图谱一致性
- 旧脚本协议
- 页面重点内容覆盖
- 静态站入口、路由、CSS、JS 约束
- 运行产物污染
- 38 个旧脚本 smoke
- 6 个重点页面渲染 smoke

Playwright E2E 检查：

- 课程页学习闭环
- 新手/进阶模式
- 中央控制台模型构建器
- Transformer mini 预设
- shape mismatch 诊断
- PyTorch 代码导出
- JSON 保存与恢复

## 部署

`.github/workflows/pages.yml` 使用 GitHub Actions 部署静态站到 GitHub Pages。构建前会先跑 Python 质量门和 Playwright E2E，只有全部通过才会发布。

发布产物只包含静态站需要的文件：

- `index.html`
- `assets/`
- `deep_learning_book/`
- `docs/`
- `part1_foundations/` 到 `part7_interview/`
- `README.md`
