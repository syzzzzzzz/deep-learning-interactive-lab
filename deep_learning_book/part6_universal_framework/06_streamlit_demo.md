# Streamlit 交互式演示：一键搭建模型演示界面

## 1. 为什么用 Streamlit？

```
训练完模型后，需要：
- 展示给老师/同事看
- 快速测试不同输入的效果
- 让非技术人员也能用

Streamlit 优势：
- 纯 Python，无需前端知识
- 几十行代码搭建完整界面
- 自动响应式布局
- 支持图表、图像、交互控件
```

---

## 2. 通用模型演示界面

```python
# app.py — 运行方式: streamlit run app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# ─────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="深度学习模型演示",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 深度学习万能演示平台")
st.markdown("选择模型 → 输入数据 → 查看结果")

# ─────────────────────────────────────────────────────────
# 模型定义（内嵌简化版，也可从文件加载）
# ─────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(h[-1])


# ─────────────────────────────────────────────────────────
# 侧边栏：模型与参数选择
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙ 配置")

    # 模型选择
    model_type = st.selectbox(
        "选择模型",
        ["CNN 图像分类", "LSTM 文本分类", "MLP 回归"],
        index=0,
    )

    # 模型加载方式
    load_mode = st.radio(
        "模型来源",
        ["随机初始化（演示用）", "加载已训练权重"],
        index=0,
    )

    checkpoint_path = None
    if load_mode == "加载已训练权重":
        checkpoint_path = st.text_input("权重文件路径", value="./experiments/best.pt")

    # 推理参数
    st.subheader("推理参数")
    temperature = st.slider("温度", 0.1, 2.0, 1.0, 0.1)
    top_k = st.slider("Top-K", 0, 100, 0, 5)

# ─────────────────────────────────────────────────────────
# 主区域：根据模型类型展示不同界面
# ─────────────────────────────────────────────────────────

if model_type == "CNN 图像分类":
    st.header("📸 手写数字识别")

    # 绘图画板
    st.subheader("在下方画板上写一个数字")
    canvas_result = st.canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
    )

    if canvas_result.image_data is not None:
        # 预处理：灰度化 → 缩放到 28×28 → 归一化
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')  # 灰度
        img = img.resize((28, 28))

        # 转张量
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        # 展示预处理后的图像
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="预处理后 (28×28)", width=200)

        # 模型推理
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN(in_channels=1, num_classes=10).to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

        model.eval()
        with torch.no_grad():
            logits = model(img_tensor.to(device))
            probs = F.softmax(logits / temperature, dim=-1)[0].cpu().numpy()

        with col2:
            # 预测结果柱状图
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['red' if i == probs.argmax() else 'steelblue' for i in range(10)]
            ax.bar(range(10), probs, color=colors)
            ax.set_xlabel('数字')
            ax.set_ylabel('概率')
            ax.set_title(f'预测结果: {probs.argmax()} (置信度 {probs.max():.1%})')
            ax.set_xticks(range(10))
            st.pyplot(fig)

        # 所有类别概率
        st.subheader("各类别概率")
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                bar_width = int(probs[i] * 100)
                st.metric(f"数字 {i}", f"{probs[i]:.2%}")


elif model_type == "LSTM 文本分类":
    st.header("📝 情感分析")

    # 文本输入
    text_input = st.text_area(
        "输入文本（中文）",
        value="这部电影太精彩了",
        height=100,
    )

    # 简化分词
    words = list(text_input)

    # 简单词汇映射
    word2idx = {c: i + 4 for i, c in enumerate(set(words))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1

    indices = [word2idx.get(c, 1) for c in words]
    input_tensor = torch.tensor([indices])

    # 模型推理
    model = LSTMClassifier(vocab_size=max(100, len(word2idx)), num_classes=2)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1)[0].numpy()

    # 展示结果
    col1, col2 = st.columns(2)

    with col1:
        sentiment = "正面 😊" if probs[1] > probs[0] else "负面 😞"
        confidence = max(probs)
        st.metric("情感", sentiment)
        st.metric("置信度", f"{confidence:.1%}")

    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(['负面', '正面'], probs, color=['#C44E52', '#55A868'])
        ax.set_xlim(0, 1)
        ax.set_xlabel('概率')
        st.pyplot(fig)

    # 注意力/关键词高亮（示意）
    st.subheader("关键词贡献")
    for i, word in enumerate(words):
        # 简化：用输入梯度模拟贡献度
        contribution = np.random.uniform(0.1, 1.0)  # 演示用
        opacity = min(1.0, contribution)
        st.markdown(
            f'<span style="background-color: rgba(85,168,104,{opacity}); '
            f'padding: 2px 6px; border-radius: 3px; margin: 2px">{word}</span>',
            unsafe_allow_html=True,
        )


elif model_type == "MLP 回归":
    st.header("📈 函数拟合")

    # 函数选择
    func_type = st.selectbox("选择函数", ["sin(x)", "x²", "sinc(x)"])

    # 噪声水平
    noise_level = st.slider("噪声水平", 0.0, 0.5, 0.1, 0.05)

    # 生成数据
    x = np.linspace(-5, 5, 200)
    if func_type == "sin(x)":
        y_true = np.sin(x)
    elif func_type == "x²":
        y_true = x ** 2
    else:
        y_true = np.sinc(x)

    y_noisy = y_true + noise_level * np.random.randn(len(x))

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true, 'b-', linewidth=2, label='真实函数')
    ax.scatter(x, y_noisy, c='gray', s=5, alpha=0.3, label='带噪声数据')
    ax.set_title(f'函数拟合: {func_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────
# 底部：模型信息
# ─────────────────────────────────────────────────────────

st.divider()
with st.expander("📊 模型详情"):
    if model_type == "CNN 图像分类":
        model = SimpleCNN()
    elif model_type == "LSTM 文本分类":
        model = LSTMClassifier()
    else:
        model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))

    st.code(str(model))
    st.metric("参数量", f"{sum(p.numel() for p in model.parameters()):,}")
```

---

## 3. 运行方式

```bash
# 安装依赖
pip install streamlit

# 启动演示
streamlit run app.py

# 指定端口
streamlit run app.py --server.port 8502

# 允许外网访问
streamlit run app.py --server.address 0.0.0.0
```

---

## 4. 多页面布局（大型项目）

```python
# 项目结构：
# streamlit_app/
# ├── app.py              ← 主入口
# ├── pages/
# │   ├── 01_图像分类.py
# │   ├── 02_文本分类.py
# │   └── 03_序列预测.py
# ├── models/
# │   ├── __init__.py
# │   └── cnn.py
# └── utils/
#     └── visualization.py

# app.py（主页）
import streamlit as st

st.set_page_config(page_title="深度学习平台", layout="wide")
st.title("🧠 深度学习万能平台")
st.markdown("左侧选择任务 → 进入对应页面")
st.info("选择左侧导航栏中的页面开始体验")

# pages/01_图像分类.py（子页面，自动出现在导航栏）
import streamlit as st
st.title("📸 图像分类")
# ... 图像分类的完整代码
```

---

## 小结

| 组件 | 功能 | 代码量 |
|------|------|--------|
| 侧边栏 | 模型选择、参数调节 | ~20行 |
| 绘图画板 | 手写输入 | 1行 |
| 文本输入 | 文本分类 | 1行 |
| 概率柱状图 | 置信度可视化 | ~10行 |
| 多页面 | 大型项目组织 | 目录结构 |
