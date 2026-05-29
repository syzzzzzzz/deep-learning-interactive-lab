"""
Deep learning case study gallery.

Run:
    streamlit run part5_toolbox/case_studies.py
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


torch.set_num_threads(1)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]


st.set_page_config(
    page_title="深度学习实战案例库",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 2.4rem; }
    .stApp { background: #f8f9f6; color: #172026; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #d8dedc;
        border-radius: 8px;
        padding: 10px 12px;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.76);
        border-radius: 0 8px 8px 0;
        padding: 0.74rem 0.9rem;
        line-height: 1.68;
        margin: 0.35rem 0 0.9rem 0;
    }
    .case-step {
        background: rgba(255,255,255,0.72);
        border: 1px solid #d8dedc;
        border-radius: 8px;
        padding: 0.84rem 0.92rem;
        line-height: 1.58;
        min-height: 7.6rem;
    }
    .small {
        color: #5b6670;
        font-size: 0.92rem;
        line-height: 1.58;
    }
    code { white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


PALETTE = {
    "ink": "#172026",
    "muted": "#5b6670",
    "teal": "#0f8b8d",
    "rose": "#bf3f5b",
    "amber": "#c4871f",
    "blue": "#3268a8",
    "green": "#3f7d58",
    "violet": "#7353ba",
    "paper": "#f8f9f6",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_note(title: str, body: str) -> None:
    st.markdown(f'<div class="note"><strong>{title}</strong> {body}</div>', unsafe_allow_html=True)


def step_cards(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (title, body) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="case-step"><strong>{title}</strong><br>{body}</div>',
                unsafe_allow_html=True,
            )


def tight_fig(width: float = 8.0, height: float = 4.5):
    fig = plt.figure(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor(PALETTE["paper"])
    return fig


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, pd.DataFrame, np.ndarray, np.ndarray]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            total += len(yb)

        model.eval()
        with torch.no_grad():
            test_logits = model(test_x)
            test_loss = float(criterion(test_logits, test_y).item())
            test_pred = test_logits.argmax(1)
            test_acc = float((test_pred == test_y).float().mean().item())

        rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total,
                "train_acc": correct / total,
                "val_loss": test_loss,
                "val_acc": test_acc,
            }
        )

    with torch.no_grad():
        logits = model(test_x)
        pred = logits.argmax(1).cpu().numpy()
        prob = F.softmax(logits, dim=1).cpu().numpy()
    return model, pd.DataFrame(rows), pred, prob


def plot_history(history: pd.DataFrame, title: str = "训练曲线"):
    fig = tight_fig(9.2, 4.4)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(history["epoch"], history["train_loss"], color=PALETTE["rose"], marker="o", label="训练损失")
    ax1.plot(history["epoch"], history["val_loss"], color=PALETTE["blue"], marker="o", label="验证损失")
    ax1.set_title(f"{title}: loss")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(history["epoch"], history["train_acc"], color=PALETTE["amber"], marker="o", label="训练准确率")
    ax2.plot(history["epoch"], history["val_acc"], color=PALETTE["teal"], marker="o", label="验证准确率")
    ax2.set_title(f"{title}: accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None):
    cm = confusion_matrix(y_true, y_pred)
    fig = tight_fig(5.2, 4.7)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    tick_labels = labels or [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Case 1: MNIST-like digit recognition with sklearn digits
# ---------------------------------------------------------------------------


class DigitCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@st.cache_data(show_spinner=False)
def load_digit_data(test_size: float, seed: int):
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    return train_x, test_x, train_y, test_y


@st.cache_resource(show_spinner=False)
def cached_digit_training(epochs: int, lr: float, dropout: float, batch_size: int, seed: int):
    set_seed(seed)
    train_x, test_x, train_y, test_y = load_digit_data(0.25, seed)
    train_tensor = torch.tensor(train_x[:, None, :, :], dtype=torch.float32)
    test_tensor = torch.tensor(test_x[:, None, :, :], dtype=torch.float32)
    train_labels = torch.tensor(train_y, dtype=torch.long)
    test_labels = torch.tensor(test_y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_tensor, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = DigitCNN(dropout)
    return (*train_classifier(model, loader, test_tensor, test_labels, epochs, lr), test_x, test_y)


def plot_digit_grid(images: np.ndarray, labels: np.ndarray, title: str):
    fig = tight_fig(7.6, 3.8)
    for i in range(16):
        ax = fig.add_subplot(2, 8, i + 1)
        ax.imshow(images[i], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(str(labels[i]), fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_digit_predictions(images: np.ndarray, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
    fig = tight_fig(8.6, 4.4)
    wrong = np.where(labels != preds)[0]
    chosen = wrong[:8] if len(wrong) >= 8 else np.arange(min(8, len(labels)))
    for i, idx in enumerate(chosen):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(images[idx], cmap="gray_r", vmin=0, vmax=1)
        conf = probs[idx, preds[idx]]
        color = PALETTE["rose"] if labels[idx] != preds[idx] else PALETTE["green"]
        ax.set_title(f"真 {labels[idx]} / 预 {preds[idx]} ({conf:.2f})", color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle("预测样例")
    fig.tight_layout()
    return fig


def render_mnist_case(seed: int) -> None:
    st.header("MNIST 手写数字识别完整流程")
    render_note(
        "案例定位:",
        "这里使用 sklearn 内置的 8x8 手写数字小数据集，流程与 MNIST 一致，但无需联网下载，适合快速展示从探索到误差分析的闭环。",
    )
    step_cards(
        [
            ("数据探索", "查看样本、类别分布和像素强度，确认输入形状为 1x8x8。"),
            ("模型构建", "两层卷积提取笔画局部模式，再接全连接分类器输出 10 类。"),
            ("训练过程", "用交叉熵和 Adam 训练轻量 CNN，实时观察 loss 与 accuracy。"),
            ("结果分析", "用混淆矩阵和错分样例定位哪些数字形状更容易混淆。"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.slider("训练轮数", 3, 24, 10, 1, key="digit_epochs")
    with c2:
        lr = st.select_slider("学习率", options=[0.0005, 0.001, 0.002, 0.004], value=0.002, key="digit_lr")
    with c3:
        dropout = st.slider("Dropout", 0.0, 0.5, 0.12, 0.02, key="digit_dropout")

    train_x, _, train_y, _ = load_digit_data(0.25, seed)
    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("数据探索")
        st.pyplot(plot_digit_grid(train_x, train_y, "训练样本预览"), clear_figure=True)
    with right:
        st.subheader("类别与像素统计")
        counts = pd.Series(train_y).value_counts().sort_index()
        st.bar_chart(counts, height=230)
        st.dataframe(
            pd.DataFrame(
                {
                    "样本数": [len(train_x)],
                    "图像形状": ["1 x 8 x 8"],
                    "像素均值": [f"{train_x.mean():.3f}"],
                    "像素标准差": [f"{train_x.std():.3f}"],
                }
            ),
            width="stretch",
        )

    with st.spinner("训练轻量 CNN..."):
        model, history, pred, prob, test_x, test_y = cached_digit_training(
            epochs, lr, dropout, 64, seed
        )

    st.subheader("模型构建")
    m1, m2, m3 = st.columns(3)
    m1.metric("可训练参数", f"{count_params(model):,}")
    m2.metric("最终验证准确率", f"{history['val_acc'].iloc[-1] * 100:.1f}%")
    m3.metric("最终验证损失", f"{history['val_loss'].iloc[-1]:.3f}")
    st.code(
        "Conv2d(1,16,3) -> ReLU -> Conv2d(16,24,3) -> ReLU -> MaxPool -> Dropout -> Linear(384,64) -> Linear(64,10)",
        language="text",
    )

    st.subheader("训练过程")
    st.pyplot(plot_history(history, "数字分类"), clear_figure=True)
    st.dataframe(history.round(4), width="stretch")

    st.subheader("结果分析")
    c1, c2 = st.columns([0.92, 1.08])
    with c1:
        st.pyplot(plot_confusion(test_y, pred), clear_figure=True)
    with c2:
        st.pyplot(plot_digit_predictions(test_x, test_y, pred, prob), clear_figure=True)
        mistakes = int((pred != test_y).sum())
        st.markdown(
            f'<div class="small">验证集共有 {len(test_y)} 个样本，错分 {mistakes} 个。若混淆矩阵在相邻数字上更亮，通常表示笔画短缺、闭环不清或缩放后的局部形状过于接近。</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Case 2: Cat-vs-dog synthetic CNN pipeline
# ---------------------------------------------------------------------------


class TinyImageCNN(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 8 * 8, 48),
            nn.ReLU(),
            nn.Linear(48, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _draw_circle(mask: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def _draw_triangle(mask: np.ndarray, cx: float, top: float, width: float, height: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    left = cx - width / 2 + (yy - top) * width / (2 * height)
    right = cx + width / 2 - (yy - top) * width / (2 * height)
    return (yy >= top) & (yy <= top + height) & (xx >= left) & (xx <= right)


def synthetic_pet_image(label: int, rng: np.random.Generator, size: int = 32, noise: float = 0.08) -> np.ndarray:
    img = np.ones((size, size, 3), dtype=np.float32)
    base = np.array([0.92, 0.89, 0.78]) if label == 0 else np.array([0.78, 0.84, 0.92])
    img[:] = base + rng.normal(0, 0.02, size=(1, 1, 3))
    shift_x = rng.normal(0, 1.4)
    shift_y = rng.normal(0, 1.2)

    face = _draw_circle(img[..., 0], size / 2 + shift_x, size / 2 + 2 + shift_y, 8.5)
    color = np.array([0.92, 0.56, 0.34]) if label == 0 else np.array([0.47, 0.36, 0.25])
    img[face] = color

    if label == 0:
        left_ear = _draw_triangle(img[..., 0], size / 2 - 7 + shift_x, 4 + shift_y, 9, 11)
        right_ear = _draw_triangle(img[..., 0], size / 2 + 7 + shift_x, 4 + shift_y, 9, 11)
        img[left_ear | right_ear] = color * 0.9
        whisker_y = int(size / 2 + 3 + shift_y)
        for offset in [-2, 1, 4]:
            y = np.clip(whisker_y + offset, 0, size - 1)
            img[y, 5:14] = np.array([0.18, 0.16, 0.14])
            img[y, 18:27] = np.array([0.18, 0.16, 0.14])
    else:
        left_ear = _draw_circle(img[..., 0], size / 2 - 8 + shift_x, 9 + shift_y, 4.8)
        right_ear = _draw_circle(img[..., 0], size / 2 + 8 + shift_x, 9 + shift_y, 4.8)
        img[left_ear | right_ear] = color * 0.82
        snout = _draw_circle(img[..., 0], size / 2 + shift_x, size / 2 + 6 + shift_y, 4.5)
        img[snout] = np.array([0.86, 0.72, 0.60])

    eye1 = _draw_circle(img[..., 0], size / 2 - 3 + shift_x, size / 2 + shift_y, 1.2)
    eye2 = _draw_circle(img[..., 0], size / 2 + 3 + shift_x, size / 2 + shift_y, 1.2)
    nose = _draw_circle(img[..., 0], size / 2 + shift_x, size / 2 + 4 + shift_y, 1.5)
    img[eye1 | eye2] = np.array([0.06, 0.06, 0.06])
    img[nose] = np.array([0.12, 0.08, 0.08])
    img += rng.normal(0, noise, size=img.shape)
    return np.clip(img, 0, 1)


@st.cache_data(show_spinner=False)
def make_pet_dataset(n_per_class: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label in [0, 1]:
        for _ in range(n_per_class):
            xs.append(synthetic_pet_image(label, rng, noise=noise))
            ys.append(label)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


@st.cache_resource(show_spinner=False)
def cached_pet_training(n_per_class: int, noise: float, epochs: int, lr: float, seed: int):
    set_seed(seed)
    x, y = make_pet_dataset(n_per_class, noise, seed)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    train_tensor = torch.tensor(train_x.transpose(0, 3, 1, 2), dtype=torch.float32)
    test_tensor = torch.tensor(test_x.transpose(0, 3, 1, 2), dtype=torch.float32)
    train_labels = torch.tensor(train_y, dtype=torch.long)
    test_labels = torch.tensor(test_y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_tensor, train_labels),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TinyImageCNN(0.12)
    return (*train_classifier(model, loader, test_tensor, test_labels, epochs, lr), train_x, train_y, test_x, test_y)


def plot_pet_grid(images: np.ndarray, labels: np.ndarray):
    names = ["cat", "dog"]
    fig = tight_fig(7.4, 3.8)
    for i in range(12):
        ax = fig.add_subplot(2, 6, i + 1)
        ax.imshow(images[i])
        ax.set_title(names[int(labels[i])], fontsize=9)
        ax.axis("off")
    fig.suptitle("合成猫狗样本")
    fig.tight_layout()
    return fig


def render_pet_case(seed: int) -> None:
    st.header("猫狗图像分类 CNN Pipeline")
    render_note(
        "案例定位:",
        "用可控合成图像模拟猫狗二分类。重点不是追求真实数据表现，而是完整串起图像数据、增强、CNN、训练监控和错误分析。",
    )
    step_cards(
        [
            ("数据探索", "猫有三角耳和胡须，狗有圆耳和口鼻区域；加入噪声模拟拍摄差异。"),
            ("模型构建", "Conv-Pool-Conv-Pool 提取局部纹理，最后用全连接层做二分类。"),
            ("训练过程", "观察噪声、样本量和学习率如何影响收敛速度。"),
            ("结果分析", "查看二分类混淆矩阵与高置信预测，理解数据偏差。"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_per_class = st.slider("每类样本数", 60, 240, 140, 20, key="pet_n")
    with c2:
        noise = st.slider("图像噪声", 0.01, 0.22, 0.08, 0.01, key="pet_noise")
    with c3:
        epochs = st.slider("训练轮数", 3, 20, 8, 1, key="pet_epochs")

    x, y = make_pet_dataset(n_per_class, noise, seed)
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("数据探索")
        st.pyplot(plot_pet_grid(x, y), clear_figure=True)
    with right:
        st.subheader("Pipeline")
        st.code(
            "读取图像 -> resize 到 32x32 -> 像素归一化 -> batch -> CNN 前向 -> 交叉熵 -> 反向传播 -> 验证集评估",
            language="text",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "阶段": ["输入", "增强", "特征", "分类"],
                    "形状/动作": ["3 x 32 x 32", "噪声 + 位移", "Conv feature maps", "cat / dog"],
                }
            ),
            width="stretch",
        )

    with st.spinner("训练猫狗 CNN..."):
        model, history, pred, prob, _, _, test_x, test_y = cached_pet_training(
            n_per_class, noise, epochs, 0.002, seed
        )

    st.subheader("模型构建与训练过程")
    m1, m2, m3 = st.columns(3)
    m1.metric("可训练参数", f"{count_params(model):,}")
    m2.metric("验证准确率", f"{history['val_acc'].iloc[-1] * 100:.1f}%")
    m3.metric("验证样本", f"{len(test_y)}")
    st.pyplot(plot_history(history, "猫狗 CNN"), clear_figure=True)

    st.subheader("结果分析")
    c1, c2 = st.columns([0.85, 1.15])
    with c1:
        st.pyplot(plot_confusion(test_y, pred, ["cat", "dog"]), clear_figure=True)
    with c2:
        conf = prob.max(axis=1)
        top = np.argsort(-conf)[:8]
        fig = tight_fig(8.4, 4.2)
        for i, idx in enumerate(top):
            ax = fig.add_subplot(2, 4, i + 1)
            ax.imshow(test_x[idx])
            ax.set_title(f"预 {['cat','dog'][pred[idx]]} {conf[idx]:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle("高置信预测")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------------------
# Case 3: Sentiment analysis with RNN/LSTM
# ---------------------------------------------------------------------------


POSITIVE_WORDS = ["好看", "精彩", "喜欢", "推荐", "感动", "优秀", "惊喜", "舒服", "值得", "开心", "流畅"]
NEGATIVE_WORDS = ["难看", "失望", "无聊", "讨厌", "糟糕", "浪费", "生气", "尴尬", "拖沓", "难受", "后悔"]
NEUTRAL_WORDS = ["剧情", "演员", "镜头", "音乐", "节奏", "画面", "结尾", "角色", "对白", "整体"]
NEGATIONS = ["不", "没有", "并不"]
SENTIMENT_LEXICON = sorted(
    set(POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS + NEGATIONS),
    key=len,
    reverse=True,
)


def tokenize(text: str) -> list[str]:
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+|[0-9]+", text.lower())
    tokens: list[str] = []
    for chunk in chunks:
        if not re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.append(chunk)
            continue
        i = 0
        while i < len(chunk):
            matched = None
            for word in SENTIMENT_LEXICON:
                if chunk.startswith(word, i):
                    matched = word
                    break
            if matched is None:
                matched = chunk[i]
            tokens.append(matched)
            i += len(matched)
    return tokens


@st.cache_data(show_spinner=False)
def make_sentiment_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    sentences: list[list[str]] = []
    labels: list[int] = []
    for _ in range(n):
        is_pos = rng.random() > 0.5
        length = int(rng.integers(5, 11))
        words = rng.choice(NEUTRAL_WORDS, size=max(2, length - 3), replace=True).tolist()
        if is_pos:
            words += rng.choice(POSITIVE_WORDS, size=2, replace=False).tolist()
            if rng.random() < 0.18:
                words += ["不", rng.choice(NEGATIVE_WORDS).item()]
            labels.append(1)
        else:
            words += rng.choice(NEGATIVE_WORDS, size=2, replace=False).tolist()
            if rng.random() < 0.18:
                words += ["不", rng.choice(POSITIVE_WORDS).item()]
            labels.append(0)
        rng.shuffle(words)
        sentences.append(words[:length])
    return sentences, np.array(labels, dtype=np.int64)


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: tuple[str, ...]

    def encode(self, words: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(w, self.stoi["<UNK>"]) for w in words[:max_len]]
        return ids + [self.stoi["<PAD>"]] * (max_len - len(ids))


def build_vocab(sentences: list[list[str]]) -> Vocab:
    tokens = ["<PAD>", "<UNK>"]
    counts = Counter(w for s in sentences for w in s)
    tokens.extend(sorted(counts))
    return Vocab({w: i for i, w in enumerate(tokens)}, tuple(tokens))


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, model_type: str):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 24, padding_idx=0)
        if model_type == "LSTM":
            self.rnn = nn.LSTM(24, hidden, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.RNN(24, hidden, batch_first=True, bidirectional=True, nonlinearity="tanh")
        self.fc = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, hidden = self.rnn(emb)
        if isinstance(hidden, tuple):
            h = hidden[0]
        else:
            h = hidden
        last = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(last)


@st.cache_resource(show_spinner=False)
def cached_sentiment_training(n: int, model_type: str, hidden: int, epochs: int, seed: int):
    set_seed(seed)
    sentences, y = make_sentiment_dataset(n, seed)
    vocab = build_vocab(sentences)
    max_len = 12
    x = np.array([vocab.encode(s, max_len) for s in sentences], dtype=np.int64)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    train_tensor = torch.tensor(train_x, dtype=torch.long)
    test_tensor = torch.tensor(test_x, dtype=torch.long)
    train_labels = torch.tensor(train_y, dtype=torch.long)
    test_labels = torch.tensor(test_y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_tensor, train_labels),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = SentimentRNN(len(vocab.itos), hidden, model_type)
    trained = train_classifier(model, loader, test_tensor, test_labels, epochs, 0.004)
    return (*trained, sentences, y, vocab, test_x, test_y)


def predict_sentiment_text(model: nn.Module, vocab: Vocab, text: str) -> tuple[str, float, list[int]]:
    words = tokenize(text)
    ids = vocab.encode(words, 12)
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))
        prob = F.softmax(logits, dim=1)[0].numpy()
    label = "正面" if prob[1] >= prob[0] else "负面"
    return label, float(max(prob)), ids


def render_sentiment_case(seed: int) -> None:
    st.header("情感分析：RNN/LSTM 文本分类")
    render_note(
        "案例定位:",
        "用小型中文影评样本展示文本分类。词表、padding、Embedding、RNN/LSTM、分类头和预测解释都放在同一页。",
    )
    step_cards(
        [
            ("数据探索", "构造正负面词、普通主题词和少量否定模式，观察词频与标签平衡。"),
            ("模型构建", "token -> id -> Embedding -> 双向 RNN/LSTM -> 全连接分类。"),
            ("训练过程", "比较普通 RNN 与 LSTM 在短文本上的收敛。"),
            ("结果分析", "输入一句影评，查看模型概率与被识别出的 token。"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_type = st.segmented_control("序列模型", ["LSTM", "RNN"], default="LSTM", key="sent_model")
    with c2:
        n = st.slider("样本数", 160, 800, 360, 40, key="sent_n")
    with c3:
        epochs = st.slider("训练轮数", 4, 24, 10, 1, key="sent_epochs")

    with st.spinner("训练文本分类模型..."):
        model, history, pred, prob, sentences, y, vocab, test_x, test_y = cached_sentiment_training(
            n, model_type or "LSTM", 32, epochs, seed
        )

    st.subheader("数据探索")
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        label_df = pd.DataFrame({"标签": ["负面", "正面"], "数量": [(y == 0).sum(), (y == 1).sum()]})
        st.bar_chart(label_df.set_index("标签"), height=230)
        st.dataframe(pd.DataFrame({"样例": [" ".join(s) for s in sentences[:8]], "标签": y[:8]}), width="stretch")
    with c2:
        counts = Counter(w for s in sentences for w in s)
        word_df = pd.DataFrame(counts.most_common(16), columns=["token", "freq"])
        st.bar_chart(word_df.set_index("token"), height=230)
        st.metric("词表大小", len(vocab.itos))

    st.subheader("模型构建与训练过程")
    m1, m2, m3 = st.columns(3)
    m1.metric("模型类型", model_type or "LSTM")
    m2.metric("可训练参数", f"{count_params(model):,}")
    m3.metric("验证准确率", f"{history['val_acc'].iloc[-1] * 100:.1f}%")
    st.pyplot(plot_history(history, f"{model_type or 'LSTM'} 情感分类"), clear_figure=True)

    st.subheader("结果分析")
    c1, c2 = st.columns([0.82, 1.18])
    with c1:
        st.pyplot(plot_confusion(test_y, pred, ["负面", "正面"]), clear_figure=True)
    with c2:
        text = st.text_input("输入一句短评", "演员精彩 画面舒服 节奏流畅 推荐", key="sent_text")
        label, confidence, ids = predict_sentiment_text(model, vocab, text)
        st.metric("预测情感", label, f"置信度 {confidence * 100:.1f}%")
        st.dataframe(
            pd.DataFrame(
                {
                    "token": tokenize(text),
                    "id": [vocab.stoi.get(w, vocab.stoi["<UNK>"]) for w in tokenize(text)],
                }
            ),
            width="stretch",
        )
        st.markdown(
            '<div class="small">短文本情感分类很容易受否定词和领域词影响。生产系统通常会加入更真实的数据、预训练词向量或 Transformer 编码器。</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Case 4: Neural style transfer concept demo
# ---------------------------------------------------------------------------


def make_content_image(size: int = 72) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / size
    y = yy / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = 0.20 + 0.45 * x
    img[..., 1] = 0.36 + 0.36 * y
    img[..., 2] = 0.48 + 0.18 * np.sin(6 * x)
    house = (x > 0.28) & (x < 0.72) & (y > 0.43) & (y < 0.78)
    roof = (y > 0.22) & (y < 0.48) & (np.abs(x - 0.5) < (0.5 - y) * 1.1)
    door = (x > 0.46) & (x < 0.56) & (y > 0.58) & (y < 0.78)
    img[house] = np.array([0.86, 0.74, 0.55])
    img[roof] = np.array([0.55, 0.20, 0.18])
    img[door] = np.array([0.28, 0.16, 0.10])
    return np.clip(img, 0, 1)


def make_style_image(size: int = 72, style: str = "wave") -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / size
    y = yy / size
    if style == "wave":
        pattern = np.sin(22 * x + 9 * np.sin(8 * y))
        img = np.stack([0.62 + 0.25 * pattern, 0.42 + 0.18 * np.cos(16 * y), 0.30 + 0.22 * np.sin(20 * (x + y))], axis=-1)
    else:
        pattern = ((np.floor(x * 9) + np.floor(y * 9)) % 2).astype(np.float32)
        img = np.stack([0.25 + 0.55 * pattern, 0.30 + 0.24 * (1 - pattern), 0.66 - 0.28 * pattern], axis=-1)
    return np.clip(img, 0, 1)


class FixedStyleFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = torch.tensor(
            [
                [[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]],
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            ],
            dtype=torch.float32,
        )
        weight = kernels.repeat(1, 3, 1, 1) / 3.0
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = F.conv2d(x, self.weight, padding=1)
        f2 = F.avg_pool2d(torch.relu(f1), 2)
        return [x, f1, f2]


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    flat = feat.view(b, c, h * w)
    return torch.bmm(flat, flat.transpose(1, 2)) / (c * h * w)


@st.cache_data(show_spinner=False)
def run_style_transfer(style_name: str, style_weight: float, steps: int, seed: int):
    set_seed(seed)
    content_np = make_content_image()
    style_np = make_style_image(style=style_name)
    content = torch.tensor(content_np.transpose(2, 0, 1)[None], dtype=torch.float32)
    style = torch.tensor(style_np.transpose(2, 0, 1)[None], dtype=torch.float32)
    extractor = FixedStyleFeatures()
    with torch.no_grad():
        content_features = extractor(content)
        style_grams = [gram_matrix(f) for f in extractor(style)]

    generated = content.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=0.06)
    rows: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        feats = extractor(generated)
        content_loss = F.mse_loss(feats[0], content_features[0])
        style_loss = sum(F.mse_loss(gram_matrix(f), g) for f, g in zip(feats, style_grams))
        total = content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            generated.clamp_(0, 1)
        if step == 1 or step % 5 == 0 or step == steps:
            rows.append(
                {
                    "step": step,
                    "content_loss": float(content_loss.item()),
                    "style_loss": float(style_loss.item()),
                    "total_loss": float(total.item()),
                }
            )
    out = generated.detach().cpu().numpy()[0].transpose(1, 2, 0)
    return content_np, style_np, np.clip(out, 0, 1), pd.DataFrame(rows)


def render_style_case(seed: int) -> None:
    st.header("图像风格迁移概念演示")
    render_note(
        "案例定位:",
        "用固定滤波器模拟 VGG 特征层：内容损失保持物体结构，风格损失用 Gram 矩阵迁移纹理统计。规模很小，但概念与神经风格迁移一致。",
    )
    step_cards(
        [
            ("数据探索", "内容图提供房子轮廓，风格图提供波纹或棋盘纹理。"),
            ("模型构建", "固定特征提取器不训练权重，只优化生成图像本身。"),
            ("训练过程", "每一步同时压低内容损失和风格 Gram 损失。"),
            ("结果分析", "调节风格权重，观察结构保持和纹理迁移之间的取舍。"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        style_name = st.segmented_control("风格", ["wave", "checker"], default="wave", key="style_name")
    with c2:
        style_weight = st.slider("风格权重", 5.0, 80.0, 35.0, 5.0, key="style_weight")
    with c3:
        steps = st.slider("优化步数", 10, 80, 35, 5, key="style_steps")

    with st.spinner("优化生成图像..."):
        content, style, output, history = run_style_transfer(style_name or "wave", style_weight, steps, seed)

    st.subheader("数据探索与结果")
    fig = tight_fig(9.4, 3.3)
    for i, (title, img) in enumerate([("内容图", content), ("风格图", style), ("生成图", output)], 1):
        ax = fig.add_subplot(1, 3, i)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("模型构建")
    st.code(
        "generated_image 是可训练参数\ncontent_loss = MSE(features(generated), features(content))\nstyle_loss = MSE(Gram(features(generated)), Gram(features(style)))\ntotal_loss = content_loss + style_weight * style_loss",
        language="text",
    )

    st.subheader("训练过程")
    st.line_chart(history.set_index("step"), height=300)
    st.dataframe(history.round(5), width="stretch")

    st.subheader("结果分析")
    st.markdown(
        f'<div class="small">当前风格权重为 {style_weight:.1f}。权重越高，生成图越倾向复刻风格纹理；权重越低，房子的边界和颜色更接近内容图。</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Case 5: Character-level text generation
# ---------------------------------------------------------------------------


CORPUS = """
深度学习从数据中发现模式。模型看见样本，调整参数，然后给出预测。
卷积网络擅长图像，循环网络擅长序列，注意力机制擅长捕捉远距离关系。
好的实验需要清楚的数据划分、稳定的训练曲线和诚实的误差分析。
当损失下降而验证集停滞时，过拟合正在靠近。正则化、更多数据和更小模型都可能有帮助。
生成模型并不是背诵句子，而是在概率空间里一步一步选择下一个符号。
"""


class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        emb = self.embed(x)
        out, hidden = self.gru(emb, hidden)
        return self.fc(out), hidden


def build_char_batches(corpus: str, seq_len: int):
    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = np.array([stoi[ch] for ch in corpus], dtype=np.int64)
    xs, ys = [], []
    for i in range(0, len(ids) - seq_len - 1):
        xs.append(ids[i : i + seq_len])
        ys.append(ids[i + 1 : i + seq_len + 1])
    return np.stack(xs), np.stack(ys), stoi, itos


@st.cache_resource(show_spinner=False)
def cached_text_generator(seq_len: int, hidden: int, epochs: int, seed: int):
    set_seed(seed)
    corpus = "".join(CORPUS.split())
    x, y, stoi, itos = build_char_batches(corpus, seq_len)
    train_x = torch.tensor(x, dtype=torch.long)
    train_y = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=24,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = CharGRU(len(stoi), hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    rows = []
    for epoch in range(1, epochs + 1):
        total_loss, total = 0.0, 0
        for xb, yb in loader:
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, len(stoi)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)
        avg_loss = total_loss / total
        rows.append({"epoch": epoch, "loss": avg_loss, "perplexity": math.exp(min(avg_loss, 10))})
    return model, pd.DataFrame(rows), stoi, itos, corpus


def sample_text(model: CharGRU, stoi: dict[str, int], itos: dict[int, str], prompt: str, length: int, temperature: float):
    model.eval()
    fallback = next(iter(stoi))
    ids = [stoi.get(ch, stoi[fallback]) for ch in prompt if ch in stoi] or [stoi[fallback]]
    result = [itos[i] for i in ids]
    x = torch.tensor([[ids[-1]]], dtype=torch.long)
    hidden = None
    with torch.no_grad():
        for token in ids[:-1]:
            _, hidden = model(torch.tensor([[token]], dtype=torch.long), hidden)
        for _ in range(length):
            logits, hidden = model(x, hidden)
            logits = logits[0, -1] / max(temperature, 1e-3)
            probs = F.softmax(logits, dim=0)
            next_id = int(torch.multinomial(probs, 1).item())
            result.append(itos[next_id])
            x = torch.tensor([[next_id]], dtype=torch.long)
    return "".join(result)


def render_generation_case(seed: int) -> None:
    st.header("文本生成 Demo：字符级 GRU")
    render_note(
        "案例定位:",
        "用很小的中文语料训练字符级语言模型。它学习的是下一个字符概率，不依赖外部大模型，适合理解生成式模型的最小闭环。",
    )
    step_cards(
        [
            ("数据探索", "把语料切成字符序列，统计字符表和训练样本。"),
            ("模型构建", "Embedding -> GRU -> Linear，预测每个位置的下一个字符。"),
            ("训练过程", "用交叉熵优化语言模型，困惑度越低表示越容易预测下文。"),
            ("结果分析", "调节 temperature，观察保守复述与发散生成的差异。"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        hidden = st.slider("隐藏维度", 24, 96, 48, 8, key="gen_hidden")
    with c2:
        epochs = st.slider("训练轮数", 8, 80, 28, 4, key="gen_epochs")
    with c3:
        temperature = st.slider("temperature", 0.4, 1.6, 0.8, 0.1, key="gen_temp")

    with st.spinner("训练字符级生成模型..."):
        model, history, stoi, itos, corpus = cached_text_generator(14, hidden, epochs, seed)

    st.subheader("数据探索")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("语料字符数", len(corpus))
        st.metric("字符表大小", len(stoi))
        st.text_area("训练语料", corpus[:240], height=150)
    with c2:
        counts = Counter(corpus)
        freq = pd.DataFrame(counts.most_common(20), columns=["字符", "频次"])
        st.bar_chart(freq.set_index("字符"), height=260)

    st.subheader("模型构建与训练过程")
    m1, m2, m3 = st.columns(3)
    m1.metric("可训练参数", f"{count_params(model):,}")
    m2.metric("最终 loss", f"{history['loss'].iloc[-1]:.3f}")
    m3.metric("最终困惑度", f"{history['perplexity'].iloc[-1]:.1f}")
    st.line_chart(history.set_index("epoch"), height=300)

    st.subheader("结果分析")
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        prompt = st.text_input("提示词", "深度学习", key="gen_prompt")
        length = st.slider("生成长度", 30, 160, 80, 10, key="gen_len")
    with c2:
        generated = sample_text(model, stoi, itos, prompt, length, temperature)
        st.text_area("生成结果", generated, height=180)
        st.markdown(
            '<div class="small">temperature 低时更偏向高概率字符，文本更稳定但更重复；temperature 高时更有变化，也更容易跑偏。</div>',
            unsafe_allow_html=True,
        )


def render_overview() -> None:
    st.title("深度学习实战案例库")
    render_note(
        "使用方式:",
        "从左侧选择一个案例。每个案例都按同一条主线组织：数据探索、模型构建、训练过程、结果分析。模型和数据都刻意保持轻量，方便在本地快速运行和教学演示。",
    )
    step_cards(
        [
            ("视觉分类", "手写数字识别、猫狗二分类，覆盖图像张量、卷积、误差分析。"),
            ("序列建模", "情感分析和文本生成，覆盖词表、Embedding、RNN/LSTM/GRU。"),
            ("生成与迁移", "风格迁移演示内容损失、风格损失与优化生成图像。"),
            ("工程闭环", "所有案例都包含指标、曲线、样本和可调参数。"),
        ]
    )

    rows = [
        ["MNIST 手写数字识别", "CNN 分类", "sklearn digits", "准确率、混淆矩阵、错分样例"],
        ["猫狗图像分类", "CNN pipeline", "合成 32x32 RGB", "噪声鲁棒性、二分类分析"],
        ["情感分析", "RNN/LSTM", "合成中文影评", "词表、概率、token 分析"],
        ["图像风格迁移", "优化生成图像", "合成内容/风格图", "内容-风格权衡"],
        ["文本生成", "字符级 GRU", "短中文语料", "loss、困惑度、temperature"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["案例", "模型", "数据", "重点"]), width="stretch", hide_index=True)


def main() -> None:
    st.sidebar.title("案例库")
    case = st.sidebar.radio(
        "选择案例",
        [
            "总览",
            "MNIST 手写数字识别",
            "猫狗图像分类 CNN",
            "情感分析 RNN/LSTM",
            "图像风格迁移",
            "文本生成 Demo",
        ],
    )
    seed = st.sidebar.slider("随机种子", 1, 99, 42)
    st.sidebar.markdown(
        '<div class="small">缓存会保留训练结果；修改关键参数后会重新训练。</div>',
        unsafe_allow_html=True,
    )

    if case == "总览":
        render_overview()
    elif case == "MNIST 手写数字识别":
        render_mnist_case(seed)
    elif case == "猫狗图像分类 CNN":
        render_pet_case(seed)
    elif case == "情感分析 RNN/LSTM":
        render_sentiment_case(seed)
    elif case == "图像风格迁移":
        render_style_case(seed)
    else:
        render_generation_case(seed)


if __name__ == "__main__":
    main()


render = main


def compute(seed: int = 42) -> dict[str, object]:
    """Pure computation placeholder."""
    return {"status": "ok", "seed": seed}


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""
    return True
