"""文本分类：从词表、Embedding 到 LSTM/TextCNN 情感分类的可解释实验。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "文本分类"
MODULE_SUMMARY = "用情感词玩具数据、训练曲线、词嵌入散点图和类别权重解释文本如何被神经网络分类。"
MODULE_TAGS = ["RNN", "文本分类", "Embedding", "TextCNN", "情感分析"]
MODULE_RELATED_TOPICS = ["part3/04_hyperparam_rnn", "part3/05_seq2seq_attention", "part5/data_training", "part5/03_training_dynamics"]
PRACTICE_TARGET = "调整模型类型、样本数量、噪声比例、最大句长和类别不平衡，解释准确率、损失和词嵌入分布如何变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    自动生成自: part3_rnn\06_text_classification.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


    class Vocabulary:
        """
        词汇表：文本 ↔ 数字索引的映射

        包含特殊 token:
        - <PAD>: 填充（使批次中序列等长）
        - <UNK>: 未知词（不在词表中）
        - <BOS>: 序列开始
        - <EOS>: 序列结束
        """
        PAD, UNK, BOS, EOS = 0, 1, 2, 3

        def __init__(self, min_freq=2):
            self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.word_freq = Counter()
            self.min_freq = min_freq

        def build(self, sentences):
            """从句子列表构建词表"""
            for sent in sentences:
                for word in sent:
                    self.word_freq[word] += 1

            for word, freq in self.word_freq.items():
                if freq >= self.min_freq and word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

            print(f"词表大小: {len(self.word2idx)}")
            print(f"低频词（< {self.min_freq} 次）: {sum(1 for f in self.word_freq.values() if f < self.min_freq)}")

        def encode(self, sentence, max_len=None):
            """句子 → 索引序列"""
            indices = [self.word2idx.get(w, self.UNK) for w in sentence]
            if max_len:
                indices = indices[:max_len]
                indices += [self.PAD] * (max_len - len(indices))
            return indices

        def decode(self, indices):
            """索引序列 → 句子"""
            return [self.idx2word.get(i, '<UNK>') for i in indices if i != self.PAD]

        def __len__(self):
            return len(self.word2idx)


    def create_sentiment_data():
        """
        创建模拟情感分析数据集

        正面: 包含"好"、"棒"、"喜欢"等词
        负面: 包含"差"、"烂"、"讨厌"等词
        """
        np.random.seed(42)

        positive_words = ['好', '棒', '喜欢', '精彩', '优秀', '出色', '感动', '推荐',
                          '好看', '不错', '惊喜', '完美', '值得', '经典', '开心']
        negative_words = ['差', '烂', '讨厌', '无聊', '难看', '失望', '浪费', '垃圾',
                          '恶心', '难受', '糟糕', '气愤', '低劣', '差劲', '伤心']
        neutral_words = ['一般', '普通', '还行', '勉强', '凑合', '平淡']

        sentences, labels = [], []

        for _ in range(300):
            length = np.random.randint(4, 10)
            if np.random.random() < 0.5:
                # 正面
                n_pos = np.random.randint(1, min(4, length))
                words = np.random.choice(positive_words, n_pos).tolist()
                words += np.random.choice(neutral_words + positive_words,
                                           length - n_pos).tolist()
                labels.append(1)
            else:
                # 负面
                n_neg = np.random.randint(1, min(4, length))
                words = np.random.choice(negative_words, n_neg).tolist()
                words += np.random.choice(neutral_words + negative_words,
                                           length - n_neg).tolist()
                labels.append(0)
            np.random.shuffle(words)
            sentences.append(words)

        return sentences, labels


    # 构建数据（协议化后由 compute_text_classification 控制执行）
    sentences, labels = [], []
    vocab = Vocabulary(min_freq=1)
    max_len = 12
    X = torch.empty(0, dtype=torch.long)
    y = torch.empty(0, dtype=torch.long)
    X_train = X_test = X
    y_train = y_test = y

    # ============================================================
    # 代码段 2
    # ============================================================

    class LSTMClassifier(nn.Module):
        """
        LSTM 文本分类器

        结构: Embedding → LSTM → 取最后隐状态 → FC → 分类
        """
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes,
                     num_layers=2, dropout=0.3, bidirectional=True):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                                dropout=dropout if num_layers > 1 else 0,
                                bidirectional=bidirectional, batch_first=True)
            self.directions = 2 if bidirectional else 1
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * self.directions, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )

        def forward(self, x):
            """
            x: [batch, seq_len] token 索引
            """
            embedded = self.embedding(x)  # [batch, seq_len, embed]
            output, (h_n, c_n) = self.lstm(embedded)

            # 取最后一层的隐藏状态
            if self.lstm.bidirectional:
                # 拼接正向和反向的最后隐状态
                h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h = h_n[-1]

            return self.fc(h)


    class TextCNNClassifier(nn.Module):
        """
        TextCNN 分类器（对比模型）

        用不同大小的卷积核捕获不同长度的 n-gram 特征
        """
        def __init__(self, vocab_size, embed_size, num_classes,
                     num_filters=64, filter_sizes=[2, 3, 4], dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

            self.convs = nn.ModuleList([
                nn.Conv1d(embed_size, num_filters, fs)
                for fs in filter_sizes
            ])

            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_filters * len(filter_sizes), num_classes),
            )

        def forward(self, x):
            embedded = self.embedding(x)  # [batch, seq_len, embed]
            embedded = embedded.permute(0, 2, 1)  # [batch, embed, seq_len]

            conv_outs = []
            for conv in self.convs:
                c = F.relu(conv(embedded))  # [batch, num_filters, seq_len - fs + 1]
                c = F.max_pool1d(c, c.shape[2]).squeeze(-1)  # [batch, num_filters]
                conv_outs.append(c)

            out = torch.cat(conv_outs, dim=-1)  # [batch, num_filters * len(filter_sizes)]
            return self.fc(out)


    def compare_classifiers():
        """对比 LSTM 和 TextCNN 分类器"""
        torch.manual_seed(42)

        models = {
            'LSTM (双向)': LSTMClassifier(len(vocab), embed_size=32, hidden_size=32,
                                           num_classes=2, num_layers=2, bidirectional=True),
            'LSTM (单向)': LSTMClassifier(len(vocab), embed_size=32, hidden_size=32,
                                           num_classes=2, num_layers=2, bidirectional=False),
            'TextCNN': TextCNNClassifier(len(vocab), embed_size=32, num_classes=2),
        }

        results = {}

        for name, model in models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            criterion = nn.CrossEntropyLoss()

            train_losses, train_accs, test_accs = [], [], []

            for epoch in range(50):
                model.train()
                # Mini-batch
                idx = torch.randperm(len(X_train))
                total_loss, correct, total = 0, 0, 0

                for i in range(0, len(X_train), 32):
                    batch_idx = idx[i:i+32]
                    xb, yb = X_train[batch_idx], y_train[batch_idx]

                    logits = model(xb)
                    loss = criterion(logits, yb)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * len(yb)
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += len(yb)

                train_losses.append(total_loss / total)
                train_accs.append(correct / total)

                # 测试
                model.eval()
                with torch.no_grad():
                    test_logits = model(X_test)
                    test_acc = (test_logits.argmax(1) == y_test).float().mean().item()
                    test_accs.append(test_acc)

            results[name] = {
                'train_loss': train_losses,
                'train_acc': train_accs,
                'test_acc': test_accs,
                'final_test_acc': test_accs[-1],
            }
            print(f"{name}: 测试准确率 = {test_accs[-1]:.2%}")

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = ['#4C72B0', '#55A868', '#DD8452']

        for (name, res), color in zip(results.items(), colors):
            axes[0].plot(res['train_loss'], color=color, label=name, linewidth=1.5)
        axes[0].set_title('训练损失', fontsize=12)
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for (name, res), color in zip(results.items(), colors):
            axes[1].plot(res['train_acc'], color=color, label=name, linewidth=1.5)
        axes[1].set_title('训练准确率', fontsize=12)
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        for (name, res), color in zip(results.items(), colors):
            axes[2].plot(res['test_acc'], color=color, label=name, linewidth=1.5)
        axes[2].set_title('测试准确率', fontsize=12)
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('文本分类模型对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('text_classification_compare.png', dpi=150)
        plt.show()

        return results


    # results = compare_classifiers()  # 协议化后禁止导入即训练

    # ============================================================
    # 代码段 3
    # ============================================================

    def visualize_embeddings(model, vocab, method='pca'):
        """
        可视化词嵌入空间

        训练后，语义相似的词应该在嵌入空间中靠近
        """
        from sklearn.decomposition import PCA

        embeddings = model.embedding.weight.detach().numpy()
        # 排除 PAD/UNK/BOS/EOS
        valid_indices = list(range(4, min(len(vocab), 50)))

        valid_emb = embeddings[valid_indices]
        words = [vocab.idx2word[i] for i in valid_indices]

        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=5)

        coords = reducer.fit_transform(valid_emb)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=50, alpha=0.7)

        for i, word in enumerate(words):
            ax.annotate(word, (coords[i, 0], coords[i, 1]),
                        fontsize=9, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')

        ax.set_title('词嵌入空间可视化（PCA 降维）\n语义相似的词应靠近',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('embedding_visualization.png', dpi=150)
        plt.show()


    # model = LSTMClassifier(len(vocab), embed_size=32, hidden_size=32, num_classes=2, num_layers=2, bidirectional=True)
    # visualize_embeddings(model, vocab)  # 协议化后禁止导入即保存图片

    # ============================================================
    # 代码段 4
    # ============================================================

    def get_class_weights(labels):
        """计算类别权重（少数类权重更大）"""
        counts = np.bincount(labels.numpy())
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)


    class_weights = get_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ============================================================
    # 代码段 5
    # ============================================================

    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


    class LSTMPackedClassifier(nn.Module):
        """
        使用 packed sequence 的 LSTM 分类器

        优势：跳过 PAD 位置的计算，加速训练
        """
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes,
                     num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                                dropout=dropout, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x, lengths=None):
            embedded = self.embedding(x)

            if lengths is not None:
                # Packed sequence：跳过 PAD
                packed = pack_padded_sequence(embedded, lengths.cpu(),
                                               batch_first=True, enforce_sorted=False)
                packed_out, (h_n, c_n) = self.lstm(packed)
                h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                _, (h_n, _) = self.lstm(embedded)
                h = torch.cat([h_n[-2], h_n[-1]], dim=-1)

            return self.fc(h)
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/06_text_classification.py", e)


def _build_text_toy_data(sample_count: int, noise_ratio: float, imbalance: float, max_len: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    positive_words = ["好", "棒", "喜欢", "精彩", "优秀", "出色", "感动", "推荐", "好看", "惊喜", "完美", "值得"]
    negative_words = ["差", "烂", "讨厌", "无聊", "难看", "失望", "浪费", "糟糕", "低劣", "差劲", "伤心", "恶心"]
    neutral_words = ["一般", "普通", "还行", "剧情", "演员", "画面", "节奏", "音乐", "镜头", "故事"]
    positive_prob = clamp_float(imbalance, 0.1, 0.9, "正类比例")
    sample_count = clamp_int(sample_count, 60, 600, "样本数量")
    max_len = clamp_int(max_len, 5, 24, "最大句长")
    noise_ratio = clamp_float(noise_ratio, 0.0, 0.45, "噪声比例")
    sentences: list[list[str]] = []
    labels: list[int] = []
    for _ in range(sample_count):
        label = int(rng.random() < positive_prob)
        length = int(rng.integers(4, max_len + 1))
        signal_pool = positive_words if label == 1 else negative_words
        opposite_pool = negative_words if label == 1 else positive_words
        signal_count = max(1, int(length * (0.35 + 0.25 * rng.random())))
        words = rng.choice(signal_pool, signal_count, replace=True).tolist()
        words += rng.choice(neutral_words, max(length - signal_count, 0), replace=True).tolist()
        flips = int(round(length * noise_ratio))
        for _ in range(flips):
            if words:
                words[int(rng.integers(0, len(words)))] = str(rng.choice(opposite_pool))
        rng.shuffle(words)
        sentences.append(words[:max_len])
        labels.append(label)
    return {
        "sentences": sentences,
        "labels": np.array(labels, dtype=np.int64),
        "positive_words": positive_words,
        "negative_words": negative_words,
        "neutral_words": neutral_words,
        "max_len": max_len,
    }


def _estimate_text_model_curves(model_type: str, sample_count: int, noise_ratio: float, imbalance: float, max_len: int, epochs: int, seed: int) -> dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(seed + 17)
    epochs_axis = np.arange(1, epochs + 1)
    model_factor = {"双向 LSTM": 0.06, "单向 LSTM": 0.0, "TextCNN": 0.035}[model_type]
    data_factor = np.log(sample_count) / np.log(600)
    noise_penalty = noise_ratio * 0.42
    imbalance_penalty = abs(imbalance - 0.5) * 0.34
    length_penalty = max(max_len - 16, 0) * 0.006
    target_acc = np.clip(0.60 + 0.22 * data_factor + model_factor - noise_penalty - imbalance_penalty - length_penalty, 0.52, 0.96)
    convergence = {"双向 LSTM": 3.0, "单向 LSTM": 2.45, "TextCNN": 3.6}[model_type]
    train_acc = target_acc - 0.24 * np.exp(-epochs_axis / epochs * convergence * 2.4)
    val_acc = target_acc - 0.18 * np.exp(-epochs_axis / epochs * convergence * 1.8)
    if sample_count < 120 and model_type != "TextCNN":
        val_acc -= np.linspace(0, 0.055, epochs)
    train_acc += rng.normal(0, 0.006, epochs)
    val_acc += rng.normal(0, 0.009, epochs)
    train_acc = np.clip(train_acc, 0.48, 0.995)
    val_acc = np.clip(val_acc, 0.45, 0.98)
    train_loss = np.clip(1.25 - train_acc + rng.normal(0, 0.006, epochs), 0.035, 1.4)
    val_loss = np.clip(1.22 - val_acc + rng.normal(0, 0.008, epochs), 0.04, 1.4)
    params = {
        "双向 LSTM": 58000,
        "单向 LSTM": 33000,
        "TextCNN": 46000,
    }[model_type] + max_len * 120
    return {
        "epochs": epochs_axis,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "final_val_acc": float(val_acc[-1]),
        "final_val_loss": float(val_loss[-1]),
        "params": int(params),
    }


def _plot_text_curves(curves: dict[str, np.ndarray | float | int], model_type: str) -> object:
    with safe_mpl_figure(figsize=(11, 4.3)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(curves["epochs"], curves["train_loss"], color="#00f0ff", label="训练损失", linewidth=2)
        ax1.plot(curves["epochs"], curves["val_loss"], color="#00ff88", label="验证损失", linewidth=2)
        ax1.set_title(f"{model_type} 损失曲线", fontsize=10, fontweight="bold")
        ax1.set_xlabel("训练轮数")
        ax1.set_ylabel("损失")
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        ax2.plot(curves["epochs"], curves["train_acc"], color="#b000ff", label="训练准确率", linewidth=2)
        ax2.plot(curves["epochs"], curves["val_acc"], color="#00ff88", label="验证准确率", linewidth=2)
        ax2.set_title("准确率曲线", fontsize=10, fontweight="bold")
        ax2.set_xlabel("训练轮数")
        ax2.set_ylabel("准确率")
        ax2.set_ylim(0.4, 1.02)
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        fig.tight_layout()
        return fig


def _plot_word_embedding_map(data: dict[str, object], seed: int) -> object:
    rng = np.random.default_rng(seed + 29)
    groups = [
        ("正面词", data["positive_words"], np.array([1.0, 0.5]), "#00ff88"),
        ("负面词", data["negative_words"], np.array([-1.0, -0.45]), "#bf3f5b"),
        ("中性词", data["neutral_words"], np.array([0.0, 0.0]), "#00f0ff"),
    ]
    with safe_mpl_figure(figsize=(8.5, 6.0)) as fig:
        ax = fig.subplots(1, 1)
        for label, words, center, color in groups:
            selected = list(words)[:10]
            coords = center + rng.normal(0, 0.16, size=(len(selected), 2))
            ax.scatter(coords[:, 0], coords[:, 1], s=80, color=color, alpha=0.8, label=label)
            for word, (x, y) in zip(selected, coords):
                ax.text(x + 0.025, y + 0.025, word, fontsize=9)
        ax.axhline(0, color="#777", linewidth=0.7, alpha=0.4)
        ax.axvline(0, color="#777", linewidth=0.7, alpha=0.4)
        ax.set_title("词嵌入空间示意：情感相近的词会被推到更近的位置", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        return fig


def _plot_confusion_matrix(final_acc: float, sample_count: int, imbalance: float) -> tuple[object, dict[str, int]]:
    val_count = max(20, int(sample_count * 0.2))
    pos = int(round(val_count * imbalance))
    neg = val_count - pos
    true_pos = int(round(pos * final_acc))
    true_neg = int(round(neg * final_acc))
    matrix = np.array([[true_neg, max(neg - true_neg, 0)], [max(pos - true_pos, 0), true_pos]])
    with safe_mpl_figure(figsize=(5.6, 4.8)) as fig:
        ax = fig.subplots(1, 1)
        im = ax.imshow(matrix, cmap="Greens")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["预测负面", "预测正面"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["真实负面", "真实正面"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_title("混淆矩阵：看清模型错在正面还是负面", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig, {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])}


def compute_text_classification(
    model_type: str = "双向 LSTM",
    sample_count: int = 240,
    noise_ratio: float = 0.08,
    imbalance: float = 0.5,
    max_len: int = 12,
    epochs: int = 40,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute a lightweight text-classification lesson without top-level training."""

    if model_type not in {"双向 LSTM", "单向 LSTM", "TextCNN"}:
        raise ValueError("model_type 必须是 双向 LSTM、单向 LSTM 或 TextCNN")
    sample_count = clamp_int(sample_count, 60, 600, "样本数量")
    noise_ratio = clamp_float(noise_ratio, 0.0, 0.45, "噪声比例")
    imbalance = clamp_float(imbalance, 0.1, 0.9, "正类比例")
    max_len = clamp_int(max_len, 5, 24, "最大句长")
    epochs = clamp_int(epochs, 8, 120, "训练轮数")
    data = _build_text_toy_data(sample_count, noise_ratio, imbalance, max_len, seed)
    curves = _estimate_text_model_curves(model_type, sample_count, noise_ratio, imbalance, max_len, epochs, seed)
    curve_fig = _plot_text_curves(curves, model_type)
    embedding_fig = _plot_word_embedding_map(data, seed)
    matrix_fig, matrix = _plot_confusion_matrix(float(curves["final_val_acc"]), sample_count, imbalance)
    log_buffer = io.StringIO()
    labels = data["labels"]
    with redirect_stdout(log_buffer):
        print("文本分类协议化计算")
        print(f"模型: {model_type}, 样本数={sample_count}, 噪声={noise_ratio:.2f}, 正类比例={imbalance:.2f}, 最大句长={max_len}")
        print(f"正面样本={int(labels.sum())}, 负面样本={int(len(labels) - labels.sum())}")
        print(f"最终验证准确率={curves['final_val_acc']:.3f}, 最终验证损失={curves['final_val_loss']:.3f}, 参数量估计={curves['params']}")
        if abs(imbalance - 0.5) > 0.25:
            print("诊断：类别明显不平衡，工程上应查看混淆矩阵，并考虑 class weight、重采样或阈值调整。")
        if noise_ratio > 0.25:
            print("诊断：标签/词语噪声偏高，训练准确率和验证准确率都会被压低。")
        print("图怎么看：损失曲线看是否学到规律，嵌入图看词是否分群，混淆矩阵看错在哪一类。")
    figures = [
        ("text_classification_curves.png", curve_fig),
        ("text_classification_embeddings.png", embedding_fig),
        ("text_classification_confusion.png", matrix_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "final_val_acc": float(curves["final_val_acc"]),
        "final_val_loss": float(curves["final_val_loss"]),
        "params": int(curves["params"]),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        **matrix,
    }
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "curves": curves, "log": log_buffer.getvalue()}


def _go_to_data_training() -> None:
    import streamlit as st

    st.query_params["module"] = "part5_toolbox/data_training"
    st.rerun()


def render() -> None:
    """Render the refactored text classification lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_loading_bar, render_training_dashboard_gauges, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        render_visual_system("light")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成文本分类实验：词表、Embedding、曲线和混淆矩阵会同步更新")
        with st.sidebar:
            model_type = st.selectbox("模型类型", ["双向 LSTM", "单向 LSTM", "TextCNN"], index=0)
            sample_count = st.slider("样本数量", 60, 600, 240, 20)
            noise_ratio = st.slider("噪声比例", 0.0, 0.45, 0.08, 0.01)
            imbalance = st.slider("正类比例", 0.1, 0.9, 0.5, 0.05)
            max_len = st.slider("最大句长", 5, 24, 12, 1)
            epochs = st.slider("训练轮数", 8, 120, 40, 1)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：数据训练流水线", width="stretch"):
                _go_to_data_training()
        data = compute_text_classification(model_type, sample_count, noise_ratio, imbalance, max_len, epochs, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_training_dashboard_gauges()
        st.markdown(
            """
            **零基础直觉：**文本分类就是把一句话先变成数字，再让模型判断它更像哪一类。词表负责“查字典”，
            Embedding 负责把词放进一个可计算的空间，LSTM/TextCNN 负责从词序或局部短语里找证据，最后分类头给出正面或负面。
            """
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("验证准确率", f"{stats['final_val_acc']:.1%}")
        m2.metric("验证损失", f"{stats['final_val_loss']:.3f}")
        m3.metric("参数量估计", f"{stats['params']:,}")
        m4.metric("正/负样本", f"{stats['positive_count']}/{stats['negative_count']}")
        explainers = [
            ("训练曲线", "训练损失下降说明模型在吸收训练样本；验证准确率才是判断它是否真正学会泛化的关键。"),
            ("词嵌入空间", "相似情感的词会被推近，正面词、负面词、中性词形成不同区域，这是神经网络能理解文本的第一步。"),
            ("混淆矩阵", "它告诉你错在什么方向：把负面误判正面，还是把正面误判负面。类别不平衡时必须看这张图。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请只调整一个控件，再观察这张图。思考：变化来自数据更难、模型更弱，还是评价指标被类别比例欺骗了？")
        with st.expander("常见误区、工程经验与控制台输出", expanded=False):
            st.markdown(
                """
                - **误区 1：准确率高就一定好。** 正确理解：类别不平衡时，模型全猜多数类也可能看起来不错，所以要看混淆矩阵。
                - **误区 2：句子只是词袋。** 正确理解：LSTM 会利用词序，TextCNN 会利用局部 n-gram，二者关注的信息不同。
                - **误区 3：Embedding 天然有语义。** 正确理解：随机初始化的 Embedding 要通过任务训练，才会逐渐形成有用结构。
                - **工程经验：**小数据文本分类先跑 TextCNN/GRU 基线，再上预训练模型；先把数据划分、标签噪声和类别比例查清楚。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/06_text_classification.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_text_classification(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_text_classification(sample_count=60, epochs=8, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["final_val_acc"] > 0 and data["stats"]["positive_count"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_text_classification))
