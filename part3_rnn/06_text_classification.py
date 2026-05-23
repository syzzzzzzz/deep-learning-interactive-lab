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


    # 构建数据
    sentences, labels = create_sentiment_data()
    vocab = Vocabulary(min_freq=1)
    vocab.build(sentences)

    # 编码并创建张量
    max_len = 12
    encoded = [vocab.encode(s, max_len) for s in sentences]
    X = torch.tensor(encoded, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    # 分割训练/测试
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"正面: {y.sum().item()}, 负面: {len(y) - y.sum().item()}")

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


    results = compare_classifiers()

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


    # 使用 LSTM 模型可视化
    model = LSTMClassifier(len(vocab), embed_size=32, hidden_size=32,
                            num_classes=2, num_layers=2, bidirectional=True)
    visualize_embeddings(model, vocab)

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
