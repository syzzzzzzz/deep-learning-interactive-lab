"""Seq2Seq 与注意力：从固定上下文瓶颈到可解释对齐矩阵。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "Seq2Seq 与注意力"
MODULE_SUMMARY = "用对齐热力图、长序列性能曲线和上下文瓶颈示意解释编码器-解码器为什么需要注意力。"
MODULE_TAGS = ["RNN", "Seq2Seq", "注意力", "机器翻译", "对齐"]
MODULE_RELATED_TOPICS = ["part3/03_sequence_toys", "part3/04_hyperparam_rnn", "part4/01_attention_mechanism", "part4/transformer_models"]
PRACTICE_TARGET = "切换源句子、目标词和注意力锐度，解释每个输出词为什么应该看向不同输入位置。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


class Encoder(nn.Module):
    """
    Seq2Seq 编码器

    将输入序列编码为隐藏状态序列
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)

    def forward(self, x):
        """
        x: [batch, src_len] 输入 token 索引
        返回: encoder_outputs [batch, src_len, hidden],
              hidden (h_n, c_n)
        """
        embedded = self.embedding(x)  # [batch, src_len, embed]
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden


class DecoderWithoutAttention(nn.Module):
    """
    基础解码器（无注意力）

    每步只依赖上一步输出和上下文向量 c
    c = 编码器最后一步的隐藏状态
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入：embedding + 上下文向量
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, context=None):
        """
        x: [batch, 1] 当前输入 token
        hidden: 编码器的隐藏状态
        context: 上下文向量（用 hidden 代替）
        """
        embedded = self.embedding(x)  # [batch, 1, embed]
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output.squeeze(1))  # [batch, vocab_size]
        return logits, hidden


class Seq2SeqBasic(nn.Module):
    """
    基础 Seq2Seq（无注意力）

# 训练：使用 teacher forcing
    推理：自回归生成
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=64,
                 hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderWithoutAttention(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]

        # 编码
        _, hidden = self.encoder(src)

        # 解码
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size)
        input_token = tgt[:, 0:1]  # <SOS> token

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = logits

            # teacher forcing：以一定概率使用真实标签作为下一步输入
            if np.random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = logits.argmax(dim=1, keepdim=True)

        return outputs

# ============================================================
# 代码段 2
# ============================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau（加性）注意力

    score(h_t, h_s) = v^T · tanh(W_1 · h_s + W_2 · h_t)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch, hidden] 解码器当前隐藏状态
        encoder_outputs: [batch, src_len, hidden] 编码器所有输出
        返回: context [batch, hidden], attention_weights [batch, src_len]
        """
        # 计算对齐分数
        # W1(h_s): [batch, src_len, hidden]
        src_energy = self.W1(encoder_outputs)
        # W2(h_t): [batch, 1, hidden] → 扩展到 src_len
        tgt_energy = self.W2(hidden).unsqueeze(1)

        # 加性得分
        energy = self.v(torch.tanh(src_energy + tgt_energy)).squeeze(-1)
        # energy: [batch, src_len]

        # softmax 归一化
        attention_weights = F.softmax(energy, dim=-1)

        # 加权求和
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch, hidden]

        return context, attention_weights


class DecoderWithBahdanau(nn.Module):
    """带 Bahdanau 注意力的解码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入维度 = embed_size + hidden_size（拼接上下文向量）
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.attention = BahdanauAttention(hidden_size)
        # 输出层：hidden + context → vocab
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        """
        x: [batch, 1] 当前输入 token
        hidden: (h_n, c_n) 解码器隐藏状态
        encoder_outputs: [batch, src_len, hidden]
        """
        embedded = self.embedding(x)  # [batch, 1, embed]

        # 计算注意力
        h_t = hidden[0][-1]  # 最后一层的 h: [batch, hidden]
        context, attn_weights = self.attention(h_t, encoder_outputs)

        # 拼接 embedding 和上下文向量
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        # rnn_input: [batch, 1, embed + hidden]

        output, hidden = self.lstm(rnn_input, hidden)

        # 拼接输出和上下文向量
        output = torch.cat([output.squeeze(1), context], dim=-1)
        logits = self.fc(output)  # [batch, vocab]

        return logits, hidden, attn_weights

# ============================================================
# 代码段 3
# ============================================================

class LuongAttention(nn.Module):
    """
    Luong（乘性）注意力

    支持 general 和 dot 两种打分方式
    general: score = h_t^T · W · h_s
    dot:     score = h_t^T · h_s  (要求 h_t 和 h_s 维度相同)
    """
    def __init__(self, hidden_size, method='general'):
        super().__init__()
        assert method in ['dot', 'general', 'concat']
        self.method = method
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch, hidden]
        encoder_outputs: [batch, src_len, hidden]
        """
        if self.method == 'dot':
            # h_t^T · h_s
            energy = torch.bmm(encoder_outputs,
                               hidden.unsqueeze(-1)).squeeze(-1)
        elif self.method == 'general':
            # h_t^T · W · h_s
            energy = torch.bmm(self.W(encoder_outputs),
                               hidden.unsqueeze(-1)).squeeze(-1)

        attention_weights = F.softmax(energy, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)

        return context, attention_weights

# ============================================================
# 代码段 4
# ============================================================

class Seq2SeqWithAttention(nn.Module):
    """
    带注意力的 Seq2Seq

    支持 Bahdanau 和 Luong 注意力
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=64,
                 hidden_size=128, num_layers=2, dropout=0.1,
                 attention_type='bahdanau'):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.attention_type = attention_type
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size

        if attention_type == 'bahdanau':
            self.decoder = DecoderWithBahdanau(tgt_vocab_size, embed_size, hidden_size,
                                                num_layers, dropout)
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]

        encoder_outputs, hidden = self.encoder(src)
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size)
        attention_maps = torch.zeros(batch_size, tgt_len, src.shape[1])

        input_token = tgt[:, 0:1]

        for t in range(1, tgt_len):
            logits, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs
            )
            outputs[:, t] = logits
            attention_maps[:, t] = attn_weights

            if np.random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = logits.argmax(dim=1, keepdim=True)

        return outputs, attention_maps

    def translate(self, src, max_len=50, sos_idx=1, eos_idx=2):
        """推理：自回归翻译"""
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src)
            input_token = torch.full((src.shape[0], 1), sos_idx, dtype=torch.long)

            generated = []
            attention_maps = []

            for _ in range(max_len):
                logits, hidden, attn_weights = self.decoder(
                    input_token, hidden, encoder_outputs
                )
                next_token = logits.argmax(dim=1, keepdim=True)
                generated.append(next_token)
                attention_maps.append(attn_weights)

                if (next_token == eos_idx).all():
                    break
                input_token = next_token

        return torch.cat(generated, dim=1), torch.stack(attention_maps, dim=1)

# ============================================================
# 代码段 5
# ============================================================

def visualize_attention(attention_weights, src_tokens, tgt_tokens):
    """
    可视化注意力对齐矩阵

    attention_weights: [tgt_len, src_len]
    src_tokens: 源语言 token 列表
    tgt_tokens: 目标语言 token 列表
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=10)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=10)

    # 在每个格子中标注权重值
    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            val = attention_weights[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if val > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='注意力权重')
    ax.set_xlabel('源语言 (输入)')
    ax.set_ylabel('目标语言 (输出)')
    ax.set_title('注意力对齐矩阵\n（每个输出词关注哪些输入词）',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('attention_alignment.png', dpi=150)
    plt.show()


# 模拟翻译注意力（示例）
def demo_attention_visualization():
    """用模拟数据展示注意力对齐"""
    # 中译英示例的模拟注意力权重
    src = ['我', '爱', '深', '度', '学', '习']
    tgt = ['I', 'love', 'deep', 'learning']

    # 模拟：每个英文词主要关注对应的中文词
    np.random.seed(42)
    attn = np.array([
        [0.8, 0.05, 0.02, 0.02, 0.02, 0.09],  # I → 我
        [0.05, 0.85, 0.02, 0.02, 0.03, 0.03],  # love → 爱
        [0.02, 0.02, 0.4, 0.4, 0.1, 0.06],     # deep → 深+度
        [0.02, 0.02, 0.05, 0.05, 0.4, 0.46],   # learning → 学+习
    ])

    visualize_attention(attn, src, tgt)

# demo_attention_visualization()  # 协议化后由 render()/compute_seq2seq_attention() 控制执行

# ============================================================
# 代码段 6
# ============================================================

def compare_with_without_attention():
    """
    对比有无注意力的 Seq2Seq 在不同序列长度下的表现

    典型结果：
    短序列（<10）：两者差不多
    长序列（>20）：注意力显著优于无注意力
    """
    seq_lengths = [5, 10, 20, 40, 80]

    # 模拟不同长度的性能曲线
    # 基于典型实验结果的示意数据
    perf_no_attn = [0.95, 0.88, 0.72, 0.55, 0.40]
    perf_bahdanau = [0.96, 0.93, 0.87, 0.78, 0.68]
    perf_luong = [0.95, 0.92, 0.86, 0.76, 0.65]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(seq_lengths, perf_no_attn, 'r-o', label='无注意力', linewidth=2)
    axes[0].plot(seq_lengths, perf_bahdanau, 'b-s', label='Bahdanau', linewidth=2)
    axes[0].plot(seq_lengths, perf_luong, 'g-^', label='Luong', linewidth=2)
    axes[0].set_xlabel('输入序列长度')
    axes[0].set_ylabel('BLEU 分数')
    axes[0].set_title('序列长度 vs 翻译质量\n（注意力解决长序列瓶颈）',
                       fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 信息瓶颈示意
    axes[1].barh(['无注意力', 'Bahdanau', 'Luong'],
                  [1, 40, 40],
                  color=['#C44E52', '#4C72B0', '#55A868'], alpha=0.85)
    axes[1].set_xlabel('上下文信息量（相对值）')
    axes[1].set_title('上下文向量信息量对比\n（无注意力=固定向量瓶颈）',
                       fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('注意力机制：Seq2Seq 的关键改进', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('attention_vs_no_attention.png', dpi=150)
    plt.show()


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sentence_pairs() -> dict[str, tuple[list[str], list[str], np.ndarray]]:
    return {
        "我 爱 深度 学习": (
            ["我", "爱", "深", "度", "学", "习"],
            ["I", "love", "deep", "learning"],
            np.array(
                [
                    [3.4, 0.2, -0.5, -0.5, -0.6, -0.1],
                    [0.1, 3.6, -0.4, -0.4, -0.2, -0.2],
                    [-0.4, -0.4, 2.2, 2.1, 0.4, 0.1],
                    [-0.4, -0.4, 0.0, 0.1, 2.0, 2.3],
                ]
            ),
        ),
        "机器 翻译 需要 上下文": (
            ["机器", "翻译", "需要", "上下文"],
            ["machine", "translation", "needs", "context"],
            np.array(
                [
                    [3.2, 1.0, -0.2, -0.5],
                    [0.7, 3.3, -0.3, -0.5],
                    [-0.5, -0.2, 3.0, 0.4],
                    [-0.4, -0.3, 0.3, 3.4],
                ]
            ),
        ),
        "今天 天气 很 适合 学习": (
            ["今天", "天气", "很", "适合", "学习"],
            ["today", "weather", "is", "good", "for", "study"],
            np.array(
                [
                    [3.3, 0.2, -0.5, -0.4, -0.5],
                    [0.0, 3.1, -0.2, -0.4, -0.5],
                    [-0.2, 1.0, 1.6, 0.5, -0.3],
                    [-0.5, -0.2, 0.8, 2.4, 0.2],
                    [-0.5, -0.4, 0.1, 2.1, 0.7],
                    [-0.5, -0.4, -0.2, 0.4, 3.2],
                ]
            ),
        ),
    }


def _compute_alignment(pair_name: str, sharpness: float) -> tuple[list[str], list[str], np.ndarray]:
    pairs = _sentence_pairs()
    if pair_name not in pairs:
        raise ValueError(f"未知句子: {pair_name}")
    src_tokens, tgt_tokens, scores = pairs[pair_name]
    sharpness = clamp_float(sharpness, 0.35, 3.0, "注意力锐度")
    weights = _softmax_numpy(scores * sharpness)
    return src_tokens, tgt_tokens, weights


def _plot_alignment(weights: np.ndarray, src_tokens: list[str], tgt_tokens: list[str], selected_target: int) -> object:
    with safe_mpl_figure(figsize=(9.5, 5.8)) as fig:
        ax = fig.subplots(1, 1)
        im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="注意力权重")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, fontsize=10)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=10)
        ax.axhline(selected_target - 0.5, color="#00f0ff", linewidth=2)
        ax.axhline(selected_target + 0.5, color="#00f0ff", linewidth=2)
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                val = weights[i, j]
                if val > 0.03:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val > 0.55 else "black")
        ax.set_xlabel("源序列：编码器记住的每个输入词")
        ax.set_ylabel("目标序列：解码器正在生成的词")
        ax.set_title("Seq2Seq 注意力对齐矩阵：一行解释一个输出词在看谁", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_length_comparison(max_length: int) -> tuple[object, dict[str, float]]:
    lengths = np.array([5, 10, 20, 40, 80])
    max_length = clamp_int(max_length, 10, 80, "最大序列长度")
    visible = lengths <= max_length
    visible_lengths = lengths[visible]
    no_attention = 0.97 * np.exp(-visible_lengths / 96) - np.maximum(visible_lengths - 12, 0) / 150
    bahdanau = 0.97 * np.exp(-visible_lengths / 260) - np.maximum(visible_lengths - 55, 0) / 500
    luong = 0.96 * np.exp(-visible_lengths / 240) - np.maximum(visible_lengths - 60, 0) / 520
    no_attention = np.clip(no_attention, 0.20, 0.98)
    bahdanau = np.clip(bahdanau, 0.35, 0.98)
    luong = np.clip(luong, 0.35, 0.98)
    with safe_mpl_figure(figsize=(10, 4.3)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(visible_lengths, no_attention, "o-", label="无注意力", color="#bf3f5b", linewidth=2)
        ax1.plot(visible_lengths, bahdanau, "s-", label="Bahdanau 加性注意力", color="#00f0ff", linewidth=2)
        ax1.plot(visible_lengths, luong, "^-", label="Luong 乘性注意力", color="#00ff88", linewidth=2)
        ax1.set_title("输入越长，固定上下文越吃力", fontsize=10, fontweight="bold")
        ax1.set_xlabel("输入序列长度")
        ax1.set_ylabel("教学化质量分数")
        ax1.set_ylim(0.15, 1.02)
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)
        ax2.barh(["固定向量", "注意力读取"], [1, int(max_length)], color=["#bf3f5b", "#00ff88"], alpha=0.88)
        ax2.set_title("上下文信息通道对比", fontsize=10, fontweight="bold")
        ax2.set_xlabel("可访问的信息位置数")
        ax2.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        stats = {
            "no_attention_last": float(no_attention[-1]),
            "bahdanau_last": float(bahdanau[-1]),
            "luong_last": float(luong[-1]),
        }
        return fig, stats


def _plot_context_flow(src_tokens: list[str], tgt_tokens: list[str], weights: np.ndarray, selected_target: int) -> object:
    selected = weights[selected_target]
    with safe_mpl_figure(figsize=(10, 3.8)) as fig:
        ax = fig.subplots(1, 1)
        x_src = np.linspace(0.1, 0.9, len(src_tokens))
        y_src = np.full(len(src_tokens), 0.72)
        x_tgt = 0.5
        y_tgt = 0.18
        for x, y, token, weight in zip(x_src, y_src, src_tokens, selected):
            ax.scatter([x], [y], s=900 * (0.35 + weight), color="#1f77b4", alpha=0.9)
            ax.text(x, y, token, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            ax.plot([x, x_tgt], [y - 0.04, y_tgt + 0.06], color="#00f0ff", linewidth=1.0 + 6 * weight, alpha=0.18 + 0.75 * weight)
        ax.scatter([x_tgt], [y_tgt], s=1100, color="#b000ff", alpha=0.9)
        ax.text(x_tgt, y_tgt, tgt_tokens[selected_target], ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.02, "线越亮、越粗，表示当前输出词从该输入词取走的信息越多", ha="center", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("上下文读取路径：解码器不是只看最后一步，而是按权重回看所有输入", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig


def compute_seq2seq_attention(
    pair_name: str = "我 爱 深度 学习",
    target_index: int = 0,
    sharpness: float = 1.2,
    max_length: int = 80,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute Seq2Seq attention teaching visuals without import-time side effects."""

    np.random.seed(seed)
    src_tokens, tgt_tokens, weights = _compute_alignment(pair_name, sharpness)
    target_index = clamp_int(int(target_index), 0, len(tgt_tokens) - 1, "目标词编号")
    log_buffer = io.StringIO()
    selected_weights = weights[target_index]
    strongest_index = int(np.argmax(selected_weights))
    entropy = float(-(selected_weights * np.log(selected_weights + 1e-12)).sum())
    with redirect_stdout(log_buffer):
        print("Seq2Seq 注意力协议化计算")
        print(f"源序列: {' '.join(src_tokens)}")
        print(f"目标词: {tgt_tokens[target_index]}")
        print(f"最关注输入词: {src_tokens[strongest_index]}，权重={selected_weights[strongest_index]:.3f}")
        print(f"注意力熵: {entropy:.3f}；熵越低代表越集中，熵越高代表同时参考多个词。")
        print("公式: context_t = Σ_s attention(t,s) * encoder_output_s")
    align_fig = _plot_alignment(weights, src_tokens, tgt_tokens, target_index)
    flow_fig = _plot_context_flow(src_tokens, tgt_tokens, weights, target_index)
    length_fig, length_stats = _plot_length_comparison(max_length)
    figures = [
        ("seq2seq_attention_alignment.png", align_fig),
        ("seq2seq_attention_context_flow.png", flow_fig),
        ("seq2seq_attention_length_comparison.png", length_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "selected_target": tgt_tokens[target_index],
        "strongest_source": src_tokens[strongest_index],
        "strongest_weight": float(selected_weights[strongest_index]),
        "attention_entropy": entropy,
        **length_stats,
    }
    return {
        "figures": figures,
        "artifacts": artifacts,
        "stats": stats,
        "weights": weights,
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "log": log_buffer.getvalue(),
    }


def _go_to_transformer_attention() -> None:
    import streamlit as st

    st.query_params["module"] = "part4_transformer/01_attention_mechanism"
    st.rerun()


def render() -> None:
    """Render the refactored Seq2Seq attention lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_attention_light_beams, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成对齐矩阵、上下文读取路径和长序列对比")
        pairs = _sentence_pairs()
        with st.sidebar:
            pair_name = st.selectbox("源句子", list(pairs.keys()))
            target_tokens = pairs[pair_name][1]
            target_word = st.selectbox("选择目标词", target_tokens)
            target_index = target_tokens.index(target_word)
            sharpness = st.slider("注意力锐度", 0.35, 3.0, 1.2, 0.05)
            max_length = st.slider("最大序列长度", 10, 80, 80, 5)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("继续看：Transformer 注意力", width="stretch"):
                _go_to_transformer_attention()

        data = compute_seq2seq_attention(pair_name, target_index, sharpness, max_length, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_attention_light_beams()
        st.markdown(
            """
            **零基础直觉：**没有注意力的 Seq2Seq 像让一个人读完整篇文章后，只允许他用一句压缩笔记回答所有问题。
            带注意力的 Seq2Seq 则允许他在回答每个词时重新回看原文。热力图中的一行就是一次“回看”：颜色越深，
            表示当前输出词越依赖那个输入词。
            """
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("当前目标词", str(stats["selected_target"]))
        k2.metric("最关注输入词", str(stats["strongest_source"]))
        k3.metric("最大权重", f"{stats['strongest_weight']:.2f}")
        k4.metric("注意力熵", f"{stats['attention_entropy']:.2f}")
        explainers = [
            (
                "对齐热力图",
                "横轴是输入词，纵轴是输出词；每一行加起来等于 1。深色格子说明当前输出词主要从这个输入位置取信息。",
            ),
            (
                "上下文读取路径",
                "线条越粗越亮，代表当前目标词从对应源词拿走的信息越多。它把矩阵里的一行变成更像人能看懂的流向图。",
            ),
            (
                "长序列瓶颈对比",
                "无注意力模型只依赖固定长度上下文，输入越长越容易漏信息；注意力可以逐步回看所有输入位置，因此下降更慢。",
            ),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请改变“选择目标词”或“注意力锐度”，观察热力图中哪一行改变最大。思考：翻译一个词时，模型为什么不应该总盯着同一个输入词？")
        with st.expander("数学、误区与控制台输出", expanded=False):
            st.markdown(
                r"""
                核心计算可以写成：

                \[
                \alpha_{t,s}=\mathrm{softmax}(\mathrm{score}(h_t,h_s)),\quad
                c_t=\sum_s \alpha_{t,s}h_s
                \]

                **误区 1：注意力就是最终解释。** 正确理解：权重能提供线索，但不是完整因果解释。
                **误区 2：越尖锐越好。** 正确理解：翻译复合词时常常需要同时看多个输入词。
                **工程经验：**长序列任务先检查注意力是否塌缩到固定位置；如果塌缩，常见原因是学习率过大、mask 错误或位置处理不合理。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/05_seq2seq_attention.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_seq2seq_attention(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_seq2seq_attention(target_index=1, sharpness=1.1, max_length=20, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["strongest_weight"] > 0 and data["weights"].shape[0] == len(data["tgt_tokens"])


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_seq2seq_attention))
