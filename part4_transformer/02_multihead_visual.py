"""
自动生成自: part4_transformer\02_multihead_visual.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple, List

class MultiHeadAttentionFull(nn.Module):
    """
    完整 Multi-Head Attention 实现
    包含所有细节：投影、分头、注意力、合并、输出投影

    参数（以 d_model=512, n_heads=8 为例）：
    - d_model=512：模型总维度
    - n_heads=8：头数
    - d_k = d_v = d_model/n_heads = 64：每头的维度

    参数量：
    - W_Q: 512×512 = 262144
    - W_K: 512×512 = 262144
    - W_V: 512×512 = 262144
    - W_O: 512×512 = 262144
    - 总计: 4×512×512 = 1,048,576 ≈ 1M 参数
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # 每头的 key/query 维度
        self.d_v = d_model // n_heads   # 每头的 value 维度
        self.scale = self.d_k ** -0.5   # 1/√d_k，避免每次计算

        # 四个投影矩阵（合并为一个大矩阵可以加速，但分开更清晰）
        self.W_q = nn.Linear(d_model, d_model, bias=bias)  # [d_model, d_model]
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)  # 输出投影

        self.dropout = nn.Dropout(dropout)

        # 保存中间值（用于可视化和调试）
        self.attn_weights: Optional[torch.Tensor] = None   # [B, H, T_q, T_k]
        self.attn_scores:  Optional[torch.Tensor] = None   # 缩放前的分数

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 [B, T, d_model] 分割为 [B, n_heads, T, d_k]

        步骤：
        1. view:      [B, T, d_model] → [B, T, n_heads, d_k]
        2. transpose: [B, T, n_heads, d_k] → [B, n_heads, T, d_k]

        以 B=2, T=10, d_model=512, n_heads=8, d_k=64 为例：
        [2, 10, 512] → [2, 10, 8, 64] → [2, 8, 10, 64]
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)   # [B, T, H, d_k]
        return x.transpose(1, 2)                     # [B, H, T, d_k]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 [B, n_heads, T, d_k] 合并为 [B, T, d_model]

        步骤：
        1. transpose: [B, H, T, d_k] → [B, T, H, d_k]
        2. contiguous + view: [B, T, H, d_k] → [B, T, d_model]

        contiguous() 是必须的：transpose 后内存不连续，view 会报错
        """
        B, H, T, d_k = x.shape
        x = x.transpose(1, 2).contiguous()   # [B, T, H, d_k]
        return x.view(B, T, self.d_model)     # [B, T, d_model]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，逐步注释每个张量的形状

        输入（以 B=2, T_q=10, T_k=10, d_model=512 为例）：
        query: [2, 10, 512]
        key:   [2, 10, 512]
        value: [2, 10, 512]
        mask:  [2, 1, 1, 10] 或 [2, 1, 10, 10]（可选）

# 输出：[2, 10, 512]
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # ── 步骤1：线性投影 ────────────────────────────────
        Q = self.W_q(query)   # [B=2, T_q=10, d_model=512]
        K = self.W_k(key)     # [B=2, T_k=10, d_model=512]
        V = self.W_v(value)   # [B=2, T_k=10, d_model=512]

        # ── 步骤2：分头 ────────────────────────────────────
        Q = self._split_heads(Q)   # [B=2, H=8, T_q=10, d_k=64]
        K = self._split_heads(K)   # [B=2, H=8, T_k=10, d_k=64]
        V = self._split_heads(V)   # [B=2, H=8, T_k=10, d_v=64]

        # ── 步骤3：Scaled Dot-Product Attention ───────────
        # K.transpose(-2,-1): [2, 8, 64, 10]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [B=2, H=8, T_q=10, T_k=10]
        self.attn_scores = scores.detach()

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)   # [2, 8, 10, 10]
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        # ── 步骤4：加权求和 ────────────────────────────────
        context = torch.matmul(attn, V)    # [2, 8, 10, 64]

        # ── 步骤5：合并头 ──────────────────────────────────
        context = self._merge_heads(context)   # [2, 10, 512]

        # ── 步骤6：输出投影 ────────────────────────────────
        output = self.W_o(context)   # [2, 10, 512]

        return output

    def visualize_all_heads(self, tokens: Optional[List[str]] = None,
                             batch_idx: int = 0, figsize_scale: float = 2.2):
        """可视化所有头的注意力权重"""
        if self.attn_weights is None:
            print("请先运行 forward()")
            return
        weights = self.attn_weights[batch_idx].numpy()   # [H, T_q, T_k]
        H, T_q, T_k = weights.shape
        n_cols = min(4, H)
        n_rows = (H + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * figsize_scale * T_k/8,
                                           n_rows * figsize_scale * T_q/8 + 0.8))
        axes = np.array(axes).reshape(n_rows, n_cols)
        for h in range(H):
            ax = axes[h // n_cols, h % n_cols]
            w = weights[h]
            im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max(), aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if tokens:
                ax.set_xticks(range(min(T_k, len(tokens))))
                ax.set_xticklabels(tokens[:T_k], rotation=45, ha='right', fontsize=7)
                ax.set_yticks(range(min(T_q, len(tokens))))
                ax.set_yticklabels(tokens[:T_q], fontsize=7)
            entropy = -(w * np.log(w + 1e-9)).sum(axis=-1).mean()
            max_attn_pos = w.mean(axis=0).argmax()
            ax.set_title(f'头 {h+1}  熵={entropy:.2f}\n最关注位置={max_attn_pos}',
                         fontsize=8, fontweight='bold')
        for h in range(H, n_rows * n_cols):
            axes[h // n_cols, h % n_cols].axis('off')
        plt.suptitle(f'Multi-Head Attention 权重（{H}个头）', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('mha_all_heads.png', dpi=150, bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────
# 三种不同输入的注意力权重热力图
# ─────────────────────────────────────────────────────────

def demo_three_input_patterns():
    """
    演示三种不同输入模式下的注意力权重差异

    输入1：均匀随机 → 注意力分散
    输入2：有重复模式 → 注意力集中在相似位置
    输入3：有明显主题词 → 注意力集中在主题词
    """
    torch.manual_seed(42)
    d_model, n_heads, T = 64, 4, 8
    mha = MultiHeadAttentionFull(d_model, n_heads, dropout=0.0)

    scenarios = {
        '均匀随机输入\n（注意力应分散）': torch.randn(1, T, d_model),
        '重复模式输入\n（相似位置注意力更强）': torch.cat([
            torch.randn(1, T//2, d_model).repeat(1, 2, 1)
        ], dim=1),
        '有主题词输入\n（主题词位置注意力集中）': torch.cat([
            torch.randn(1, 1, d_model) * 5,   # 主题词（幅度大）
            torch.randn(1, T-1, d_model) * 0.5
        ], dim=1),
    }

    fig, axes = plt.subplots(n_heads, len(scenarios),
                              figsize=(len(scenarios) * 3.5, n_heads * 2.8))

    for col, (title, x) in enumerate(scenarios.items()):
        with torch.no_grad():
            mha(x, x, x)
        weights = mha.attn_weights[0].numpy()   # [H, T, T]
        for h in range(n_heads):
            ax = axes[h, col]
            w = weights[h]
            im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max(), aspect='auto')
            entropy = -(w * np.log(w + 1e-9)).sum(axis=-1).mean()
            if h == 0:
                ax.set_title(title, fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'头{h+1}\n熵={entropy:.2f}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle('三种输入模式下的注意力权重对比\n（每行=一个头，每列=一种输入）',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mha_three_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印每种输入的注意力熵统计
    print("\n注意力熵统计（熵越高=注意力越分散）：")
    for title, x in scenarios.items():
        with torch.no_grad():
            mha(x, x, x)
        w = mha.attn_weights[0].numpy()
        entropy = -(w * np.log(w + 1e-9)).sum(axis=-1).mean()
        print(f"  {title.split(chr(10))[0]:20s}: 平均熵={entropy:.4f}")


# ─────────────────────────────────────────────────────────
# 各头学到的不同特征分析
# ─────────────────────────────────────────────────────────

def analyze_head_specialization():
    """
    分析多头注意力中各头的专业化程度

    通过训练一个简单任务，观察各头是否学到不同的关注模式
    """
    torch.manual_seed(42)
    d_model, n_heads, T = 64, 4, 12
    mha = MultiHeadAttentionFull(d_model, n_heads, dropout=0.0)
    fc  = nn.Linear(d_model, 1)

    optimizer = torch.optim.Adam(list(mha.parameters()) + list(fc.parameters()), lr=0.001)

    # 任务：预测序列中第一个位置的值（需要关注位置0）
    for step in range(300):
        x = torch.randn(16, T, d_model)
        target = x[:, 0, 0:1]   # 第一个位置的第一个特征
        out = mha(x, x, x)
        pred = fc(out[:, 0])
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 分析各头的注意力模式
    x_test = torch.randn(1, T, d_model)
    with torch.no_grad():
        mha(x_test, x_test, x_test)

    weights = mha.attn_weights[0].numpy()   # [H, T, T]

    fig, axes = plt.subplots(2, n_heads, figsize=(n_heads * 3.5, 6))

    for h in range(n_heads):
        w = weights[h]
        # 注意力权重热力图
        im = axes[0, h].imshow(w, cmap='Blues', vmin=0, vmax=w.max(), aspect='auto')
        plt.colorbar(im, ax=axes[0, h], fraction=0.046)
        entropy = -(w * np.log(w + 1e-9)).sum(axis=-1).mean()
        axes[0, h].set_title(f'头 {h+1}\n熵={entropy:.3f}', fontsize=10, fontweight='bold')
        axes[0, h].set_xlabel('Key 位置'); axes[0, h].set_ylabel('Query 位置')

        # 每个 Query 位置最关注的 Key 位置
        top_key = w.argmax(axis=1)
        axes[1, h].bar(range(T), top_key, color='steelblue', alpha=0.8)
        axes[1, h].axhline(0, color='red', linestyle='--', alpha=0.7, label='位置0（目标）')
        axes[1, h].set_title(f'头 {h+1} 最关注的 Key 位置', fontsize=9)
        axes[1, h].set_xlabel('Query 位置'); axes[1, h].set_ylabel('最关注的 Key')
        axes[1, h].set_ylim(-0.5, T - 0.5)
        axes[1, h].legend(fontsize=7)
        axes[1, h].grid(True, alpha=0.3)

    plt.suptitle('各头专业化分析（任务：关注位置0）\n训练后各头是否都学会关注位置0？',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('head_specialization.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_multihead_complete():
    torch.manual_seed(42)
    print("Multi-Head Attention 完整演示")
    print("=" * 50)

    d_model, n_heads = 512, 8
    mha = MultiHeadAttentionFull(d_model, n_heads)

    # 参数量统计
    total = sum(p.numel() for p in mha.parameters())
    print(f"d_model={d_model}, n_heads={n_heads}, d_k={d_model//n_heads}")
    print(f"总参数量: {total:,}  ({total/1e6:.2f}M)")
    print(f"  W_Q: {d_model}×{d_model} = {d_model*d_model:,}")
    print(f"  W_K: {d_model}×{d_model} = {d_model*d_model:,}")
    print(f"  W_V: {d_model}×{d_model} = {d_model*d_model:,}")
    print(f"  W_O: {d_model}×{d_model} = {d_model*d_model:,}")

    # 前向传播
    B, T = 2, 10
    x = torch.randn(B, T, d_model)
    tokens = ['我', '爱', '深度', '学习', '它', '很', '有趣', '也', '很', '强大']
    out = mha(x, x, x)
    print(f"\n输入形状:  {tuple(x.shape)}")
    print(f"输出形状:  {tuple(out.shape)}")
    print(f"注意力权重形状: {tuple(mha.attn_weights.shape)}")

    # 可视化
    mha.visualize_all_heads(tokens=tokens)
    demo_three_input_patterns()
    analyze_head_specialization()

    return mha

mha = demo_multihead_complete()
