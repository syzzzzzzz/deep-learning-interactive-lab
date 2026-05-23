MODULE_TITLE = "注意力机制"
MODULE_SUMMARY = "从查询、键、值和权重矩阵理解注意力的信息检索过程。"
MODULE_TAGS = ["Transformer", "注意力", "NLP", "可视化"]

import streamlit as st


PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _go_to_playground(example: str) -> None:
    st.query_params["module"] = PLAYGROUND_TARGET
    st.query_params["example"] = example
    st.rerun()


def _render_streamlit_entry() -> None:
    st.set_page_config(page_title="注意力机制", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #fbfcfb 0%, #eef5f2 100%); color: #172026; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; }
        .hero { border-bottom: 1px solid #d7dde1; padding-bottom: 0.9rem; margin-bottom: 1rem; }
        .hero h1 { margin: 0; font-size: clamp(2rem, 3vw, 3rem); letter-spacing: 0; }
        .hero p { color: #58646d; max-width: 980px; line-height: 1.7; margin: 0.45rem 0 0; }
        .note { border-left: 4px solid #0f8b8d; background: rgba(255,255,255,0.78); border-radius: 0 8px 8px 0; padding: 0.75rem 0.95rem; line-height: 1.7; }
        </style>
        <div class="hero">
          <h1>注意力机制</h1>
          <p>从 Query、Key、Value 的投影出发，理解注意力如何把“当前位置的问题”映射到“上下文中的信息检索”。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.7, 0.3])
    with left:
        st.markdown(
            """
            <div class="note">
            注意力里的 Q/K/V 本质上是线性投影后的表示。先在中央控制台里观察 Linear 层如何保持序列长度、替换最后一维，
            再回到完整多头注意力，会更容易看懂 reshape、分头和加权求和的形状流动。
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("去实战：Transformer 示例", width="stretch"):
            _go_to_playground("transformer")
    st.subheader("核心张量形状")
    st.code(
        """X: [batch, seq_len, d_model]
Q = X @ W_Q: [batch, seq_len, d_k]
K = X @ W_K: [batch, seq_len, d_k]
V = X @ W_V: [batch, seq_len, d_v]
scores = Q @ K.transpose(-2, -1): [batch, seq_len, seq_len]
output = softmax(scores / sqrt(d_k)) @ V: [batch, seq_len, d_v]""",
        language="text",
    )


if _running_under_streamlit():
    _render_streamlit_entry()
    st.stop()

try:
    """
    自动生成自: part4_transformer\01_attention_mechanism.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from typing import Optional, Tuple

    # ─────────────────────────────────────────────────────────
    # 完整自注意力实现，每行附带形状注释
    # ─────────────────────────────────────────────────────────

    def scaled_dot_product_attention_verbose(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的 Scaled Dot-Product Attention，附带逐步形状打印

        参数（以 batch_size=2, n_heads=8, seq_len=10, d_k=64 为例）：
        Q: [2, 8, 10, 64]  — batch × heads × seq_len × d_k
        K: [2, 8, 10, 64]
        V: [2, 8, 10, 64]
        mask: [2, 1, 1, 10] 或 [2, 1, 10, 10]（可选）

        返回：
        output:       [2, 8, 10, 64]
        attn_weights: [2, 8, 10, 10]
        """
        B, H, T_q, d_k = Q.shape   # 2, 8, 10, 64
        _, _, T_k, _   = K.shape   # 2, 8, 10, 64
        # Q:  [B=2, H=8, T_q=10, d_k=64]
        # K:  [B=2, H=8, T_k=10, d_k=64]
        # V:  [B=2, H=8, T_k=10, d_v=64]

        # ── 步骤1：计算注意力分数 ──────────────────────────────
        # K.transpose(-2, -1): [2, 8, 64, 10]  ← 转置最后两维
        # torch.matmul(Q, K^T): [2, 8, 10, 10] ← 每个 query 与所有 key 的点积
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # scores: [B=2, H=8, T_q=10, T_k=10]
        # scores[b, h, i, j] = Q[b,h,i,:] · K[b,h,j,:]  （位置i对位置j的原始分数）

        # ── 步骤2：缩放 ────────────────────────────────────────
        scale = d_k ** 0.5   # √64 = 8.0
        scores = scores / scale
        # scores: [2, 8, 10, 10]  值域从 ~N(0, d_k) 缩放到 ~N(0, 1)

        # ── 步骤3：应用 mask（可选）────────────────────────────
        if mask is not None:
            # mask: [2, 1, 1, 10] 或 [2, 1, 10, 10]，True 的位置设为 -inf
            # 广播后与 scores 形状对齐
            scores = scores.masked_fill(mask, float('-inf'))
            # scores: [2, 8, 10, 10]  被 mask 的位置 = -inf → softmax 后 = 0

        # ── 步骤4：Softmax 归一化 ──────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [2, 8, 10, 10]
        # attn_weights[b, h, i, :].sum() == 1.0  （每行之和为1）
        # attn_weights[b, h, i, j] = 位置i分配给位置j的注意力权重

        # ── 步骤5：Dropout（训练时）────────────────────────────
        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        # attn_weights: [2, 8, 10, 10]  部分权重被随机置0

        # ── 步骤6：加权求和 ────────────────────────────────────
        output = torch.matmul(attn_weights, V)
        # attn_weights: [2, 8, 10, 10]
        # V:            [2, 8, 10, 64]
        # output:       [2, 8, 10, 64]
        # output[b, h, i, :] = Σ_j attn_weights[b,h,i,j] * V[b,h,j,:]

        return output, attn_weights


    def demonstrate_tensor_shapes():
        """用具体数字演示每一步的张量形状"""
        torch.manual_seed(42)
        B, T, d_model = 2, 10, 512
        d_k = d_v = 64
        H = 8  # 头数，d_model / H = 64 = d_k

        print("=" * 60)
        print("自注意力张量形状逐步演示")
        print(f"batch_size={B}, seq_len={T}, d_model={d_model}, d_k={d_k}, H={H}")
        print("=" * 60)

        # 输入
        X = torch.randn(B, T, d_model)
        print(f"\n输入 X:          {tuple(X.shape)}  ← [batch, seq_len, d_model]")

        # 投影矩阵
        W_Q = torch.randn(d_model, d_k)
        W_K = torch.randn(d_model, d_k)
        W_V = torch.randn(d_model, d_v)
        print(f"W_Q:             {tuple(W_Q.shape)}  ← [d_model, d_k]")
        print(f"W_K:             {tuple(W_K.shape)}")
        print(f"W_V:             {tuple(W_V.shape)}")

        # 单头注意力
        Q = X @ W_Q   # [2, 10, 64]
        K = X @ W_K
        V = X @ W_V
        print(f"\nQ = X @ W_Q:     {tuple(Q.shape)}  ← [batch, seq_len, d_k]")
        print(f"K = X @ W_K:     {tuple(K.shape)}")
        print(f"V = X @ W_V:     {tuple(V.shape)}")

        scores = Q @ K.transpose(-2, -1)  # [2, 10, 10]
        print(f"\nscores = Q@K^T:  {tuple(scores.shape)}  ← [batch, seq_len, seq_len]")
        print(f"  scores[0,0,:] = {scores[0,0,:].detach().numpy().round(2)}")
        print(f"  方差（缩放前）: {scores.var().item():.2f}  期望≈{float(d_k):.1f}")

        scores_scaled = scores / (d_k ** 0.5)
        print(f"\nscores/√{d_k}:    {tuple(scores_scaled.shape)}")
        print(f"  方差（缩放后）: {scores_scaled.var().item():.2f}  期望≈1.0")

        attn = F.softmax(scores_scaled, dim=-1)
        print(f"\nattn = softmax:  {tuple(attn.shape)}  ← [batch, seq_len, seq_len]")
        print(f"  attn[0,0,:].sum() = {attn[0,0,:].sum().item():.6f}  （应为1.0）")
        print(f"  attn[0,0,:] = {attn[0,0,:].detach().numpy().round(3)}")

        output = attn @ V
        print(f"\noutput = attn@V: {tuple(output.shape)}  ← [batch, seq_len, d_v]")

        # 多头版本
        print("\n" + "─" * 60)
        print("多头版本（H=8个头）")
        print("─" * 60)

        W_Q_mh = torch.randn(d_model, d_model)  # [512, 512]
        Q_mh = X @ W_Q_mh                        # [2, 10, 512]
        print(f"\nQ_mh = X @ W_Q:  {tuple(Q_mh.shape)}  ← [batch, seq_len, d_model]")

        # 分头：reshape + transpose
        Q_heads = Q_mh.view(B, T, H, d_k).transpose(1, 2)
        print(f"view(B,T,H,d_k): {tuple(Q_mh.view(B,T,H,d_k).shape)}  ← [batch, seq_len, heads, d_k]")
        print(f"transpose(1,2):  {tuple(Q_heads.shape)}  ← [batch, heads, seq_len, d_k]")

        K_heads = (X @ torch.randn(d_model, d_model)).view(B, T, H, d_k).transpose(1, 2)
        V_heads = (X @ torch.randn(d_model, d_model)).view(B, T, H, d_v).transpose(1, 2)

        out_heads, attn_heads = scaled_dot_product_attention_verbose(Q_heads, K_heads, V_heads)
        print(f"\n注意力输出:      {tuple(out_heads.shape)}  ← [batch, heads, seq_len, d_k]")
        print(f"注意力权重:      {tuple(attn_heads.shape)}  ← [batch, heads, seq_len, seq_len]")

        # 合并头
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, d_model)
        print(f"\ntranspose(1,2):  {tuple(out_heads.transpose(1,2).shape)}  ← [batch, seq_len, heads, d_k]")
        print(f"view(B,T,d_model):{tuple(out_concat.shape)}  ← [batch, seq_len, d_model]")

        W_O = torch.randn(d_model, d_model)
        final_out = out_concat @ W_O
        print(f"@ W_O:           {tuple(final_out.shape)}  ← [batch, seq_len, d_model]  最终输出")

        return attn_heads


    attn_heads = demonstrate_tensor_shapes()

    # ============================================================
    # 代码段 2
    # ============================================================

    def analyze_attention_numerics():
        """
        分析 √d_k 缩放的必要性
        用具体数字说明不缩放时 softmax 饱和的问题
        """
        torch.manual_seed(0)
        T, d_k = 10, 64

        Q = torch.randn(T, d_k)
        K = torch.randn(T, d_k)

        scores_raw    = Q @ K.T                  # 未缩放
        scores_scaled = Q @ K.T / (d_k ** 0.5)  # 缩放后

        attn_raw    = F.softmax(scores_raw,    dim=-1)
        attn_scaled = F.softmax(scores_scaled, dim=-1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 分数分布
        axes[0, 0].hist(scores_raw.flatten().numpy(), bins=40,
                        color='#C44E52', alpha=0.8, edgecolor='white')
        axes[0, 0].set_title(f'原始分数分布\n均值={scores_raw.mean():.2f}, 方差={scores_raw.var():.2f}',
                              fontsize=10, fontweight='bold')
        axes[0, 0].set_xlabel('分数值'); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(scores_scaled.flatten().numpy(), bins=40,
                        color='#4C72B0', alpha=0.8, edgecolor='white')
        axes[0, 1].set_title(f'缩放后分数分布\n均值={scores_scaled.mean():.2f}, 方差={scores_scaled.var():.2f}',
                              fontsize=10, fontweight='bold')
        axes[0, 1].set_xlabel('分数值'); axes[0, 1].grid(True, alpha=0.3)

        # 注意力权重分布
        axes[0, 2].hist(attn_raw.flatten().numpy(), bins=40,
                        color='#C44E52', alpha=0.8, edgecolor='white', label='未缩放')
        axes[0, 2].hist(attn_scaled.flatten().numpy(), bins=40,
                        color='#4C72B0', alpha=0.5, edgecolor='white', label='缩放后')
        axes[0, 2].set_title('注意力权重分布对比\n（未缩放更极端，接近 one-hot）',
                              fontsize=10, fontweight='bold')
        axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

        # 注意力热力图
        im1 = axes[1, 0].imshow(attn_raw.numpy(), cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im1, ax=axes[1, 0])
        axes[1, 0].set_title('未缩放注意力权重\n（极度集中，信息丢失）', fontsize=10, fontweight='bold')
        axes[1, 0].set_xlabel('Key 位置'); axes[1, 0].set_ylabel('Query 位置')

        im2 = axes[1, 1].imshow(attn_scaled.numpy(), cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[1, 1])
        axes[1, 1].set_title('缩放后注意力权重\n（分布均匀，信息丰富）', fontsize=10, fontweight='bold')
        axes[1, 1].set_xlabel('Key 位置'); axes[1, 1].set_ylabel('Query 位置')

        # 熵对比（熵越高=注意力越分散=信息越丰富）
        entropy_raw    = -(attn_raw    * (attn_raw    + 1e-9).log()).sum(dim=-1)
        entropy_scaled = -(attn_scaled * (attn_scaled + 1e-9).log()).sum(dim=-1)
        x = range(T)
        axes[1, 2].plot(x, entropy_raw.numpy(),    'r-o', label=f'未缩放 均值={entropy_raw.mean():.2f}',
                        linewidth=2, markersize=6)
        axes[1, 2].plot(x, entropy_scaled.numpy(), 'b-o', label=f'缩放后 均值={entropy_scaled.mean():.2f}',
                        linewidth=2, markersize=6)
        axes[1, 2].axhline(np.log(T), color='gray', linestyle='--', alpha=0.7,
                            label=f'最大熵 log({T})={np.log(T):.2f}')
        axes[1, 2].set_title('注意力熵（越高=越分散=越好）', fontsize=10, fontweight='bold')
        axes[1, 2].set_xlabel('Query 位置'); axes[1, 2].set_ylabel('熵')
        axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'√d_k 缩放的必要性分析（d_k={d_k}）', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('attention_scaling_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"未缩放注意力熵均值:  {entropy_raw.mean():.4f}")
        print(f"缩放后注意力熵均值:  {entropy_scaled.mean():.4f}")
        print(f"最大可能熵 log({T}): {np.log(T):.4f}")
        print(f"缩放后熵/最大熵:     {entropy_scaled.mean()/np.log(T):.1%}")

    analyze_attention_numerics()

    # ============================================================
    # 代码段 3
    # ============================================================

    def demonstrate_causal_mask():
        """
        演示因果掩码（Causal Mask）的工作原理
        用于 Decoder 的自注意力：位置 i 只能看到位置 0..i
        """
        T = 6
        # 创建上三角掩码（True = 被遮住）
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        print("因果掩码（True=被遮住）：")
        print(mask.int().numpy())
        print()
        print("含义：行=Query位置，列=Key位置")
        print("位置0只能看位置0，位置3能看位置0,1,2,3")

        # 演示 masked_fill
        scores = torch.randn(1, 1, T, T)
        scores_masked = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores_masked, dim=-1)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        im0 = axes[0].imshow(scores[0, 0].numpy(), cmap='RdBu', aspect='auto')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title('原始分数', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Key 位置'); axes[0].set_ylabel('Query 位置')

        im1 = axes[1].imshow(mask.float().numpy(), cmap='Reds', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title('因果掩码\n（红色=被遮住，设为-∞）', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Key 位置'); axes[1].set_ylabel('Query 位置')
        for i in range(T):
            for j in range(T):
                axes[1].text(j, i, '✗' if mask[i,j] else '✓',
                             ha='center', va='center', fontsize=10,
                             color='white' if mask[i,j] else 'black')

        im2 = axes[2].imshow(attn[0, 0].detach().numpy(), cmap='Blues', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title('掩码后注意力权重\n（上三角=0，下三角=有效）', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Key 位置'); axes[2].set_ylabel('Query 位置')
        for i in range(T):
            for j in range(T):
                val = attn[0, 0, i, j].item()
                axes[2].text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=8, color='white' if val > 0.5 else 'black')

        plt.suptitle('因果掩码（Causal Mask）工作原理', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('causal_mask.png', dpi=150, bbox_inches='tight')
        plt.show()

    demonstrate_causal_mask()
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part4_transformer/01_attention_mechanism.py", e)
