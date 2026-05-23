try:
    """
    自动生成自: part4_transformer\03_encoder_decoder.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    class PositionalEncodingFull(nn.Module):
        """
        正弦位置编码完整实现

        参数：
        - d_model: 模型维度（如512）
        - max_len: 最大序列长度（如5000）
        - dropout: dropout 概率

        工作原理：
        1. 预计算所有位置的编码（max_len x d_model 矩阵）
        2. 前向传播时直接加到输入嵌入上
        3. 注册为 buffer（不参与梯度更新，但会随模型保存）
        """
        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.d_model = d_model
            self.max_len = max_len
            # 预计算位置编码矩阵
            pe = self._compute_pe(max_len, d_model)
            # pe: [max_len, d_model]
            # 注册为 buffer：不是参数（不更新），但会被 state_dict 保存
            self.register_buffer('pe', pe.unsqueeze(0))
            # self.pe: [1, max_len, d_model]

        def _compute_pe(self, max_len: int, d_model: int) -> torch.Tensor:
            """
            逐步计算位置编码矩阵

            步骤1：创建位置索引 [max_len, 1]
            步骤2：计算频率分母 [d_model/2]
            步骤3：计算角度 [max_len, d_model/2]
            步骤4：填充 sin/cos
            """
            # 步骤1：位置索引
            position = torch.arange(max_len).unsqueeze(1).float()
            # position: [max_len=5000, 1]
            # position[i] = i（位置索引）

            # 步骤2：频率分母
            # div_term[i] = 10000^(2i/d_model)
            # 等价于 exp(2i * (-log(10000)/d_model))
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            # div_term: [d_model/2]
            # div_term[0] = 1.0（最高频）
            # div_term[-1] = 10000^(-1) = 0.0001（最低频）

            # 步骤3：计算角度 = position / div_term
            # 广播：[max_len, 1] * [d_model/2] -> [max_len, d_model/2]
            angles = position * div_term
            # angles: [max_len, d_model/2]
            # angles[pos, i] = pos / 10000^(2i/d_model)

            # 步骤4：填充 sin/cos
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(angles)   # 偶数维度用 sin
            pe[:, 1::2] = torch.cos(angles)   # 奇数维度用 cos
            # pe: [max_len, d_model]
            return pe

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: [B, T, d_model]
            返回: [B, T, d_model]  （x + 位置编码）
            """
            # self.pe[:, :x.size(1)]: [1, T, d_model]
            # 广播加法：[B, T, d_model] + [1, T, d_model] -> [B, T, d_model]
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

        def get_encoding(self, max_pos: int = None) -> np.ndarray:
            """获取位置编码矩阵（用于可视化）"""
            n = max_pos or self.max_len
            return self.pe[0, :n].numpy()


    def visualize_positional_encoding():
        """全面可视化位置编码的各种特性"""
        d_model = 128
        max_len = 100
        pe_module = PositionalEncodingFull(d_model, max_len, dropout=0.0)
        pe = pe_module.get_encoding(max_len)   # [100, 128]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 图1：完整位置编码热力图
        im = axes[0, 0].imshow(pe.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1, origin='lower')
        plt.colorbar(im, ax=axes[0, 0])
        axes[0, 0].set_title(f'位置编码矩阵（d_model={d_model}）\n横轴=位置，纵轴=维度',
                              fontsize=10, fontweight='bold')
        axes[0, 0].set_xlabel('位置 pos')
        axes[0, 0].set_ylabel('维度索引 i')

        # 图2：不同维度的频率
        for d in [0, 2, 10, 30, 60, 100, 126]:
            if d < d_model:
                axes[0, 1].plot(pe[:, d], linewidth=1.5, label=f'dim={d}', alpha=0.8)
        axes[0, 1].set_title('不同维度的位置编码值\n（低维=高频，高维=低频）',
                              fontsize=10, fontweight='bold')
        axes[0, 1].set_xlabel('位置 pos')
        axes[0, 1].set_ylabel('编码值')
        axes[0, 1].legend(fontsize=7, ncol=2)
        axes[0, 1].grid(True, alpha=0.3)

        # 图3：位置间余弦相似度
        pe_norm = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-8)
        sim_matrix = pe_norm @ pe_norm.T
        im3 = axes[0, 2].imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im3, ax=axes[0, 2])
        axes[0, 2].set_title('位置间余弦相似度\n（对角线=1，越远越不相似）',
                              fontsize=10, fontweight='bold')
        axes[0, 2].set_xlabel('位置 j')
        axes[0, 2].set_ylabel('位置 i')

        # 图4：位置0与其他位置的相似度
        sim_from_0 = sim_matrix[0]
        axes[1, 0].plot(sim_from_0, 'b-', linewidth=2)
        axes[1, 0].fill_between(range(max_len), sim_from_0, alpha=0.3)
        axes[1, 0].set_title('位置0与其他位置的相似度\n（随距离单调递减）',
                              fontsize=10, fontweight='bold')
        axes[1, 0].set_xlabel('位置 pos')
        axes[1, 0].set_ylabel('余弦相似度')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)

        # 图5：频率分析
        freqs = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
        axes[1, 1].semilogy(range(d_model // 2), freqs, 'r-', linewidth=2)
        axes[1, 1].set_title('各维度的频率\n（指数衰减，低维高频，高维低频）',
                              fontsize=10, fontweight='bold')
        axes[1, 1].set_xlabel('维度对索引 i')
        axes[1, 1].set_ylabel('频率（log scale）')
        axes[1, 1].grid(True, alpha=0.3)

        # 图6：词嵌入+位置编码
        torch.manual_seed(42)
        vocab_size, d = 100, 32
        emb = nn.Embedding(vocab_size, d)
        pe_small = PositionalEncodingFull(d, 20, dropout=0.0)
        tokens = torch.randint(0, vocab_size, (1, 10))
        x_emb = emb(tokens)
        x_pe  = pe_small(x_emb)
        axes[1, 2].imshow(x_pe[0].detach().numpy().T, aspect='auto', cmap='RdBu', vmin=-2, vmax=2)
        axes[1, 2].set_title('词嵌入 + 位置编码后的表示\n（每列=一个位置的向量）',
                              fontsize=10, fontweight='bold')
        axes[1, 2].set_xlabel('位置')
        axes[1, 2].set_ylabel('维度')

        plt.suptitle('正弦位置编码全面分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('positional_encoding_full.png', dpi=150, bbox_inches='tight')
        plt.show()


    class LearnablePositionalEncoding(nn.Module):
        """
        可学习位置编码（BERT 风格）

        与正弦编码的区别：
        - 正弦编码：固定，不参与训练，可外推到更长序列
        - 可学习编码：参与训练，通常效果更好，但不能外推
        """
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.pe = nn.Embedding(max_len, d_model)
            pe_init = PositionalEncodingFull(d_model, max_len, dropout=0.0).get_encoding(max_len)
            self.pe.weight.data = torch.from_numpy(pe_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, _ = x.shape
            positions = torch.arange(T, device=x.device)
            return self.dropout(x + self.pe(positions))


    def demo_positional_encoding():
        print("位置编码完整演示")
        print("=" * 50)
        d_model = 512
        pe = PositionalEncodingFull(d_model, max_len=5000, dropout=0.1)
        x = torch.randn(2, 10, d_model)
        out = pe(x)
        print(f"输入形状:  {tuple(x.shape)}")
        print(f"输出形状:  {tuple(out.shape)}")
        print(f"PE矩阵形状: {tuple(pe.pe.shape)}")
        pe_matrix = pe.get_encoding(100)
        all_unique = True
        for i in range(100):
            for j in range(i+1, 100):
                if np.allclose(pe_matrix[i], pe_matrix[j], atol=1e-6):
                    all_unique = False
        print(f"所有位置编码唯一: {all_unique}")
        visualize_positional_encoding()
        return pe

    pe = demo_positional_encoding()
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part4_transformer/03_encoder_decoder.py", e)
