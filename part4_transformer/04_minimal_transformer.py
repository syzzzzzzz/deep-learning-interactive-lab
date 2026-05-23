try:
    """
    自动生成自: part4_transformer\04_minimal_transformer.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math


    # ─────────────────────────────────────────────────────────
    # 基础组件（复用前面章节）
    # ─────────────────────────────────────────────────────────

    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights


    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.W_Q = nn.Linear(d_model, d_model, bias=False)
            self.W_K = nn.Linear(d_model, d_model, bias=False)
            self.W_V = nn.Linear(d_model, d_model, bias=False)
            self.W_O = nn.Linear(d_model, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.attention_weights = None

        def split_heads(self, x):
            b, s, _ = x.shape
            return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

        def forward(self, query, key, value, mask=None):
            b = query.shape[0]
            Q = self.split_heads(self.W_Q(query))
            K = self.split_heads(self.W_K(key))
            V = self.split_heads(self.W_V(value))
            out, self.attention_weights = scaled_dot_product_attention(Q, K, V, mask)
            out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
            return self.W_O(out)


    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])


    # ─────────────────────────────────────────────────────────
    # Feed-Forward Network（FFN）
    # ─────────────────────────────────────────────────────────

    class FeedForward(nn.Module):
        """
        两层全连接网络，中间维度扩大4倍
        FFN(x) = max(0, xW1 + b1)W2 + b2
        """
        def __init__(self, d_model, d_ff=None, dropout=0.1):
            super().__init__()
            d_ff = d_ff or d_model * 4
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)


    # ─────────────────────────────────────────────────────────
    # Encoder Block
    # ─────────────────────────────────────────────────────────

    class EncoderBlock(nn.Module):
        """
        Transformer Encoder 的一个块：
        1. Multi-Head Self-Attention
        2. Add & LayerNorm（残差连接）
        3. Feed-Forward Network
        4. Add & LayerNorm（残差连接）
        """
        def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, src_mask=None):
            # Self-Attention + 残差
            attn_out = self.self_attn(x, x, x, src_mask)
            x = self.norm1(x + self.dropout(attn_out))

            # FFN + 残差
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x


    # ─────────────────────────────────────────────────────────
    # Decoder Block
    # ─────────────────────────────────────────────────────────

    class DecoderBlock(nn.Module):
        """
        Transformer Decoder 的一个块：
        1. Masked Multi-Head Self-Attention（只看过去）
        2. Add & LayerNorm
        3. Cross-Attention（关注 Encoder 输出）
        4. Add & LayerNorm
        5. Feed-Forward Network
        6. Add & LayerNorm
        """
        def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
            # 1. Masked Self-Attention（只看过去的目标词）
            self_attn_out = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(self_attn_out))

            # 2. Cross-Attention（Q来自Decoder，K/V来自Encoder）
            cross_attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(cross_attn_out))

            # 3. FFN
            x = self.norm3(x + self.ffn(x))
            return x


    # ─────────────────────────────────────────────────────────
    # 完整 Transformer
    # ─────────────────────────────────────────────────────────

    class Transformer(nn.Module):
        """
        完整的 Encoder-Decoder Transformer

        参数：
          src_vocab_size: 源语言词汇表大小
          tgt_vocab_size: 目标语言词汇表大小
          d_model: 模型维度（默认512）
          num_heads: 注意力头数（默认8）
          num_encoder_layers: Encoder 层数（默认6）
          num_decoder_layers: Decoder 层数（默认6）
          d_ff: FFN 中间维度（默认2048）
          max_len: 最大序列长度
          dropout: Dropout 概率
        """
        def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            max_len=5000,
            dropout=0.1,
        ):
            super().__init__()
            self.d_model = d_model

            # 嵌入层
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

            # Encoder
            self.encoder_layers = nn.ModuleList([
                EncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_encoder_layers)
            ])
            self.encoder_norm = nn.LayerNorm(d_model)

            # Decoder
            self.decoder_layers = nn.ModuleList([
                DecoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_decoder_layers)
            ])
            self.decoder_norm = nn.LayerNorm(d_model)

            # 输出投影
            self.output_proj = nn.Linear(d_model, tgt_vocab_size)

            # 权重初始化
            self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def make_src_mask(self, src, pad_idx=0):
            """源序列掩码：遮住 padding"""
            return (src != pad_idx).unsqueeze(1).unsqueeze(2)

        def make_tgt_mask(self, tgt, pad_idx=0):
            """目标序列掩码：遮住 padding + 未来位置"""
            tgt_len = tgt.shape[1]
            pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
            return pad_mask & causal_mask

        def encode(self, src, src_mask=None):
            """编码源序列"""
            x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
            for layer in self.encoder_layers:
                x = layer(x, src_mask)
            return self.encoder_norm(x)

        def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
            """解码目标序列"""
            x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
            for layer in self.decoder_layers:
                x = layer(x, enc_output, src_mask, tgt_mask)
            return self.decoder_norm(x)

        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            enc_output = self.encode(src, src_mask)
            dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
            return self.output_proj(dec_output)

        def get_all_attention_weights(self):
            """获取所有层的注意力权重（用于可视化）"""
            weights = {
                'encoder_self_attn': [],
                'decoder_self_attn': [],
                'decoder_cross_attn': [],
            }
            for layer in self.encoder_layers:
                if layer.self_attn.attention_weights is not None:
                    weights['encoder_self_attn'].append(
                        layer.self_attn.attention_weights.detach()
                    )
            for layer in self.decoder_layers:
                if layer.self_attn.attention_weights is not None:
                    weights['decoder_self_attn'].append(
                        layer.self_attn.attention_weights.detach()
                    )
                if layer.cross_attn.attention_weights is not None:
                    weights['decoder_cross_attn'].append(
                        layer.cross_attn.attention_weights.detach()
                    )
            return weights


    # ─────────────────────────────────────────────────────────
    # 极简 GPT（仅 Decoder，用于语言模型）
    # ─────────────────────────────────────────────────────────

    class MiniGPT(nn.Module):
        """
        极简 GPT：仅使用 Decoder（无 Encoder）
        用于语言模型、文本生成

        与完整 Transformer 的区别：
        - 无 Encoder，无 Cross-Attention
        - 只有 Masked Self-Attention
        - 自回归生成：每次生成一个 token
        """
        def __init__(self, vocab_size, d_model=128, num_heads=4,
                     num_layers=4, d_ff=512, max_len=512, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

            self.blocks = nn.ModuleList([
                self._make_block(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            # 权重绑定（嵌入层和输出层共享权重）
            self.head.weight = self.embedding.weight

            self._init_weights()

        def _make_block(self, d_model, num_heads, d_ff, dropout):
            return nn.ModuleDict({
                'attn': MultiHeadAttention(d_model, num_heads, dropout),
                'ffn': FeedForward(d_model, d_ff, dropout),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0, std=0.02)

        def forward(self, x):
            seq_len = x.shape[1]
            # 因果掩码
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

            h = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))

            for block in self.blocks:
                attn_out = block['attn'](h, h, h, mask)
                h = block['norm1'](h + attn_out)
                h = block['norm2'](h + block['ffn'](h))

            h = self.norm(h)
            return self.head(h)

        @torch.no_grad()
        def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
            """
            自回归生成文本
            idx: [1, seq_len] 起始 token 序列
            """
            self.eval()
            for _ in range(max_new_tokens):
                logits = self(idx)[:, -1, :]  # 只取最后一个位置的 logits

                # 温度缩放
                logits = logits / temperature

                # Top-k 采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)

            return idx


    # ─────────────────────────────────────────────────────────
    # 训练一个玩具翻译任务
    # ─────────────────────────────────────────────────────────

    def train_toy_translation():
        """
        训练一个极简翻译模型：数字序列反转
    # 输入: [1, 2, 3, 4, 5]
    # 输出: [5, 4, 3, 2, 1]
        """
        torch.manual_seed(42)

        VOCAB_SIZE = 12   # 0-9 + PAD(10) + BOS(11)
        PAD_IDX = 10
        BOS_IDX = 11
        SEQ_LEN = 6
        N_SAMPLES = 1000

        # 生成数据
        def make_batch(n, seq_len=SEQ_LEN):
            src = torch.randint(1, 10, (n, seq_len))
            tgt_in = torch.cat([
                torch.full((n, 1), BOS_IDX),
                src.flip(1)[:, :-1]
            ], dim=1)
            tgt_out = src.flip(1)
            return src, tgt_in, tgt_out

        # 极简 Transformer（小参数量，快速训练）
        model = Transformer(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=256,
            dropout=0.1,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        losses = []
        for epoch in range(200):
            model.train()
            src, tgt_in, tgt_out = make_batch(64)

            src_mask = model.make_src_mask(src, PAD_IDX)
            tgt_mask = model.make_tgt_mask(tgt_in, PAD_IDX)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_out.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}")

        # 测试
        model.eval()
        src_test = torch.tensor([[3, 1, 4, 1, 5, 9]])
        print(f"\n测试: 输入={src_test[0].tolist()}")
        print(f"期望输出: {src_test[0].flip(0).tolist()}")

        # 贪心解码
        with torch.no_grad():
            enc_out = model.encode(src_test)
            tgt = torch.tensor([[BOS_IDX]])
            for _ in range(SEQ_LEN):
                tgt_mask = model.make_tgt_mask(tgt, PAD_IDX)
                dec_out = model.decode(tgt, enc_out, tgt_mask=tgt_mask)
                next_token = model.output_proj(dec_out[:, -1]).argmax(-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)

        print(f"模型输出: {tgt[0, 1:].tolist()}")

        # 可视化注意力权重
        with torch.no_grad():
            _ = model(src_test, tgt[:, :-1])

        attn_weights = model.get_all_attention_weights()
        src_tokens = [str(t.item()) for t in src_test[0]]
        tgt_tokens = [str(t.item()) for t in tgt[0, 1:]]

        if attn_weights['decoder_cross_attn']:
            w = attn_weights['decoder_cross_attn'][0][0]  # 第一层，第一个样本
            fig, axes = plt.subplots(1, min(4, w.shape[0]),
                                      figsize=(min(4, w.shape[0]) * 4, 4))
            if not hasattr(axes, '__len__'):
                axes = [axes]
            for h, ax in enumerate(axes):
                sns.heatmap(w[h].numpy(), annot=True, fmt='.2f', cmap='Blues',
                            xticklabels=src_tokens, yticklabels=tgt_tokens,
                            ax=ax, vmin=0, vmax=1)
                ax.set_title(f'Cross-Attention Head {h+1}', fontsize=10)
                ax.set_xlabel('源序列（Encoder）')
                ax.set_ylabel('目标序列（Decoder）')
            plt.suptitle('Decoder Cross-Attention 权重\n（Decoder 如何关注 Encoder 输出）',
                         fontsize=12)
            plt.tight_layout()
            plt.savefig('cross_attention.png', dpi=150, bbox_inches='tight')
            plt.show()

        # 损失曲线
        plt.figure(figsize=(8, 4))
        plt.plot(losses, 'b-', alpha=0.8)
        plt.title('Transformer 训练损失（序列反转任务）', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('transformer_loss.png', dpi=150, bbox_inches='tight')
        plt.show()

        return model, losses

    model, losses = train_toy_translation()


    # ─────────────────────────────────────────────────────────
    # 训练 MiniGPT 做字符级语言模型
    # ─────────────────────────────────────────────────────────

    def train_mini_gpt():
        """训练 MiniGPT 学习简单的文本模式"""
        torch.manual_seed(42)

        # 训练数据：重复的数字模式
        text = "0123456789" * 200
        chars = sorted(set(text))
        char2idx = {c: i for i, c in enumerate(chars)}
        idx2char = {i: c for c, i in char2idx.items()}
        vocab_size = len(chars)

        data = torch.tensor([char2idx[c] for c in text])

        model = MiniGPT(vocab_size=vocab_size, d_model=64, num_heads=4,
                        num_layers=2, d_ff=256, max_len=100)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        seq_len = 20
        batch_size = 32

        losses = []
        for step in range(500):
            # 随机采样批次
            idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x = torch.stack([data[i:i+seq_len] for i in idx])
            y = torch.stack([data[i+1:i+seq_len+1] for i in idx])

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # 生成文本
        start = torch.tensor([[char2idx['0']]])
        generated = model.generate(start, max_new_tokens=30, temperature=0.5)
        generated_text = ''.join([idx2char[i.item()] for i in generated[0]])
        print(f"\nMiniGPT 生成文本: {generated_text}")
        print(f"期望模式: 0123456789012345678901234567890")

        return model, losses

    gpt_model, gpt_losses = train_mini_gpt()

    # ============================================================
    # 代码段 2
    # ============================================================

    def transformer_debug_panel():
        """
        交互式参数调试：改变超参数，观察模型行为
        """
        configs = [
            {'d_model': 32,  'num_heads': 2, 'num_layers': 1, 'label': '极小'},
            {'d_model': 64,  'num_heads': 4, 'num_layers': 2, 'label': '小'},
            {'d_model': 128, 'num_heads': 8, 'num_layers': 4, 'label': '中'},
            {'d_model': 256, 'num_heads': 8, 'num_layers': 6, 'label': '大'},
        ]

        print("Transformer 参数对比：")
        print(f"{'配置':6s} {'d_model':8s} {'heads':6s} {'layers':7s} {'参数量':12s} {'内存(MB)':10s}")
        print("-" * 55)

        for cfg in configs:
            m = Transformer(
                src_vocab_size=1000, tgt_vocab_size=1000,
                d_model=cfg['d_model'], num_heads=cfg['num_heads'],
                num_encoder_layers=cfg['num_layers'],
                num_decoder_layers=cfg['num_layers'],
                d_ff=cfg['d_model'] * 4,
            )
            n_params = sum(p.numel() for p in m.parameters())
            mem_mb = n_params * 4 / 1024 / 1024  # float32
            print(f"{cfg['label']:6s} {cfg['d_model']:8d} {cfg['num_heads']:6d} "
                  f"{cfg['num_layers']:7d} {n_params:12,d} {mem_mb:10.2f}")

    transformer_debug_panel()
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part4_transformer/04_minimal_transformer.py", e)
