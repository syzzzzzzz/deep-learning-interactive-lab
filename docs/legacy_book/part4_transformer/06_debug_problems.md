# Transformer 调试问题集：15个常见错误与解决方案

## 1. NaN Loss：注意力分数溢出

**症状：**
```python
Epoch 1, Step 100: loss = nan
```

**根本原因：**
```python
# 错误代码
scores = torch.matmul(Q, K.transpose(-2, -1))  # 未缩放
attn = F.softmax(scores, dim=-1)  # softmax 输入过大导致 exp 溢出
```

**解决方案：**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    正确的缩放点积注意力
    Q, K, V: [batch, num_heads, seq_len, d_k]
    """
    d_k = Q.size(-1)
    # 关键：除以 sqrt(d_k) 防止内积过大
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 不要用 -inf
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

## 2. 梯度消失：深层 Transformer 训练困难

**症状：**
```
Layer 1 grad norm: 0.5
Layer 6 grad norm: 0.001
Layer 12 grad norm: 1e-8  # 梯度几乎消失
```

**根本原因：Post-Norm 结构导致梯度衰减**

**错误实现（Post-Norm）：**
```python
class TransformerLayerPostNorm(nn.Module):
    def forward(self, x):
        # 先计算，后归一化
        x = x + self.attention(x)
        x = self.norm1(x)  # 梯度需要经过 norm
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x
```

**正确实现（Pre-Norm）：**
```python
class TransformerLayerPreNorm(nn.Module):
    def forward(self, x):
        # 先归一化，后计算
        x = x + self.attention(self.norm1(x))  # 残差直连
        x = x + self.ffn(self.norm2(x))
        return x
```

**对比：**
- Post-Norm：梯度路径经过 12 个 LayerNorm → 梯度衰减
- Pre-Norm：残差连接提供直通路径 → 梯度稳定

## 3. 因果掩码错误：未来信息泄露

**症状：**
```python
# 验证集困惑度异常低，但生成质量差
Val perplexity: 1.2  # 太好了，不正常
```

**错误代码：**
```python
# 错误：掩码形状不对
mask = torch.tril(torch.ones(seq_len, seq_len))  # [seq_len, seq_len]
attn = self.attention(x, mask=mask)  # 广播错误
```

**正确实现：**
```python
def create_causal_mask(seq_len, device):
    """
    创建因果掩码，防止看到未来 token
    返回: [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # 扩展维度以匹配 [batch, heads, seq_len, seq_len]
    return mask.unsqueeze(0).unsqueeze(0)

# 使用
mask = create_causal_mask(seq_len, device)
scores = scores.masked_fill(mask == 0, -1e9)
```

**验证掩码：**
```python
# 可视化掩码
import matplotlib.pyplot as plt
mask = create_causal_mask(10, 'cpu')
plt.imshow(mask[0, 0], cmap='Blues')
plt.title('因果掩码（白色=可见，蓝色=遮蔽）')
plt.xlabel('Key 位置')
plt.ylabel('Query 位置')
```

## 4. Padding 掩码未正确应用

**症状：**
```python
# 短序列的 loss 异常高
Seq len 10: loss = 0.5
Seq len 50: loss = 2.3  # padding 干扰了注意力
```

**错误代码：**
```python
# 只在输入做了 padding，但注意力没有掩码
x = F.pad(x, (0, 0, 0, max_len - seq_len))  # padding
attn = self.attention(x)  # 错误：padding 位置参与了注意力计算
```

**正确实现：**
```python
def create_padding_mask(seq_lengths, max_len):
    """
    seq_lengths: [batch] 每个样本的真实长度
    返回: [batch, 1, 1, max_len]
    """
    batch_size = len(seq_lengths)
    mask = torch.arange(max_len).expand(batch_size, max_len)
    mask = mask < seq_lengths.unsqueeze(1)  # [batch, max_len]
    return mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_len]

# 使用
padding_mask = create_padding_mask(seq_lengths, max_len)
scores = scores.masked_fill(~padding_mask, -1e9)
```

## 5. 位置编码在批次中错位

**症状：**
```python
# Batch size > 1 时性能下降
Batch=1: accuracy = 85%
Batch=32: accuracy = 60%  # 位置编码加错了
```

**错误代码：**
```python
class PositionalEncoding(nn.Module):
    def forward(self, x):
        # 错误：直接加到整个 batch
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len, :]  # [seq_len, d_model]
        return x + pos_enc  # 广播错误
```

**正确实现：**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        # 正确：扩展 batch 维度
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        return x + pos_enc  # 正确广播到 [batch, seq_len, d_model]
```

## 6. 内存溢出：存储所有注意力权重

**症状：**
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**错误代码：**
```python
class Transformer(nn.Module):
    def forward(self, x):
        attn_weights_list = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_weights_list.append(attn)  # 错误：存储 [batch, heads, N, N]
        return x, attn_weights_list  # 12层 × N² 内存
```

**解决方案：**
```python
class Transformer(nn.Module):
    def forward(self, x, return_attention=False):
        attn_weights = None
        for layer in self.layers:
            x, attn = layer(x)
            # 只在需要时返回最后一层的注意力
            if return_attention:
                attn_weights = attn
        return x, attn_weights

# 训练时不返回注意力
output, _ = model(x, return_attention=False)
```

## 7. 低效注意力实现导致训练慢

**症状：**
```
Training speed: 50 tokens/sec  # 应该 >500
```

**错误代码：**
```python
# 逐头计算注意力
for i in range(num_heads):
    Q_head = Q[:, i, :, :]
    K_head = K[:, i, :, :]
    V_head = V[:, i, :, :]
    out[:, i, :, :] = attention(Q_head, K_head, V_head)  # 慢
```

**正确实现：**
```python
def multi_head_attention(Q, K, V):
    """
    Q, K, V: [batch, num_heads, seq_len, d_k]
    一次性计算所有头
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    # 所有头并行计算
    output = torch.matmul(attn, V)  # [batch, num_heads, seq_len, d_k]
    return output
```

## 8. 交叉注意力的 Key/Value 来源错误

**症状：**
```python
# Encoder-Decoder 模型翻译质量差
BLEU score: 5.2  # 应该 >25
```

**错误代码：**
```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_output):
        # 错误：交叉注意力的 K, V 应该来自 encoder
        x = x + self.self_attn(x, x, x)  # 自注意力 ✓
        x = x + self.cross_attn(x, x, x)  # 错误：K, V 应该是 encoder_output
        x = x + self.ffn(x)
        return x
```

**正确实现：**
```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_output):
        # 自注意力：Q, K, V 都来自 decoder
        x = x + self.self_attn(
            query=x, key=x, value=x
        )
        
        # 交叉注意力：Q 来自 decoder，K, V 来自 encoder
        x = x + self.cross_attn(
            query=x, 
            key=encoder_output,    # 来自 encoder
            value=encoder_output   # 来自 encoder
        )
        
        x = x + self.ffn(x)
        return x
```

## 9. Label Smoothing 实现错误

**症状：**
```python
# 训练 loss 不下降
Epoch 10: loss = 4.5 (no improvement)
```

**错误代码：**
```python
# 错误：直接修改 one-hot 标签
labels_smooth = labels * 0.9 + 0.1  # 错误
loss = F.cross_entropy(logits, labels_smooth)
```

**正确实现：**
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, targets):
        """
        logits: [batch, num_classes]
        targets: [batch] (类别索引)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
```

## 10. 学习率预热未应用

**症状：**
```python
# 训练初期 loss 爆炸
Step 10: loss = 15.3
Step 20: loss = nan
```

**错误代码：**
```python
# 直接使用大学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 太大
```

**正确实现：**
```python
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 使用
optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)

for batch in dataloader:
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # 每步更新学习率
```

## 11. 权重共享：Embedding 和输出层

**症状：**
```python
# 模型参数过多，训练慢
Total params: 180M  # 应该 ~120M
```

**未优化代码：**
```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)  # 独立参数
```

**优化实现（权重共享）：**
```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 输出层共享 embedding 权重
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # 共享
    
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        x = self.transformer(x)
        logits = self.output_proj(x)
        return logits
```

## 12. Beam Search 实现错误

**症状：**
```python
# 生成结果重复或质量差
Output: "the the the the..."
```

**错误代码：**
```python
# 错误：没有正确扩展 beam
for step in range(max_len):
    logits = model(current_tokens)
    top_k = torch.topk(logits, k=beam_size)
    current_tokens = top_k.indices  # 错误：丢失了之前的路径
```

**正确实现：**
```python
def beam_search(model, start_token, max_len, beam_size=5):
    device = next(model.parameters()).device
    
    # 初始化：[beam_size, 1]
    beams = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = -float('inf')  # 只有第一个 beam 有效
    
    for step in range(max_len):
        # 获取所有 beam 的预测
        logits = model(beams)[:, -1, :]  # [beam_size, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算所有候选的分数
        candidate_scores = beam_scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
        candidate_scores = candidate_scores.view(-1)  # [beam_size * vocab_size]
        
        # 选择 top-k
        top_scores, top_indices = torch.topk(candidate_scores, beam_size)
        
        # 恢复 beam 索引和 token 索引
        beam_indices = top_indices // logits.size(-1)
        token_indices = top_indices % logits.size(-1)
        
        # 更新 beams
        beams = torch.cat([
            beams[beam_indices],
            token_indices.unsqueeze(1)
        ], dim=1)
        beam_scores = top_scores
    
    # 返回得分最高的序列
    best_beam = beams[beam_scores.argmax()]
    return best_beam
```

## 13. Tokenizer Padding 方向不匹配

**症状：**
```python
# 批次训练时性能下降
Single sample: 90% accuracy
Batched: 65% accuracy
```

**错误代码：**
```python
# Tokenizer 左填充，但模型期望右填充
tokens = tokenizer(texts, padding='left')  # 错误
# 位置编码错位
```

**解决方案：**
```python
# 统一使用右填充（推荐）
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'right'  # 明确指定

tokens = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# 创建对应的 padding mask
attention_mask = tokens['attention_mask']  # [batch, seq_len]
```

## 14. 梯度累积的 Loss 缩放错误

**症状：**
```python
# 梯度累积后 loss 异常
Accumulation steps=4: loss = 0.25  # 应该 ~1.0
```

**错误代码：**
```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()  # 错误：没有缩放
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**正确实现：**
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch)
    # 关键：除以累积步数
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # 记录时恢复原始 loss
    wandb.log({'loss': loss.item() * accumulation_steps})
```

## 15. 混合精度训练 NaN 问题

**症状：**
```python
# 使用 FP16 后出现 NaN
Step 100: loss = nan
```

**错误代码：**
```python
# 直接使用 autocast 没有梯度缩放
with torch.cuda.amp.autocast():
    loss = model(batch)
loss.backward()  # 梯度下溢
optimizer.step()
```

**正确实现：**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 前向传播使用混合精度
    with autocast():
        loss = model(batch)
    
    # 梯度缩放防止下溢
    scaler.scale(loss).backward()
    
    # 梯度裁剪（在 unscale 后）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

**额外检查：**
```python
# 监控梯度和激活值
def check_nan_inf(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN in {name}.grad")
            if torch.isinf(param.grad).any():
                print(f"Inf in {name}.grad")

# 训练循环中
if step % 100 == 0:
    check_nan_inf(model)
```

---

## 调试工具箱

```python
# 1. 可视化注意力权重
def plot_attention(attn_weights, tokens):
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights.cpu().numpy(), 
                xticklabels=tokens, yticklabels=tokens,
                cmap='viridis')
    plt.title('Attention Weights')
    plt.show()

# 2. 检查梯度流
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

# 3. 监控激活值分布
def log_activation_stats(activations, name):
    print(f"{name}: mean={activations.mean():.3f}, "
          f"std={activations.std():.3f}, "
          f"min={activations.min():.3f}, "
          f"max={activations.max():.3f}")
```

掌握这 15 个常见问题，你的 Transformer 调试能力将大幅提升！
