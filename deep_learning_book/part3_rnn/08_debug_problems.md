# RNN/LSTM 调试问题集：15个常见错误与解决方案

## 1. 梯度爆炸：Loss 突然变 NaN

**症状：**
```python
Epoch 5: loss = 0.45
Epoch 6: loss = nan
```

**根本原因：** LSTM 的梯度在长序列中指数增长

**解决方案：**
```python
# 始终添加梯度裁剪！
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 关键
optimizer.step()
```

## 2. 梯度消失：长序列前半段学不到

**症状：**
```python
# 只记住最近的输入
输入: [A, B, C, D, E, F, G]
模型只对 E, F, G 有响应
```

**根本原因：** 遗忘门偏置初始化不当，过早遗忘

**解决方案：**
```python
# LSTM 遗忘门偏置设为 1（鼓励记忆保留）
for name, param in model.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        # PyTorch LSTM bias 顺序: [input, forget, cell, output]
        param.data[n//4:n//2].fill_(1.0)  # 遗忘门偏置=1
```

## 3. 隐藏状态维度错误

**症状：**
```python
RuntimeError: Expected hidden[0] size (2, 8, 64), got (1, 8, 64)
```

**根本原因：** 隐藏状态的层数/方向数不匹配

**解决方案：**
```python
def init_hidden(model, batch_size):
    """正确初始化隐藏状态"""
    num_layers = model.num_layers
    hidden_size = model.hidden_size
    directions = 2 if model.bidirectional else 1

    h = torch.zeros(num_layers * directions, batch_size, hidden_size)
    c = torch.zeros(num_layers * directions, batch_size, hidden_size)
    return (h, c)

# 使用
h = init_hidden(model, batch_size=32)
output, h = model(x, h)
```

## 4. 序列长度不一致导致报错

**症状：**
```python
RuntimeError: stack expects each tensor to be equal size
```

**根本原因：** 批次中不同样本长度不同

**解决方案：**
```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# 方法1：填充到相同长度
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
lengths = torch.tensor([len(s) for s in sequences])

# 方法2：PackedSequence（推荐）
embedded = model.embedding(padded)
packed = pack_padded_sequence(embedded, lengths.cpu(),
                               batch_first=True, enforce_sorted=False)
output, hidden = model.lstm(packed)
```

## 5. PAD token 参与了损失计算

**症状：**
```python
# 分类准确率异常低
Test accuracy: 30%  # 应该 >80%
```

**根本原因：** PAD 位置的预测被计入损失

**解决方案：**
```python
# 方法1：CrossEntropyLoss 忽略 PAD
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD

# 方法2：手动 mask
mask = (targets != 0)  # 非 PAD 位置
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='none')
loss = (loss.view(-1) * mask.view(-1)).sum() / mask.sum()
```

## 6. Teacher Forcing 导致 Exposure Bias

**症状：**
```python
# 训练 BLEU=35，推理 BLEU=15（差距巨大）
```

**根本原因：** 训练用真实标签，推理用模型预测，分布不一致

**解决方案：**
```python
# 预定采样：逐步降低 teacher forcing 比率
def get_tf_ratio(epoch, k=10):
    """逆 sigmoid 衰减"""
    return k / (k + np.exp(epoch / k))

for epoch in range(num_epochs):
    tf_ratio = get_tf_ratio(epoch)
    output = model(src, tgt, teacher_forcing_ratio=tf_ratio)
```

## 7. 双向 LSTM 隐藏状态拼接错误

**症状：**
```python
# 分类性能不如单向 LSTM
BiLSTM: 85% < LSTM: 88%
```

**根本原因：** 双向 LSTM 隐藏状态拼接方式错误

**解决方案：**
```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True,
                            batch_first=True)
        # 注意：全连接层输入是 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embedded)

        # 正确：拼接正向最后层和反向最后层
        # h_n 形状: [num_layers * 2, batch, hidden]
        h_forward = h_n[-2]   # 正向最后层
        h_backward = h_n[-1]  # 反向最后层
        h = torch.cat([h_forward, h_backward], dim=-1)  # [batch, hidden*2]

        return self.fc(h)
```

## 8. Dropout 位置不对

**症状：**
```python
# 过拟合，训练准确率远高于测试
Train: 95%, Test: 70%
```

**根本原因：** Dropout 加在了错误的位置

**解决方案：**
```python
class ProperDropoutLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        # LSTM 的 dropout 加在层间（不是时间步间！）
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        # 全连接层前加 Dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.lstm(self.embedding(x))
        # 取最后时间步 + Dropout
        out = self.dropout(output[:, -1, :])
        return self.fc(out)
```

## 9. Embedding 梯度更新问题

**症状：**
```python
# 词汇表很大但训练集小，罕见词 embedding 学不好
```

**根本原因：** 罕见词梯度更新不足

**解决方案：**
```python
# 方法1：冻结低频词的 embedding
model.embedding.weight.requires_grad = False

# 方法2：使用预训练 embedding
with torch.no_grad():
    model.embedding.weight.copy_(pretrained_embeddings)
    # 可选：冻结
    model.embedding.weight.requires_grad = False

# 方法3：分层学习率
optimizer = torch.optim.Adam([
    {'params': model.embedding.parameters(), 'lr': 0.0001},   # 慢
    {'params': model.lstm.parameters(), 'lr': 0.001},
    {'params': model.fc.parameters(), 'lr': 0.01},             # 快
])
```

## 10. 自回归生成重复/退化

**症状：**
```python
# 生成文本重复
Output: "the the the the the the the"
```

**根本原因：** temperature 太低或 top-k 太小

**解决方案：**
```python
def generate_with_control(model, start_token, max_len=50,
                          temperature=1.0, top_k=0, top_p=0.0):
    """带温度和采样的文本生成"""
    model.eval()
    x = torch.tensor([[start_token]])
    h = None
    generated = []

    with torch.no_grad():
        for _ in range(max_len):
            logits, h = model(x, h)
            logits = logits[0, -1, :] / temperature  # 温度控制

            # Top-k 过滤
            if top_k > 0:
                top_vals, top_idx = logits.topk(top_k)
                logits[logits < top_vals[-1]] = -float('inf')

            # Top-p（核采样）
            if top_p > 0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                indices_to_remove = sorted_idx[remove_mask]
                logits[indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())

            x = next_token.unsqueeze(0)

    return generated

# 使用：temperature=0.8, top_k=50 是较好的默认值
```

## 11. PackedSequence 解包错误

**症状：**
```python
RuntimeError: cannot call DATA on non-data tensor
```

**根本原因：** 对 PackedSequence 直接做操作

**解决方案：**
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 正确流程
embedded = model.embedding(padded_input)
packed = pack_padded_sequence(embedded, lengths, batch_first=True)
packed_output, hidden = model.lstm(packed)
# 解包后再做后续操作
output, _ = pad_packed_sequence(packed_output, batch_first=True)
logits = model.fc(output)  # 现在是普通张量
```

## 12. 序列批次顺序影响 PackedSequence

**症状：**
```python
RuntimeError: lengths array has to be sorted in decreasing order
```

**根本原因：** 旧版 PyTorch 要求 PackedSequence 的输入按长度降序排列

**解决方案：**
```python
# 方法1：使用 enforce_sorted=False（PyTorch 1.1+）
packed = pack_padded_sequence(embedded, lengths,
                               batch_first=True, enforce_sorted=False)

# 方法2：手动排序
sorted_lengths, sorted_idx = lengths.sort(descending=True)
sorted_input = padded_input[sorted_idx]
embedded = model.embedding(sorted_input)
packed = pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
```

## 13. GPU 上隐藏状态未移到正确设备

**症状：**
```python
RuntimeError: Expected all tensors to be on the same device
```

**根本原因：** 初始化的隐藏状态在 CPU，模型在 GPU

**解决方案：**
```python
def init_hidden(model, batch_size, device):
    num_layers = model.num_layers
    hidden_size = model.hidden_size
    directions = 2 if model.bidirectional else 1

    h = torch.zeros(num_layers * directions, batch_size, hidden_size, device=device)
    c = torch.zeros(num_layers * directions, batch_size, hidden_size, device=device)
    return (h, c)

# 使用：传入正确的设备
device = next(model.parameters()).device
h = init_hidden(model, batch_size=32, device=device)
```

## 14. 多层 LSTM 的 Dropout 只在层间

**症状：**
```python
# 单层 LSTM 设了 dropout 但没有效果
model = nn.LSTM(embed_size, hidden_size, num_layers=1, dropout=0.3)
# dropout=0.3 被忽略！因为只有1层就没有层间
```

**根本原因：** PyTorch LSTM 的 dropout 只在层间生效

**解决方案：**
```python
# 单层 LSTM 需要手动加 Dropout
class SingleLayerLSTMWithDropout(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1)
        self.dropout = nn.Dropout(dropout)  # 手动加

    def forward(self, x, h=None):
        out, h = self.lstm(self.dropout(x))  # 输入前 dropout
        return out, h
```

## 15. 推理时忘记 model.eval()

**症状：**
```python
# 推理结果不稳定（每次运行不同）
Run 1: "I love deep"
Run 2: "I like machine"  # 同一输入，不同输出
```

**根本原因：** Dropout 和 BatchNorm 在 train 模式下随机

**解决方案：**
```python
# 推理前必须 eval()
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# 推理后恢复 train
model.train()
```

---

## 调试工具箱

```python
# 1. 检查梯度流
def check_rnn_gradient_flow(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            grad_mean = p.grad.abs().mean().item()
            grad_max = p.grad.abs().max().item()
            if grad_norm < 1e-7:
                status = "⚠ 梯度消失"
            elif grad_norm > 10:
                status = "⚠ 梯度爆炸"
            else:
                status = "✓ 正常"
            print(f"{name:30s} norm={grad_norm:.6f} mean={grad_mean:.6f} {status}")

# 2. 检查隐藏状态
def check_hidden_state(hidden, name="hidden"):
    h, c = hidden
    print(f"{name}: h_norm={h.norm():.4f}, c_norm={c.norm():.4f}")
    if torch.isnan(h).any():
        print(f"  ⚠ {name} 包含 NaN！")
    if torch.isinf(c).any():
        print(f"  ⚠ {name} 包含 Inf！")

# 3. 梯度范数随时间步的变化
def plot_gradient_per_step(model, x, y):
    """可视化每个时间步的梯度强度"""
    model.zero_grad()
    output, _ = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()

    # LSTM 的 input gradient 反映了每个时间步的梯度强度
    if x.grad is not None:
        grad_per_step = x.grad[0, :, 0].abs().detach().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(grad_per_step, 'b-', linewidth=2)
        plt.xlabel('时间步（0=最早）')
        plt.ylabel('梯度绝对值')
        plt.title('梯度强度随时间步的变化')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()
```

掌握这 15 个问题，RNN/LSTM 调试能力将大幅提升！
