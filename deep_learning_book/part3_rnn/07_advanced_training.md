# RNN 高级训练技术：Teacher Forcing / 预定采样 / BPTT / 梯度策略

## 1. Teacher Forcing

### 1.1 原理

```
解码器生成第 t 步时，需要第 t-1 步的输出作为输入

两种策略：

自回归（无 Teacher Forcing）：
  input_t = model_output_{t-1}  ← 用模型自己的预测
  问题：训练早期预测很差，错误会累积

Teacher Forcing：
  input_t = ground_truth_{t-1}   ← 用真实标签
  优势：训练稳定，收敛快
  问题：推理时没有真实标签，训练-推理不一致（exposure bias）
```

### 1.2 实现与对比

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SimpleSeq2Seq(nn.Module):
    """简化 Seq2Seq 用于演示 Teacher Forcing"""
    def __init__(self, vocab_size=20, embed_size=16, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt, tf_ratio=0.0):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        tf_ratio: teacher forcing 比率 (0=纯自回归, 1=纯 teacher forcing)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]

        # 编码
        _, hidden = self.encoder(self.embedding(src))

        # 解码
        outputs = []
        input_tok = tgt[:, 0:1]  # <SOS>

        for t in range(1, tgt_len):
            emb = self.embedding(input_tok)
            out, hidden = self.decoder(emb, hidden)
            logits = self.fc(out.squeeze(1))
            outputs.append(logits)

            # 决定下一步输入
            if np.random.random() < tf_ratio:
                input_tok = tgt[:, t:t+1]      # Teacher Forcing
            else:
                input_tok = logits.argmax(1, keepdim=True)  # 自回归

        return torch.stack(outputs, dim=1)


def compare_teacher_forcing_ratios():
    """
    对比不同 Teacher Forcing 比率的效果

    典型发现：
    - tf=1.0：训练最快，但推理时性能骤降（exposure bias）
    - tf=0.0：训练慢，但推理性能更一致
    - tf=0.5：折中方案
    - 预定采样：从 1.0 逐渐降到 0.0，最佳实践
    """
    torch.manual_seed(42)
    np.random.seed(42)

    vocab_size = 20
    seq_len = 10
    n_samples = 500

    # 生成简单复制任务：输出 = 输入的逆序
    src = torch.randint(3, vocab_size, (n_samples, seq_len))
    tgt = torch.cat([
        torch.full((n_samples, 1), 1),  # <SOS>
        src.flip(dims=[1]),
        torch.full((n_samples, 1), 2),  # <EOS>
    ], dim=1)

    split = 400
    src_train, tgt_train = src[:split], tgt[:split]
    src_test, tgt_test = src[split:], tgt[split:]

    tf_ratios = [0.0, 0.3, 0.5, 1.0]
    results = {}

    for tf in tf_ratios:
        model = SimpleSeq2Seq(vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        train_losses, val_accs = [], []

        for epoch in range(60):
            model.train()
            logits = model(src_train, tgt_train, tf_ratio=tf)
            loss = criterion(logits.reshape(-1, vocab_size),
                            tgt_train[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # 验证（始终无 teacher forcing）
            model.eval()
            with torch.no_grad():
                val_logits = model(src_test, tgt_test, tf_ratio=0.0)
                preds = val_logits.argmax(-1)
                acc = (preds == tgt_test[:, 1:]).float().mean().item()
                val_accs.append(acc)

        results[f'tf={tf}'] = {
            'train_loss': train_losses,
            'val_acc': val_accs,
            'final_acc': val_accs[-1],
        }
        print(f"tf={tf}: 最终验证准确率 = {val_accs[-1]:.2%}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#C44E52', '#DD8452', '#4C72B0', '#55A868']

    for (name, res), color in zip(results.items(), colors):
        axes[0].plot(res['train_loss'], color=color, label=name, linewidth=1.5)
    axes[0].set_title('训练损失', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for (name, res), color in zip(results.items(), colors):
        axes[1].plot(res['val_acc'], color=color, label=name, linewidth=1.5)
    axes[1].set_title('验证准确率（推理模式）', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Teacher Forcing 比率对比\n（验证时统一用自回归模式）',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('teacher_forcing_compare.png', dpi=150)
    plt.show()


compare_teacher_forcing_ratios()
```

---

## 2. 预定采样（Scheduled Sampling）

### 2.1 原理

```
训练初期：高 Teacher Forcing 比率（稳定训练）
训练后期：低 Teacher Forcing 比率（减少 exposure bias）

渐进过渡策略：
- 线性衰减：tf_ratio = max(0, 1 - epoch / k)
- 指数衰减：tf_ratio = k^epoch (k < 1)
- 逆 sigmoid：tf_ratio = k / (k + exp(epoch / k))
```

### 2.2 实现

```python
class ScheduledSampling:
    """
    预定采样策略

    用法：
        scheduler = ScheduledSampling(strategy='linear', k=20)
        for epoch in range(num_epochs):
            tf_ratio = scheduler.get_ratio(epoch)
            # ... 训练时使用 tf_ratio
    """
    def __init__(self, strategy='inverse_sigmoid', k=5):
        self.strategy = strategy
        self.k = k

    def get_ratio(self, epoch):
        if self.strategy == 'linear':
            return max(0.0, 1.0 - epoch / self.k)
        elif self.strategy == 'exponential':
            return self.k ** epoch  # k < 1, 如 0.95
        elif self.strategy == 'inverse_sigmoid':
            return self.k / (self.k + np.exp(epoch / self.k))
        else:
            raise ValueError(f"未知策略: {self.strategy}")


def visualize_scheduled_sampling():
    """可视化不同预定采样策略"""
    epochs = np.arange(0, 50)

    strategies = {
        '线性 (k=30)': ScheduledSampling('linear', k=30),
        '指数 (k=0.95)': ScheduledSampling('exponential', k=0.95),
        '逆 sigmoid (k=5)': ScheduledSampling('inverse_sigmoid', k=5),
        '逆 sigmoid (k=10)': ScheduledSampling('inverse_sigmoid', k=10),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, scheduler in strategies.items():
        ratios = [scheduler.get_ratio(e) for e in epochs]
        ax.plot(epochs, ratios, linewidth=2, label=name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Teacher Forcing 比率')
    ax.set_title('预定采样策略对比\n（从 1.0 逐步降到 0.0）', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('scheduled_sampling.png', dpi=150)
    plt.show()


visualize_scheduled_sampling()
```

---

## 3. BPTT：沿时间反向传播

### 3.1 原理

```
标准 BPTT：序列长度 T，梯度要穿越 T 步
问题：T 很大时，梯度消失/爆炸，内存不足

截断 BPTT（Truncated BPTT）：
- 将长序列分成 K 段，每段长度 T/K
- 每段独立计算梯度
- 段间传递隐藏状态（但梯度不跨段）

标准 BPTT：
  ┌───────────────────────────────────────┐
  │ 梯度穿越整个序列（T=100 步）           │
  └───────────────────────────────────────┘

截断 BPTT（segment=20）：
  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
  │ 梯度20步  ││ 梯度20步  ││ 梯度20步  ││ 梯度20步  ││ 梯度20步  │
  └────h─────┘└────h─────┘└────h─────┘└────h─────┘└────h─────┘
       ↑ 隐藏状态传递，但梯度不跨段
```

### 3.2 实现

```python
def truncated_bptt_train(model, data, seq_len=20, segment_len=5,
                          lr=0.01, n_epochs=50):
    """
    截断 BPTT 训练

    将长序列分成短段，每段独立反向传播
    隐藏状态在段间传递（detach 梯度图）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        total_loss = 0
        h = None  # 初始隐藏状态

        # 将数据分成段
        n_segments = seq_len // segment_len

        for seg in range(n_segments):
            start = seg * segment_len
            end = start + segment_len
            x_seg = data[start:end].unsqueeze(0).unsqueeze(-1)  # [1, seg_len, 1]
            y_seg = data[start+1:end+1].unsqueeze(0).unsqueeze(-1)

            # 前向传播
            if h is not None:
                # detach：断开梯度图，防止梯度跨段
                h = (h[0].detach(), h[1].detach())

            out, h = model(x_seg, h)
            loss = criterion(out, y_seg)

            # 反向传播（只在当前段内）
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss / n_segments)

    return losses


def compare_bptt_strategies():
    """对比标准 BPTT 和截断 BPTT"""
    torch.manual_seed(42)

    # 生成长序列
    t = np.linspace(0, 20 * np.pi, 2000)
    data = torch.tensor(np.sin(t) + 0.1 * np.random.randn(len(t)),
                        dtype=torch.float32)

    seq_len = 100
    segment_lens = [5, 10, 20, 50, 100]  # 100 = 标准 BPTT

    results = {}

    for seg_len in segment_lens:
        model = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        losses = truncated_bptt_train(model, data, seq_len=seq_len,
                                       segment_len=seg_len, n_epochs=80)
        label = f'段长={seg_len}' if seg_len < 100 else '标准 BPTT'
        results[label] = losses
        print(f"{label}: 最终损失 = {losses[-1]:.4f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, losses in results.items():
        ax.plot(losses, linewidth=1.5, label=name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('平均段损失')
    ax.set_title('截断 BPTT 段长度对比\n（段越长，梯度传播越远，但内存越多）',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('truncated_bptt.png', dpi=150)
    plt.show()


compare_bptt_strategies()
```

---

## 4. 梯度裁剪策略

### 4.1 三种裁剪方法

```python
def demonstrate_gradient_clipping():
    """
    对比三种梯度裁剪策略

    1. 按范数裁剪：||g|| > max_norm 时缩放
    2. 按值裁剪：g = clamp(g, -clip_value, clip_value)
    3. 按全局范数裁剪：所有参数的梯度范数一起限制
    """
    torch.manual_seed(42)

    model = nn.LSTM(input_size=4, hidden_size=64, num_layers=3, batch_first=True)

    x = torch.randn(8, 50, 4)
    y = torch.randn(8, 50, 64)

    output, _ = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()

    # 原始梯度分布
    grad_norms_orig = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms_orig.append(p.grad.norm().item())

    # 方法1：按全局范数裁剪
    model_clone = {name: p.clone() for name, p in model.named_parameters()}
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norms_norm = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms_norm.append(p.grad.norm().item())

    # 重新计算梯度
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    output, _ = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()

    # 方法2：按值裁剪
    for p in model.parameters():
        if p.grad is not None:
            p.grad.clamp_(-0.5, 0.5)
    grad_norms_value = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms_value.append(p.grad.norm().item())

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, norms, title in zip(axes,
        [grad_norms_orig, grad_norms_norm, grad_norms_value],
        ['原始梯度', f'按范数裁剪 (max=1.0)', '按值裁剪 (±0.5)']):
        ax.hist(norms, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('梯度范数')
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='裁剪阈值')
        ax.legend()

    plt.suptitle('梯度裁剪策略对比', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradient_clipping.png', dpi=150)
    plt.show()


demonstrate_gradient_clipping()
```

---

## 5. 训练最佳实践总结

```python
def training_best_practices():
    """RNN 训练最佳实践汇总"""

    practices = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              RNN/LSTM 训练最佳实践                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  1. 梯度裁剪：始终使用！                                      ║
    ║     clip_grad_norm_(params, max_norm=1.0~5.0)                ║
    ║                                                              ║
    ║  2. Teacher Forcing + 预定采样：                              ║
    ║     起步: tf_ratio=1.0                                       ║
    ║     逐步: tf_ratio → 0.0（逆 sigmoid 衰减）                   ║
    ║                                                              ║
    ║  3. 截断 BPTT：                                              ║
    ║     段长度 20~50 步                                          ║
    ║     段间 detach 隐藏状态                                      ║
    ║                                                              ║
    ║  4. 学习率：                                                  ║
    ║     Adam: lr=0.001~0.01                                      ║
    ║     SGD: lr=0.1 + 余弦退火                                   ║
    ║                                                              ║
    ║  5. 正则化：                                                  ║
    ║     Dropout: 层间 0.1~0.3                                    ║
    ║     Weight decay: 1e-4~1e-3                                  ║
    ║     Variational Dropout: 同一序列共享 dropout mask            ║
    ║                                                              ║
    ║  6. 初始化：                                                  ║
    ║     遗忘门偏置设为 1（LSTM 初始化技巧）                        ║
    ║     for name, p in model.named_parameters():                 ║
    ║         if 'bias' in name:                                   ║
    ║             n = p.size(0)                                   ║
    ║             p[n//4:n//2].fill_(1.0)  # 遗忘门偏置=1          ║
    ║                                                              ║
    ║  7. 序列长度：                                                ║
    ║     按长度分桶（减少 padding）                                ║
    ║     使用 pack_padded_sequence 跳过 PAD                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    print(practices)


training_best_practices()
```

---

## 小结

| 技术 | 作用 | 推荐设置 |
|------|------|----------|
| Teacher Forcing | 加速训练收敛 | 起步 1.0，逐步降到 0 |
| 预定采样 | 减少 exposure bias | 逆 sigmoid 衰减 |
| 截断 BPTT | 控制内存和梯度范围 | 段长 20~50 |
| 梯度裁剪（范数） | 防止梯度爆炸 | max_norm=1~5 |
| 梯度裁剪（值） | 限制单个梯度 | ±0.5 |
| 遗忘门偏置初始化 | 促进记忆保留 | 偏置=1 |
