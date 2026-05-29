# FlashAttention：高效注意力机制

## 1. 标准注意力的内存瓶颈

标准的 Transformer 注意力机制存在严重的内存问题：

```python
import torch
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """
    标准注意力实现
    Q, K, V: [batch, seq_len, d_model]
    内存复杂度: O(N^2) 其中 N = seq_len
    """
    d_k = Q.size(-1)
    # 计算注意力分数矩阵 [batch, N, N]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # 问题：这里需要存储完整的 N×N 矩阵到 HBM（高带宽内存）
    attn_weights = F.softmax(scores, dim=-1)  # [batch, N, N]
    # 再次访问 HBM 读取 attn_weights
    output = torch.matmul(attn_weights, V)  # [batch, N, d_model]
    return output
```

**内存瓶颈分析：**
- 注意力矩阵大小：`N × N`（N 是序列长度）
- 当 N=1024 时，单个注意力头需要 1024² = 1M 个浮点数
- 16 个注意力头 × 4 字节/float32 = 64MB
- 对于长序列（N=4096），单层就需要 1GB+ 内存

## 2. FlashAttention 的核心思想

FlashAttention 通过**分块计算**和**在线 softmax** 避免存储完整的注意力矩阵：

### 2.1 分块策略

```
标准注意力：
Q [N×d] × K^T [d×N] → S [N×N] → softmax → P [N×N] × V [N×d] → O [N×d]
         ↑ 需要存储到 HBM

FlashAttention：
将 Q, K, V 分成小块，每次只在 SRAM 中计算一小块
Q_blocks: [B_q × d]  (B_q << N)
K_blocks: [B_k × d]  (B_k << N)
V_blocks: [B_k × d]
```

### 2.2 在线 Softmax 算法

关键技巧：增量更新 softmax 的最大值和归一化因子

```python
def online_softmax_update(m_old, l_old, block_scores, block_values):
    """
    在线更新 softmax
    m_old: 之前块的最大值
    l_old: 之前块的归一化因子 (sum of exp)
    block_scores: 当前块的注意力分数
    """
    m_new = torch.max(m_old, torch.max(block_scores, dim=-1, keepdim=True)[0])
    
    # 重新缩放之前的结果
    scale_old = torch.exp(m_old - m_new)
    l_old_scaled = l_old * scale_old
    
    # 计算当前块的贡献
    exp_scores = torch.exp(block_scores - m_new)
    l_new = l_old_scaled + torch.sum(exp_scores, dim=-1, keepdim=True)
    
    return m_new, l_new, exp_scores
```

## 3. FlashAttention 完整实现

```python
import torch
import math

class FlashAttention:
    def __init__(self, block_size_q=64, block_size_k=64):
        """
        block_size_q: Q 的分块大小
        block_size_k: K, V 的分块大小
        """
        self.B_q = block_size_q
        self.B_k = block_size_k
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        返回: [batch, num_heads, seq_len, head_dim]
        """
        batch, num_heads, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        
        # 输出累加器
        O = torch.zeros_like(Q)
        # 每个 Q 块的统计量
        l = torch.zeros(batch, num_heads, N, 1, device=Q.device)  # 归一化因子
        m = torch.full((batch, num_heads, N, 1), -float('inf'), device=Q.device)  # 最大值
        
        # 外层循环：遍历 Q 的块
        for i in range(0, N, self.B_q):
            Q_block = Q[:, :, i:i+self.B_q, :]  # [batch, heads, B_q, d]
            O_block = torch.zeros_like(Q_block)
            l_block = torch.zeros(batch, num_heads, self.B_q, 1, device=Q.device)
            m_block = torch.full((batch, num_heads, self.B_q, 1), -float('inf'), device=Q.device)
            
            # 内层循环：遍历 K, V 的块
            for j in range(0, N, self.B_k):
                K_block = K[:, :, j:j+self.B_k, :]  # [batch, heads, B_k, d]
                V_block = V[:, :, j:j+self.B_k, :]  # [batch, heads, B_k, d]
                
                # 计算当前块的注意力分数
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale
                # S_block: [batch, heads, B_q, B_k]
                
                # 应用 mask（如果有）
                if mask is not None:
                    mask_block = mask[:, :, i:i+self.B_q, j:j+self.B_k]
                    S_block = S_block.masked_fill(mask_block == 0, -float('inf'))
                
                # 在线 softmax 更新
                m_new = torch.max(m_block, torch.max(S_block, dim=-1, keepdim=True)[0])
                
                # 重新缩放之前的累加结果
                scale_old = torch.exp(m_block - m_new)
                O_block = O_block * scale_old
                l_block = l_block * scale_old
                
                # 计算当前块的 softmax 和输出贡献
                P_block = torch.exp(S_block - m_new)  # [batch, heads, B_q, B_k]
                l_new = l_block + torch.sum(P_block, dim=-1, keepdim=True)
                
                # 累加当前块的输出
                O_block = O_block + torch.matmul(P_block, V_block)
                
                # 更新统计量
                m_block = m_new
                l_block = l_new
            
            # 最终归一化
            O[:, :, i:i+self.B_q, :] = O_block / l_block
            l[:, :, i:i+self.B_q, :] = l_block
            m[:, :, i:i+self.B_q, :] = m_block
        
        return O

# 使用示例
def test_flash_attention():
    batch, num_heads, seq_len, head_dim = 2, 8, 512, 64
    
    Q = torch.randn(batch, num_heads, seq_len, head_dim)
    K = torch.randn(batch, num_heads, seq_len, head_dim)
    V = torch.randn(batch, num_heads, seq_len, head_dim)
    
    # 标准注意力
    flash_attn = FlashAttention(block_size_q=64, block_size_k=64)
    output_flash = flash_attn.forward(Q, K, V)
    
    # 验证正确性（与标准实现对比）
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output_standard = torch.matmul(attn_weights, V)
    
    print(f"输出差异: {torch.max(torch.abs(output_flash - output_standard)).item():.6f}")
    print(f"Flash 输出形状: {output_flash.shape}")
```

## 4. IO 复杂度分析

### 4.1 标准注意力的 IO 复杂度

```
HBM 读取次数：
1. 读取 Q, K: 2Nd 次
2. 读取注意力矩阵 S: N² 次
3. 读取 V: Nd 次
总计: O(N² + Nd)

HBM 写入次数：
1. 写入 S: N² 次
2. 写入 P (softmax后): N² 次
3. 写入 O: Nd 次
总计: O(N²)

当 N >> d 时，主导项是 O(N²)
```

### 4.2 FlashAttention 的 IO 复杂度

```
假设 SRAM 大小为 M，块大小 B = sqrt(M/d)

HBM 读取次数：
- Q 的每个块读取一次: N/B 次
- K, V 的每个块被读取 N/B 次: (N/B)² 次
- 每次读取大小为 Bd
总计: O(N²d²/M)

当 M >> d 时，复杂度降为 O(N²d²/M)，相比标准的 O(N²d) 减少了 d/M 倍
```

## 5. 性能对比实验

```python
import time
import torch
import matplotlib.pyplot as plt

def benchmark_attention(seq_lengths, num_heads=8, head_dim=64):
    """
    对比标准注意力和 FlashAttention 的性能
    """
    results = {'standard': [], 'flash': [], 'memory_standard': [], 'memory_flash': []}
    
    for N in seq_lengths:
        Q = torch.randn(1, num_heads, N, head_dim, device='cuda')
        K = torch.randn(1, num_heads, N, head_dim, device='cuda')
        V = torch.randn(1, num_heads, N, head_dim, device='cuda')
        
        # 标准注意力
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out_std = torch.matmul(attn, V)
        torch.cuda.synchronize()
        time_std = time.time() - start
        mem_std = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # FlashAttention
        torch.cuda.reset_peak_memory_stats()
        flash_attn = FlashAttention(block_size_q=64, block_size_k=64)
        start = time.time()
        out_flash = flash_attn.forward(Q, K, V)
        torch.cuda.synchronize()
        time_flash = time.time() - start
        mem_flash = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        results['standard'].append(time_std * 1000)  # ms
        results['flash'].append(time_flash * 1000)
        results['memory_standard'].append(mem_std)
        results['memory_flash'].append(mem_flash)
        
        print(f"N={N}: 标准 {time_std*1000:.2f}ms ({mem_std:.1f}MB), "
              f"Flash {time_flash*1000:.2f}ms ({mem_flash:.1f}MB)")
    
    return results

# 运行基准测试
seq_lengths = [256, 512, 1024, 2048]
results = benchmark_attention(seq_lengths)

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(seq_lengths, results['standard'], 'o-', label='标准注意力')
ax1.plot(seq_lengths, results['flash'], 's-', label='FlashAttention')
ax1.set_xlabel('序列长度')
ax1.set_ylabel('时间 (ms)')
ax1.set_title('计算时间对比')
ax1.legend()
ax1.grid(True)

ax2.plot(seq_lengths, results['memory_standard'], 'o-', label='标准注意力')
ax2.plot(seq_lengths, results['memory_flash'], 's-', label='FlashAttention')
ax2.set_xlabel('序列长度')
ax2.set_ylabel('内存 (MB)')
ax2.set_title('内存使用对比')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('flash_attention_comparison.png', dpi=150)
```

## 6. 分块模式可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_tiling_pattern(N=16, B_q=4, B_k=4):
    """
    可视化 FlashAttention 的分块访问模式
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 标准注意力：一次性计算整个矩阵
    full_matrix = np.ones((N, N))
    axes[0].imshow(full_matrix, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('标准注意力\n一次性存储 N×N 矩阵', fontsize=12)
    axes[0].set_xlabel('Key 位置')
    axes[0].set_ylabel('Query 位置')
    
    # FlashAttention：分块计算
    tiled_matrix = np.zeros((N, N))
    for i in range(0, N, B_q):
        for j in range(0, N, B_k):
            tiled_matrix[i:i+B_q, j:j+B_k] = (i // B_q + j // B_k) % 2 + 0.3
    
    axes[1].imshow(tiled_matrix, cmap='Greens', vmin=0, vmax=1.5)
    axes[1].set_title(f'FlashAttention 分块\n块大小 {B_q}×{B_k}', fontsize=12)
    axes[1].set_xlabel('Key 位置')
    axes[1].set_ylabel('Query 位置')
    
    # 绘制网格线
    for i in range(0, N, B_q):
        axes[1].axhline(i - 0.5, color='black', linewidth=1)
    for j in range(0, N, B_k):
        axes[1].axvline(j - 0.5, color='black', linewidth=1)
    
    # 计算顺序示意
    order_matrix = np.zeros((N, N))
    order = 1
    for i in range(0, N, B_q):
        for j in range(0, N, B_k):
            order_matrix[i:i+B_q, j:j+B_k] = order
            order += 1
    
    im = axes[2].imshow(order_matrix, cmap='viridis')
    axes[2].set_title('块的计算顺序', fontsize=12)
    axes[2].set_xlabel('Key 位置')
    axes[2].set_ylabel('Query 位置')
    plt.colorbar(im, ax=axes[2], label='计算顺序')
    
    plt.tight_layout()
    plt.savefig('flash_attention_tiling.png', dpi=150)
    plt.show()

visualize_tiling_pattern(N=16, B_q=4, B_k=4)
```

## 7. 关键优势总结

| 特性 | 标准注意力 | FlashAttention |
|------|-----------|----------------|
| 内存复杂度 | O(N²) | O(N) |
| IO 复杂度 | O(N²d) | O(N²d²/M) |
| HBM 访问 | 多次读写 N² 矩阵 | 只读写 O(Nd) |
| 最大序列长度 | ~2K (16GB GPU) | ~16K (16GB GPU) |
| 训练速度 | 基准 | 2-4× 更快 |

**核心洞察：**
- GPU 的瓶颈不是计算（FLOPS），而是内存带宽（HBM ↔ SRAM）
- FlashAttention 通过分块计算，将中间结果保留在快速 SRAM 中
- 牺牲少量重复计算，换取大幅减少 HBM 访问
- 使得训练超长序列（16K+）成为可能

## 8. 实际应用建议

```python
# 在 PyTorch 中使用官方 FlashAttention
# pip install flash-attn

from flash_attn import flash_attn_func

def efficient_attention(q, k, v, causal=False):
    """
    q, k, v: [batch, seq_len, num_heads, head_dim]
    注意：官方实现要求输入格式与标准 PyTorch 不同
    """
    # FlashAttention 要求输入为 half 精度
    q, k, v = q.half(), k.half(), v.half()
    
    output = flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        causal=causal,
        softmax_scale=None  # 自动使用 1/sqrt(d)
    )
    
    return output.float()
```

**使用场景：**
- 长文档理解（>4K tokens）
- 高分辨率图像 Transformer
- 长视频处理
- 蛋白质序列建模

FlashAttention 是现代 Transformer 训练的标配优化技术！
