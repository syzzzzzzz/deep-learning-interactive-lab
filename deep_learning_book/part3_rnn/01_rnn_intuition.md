# 第八章：循环结构、梯度消失与门机制直观解释

## 来源标注版：RNN 到底记住了什么

**这一节真正要学的不是“RNN 有记忆”这句口号，而是：模型每读入一个时间步，都会把当前输入和上一刻隐藏状态合成新的隐藏状态。** 这份正文对照 D2L、PyTorch recurrent layers 文档、LSTM 原论文和 GRU/Encoder-Decoder 论文；页面里的状态柱和时间箭头是低维教学图，不等于真实模型内部只有几个可直接解释的记忆格。[S1][S2][S3][S4]

来源标符：

- [S1] Dive into Deep Learning, Recurrent Neural Networks: https://d2l.ai/chapter_recurrent-neural-networks/index.html
- [S2] PyTorch recurrent layers: https://pytorch.org/docs/stable/nn.html#recurrent-layers
- [S3] Hochreiter & Schmidhuber, Long Short-Term Memory: https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory
- [S4] Cho et al., Learning Phrase Representations using RNN Encoder-Decoder: https://arxiv.org/abs/1406.1078

### 1. 序列数据：不能把每个时刻当成孤立样本

句子、音频、日志和传感器曲线都有顺序。第 `t` 个输入不仅要看自己，还要看前面发生过什么；否则“我不喜欢这部电影”和“我喜欢这部电影”就可能被模型看得太像。[S1]

> 操作建议：先调大“序列长度”。观察页面里时间步变多后，最早输入的影响是否还能到达最后输出。

### 2. 隐藏状态：RNN 的记忆不是仓库，而是压缩摘要

隐藏状态 `h_t` 可以理解成模型读到第 `t` 步时的当前笔记。它不是把历史逐字保存，而是把历史压缩成一个固定长度向量。页面中的状态柱表示隐藏状态不同维度的强弱，柱子变化越大，说明当前输入正在明显改写内部摘要。[S1][S2]

```text
h_t = f(x_t, h_{t-1})
```

> 操作建议：把“隐藏维度”从小调到大，盯住状态柱和最终输出。思考：维度变多后，信息更清楚了，还是只是更难解释了？

### 3. 一步递推：当前输入和旧记忆一起决定新记忆

最基本的 RNN 单元会把当前输入 `x_t` 和上一时刻隐藏状态 `h_{t-1}` 分别线性变换，再加起来过非线性函数，得到新的 `h_t`。`W_xh` 负责读当前输入，`W_hh` 负责读旧记忆；页面里的循环箭头对应的就是这条旧记忆路径。[S1][S2]

```text
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
```

> 操作建议：调整“记忆保留率”。如果保留率低，早期状态会很快被新输入盖住；如果保留率高，旧信息会更久地影响输出。

### 4. 为什么长序列难：梯度也要沿时间反复穿过同一条路

RNN 训练时，误差信号要从后面的时间步反传回前面的时间步。序列越长，梯度要穿过的递推链越长。如果每一步都让梯度缩小一点，连乘很多次后就会接近 0；如果每一步都放大一点，梯度又可能爆炸。[S1][S3]

> 操作建议：把“序列长度”调到较大，再加入“输入噪声”。观察早期 token 的影响是否变得更模糊。

### 5. LSTM 和 GRU：给记忆加门，不是简单换个名字

LSTM 的历史动机是缓解长期依赖问题：它用门控机制控制哪些信息保留、哪些信息写入、哪些信息输出。GRU 把门控结构做得更紧凑，常用更新门和重置门控制状态变化。它不是“更低级的 LSTM”，而是在参数量和表达能力之间做另一种取舍。[S2][S3][S4]

> 操作建议：看完本页后跳到“隐藏状态”或“序列模型”，对比普通 RNN、LSTM、GRU 的状态更新方式。重点问：门到底在保护哪一段信息？

---

## 8.1 为什么需要 RNN？

CNN 处理固定大小的输入（一张图）。但现实中很多数据是**序列**：
- 句子：词与词之间有顺序依赖
- 时间序列：今天的股价依赖昨天
- 音频：每帧声音依赖前面的帧

RNN 的核心思想：**用隐藏状态记住过去**。

```
普通网络：  x → [网络] → y
RNN：       x_t, h_{t-1} → [网络] → y_t, h_t
                                          ↑
                                    传给下一步
```

---

## 8.2 RNN 手动推导

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────
# 手动实现 RNN，逐步展示计算过程
# ─────────────────────────────────────────────────────────

class ManualRNNStep:
    """
    单步 RNN 计算的完整展示

    公式：
        h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
        y_t = W_hy · h_t + b_y
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 4, output_size: int = 2):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重（Xavier）
        torch.manual_seed(42)
        scale = (2 / (input_size + hidden_size)) ** 0.5
        self.W_xh = torch.randn(hidden_size, input_size)  * scale
        self.W_hh = torch.randn(hidden_size, hidden_size) * scale
        self.b_h  = torch.zeros(hidden_size)
        self.W_hy = torch.randn(output_size, hidden_size) * scale
        self.b_y  = torch.zeros(output_size)

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor, verbose: bool = True):
        """
        执行一步 RNN 计算，打印每个中间值

        x_t:    [input_size]
        h_prev: [hidden_size]
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"输入 x_t:      {x_t.numpy().round(3)}")
            print(f"上一隐状态 h_{{t-1}}: {h_prev.numpy().round(3)}")

        # 计算各部分
        xh_part = self.W_xh @ x_t
        hh_part = self.W_hh @ h_prev
        pre_act = xh_part + hh_part + self.b_h
        h_t = torch.tanh(pre_act)
        y_t = self.W_hy @ h_t + self.b_y

        if verbose:
            print(f"\nW_xh · x_t:    {xh_part.numpy().round(3)}")
            print(f"W_hh · h_{{t-1}}: {hh_part.numpy().round(3)}")
            print(f"pre_activation: {pre_act.numpy().round(3)}")
            print(f"h_t = tanh(...): {h_t.numpy().round(3)}")
            print(f"y_t = W_hy·h_t: {y_t.numpy().round(3)}")

        return h_t, y_t

    def run_sequence(self, sequence: torch.Tensor, verbose: bool = False):
        """
        处理完整序列

        sequence: [T, input_size]
        返回: all_h [T, hidden_size], all_y [T, output_size]
        """
        T = sequence.shape[0]
        h = torch.zeros(self.hidden_size)
        all_h, all_y = [], []

        for t in range(T):
            h, y = self.step(sequence[t], h, verbose=(verbose and t < 3))
            all_h.append(h.clone())
            all_y.append(y.clone())

        return torch.stack(all_h), torch.stack(all_y)


# 演示
rnn_step = ManualRNNStep(input_size=3, hidden_size=4, output_size=2)
x_t = torch.tensor([0.5, -0.3, 0.8])
h_prev = torch.zeros(4)
h_t, y_t = rnn_step.step(x_t, h_prev, verbose=True)

# 处理一个序列
seq = torch.randn(10, 3)
all_h, all_y = rnn_step.run_sequence(seq)
print(f"\n序列长度=10，隐状态形状: {all_h.shape}，输出形状: {all_y.shape}")
```

---

## 8.3 梯度消失的数学直觉

```python
def visualize_vanishing_gradient():
    """
    可视化 RNN 中梯度消失的原因

    反向传播时，梯度需要穿越 T 步：
    ∂L/∂h_0 = ∏_{t=1}^{T} (∂h_t/∂h_{t-1})
             = ∏_{t=1}^{T} diag(1 - h_t²) · W_hh

    如果 |W_hh| < 1，乘积指数衰减 → 梯度消失
    如果 |W_hh| > 1，乘积指数爆炸 → 梯度爆炸
    """
    T = 50  # 序列长度
    hidden_size = 32

    torch.manual_seed(42)

    # 三种情况：消失、正常、爆炸
    scenarios = {
        '梯度消失 (‖W‖<1)': 0.9,
        '梯度正常 (‖W‖≈1)': 1.0,
        '梯度爆炸 (‖W‖>1)': 1.1,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (label, scale) in zip(axes, scenarios.items()):
        # 初始化权重，控制谱范数
        W = torch.randn(hidden_size, hidden_size)
        # 将谱范数缩放到目标值
        sigma = torch.linalg.norm(W, ord=2)
        W = W * (scale / sigma)

        # 模拟梯度传播：从最后一步往前
        grad_norms = []
        grad = torch.ones(hidden_size)  # 初始梯度

        for t in range(T):
            grad = W.T @ grad
            grad_norms.append(grad.norm().item())

        steps = list(range(1, T + 1))
        color = '#C44E52' if scale < 1 else ('#55A868' if scale == 1 else '#DD8452')
        ax.semilogy(steps, grad_norms, color=color, linewidth=2)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('反向传播步数（从输出往输入）')
        ax.set_ylabel('梯度范数（log scale）')
        ax.grid(True, alpha=0.3)
        ax.axhline(1e-4, color='gray', linestyle='--', alpha=0.5, label='消失阈值')
        ax.axhline(1e4,  color='gray', linestyle=':',  alpha=0.5, label='爆炸阈值')
        ax.legend(fontsize=8)

        # 标注最终梯度值
        final = grad_norms[-1]
        ax.text(0.95, 0.05, f'最终梯度: {final:.2e}',
                transform=ax.transAxes, ha='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('RNN 梯度消失/爆炸的根本原因：矩阵连乘',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vanishing_gradient_math.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_vanishing_gradient()
```

---

## 8.4 LSTM 门机制直观解释

```python
def explain_lstm_gates():
    """
    用图示解释 LSTM 的四个门

    LSTM 公式：
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   ← 遗忘门：决定忘掉多少
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← 输入门：决定记住多少
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) ← 候选记忆
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   ← 输出门：决定输出多少

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       ← 细胞状态更新
    h_t = o_t ⊙ tanh(c_t)                  ← 隐藏状态
    """
    torch.manual_seed(42)

    input_size  = 4
    hidden_size = 6

    # 模拟一个 LSTM 步骤，记录所有门的值
    lstm_cell = nn.LSTMCell(input_size, hidden_size)

    x_seq = torch.randn(20, input_size)
    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)

    gate_history = {'forget': [], 'input': [], 'output': [], 'cell': [], 'hidden': []}

    # 手动提取门值（通过权重矩阵计算）
    for t in range(20):
        x_t = x_seq[t:t+1]
        combined = torch.cat([h, x_t], dim=1)

        # LSTM cell 的权重：weight_ih [4*hidden, input], weight_hh [4*hidden, hidden]
        gates_raw = (combined @ torch.cat([lstm_cell.weight_hh,
                                            lstm_cell.weight_ih], dim=1).T
                     + lstm_cell.bias_hh + lstm_cell.bias_ih)

        f = torch.sigmoid(gates_raw[:, :hidden_size])
        i = torch.sigmoid(gates_raw[:, hidden_size:2*hidden_size])
        g = torch.tanh(gates_raw[:, 2*hidden_size:3*hidden_size])
        o = torch.sigmoid(gates_raw[:, 3*hidden_size:])

        c = f * c + i * g
        h = o * torch.tanh(c)

        gate_history['forget'].append(f[0].detach().numpy())
        gate_history['input'].append(i[0].detach().numpy())
        gate_history['output'].append(o[0].detach().numpy())
        gate_history['cell'].append(c[0].detach().numpy())
        gate_history['hidden'].append(h[0].detach().numpy())

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    gate_names = ['forget', 'input', 'output', 'cell', 'hidden']
    gate_labels = ['遗忘门 f_t（σ）', '输入门 i_t（σ）', '输出门 o_t（σ）',
                   '细胞状态 c_t', '隐藏状态 h_t']
    cmaps = ['Reds', 'Blues', 'Greens', 'RdBu', 'PuOr']

    for idx, (name, label, cmap) in enumerate(zip(gate_names, gate_labels, cmaps)):
        ax = axes[idx // 3, idx % 3]
        data = np.array(gate_history[name]).T  # [hidden_size, T]
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                       vmin=0 if name in ['forget','input','output'] else -1,
                       vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('时间步 t')
        ax.set_ylabel('隐藏单元索引')

    # 最后一个子图：门值随时间的均值变化
    ax = axes[1, 2]
    for name, label, color in zip(
        ['forget', 'input', 'output'],
        ['遗忘门', '输入门', '输出门'],
        ['red', 'blue', 'green']
    ):
        means = [np.mean(v) for v in gate_history[name]]
        ax.plot(means, color=color, linewidth=2, label=label, alpha=0.8)
    ax.set_title('各门平均激活值随时间变化', fontsize=11, fontweight='bold')
    ax.set_xlabel('时间步 t')
    ax.set_ylabel('平均门值')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('LSTM 门机制可视化（序列长度=20）',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lstm_gates.png', dpi=150, bbox_inches='tight')
    plt.show()

explain_lstm_gates()
```

---

## 8.5 GRU vs LSTM：门的简化

```python
def compare_gru_lstm_gates():
    """
    对比 GRU 和 LSTM 的门机制

    GRU 只有两个门（比 LSTM 少一个）：
    z_t = σ(W_z · [h_{t-1}, x_t])   ← 更新门（合并了遗忘+输入）
    r_t = σ(W_r · [h_{t-1}, x_t])   ← 重置门
    n_t = tanh(W_n · [r_t⊙h_{t-1}, x_t])  ← 候选隐状态
    h_t = (1-z_t)⊙h_{t-1} + z_t⊙n_t
    """
    print("LSTM vs GRU 参数量对比：")
    print("=" * 40)

    for hidden_size in [32, 64, 128, 256]:
        input_size = 32
        lstm_params = 4 * (hidden_size * input_size + hidden_size * hidden_size + hidden_size)
        gru_params  = 3 * (hidden_size * input_size + hidden_size * hidden_size + hidden_size)
        print(f"hidden={hidden_size:4d}: LSTM={lstm_params:8,}  GRU={gru_params:8,}  "
              f"GRU/LSTM={gru_params/lstm_params:.1%}")

    # 可视化门结构对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LSTM 门结构
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('LSTM 门结构', fontsize=13, fontweight='bold')

    # 绘制 LSTM 示意图
    gates_lstm = [
        (2, 6, '遗忘门\nf_t = σ(·)', '#FF6B6B'),
        (5, 6, '输入门\ni_t = σ(·)', '#4ECDC4'),
        (8, 6, '输出门\no_t = σ(·)', '#45B7D1'),
        (5, 3, '候选记忆\ng_t = tanh(·)', '#96CEB4'),
    ]
    for x, y, label, color in gates_lstm:
        rect = FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2,
                               boxstyle='round,pad=0.1',
                               facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    ax.text(5, 1, '细胞状态 c_t = f_t⊙c_{t-1} + i_t⊙g_t\n隐状态 h_t = o_t⊙tanh(c_t)',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # GRU 门结构
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('GRU 门结构（更简洁）', fontsize=13, fontweight='bold')

    gates_gru = [
        (3, 6, '更新门\nz_t = σ(·)', '#FF6B6B'),
        (7, 6, '重置门\nr_t = σ(·)', '#4ECDC4'),
        (5, 3, '候选状态\nn_t = tanh(·)', '#96CEB4'),
    ]
    for x, y, label, color in gates_gru:
        rect = FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2,
                               boxstyle='round,pad=0.1',
                               facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    ax.text(5, 1, 'h_t = (1-z_t)⊙h_{t-1} + z_t⊙n_t\n（无独立细胞状态）',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('LSTM（4门）vs GRU（2门）结构对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lstm_vs_gru.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_gru_lstm_gates()
```

---

## 小结

| 模型 | 门数量 | 核心公式 | 适用场景 |
|------|--------|----------|----------|
| RNN | 0 | h_t = tanh(W·[h,x]) | 短序列，简单任务 |
| LSTM | 4（f/i/g/o） | c_t = f⊙c + i⊙g | 长序列，需要长期记忆 |
| GRU | 2（z/r） | h_t = (1-z)⊙h + z⊙n | 中等序列，参数更少 |

## 内容可信度与来源

**可信度：已校对。** 本节关于隐藏状态递推、序列记忆、梯度衰减/爆炸、LSTM 与 GRU 门控的说明，已对照 D2L、PyTorch 循环层文档和经典 RNN 论文来源检查。页面里的状态箭头和门控图是低维教学图，用来解释“信息怎样随时间传递”，不代表真实模型内部向量可以被这样逐项解释。

参考来源：

- Dive into Deep Learning, recurrent neural networks: https://d2l.ai/chapter_recurrent-neural-networks/index.html
- PyTorch recurrent layers: https://pytorch.org/docs/stable/nn.html#recurrent-layers
- Hochreiter & Schmidhuber, Long Short-Term Memory: https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory
- Cho et al., Learning Phrase Representations using RNN Encoder-Decoder: https://arxiv.org/abs/1406.1078

边界说明：

- 隐藏状态图是低维可视化，真实 RNN/LSTM/GRU 的状态通常是高维向量。
- LSTM/GRU 门控公式在不同库实现里会有门顺序、偏置初始化和融合 kernel 的差异，工程细节以框架文档为准。
- 长程依赖问题不能只看模型类型，还要结合序列长度、初始化、优化器、梯度裁剪和数据分布判断。
