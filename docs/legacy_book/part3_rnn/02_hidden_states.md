# 第九章：隐藏状态变化与序列预测可视化

---

## 9.1 隐藏状态是什么？

隐藏状态 h_t 是 RNN 的"记忆"——它压缩了从 t=0 到 t 的所有历史信息。
可视化隐藏状态能告诉我们：
- 模型在序列的哪个位置"注意到"了什么
- 不同输入如何影响记忆
- 模型是否真的学到了有意义的表示

---

## 9.1.1 隐藏状态轨迹可视化

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from typing import List, Optional

# ─────────────────────────────────────────────────────────
# 隐藏状态追踪器
# ─────────────────────────────────────────────────────────

class HiddenStateTracker:
    """
    追踪并可视化 RNN/LSTM/GRU 的隐藏状态变化

    使用方法：
        tracker = HiddenStateTracker(model, model_type='lstm')
        tracker.run_sequence(input_seq)
        tracker.plot_heatmap()
        tracker.plot_pca_trajectory()
    """

    def __init__(self, model: nn.Module, model_type: str = 'rnn'):
        """
        model_type: 'rnn', 'lstm', 'gru'
        """
        self.model = model
        self.model_type = model_type.lower()
        self.hidden_states: List[np.ndarray] = []
        self.cell_states: List[np.ndarray] = []   # 仅 LSTM
        self.inputs: List[np.ndarray] = []

    def run_sequence(self, x: torch.Tensor, reset: bool = True):
        """
        逐步运行序列，记录每步隐藏状态

        x: [T, input_size] 或 [T, batch, input_size]
        """
        if reset:
            self.hidden_states.clear()
            self.cell_states.clear()
            self.inputs.clear()

        self.model.eval()
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [T, 1, input_size]

        T = x.shape[0]
        h = torch.zeros(1, 1, self.model.hidden_size)
        c = torch.zeros(1, 1, self.model.hidden_size)

        with torch.no_grad():
            for t in range(T):
                x_t = x[t:t+1]  # [1, 1, input_size]

                if self.model_type == 'lstm':
                    out, (h, c) = self.model(x_t, (h, c))
                    self.cell_states.append(c[0, 0].numpy().copy())
                elif self.model_type == 'gru':
                    out, h = self.model(x_t, h)
                else:  # rnn
                    out, h = self.model(x_t, h)

                self.hidden_states.append(h[0, 0].numpy().copy())
                self.inputs.append(x_t[0, 0].numpy().copy())

        return np.array(self.hidden_states)

    def plot_heatmap(self, figsize=(14, 6)):
        """热力图：横轴=时间步，纵轴=隐藏单元"""
        if not self.hidden_states:
            print("请先运行 run_sequence()")
            return

        H = np.array(self.hidden_states).T  # [hidden_size, T]
        T = H.shape[1]

        n_plots = 2 if self.cell_states else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        im0 = axes[0].imshow(H, aspect='auto', cmap='RdBu',
                              vmin=-1, vmax=1, interpolation='nearest')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title('隐藏状态 h_t（每行=一个隐藏单元，每列=一个时间步）',
                          fontsize=11, fontweight='bold')
        axes[0].set_xlabel('时间步 t')
        axes[0].set_ylabel('隐藏单元索引')

        if self.cell_states:
            C = np.array(self.cell_states).T
            im1 = axes[1].imshow(C, aspect='auto', cmap='RdBu',
                                  vmin=-2, vmax=2, interpolation='nearest')
            plt.colorbar(im1, ax=axes[1])
            axes[1].set_title('细胞状态 c_t（LSTM 专有）',
                              fontsize=11, fontweight='bold')
            axes[1].set_xlabel('时间步 t')
            axes[1].set_ylabel('细胞单元索引')

        plt.suptitle(f'{self.model_type.upper()} 隐藏状态热力图',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'hidden_state_heatmap_{self.model_type}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_pca_trajectory(self, figsize=(10, 8)):
        """
        用 PCA 将高维隐藏状态降到 2D，可视化状态轨迹

        轨迹的形状反映了模型如何在状态空间中"移动"
        """
        if len(self.hidden_states) < 3:
            print("序列太短，无法做 PCA")
            return

        H = np.array(self.hidden_states)  # [T, hidden_size]
        T = H.shape[0]

        pca = PCA(n_components=2)
        H_2d = pca.fit_transform(H)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 轨迹图（带时间颜色编码）
        ax = axes[0]
        scatter = ax.scatter(H_2d[:, 0], H_2d[:, 1],
                             c=range(T), cmap='viridis', s=60, zorder=5)
        # 连接轨迹
        ax.plot(H_2d[:, 0], H_2d[:, 1], 'gray', alpha=0.4, linewidth=1, zorder=3)
        # 标注起点和终点
        ax.scatter(*H_2d[0], s=200, c='green', marker='*', zorder=6, label='起点 t=0')
        ax.scatter(*H_2d[-1], s=200, c='red', marker='X', zorder=6, label=f'终点 t={T-1}')
        plt.colorbar(scatter, ax=ax, label='时间步')
        ax.set_title('隐藏状态 PCA 轨迹\n（颜色=时间，绿星=起点，红X=终点）',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 各隐藏单元随时间的变化（前8个）
        ax2 = axes[1]
        n_show = min(8, H.shape[1])
        for i in range(n_show):
            ax2.plot(H[:, i], alpha=0.7, linewidth=1.2, label=f'h[{i}]')
        ax2.set_title(f'前{n_show}个隐藏单元随时间的变化',
                      fontsize=10, fontweight='bold')
        ax2.set_xlabel('时间步 t')
        ax2.set_ylabel('激活值')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

        plt.suptitle(f'{self.model_type.upper()} 隐藏状态空间分析',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'hidden_state_pca_{self.model_type}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_input_vs_hidden(self, input_dim: int = 0, hidden_dim: int = 0):
        """可视化输入与隐藏状态的对应关系"""
        if not self.hidden_states:
            return

        inputs = np.array(self.inputs)
        hiddens = np.array(self.hidden_states)
        T = len(inputs)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        t = range(T)
        axes[0].plot(t, inputs[:, input_dim], 'b-', linewidth=1.5, label=f'输入 x[{input_dim}]')
        axes[0].set_ylabel('输入值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('输入序列', fontsize=10)

        axes[1].plot(t, hiddens[:, hidden_dim], 'r-', linewidth=1.5,
                     label=f'隐藏状态 h[{hidden_dim}]')
        axes[1].set_ylabel('隐藏状态值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('对应隐藏单元', fontsize=10)

        # 相关性
        corr = np.corrcoef(inputs[:, input_dim], hiddens[:, hidden_dim])[0, 1]
        axes[2].scatter(inputs[:, input_dim], hiddens[:, hidden_dim],
                        c=range(T), cmap='viridis', s=30, alpha=0.7)
        axes[2].set_xlabel(f'输入 x[{input_dim}]')
        axes[2].set_ylabel(f'隐藏状态 h[{hidden_dim}]')
        axes[2].set_title(f'输入 vs 隐藏状态（相关系数={corr:.3f}）', fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('输入序列与隐藏状态的对应关系', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('input_vs_hidden.png', dpi=150, bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────
# 序列预测可视化
# ─────────────────────────────────────────────────────────

class SequencePredictionVisualizer:
    """
    可视化序列预测任务的训练过程和预测结果

    支持：正弦波预测、随机游走预测
    """

    def __init__(self, model_type: str = 'lstm', hidden_size: int = 32,
                 seq_len: int = 50, pred_len: int = 20):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 构建模型
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[model_type]
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden_size,
                           num_layers=1, batch_first=False)
        self.fc = nn.Linear(hidden_size, 1)
        self.params = list(self.rnn.parameters()) + list(self.fc.parameters())

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, 1, 1]"""
        out, _ = self.rnn(x)
        return self.fc(out[-1])  # 取最后一步的输出

    def train_and_visualize(self, n_epochs: int = 200, lr: float = 0.01):
        """训练正弦波预测，并可视化训练过程"""
        torch.manual_seed(42)
        np.random.seed(42)

        # 生成正弦波数据
        t = np.linspace(0, 4 * np.pi, 500)
        signal = np.sin(t) + 0.1 * np.random.randn(len(t))

        def make_batch(signal, seq_len, n_samples=32):
            X, y = [], []
            for _ in range(n_samples):
                start = np.random.randint(0, len(signal) - seq_len - 1)
                X.append(signal[start:start+seq_len])
                y.append(signal[start+seq_len])
            return (torch.tensor(X, dtype=torch.float32).unsqueeze(-1).permute(1, 0, 2),
                    torch.tensor(y, dtype=torch.float32).unsqueeze(-1))

        optimizer = torch.optim.Adam(self.params, lr=lr)
        criterion = nn.MSELoss()
        losses = []

        for epoch in range(n_epochs):
            X_batch, y_batch = make_batch(signal, self.seq_len)
            pred = self.predict_step(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, 1.0)
            optimizer.step()
            losses.append(loss.item())

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 训练损失
        axes[0, 0].semilogy(losses, 'b-', linewidth=1.5)
        axes[0, 0].set_title('训练损失（MSE）', fontsize=11)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss（log scale）')
        axes[0, 0].grid(True, alpha=0.3)

        # 预测 vs 真实
        test_start = 200
        test_input = signal[test_start:test_start + self.seq_len]
        test_true  = signal[test_start + self.seq_len:
                            test_start + self.seq_len + self.pred_len]

        # 自回归预测
        preds = []
        current_seq = list(test_input)
        with torch.no_grad():
            for _ in range(self.pred_len):
                x = torch.tensor(current_seq[-self.seq_len:],
                                  dtype=torch.float32).unsqueeze(-1).unsqueeze(1)
                p = self.predict_step(x).item()
                preds.append(p)
                current_seq.append(p)

        axes[0, 1].plot(range(self.seq_len), test_input, 'b-', linewidth=1.5, label='输入序列')
        axes[0, 1].plot(range(self.seq_len, self.seq_len + self.pred_len),
                        test_true, 'g-', linewidth=2, label='真实值')
        axes[0, 1].plot(range(self.seq_len, self.seq_len + self.pred_len),
                        preds, 'r--', linewidth=2, label='预测值')
        axes[0, 1].axvline(self.seq_len, color='gray', linestyle='--', alpha=0.7)
        axes[0, 1].set_title(f'{self.model_type.upper()} 序列预测（自回归）', fontsize=11)
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 隐藏状态热力图
        with torch.no_grad():
            x_vis = torch.tensor(test_input, dtype=torch.float32).unsqueeze(-1).unsqueeze(1)
            # 逐步运行
            h_states = []
            h = torch.zeros(1, 1, self.hidden_size)
            c = torch.zeros(1, 1, self.hidden_size)
            for t in range(len(test_input)):
                xt = x_vis[t:t+1]
                if self.model_type == 'lstm':
                    _, (h, c) = self.rnn(xt, (h, c))
                else:
                    _, h = self.rnn(xt, h)
                h_states.append(h[0, 0].numpy())

        H = np.array(h_states).T
        im = axes[1, 0].imshow(H[:16], aspect='auto', cmap='RdBu',
                                vmin=-1, vmax=1)
        plt.colorbar(im, ax=axes[1, 0])
        axes[1, 0].set_title('隐藏状态热力图（前16个单元）', fontsize=11)
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('隐藏单元')

        # 预测误差随时间
        errors = [abs(p - r) for p, r in zip(preds, test_true)]
        axes[1, 1].bar(range(self.pred_len), errors, color='salmon', alpha=0.8)
        axes[1, 1].set_title('逐步预测误差（越远越大）', fontsize=11)
        axes[1, 1].set_xlabel('预测步数')
        axes[1, 1].set_ylabel('|预测 - 真实|')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{self.model_type.upper()} 正弦波预测实验',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'seq_pred_{self.model_type}.png', dpi=150, bbox_inches='tight')
        plt.show()

        return losses, preds


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_hidden_states():
    torch.manual_seed(42)

    # 1. 隐藏状态追踪
    print("1. LSTM 隐藏状态追踪")
    lstm = nn.LSTM(input_size=4, hidden_size=16, batch_first=False)
    lstm.hidden_size = 16

    tracker = HiddenStateTracker(lstm, model_type='lstm')
    seq = torch.randn(30, 4)
    tracker.run_sequence(seq)
    tracker.plot_heatmap()
    tracker.plot_pca_trajectory()
    tracker.plot_input_vs_hidden(input_dim=0, hidden_dim=0)

    # 2. 序列预测
    print("\n2. 序列预测可视化")
    for model_type in ['rnn', 'lstm', 'gru']:
        print(f"\n训练 {model_type.upper()}...")
        viz = SequencePredictionVisualizer(model_type=model_type, hidden_size=32)
        losses, preds = viz.train_and_visualize(n_epochs=150)
        print(f"  最终损失: {losses[-1]:.6f}")

    return tracker

tracker = demo_hidden_states()
```

---

## 小结

| 工具 | 功能 | 关键输出 |
|------|------|----------|
| HiddenStateTracker | 逐步记录隐藏状态 | 热力图 + PCA 轨迹 |
| plot_heatmap | 时间×单元的激活热力图 | 看哪些单元在何时激活 |
| plot_pca_trajectory | 状态空间轨迹 | 理解模型"记忆"的演化 |
| SequencePredictionVisualizer | 训练+预测+误差分析 | 自回归预测可视化 |
