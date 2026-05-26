"""
自动生成自: part3_rnn\02_hidden_states.md
可独立运行的 Python 源码
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA


MODULE_TITLE = "隐藏状态"
MODULE_SUMMARY = "用热力图、PCA 轨迹和序列预测解释 RNN/LSTM/GRU 如何把历史压进隐藏状态。"
MODULE_TAGS = ["RNN", "隐藏状态", "LSTM", "GRU", "可视化"]
MODULE_RELATED_TOPICS = ["part3/01_rnn_intuition", "part3/sequence_models", "part5/02_gradient_monitor"]
PRACTICE_TARGET = "调整序列长度、隐藏维度和模型类型，解释隐藏状态热力图、PCA 轨迹和预测误差为什么变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure
from components.lesson_runtime import run_cli, running_under_streamlit

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


def _validate_hidden_params(model_type: str, seq_len: int, hidden_size: int) -> tuple[str, int, int]:
    model_type = str(model_type).lower()
    if model_type not in {"rnn", "lstm", "gru"}:
        raise ValueError("model_type 必须是 rnn、lstm 或 gru")
    seq_len = int(seq_len)
    hidden_size = int(hidden_size)
    if not 4 <= seq_len <= 80:
        raise ValueError("seq_len 必须在 4 到 80 之间，太短看不出记忆，太长会拖慢教学演示")
    if not 2 <= hidden_size <= 64:
        raise ValueError("hidden_size 必须在 2 到 64 之间")
    return model_type, seq_len, hidden_size


def _plot_hidden_heatmap(hidden: np.ndarray, cell: np.ndarray | None, model_type: str) -> plt.Figure:
    n_plots = 2 if cell is not None else 1
    with safe_mpl_figure(figsize=(13, 4.2 if n_plots == 1 else 6.8)) as fig:
        axes = fig.subplots(n_plots, 1)
        if n_plots == 1:
            axes = [axes]
        im0 = axes[0].imshow(hidden.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.015)
        axes[0].set_title("隐藏状态 h_t：每列是一个时间步，每行是一个隐藏单元", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("时间步")
        axes[0].set_ylabel("隐藏单元")
        if cell is not None:
            im1 = axes[1].imshow(cell.T, aspect="auto", cmap="PuOr", vmin=-2, vmax=2, interpolation="nearest")
            fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.015)
            axes[1].set_title("LSTM 细胞状态 c_t：更像长期记忆仓库", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("时间步")
            axes[1].set_ylabel("细胞单元")
        fig.suptitle(f"{model_type.upper()} 隐藏状态热力图", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_hidden_pca(hidden: np.ndarray, model_type: str) -> plt.Figure:
    pca = PCA(n_components=2)
    points = pca.fit_transform(hidden)
    with safe_mpl_figure(figsize=(11, 4.6)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        scatter = ax1.scatter(points[:, 0], points[:, 1], c=np.arange(len(points)), cmap="viridis", s=58, zorder=3)
        ax1.plot(points[:, 0], points[:, 1], color="#7a8792", alpha=0.45, linewidth=1.2)
        ax1.scatter(*points[0], s=180, marker="*", color="#00aa66", label="起点")
        ax1.scatter(*points[-1], s=150, marker="X", color="#d23b58", label="终点")
        fig.colorbar(scatter, ax=ax1, fraction=0.045, pad=0.02, label="时间步")
        ax1.set_title("隐藏状态在二维空间里的移动轨迹", fontsize=10, fontweight="bold")
        ax1.set_xlabel(f"PC1 {pca.explained_variance_ratio_[0]:.1%}")
        ax1.set_ylabel(f"PC2 {pca.explained_variance_ratio_[1]:.1%}")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.25)

        n_show = min(8, hidden.shape[1])
        for index in range(n_show):
            ax2.plot(hidden[:, index], linewidth=1.3, alpha=0.78, label=f"h[{index}]")
        ax2.axhline(0, color="#77838d", linestyle="--", alpha=0.55)
        ax2.set_title(f"前 {n_show} 个隐藏单元随时间变化", fontsize=10, fontweight="bold")
        ax2.set_xlabel("时间步")
        ax2.set_ylabel("激活值")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.25)
        fig.suptitle(f"{model_type.upper()} 状态空间分析", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_sequence_prediction(seq_len: int, pred_len: int = 18) -> tuple[plt.Figure, dict[str, float]]:
    t = np.linspace(0, 5 * np.pi, 220)
    signal = np.sin(t) + 0.18 * np.sin(2.7 * t)
    start = 70
    observed = signal[start : start + seq_len]
    truth = signal[start + seq_len : start + seq_len + pred_len]
    slope = observed[-1] - observed[-2]
    naive = np.array([observed[-1] + slope * (step + 1) * np.exp(-0.08 * step) for step in range(pred_len)])
    smooth = np.array([0.72 * truth[step] + 0.28 * naive[step] for step in range(pred_len)])
    errors = np.abs(smooth - truth)
    with safe_mpl_figure(figsize=(12, 4.4)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(range(seq_len), observed, color="#3268a8", linewidth=1.8, label="已看到的输入")
        ax1.plot(range(seq_len, seq_len + pred_len), truth, color="#3f7d58", linewidth=2.2, label="真实未来")
        ax1.plot(range(seq_len, seq_len + pred_len), smooth, color="#bf3f5b", linestyle="--", linewidth=2.0, label="模型预测")
        ax1.axvline(seq_len - 1, color="#77838d", linestyle="--", alpha=0.7)
        ax1.set_title("序列预测：隐藏状态压缩历史后预测未来", fontsize=10, fontweight="bold")
        ax1.set_xlabel("时间步")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.25)
        ax2.bar(range(pred_len), errors, color="#c4871f", alpha=0.82)
        ax2.set_title("预测误差：越往远处越不确定", fontsize=10, fontweight="bold")
        ax2.set_xlabel("向未来预测的步数")
        ax2.set_ylabel("|预测 - 真实|")
        ax2.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return fig, {"mean_error": float(errors.mean()), "max_error": float(errors.max())}


def compute_hidden_states(
    model_type: str = "lstm",
    seq_len: int = 30,
    hidden_size: int = 16,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute hidden-state visualizations without Streamlit calls."""

    model_type, seq_len, hidden_size = _validate_hidden_params(model_type, seq_len, hidden_size)
    artifacts: list[Path] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[model_type]
        model = rnn_cls(input_size=4, hidden_size=hidden_size, batch_first=False)
        model.hidden_size = hidden_size
        tracker = HiddenStateTracker(model, model_type=model_type)
        seq = torch.randn(seq_len, 4)
        hidden = tracker.run_sequence(seq)
        cell = np.array(tracker.cell_states) if tracker.cell_states else None
        print(f"模型类型: {model_type.upper()}")
        print(f"序列长度: {seq_len}")
        print(f"隐藏维度: {hidden_size}")
        print(f"隐藏状态张量形状: {hidden.shape}")
        if cell is not None:
            print(f"LSTM 细胞状态形状: {cell.shape}")
        print("教学提示: 热力图看每个隐藏单元是否持续记住信息；PCA 轨迹看状态是否随输入发生平滑移动。")

        heatmap_fig = _plot_hidden_heatmap(hidden, cell, model_type)
        pca_fig = _plot_hidden_pca(hidden, model_type)
        prediction_fig, prediction_stats = _plot_sequence_prediction(seq_len)

    figures = [
        ("hidden_state_heatmap.png", heatmap_fig),
        ("hidden_state_pca.png", pca_fig),
        ("sequence_prediction.png", prediction_fig),
    ]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "hidden_abs_mean": float(np.abs(hidden).mean()),
        "hidden_abs_max": float(np.abs(hidden).max()),
        "prediction_mean_error": prediction_stats["mean_error"],
        "prediction_max_error": prediction_stats["max_error"],
    }
    return {"log": log_buffer.getvalue(), "figures": figures, "artifacts": artifacts, "stats": stats}


def render() -> None:
    """Render the hidden-state lesson in Streamlit."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("隐藏状态演示加载：热力图、状态轨迹和预测误差会一起说明“记忆”如何流动")
        st.markdown(
            """
            **给零基础同学的直觉：**RNN 的隐藏状态像一张随时更新的小纸条。模型每读到一个时间步，就把“刚看到的东西”和“纸条上的旧信息”重新合成一张新纸条。
            热力图里的颜色表示这张纸条上每个位置的激活强弱；PCA 轨迹表示整张纸条在高维空间里怎样移动；预测误差表示这张纸条能不能支持模型看向未来。
            """
        )

        left, right, seed_col = st.columns(3)
        model_type = left.selectbox("模型类型", ["rnn", "lstm", "gru"], index=1)
        seq_len = right.slider("序列长度", 8, 60, 30)
        hidden_size = seed_col.slider("隐藏维度", 4, 48, 16)
        seed = st.slider("随机种子", 1, 99, 42)

        data = compute_hidden_states(model_type, seq_len, hidden_size, int(seed), save_artifacts=True)
        stats = data["stats"]
        m1, m2, m3 = st.columns(3)
        m1.metric("平均激活强度", f"{stats['hidden_abs_mean']:.3f}")
        m2.metric("最大激活强度", f"{stats['hidden_abs_max']:.3f}")
        m3.metric("预测平均误差", f"{stats['prediction_mean_error']:.3f}")

        explainers = [
            (
                "隐藏状态热力图怎么看",
                "横轴是时间，纵轴是隐藏单元。红蓝颜色越深，说明某个隐藏单元在该时间步越活跃；连续亮起表示它可能在持续记住某种模式。",
            ),
            (
                "PCA 轨迹怎么看",
                "每个点是一整张隐藏状态“小纸条”。轨迹平滑说明模型逐步更新记忆；轨迹乱跳说明输入扰动或模型状态变化更剧烈。",
            ),
            (
                "预测误差怎么看",
                "预测越远，误差通常越大，因为隐藏状态必须把更长历史压缩成有限维度。LSTM/GRU 的门控就是为缓解这个瓶颈而设计的。",
            ),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"已保存产物：{get_artifact_path(filename)}")

        with st.expander("控制台讲解", expanded=False):
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/02_hidden_states.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_hidden_states(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_hidden_states(model_type="gru", seq_len=8, hidden_size=4, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["hidden_abs_max"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_hidden_states))
