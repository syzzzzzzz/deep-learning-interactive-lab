"""RNN 超参数实验：把模型类型、隐藏维度、层数、Dropout 和学习率放进同一张可解释面板。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "RNN 超参数实验"
MODULE_SUMMARY = "用轻量可控实验解释 RNN/LSTM/GRU 的隐藏维度、层数、Dropout、双向结构和学习率如何影响损失曲线。"
MODULE_TAGS = ["RNN", "超参数", "LSTM", "GRU", "训练曲线", "鲁棒性"]
MODULE_RELATED_TOPICS = ["part3/01_rnn_intuition", "part3/02_hidden_states", "part3/03_sequence_toys", "part5/03_training_dynamics"]
PRACTICE_TARGET = "调整模型类型、隐藏维度、层数、Dropout 和学习率，解释训练损失、验证损失、参数量和过拟合风险如何变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure

try:
    """
    自动生成自: part3_rnn\04_hyperparam_rnn.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import product
    from typing import Dict, List, Tuple

    # ─────────────────────────────────────────────────────────
    # 统一的 RNN 实验模型
    # ─────────────────────────────────────────────────────────

    class RNNExperiment(nn.Module):
        """
        可配置的 RNN 实验模型

        支持 RNN / LSTM / GRU，可调所有关键超参数
        """

        def __init__(self,
                     model_type: str = 'lstm',
                     input_size: int = 1,
                     hidden_size: int = 64,
                     num_layers: int = 1,
                     dropout: float = 0.0,
                     bidirectional: bool = False,
                     output_size: int = 1):
            super().__init__()
            self.model_type = model_type
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.directions = 2 if bidirectional else 1

            rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[model_type]
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size * self.directions, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: [batch, seq_len, input_size]"""
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])  # 取最后时间步

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def run_experiment(config: dict, n_epochs: int = 100,
                       seq_len: int = 50, n_samples: int = 500,
                       lr: float = 0.001) -> dict:
        """
        运行单个超参数配置的实验

        返回：{'train_loss': [...], 'val_loss': [...], 'params': int}
        """
        torch.manual_seed(42)
        np.random.seed(42)

        # 生成正弦波数据
        t = np.linspace(0, 8 * np.pi, n_samples + seq_len + 1)
        signal = np.sin(t) + 0.05 * np.random.randn(len(t))

        X = np.array([signal[i:i+seq_len] for i in range(n_samples)])
        y = np.array([signal[i+seq_len] for i in range(n_samples)])

        split = int(0.8 * n_samples)
        X_train = torch.tensor(X[:split], dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y[:split], dtype=torch.float32).unsqueeze(-1)
        X_val   = torch.tensor(X[split:], dtype=torch.float32).unsqueeze(-1)
        y_val   = torch.tensor(y[split:], dtype=torch.float32).unsqueeze(-1)

        model = RNNExperiment(**config)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses, val_losses = [], []
        batch_size = 64

        for epoch in range(n_epochs):
            model.train()
            idx = torch.randperm(len(X_train))
            epoch_loss = []
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb, yb = X_train[batch_idx], y_train[batch_idx]
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss.append(loss.item())
            train_losses.append(np.mean(epoch_loss))

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            val_losses.append(val_loss)

        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'final_val_loss': val_losses[-1],
            'params': model.count_params(),
            'config': config,
        }


    # ─────────────────────────────────────────────────────────
    # 超参数对比实验：hidden_size
    # ─────────────────────────────────────────────────────────

    def experiment_hidden_size():
        """对比不同 hidden_size 的效果"""
        hidden_sizes = [8, 16, 32, 64, 128]
        results = {}

        for hs in hidden_sizes:
            print(f"  hidden_size={hs}...")
            config = dict(model_type='lstm', input_size=1, hidden_size=hs,
                          num_layers=1, dropout=0.0, bidirectional=False, output_size=1)
            results[hs] = run_experiment(config, n_epochs=80)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 验证损失曲线
        for hs, res in results.items():
            axes[0].semilogy(res['val_loss'], linewidth=1.5, label=f'hidden={hs}')
        axes[0].set_title('验证损失 vs Epoch', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss（log）')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 最终验证损失 vs hidden_size
        hs_list = list(results.keys())
        final_losses = [results[hs]['final_val_loss'] for hs in hs_list]
        axes[1].plot(hs_list, final_losses, 'b-o', linewidth=2, markersize=8)
        axes[1].set_title('最终验证损失 vs hidden_size', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('hidden_size')
        axes[1].set_ylabel('最终 MSE Loss')
        axes[1].grid(True, alpha=0.3)

        # 参数量 vs hidden_size
        params = [results[hs]['params'] for hs in hs_list]
        axes[2].plot(hs_list, params, 'r-o', linewidth=2, markersize=8)
        axes[2].set_title('参数量 vs hidden_size', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('hidden_size')
        axes[2].set_ylabel('参数量')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('LSTM hidden_size 超参数实验', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('exp_hidden_size.png', dpi=150, bbox_inches='tight')
        plt.show()

        return results


    # ─────────────────────────────────────────────────────────
    # 超参数对比实验：模型类型 + 层数
    # ─────────────────────────────────────────────────────────

    def experiment_model_type_and_layers():
        """对比 RNN/LSTM/GRU 和不同层数"""
        configs = []
        for model_type in ['rnn', 'lstm', 'gru']:
            for num_layers in [1, 2, 3]:
                configs.append({
                    'label': f'{model_type.upper()}-{num_layers}层',
                    'config': dict(model_type=model_type, input_size=1, hidden_size=32,
                                   num_layers=num_layers,
                                   dropout=0.1 if num_layers > 1 else 0.0,
                                   bidirectional=False, output_size=1),
                })

        results = {}
        for item in configs:
            label = item['label']
            print(f"  {label}...")
            results[label] = run_experiment(item['config'], n_epochs=80)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        for (label, res), color in zip(results.items(), colors):
            axes[0].semilogy(res['val_loss'], linewidth=1.5, label=label,
                             color=color, alpha=0.85)

        axes[0].set_title('验证损失对比（模型类型 × 层数）', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss（log）')
        axes[0].legend(fontsize=7, ncol=3)
        axes[0].grid(True, alpha=0.3)

        # 最终损失热力图
        model_types = ['RNN', 'LSTM', 'GRU']
        n_layers_list = [1, 2, 3]
        heatmap = np.zeros((3, 3))
        for i, mt in enumerate(model_types):
            for j, nl in enumerate(n_layers_list):
                key = f'{mt}-{nl}层'
                heatmap[i, j] = results[key]['final_val_loss']

        im = axes[1].imshow(heatmap, cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(im, ax=axes[1], label='最终验证 MSE')
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(['1层', '2层', '3层'])
        axes[1].set_yticks(range(3))
        axes[1].set_yticklabels(model_types)
        axes[1].set_title('最终验证损失热力图\n（绿=好，红=差）', fontsize=11, fontweight='bold')
        for i in range(3):
            for j in range(3):
                axes[1].text(j, i, f'{heatmap[i,j]:.4f}',
                             ha='center', va='center', fontsize=9, fontweight='bold')

        plt.suptitle('RNN / LSTM / GRU × 层数 超参数实验', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('exp_model_layers.png', dpi=150, bbox_inches='tight')
        plt.show()

        return results


    # ─────────────────────────────────────────────────────────
    # 双向 RNN 对比
    # ─────────────────────────────────────────────────────────

    def experiment_bidirectional():
        """对比单向 vs 双向 LSTM"""
        configs = {
            '单向 LSTM': dict(model_type='lstm', input_size=1, hidden_size=32,
                              num_layers=1, dropout=0.0, bidirectional=False, output_size=1),
            '双向 LSTM': dict(model_type='lstm', input_size=1, hidden_size=32,
                              num_layers=1, dropout=0.0, bidirectional=True, output_size=1),
            '单向 GRU':  dict(model_type='gru', input_size=1, hidden_size=32,
                              num_layers=1, dropout=0.0, bidirectional=False, output_size=1),
            '双向 GRU':  dict(model_type='gru', input_size=1, hidden_size=32,
                              num_layers=1, dropout=0.0, bidirectional=True, output_size=1),
        }

        results = {}
        for label, config in configs.items():
            print(f"  {label}...")
            results[label] = run_experiment(config, n_epochs=80)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
        for (label, res), color in zip(results.items(), colors):
            axes[0].semilogy(res['val_loss'], linewidth=2, label=label,
                             color=color, alpha=0.85)
        axes[0].set_title('单向 vs 双向：验证损失', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss（log）')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        labels = list(results.keys())
        final = [results[l]['final_val_loss'] for l in labels]
        params = [results[l]['params'] for l in labels]

        ax2 = axes[1]
        scatter = ax2.scatter(params, final, c=colors, s=200, zorder=5,
                              edgecolors='white', linewidth=2)
        for i, label in enumerate(labels):
            ax2.annotate(label, (params[i], final[i]),
                         textcoords='offset points', xytext=(8, 4), fontsize=9)
        ax2.set_xlabel('参数量')
        ax2.set_ylabel('最终验证 MSE')
        ax2.set_title('参数量 vs 性能（双向参数更多）', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('单向 vs 双向 RNN 实验', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('exp_bidirectional.png', dpi=150, bbox_inches='tight')
        plt.show()

        return results


    # ─────────────────────────────────────────────────────────
    # 完整演示
    # ─────────────────────────────────────────────────────────

    def demo_rnn_hyperparams():
        print("实验 1：hidden_size 对比")
        res1 = experiment_hidden_size()

        print("\n实验 2：模型类型 × 层数")
        res2 = experiment_model_type_and_layers()

        print("\n实验 3：单向 vs 双向")
        res3 = experiment_bidirectional()

        # 汇总最优配置
        print("\n" + "="*50)
        print("实验汇总：")
        best_hs = min(res1, key=lambda k: res1[k]['final_val_loss'])
        print(f"  最优 hidden_size: {best_hs}  (val_loss={res1[best_hs]['final_val_loss']:.6f})")

        best_arch = min(res2, key=lambda k: res2[k]['final_val_loss'])
        print(f"  最优架构: {best_arch}  (val_loss={res2[best_arch]['final_val_loss']:.6f})")

        best_bi = min(res3, key=lambda k: res3[k]['final_val_loss'])
        print(f"  最优方向: {best_bi}  (val_loss={res3[best_bi]['final_val_loss']:.6f})")

        return res1, res2, res3

    if __name__ == '__main__':
        smoke_config = dict(model_type='lstm', input_size=1, hidden_size=8,
                            num_layers=1, dropout=0.0, bidirectional=False,
                            output_size=1)
        smoke = run_experiment(smoke_config, n_epochs=1, seq_len=8, n_samples=32)
        print(f"RNN hyperparameter module smoke test passed: val_loss={smoke['final_val_loss']:.6f}")
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/04_hyperparam_rnn.py", e)


def _validate_hyperparam_inputs(
    model_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    sequence_length: int,
) -> tuple[str, int, int, float, float, int]:
    model_type = str(model_type).lower()
    if model_type not in {"rnn", "lstm", "gru"}:
        raise ValueError("model_type 必须是 rnn、lstm 或 gru")
    hidden_size = clamp_int(int(hidden_size), 4, 128, "隐藏维度")
    num_layers = clamp_int(int(num_layers), 1, 4, "层数")
    dropout = clamp_float(float(dropout), 0.0, 0.8, "Dropout")
    learning_rate = clamp_float(float(learning_rate), 0.0001, 0.05, "学习率")
    sequence_length = clamp_int(int(sequence_length), 8, 96, "序列长度")
    return model_type, hidden_size, num_layers, dropout, learning_rate, sequence_length


def _estimate_rnn_params(model_type: str, hidden_size: int, num_layers: int, bidirectional: bool = False) -> int:
    gates = {"rnn": 1, "gru": 3, "lstm": 4}[model_type]
    directions = 2 if bidirectional else 1
    total = 0
    input_size = 1
    for layer in range(num_layers):
        layer_input = input_size if layer == 0 else hidden_size * directions
        one_direction = gates * (layer_input * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
        total += one_direction * directions
    total += hidden_size * directions + 1
    return int(total)


def _score_config(model_type: str, hidden_size: int, num_layers: int, dropout: float, learning_rate: float, sequence_length: int) -> float:
    capacity = np.log2(hidden_size) / 7.0
    memory_bonus = {"rnn": -0.16, "gru": 0.06, "lstm": 0.09}[model_type]
    long_sequence_penalty = max(sequence_length - 32, 0) / 180.0
    layer_bonus = min(num_layers - 1, 2) * 0.035
    layer_penalty = max(num_layers - 2, 0) * 0.045
    dropout_penalty = abs(dropout - (0.12 if num_layers > 1 else 0.02)) * 0.22
    lr_penalty = abs(np.log10(learning_rate) - np.log10(0.003)) * 0.08
    score = 0.58 - 0.30 * capacity - memory_bonus - layer_bonus + layer_penalty + dropout_penalty + lr_penalty + long_sequence_penalty
    return float(np.clip(score, 0.035, 0.92))


def _simulate_training_curves(
    model_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    sequence_length: int,
    epochs: int,
    seed: int,
) -> dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(seed)
    epochs_axis = np.arange(1, epochs + 1)
    final_val = _score_config(model_type, hidden_size, num_layers, dropout, learning_rate, sequence_length)
    convergence = np.clip(learning_rate * 850 / (1 + 0.18 * num_layers), 0.08, 1.6)
    start = 1.15 + sequence_length / 210.0 + rng.normal(0, 0.015)
    train_floor = final_val * (0.72 + max(hidden_size - 48, 0) / 240.0)
    train_loss = train_floor + (start - train_floor) * np.exp(-epochs_axis * convergence / max(epochs, 1) * 3.5)
    val_loss = final_val + (start * 0.9 - final_val) * np.exp(-epochs_axis * convergence / max(epochs, 1) * 2.9)
    if learning_rate > 0.02:
        oscillation = np.sin(epochs_axis * 0.85) * (learning_rate - 0.02) * 7.0
        train_loss = np.maximum(train_loss + oscillation, 0.02)
        val_loss = np.maximum(val_loss + np.abs(oscillation) * 0.55, 0.02)
    if hidden_size > 64 and dropout < 0.05:
        overfit = np.linspace(0, (hidden_size - 64) / 260.0, epochs)
        val_loss = val_loss + overfit
    train_loss += rng.normal(0, 0.006, size=epochs)
    val_loss += rng.normal(0, 0.008, size=epochs)
    train_loss = np.maximum.accumulate(train_loss[::-1])[::-1]
    val_loss = np.maximum(val_loss, 0.015)
    params = _estimate_rnn_params(model_type, hidden_size, num_layers)
    return {
        "epochs": epochs_axis,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "final_val_loss": float(val_loss[-1]),
        "best_val_loss": float(val_loss.min()),
        "params": params,
    }


def _plot_main_curves(curves: dict[str, np.ndarray | float | int], model_type: str) -> object:
    with safe_mpl_figure(figsize=(10, 4.8)) as fig:
        ax = fig.subplots(1, 1)
        ax.plot(curves["epochs"], curves["train_loss"], label="训练损失", color="#00f0ff", linewidth=2)
        ax.plot(curves["epochs"], curves["val_loss"], label="验证损失", color="#00ff88", linewidth=2)
        ax.scatter([int(np.argmin(curves["val_loss"])) + 1], [curves["best_val_loss"]], color="#b000ff", s=80, zorder=5, label="最佳验证点")
        ax.set_title(f"{model_type.upper()} 当前配置的损失曲线", fontsize=12, fontweight="bold")
        ax.set_xlabel("训练轮数")
        ax.set_ylabel("MSE 损失")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig


def _plot_hidden_sweep(
    model_type: str,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    sequence_length: int,
) -> tuple[object, dict[int, float]]:
    sizes = [8, 16, 32, 64, 96, 128]
    losses = {
        size: _score_config(model_type, size, num_layers, dropout, learning_rate, sequence_length)
        + max(size - 80, 0) / 650.0
        for size in sizes
    }
    params = [_estimate_rnn_params(model_type, size, num_layers) for size in sizes]
    with safe_mpl_figure(figsize=(10, 4.2)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(sizes, list(losses.values()), "o-", color="#00ff88", linewidth=2)
        ax1.set_title("隐藏维度 vs 验证损失", fontsize=10, fontweight="bold")
        ax1.set_xlabel("隐藏维度")
        ax1.set_ylabel("估计验证损失")
        ax1.grid(True, alpha=0.25)
        ax2.plot(sizes, params, "o-", color="#b000ff", linewidth=2)
        ax2.set_title("隐藏维度 vs 参数量", fontsize=10, fontweight="bold")
        ax2.set_xlabel("隐藏维度")
        ax2.set_ylabel("参数量")
        ax2.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, losses


def _plot_architecture_heatmap(dropout: float, learning_rate: float, sequence_length: int) -> tuple[object, dict[str, float]]:
    model_types = ["rnn", "lstm", "gru"]
    layers = [1, 2, 3, 4]
    heatmap = np.zeros((len(model_types), len(layers)))
    results: dict[str, float] = {}
    for i, mt in enumerate(model_types):
        for j, layer_count in enumerate(layers):
            value = _score_config(mt, 32, layer_count, dropout if layer_count > 1 else 0.0, learning_rate, sequence_length)
            heatmap[i, j] = value
            results[f"{mt.upper()}-{layer_count}层"] = float(value)
    with safe_mpl_figure(figsize=(8.2, 4.3)) as fig:
        ax = fig.subplots(1, 1)
        im = ax.imshow(heatmap, cmap="RdYlGn_r", aspect="auto")
        fig.colorbar(im, ax=ax, label="估计验证损失")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"{layer}层" for layer in layers])
        ax.set_yticks(range(len(model_types)))
        ax.set_yticklabels([mt.upper() for mt in model_types])
        ax.set_title("模型类型 × 层数：越绿表示越容易训好", fontsize=11, fontweight="bold")
        for i in range(len(model_types)):
            for j in range(len(layers)):
                ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
        fig.tight_layout()
        return fig, results


def _diagnose_hyperparams(
    model_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    sequence_length: int,
    final_val_loss: float,
) -> list[str]:
    notes = []
    if model_type == "rnn" and sequence_length > 32:
        notes.append("普通 RNN 正在处理较长序列，容易出现梯度衰减；真实项目优先试 GRU/LSTM。")
    if hidden_size < 16:
        notes.append("隐藏维度偏小，模型像笔记本页数太少，可能记不住足够模式。")
    if hidden_size > 96:
        notes.append("隐藏维度很大，参数量上升明显；若数据少，验证损失可能先降后升。")
    if num_layers > 2 and dropout < 0.08:
        notes.append("多层 RNN 没有足够 Dropout 时容易过拟合，验证曲线可能抬头。")
    if learning_rate > 0.02:
        notes.append("学习率偏大，曲线可能震荡；工程上先降到 0.001~0.005 再观察。")
    if learning_rate < 0.0005:
        notes.append("学习率偏小，训练会很慢；如果曲线平得像直线，先把学习率放大 2~5 倍。")
    if final_val_loss < 0.18:
        notes.append("当前配置趋势较好：损失低、参数量还可控，可以作为下一轮实训基线。")
    return notes or ["当前参数没有明显危险信号，适合继续做局部微调。"]


def compute_rnn_hyperparams(
    model_type: str = "lstm",
    hidden_size: int = 32,
    num_layers: int = 2,
    dropout: float = 0.12,
    learning_rate: float = 0.003,
    sequence_length: int = 32,
    epochs: int = 45,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute a lightweight, deterministic RNN hyperparameter lesson."""

    model_type, hidden_size, num_layers, dropout, learning_rate, sequence_length = _validate_hyperparam_inputs(
        model_type, hidden_size, num_layers, dropout, learning_rate, sequence_length
    )
    epochs = clamp_int(int(epochs), 8, 160, "训练轮数")
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        curves = _simulate_training_curves(model_type, hidden_size, num_layers, dropout, learning_rate, sequence_length, epochs, seed)
        notes = _diagnose_hyperparams(
            model_type,
            hidden_size,
            num_layers,
            dropout,
            learning_rate,
            sequence_length,
            float(curves["final_val_loss"]),
        )
        print("RNN 超参数协议化计算")
        print(f"配置: model={model_type.upper()}, hidden={hidden_size}, layers={num_layers}, dropout={dropout:.2f}, lr={learning_rate:.4f}, seq_len={sequence_length}")
        print(f"参数量估计: {curves['params']}")
        print(f"最佳验证损失: {curves['best_val_loss']:.4f}, 最终验证损失: {curves['final_val_loss']:.4f}")
        for index, note in enumerate(notes, 1):
            print(f"诊断 {index}: {note}")
    main_fig = _plot_main_curves(curves, model_type)
    sweep_fig, hidden_sweep = _plot_hidden_sweep(model_type, num_layers, dropout, learning_rate, sequence_length)
    heatmap_fig, architecture_scores = _plot_architecture_heatmap(dropout, learning_rate, sequence_length)
    figures = [
        ("rnn_hyperparam_loss_curves.png", main_fig),
        ("rnn_hyperparam_hidden_sweep.png", sweep_fig),
        ("rnn_hyperparam_architecture_heatmap.png", heatmap_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "params": int(curves["params"]),
        "best_val_loss": float(curves["best_val_loss"]),
        "final_val_loss": float(curves["final_val_loss"]),
        "best_hidden_size": min(hidden_sweep, key=hidden_sweep.get),
        "best_architecture": min(architecture_scores, key=architecture_scores.get),
    }
    return {
        "figures": figures,
        "artifacts": artifacts,
        "stats": stats,
        "curves": curves,
        "hidden_sweep": hidden_sweep,
        "architecture_scores": architecture_scores,
        "log": log_buffer.getvalue(),
    }


def _go_to_training_dynamics() -> None:
    import streamlit as st

    st.query_params["module"] = "part5_toolbox/03_training_dynamics"
    st.rerun()


def render() -> None:
    """Render the refactored RNN hyperparameter lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在组织超参数实验：损失曲线、容量扫描、架构热力图会同步更新")
        st.markdown(
            """
            **零基础直觉：**调 RNN 超参数像给一个学生安排学习方式。隐藏维度决定笔记本有多厚，层数决定思考要转几道弯，
            Dropout 像故意遮住一部分提示防止死记硬背，学习率决定每次改错迈多大步。图里的每条曲线都在回答同一个问题：
            **这个模型是在认真学规律，还是只是在记训练样本、乱跳或者学得太慢？**
            """
        )
        with st.sidebar:
            model_type = st.selectbox("模型类型", ["rnn", "lstm", "gru"], index=1, format_func=str.upper)
            hidden_size = st.slider("隐藏维度", 4, 128, 32, 4)
            num_layers = st.slider("层数", 1, 4, 2)
            dropout = st.slider("Dropout", 0.0, 0.8, 0.12, 0.02)
            learning_rate = st.slider("学习率", 0.0001, 0.05, 0.003, 0.0001, format="%.4f")
            sequence_length = st.slider("序列长度", 8, 96, 32, 4)
            epochs = st.slider("训练轮数", 8, 120, 45, 1)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：训练动态分析", width="stretch"):
                _go_to_training_dynamics()

        data = compute_rnn_hyperparams(
            model_type,
            hidden_size,
            num_layers,
            dropout,
            learning_rate,
            sequence_length,
            epochs,
            int(seed),
            save_artifacts=True,
        )
        stats = data["stats"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("参数量估计", f"{stats['params']:,}")
        k2.metric("最佳验证损失", f"{stats['best_val_loss']:.3f}")
        k3.metric("推荐隐藏维度", str(stats["best_hidden_size"]))
        k4.metric("推荐结构", str(stats["best_architecture"]))
        st.info("本页使用可重复的轻量教学实验来呈现趋势；真实训练请进入工具箱训练动态页，用同一组参数做完整数据验证。")

        explainers = [
            (
                "训练/验证损失曲线怎么看",
                "训练损失下降说明模型在记住训练样本；验证损失同步下降才说明它学到了可迁移规律。如果训练线继续下降、验证线抬头，就是过拟合。",
            ),
            (
                "隐藏维度扫描怎么看",
                "隐藏维度越大，模型能保存的信息越多，但参数量也按平方级上升。绿色曲线下降到某处后变平，说明再加容量收益变小。",
            ),
            (
                "架构热力图怎么看",
                "每个格子是一种“模型类型 × 层数”。颜色越绿表示估计验证损失越低；普通 RNN 在长序列上通常吃亏，LSTM/GRU 更稳。",
            ),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请拖动左侧一个参数，只改这一项，再观察三张图哪一张最先变化。思考：这是容量变化、优化变化，还是正则化变化？")

        with st.expander("常见误区、工程经验与控制台输出", expanded=False):
            st.markdown(
                """
                - **误区 1：隐藏维度越大越好。** 正确理解：容量变大后训练损失更容易下降，但数据少时验证损失会反弹。
                - **误区 2：层数越深越高级。** 正确理解：RNN 层数增加会让优化更难，2 层常常是教学和小项目的稳妥起点。
                - **误区 3：Dropout 只是随便关神经元。** 正确理解：它是在训练时制造扰动，逼模型不要依赖某个固定捷径。
                - **工程经验：**序列预测基线通常从 `GRU/LSTM + hidden=32/64 + layers=1/2 + lr=0.001~0.003` 开始，再按验证集微调。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/04_hyperparam_rnn.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_rnn_hyperparams(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_rnn_hyperparams(model_type="gru", hidden_size=8, num_layers=1, epochs=8, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["params"] > 0 and data["stats"]["best_val_loss"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_rnn_hyperparams))
