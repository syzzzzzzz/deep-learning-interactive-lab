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
