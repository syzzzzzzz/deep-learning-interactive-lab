"""RNN 高级训练：Teacher Forcing、预定采样、截断 BPTT 与梯度裁剪。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "RNN 高级训练"
MODULE_SUMMARY = "用可控曲线解释 Teacher Forcing、预定采样、截断 BPTT 和梯度裁剪如何让序列模型更稳定。"
MODULE_TAGS = ["RNN", "训练技巧", "Teacher Forcing", "BPTT", "梯度裁剪"]
MODULE_RELATED_TOPICS = ["part3/04_hyperparam_rnn", "part3/05_seq2seq_attention", "part5/02_gradient_monitor", "part5/03_training_dynamics"]
PRACTICE_TARGET = "调整 teacher forcing 比率、采样策略、BPTT 段长和裁剪阈值，解释训练速度、推理偏差和梯度稳定性如何变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    自动生成自: part3_rnn\07_advanced_training.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


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


    # compare_teacher_forcing_ratios()  # 协议化后由 compute_advanced_training() 控制执行

    # ============================================================
    # 代码段 2
    # ============================================================

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


    # visualize_scheduled_sampling()  # 协议化后由 compute_advanced_training() 控制执行

    # ============================================================
    # 代码段 3
    # ============================================================

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


    # compare_bptt_strategies()  # 协议化后由 compute_advanced_training() 控制执行

    # ============================================================
    # 代码段 4
    # ============================================================

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


    # demonstrate_gradient_clipping()  # 协议化后由 compute_advanced_training() 控制执行

    # ============================================================
    # 代码段 5
    # ============================================================

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


    # training_best_practices()  # 协议化后由 render() 展示
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/07_advanced_training.py", e)


def _scheduled_ratio(strategy: str, epoch: np.ndarray, k: float) -> np.ndarray:
    if strategy == "linear":
        return np.maximum(0.0, 1.0 - epoch / max(k, 1e-6))
    if strategy == "exponential":
        return np.power(clamp_float(k, 0.80, 0.995, "指数衰减系数"), epoch)
    if strategy == "inverse_sigmoid":
        return k / (k + np.exp(epoch / max(k, 1e-6)))
    raise ValueError("strategy 必须是 linear、exponential 或 inverse_sigmoid")


def _plot_teacher_forcing(tf_ratio: float, epochs: int, seed: int) -> tuple[object, dict[str, float]]:
    rng = np.random.default_rng(seed)
    axis = np.arange(1, epochs + 1)
    train_speed = 1.8 + 2.4 * tf_ratio
    exposure_gap = max(tf_ratio - 0.45, 0) * 0.22
    train_loss = 0.18 + (1.15 - 0.18) * np.exp(-axis / epochs * train_speed * 2.1)
    inference_acc = 0.54 + 0.34 * (1 - np.exp(-axis / epochs * (2.4 - exposure_gap * 2))) - exposure_gap * np.linspace(0.2, 1.0, epochs)
    train_loss += rng.normal(0, 0.006, epochs)
    inference_acc += rng.normal(0, 0.007, epochs)
    inference_acc = np.clip(inference_acc, 0.35, 0.96)
    with safe_mpl_figure(figsize=(10.5, 4.2)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(axis, train_loss, color="#00f0ff", linewidth=2)
        ax1.set_title("Teacher Forcing 越高，训练越像看答案", fontsize=10, fontweight="bold")
        ax1.set_xlabel("训练轮数")
        ax1.set_ylabel("训练损失")
        ax1.grid(True, alpha=0.25)
        ax2.plot(axis, inference_acc, color="#00ff88", linewidth=2)
        ax2.set_title("推理准确率：验证时不再给答案", fontsize=10, fontweight="bold")
        ax2.set_xlabel("训练轮数")
        ax2.set_ylabel("自回归准确率")
        ax2.set_ylim(0.3, 1.02)
        ax2.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, {"final_train_loss": float(train_loss[-1]), "final_inference_acc": float(inference_acc[-1])}


def _plot_scheduled_sampling(strategy: str, k: float, epochs: int) -> tuple[object, dict[str, float]]:
    axis = np.arange(0, epochs)
    ratios = _scheduled_ratio(strategy, axis, k)
    baseline = np.full_like(ratios, 1.0, dtype=float)
    with safe_mpl_figure(figsize=(8.6, 4.3)) as fig:
        ax = fig.subplots(1, 1)
        ax.plot(axis, ratios, color="#b000ff", linewidth=2.4, label=f"{strategy}")
        ax.plot(axis, baseline, color="#bf3f5b", linestyle="--", alpha=0.7, label="一直 teacher forcing")
        ax.axhline(0.5, color="#777", linestyle=":", alpha=0.7)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("训练轮数")
        ax.set_ylabel("Teacher Forcing 比率")
        ax.set_title("预定采样：从看答案逐步过渡到自己生成", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, {"start_ratio": float(ratios[0]), "end_ratio": float(ratios[-1]), "mean_ratio": float(ratios.mean())}


def _plot_bptt(segment_len: int, sequence_length: int) -> tuple[object, dict[str, float]]:
    segments = np.array([4, 8, 16, 32, 48, 64])
    sequence_length = clamp_int(sequence_length, 20, 160, "序列长度")
    visible = segments[segments <= sequence_length]
    memory = visible / max(sequence_length, 1)
    compute_cost = visible * np.log2(visible + 1)
    stability = np.exp(-visible / 90)
    selected_idx = int(np.argmin(np.abs(visible - segment_len)))
    with safe_mpl_figure(figsize=(10.2, 4.2)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(visible, memory, "o-", color="#00ff88", label="可回看的历史比例")
        ax1.plot(visible, stability, "s-", color="#00f0ff", label="梯度稳定性")
        ax1.scatter([visible[selected_idx]], [memory[selected_idx]], color="#b000ff", s=90, zorder=5)
        ax1.set_title("截断 BPTT 段长的取舍", fontsize=10, fontweight="bold")
        ax1.set_xlabel("段长")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)
        ax2.plot(visible, compute_cost, "o-", color="#bf3f5b", linewidth=2)
        ax2.set_title("段长越大，显存和计算越贵", fontsize=10, fontweight="bold")
        ax2.set_xlabel("段长")
        ax2.set_ylabel("相对计算代价")
        ax2.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, {"selected_memory_ratio": float(memory[selected_idx]), "selected_compute_cost": float(compute_cost[selected_idx])}


def _plot_gradient_clipping(clip_norm: float, seed: int) -> tuple[object, dict[str, float]]:
    rng = np.random.default_rng(seed + 73)
    raw = rng.lognormal(mean=0.2, sigma=0.9, size=160)
    clipped = np.minimum(raw, clip_norm)
    with safe_mpl_figure(figsize=(9.5, 4.2)) as fig:
        ax = fig.subplots(1, 1)
        ax.hist(raw, bins=26, color="#bf3f5b", alpha=0.45, label="原始梯度范数")
        ax.hist(clipped, bins=26, color="#00ff88", alpha=0.65, label="裁剪后")
        ax.axvline(clip_norm, color="#00f0ff", linestyle="--", linewidth=2, label=f"阈值 {clip_norm:.1f}")
        ax.set_title("梯度裁剪：把过大的更新压回安全范围", fontsize=11, fontweight="bold")
        ax.set_xlabel("梯度范数")
        ax.set_ylabel("频次")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        return fig, {"raw_max": float(raw.max()), "clipped_max": float(clipped.max()), "clipped_fraction": float((raw > clip_norm).mean())}


def compute_advanced_training(
    teacher_forcing_ratio: float = 0.6,
    sampling_strategy: str = "inverse_sigmoid",
    sampling_k: float = 8.0,
    sequence_length: int = 80,
    bptt_segment_len: int = 16,
    clip_norm: float = 1.0,
    epochs: int = 50,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute advanced RNN training visuals without heavy top-level training."""

    teacher_forcing_ratio = clamp_float(teacher_forcing_ratio, 0.0, 1.0, "Teacher Forcing 比率")
    sampling_k = clamp_float(sampling_k, 1.0, 40.0, "采样参数 k")
    sequence_length = clamp_int(sequence_length, 20, 160, "序列长度")
    bptt_segment_len = clamp_int(bptt_segment_len, 4, min(sequence_length, 64), "BPTT 段长")
    clip_norm = clamp_float(clip_norm, 0.2, 8.0, "梯度裁剪阈值")
    epochs = clamp_int(epochs, 12, 160, "训练轮数")
    tf_fig, tf_stats = _plot_teacher_forcing(teacher_forcing_ratio, epochs, seed)
    schedule_fig, schedule_stats = _plot_scheduled_sampling(sampling_strategy, sampling_k, epochs)
    bptt_fig, bptt_stats = _plot_bptt(bptt_segment_len, sequence_length)
    clip_fig, clip_stats = _plot_gradient_clipping(clip_norm, seed)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("RNN 高级训练协议化计算")
        print(f"teacher_forcing_ratio={teacher_forcing_ratio:.2f}, strategy={sampling_strategy}, k={sampling_k:.2f}")
        print(f"sequence_length={sequence_length}, bptt_segment_len={bptt_segment_len}, clip_norm={clip_norm:.2f}")
        print(f"推理准确率估计={tf_stats['final_inference_acc']:.3f}, 裁剪比例={clip_stats['clipped_fraction']:.2%}")
        if teacher_forcing_ratio > 0.8:
            print("诊断：Teacher Forcing 很高，训练会快，但推理时可能出现 exposure bias。")
        if clip_stats["clipped_fraction"] > 0.35:
            print("诊断：很多梯度被裁剪，可能学习率偏大或序列太长。")
        print("工程经验：Seq2Seq 通常从高 teacher forcing 起步，再用预定采样降低；RNN 长序列几乎总要配梯度裁剪。")
    figures = [
        ("advanced_training_teacher_forcing.png", tf_fig),
        ("advanced_training_scheduled_sampling.png", schedule_fig),
        ("advanced_training_bptt.png", bptt_fig),
        ("advanced_training_gradient_clipping.png", clip_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {**tf_stats, **schedule_stats, **bptt_stats, **clip_stats}
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "log": log_buffer.getvalue()}


def _go_to_gradient_monitor() -> None:
    import streamlit as st

    st.query_params["module"] = "part5_toolbox/02_gradient_monitor"
    st.rerun()


def render() -> None:
    """Render the refactored advanced RNN training lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_backprop_current_flow, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成训练技巧图谱：Teacher Forcing、预定采样、BPTT 与梯度裁剪")
        with st.sidebar:
            teacher_forcing_ratio = st.slider("Teacher Forcing 比率", 0.0, 1.0, 0.6, 0.05)
            sampling_strategy = st.selectbox("预定采样策略", ["inverse_sigmoid", "linear", "exponential"])
            sampling_k = st.slider("采样参数 k", 1.0, 40.0, 8.0, 0.5)
            sequence_length = st.slider("序列长度", 20, 160, 80, 4)
            bptt_segment_len = st.slider("BPTT 段长", 4, 64, 16, 4)
            clip_norm = st.slider("梯度裁剪阈值", 0.2, 8.0, 1.0, 0.1)
            epochs = st.slider("训练轮数", 12, 160, 50, 1)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：梯度监控", width="stretch"):
                _go_to_gradient_monitor()
        data = compute_advanced_training(
            teacher_forcing_ratio,
            sampling_strategy,
            sampling_k,
            sequence_length,
            bptt_segment_len,
            clip_norm,
            epochs,
            int(seed),
            save_artifacts=True,
        )
        stats = data["stats"]
        render_backprop_current_flow()
        st.markdown(
            """
            **零基础直觉：**RNN 训练不是只按“开始训练”就完事。Teacher Forcing 决定训练时给不给标准答案，
            预定采样决定什么时候撤掉答案，截断 BPTT 决定反向传播回看多远，梯度裁剪决定更新步子会不会失控。
            """
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("推理准确率估计", f"{stats['final_inference_acc']:.1%}")
        m2.metric("采样末端比例", f"{stats['end_ratio']:.2f}")
        m3.metric("可回看比例", f"{stats['selected_memory_ratio']:.1%}")
        m4.metric("梯度裁剪比例", f"{stats['clipped_fraction']:.1%}")
        explainers = [
            ("Teacher Forcing", "训练时一直给答案会让损失下降很快，但推理时模型必须吃自己的输出，错误会连锁传播。"),
            ("预定采样", "它像训练辅助轮：一开始扶着模型，随后逐步放手，让训练环境更接近真实推理。"),
            ("截断 BPTT", "段长越大，模型能回看更久，但显存和不稳定性也上升；段长太短又学不到长依赖。"),
            ("梯度裁剪", "当梯度像电流过载时，裁剪把它限制在安全范围，防止一次更新把模型参数冲坏。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请只调整对应控件一次，观察曲线是否更稳。思考：你是在提高训练速度，还是在降低训练和推理之间的差距？")
        with st.expander("工程清单与控制台输出", expanded=False):
            st.markdown(
                """
                - **Teacher Forcing**：翻译/生成任务常从 0.5~1.0 起步，再逐步降低。
                - **截断 BPTT**：真实长序列常用 20~50 步做段长，段间 detach 隐藏状态。
                - **梯度裁剪**：RNN/LSTM/GRU 几乎默认加，`max_norm=1.0~5.0` 是常见起点。
                - **判断标准**：不要只看训练损失，要同时看推理模式下的验证准确率和梯度范数。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/07_advanced_training.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_advanced_training(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_advanced_training(epochs=12, sequence_length=32, bptt_segment_len=8, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["clipped_max"] <= 1.0 and data["stats"]["final_inference_acc"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_advanced_training))
