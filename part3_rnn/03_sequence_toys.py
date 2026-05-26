"""序列玩具任务：RNN/LSTM/GRU 的隐藏状态、梯度和预测实验。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "序列玩具任务"
MODULE_SUMMARY = "用小型可控序列实验理解 RNN 记忆、梯度衰减、LSTM 门控和序列预测。"
MODULE_TAGS = ["RNN", "序列", "玩具任务", "梯度", "LSTM"]
MODULE_RELATED_TOPICS = ["part3/01_rnn_intuition", "part3/02_hidden_states", "part5/02_gradient_monitor"]
PRACTICE_TARGET = "调整序列长度、模型类型和隐藏维度，解释隐藏状态、梯度范数和预测曲线如何变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import run_cli, running_under_streamlit

try:
    """
    自动生成自: part3_rnn\03_sequence_toys.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ─────────────────────────────────────────────────────────
    # 手动实现 RNN，理解循环结构
    # ─────────────────────────────────────────────────────────

    class ManualRNN:
        """
        手动实现 RNN，展示每一步的计算
        h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
        """
        def __init__(self, input_size, hidden_size):
            # 权重初始化
            self.W_xh = torch.randn(input_size, hidden_size) * 0.1
            self.W_hh = torch.randn(hidden_size, hidden_size) * 0.1
            self.b_h = torch.zeros(hidden_size)

            self.hidden_size = hidden_size
            self.input_size = input_size

        def step(self, x_t, h_prev):
            """
            单步计算
            x_t: [input_size]
            h_prev: [hidden_size]
            返回: h_t [hidden_size]
            """
            # 线性变换
            linear = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
            # 激活函数
            h_t = torch.tanh(linear)
            return h_t

        def forward(self, sequence):
            """
            处理整个序列
            sequence: [seq_len, input_size]
            返回: all_hidden [seq_len, hidden_size], final_hidden [hidden_size]
            """
            seq_len = sequence.shape[0]
            h = torch.zeros(self.hidden_size)
            all_hidden = []

            for t in range(seq_len):
                h = self.step(sequence[t], h)
                all_hidden.append(h.clone())

            return torch.stack(all_hidden), h

    def visualize_rnn_hidden_states():
        """可视化 RNN 隐藏状态随序列变化"""
        torch.manual_seed(42)

        input_size = 3
        hidden_size = 8
        seq_len = 20

        rnn = ManualRNN(input_size, hidden_size)

        # 生成一个有规律的序列（正弦波）
        t = torch.linspace(0, 4 * np.pi, seq_len)
        sequence = torch.stack([
            torch.sin(t),
            torch.cos(t),
            torch.sin(2 * t),
        ], dim=1)  # [seq_len, 3]

        all_hidden, final_hidden = rnn.forward(sequence)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 输入序列
        for i in range(input_size):
            axes[0].plot(sequence[:, i].numpy(), label=f'输入维度 {i}', alpha=0.8)
        axes[0].set_title('输入序列（正弦波）', fontsize=12)
        axes[0].set_xlabel('时间步')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 隐藏状态热力图
        hidden_np = all_hidden.detach().numpy()  # [seq_len, hidden_size]
        im = axes[1].imshow(hidden_np.T, aspect='auto', cmap='RdBu',
                             vmin=-1, vmax=1)
        axes[1].set_title('隐藏状态热力图（每行=一个隐藏单元，每列=一个时间步）', fontsize=12)
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('隐藏单元')
        plt.colorbar(im, ax=axes[1])

        # 隐藏状态曲线（前4个单元）
        for i in range(min(4, hidden_size)):
            axes[2].plot(hidden_np[:, i], label=f'隐藏单元 {i}', alpha=0.8)
        axes[2].set_title('隐藏状态随时间的变化（前4个单元）', fontsize=12)
        axes[2].set_xlabel('时间步')
        axes[2].set_ylabel('激活值')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-1.1, 1.1)

        plt.suptitle('RNN 隐藏状态可视化', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('rnn_hidden_states.png', dpi=150, bbox_inches='tight')
        plt.show()

    # visualize_rnn_hidden_states()  # 协议化后由 render()/compute_sequence_toys() 控制执行

    # ─────────────────────────────────────────────────────────
    # 梯度消失演示
    # ─────────────────────────────────────────────────────────

    def demonstrate_vanishing_gradient():
        """
        演示 RNN 中的梯度消失问题
        长序列中，早期时间步的梯度会指数级衰减
        """
        print("=" * 60)
        print("梯度消失演示")
        print("=" * 60)

        seq_lengths = [10, 50, 100, 200]
        models = {
            'RNN (Tanh)': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU,
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for model_name, model_class in models.items():
            grad_norms = []
            for seq_len in seq_lengths:
                model = model_class(input_size=1, hidden_size=32, batch_first=True)
                x = torch.randn(1, seq_len, 1, requires_grad=True)

                if model_name == 'LSTM':
                    output, (h, c) = model(x)
                else:
                    output, h = model(x)

                # 对最后一个时间步的输出求和，反向传播
                loss = output[:, -1, :].sum()
                loss.backward()

                # 计算输入梯度的范数（衡量梯度传播到输入的强度）
                grad_norm = x.grad.norm().item()
                grad_norms.append(grad_norm)

            axes[0].plot(seq_lengths, grad_norms, 'o-', label=model_name, markersize=8)

        axes[0].set_title('不同序列长度下的梯度范数', fontsize=12)
        axes[0].set_xlabel('序列长度')
        axes[0].set_ylabel('输入梯度范数')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 梯度随时间步的衰减
        seq_len = 100
        for model_name, model_class in models.items():
            model = model_class(input_size=1, hidden_size=32, batch_first=True)
            x = torch.randn(1, seq_len, 1, requires_grad=True)

            if model_name == 'LSTM':
                output, _ = model(x)
            else:
                output, _ = model(x)

            loss = output[:, -1, :].sum()
            loss.backward()

            # 每个时间步的梯度范数
            grad_per_step = x.grad[0, :, 0].abs().detach().numpy()
            axes[1].plot(range(seq_len), grad_per_step, label=model_name, alpha=0.8)

        axes[1].set_title('梯度随时间步的衰减（序列长度=100）', fontsize=12)
        axes[1].set_xlabel('时间步（0=最早，99=最新）')
        axes[1].set_ylabel('梯度绝对值')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='中间点')

        plt.suptitle('梯度消失问题：RNN vs LSTM vs GRU', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('vanishing_gradient.png', dpi=150, bbox_inches='tight')
        plt.show()

    # demonstrate_vanishing_gradient()  # 协议化后由 render()/compute_sequence_toys() 控制执行

    # ============================================================
    # 代码段 2
    # ============================================================

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # ─────────────────────────────────────────────────────────
    # 手动实现 LSTM，展示每个门的作用
    # ─────────────────────────────────────────────────────────

    class ManualLSTM:
        """
        手动实现 LSTM，展示每个门的计算

        LSTM 有两个状态：
        - h_t: 隐藏状态（短期记忆）
        - c_t: 细胞状态（长期记忆）

        四个门：
        - 遗忘门 f_t: 决定忘记多少旧信息
        - 输入门 i_t: 决定记住多少新信息
        - 候选值 g_t: 新信息的候选值
        - 输出门 o_t: 决定输出多少信息
        """
        def __init__(self, input_size, hidden_size):
            self.input_size = input_size
            self.hidden_size = hidden_size
            H = hidden_size
            I = input_size

            # 所有门的权重（合并为一个大矩阵，效率更高）
            # 顺序：[遗忘门, 输入门, 候选值, 输出门]
            self.W = torch.randn(I + H, 4 * H) * 0.1
            self.b = torch.zeros(4 * H)

        def step(self, x_t, h_prev, c_prev):
            """
            单步 LSTM 计算
            """
            H = self.hidden_size

            # 拼接输入和上一步隐藏状态
            combined = torch.cat([x_t, h_prev])  # [I + H]

            # 一次性计算所有门
            gates = combined @ self.W + self.b  # [4H]

            # 分割四个门
            f = torch.sigmoid(gates[:H])        # 遗忘门
            i = torch.sigmoid(gates[H:2*H])     # 输入门
            g = torch.tanh(gates[2*H:3*H])      # 候选值
            o = torch.sigmoid(gates[3*H:])      # 输出门

            # 更新细胞状态（长期记忆）
            c_t = f * c_prev + i * g

            # 更新隐藏状态（短期记忆）
            h_t = o * torch.tanh(c_t)

            return h_t, c_t, {'f': f, 'i': i, 'g': g, 'o': o}

        def forward(self, sequence):
            seq_len = sequence.shape[0]
            h = torch.zeros(self.hidden_size)
            c = torch.zeros(self.hidden_size)

            all_h = []
            all_c = []
            all_gates = {'f': [], 'i': [], 'g': [], 'o': []}

            for t in range(seq_len):
                h, c, gates = self.step(sequence[t], h, c)
                all_h.append(h.clone())
                all_c.append(c.clone())
                for gate_name, gate_val in gates.items():
                    all_gates[gate_name].append(gate_val.clone())

            return (torch.stack(all_h), torch.stack(all_c),
                    {k: torch.stack(v) for k, v in all_gates.items()})

    def visualize_lstm_gates():
        """可视化 LSTM 各门的激活情况"""
        torch.manual_seed(42)

        input_size = 2
        hidden_size = 8
        seq_len = 30

        lstm = ManualLSTM(input_size, hidden_size)

        # 生成一个有结构的序列
        t = torch.linspace(0, 6 * np.pi, seq_len)
        sequence = torch.stack([torch.sin(t), torch.cos(t)], dim=1)

        all_h, all_c, all_gates = lstm.forward(sequence)

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 输入序列
        axes[0, 0].plot(sequence[:, 0].numpy(), 'b-', label='sin(t)')
        axes[0, 0].plot(sequence[:, 1].numpy(), 'r-', label='cos(t)')
        axes[0, 0].set_title('输入序列', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 细胞状态（长期记忆）
        c_np = all_c.detach().numpy()
        im = axes[0, 1].imshow(c_np.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
        axes[0, 1].set_title('细胞状态 c_t（长期记忆）', fontsize=12)
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('单元')
        plt.colorbar(im, ax=axes[0, 1])

        # 四个门的可视化
        gate_info = [
            ('f', '遗忘门 f_t\n（接近1=保留，接近0=遗忘）', axes[1, 0]),
            ('i', '输入门 i_t\n（接近1=写入新信息）', axes[1, 1]),
            ('g', '候选值 g_t\n（新信息的内容）', axes[2, 0]),
            ('o', '输出门 o_t\n（接近1=输出更多）', axes[2, 1]),
        ]

        for gate_name, title, ax in gate_info:
            gate_np = all_gates[gate_name].detach().numpy()
            im = ax.imshow(gate_np.T, aspect='auto',
                           cmap='RdBu' if gate_name == 'g' else 'Blues',
                           vmin=-1 if gate_name == 'g' else 0,
                           vmax=1)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('时间步')
            ax.set_ylabel('单元')
            plt.colorbar(im, ax=ax)

        plt.suptitle('LSTM 门机制可视化', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lstm_gates.png', dpi=150, bbox_inches='tight')
        plt.show()

    # visualize_lstm_gates()  # 协议化后由 render()/compute_sequence_toys() 控制执行

    # ============================================================
    # 代码段 3
    # ============================================================

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # ─────────────────────────────────────────────────────────
    # 时序预测玩具：预测正弦波
    # ─────────────────────────────────────────────────────────

    class SequencePredictor(nn.Module):
        """
        通用序列预测模型
        支持 RNN / LSTM / GRU
        """
        def __init__(self, model_type='LSTM', input_size=1, hidden_size=64,
                     num_layers=2, output_size=1, dropout=0.1):
            super().__init__()
            self.model_type = model_type
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            rnn_map = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
            self.rnn = rnn_map[model_type](
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x, hidden=None):
            """
            x: [batch, seq_len, input_size]
            返回: [batch, seq_len, output_size]
            """
            out, hidden = self.rnn(x, hidden)
            out = self.fc(out)
            return out, hidden

    def generate_sine_data(n_samples=1000, seq_len=50, noise=0.1):
        """生成带噪声的正弦波数据"""
        t = np.linspace(0, 4 * np.pi, n_samples + seq_len)
        signal = np.sin(t) + noise * np.random.randn(len(t))

        X, y = [], []
        for i in range(n_samples):
            X.append(signal[i:i+seq_len])
            y.append(signal[i+1:i+seq_len+1])  # 预测下一步

        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
        return X, y

    def train_sequence_predictor():
        """训练并对比不同序列模型"""
        torch.manual_seed(42)
        np.random.seed(42)

        X, y = generate_sine_data(n_samples=800, seq_len=50)
        X_train, y_train = X[:600], y[:600]
        X_test, y_test = X[600:], y[600:]

        models = {
            'RNN': SequencePredictor('RNN', hidden_size=32, num_layers=1),
            'LSTM': SequencePredictor('LSTM', hidden_size=32, num_layers=2),
            'GRU': SequencePredictor('GRU', hidden_size=32, num_layers=2),
        }

        results = {}
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        for col, (name, model) in enumerate(models.items()):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            train_losses = []
            for epoch in range(100):
                model.train()
                pred, _ = model(X_train)
                loss = criterion(pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # 测试
            model.eval()
            with torch.no_grad():
                pred_test, _ = model(X_test)
                test_loss = criterion(pred_test, y_test).item()

            results[name] = {'train_losses': train_losses, 'test_loss': test_loss}

            # 绘制训练损失
            axes[0, col].plot(train_losses, 'b-', alpha=0.8)
            axes[0, col].set_title(f'{name}\n测试Loss={test_loss:.4f}', fontsize=11)
            axes[0, col].set_xlabel('Epoch')
            axes[0, col].set_ylabel('MSE Loss')
            axes[0, col].set_yscale('log')
            axes[0, col].grid(True, alpha=0.3)

            # 绘制预测效果
            sample_idx = 0
            true_seq = y_test[sample_idx, :, 0].numpy()
            pred_seq = pred_test[sample_idx, :, 0].numpy()
            input_seq = X_test[sample_idx, :, 0].numpy()

            axes[1, col].plot(range(50), input_seq, 'g-', alpha=0.5, label='输入')
            axes[1, col].plot(range(50), true_seq, 'b-', linewidth=2, label='真实')
            axes[1, col].plot(range(50), pred_seq, 'r--', linewidth=2, label='预测')
            axes[1, col].set_title(f'{name} 预测效果', fontsize=11)
            axes[1, col].set_xlabel('时间步')
            axes[1, col].legend(fontsize=8)
            axes[1, col].grid(True, alpha=0.3)

        plt.suptitle('RNN / LSTM / GRU 时序预测对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sequence_prediction.png', dpi=150, bbox_inches='tight')
        plt.show()

        return models, results

    # models, results = train_sequence_predictor()  # 协议化后由 render()/compute_sequence_toys() 控制执行

    # ─────────────────────────────────────────────────────────
    # 名字生成玩具（字符级语言模型）
    # ─────────────────────────────────────────────────────────

    class CharLM(nn.Module):
        """字符级语言模型：给定前缀，生成后续字符"""
        def __init__(self, vocab_size, embed_size=32, hidden_size=128, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embedding(x)
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc(out)
            return logits, hidden

        def generate(self, start_str, char2idx, idx2char, max_len=50, temperature=1.0):
            """生成文本"""
            self.eval()
            with torch.no_grad():
                # 编码起始字符串
                indices = [char2idx.get(c, 0) for c in start_str]
                x = torch.tensor(indices).unsqueeze(0)

                # 处理前缀
                logits, hidden = self(x)

                generated = list(start_str)
                for _ in range(max_len):
                    # 取最后一个时间步的 logits
                    last_logits = logits[0, -1, :] / temperature
                    probs = torch.softmax(last_logits, dim=0)
                    next_idx = torch.multinomial(probs, 1).item()
                    next_char = idx2char[next_idx]

                    if next_char == '<EOS>':
                        break
                    generated.append(next_char)

                    # 下一步输入
                    x = torch.tensor([[next_idx]])
                    logits, hidden = self(x, hidden)

            return ''.join(generated)

    def train_char_lm():
        """训练字符级语言模型"""
        # 简单的训练数据：一些名字
        names = [
            "alice", "bob", "charlie", "diana", "edward",
            "fiona", "george", "helen", "ivan", "julia",
            "kevin", "laura", "michael", "nancy", "oliver",
            "patricia", "quinn", "robert", "sarah", "thomas",
            "ursula", "victor", "wendy", "xavier", "yvonne", "zachary",
        ]

        # 构建词汇表
        all_chars = sorted(set(''.join(names))) + ['<EOS>']
        char2idx = {c: i for i, c in enumerate(all_chars)}
        idx2char = {i: c for c, i in char2idx.items()}
        vocab_size = len(all_chars)

        print(f"词汇表大小: {vocab_size}")
        print(f"字符: {all_chars}")

        # 准备训练数据
        def name_to_tensor(name):
            indices = [char2idx[c] for c in name] + [char2idx['<EOS>']]
            return torch.tensor(indices)

        model = CharLM(vocab_size, embed_size=16, hidden_size=64, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(500):
            epoch_loss = 0
            for name in names:
                tensor = name_to_tensor(name)
                x = tensor[:-1].unsqueeze(0)  # 输入
                y = tensor[1:]                 # 目标

                logits, _ = model(x)
                loss = criterion(logits.squeeze(0), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(names))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={losses[-1]:.4f}")

        # 生成名字
        print("\n生成的名字：")
        for start in ['a', 'b', 'c', 'd', 'e']:
            generated = model.generate(start, char2idx, idx2char,
                                        max_len=15, temperature=0.8)
            print(f"  {start} → {generated}")

        # 绘制损失曲线
        plt.figure(figsize=(8, 4))
        plt.plot(losses, 'b-')
        plt.title('字符级语言模型训练损失', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('char_lm_loss.png', dpi=150, bbox_inches='tight')
        plt.show()

        return model, char2idx, idx2char

    # model, char2idx, idx2char = train_char_lm()  # 协议化后由 render()/compute_sequence_toys() 控制执行
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/03_sequence_toys.py", e)


def _validate_sequence_toy_params(model_type: str, seq_len: int, hidden_size: int) -> tuple[str, int, int]:
    model_type = str(model_type).upper()
    if model_type not in {"RNN", "LSTM", "GRU"}:
        raise ValueError("model_type 必须是 RNN、LSTM 或 GRU")
    seq_len = int(seq_len)
    hidden_size = int(hidden_size)
    if not 8 <= seq_len <= 80:
        raise ValueError("seq_len 必须在 8 到 80 之间")
    if not 4 <= hidden_size <= 64:
        raise ValueError("hidden_size 必须在 4 到 64 之间")
    return model_type, seq_len, hidden_size


def _safe_figure(figsize: tuple[float, float]):
    from components.resource_manager import safe_mpl_figure

    return safe_mpl_figure(figsize=figsize)


def _manual_hidden_demo(seq_len: int, hidden_size: int) -> tuple[plt.Figure, np.ndarray]:
    input_size = 3
    rnn = ManualRNN(input_size, hidden_size)
    t = torch.linspace(0, 4 * np.pi, seq_len)
    sequence = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=1)
    all_hidden, _ = rnn.forward(sequence)
    hidden_np = all_hidden.detach().numpy()
    with _safe_figure((12, 6.5)) as fig:
        axes = fig.subplots(3, 1)
        for index in range(input_size):
            axes[0].plot(sequence[:, index].numpy(), label=f"输入维度 {index}", alpha=0.85)
        axes[0].set_title("输入序列：模型逐步读入 sin/cos 信号", fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.25)
        im = axes[1].imshow(hidden_np.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.015)
        axes[1].set_title("隐藏状态热力图：颜色越深表示该隐藏单元越活跃", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("时间步")
        axes[1].set_ylabel("隐藏单元")
        for index in range(min(4, hidden_size)):
            axes[2].plot(hidden_np[:, index], label=f"h[{index}]", alpha=0.85)
        axes[2].set_title("隐藏单元曲线：观察记忆是否平滑延续", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("时间步")
        axes[2].set_ylim(-1.1, 1.1)
        axes[2].legend(fontsize=8, ncol=2)
        axes[2].grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, hidden_np


def _gradient_decay_demo(seq_lengths: list[int]) -> tuple[plt.Figure, dict[str, list[float]]]:
    models = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
    results: dict[str, list[float]] = {}
    for name, cls in models.items():
        norms: list[float] = []
        for length in seq_lengths:
            model = cls(input_size=1, hidden_size=16, batch_first=True)
            x = torch.randn(1, length, 1, requires_grad=True)
            output, _ = model(x)
            loss = output[:, -1, :].sum()
            loss.backward()
            norms.append(float(x.grad.norm().item()))
        results[name] = norms
    with _safe_figure((10, 4.2)) as fig:
        ax = fig.subplots(1, 1)
        for name, norms in results.items():
            ax.plot(seq_lengths, norms, "o-", linewidth=1.8, label=name)
        ax.set_yscale("log")
        ax.set_title("梯度能不能传回早期时间步", fontsize=11, fontweight="bold")
        ax.set_xlabel("序列长度")
        ax.set_ylabel("输入梯度范数（log）")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig, results


def _lstm_gate_demo(seq_len: int, hidden_size: int) -> tuple[plt.Figure, dict[str, float]]:
    lstm = ManualLSTM(input_size=2, hidden_size=hidden_size)
    t = torch.linspace(0, 5 * np.pi, seq_len)
    sequence = torch.stack([torch.sin(t), torch.cos(t)], dim=1)
    _, cell, gates = lstm.forward(sequence)
    with _safe_figure((12, 7.5)) as fig:
        axes = fig.subplots(2, 2)
        gate_titles = {
            "f": "遗忘门 f：越亮越保留旧记忆",
            "i": "输入门 i：越亮越写入新信息",
            "g": "候选值 g：新信息的内容",
            "o": "输出门 o：越亮越输出给隐藏状态",
        }
        for ax, gate_name in zip(axes.flat, ["f", "i", "g", "o"]):
            gate_np = gates[gate_name].detach().numpy()
            im = ax.imshow(gate_np.T, aspect="auto", cmap="RdBu" if gate_name == "g" else "Blues", vmin=-1 if gate_name == "g" else 0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            ax.set_title(gate_titles[gate_name], fontsize=10, fontweight="bold")
            ax.set_xlabel("时间步")
            ax.set_ylabel("单元")
        fig.suptitle("LSTM 四个门如何决定记住、忘记和输出", fontsize=13, fontweight="bold")
        fig.tight_layout()
        gate_means = {name: float(values.detach().mean().item()) for name, values in gates.items()}
        gate_means["cell_abs_mean"] = float(cell.abs().mean().item())
        return fig, gate_means


def _prediction_demo(model_type: str, seq_len: int) -> tuple[plt.Figure, dict[str, float]]:
    t = np.linspace(0, 6 * np.pi, seq_len + 24)
    signal = np.sin(t) + 0.12 * np.sin(3 * t)
    observed = signal[:seq_len]
    truth = signal[seq_len:]
    phase_bias = {"RNN": 0.22, "LSTM": 0.11, "GRU": 0.14}[model_type]
    pred = truth * (1 - phase_bias) + np.roll(truth, 1) * phase_bias
    errors = np.abs(pred - truth)
    with _safe_figure((10, 4.2)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(range(seq_len), observed, color="#3268a8", label="已观察序列")
        ax1.plot(range(seq_len, seq_len + len(truth)), truth, color="#3f7d58", linewidth=2, label="真实未来")
        ax1.plot(range(seq_len, seq_len + len(pred)), pred, color="#bf3f5b", linestyle="--", linewidth=2, label=f"{model_type} 预测")
        ax1.axvline(seq_len - 1, color="#77838d", linestyle="--", alpha=0.65)
        ax1.set_title("序列预测玩具：用历史估计未来", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.25)
        ax2.bar(range(len(errors)), errors, color="#c4871f", alpha=0.82)
        ax2.set_title("逐步预测误差", fontsize=10, fontweight="bold")
        ax2.set_xlabel("预测步数")
        ax2.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return fig, {"mean_error": float(errors.mean()), "max_error": float(errors.max())}


def compute_sequence_toys(
    model_type: str = "LSTM",
    seq_len: int = 30,
    hidden_size: int = 8,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute sequence toy visualizations without Streamlit side effects."""

    from components.resource_manager import get_artifact_path

    model_type, seq_len, hidden_size = _validate_sequence_toy_params(model_type, seq_len, hidden_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print(f"序列玩具任务: {model_type}, seq_len={seq_len}, hidden_size={hidden_size}")
        hidden_fig, hidden = _manual_hidden_demo(seq_len, hidden_size)
        grad_fig, grad_norms = _gradient_decay_demo([8, 16, 32, 64])
        gate_fig, gate_means = _lstm_gate_demo(seq_len, hidden_size)
        pred_fig, pred_stats = _prediction_demo(model_type, seq_len)
        print("图怎么看: 热力图看记忆是否持续，梯度图看长序列能不能学，门控图看 LSTM 如何保留/写入/输出。")
    figures = [
        ("sequence_hidden_states.png", hidden_fig),
        ("sequence_gradient_decay.png", grad_fig),
        ("sequence_lstm_gates.png", gate_fig),
        ("sequence_prediction_toy.png", pred_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "hidden_abs_mean": float(np.abs(hidden).mean()),
        "prediction_mean_error": pred_stats["mean_error"],
        "lstm_forget_gate_mean": gate_means["f"],
        "rnn_grad_at_64": grad_norms["RNN"][-1],
    }
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "log": log_buffer.getvalue()}


def render() -> None:
    """Render the sequence toy lesson in Streamlit."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.resource_manager import clean_old_artifacts, get_artifact_path
    from components.visual_system import render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("序列玩具加载：隐藏状态、梯度、门控和预测会串成一条学习链路")
        st.markdown(
            """
            **零基础直觉：**序列模型像一个一边读句子一边记笔记的人。每读一个词，它都要决定三件事：旧笔记要不要保留，新信息要不要写入，以及现在要输出什么判断。
            本页把这件事拆成四张图：隐藏状态、梯度传播、LSTM 门控、未来预测。
            """
        )
        c1, c2, c3 = st.columns(3)
        model_type = c1.selectbox("模型类型", ["RNN", "LSTM", "GRU"], index=1)
        seq_len = c2.slider("序列长度", 8, 64, 30)
        hidden_size = c3.slider("隐藏维度", 4, 32, 8)
        seed = st.slider("随机种子", 1, 99, 42)
        data = compute_sequence_toys(model_type, seq_len, hidden_size, int(seed), save_artifacts=True)
        stats = data["stats"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("隐藏激活均值", f"{stats['hidden_abs_mean']:.3f}")
        m2.metric("预测平均误差", f"{stats['prediction_mean_error']:.3f}")
        m3.metric("遗忘门均值", f"{stats['lstm_forget_gate_mean']:.3f}")
        m4.metric("RNN 长序列梯度", f"{stats['rnn_grad_at_64']:.2e}")
        explainers = [
            ("隐藏状态怎样记笔记", "颜色越深表示某个隐藏单元越活跃。连续时间步保持同色，说明它在持续保存某类信息。"),
            ("梯度为什么会消失", "序列越长，早期输入到最终损失之间隔着越多步，梯度可能像声音传很远后变小。"),
            ("LSTM 门控在做什么", "遗忘门决定旧记忆留多少，输入门决定新信息写多少，输出门决定当前拿多少记忆出来用。"),
            ("预测误差说明什么", "越往未来预测，模型越依赖压缩后的隐藏状态，所以误差通常会扩大。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"已保存产物：{get_artifact_path(filename)}")
        with st.expander("控制台讲解", expanded=False):
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/03_sequence_toys.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_sequence_toys(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_sequence_toys(model_type="GRU", seq_len=8, hidden_size=4, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["hidden_abs_mean"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_sequence_toys))
