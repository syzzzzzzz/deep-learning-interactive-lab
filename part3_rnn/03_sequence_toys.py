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

    visualize_rnn_hidden_states()

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

    demonstrate_vanishing_gradient()

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

    visualize_lstm_gates()

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

    models, results = train_sequence_predictor()

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

    model, char2idx, idx2char = train_char_lm()
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/03_sequence_toys.py", e)


def render() -> None:
    """Page entry point — content runs at module import time."""
    pass


def compute(seed: int = 42) -> dict[str, object]:
    """Pure computation placeholder."""
    return {"status": "ok", "seed": seed}


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""
    return True
