"""RNN 调试问题：把常见训练故障变成可检查、可解释、可复用的诊断面板。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "RNN 调试问题"
MODULE_SUMMARY = "用梯度流、隐藏状态、padding mask 和采样控制解释 RNN 训练中最常见的故障与修复方法。"
MODULE_TAGS = ["RNN", "调试", "梯度", "Padding", "采样"]
MODULE_RELATED_TOPICS = ["part3/07_advanced_training", "part5/02_gradient_monitor", "part5/03_training_dynamics", "part3/06_text_classification"]
PRACTICE_TARGET = "切换故障类型并调整裁剪阈值、温度和 top-k/top-p，判断训练问题来自梯度、隐藏状态、mask 还是采样策略。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    RNN debugging snippets collected as safe, runnable helpers.

    The original lesson contains intentionally broken examples. This module keeps
    the corrected patterns executable without running undefined pseudo-code at the
    top level.
    """

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

    from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


    def clip_training_step(model, optimizer, loss, max_norm=1.0):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()


    def init_lstm_hidden(model, batch_size, device=None):
        device = device or next(model.parameters()).device
        directions = 2 if getattr(model, "bidirectional", False) else 1
        num_layers = getattr(model, "num_layers", 1)
        hidden_size = getattr(model, "hidden_size", 1)
        h = torch.zeros(num_layers * directions, batch_size, hidden_size, device=device)
        c = torch.zeros(num_layers * directions, batch_size, hidden_size, device=device)
        return h, c


    def pad_and_pack(sequences, embedding, lstm):
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(s) for s in sequences], device=padded.device)
        embedded = embedding(padded)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden


    def masked_cross_entropy(logits, targets, pad_index=0):
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        mask = targets.reshape(-1) != pad_index
        denom = mask.sum().clamp_min(1)
        return (flat_loss * mask).sum() / denom


    def get_teacher_forcing_ratio(epoch, k=10):
        return k / (k + np.exp(epoch / k))


    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (h_n, _) = self.lstm(embedded)
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            return self.fc(h)


    class ProperDropoutLSTM(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            output, _ = self.lstm(self.embedding(x))
            return self.fc(self.dropout(output[:, -1, :]))


    class SingleLayerLSTMWithDropout(nn.Module):
        def __init__(self, embed_size, hidden_size, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, h=None):
            return self.lstm(self.dropout(x), h)


    def generate_with_control(model, start_token, max_len=50, temperature=1.0, top_k=0, top_p=0.0):
        model.eval()
        x = torch.tensor([[start_token]], device=next(model.parameters()).device)
        h = None
        generated = []

        with torch.no_grad():
            for _ in range(max_len):
                logits, h = model(x, h)
                logits = logits[0, -1, :] / max(temperature, 1e-8)

                if top_k > 0:
                    top_vals, _ = logits.topk(min(top_k, logits.numel()))
                    logits[logits < top_vals[-1]] = -float("inf")

                if top_p > 0:
                    sorted_logits, sorted_idx = logits.sort(descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[1:] = remove_mask[:-1].clone()
                    remove_mask[0] = False
                    logits[sorted_idx[remove_mask]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                x = next_token.view(1, 1)

        return generated


    def check_rnn_gradient_flow(model):
        rows = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grad_norm = p.grad.norm().item()
            grad_mean = p.grad.abs().mean().item()
            grad_max = p.grad.abs().max().item()
            rows.append((name, grad_norm, grad_mean, grad_max))
        return rows


    def check_hidden_state(hidden):
        h, c = hidden
        return {
            "h_norm": h.norm().item(),
            "c_norm": c.norm().item(),
            "h_has_nan": torch.isnan(h).any().item(),
            "c_has_inf": torch.isinf(c).any().item(),
        }


    if __name__ == "__main__":
        model = BiLSTMClassifier(vocab_size=12, embed_size=4, hidden_size=6, num_classes=3)
        x = torch.randint(0, 12, (2, 5))
        y = model(x)
        assert y.shape == (2, 3)
        print("RNN debug helpers smoke test passed.")
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/08_debug_problems.py", e)


def _simulate_debug_signals(problem_type: str, clip_norm: float, temperature: float, top_k: int, top_p: float, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    layers = ["embedding", "lstm.weight_ih", "lstm.weight_hh", "fc.weight"]
    base = rng.lognormal(mean=-0.4, sigma=0.45, size=len(layers))
    if problem_type == "梯度爆炸":
        base *= np.array([1.2, 4.5, 9.0, 2.0])
    elif problem_type == "梯度消失":
        base *= np.array([0.08, 0.035, 0.018, 0.12])
    elif problem_type == "Padding 未 mask":
        base *= np.array([1.6, 1.1, 1.0, 1.7])
    else:
        base *= np.array([0.9, 1.0, 1.0, 1.1])
    clipped = np.minimum(base, clip_norm)
    time = np.arange(1, 41)
    hidden_norm = 1.0 + 0.08 * np.sin(time / 4)
    if problem_type == "梯度爆炸":
        hidden_norm = np.exp(time / 15) / 3
    elif problem_type == "梯度消失":
        hidden_norm = np.exp(-time / 12)
    elif problem_type == "采样退化":
        hidden_norm = 0.85 + 0.03 * np.sin(time / 2)
    vocab_logits = rng.normal(0, 1, size=24) / max(temperature, 1e-6)
    probs = np.exp(vocab_logits - vocab_logits.max())
    probs = probs / probs.sum()
    if top_k > 0:
        keep = np.argsort(probs)[-top_k:]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()
    if top_p > 0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep_order = order[cumulative <= top_p]
        if len(keep_order) == 0:
            keep_order = order[:1]
        masked = np.zeros_like(probs)
        masked[keep_order] = probs[keep_order]
        probs = masked / masked.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {"layers": layers, "raw_grad": base, "clipped_grad": clipped, "hidden_norm": hidden_norm, "sampling_probs": probs, "entropy": entropy}


def _plot_gradient_debug(signals: dict[str, object], clip_norm: float) -> object:
    layers = signals["layers"]
    x = np.arange(len(layers))
    with safe_mpl_figure(figsize=(9.8, 4.3)) as fig:
        ax = fig.subplots(1, 1)
        ax.bar(x - 0.18, signals["raw_grad"], width=0.36, label="原始梯度", color="#bf3f5b", alpha=0.75)
        ax.bar(x + 0.18, signals["clipped_grad"], width=0.36, label="裁剪后", color="#00ff88", alpha=0.82)
        ax.axhline(clip_norm, color="#00f0ff", linestyle="--", linewidth=2, label="裁剪阈值")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=15, ha="right")
        ax.set_ylabel("梯度范数")
        ax.set_title("梯度流诊断：看哪一层过大、过小或被裁剪", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig


def _plot_hidden_debug(signals: dict[str, object]) -> object:
    with safe_mpl_figure(figsize=(8.8, 4.1)) as fig:
        ax = fig.subplots(1, 1)
        ax.plot(np.arange(1, len(signals["hidden_norm"]) + 1), signals["hidden_norm"], color="#b000ff", linewidth=2.4)
        ax.axhline(1.0, color="#777", linestyle=":", alpha=0.7)
        ax.set_xlabel("时间步")
        ax.set_ylabel("隐藏状态范数")
        ax.set_title("隐藏状态诊断：过大可能爆炸，过小可能遗忘", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig


def _plot_sampling_debug(signals: dict[str, object]) -> object:
    probs = signals["sampling_probs"]
    top = np.argsort(probs)[::-1][:12]
    with safe_mpl_figure(figsize=(8.8, 4.1)) as fig:
        ax = fig.subplots(1, 1)
        ax.bar(range(len(top)), probs[top], color="#00f0ff", alpha=0.82)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([f"tok{idx}" for idx in top], rotation=30, ha="right")
        ax.set_ylabel("采样概率")
        ax.set_title("生成采样诊断：概率太尖会重复，太平会发散", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return fig


def compute_debug_problems(
    problem_type: str = "梯度爆炸",
    clip_norm: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 6,
    top_p: float = 0.0,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute RNN debugging diagnostics from public helper concepts."""

    if problem_type not in {"梯度爆炸", "梯度消失", "Padding 未 mask", "采样退化"}:
        raise ValueError("problem_type 必须是 梯度爆炸、梯度消失、Padding 未 mask 或 采样退化")
    clip_norm = clamp_float(clip_norm, 0.1, 8.0, "裁剪阈值")
    temperature = clamp_float(temperature, 0.2, 2.5, "采样温度")
    top_k = clamp_int(top_k, 0, 24, "top-k")
    top_p = clamp_float(top_p, 0.0, 0.98, "top-p")
    signals = _simulate_debug_signals(problem_type, clip_norm, temperature, top_k, top_p, seed)
    grad_fig = _plot_gradient_debug(signals, clip_norm)
    hidden_fig = _plot_hidden_debug(signals)
    sampling_fig = _plot_sampling_debug(signals)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("RNN 调试协议化计算")
        print(f"故障类型: {problem_type}, clip_norm={clip_norm:.2f}, temperature={temperature:.2f}, top_k={top_k}, top_p={top_p:.2f}")
        print(f"最大原始梯度={float(np.max(signals['raw_grad'])):.3f}, 最大裁剪后梯度={float(np.max(signals['clipped_grad'])):.3f}")
        print(f"隐藏状态末端范数={float(signals['hidden_norm'][-1]):.3f}, 采样熵={signals['entropy']:.3f}")
        if problem_type == "Padding 未 mask":
            print("诊断：PAD 位置参与 loss 会污染梯度，应使用 mask 或 ignore_index。")
        elif problem_type == "采样退化":
            print("诊断：温度/top-k/top-p 会改变输出分布，重复或胡言乱语都先查采样策略。")
        else:
            print("诊断：先看梯度范数，再看 hidden state 范数，最后检查学习率和裁剪阈值。")
    figures = [
        ("rnn_debug_gradient_flow.png", grad_fig),
        ("rnn_debug_hidden_state.png", hidden_fig),
        ("rnn_debug_sampling.png", sampling_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "max_raw_grad": float(np.max(signals["raw_grad"])),
        "max_clipped_grad": float(np.max(signals["clipped_grad"])),
        "hidden_final_norm": float(signals["hidden_norm"][-1]),
        "sampling_entropy": float(signals["entropy"]),
    }
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "signals": signals, "log": log_buffer.getvalue()}


def _go_to_gradient_monitor() -> None:
    import streamlit as st

    st.query_params["module"] = "part5_toolbox/02_gradient_monitor"
    st.rerun()


def render() -> None:
    """Render the RNN debugging lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_backprop_current_flow, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        render_visual_system("light")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成调试诊断：梯度、隐藏状态、采样分布同步刷新")
        with st.sidebar:
            problem_type = st.selectbox("故障类型", ["梯度爆炸", "梯度消失", "Padding 未 mask", "采样退化"])
            clip_norm = st.slider("裁剪阈值", 0.1, 8.0, 1.0, 0.1)
            temperature = st.slider("采样温度", 0.2, 2.5, 1.0, 0.05)
            top_k = st.slider("top-k", 0, 24, 6, 1)
            top_p = st.slider("top-p", 0.0, 0.98, 0.0, 0.02)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：梯度监控", width="stretch"):
                _go_to_gradient_monitor()
        data = compute_debug_problems(problem_type, clip_norm, temperature, top_k, top_p, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_backprop_current_flow()
        st.markdown(
            """
            **零基础直觉：**调试 RNN 像给一台会写句子的机器体检。梯度是电流，隐藏状态是记忆，mask 是告诉它哪些位置是假字，
            采样参数是控制它说话时“保守还是发散”。哪里异常，就先盯哪张图。
            """
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最大原始梯度", f"{stats['max_raw_grad']:.2f}")
        c2.metric("裁剪后最大梯度", f"{stats['max_clipped_grad']:.2f}")
        c3.metric("隐藏末端范数", f"{stats['hidden_final_norm']:.2f}")
        c4.metric("采样熵", f"{stats['sampling_entropy']:.2f}")
        explainers = [
            ("梯度流诊断", "梯度太大会让参数一次跳坏，太小会几乎学不动。裁剪阈值线帮助你看哪些层被压住。"),
            ("隐藏状态诊断", "隐藏状态范数持续膨胀是爆炸信号，快速贴近 0 是遗忘信号，稳定波动通常更健康。"),
            ("采样诊断", "温度低或 top-k 太小会让输出重复；温度高或限制太少会让输出变得散乱。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请切换一个故障类型，再观察三张图哪张先发出警报。思考：这个问题应该先改数据、模型、优化器，还是采样策略？")
        with st.expander("排查步骤与控制台输出", expanded=False):
            st.markdown(
                """
                1. **先看 loss 是否 NaN/不降**：若 NaN，优先查学习率、梯度爆炸、非法输入。
                2. **再看梯度范数**：若几乎为 0，查激活饱和、序列太长、detach 是否误用。
                3. **再看 mask**：文本任务里 PAD 没屏蔽，会让模型认真学习“空白”。
                4. **最后看采样**：训练正常但生成怪，往往是 temperature、top-k、top-p 的问题。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part3_rnn/08_debug_problems.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_debug_problems(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_debug_problems(problem_type="梯度爆炸", clip_norm=1.0, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["max_clipped_grad"] <= 1.0 and data["stats"]["sampling_entropy"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_debug_problems))
