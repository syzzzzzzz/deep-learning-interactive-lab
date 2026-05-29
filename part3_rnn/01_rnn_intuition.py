MODULE_TITLE = "RNN 直觉"
MODULE_SUMMARY = "从循环隐藏状态、时间展开和梯度衰减理解序列模型的基本机制。"
MODULE_TAGS = ["RNN", "序列", "隐藏状态", "梯度"]
MODULE_RELATED_TOPICS = ["隐藏状态", "RNN 超参实验", "LSTM/GRU", "梯度监控"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground?example=mlp"

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


def simulate_rnn_sequence(
    sequence_length: int = 8,
    input_size: int = 3,
    hidden_size: int = 5,
    recurrent_scale: float = 0.85,
    input_strength: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run a tiny hand-written RNN and return all intermediate states."""

    sequence_length = clamp_int(sequence_length, 3, 30, "序列长度")
    input_size = clamp_int(input_size, 1, 8, "输入维度")
    hidden_size = clamp_int(hidden_size, 2, 16, "隐藏单元数")
    recurrent_scale = clamp_float(recurrent_scale, 0.05, 1.6, "循环权重尺度")
    input_strength = clamp_float(input_strength, 0.05, 3.0, "输入强度")
    torch.manual_seed(seed)

    x = torch.randn(sequence_length, input_size) * input_strength
    w_xh = torch.randn(hidden_size, input_size) / np.sqrt(input_size)
    w_hh_raw = torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
    spectral = torch.linalg.matrix_norm(w_hh_raw, ord=2)
    w_hh = w_hh_raw / spectral * recurrent_scale
    b_h = torch.zeros(hidden_size)

    hidden_states = []
    pre_activations = []
    h = torch.zeros(hidden_size)
    for t in range(sequence_length):
        pre = w_xh @ x[t] + w_hh @ h + b_h
        h = torch.tanh(pre)
        pre_activations.append(pre.numpy())
        hidden_states.append(h.numpy())

    return {
        "inputs": x.numpy(),
        "hidden_states": np.stack(hidden_states),
        "pre_activations": np.stack(pre_activations),
        "w_hh": w_hh.numpy(),
    }


def gradient_decay_curve(sequence_length: int, recurrent_scale: float, saturation: float) -> np.ndarray:
    """Approximate how gradients change as they travel backwards in time."""

    sequence_length = clamp_int(sequence_length, 3, 80, "反向传播时间距离")
    recurrent_scale = clamp_float(recurrent_scale, 0.05, 1.8, "循环 Jacobian 尺度")
    saturation = clamp_float(saturation, 0.0, 0.95, "tanh 饱和程度")
    effective_gain = recurrent_scale * (1.0 - saturation)
    steps = np.arange(sequence_length)
    return effective_gain ** steps


def _plot_hidden_heatmap(hidden_states: np.ndarray) -> plt.Figure:
    with safe_mpl_figure(figsize=(8, 4.5)) as fig:
        ax = fig.subplots()
        im = ax.imshow(hidden_states.T, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("隐藏状态随时间变化", fontsize=13, fontweight="bold")
        ax.set_xlabel("时间步 t")
        ax.set_ylabel("隐藏单元")
        fig.tight_layout()
        return fig


def _plot_gradient_curve(curve: np.ndarray, recurrent_scale: float) -> plt.Figure:
    with safe_mpl_figure(figsize=(7.5, 4.2)) as fig:
        ax = fig.subplots()
        ax.plot(curve, marker="o", linewidth=2, color="#3268a8")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_title(f"反向传播中的梯度倍率（循环尺度={recurrent_scale:.2f}）", fontsize=12, fontweight="bold")
        ax.set_xlabel("向前追溯的时间距离")
        ax.set_ylabel("梯度相对倍率（log）")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def _plot_gate_intuition() -> plt.Figure:
    with safe_mpl_figure(figsize=(8.5, 4.5)) as fig:
        ax = fig.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        boxes = [
            (0.7, 2.8, "输入 x_t", "#d8ecff"),
            (0.7, 1.0, "旧状态 h_{t-1}", "#f7e0b5"),
            (3.6, 1.9, "tanh\n候选记忆", "#d7f2dc"),
            (6.5, 1.9, "新状态 h_t", "#f2d7e6"),
        ]
        for x, y, label, color in boxes:
            rect = plt.Rectangle((x, y), 1.9, 0.9, facecolor=color, edgecolor="#333", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(x + 0.95, y + 0.45, label, ha="center", va="center", fontsize=11, fontweight="bold")
        arrow_props = dict(arrowstyle="->", linewidth=2, color="#555")
        ax.annotate("", xy=(3.6, 2.35), xytext=(2.6, 3.25), arrowprops=arrow_props)
        ax.annotate("", xy=(3.6, 2.1), xytext=(2.6, 1.45), arrowprops=arrow_props)
        ax.annotate("", xy=(6.5, 2.35), xytext=(5.5, 2.35), arrowprops=arrow_props)
        ax.text(5.0, 3.6, "普通 RNN 只有一个状态通道\n长距离信息容易被反复乘法冲淡", ha="center", fontsize=11)
        ax.set_title("RNN 单步更新直觉", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


def compute_rnn_intuition(
    sequence_length: int = 8,
    hidden_size: int = 5,
    recurrent_scale: float = 0.85,
    input_strength: float = 1.0,
    saturation: float = 0.15,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute the RNN lesson without Streamlit calls."""

    artifacts: list[Path] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        result = simulate_rnn_sequence(
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            recurrent_scale=recurrent_scale,
            input_strength=input_strength,
            seed=seed,
        )
        curve = gradient_decay_curve(sequence_length * 4, recurrent_scale, saturation)
        print("RNN 计算公式：h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)")
        print(f"序列长度={sequence_length}, 隐藏单元={hidden_size}, 循环尺度={recurrent_scale:.2f}")
        print(f"隐藏状态 shape={result['hidden_states'].shape}")
        print(f"最后一步隐藏状态={np.round(result['hidden_states'][-1], 3)}")
        print(f"梯度末端倍率={curve[-1]:.6f}")
        if curve[-1] < 1e-3:
            print("诊断：梯度明显消失。工程上可考虑 LSTM/GRU、残差连接、归一化或截断 BPTT。")
        elif curve[-1] > 10:
            print("诊断：梯度明显爆炸。工程上应降低学习率或使用梯度裁剪。")
        else:
            print("诊断：梯度倍率处在可观察范围，适合做教学演示。")

    heatmap_fig = _plot_hidden_heatmap(result["hidden_states"])
    gradient_fig = _plot_gradient_curve(curve, recurrent_scale)
    gate_fig = _plot_gate_intuition()
    figures = [
        ("rnn_hidden_heatmap_refactored.png", heatmap_fig),
        ("rnn_gradient_curve_refactored.png", gradient_fig),
        ("rnn_gate_intuition.png", gate_fig),
    ]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {"log": log_buffer.getvalue(), "figures": figures, "artifacts": artifacts, "curve": curve}


def _go_to_sequence_models() -> None:
    import streamlit as st

    st.query_params["module"] = "part3_rnn/sequence_models"
    st.rerun()


def render() -> None:
    """Render the refactored RNN intuition lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        st.info("RNN 的核心不是“会循环”四个字，而是同一个状态向量在每个时间步反复更新：旧状态决定新状态，新状态继续影响未来。")

        with st.sidebar:
            sequence_length = st.slider("序列长度", 3, 30, 8)
            hidden_size = st.slider("隐藏单元数", 2, 16, 5)
            recurrent_scale = st.slider("循环权重尺度", 0.05, 1.60, 0.85, 0.05)
            input_strength = st.slider("输入强度", 0.05, 3.0, 1.0, 0.05)
            saturation = st.slider("tanh 饱和程度", 0.0, 0.95, 0.15, 0.05)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("继续看：序列模型总览", width="stretch"):
                _go_to_sequence_models()

        data = compute_rnn_intuition(sequence_length, hidden_size, recurrent_scale, input_strength, saturation, int(seed), save_artifacts=True)
        st.subheader("直觉闭环")
        st.markdown(
            """
            - **隐藏状态热力图**：横轴是时间，纵轴是隐藏单元，颜色表示该单元在当前时间步的激活强弱。
            - **梯度倍率曲线**：展示反向传播越过很多时间步时，梯度会被反复乘大或乘小。
            - **单步更新图**：说明普通 RNN 为什么容易忘掉很久以前的信息。
            """
        )
        cols = st.columns(3)
        for col, title, (_, fig) in zip(cols, ("隐藏状态", "梯度传播", "单步更新"), data["figures"]):
            with col:
                st.subheader(title)
                st.pyplot(fig, clear_figure=False)

        with st.expander("控制台输出与工程解释", expanded=False):
            st.code(str(data["log"]), language="text")
            st.markdown("工程经验：普通 RNN 适合短序列教学和基线实验；真实长序列任务通常先考虑 LSTM、GRU、Transformer 或状态空间模型。")
    except Exception as exc:
        render_module_error("part3_rnn/01_rnn_intuition.py", exc)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_rnn_intuition(sequence_length=5, hidden_size=3, save_artifacts=False)
    return bool(data["figures"]) and len(data["curve"]) > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_rnn_intuition))
