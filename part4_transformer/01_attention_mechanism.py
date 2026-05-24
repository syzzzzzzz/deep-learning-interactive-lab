MODULE_TITLE = "注意力机制"
MODULE_SUMMARY = "从查询、键、值和权重矩阵理解注意力的信息检索过程。"
MODULE_TAGS = ["Transformer", "注意力", "NLP", "可视化"]

import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"


def scaled_dot_product_attention_verbose(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, T_q, d_k = Q.shape
    _, _, T_k, _ = K.shape
    print(f"Q/K/V: {tuple(Q.shape)} / {tuple(K.shape)} / {tuple(V.shape)}")
    print(f"batch={B}, heads={H}, query_len={T_q}, key_len={T_k}, d_k={d_k}")
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    output = torch.matmul(attn_weights, V)
    print(f"scores={tuple(scores.shape)}, attn={tuple(attn_weights.shape)}, output={tuple(output.shape)}")
    return output, attn_weights


def demonstrate_tensor_shapes() -> torch.Tensor:
    torch.manual_seed(42)
    B, T, d_model = 2, 10, 512
    d_k = d_v = 64
    H = 8
    print("=" * 60)
    print("自注意力张量形状逐步演示")
    print(f"batch_size={B}, seq_len={T}, d_model={d_model}, d_k={d_k}, H={H}")
    X = torch.randn(B, T, d_model)
    W_Q = torch.randn(d_model, d_k)
    W_K = torch.randn(d_model, d_k)
    W_V = torch.randn(d_model, d_v)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    print(f"X={tuple(X.shape)}, Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}")
    scores = Q @ K.transpose(-2, -1)
    scores_scaled = scores / (d_k ** 0.5)
    attn = F.softmax(scores_scaled, dim=-1)
    output = attn @ V
    print(f"scores={tuple(scores.shape)}, attn={tuple(attn.shape)}, output={tuple(output.shape)}")
    print(f"缩放前方差={scores.var().item():.2f}，缩放后方差={scores_scaled.var().item():.2f}")
    Q_heads = (X @ torch.randn(d_model, d_model)).view(B, T, H, d_k).transpose(1, 2)
    K_heads = (X @ torch.randn(d_model, d_model)).view(B, T, H, d_k).transpose(1, 2)
    V_heads = (X @ torch.randn(d_model, d_model)).view(B, T, H, d_v).transpose(1, 2)
    out_heads, attn_heads = scaled_dot_product_attention_verbose(Q_heads, K_heads, V_heads)
    out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, d_model)
    final_out = out_concat @ torch.randn(d_model, d_model)
    print(f"多头输出={tuple(out_heads.shape)}，合并后={tuple(out_concat.shape)}，最终={tuple(final_out.shape)}")
    return attn_heads


def plot_attention_scaling() -> tuple[plt.Figure, dict[str, float]]:
    torch.manual_seed(0)
    T, d_k = 10, 64
    Q = torch.randn(T, d_k)
    K = torch.randn(T, d_k)
    scores_raw = Q @ K.T
    scores_scaled = scores_raw / (d_k ** 0.5)
    attn_raw = F.softmax(scores_raw, dim=-1)
    attn_scaled = F.softmax(scores_scaled, dim=-1)
    entropy_raw = -(attn_raw * (attn_raw + 1e-9).log()).sum(dim=-1)
    entropy_scaled = -(attn_scaled * (attn_scaled + 1e-9).log()).sum(dim=-1)

    with safe_mpl_figure(figsize=(15, 8)) as fig:
        axes = fig.subplots(2, 3)
        axes[0, 0].hist(scores_raw.flatten().numpy(), bins=40, color="#C44E52", alpha=0.8, edgecolor="white")
        axes[0, 0].set_title(f"原始分数分布\n均值={scores_raw.mean():.2f}, 方差={scores_raw.var():.2f}")
        axes[0, 1].hist(scores_scaled.flatten().numpy(), bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
        axes[0, 1].set_title(f"缩放后分数分布\n均值={scores_scaled.mean():.2f}, 方差={scores_scaled.var():.2f}")
        axes[0, 2].hist(attn_raw.flatten().numpy(), bins=40, color="#C44E52", alpha=0.8, edgecolor="white", label="未缩放")
        axes[0, 2].hist(attn_scaled.flatten().numpy(), bins=40, color="#4C72B0", alpha=0.5, edgecolor="white", label="缩放后")
        axes[0, 2].set_title("注意力权重分布对比")
        axes[0, 2].legend()
        im1 = axes[1, 0].imshow(attn_raw.numpy(), cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im1, ax=axes[1, 0])
        axes[1, 0].set_title("未缩放注意力权重")
        im2 = axes[1, 1].imshow(attn_scaled.numpy(), cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im2, ax=axes[1, 1])
        axes[1, 1].set_title("缩放后注意力权重")
        x = range(T)
        axes[1, 2].plot(x, entropy_raw.numpy(), "r-o", label=f"未缩放 {entropy_raw.mean():.2f}")
        axes[1, 2].plot(x, entropy_scaled.numpy(), "b-o", label=f"缩放后 {entropy_scaled.mean():.2f}")
        axes[1, 2].axhline(np.log(T), color="gray", linestyle="--", label=f"最大熵 {np.log(T):.2f}")
        axes[1, 2].set_title("注意力熵")
        axes[1, 2].legend(fontsize=8)
        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"√d_k 缩放的必要性分析（d_k={d_k}）", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig, {
            "entropy_raw": float(entropy_raw.mean()),
            "entropy_scaled": float(entropy_scaled.mean()),
            "max_entropy": float(np.log(T)),
        }


def plot_causal_mask() -> tuple[plt.Figure, np.ndarray]:
    torch.manual_seed(0)
    T = 6
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = torch.randn(1, 1, T, T)
    attn = F.softmax(scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf")), dim=-1)
    with safe_mpl_figure(figsize=(14, 4)) as fig:
        axes = fig.subplots(1, 3)
        im0 = axes[0].imshow(scores[0, 0].numpy(), cmap="RdBu", aspect="auto")
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title("原始分数")
        im1 = axes[1].imshow(mask.float().numpy(), cmap="Reds", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title("因果掩码")
        im2 = axes[2].imshow(attn[0, 0].detach().numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title("掩码后注意力权重")
        for ax in axes:
            ax.set_xlabel("Key 位置")
            ax.set_ylabel("Query 位置")
        fig.suptitle("因果掩码（Causal Mask）工作原理", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig, mask.int().numpy()


def compute_注意力演示(save_artifacts: bool = False) -> dict[str, object]:
    """Compute all attention examples without Streamlit calls."""

    log_buffer = io.StringIO()
    artifacts: list[Path] = []
    with redirect_stdout(log_buffer):
        attn_heads = demonstrate_tensor_shapes()
        scaling_fig, stats = plot_attention_scaling()
        print(f"未缩放注意力熵均值: {stats['entropy_raw']:.4f}")
        print(f"缩放后注意力熵均值: {stats['entropy_scaled']:.4f}")
        print(f"最大可能熵: {stats['max_entropy']:.4f}")
        mask_fig, mask = plot_causal_mask()
        print("因果掩码（1=被遮住）：")
        print(mask)

    figures = [("attention_scaling_analysis.png", scaling_fig), ("causal_mask.png", mask_fig)]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {"log": log_buffer.getvalue(), "figures": figures, "artifacts": artifacts, "attn_shape": tuple(attn_heads.shape)}


def _go_to_playground(example: str) -> None:
    import streamlit as st

    st.query_params["module"] = PLAYGROUND_TARGET
    st.query_params["example"] = example
    st.rerun()


def render_注意力演示() -> None:
    """Render the attention lesson in Streamlit."""

    import streamlit as st
    from components.error_boundary import render_module_error

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        left, right = st.columns([0.7, 0.3])
        with left:
            st.info("Q/K/V 是线性投影后的表示；注意力权重描述当前位置从上下文哪里取信息。")
        with right:
            if st.button("去实战：Transformer 示例", width="stretch"):
                _go_to_playground("transformer")
        data = compute_注意力演示(save_artifacts=True)
        st.subheader("核心张量形状")
        st.code(
            "X: [batch, seq_len, d_model]\n"
            "Q/K/V: [batch, seq_len, d_k]\n"
            "scores = Q @ K.transpose(-2, -1): [batch, seq_len, seq_len]\n"
            "output = softmax(scores / sqrt(d_k)) @ V",
            language="text",
        )
        st.caption(f"多头注意力权重形状：{data['attn_shape']}")
        for title, (filename, fig) in zip(("缩放为什么必要", "因果掩码如何工作"), data["figures"]):
            st.subheader(title)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"已保存产物：{get_artifact_path(filename)}")
        with st.expander("控制台讲解", expanded=False):
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part4_transformer/01_attention_mechanism.py", exc)


render = render_注意力演示


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_注意力演示(save_artifacts=False)
    return bool(data["figures"]) and bool(data["attn_shape"])


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_stdio()
    if _running_under_streamlit():
        render_注意力演示()
    else:
        try:
            result = compute_注意力演示(save_artifacts=True)
            print(result["log"])
            for path in result["artifacts"]:
                print(f"图像已保存: {path}")
        except Exception as e:
            traceback.print_exception(e)
            raise SystemExit(1)
