MODULE_TITLE = "多头注意力可视化"
MODULE_SUMMARY = "观察不同 attention head 如何在不同关系子空间里分工。"
MODULE_TAGS = ["Transformer", "多头注意力", "QKV", "可视化"]
MODULE_RELATED_TOPICS = ["注意力机制", "Transformer 架构", "Flash Attention", "训练动态"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground?example=transformer"

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


DEFAULT_TOKENS = ["我", "正在", "学习", "多头", "注意力", "它", "能", "捕捉", "多种", "关系"]


def _make_pattern(head_index: int, token_count: int) -> torch.Tensor:
    """Create an interpretable synthetic relation pattern for one head."""

    i = torch.arange(token_count).float().view(-1, 1)
    j = torch.arange(token_count).float().view(1, -1)
    if head_index % 4 == 0:
        scores = -torch.abs(i - j)  # local neighborhood
    elif head_index % 4 == 1:
        scores = -torch.abs(j - 0) * 0.6  # anchor to first token
    elif head_index % 4 == 2:
        scores = -torch.abs(j - (token_count - 1 - i)) * 0.8  # mirrored dependency
    else:
        scores = torch.sin((i + 1) * (j + 1) / max(token_count, 1))  # periodic relation
    return scores


def compute_multihead_attention(
    token_text: str = "我 正在 学习 多头 注意力 它 能 捕捉 多种 关系",
    d_model: int = 64,
    n_heads: int = 4,
    sharpness: float = 1.4,
    seed: int = 7,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute synthetic multi-head attention patterns without Streamlit."""

    d_model = clamp_int(d_model, 16, 256, "模型维度")
    n_heads = clamp_int(n_heads, 1, 8, "注意力头数")
    sharpness = clamp_float(sharpness, 0.2, 4.0, "注意力锐度")
    tokens = [token for token in token_text.strip().split() if token] or DEFAULT_TOKENS
    tokens = tokens[:16]
    if d_model % n_heads != 0:
        adjusted = max(1, min(n_heads, d_model))
        while d_model % adjusted != 0 and adjusted > 1:
            adjusted -= 1
        n_heads = adjusted

    torch.manual_seed(seed)
    token_count = len(tokens)
    raw_scores = []
    attentions = []
    entropies = []
    with redirect_stdout(io.StringIO()) as buffer:
        print(f"tokens={tokens}")
        print(f"d_model={d_model}, n_heads={n_heads}, 每头维度 d_k={d_model // n_heads}")
        for head in range(n_heads):
            scores = _make_pattern(head, token_count)
            scores = scores + torch.randn_like(scores) * 0.08
            attn = F.softmax(scores * sharpness, dim=-1)
            entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()
            raw_scores.append(scores.numpy())
            attentions.append(attn.numpy())
            entropies.append(entropy)
            print(f"head {head + 1}: 平均熵={entropy:.3f}, 主要模式={_head_pattern_name(head)}")
        log = buffer.getvalue()

    heatmap_fig = _plot_all_heads(tokens, attentions, sharpness)
    entropy_fig = _plot_entropy(entropies)
    shape_rows = [
        {"张量": "X", "shape": f"[batch, {token_count}, {d_model}]", "含义": "输入 token 表示"},
        {"张量": "Q/K/V", "shape": f"[batch, {n_heads}, {token_count}, {d_model // n_heads}]", "含义": "分头后的查询/键/值"},
        {"张量": "Attention", "shape": f"[batch, {n_heads}, {token_count}, {token_count}]", "含义": "每个 head 的关系矩阵"},
        {"张量": "Concat", "shape": f"[batch, {token_count}, {d_model}]", "含义": "所有 head 合并后的输出"},
    ]
    figures = [("multihead_attention_heads.png", heatmap_fig), ("multihead_entropy.png", entropy_fig)]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {
        "log": log,
        "figures": figures,
        "artifacts": artifacts,
        "tokens": tokens,
        "entropies": entropies,
        "shape_rows": shape_rows,
        "n_heads": n_heads,
    }


def _head_pattern_name(head_index: int) -> str:
    names = ["局部邻近", "锚点聚焦", "镜像依赖", "周期关系"]
    return names[head_index % len(names)]


def _plot_all_heads(tokens: list[str], attentions: list[np.ndarray], sharpness: float) -> plt.Figure:
    n_heads = len(attentions)
    cols = min(4, n_heads)
    rows = int(np.ceil(n_heads / cols))
    with safe_mpl_figure(figsize=(cols * 3.4, rows * 3.2)) as fig:
        axes = fig.subplots(rows, cols, squeeze=False)
        for idx, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if idx >= n_heads:
                continue
            attn = attentions[idx]
            im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=max(0.2, attn.max()))
            ax.set_title(f"Head {idx + 1}: {_head_pattern_name(idx)}", fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"多头注意力权重热力图（锐度={sharpness:.2f}）", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_entropy(entropies: list[float]) -> plt.Figure:
    with safe_mpl_figure(figsize=(6.8, 3.8)) as fig:
        ax = fig.subplots()
        ax.bar([f"H{i + 1}" for i in range(len(entropies))], entropies, color="#0f8b8d")
        ax.set_title("各头注意力熵：越低越集中", fontsize=12, fontweight="bold")
        ax.set_ylabel("平均熵")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig


def _go_to_playground() -> None:
    import streamlit as st

    st.query_params["module"] = "part6_universal_framework/neural_network_playground"
    st.query_params["example"] = "transformer"
    st.rerun()


def render() -> None:
    """Render the refactored multi-head attention lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        st.info("多头注意力不是简单重复算很多遍；不同 head 可以在不同子空间里学习不同关系，例如邻近、指代、句法或全局锚点。")

        with st.sidebar:
            token_text = st.text_input("输入 token（空格分隔）", "我 正在 学习 多头 注意力 它 能 捕捉 多种 关系")
            d_model = st.select_slider("模型维度 d_model", options=[32, 64, 96, 128, 192, 256], value=64)
            n_heads = st.slider("注意力头数", 1, 8, 4)
            sharpness = st.slider("多头注意力锐度", 0.2, 4.0, 1.4, 0.1)
            seed = st.number_input("随机种子", 0, 9999, 7, 1)
            if st.button("去实战：Transformer 构建器", width="stretch"):
                _go_to_playground()

        data = compute_multihead_attention(token_text, d_model, n_heads, sharpness, int(seed), save_artifacts=True)
        if data["n_heads"] != n_heads:
            st.warning(f"`d_model` 必须能被头数整除，页面已把头数调整为 {data['n_heads']}。")

        st.subheader("张量形状")
        st.dataframe(data["shape_rows"], width="stretch")
        left, right = st.columns([0.68, 0.32])
        with left:
            st.subheader("所有 Head 的热力图")
            st.pyplot(data["figures"][0][1], clear_figure=False)
        with right:
            st.subheader("集中程度")
            st.pyplot(data["figures"][1][1], clear_figure=False)
            st.markdown("锐度越高，softmax 越容易把权重压到少数 token 上；但过度集中会丢掉多个上下文线索。")

        with st.expander("控制台输出与公式", expanded=False):
            st.code(str(data["log"]), language="text")
            st.latex(r"\mathrm{head}_i=\mathrm{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i")
            st.latex(r"\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O")
    except Exception as exc:
        render_module_error("part4_transformer/02_multihead_visual.py", exc)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_multihead_attention(d_model=32, n_heads=4, save_artifacts=False)
    return bool(data["figures"]) and len(data["entropies"]) == 4


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_multihead_attention))
