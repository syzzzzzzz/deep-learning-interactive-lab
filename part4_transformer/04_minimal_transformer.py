"""Minimal Transformer legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

import numpy as np

from components.legacy_protocol import (
    LegacyLessonSpec,
    make_curve,
    matrix_figure,
    parameter_count_linear,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_curve_figure,
    softmax,
    stable_rng,
)


MODULE_TITLE = "极简 Transformer 实现"
MODULE_SUMMARY = "用最小组件串起 Embedding、Multi-Head Attention、FFN、残差和 LayerNorm。"
MODULE_TAGS = ["Transformer", "最小实现", "注意力", "架构"]
MODULE_RELATED_TOPICS = ["part4/01_attention_mechanism", "part4/02_multihead_visual", "part4/03_encoder_decoder", "part6_universal_framework/neural_network_playground"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("d_model", 64), ("head 数", 4), ("层数", 2), ("随机种子", 42)),
    observations=("最小 Transformer 的核心是形状守恒：残差连接要求每个子层输出仍是 d_model。",),
    misconceptions=("常见误区：多头注意力不是把模型复制多份，而是把 d_model 切成多个子空间并行取信息。",),
    engineering=("工程用途：先打印每一步 tensor shape，再检查 mask、head 维度和 LayerNorm 位置。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：先让张量形状全程对齐，再讨论训练质量。",
            "工程坑案例：d_model 不能被 head 数整除时，split heads 会直接错位。",
            "进阶思考：为什么 FFN 中间维度通常扩大到 4 倍？",
        ],
    )


def compute_minimal_transformer(d_model: int = 64, heads: int = 4, layers: int = 2, seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    d_model = max(16, int(d_model))
    heads = max(1, int(heads))
    if d_model % heads != 0:
        heads = 1
    layers = max(1, min(int(layers), 8))
    rng = stable_rng(seed)
    tokens = 8
    x = rng.normal(size=(tokens, d_model))
    q = x @ rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    k = x @ rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    attn = softmax(q @ k.T / np.sqrt(d_model), axis=-1)
    loss = make_curve(seed, 3.2, 0.84, steps=26, noise=0.04)
    rows = [
        {"指标": "head_dim", "数值": d_model // heads, "解释": "每个 head 分到的子空间维度。"},
        {"指标": "QKV 参数", "数值": parameter_count_linear(d_model, d_model, False) * 3, "解释": "Q/K/V 三个线性投影。"},
        {"指标": "FFN 参数", "数值": parameter_count_linear(d_model, d_model * 4) + parameter_count_linear(d_model * 4, d_model), "解释": "两层 MLP，通常先扩到 4 倍再压回。"},
        {"指标": "总 block 数", "数值": layers, "解释": "堆叠越深，表达力更强，训练稳定性要求也更高。"},
    ]
    figures = [
        ("minimal_attention.png", matrix_figure(attn, title="最小注意力权重")),
        ("minimal_transformer_loss.png", small_curve_figure({"toy loss": loss}, title="极简训练 loss 示意", ylabel="loss")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "Multi-Head Attention 输入输出都保持 B x T x d_model。",
            "FFN 在每个 token 上独立应用，不在时间维度混合信息。",
            "残差连接要求子层输出形状和输入完全一致。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"attention": attn.tolist()},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_minimal_transformer, module_path="part4_transformer/04_minimal_transformer.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_minimal_transformer(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_minimal_transformer(d_model=32, heads=4, layers=1, save_artifacts=False)
    return data["rows"][0]["数值"] == 8 and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
