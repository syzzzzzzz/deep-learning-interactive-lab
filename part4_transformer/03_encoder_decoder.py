"""Encoder, decoder and positional encoding legacy lesson."""

from __future__ import annotations

import numpy as np

from components.legacy_protocol import (
    LegacyLessonSpec,
    matrix_figure,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "编码器、解码器与位置编码"
MODULE_SUMMARY = "拆解 Transformer 编码器/解码器的数据流、mask 和正弦位置编码。"
MODULE_TAGS = ["Transformer", "位置编码", "Mask", "Encoder-Decoder"]
MODULE_RELATED_TOPICS = ["part4/01_attention_mechanism", "part4/02_multihead_visual", "part4/04_minimal_transformer", "part4/05_flash_attention"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("序列长度", 16), ("模型维度", 32), ("随机种子", 42)),
    observations=("位置编码把顺序注入 token 表示，causal mask 防止 decoder 偷看未来。",),
    misconceptions=("常见误区：Transformer 不是天然知道顺序，必须通过位置编码或位置偏置注入位置信息。",),
    engineering=("工程用途：先检查 mask 形状和广播规则，再排查 attention 输出。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：Encoder 读完整输入，Decoder 一边看已生成 token，一边 cross-attend 到 Encoder 输出。",
            "工程坑案例：causal mask 维度错一格，会让训练指标虚高但推理崩掉。",
            "进阶思考：正弦位置编码为什么可以外推到比训练更长的位置？",
        ],
    )


def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(0, d_model, 2)[None, :]
    div = np.exp(-np.log(10000.0) * dims / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div)
    pe[:, 1::2] = np.cos(positions * div)
    return pe


def compute_encoder_decoder(seq_len: int = 16, d_model: int = 32, seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    seq_len = max(4, min(int(seq_len), 64))
    d_model = max(8, min(int(d_model // 2 * 2), 128))
    pe = _positional_encoding(seq_len, d_model)
    causal = np.tril(np.ones((seq_len, seq_len)))
    sim = pe @ pe.T / (np.linalg.norm(pe, axis=1, keepdims=True) @ np.linalg.norm(pe, axis=1, keepdims=True).T + 1e-8)
    rows = [
        {"指标": "Encoder 输入", "数值": f"B x {seq_len} x {d_model}", "解释": "所有源 token 同时自注意力。"},
        {"指标": "Decoder 自注意力 mask", "数值": f"{seq_len} x {seq_len}", "解释": "下三角为 1，未来位置为 0。"},
        {"指标": "Cross-Attention", "数值": "Q=decoder, K/V=encoder", "解释": "生成端向输入端取信息。"},
        {"指标": "位置 0 与末位相似度", "数值": round(float(sim[0, -1]), 4), "解释": "距离越远，位置编码相似度通常越低但不是简单线性。"},
    ]
    figures = [
        ("positional_encoding.png", matrix_figure(pe[:, : min(32, d_model)].T, title="正弦位置编码")),
        ("causal_mask.png", matrix_figure(causal, title="Decoder Causal Mask")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "横轴是位置，纵轴是维度；低维振荡快，高维振荡慢。",
            "decoder self-attention 只能看当前位置及以前的 token。",
            "cross-attention 的 key/value 来自 encoder 输出，query 来自 decoder 状态。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"position_similarity": sim.tolist(), "seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_encoder_decoder, module_path="part4_transformer/03_encoder_decoder.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_encoder_decoder(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_encoder_decoder(seq_len=8, d_model=16, save_artifacts=False)
    return bool(data["figures"]) and "Cross-Attention" in data["log"]


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
