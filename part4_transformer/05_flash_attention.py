"""FlashAttention legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

import numpy as np

from components.legacy_protocol import (
    LegacyLessonSpec,
    attention_memory,
    matrix_figure,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "Flash Attention"
MODULE_SUMMARY = "理解标准注意力的 N x N 内存瓶颈，以及分块在线 softmax 如何减少显存访问。"
MODULE_TAGS = ["Transformer", "FlashAttention", "性能", "注意力"]
MODULE_RELATED_TOPICS = ["part4/01_attention_mechanism", "part4/02_multihead_visual", "part4/04_minimal_transformer", "part5/03_training_dynamics"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("序列长度", 512), ("head 数", 8), ("块大小", 64)),
    observations=("FlashAttention 不是减少数学量级，而是减少 N x N 注意力矩阵反复写回高带宽显存。",),
    misconceptions=("常见误区：FlashAttention 不是近似注意力，它通过分块和在线 softmax 保持数值等价。",),
    engineering=("工程用途：长序列训练先估算 attention matrix 显存，再决定是否用 flash/scaled_dot_product_attention。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：标准注意力的瓶颈常在 N x N 分数矩阵的读写。",
            "工程坑案例：只看 FLOPs 会低估显存带宽和中间矩阵的代价。",
            "进阶思考：为什么在线 softmax 需要同时维护最大值 m 和归一化因子 l？",
        ],
    )


def compute_flash_attention(seq_len: int = 512, heads: int = 8, block_size: int = 64, seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    seq_len = max(16, min(int(seq_len), 4096))
    heads = max(1, min(int(heads), 32))
    block_size = max(8, min(int(block_size), seq_len))
    full_mb = attention_memory(seq_len, heads, bytes_per_value=2)
    block_mb = attention_memory(block_size, heads, bytes_per_value=2)
    blocks = int(np.ceil(seq_len / block_size)) ** 2
    tiled = np.zeros((min(seq_len, 64), min(seq_len, 64)))
    b = max(1, min(block_size, 16))
    order = 1
    for i in range(0, tiled.shape[0], b):
        for j in range(0, tiled.shape[1], b):
            tiled[i : i + b, j : j + b] = order
            order += 1
    rows = [
        {"指标": "标准注意力矩阵", "数值": round(full_mb, 2), "解释": "完整 scores/weights 的估算显存 MB，随 N^2 增长。"},
        {"指标": "单块注意力矩阵", "数值": round(block_mb, 2), "解释": "分块后每次只处理 Bq x Bk 局部矩阵。"},
        {"指标": "块访问次数", "数值": blocks, "解释": "块越小，峰值显存更低，但循环次数更多。"},
        {"指标": "在线 softmax 状态", "数值": "m/l/O", "解释": "维护最大值、归一化因子和输出累加器，避免保存完整权重。"},
    ]
    figures = [
        ("flash_tiling.png", matrix_figure(tiled, title="FlashAttention 分块访问顺序")),
        ("flash_memory.png", small_bar_figure(rows[:2], title="标准 vs 分块峰值注意力矩阵 MB")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "标准注意力需要显式保存 N x N 权重矩阵。",
            "分块计算让中间矩阵留在更快的片上缓存里。",
            "块大小需要平衡峰值显存、并行度和循环开销。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"full_mb": full_mb, "block_mb": block_mb, "seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_flash_attention, module_path="part4_transformer/05_flash_attention.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_flash_attention(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_flash_attention(seq_len=128, heads=4, block_size=32, save_artifacts=False)
    return data["full_mb"] > data["block_mb"] and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
