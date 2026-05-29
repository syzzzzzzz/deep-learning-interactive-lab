"""Legacy Streamlit demo, now protocolized for the HTML-first site."""

from __future__ import annotations

import numpy as np

from components.legacy_protocol import (
    LegacyLessonSpec,
    matrix_figure,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_curve_figure,
    softmax,
    stable_rng,
)


MODULE_TITLE = "可视化实验台"
MODULE_SUMMARY = "保留旧 Streamlit 实验台的三类核心计算：决策边界、卷积响应和注意力热力图。"
MODULE_TAGS = ["实验台", "可视化", "HTML 迁移", "教学演示"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/neural_network_playground", "part6_universal_framework/training_demo", "part4/02_multihead_visual", "part2/01_convolution_visual"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("实验", ("决策边界", "卷积响应", "注意力热力图")), ("样本数", 240), ("随机种子", 42)),
    observations=("课程交互已迁移到 HTML 主站；这里保留旧实验台的轻量 compute 合约。",),
    misconceptions=("常见误区：可视化演示不等于真实训练报告，它用于建立直觉和做 sanity check。",),
    engineering=("工程用途：先在轻量实验台看趋势，再进入中央控制台做完整训练联动。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：旧 Streamlit demo 已迁移为 HTML 优先，Python 侧保留核心计算。",
            "决策边界看模型如何把平面切开。",
            "卷积响应看小滤波器如何重读图像。",
            "注意力热力图看每个 token 把权重分给谁。",
        ],
    )


def compute_streamlit_demo(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rng = stable_rng(seed)
    grid = np.linspace(-2, 2, 32)
    xx, yy = np.meshgrid(grid, grid)
    boundary = 1 / (1 + np.exp(-(xx**2 - yy + 0.25 * np.sin(xx * 2))))
    image = np.exp(-((xx + 0.4) ** 2 + (yy - 0.2) ** 2)) + 0.7 * np.exp(-((xx - 0.35) ** 2 + (yy + 0.35) ** 2) / 0.2)
    kernel_response = np.abs(np.gradient(image)[0]) + np.abs(np.gradient(image)[1])
    tokens = 8
    q = rng.normal(size=(tokens, 16))
    k = rng.normal(size=(tokens, 16))
    attention = softmax(q @ k.T / 4.0, axis=-1)
    loss = [0.9, 0.68, 0.53, 0.43, 0.37, 0.34]
    rows = [
        {"指标": "决策边界", "数值": round(float(boundary.mean()), 4), "解释": "背景概率地形，0.5 附近就是分类边界。"},
        {"指标": "卷积响应", "数值": round(float(kernel_response.max()), 4), "解释": "局部变化越强，边缘响应越亮。"},
        {"指标": "注意力熵", "数值": round(float(-(attention * np.log(attention + 1e-8)).sum(axis=1).mean()), 4), "解释": "熵越低，注意力越集中。"},
        {"指标": "迁移状态", "数值": "HTML first", "解释": "旧 demo 保留计算，主体验迁移到静态 HTML 站。"},
    ]
    figures = [
        ("streamlit_demo_boundary.png", matrix_figure(boundary, title="旧实验台：决策边界")),
        ("streamlit_demo_attention.png", matrix_figure(attention, title="旧实验台：注意力热力图")),
        ("streamlit_demo_loss.png", small_curve_figure({"loss": loss}, title="旧实验台：训练曲线", ylabel="loss")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "旧 Streamlit 页不再承担主站体验，避免切换页面时重训造成卡顿。",
            "决策边界、卷积和注意力三个核心计算仍可 smoke 和复用。",
            "完整课程交互应进入 HTML 主站和中央控制台。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_streamlit_demo, module_path="part6_universal_framework/06_streamlit_demo.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_streamlit_demo(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_streamlit_demo(save_artifacts=False)
    return "HTML first" in data["log"] and len(data["figures"]) == 3


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
