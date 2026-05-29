"""Transformer debugging legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "Transformer 调试问题集"
MODULE_SUMMARY = "用结构化清单定位 mask、位置编码、梯度、学习率和生成退化问题。"
MODULE_TAGS = ["Transformer", "调试", "Mask", "训练诊断"]
MODULE_RELATED_TOPICS = ["part4/03_encoder_decoder", "part4/04_minimal_transformer", "part5/02_gradient_monitor", "part5/03_training_dynamics"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("问题类型", ("mask", "nan", "重复生成", "梯度爆炸")), ("裁剪阈值", 1.0), ("随机种子", 42)),
    observations=("Transformer 故障通常先从 shape/mask 检查开始，再看梯度、学习率和生成采样。",),
    misconceptions=("常见误区：训练 loss 很低不代表生成正确，decoder 如果偷看未来会给出虚假好指标。",),
    engineering=("工程用途：为每类故障准备最小复现、检测函数和修复动作。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：调试 Transformer 先查 mask 和 shape，再查数值稳定性。",
            "工程坑案例：padding mask 方向反了，训练能跑但注意力全看 PAD。",
            "进阶思考：重复生成时应该先调 temperature/top-p，还是先查训练数据？",
        ],
    )


def compute_debug_problems(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    cases = [
        {"指标": "mask 维度错", "数值": 5, "解释": "检查形状是否能广播到 B x H x T x T。"},
        {"指标": "NaN/Inf", "数值": 4, "解释": "检查 softmax 前分数、学习率和混合精度缩放。"},
        {"指标": "梯度爆炸", "数值": 4, "解释": "看 grad_norm，加入裁剪并降低学习率。"},
        {"指标": "重复生成", "数值": 3, "解释": "检查采样温度、top-p、重复惩罚和训练语料。"},
        {"指标": "位置编码错位", "数值": 3, "解释": "检查 position id 是否从 0 开始且和 padding 对齐。"},
    ]
    figures = [("transformer_debug_severity.png", small_bar_figure(cases, title="Transformer 常见问题严重度"))]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=cases,
        notes=[
            "mask 错误会直接改变模型能看到的信息范围，是第一优先级。",
            "NaN 多数来自过大学习率、未裁剪梯度或 softmax 前分数过大。",
            "生成质量问题要同时看训练数据、采样参数和长度惩罚。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_debug_problems, module_path="part4_transformer/06_debug_problems.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_debug_problems(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_debug_problems(save_artifacts=False)
    return "mask" in data["log"] and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
