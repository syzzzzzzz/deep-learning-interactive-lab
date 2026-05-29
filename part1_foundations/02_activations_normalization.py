"""Activation, initialization, normalization and overfitting legacy lesson."""

from __future__ import annotations

import numpy as np

from components.legacy_protocol import (
    LegacyLessonSpec,
    make_curve,
    matrix_figure,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
    small_curve_figure,
    stable_rng,
)


MODULE_TITLE = "激活函数、初始化与归一化"
MODULE_SUMMARY = "比较激活函数、权重初始化、归一化和过拟合控制如何影响训练稳定性。"
MODULE_TAGS = ["基础", "激活函数", "初始化", "归一化", "正则化"]
MODULE_RELATED_TOPICS = ["part1/01_tensors_gradients", "part1/03_datasets_optimizers", "part5/02_gradient_monitor", "part5/03_training_dynamics"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("激活函数", ("ReLU", "GELU", "Tanh", "Sigmoid")), ("网络深度", 20), ("Dropout", 0.2), ("随机种子", 42)),
    observations=(
        "Sigmoid/Tanh 在两端容易饱和，梯度变小；ReLU/GELU 更适合深层网络。",
        "He 初始化让 ReLU 网络的信号方差更稳定，LayerNorm 不依赖 batch size。",
    ),
    misconceptions=("常见误区：归一化不是为了让 loss 更好看，而是为了让每层输入尺度更可控。",),
    engineering=("工程用途：先看激活饱和率和梯度范数，再决定换激活函数、初始化或加归一化。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：激活函数决定非线性，初始化决定训练起点，归一化决定信号尺度。",
            "工程坑案例：深层 Sigmoid loss 不动时，不要先加层数，先查梯度是否消失。",
            "进阶思考：为什么 He 初始化更适合 ReLU，而 Xavier 更适合 Tanh/Sigmoid？",
        ],
    )


def _activation_curves() -> dict[str, list[float]]:
    x = np.linspace(-4, 4, 80)
    sigmoid = 1 / (1 + np.exp(-x))
    relu = np.maximum(x, 0)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return {"Sigmoid": sigmoid.tolist(), "ReLU": relu.tolist(), "GELU": gelu.tolist()}


def compute_activations_normalization(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rng = stable_rng(seed)
    init_rows = [
        {"指标": "零初始化", "数值": 0.0, "解释": "所有神经元收到同样梯度，无法打破对称性。"},
        {"指标": "小随机", "数值": 0.04, "解释": "深层信号快速衰减，容易梯度消失。"},
        {"指标": "Xavier", "数值": 0.86, "解释": "适合 Tanh/Sigmoid，尽量保持前后层方差。"},
        {"指标": "He", "数值": 1.03, "解释": "适合 ReLU，补偿一半激活被截断的方差损失。"},
    ]
    batch = rng.normal(2.0, 3.0, (4, 3, 2, 2))
    batch_normed = (batch - batch.mean(axis=(0, 2, 3), keepdims=True)) / (batch.std(axis=(0, 2, 3), keepdims=True) + 1e-8)
    layer_normed = (batch - batch.mean(axis=(1, 2, 3), keepdims=True)) / (batch.std(axis=(1, 2, 3), keepdims=True) + 1e-8)
    train_loss = make_curve(seed, 0.72, 0.08, steps=30, noise=0.012)
    val_loss = make_curve(seed + 1, 0.82, 0.24, steps=30, noise=0.025)
    overfit_gap = round(float(val_loss[-1] - train_loss[-1]), 4)
    rows = [
        {"指标": "BatchNorm 全局均值", "数值": round(float(batch_normed.mean()), 4), "解释": "按通道归一化后，统计量接近 0。"},
        {"指标": "LayerNorm 全局标准差", "数值": round(float(layer_normed.std()), 4), "解释": "按样本归一化后，不依赖 batch size。"},
        {"指标": "过拟合差距", "数值": overfit_gap, "解释": "验证损失高于训练损失，说明模型开始记住训练噪声。"},
    ] + init_rows
    figures = [
        ("activation_curves.png", small_curve_figure(_activation_curves(), title="常见激活函数曲线")),
        ("initialization_stability.png", small_bar_figure(init_rows, title="初始化后的深层信号标准差")),
        ("normalization_matrix.png", matrix_figure(batch_normed[0, 0], title="归一化后局部响应")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "观察激活函数两端是否饱和，饱和越多，梯度越容易变小。",
            "观察初始化后的信号标准差是否长期接近 1。",
            "调 Dropout 或权重衰减时，看训练/验证差距是否缩小。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"train_loss": train_loss, "val_loss": val_loss},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_activations_normalization, module_path="part1_foundations/02_activations_normalization.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_activations_normalization(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_activations_normalization(save_artifacts=False)
    return bool(data["rows"]) and bool(data["figures"]) and "过拟合差距" in data["log"]


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
