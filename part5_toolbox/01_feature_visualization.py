"""Feature visualization legacy lesson, split into compute/render/smoke."""

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
    stable_rng,
)


MODULE_TITLE = "特征可视化工具"
MODULE_SUMMARY = "用 hook、滤波器响应和激活最大化观察模型内部学到了什么。"
MODULE_TAGS = ["工具箱", "特征可视化", "解释性", "调试"]
MODULE_RELATED_TOPICS = ["part2/02_feature_maps", "part2/08_visualization_gradcam", "part5/02_gradient_monitor", "part6_universal_framework/training_demo"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("观察层", ("conv1", "conv2", "fc1")), ("显示通道数", 8), ("激活最大化步数", 40)),
    observations=(
        "浅层特征通常看边缘和纹理，深层特征更稀疏、更接近任务语义。",
        "激活最大化图像越稳定，说明该通道越可能捕获了可重复模式。",
    ),
    misconceptions=("工程坑案例：特征图不是 Grad-CAM，不能直接当成最终分类因果解释。",),
    engineering=("先确认 layer_name、输入归一化和模型训练状态，再比较多张样本的稳定模式。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：特征可视化不是看漂亮图片，而是检查模型内部表征。",
            "工程坑案例：把特征图直接当最终关注区域会误判。",
            "进阶思考：同一输入在浅层和深层的稀疏性为什么不同？",
        ],
    )


def compute_feature_visualization(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rng = stable_rng(seed)
    activations = np.abs(rng.normal(0.0, 1.0, (8, 8)))
    activations[:2, :2] *= 2.2
    channel_strengths = np.sort(rng.uniform(0.18, 0.95, 6))[::-1]
    sparsity = float((activations < 0.25).mean())
    contrast = float(activations.max() / (activations.mean() + 1e-8))
    max_trace = np.maximum.accumulate(rng.normal(0.04, 0.015, 24).cumsum() + 0.4)

    rows = [
        {"指标": "hook 捕获层数", "数值": 3, "解释": "conv1、conv2、fc1 分别观察纹理、组合特征和高层激活。"},
        {"指标": "特征稀疏率", "数值": round(sparsity, 3), "解释": "越高表示多数位置不响应，常见于 ReLU 后的选择性特征。"},
        {"指标": "滤波器对比度", "数值": round(contrast, 3), "解释": "越高表示少数区域响应更强，需要结合多样本判断是否稳定。"},
        {"指标": "激活最大化终值", "数值": round(float(max_trace[-1]), 3), "解释": "梯度上升能找到让目标通道更兴奋的输入模式。"},
    ]
    figures = [
        ("feature_activation_map.png", matrix_figure(activations, title="特征图响应热力图")),
        ("feature_channel_strength.png", small_bar_figure([{"指标": f"Ch{i}", "数值": v, "解释": ""} for i, v in enumerate(channel_strengths)], title="通道响应强度")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "亮区表示该层某通道响应强，但不等价于最终分类依据。",
            "如果所有通道都像随机噪声，优先检查输入预处理和模型是否训练过。",
            "激活最大化需要正则化，否则容易生成极端像素噪声。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"activation_trace": [round(float(x), 4) for x in max_trace]},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_feature_visualization, module_path="part5_toolbox/01_feature_visualization.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_feature_visualization(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_feature_visualization(seed=7, save_artifacts=False)
    return bool(data["rows"]) and bool(data["figures"]) and data["rows"][1]["数值"] >= 0


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
