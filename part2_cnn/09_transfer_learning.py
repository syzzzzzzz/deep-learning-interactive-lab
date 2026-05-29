"""Transfer learning legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
    small_curve_figure,
)


MODULE_TITLE = "迁移学习"
MODULE_SUMMARY = "复用预训练模型，比较特征提取、部分微调、差异化学习率和渐进解冻。"
MODULE_TAGS = ["CNN", "迁移学习", "微调", "工程"]
MODULE_RELATED_TOPICS = ["part2/03_classic_architectures", "part2/08_visualization_gradcam", "part5/04_hyperparam_search", "part6_universal_framework/training_demo"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("目标类别数", 10), ("样本量", 1000), ("解冻起点", ("fc", "layer4", "layer3", "all"))),
    observations=("小数据优先特征提取，中等数据解冻高层，大数据再考虑全模型微调。",),
    misconceptions=("常见误区：迁移学习不是直接换分类头就完事，输入预处理必须和预训练分布对齐。",),
    engineering=("工程用途：差异化学习率让预训练层慢慢改，分类头快速适配新任务。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：先冻结大部分视觉特征，再逐步让高层适应新数据。",
            "工程坑案例：忘记 ImageNet mean/std 会让预训练特征整体偏移。",
            "进阶思考：为什么浅层边缘纹理更通用，而高层语义更任务相关？",
        ],
    )


def compute_transfer_learning(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    total_params = 11_689_512
    strategies = [
        {"指标": "只训分类头", "数值": 5_130, "解释": "冻结 backbone，适合样本少、类别相近的任务。"},
        {"指标": "解冻 layer4", "数值": 8_398_858, "解释": "只微调高层语义，常是迁移学习默认起点。"},
        {"指标": "解冻 layer3+", "数值": 10_497_546, "解释": "数据更多或目标域差异更大时使用。"},
        {"指标": "全模型微调", "数值": total_params, "解释": "表达力最强，但更容易过拟合，也更吃显存。"},
    ]
    acc_curves = {
        "从零训练": [35, 50, 58, 64, 70],
        "特征提取": [65, 75, 80, 83, 85],
        "部分微调": [70, 80, 85, 88, 90],
        "渐进解冻": [65, 78, 85, 89, 93],
    }
    rows = [
        {"指标": "预处理均值", "数值": "0.485/0.456/0.406", "解释": "ImageNet RGB 均值，目标数据应使用同一尺度。"},
        {"指标": "分类头学习率", "数值": "1e-2", "解释": "新头随机初始化，需要更快学习。"},
        {"指标": "backbone 学习率", "数值": "1e-3", "解释": "预训练层已有知识，微调用更小步长。"},
    ] + strategies
    figures = [
        ("transfer_strategy_acc.png", small_curve_figure(acc_curves, title="迁移学习策略准确率示意", ylabel="acc")),
        ("transfer_trainable_params.png", small_bar_figure(strategies, title="不同解冻策略的可训练参数")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "先确认输入尺寸、通道顺序和归一化与预训练模型一致。",
            "样本少时不要一开始全模型微调，先让分类头收敛。",
            "渐进解冻时，解冻越深，学习率通常越小。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"total_params": total_params, "seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_transfer_learning, module_path="part2_cnn/09_transfer_learning.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_transfer_learning(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_transfer_learning(save_artifacts=False)
    return data["total_params"] > 1_000_000 and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
