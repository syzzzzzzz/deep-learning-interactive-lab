"""Hyperparameter search legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

import numpy as np

from components.legacy_protocol import (
    LegacyLessonSpec,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
    small_curve_figure,
)


MODULE_TITLE = "超参搜索进阶"
MODULE_SUMMARY = "用 LR Finder、学习率调度和敏感性分析缩小稳定训练范围。"
MODULE_TAGS = ["超参数", "LR Finder", "实验记录", "训练"]
MODULE_RELATED_TOPICS = ["part1/03_datasets_optimizers", "part5/03_training_dynamics", "part6_universal_framework/training_demo", "part5/data_training"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("搜索预算", 12), ("最大学习率", 0.1), ("随机种子", 42)),
    observations=("LR Finder 选择 loss 下降最快附近，但不要选已经反弹或发散的点。",),
    misconceptions=("真实踩坑：验证集和测试集混用会把最好参数调到测试集噪声上。",),
    engineering=("工程经验：学习率优先用对数尺度粗搜，再用验证集选择，测试集只做最终报告。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：超参搜索不是碰运气，而是缩小不稳定、欠拟合、过拟合的可能范围。",
            "LR Finder 横轴最好看 log scale，因为学习率跨越多个数量级。",
            "工程经验：保存每次失败配置，失败样本是调参知识库。",
            "真实踩坑：不要用测试集调参。",
            "进阶思考：随机搜索为什么在高维空间常比网格搜索更划算？",
        ],
    )


def compute_hyperparam_search(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    lrs = np.logspace(-5, 0, 28)
    losses = 1.2 - 0.24 * np.log10(lrs / lrs[0]) + 0.18 * (np.log10(lrs) + 2.2) ** 2
    losses = losses + rng.normal(0.0, 0.015, len(losses))
    best_idx = int(np.argmin(np.gradient(losses)))
    suggested_lr = float(lrs[best_idx])

    sensitivities = {"lr": 0.118, "hidden_size": 0.071, "dropout": 0.034, "weight_decay": 0.022}
    rows = [
        {"指标": "LR Finder 建议", "数值": f"{suggested_lr:.2e}", "解释": "loss 下降最快附近，后续用验证集细搜。"},
        {"指标": "最敏感超参", "数值": "lr", "解释": "改变学习率对验证分数影响最大。"},
        {"指标": "搜索预算", "数值": 12, "解释": "先随机粗搜，再围绕稳定区域局部细搜。"},
        {"指标": "调度策略", "数值": "Cosine / OneCycle", "解释": "平滑调度通常比硬阶梯更适合作为默认候选。"},
    ]
    figures = [
        ("lr_finder.png", small_curve_figure({"LR Finder loss": [round(float(x), 4) for x in losses]}, title="LR Finder")),
        ("hyperparam_sensitivity.png", small_bar_figure([{"指标": k, "数值": v, "解释": ""} for k, v in sensitivities.items()], title="超参敏感性分析")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "最高分配置周围也要稳定，否则可能只是随机噪声。",
            "搜索空间越高维，随机搜索越容易覆盖真正重要的维度。",
            "训练集训练、验证集选参、测试集最后一次评估。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"suggested_lr": suggested_lr, "sensitivities": sensitivities},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_hyperparam_search, module_path="part5_toolbox/04_hyperparam_search.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_hyperparam_search(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_hyperparam_search(save_artifacts=False)
    return "suggested_lr" in data and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
