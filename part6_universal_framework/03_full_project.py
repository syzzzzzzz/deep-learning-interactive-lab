"""Full project skeleton legacy lesson, split into compute/render/smoke."""

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


MODULE_TITLE = "完整项目骨架"
MODULE_SUMMARY = "用 UniversalTrainer、UniversalVisualizer 和实验目录保证训练可复现、可评估、可交接。"
MODULE_TAGS = ["项目工程", "UniversalTrainer", "UniversalVisualizer", "复现实验"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/01_unified_interface", "part6_universal_framework/05_one_click_training", "part6_universal_framework/07_project_template"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("epochs", 10), ("grad_clip", 1.0), ("patience", 5), ("随机种子", 42)),
    observations=("完整项目骨架要保证下个月还能复现实验，而不只是今天能跑一次。",),
    misconceptions=("工程坑案例：只保存 best_model.pth，不保存 config 和数据切分，三周后无法复现指标。",),
    engineering=("工程用途：UniversalTrainer 管训练闭环，UniversalVisualizer 管模型摘要、参数分布和错误样本。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：完整项目骨架解决复现实验和交接问题。",
            "UniversalTrainer 统一 train/eval、调度、早停和保存最优模型。",
            "UniversalVisualizer 负责模型结构、参数分布、预测样本和错误案例。",
            "复现实验必须保存 config、checkpoint、training_history.json、曲线和最终评估。",
        ],
    )


def compute_full_project(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rows = [
        {"指标": "UniversalTrainer", "数值": 6, "解释": "训练、验证、调度、早停、保存、历史记录。"},
        {"指标": "UniversalVisualizer", "数值": 4, "解释": "模型摘要、参数分布、预测样本、错误案例。"},
        {"指标": "config.json", "数值": 1, "解释": "记录模型、数据、优化器、seed 和输出目录。"},
        {"指标": "training_history.json", "数值": 1, "解释": "记录 loss、metric、lr，支撑复现实验。"},
    ]
    curves = {"train_loss": [1.0, 0.72, 0.52, 0.41, 0.35], "val_loss": [1.1, 0.79, 0.59, 0.5, 0.51], "lr": [0.001, 0.0009, 0.00065, 0.00035, 0.0001]}
    figures = [
        ("full_project_contract.png", small_bar_figure(rows, title="完整项目职责清单")),
        ("full_project_history.png", small_curve_figure(curves, title="训练历史产物", ylabel="value")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "evaluate 只做评估，不应偷偷 optimizer.step()。",
            "实验目录要把 config、checkpoint、history、曲线和评估结果放在一起。",
            "业务指标和验证 loss 冲突时，先查 metric_fn 和数据切分。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_full_project, module_path="part6_universal_framework/03_full_project.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_full_project(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_full_project(save_artifacts=False)
    return "UniversalTrainer" in data["log"] and "复现实验" in data["log"]


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
