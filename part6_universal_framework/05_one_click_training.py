"""One-click training legacy lesson, split into compute/render/smoke."""

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


MODULE_TITLE = "一键训练与评估"
MODULE_SUMMARY = "把 config、训练、验证、checkpoint、日志和曲线串成可复现实验闭环。"
MODULE_TAGS = ["训练工程", "一键训练", "checkpoint", "日志"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/03_full_project", "part6_universal_framework/04_plugin_system", "part6_universal_framework/07_project_template", "part5/03_training_dynamics"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("epochs", 10), ("patience", 5), ("monitor", ("accuracy", "val_loss", "f1")), ("随机种子", 42)),
    observations=("一键训练不是隐藏训练，而是把一次实验从配置到产物完整串起来。",),
    misconceptions=("工程坑案例：只保存 last.pt 容易上线过拟合后的最后一轮，默认应保存 best.pt。",),
    engineering=("工程用途：best.pt、training_log.csv、config.json 和 training_curves.png 必须在同一个 save_dir。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：一键训练要把配置、训练、评估、保存和日志串成闭环。",
            "best.pt 保存验证集最好的模型，通常比最后一轮更可靠。",
            "training_log.csv 是排查曲线异常的第一证据。",
            "config.json 是复现实验的说明书。",
            "工程坑案例：last.pt 可能是过拟合后的坏模型。",
        ],
    )


def compute_one_click_training(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rows = [
        {"指标": "config.json", "数值": 1, "解释": "记录实验名称、模型、学习率、优化器、seed 和输出目录。"},
        {"指标": "best.pt", "数值": 1, "解释": "monitor 指标最好的一轮 checkpoint。"},
        {"指标": "training_log.csv", "数值": 10, "解释": "逐 epoch 保存 loss、metric、lr 和耗时。"},
        {"指标": "training_curves.png", "数值": 1, "解释": "训练/验证曲线与学习率曲线，判断过拟合和欠拟合。"},
        {"指标": "final_result.json", "数值": 1, "解释": "最终测试集指标和最佳验证指标。"},
    ]
    curves = {"train_loss": [0.9, 0.66, 0.49, 0.38, 0.32], "val_acc": [0.62, 0.74, 0.81, 0.84, 0.835], "lr": [0.001, 0.0009, 0.00065, 0.00035, 0.0001]}
    figures = [
        ("one_click_artifacts.png", small_bar_figure(rows, title="一键训练产物清单")),
        ("one_click_curves.png", small_curve_figure(curves, title="一键训练日志曲线", ylabel="value")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "monitor 指标决定 best.pt 保存哪一轮，必须写入 checkpoint。",
            "训练中断恢复至少需要 config、checkpoint、optimizer_state 和日志。",
            "真实项目 save_dir 应包含任务、模型、日期和关键超参。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_one_click_training, module_path="part6_universal_framework/05_one_click_training.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_one_click_training(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_one_click_training(save_artifacts=False)
    return "best.pt" in data["log"] and "training_log.csv" in data["log"] and "config.json" in data["log"]


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
