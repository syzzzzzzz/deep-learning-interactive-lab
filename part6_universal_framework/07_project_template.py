"""Project template legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "项目模板"
MODULE_SUMMARY = "把训练入口、评估脚本、K-Fold、集成预测和实验产物组织成可复用模板。"
MODULE_TAGS = ["项目模板", "训练入口", "评估脚本", "K-Fold", "集成"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/03_full_project", "part6_universal_framework/05_one_click_training", "part6_universal_framework/04_plugin_system"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("fold 数", 5), ("集成模型数", 3), ("随机种子", 42)),
    observations=("项目模板的价值在于复现和交接，不只是把训练代码拆成几个文件。",),
    misconceptions=("工程坑案例：评估脚本只打印 checkpoint 信息不算闭环，必须加载数据并输出指标。",),
    engineering=("工程用途：训练入口和评估脚本分开，K-Fold 和 ensemble 放在可选工具层。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：项目模板的价值是复现、交接和扩展。",
            "训练入口 parse_args 允许命令行覆盖 config、model、lr、epochs 和 gpu。",
            "评估脚本只加载 checkpoint 和测试数据，不再训练。",
            "K-Fold 用多次切分估计稳定性，ensemble_predict 平均多个 checkpoint 概率。",
            "工程坑案例：模板复制后忘记补评估数据加载，会让项目看似闭环但无法报告。",
        ],
    )


def compute_project_template(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rows = [
        {"指标": "训练入口", "数值": 5, "解释": "config、model、lr、epochs、gpu 可从命令行覆盖。"},
        {"指标": "评估脚本", "数值": 3, "解释": "checkpoint、config、split 三个关键输入。"},
        {"指标": "K-Fold", "数值": 5, "解释": "每个 fold 独立 save_dir，避免 checkpoint 覆盖。"},
        {"指标": "ensemble_predict", "数值": 3, "解释": "平均多个模型概率，降低方差但增加推理成本。"},
        {"指标": "产物追溯", "数值": 6, "解释": "config、seed、数据切分、checkpoint、training_log、final_result。"},
    ]
    figures = [("project_template_checklist.png", small_bar_figure(rows, title="项目模板闭环检查"))]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "训练脚本和评估脚本分开，能避免测试阶段意外更新参数。",
            "每个 fold 必须写入独立目录，日志和 checkpoint 不能互相覆盖。",
            "线上指标和离线验证冲突时，先查数据切分和指标定义。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_project_template, module_path="part6_universal_framework/07_project_template.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_project_template(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_project_template(save_artifacts=False)
    return "K-Fold" in data["log"] and "评估脚本" in data["log"]


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
