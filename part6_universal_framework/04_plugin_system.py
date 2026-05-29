"""Plugin system legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "插件系统"
MODULE_SUMMARY = "用注册表、配置模板和 hook 插槽扩展模型、数据集、任务和训练逻辑。"
MODULE_TAGS = ["统一框架", "插件", "注册表", "配置"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/02_modular_structure", "part6_universal_framework/03_full_project", "part6_universal_framework/05_one_click_training"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("插件类型", ("model", "dataset", "task", "hook")), ("hook 数", 3), ("随机种子", 42)),
    observations=("注册表让配置模板只写 name 和 params，就能替换模型、数据集和任务。",),
    misconceptions=("工程坑案例：插件静默加载失败最危险，必须打印文件名、异常和最终合并配置。",),
    engineering=("工程用途：让变化频繁的模型、数据集、任务和 hook 成为可插拔组件。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：插件系统的核心是稳定替换，不是炫技。",
            "注册表负责把名称映射到类和默认参数。",
            "配置模板把 model/dataset/task/training/hooks/output 分开写清楚。",
            "工程坑案例：名称冲突和默认参数覆盖不清是最常见事故。",
            "进阶思考：如果插件要改训练循环内部很多行，说明抽象边界可能错了。",
        ],
    )


def compute_plugin_system(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rows = [
        {"指标": "MODEL_REGISTRY", "数值": 5, "解释": "mlp、cnn、lstm、transformer、resnet 等模型入口。"},
        {"指标": "DATASET_REGISTRY", "数值": 3, "解释": "mnist、synthetic、cifar10 等数据源入口。"},
        {"指标": "TASK_REGISTRY", "数值": 3, "解释": "classification、regression、multilabel 等任务入口。"},
        {"指标": "HookSystem", "数值": 6, "解释": "on_train_start、on_epoch_end、on_train_end 等训练插槽。"},
    ]
    config_rows = [
        {"指标": "model", "数值": 1, "解释": "name + params。"},
        {"指标": "dataset", "数值": 1, "解释": "name + batch_size + val_ratio。"},
        {"指标": "training", "数值": 1, "解释": "epochs、lr、weight_decay、grad_clip。"},
        {"指标": "hooks", "数值": 1, "解释": "lr_logger、gradient_monitor、early_stopping。"},
    ]
    figures = [
        ("plugin_registry_counts.png", small_bar_figure(rows, title="注册表组件数量")),
        ("plugin_config_template.png", small_bar_figure(config_rows, title="实验配置模板")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows + config_rows,
        notes=[
            "插件加载时要检查重复注册和默认参数覆盖结果。",
            "hook 应该做可观察、可关闭、可记录的事情，不应暗改训练语义。",
            "配置模板要在启动时完整打印，方便复现实验和排查。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_plugin_system, module_path="part6_universal_framework/04_plugin_system.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_plugin_system(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_plugin_system(save_artifacts=False)
    return "注册表" in data["log"] and "配置模板" in data["log"]


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
