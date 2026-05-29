"""Modular structure legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    parameter_count_linear,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "模块化结构"
MODULE_SUMMARY = "用注册表、TrainConfig 和模块化训练器，让切换模型只改配置。"
MODULE_TAGS = ["统一框架", "模块化", "注册表", "配置"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/01_unified_interface", "part6_universal_framework/04_plugin_system", "part6_universal_framework/05_one_click_training"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("模型", ("mlp", "cnn", "lstm", "transformer_encoder")), ("epochs", 50), ("随机种子", 42)),
    observations=("模型、数据、任务、训练器分开后，切换模型只影响配置和注册表入口。",),
    misconceptions=("常见误区：模块化不是把文件拆碎，而是让依赖方向和职责稳定。",),
    engineering=("工程用途：注册表让配置文件可以声明模型名称，训练代码不再写 if/else。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：模块化结构的目标是降低替换模型和任务的成本。",
            "注册表负责 name -> class，TrainConfig 集中管理超参。",
            "进阶思考：哪些变化频繁的边界值得抽象？",
        ],
    )


def compute_modular_structure(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    rows = [
        {"指标": "MLP", "数值": parameter_count_linear(20, 64) + parameter_count_linear(64, 2), "解释": "二维/表格任务的默认基线。"},
        {"指标": "CNN", "数值": 32 * 1 * 3 * 3 + 64 * 32 * 3 * 3, "解释": "图像任务复用卷积特征提取。"},
        {"指标": "LSTM", "数值": 4 * (10 + 64 + 1) * 64, "解释": "序列任务用门控隐藏状态建模。"},
        {"指标": "Transformer", "数值": parameter_count_linear(64, 64, False) * 4, "解释": "注意力投影和输出投影构成主要参数。"},
    ]
    figures = [("modular_model_params.png", small_bar_figure(rows, title="同一训练器下可替换模型参数量"))]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "注册表替代散落的 if/else，让配置成为模型切换入口。",
            "TrainConfig 集中保存 epochs、batch_size、lr、grad_clip、scheduler。",
            "模块之间只通过稳定协议交互，才是真正模块化。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_modular_structure, module_path="part6_universal_framework/02_modular_structure.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_modular_structure(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_modular_structure(save_artifacts=False)
    return len(data["rows"]) >= 4 and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
