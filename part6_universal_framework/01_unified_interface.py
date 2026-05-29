"""Unified interface legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    parameter_count_linear,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
    small_curve_figure,
)


MODULE_TITLE = "统一接口"
MODULE_SUMMARY = "把数据、模型、训练、保存和推理整理成稳定边界，降低项目扩展成本。"
MODULE_TAGS = ["统一框架", "接口设计", "训练工程", "复现"]
MODULE_RELATED_TOPICS = ["part6_universal_framework/02_modular_structure", "part6_universal_framework/03_full_project", "part6_universal_framework/05_one_click_training", "part5/03_training_dynamics"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("batch_size", 64), ("lr", 0.001), ("patience", 10), ("随机种子", 42)),
    observations=("统一接口的核心不是少写几行代码，而是让数据、模型、训练和推理职责可替换。",),
    misconceptions=("工程坑案例：训练集和验证集各自 normalize，会造成线上分布对不上。",),
    engineering=("工程用途：把 fit/save/load/predict/count_params 作为模型生命周期协议。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：统一接口要给数据、模型、训练和推理划清边界。",
            "统一接口包含 TensorDatasetWrapper、TrainableMixin、MLP/SimpleCNN 三层职责。",
            "工程坑案例：统计量必须由训练集拟合，再复用到验证、测试和线上。",
            "进阶思考：分类变回归时，应替换 criterion/metric，而不是重写 Dataset。",
        ],
    )


def compute_unified_interface(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    model_rows = [
        {"指标": "TensorDatasetWrapper", "数值": 1, "解释": "负责类型转换、normalize、split 和 DataLoader。"},
        {"指标": "TrainableMixin", "数值": 5, "解释": "fit/save/load/predict/count_params 统一训练生命周期。"},
        {"指标": "MLP 参数量", "数值": parameter_count_linear(20, 64) + parameter_count_linear(64, 32) + parameter_count_linear(32, 2), "解释": "模型只描述结构，不关心训练循环。"},
        {"指标": "推荐 lr", "数值": 0.001, "解释": "Adam 小实验的稳妥起点。"},
    ]
    curves = {"train_loss": [0.72, 0.48, 0.34, 0.25, 0.2], "val_loss": [0.76, 0.52, 0.39, 0.33, 0.34]}
    figures = [
        ("unified_interface_blocks.png", small_bar_figure(model_rows[:3], title="统一接口职责分层")),
        ("unified_interface_curve.png", small_curve_figure(curves, title="统一 fit() 输出的训练曲线", ylabel="loss")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=model_rows,
        notes=[
            "Dataset 负责数据形态和统计状态，模型只负责 forward。",
            "TrainableMixin 让任意 nn.Module 拥有一致的 fit/save/load/predict。",
            "统一接口必须保存 normalize 统计量，否则复现和上线会漂移。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_unified_interface, module_path="part6_universal_framework/01_unified_interface.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_unified_interface(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_unified_interface(save_artifacts=False)
    return "统一接口" in data["log"] and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
