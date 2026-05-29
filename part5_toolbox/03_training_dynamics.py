"""Training dynamics legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    make_curve,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
    small_curve_figure,
)


MODULE_TITLE = "训练动态分析"
MODULE_SUMMARY = "联合观察 loss、权重分布、激活饱和率和更新幅度比。"
MODULE_TAGS = ["训练", "动态监控", "诊断", "工具箱"]
MODULE_RELATED_TOPICS = ["part5/02_gradient_monitor", "part5/04_hyperparam_search", "part6_universal_framework/training_demo", "part1/03_datasets_optimizers"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("学习率倍率", 1.0), ("网络深度", 4), ("随机种子", 42)),
    observations=("更新幅度比约在 1e-4 到 1e-2 较健康，中心参考值约 1e-3。",),
    misconceptions=("工程坑案例：loss 不降不一定是模型太小，可能是更新幅度比长期低于 1e-4。",),
    engineering=("把 loss、激活饱和率和更新幅度比放在一起看，能更快定位学习率、初始化或归一化问题。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：训练动态页要把 loss、权重分布、激活饱和率和更新幅度比放在一起看。",
            "更新幅度比 = lr * grad_norm / weight_norm，比单看梯度更接近参数实际改动。",
            "进阶思考：为什么更新幅度比比梯度范数更能判断学习率？",
        ],
    )


def compute_training_dynamics(seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    loss = make_curve(seed, 1.25, 0.28, steps=28, noise=0.025)
    saturation = make_curve(seed + 1, 0.12, 0.34, steps=28, noise=0.018)
    update_ratio = make_curve(seed + 2, 0.0016, 0.0009, steps=28, noise=0.00008)
    rows = [
        {"指标": "loss 下降幅度", "数值": round(loss[0] - loss[-1], 4), "解释": "下降越明显，说明优化方向整体有效。"},
        {"指标": "激活饱和率", "数值": round(saturation[-1], 4), "解释": "超过 0.2 要留意，超过 0.5 常要查初始化、输入尺度或激活函数。"},
        {"指标": "更新幅度比", "数值": round(update_ratio[-1], 6), "解释": "健康带大致在 1e-4 到 1e-2，太小学得慢，太大容易震荡。"},
        {"指标": "权重分布漂移", "数值": 0.083, "解释": "均值长期偏离 0 时，要检查学习率和正则化。"},
    ]
    figures = [
        ("training_dynamics_curves.png", small_curve_figure({"loss": loss, "激活饱和率": saturation}, title="训练动态曲线")),
        ("update_ratio.png", small_bar_figure([rows[1], rows[2], rows[3]], title="关键诊断指标")),
    ]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=rows,
        notes=[
            "loss 正常下降但激活饱和率升高时，可能后期出现梯度变小。",
            "更新幅度比太低时，不要急着加模型层数，先查学习率和梯度裁剪。",
            "权重百分位范围快速变宽，常见于学习率过高。",
        ],
        figures=figures,
        artifacts=artifacts,
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_training_dynamics, module_path="part5_toolbox/03_training_dynamics.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_training_dynamics(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_training_dynamics(save_artifacts=False)
    return data["rows"][2]["指标"] == "更新幅度比" and bool(data["figures"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
