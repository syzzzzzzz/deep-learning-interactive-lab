"""Gradient monitor legacy lesson, split into compute/render/smoke."""

from __future__ import annotations

from components.legacy_protocol import (
    LegacyLessonSpec,
    print_learning_guide as _print_learning_guide,
    protocol_payload,
    run_or_render,
    save_figures,
    small_bar_figure,
)


MODULE_TITLE = "梯度监控"
MODULE_SUMMARY = "监控梯度范数、梯度消失、梯度爆炸和训练健康状态。"
MODULE_TAGS = ["工具箱", "梯度", "训练诊断", "可视化"]
MODULE_RELATED_TOPICS = ["part1/02_activations_normalization", "part3/07_advanced_training", "part5/03_training_dynamics", "part6_universal_framework/training_demo"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

SPEC = LegacyLessonSpec(
    title=MODULE_TITLE,
    summary=MODULE_SUMMARY,
    tags=tuple(MODULE_TAGS),
    related_topics=tuple(MODULE_RELATED_TOPICS),
    practice_target=PRACTICE_TARGET,
    controls=(("场景", ("混合状态", "全部正常", "严重消失", "严重爆炸")), ("梯度裁剪阈值", 1.0), ("随机种子", 42)),
    observations=("绿色是健康梯度，蓝色低矮代表梯度消失，红色高亮代表梯度爆炸。",),
    misconceptions=("工程坑案例：只看验证准确率会错过 embedding 层长期无梯度这类问题。",),
    engineering=("loss 发散加梯度暴涨先降学习率；loss 不动加前层梯度接近 0 先查初始化和激活函数。",),
)


def print_learning_guide() -> None:
    _print_learning_guide(
        MODULE_TITLE,
        [
            "学习导读：梯度监控是在给训练过程做体检。",
            "梯度消失：参数几乎学不动，常见于深层 Sigmoid/Tanh。",
            "梯度爆炸：更新过猛，常用梯度裁剪和降低学习率处理。",
            "进阶思考：只有最后一层梯度很大时，先查标签、loss 和学习率。",
        ],
    )


def _scenario_norms(scenario: str) -> list[float]:
    if scenario == "全部正常":
        return [0.48, 0.35, 0.61, 0.42, 0.29]
    if scenario == "严重消失":
        return [0.12, 0.004, 0.00003, 0.0000002, 0.00000001]
    if scenario == "严重爆炸":
        return [1.2, 45.0, 320.0, 1500.0, 8900.0]
    return [0.85, 0.42, 0.003, 0.000001, 0.95, 120.5, 0.67]


def compute_gradient_monitor(scenario: str = "混合状态", seed: int = 42, save_artifacts: bool = False, **_: object) -> dict[str, object]:
    norms = _scenario_norms(scenario)
    rows = []
    for index, norm in enumerate(norms, 1):
        if norm <= 1e-6:
            status = "梯度消失"
        elif norm > 100:
            status = "梯度爆炸"
        else:
            status = "正常"
        rows.append({"指标": f"Layer{index}", "数值": norm, "解释": status})

    status_counts = [
        {"指标": "正常层数", "数值": sum(1 for item in norms if 1e-6 < item <= 100), "解释": "可以稳定更新的层。"},
        {"指标": "梯度消失", "数值": sum(1 for item in norms if item <= 1e-6), "解释": "更新信号过小，参数几乎不动。"},
        {"指标": "梯度爆炸", "数值": sum(1 for item in norms if item > 100), "解释": "更新信号过大，训练容易发散。"},
    ]
    figures = [("gradient_health.png", small_bar_figure(rows, title="梯度健康仪表盘"))]
    artifacts = save_figures(figures, save_artifacts)
    return protocol_payload(
        SPEC,
        rows=status_counts + rows,
        notes=[
            "至少观察 10 到 50 步趋势，不要用单个 batch 下结论。",
            "梯度消失优先查激活函数、初始化、残差连接和归一化。",
            "梯度爆炸优先降低学习率并加入 clip_grad_norm_。",
        ],
        figures=figures,
        artifacts=artifacts,
        extra={"scenario": scenario, "norms": norms, "seed": seed},
    )


def render() -> None:
    import streamlit as st  # noqa: F401

    from components.legacy_protocol import render_protocol_page

    render_protocol_page(spec=SPEC, compute=compute_gradient_monitor, module_path="part5_toolbox/02_gradient_monitor.py")


def compute(seed: int = 42, save_artifacts: bool = False, **kwargs: object) -> dict[str, object]:
    return compute_gradient_monitor(seed=seed, save_artifacts=save_artifacts, **kwargs)


def smoke() -> bool:
    data = compute_gradient_monitor(save_artifacts=False)
    return any(row["解释"] == "梯度爆炸" for row in data["rows"]) and any(row["解释"] == "梯度消失" for row in data["rows"])


if __name__ == "__main__":
    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
