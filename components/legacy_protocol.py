"""Shared render/compute helpers for protocolized legacy lessons."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.legacy_runtime import run_cli, run_or_render, running_under_streamlit


@dataclass(frozen=True)
class LegacyLessonSpec:
    title: str
    summary: str
    tags: tuple[str, ...]
    related_topics: tuple[str, ...]
    practice_target: str
    controls: tuple[tuple[str, Any], ...]
    observations: tuple[str, ...]
    misconceptions: tuple[str, ...]
    engineering: tuple[str, ...]


def print_learning_guide(title: str, points: Iterable[str] | None = None) -> None:
    """Print a short learning guide for direct script runs."""

    print("学习导读")
    print("=" * 48)
    print(title)
    if points:
        for index, point in enumerate(points, 1):
            print(f"{index}. {point}")
    print("工程坑案例：先看可观测指标，再动模型结构。")
    print("进阶思考：把本页结论迁移到中央控制台的真实训练配置。")


def stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def make_curve(seed: int, start: float, end: float, steps: int = 24, noise: float = 0.02) -> list[float]:
    rng = stable_rng(seed)
    base = np.linspace(start, end, steps)
    wiggle = rng.normal(0.0, noise, steps)
    curve = np.maximum(base + wiggle, 0.0)
    return [round(float(value), 4) for value in curve]


def row(label: str, value: object, note: str) -> dict[str, object]:
    return {"指标": label, "数值": value, "解释": note}


def summarize_rows(rows: Iterable[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in rows:
        label = item.get("指标", item.get("项目", "项目"))
        value = item.get("数值", item.get("结果", ""))
        note = item.get("解释", item.get("说明", ""))
        lines.append(f"- {label}: {value}。{note}")
    return "\n".join(lines)


def small_bar_figure(rows: list[dict[str, object]], *, title: str, value_key: str = "数值"):
    from components.resource_manager import safe_mpl_figure

    labels = [str(item.get("指标", item.get("项目", ""))) for item in rows]
    values = [float(item.get(value_key, 0) or 0) for item in rows]
    with safe_mpl_figure(figsize=(7.2, 3.8)) as fig:
        ax = fig.subplots()
        ax.bar(labels, values, color=["#2a2118", "#b08a4f", "#7c756c", "#d8c7ad", "#4f5d52", "#8a6f49"][: len(labels)])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        return fig


def small_curve_figure(curves: dict[str, list[float]], *, title: str, ylabel: str = "value"):
    from components.resource_manager import safe_mpl_figure

    with safe_mpl_figure(figsize=(7.5, 3.8)) as fig:
        ax = fig.subplots()
        for label, values in curves.items():
            ax.plot(values, label=label, linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig


def matrix_figure(matrix: np.ndarray, *, title: str):
    from components.resource_manager import safe_mpl_figure

    with safe_mpl_figure(figsize=(5.8, 4.8)) as fig:
        ax = fig.subplots()
        image = ax.imshow(matrix, cmap="cividis", aspect="auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("列 / key")
        ax.set_ylabel("行 / query")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig


def protocol_payload(
    spec: LegacyLessonSpec,
    *,
    rows: list[dict[str, object]],
    notes: list[str],
    figures: list[tuple[str, object]] | None = None,
    artifacts: list[Path] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    log = "\n".join(
        [
            spec.title,
            spec.summary,
            "",
            summarize_rows(rows),
            "",
            "观察重点：",
            *[f"- {item}" for item in notes],
        ]
    )
    payload: dict[str, object] = {
        "title": spec.title,
        "summary": spec.summary,
        "rows": rows,
        "notes": notes,
        "figures": figures or [],
        "artifacts": artifacts or [],
        "log": log,
    }
    if extra:
        payload.update(extra)
    return payload


def save_figures(figures: list[tuple[str, object]], save_artifacts: bool) -> list[Path]:
    if not save_artifacts:
        return []
    from components.resource_manager import get_artifact_path

    artifacts: list[Path] = []
    for filename, fig in figures:
        path = get_artifact_path(filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        artifacts.append(path)
    return artifacts


def render_protocol_page(
    *,
    spec: LegacyLessonSpec,
    compute: Callable[..., dict[str, object]],
    module_path: str,
    sidebar: Callable[[Any], dict[str, object]] | None = None,
) -> None:
    import streamlit as st
    from components.error_boundary import render_module_error
    from components.resource_manager import clean_old_artifacts

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=spec.title, layout="wide", initial_sidebar_state="auto")
        st.link_button("返回主界面", "/", width="content")
        st.title(spec.title)
        st.caption(spec.summary)
        st.info("学习导读：先看指标含义，再调参数，最后去中央控制台实战。")

        params: dict[str, object] = {}
        with st.sidebar:
            st.subheader("参数")
            if sidebar is not None:
                params = sidebar(st)
            else:
                params = default_sidebar_controls(st, spec)
            if st.button("去中央控制台实战", width="stretch"):
                st.query_params["module"] = spec.practice_target
                st.rerun()

        data = compute(**params, save_artifacts=True)
        left, right = st.columns([0.58, 0.42])
        with left:
            st.subheader("核心观测")
            if data.get("figures"):
                for _, fig in data["figures"][:2]:
                    st.pyplot(fig, clear_figure=False)
            else:
                st.dataframe(data["rows"], width="stretch")
        with right:
            st.subheader("诊断表")
            st.dataframe(data["rows"], width="stretch")
            st.markdown("#### 观察什么变化")
            for note in data["notes"]:
                st.markdown(f"- {note}")

        with st.expander("零基础解释模板", expanded=False):
            st.markdown(f"**这是什么？** {spec.summary}")
            st.markdown("**生活类比** 像调一台学习机器的旋钮：先知道仪表盘读数，再决定往哪边调。")
            st.markdown("**一句话直觉** 参数改变的是信息流、梯度流或工程流程的约束条件。")
            st.markdown("**严谨定义** 本页把旧教材脚本拆成纯计算 `compute*()`、页面 `render()` 和轻量 `smoke()`。")
            st.markdown("**图中每个元素代表什么** 横轴通常是步骤、层或序列位置；纵轴是响应、损失、内存或健康指标。")
            st.markdown("**颜色/亮度/方向/速度代表什么** 深色通常代表更强响应；曲线上升/下降代表训练或复杂度趋势。")
            st.markdown("**用户应该调哪个参数** " + "、".join(name for name, _ in spec.controls))
            st.markdown("**为什么会这样** " + (spec.observations[0] if spec.observations else spec.summary))
            st.markdown("**常见误区** " + (spec.misconceptions[0] if spec.misconceptions else "只看单一指标会误判训练状态。"))
            st.markdown("**工程用途** " + (spec.engineering[0] if spec.engineering else "用于快速定位训练和模型设计问题。"))

        with st.expander("控制台输出", expanded=False):
            st.code(str(data.get("log", "")), language="text")
    except Exception as exc:
        render_module_error(module_path, exc)


def default_sidebar_controls(st: Any, spec: LegacyLessonSpec) -> dict[str, object]:
    params: dict[str, object] = {}
    for name, value in spec.controls:
        key = name.lower().replace(" ", "_").replace("-", "_")
        if isinstance(value, bool):
            params[key] = st.checkbox(name, value=value)
        elif isinstance(value, int):
            params[key] = st.slider(name, 1, max(2, value * 4), value)
        elif isinstance(value, float):
            params[key] = st.slider(name, 0.0, max(1.0, value * 4), value, 0.01)
        elif isinstance(value, (list, tuple)) and value:
            params[key] = st.selectbox(name, list(value), index=0)
    params.setdefault("seed", int(st.number_input("随机种子", 0, 9999, 42, 1)))
    return params


def attention_memory(seq_len: int, heads: int, bytes_per_value: int = 2) -> float:
    return seq_len * seq_len * heads * bytes_per_value / 1024 / 1024


def parameter_count_linear(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def safe_ratio(numerator: float, denominator: float) -> float:
    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator
