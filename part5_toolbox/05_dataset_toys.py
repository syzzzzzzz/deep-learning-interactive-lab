MODULE_TITLE = "玩具数据集"
MODULE_SUMMARY = "用可控二维数据集快速观察模型边界、噪声、样本量和欠拟合/过拟合。"
MODULE_TAGS = ["数据", "玩具实验", "决策边界", "诊断"]
MODULE_RELATED_TOPICS = ["经典机器学习", "数据与训练", "训练动态", "超参搜索"]
PRACTICE_TARGET = "part6_universal_framework/training_demo"

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


def make_toy_dataset(dataset: str, samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic 2D toy dataset."""

    samples = clamp_int(samples, 80, 1200, "样本数")
    noise = clamp_float(noise, 0.0, 0.6, "噪声强度")
    if dataset == "双月":
        return make_moons(n_samples=samples, noise=noise, random_state=seed)
    if dataset == "同心圆":
        return make_circles(n_samples=samples, noise=noise, factor=0.45, random_state=seed)
    if dataset == "高斯团":
        return make_blobs(n_samples=samples, centers=3, cluster_std=0.75 + noise * 2, random_state=seed)
    raise ValueError(f"未知数据集：{dataset}")


def build_classifier(model_name: str, regularization: float, max_depth: int) -> object:
    """Build a small classifier for decision-boundary comparison."""

    regularization = clamp_float(regularization, 0.01, 20.0, "正则化强度")
    max_depth = clamp_int(max_depth, 1, 12, "最大树深度")
    if model_name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=500))
    if model_name == "RBF-SVM":
        return make_pipeline(StandardScaler(), SVC(C=regularization, gamma="scale"))
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=7)
    if model_name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    raise ValueError(f"未知模型：{model_name}")


def _plot_decision_boundary(model: object, x: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
    x_min, x_max = x[:, 0].min() - 0.8, x[:, 0].max() + 0.8
    y_min, y_max = x[:, 1].min() - 0.8, x[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model.predict(grid).reshape(xx.shape)
    with safe_mpl_figure(figsize=(6.2, 5.2)) as fig:
        ax = fig.subplots()
        ax.contourf(xx, yy, pred, alpha=0.25, cmap="Set2")
        scatter = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="Set2", s=18, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)
        ax.legend(*scatter.legend_elements(), title="类别", loc="upper right")
        fig.tight_layout()
        return fig


def _plot_dataset_difficulty(rows: list[dict[str, object]]) -> plt.Figure:
    labels = [str(row["模型"]) for row in rows]
    scores = [float(row["训练准确率"]) for row in rows]
    with safe_mpl_figure(figsize=(7, 3.8)) as fig:
        ax = fig.subplots()
        ax.bar(labels, scores, color=["#3268a8", "#0f8b8d", "#c4871f", "#8f5aa8"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("训练准确率")
        ax.set_title("不同模型在当前数据集上的拟合能力", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig


def compute_dataset_toys(
    dataset: str = "双月",
    model_name: str = "RBF-SVM",
    samples: int = 300,
    noise: float = 0.18,
    regularization: float = 1.0,
    max_depth: int = 4,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute toy dataset diagnostics without Streamlit."""

    artifacts: list[Path] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        x, y = make_toy_dataset(dataset, samples, noise, seed)
        model = build_classifier(model_name, regularization, max_depth)
        model.fit(x, y)
        pred = model.predict(x)
        acc = accuracy_score(y, pred)
        print(f"数据集={dataset}, 样本数={samples}, 噪声={noise:.2f}")
        print(f"主模型={model_name}, 训练准确率={acc:.3f}")
        print("读图提示：边界过直通常欠拟合；边界太碎通常过拟合；噪声越高，完美边界越不可信。")

        rows: list[dict[str, object]] = []
        for candidate in ["LogisticRegression", "RBF-SVM", "DecisionTree", "KNN"]:
            clf = build_classifier(candidate, regularization, max_depth)
            clf.fit(x, y)
            rows.append({"模型": candidate, "训练准确率": round(float(accuracy_score(y, clf.predict(x))), 4)})

    boundary_fig = _plot_decision_boundary(model, x, y, f"{dataset} + {model_name}")
    score_fig = _plot_dataset_difficulty(rows)
    figures = [("toy_decision_boundary.png", boundary_fig), ("toy_model_scores.png", score_fig)]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {"log": log_buffer.getvalue(), "figures": figures, "artifacts": artifacts, "scores": rows}


def _go_to_training_demo() -> None:
    import streamlit as st

    st.query_params["module"] = "part6_universal_framework/training_demo"
    st.rerun()


def render() -> None:
    """Render the refactored toy-dataset lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        st.info("玩具数据集的价值不是“数据简单”，而是可以精确控制难度：先看模型能不能学会最小问题，再把经验迁移到真实项目。")

        with st.sidebar:
            dataset = st.selectbox("选择数据集", ["双月", "同心圆", "高斯团"], index=0)
            model_name = st.selectbox("选择模型", ["LogisticRegression", "RBF-SVM", "DecisionTree", "KNN"], index=1)
            samples = st.slider("样本数", 80, 1200, 300, 20)
            noise = st.slider("噪声强度", 0.0, 0.6, 0.18, 0.02)
            regularization = st.slider("正则化强度 C", 0.01, 20.0, 1.0, 0.05)
            max_depth = st.slider("最大树深度", 1, 12, 4)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：训练过程可视化", width="stretch"):
                _go_to_training_demo()

        data = compute_dataset_toys(dataset, model_name, samples, noise, regularization, max_depth, int(seed), save_artifacts=True)
        left, right = st.columns([0.62, 0.38])
        with left:
            st.subheader("决策边界")
            st.pyplot(data["figures"][0][1], clear_figure=False)
        with right:
            st.subheader("模型对比")
            st.dataframe(data["scores"], width="stretch")
            st.pyplot(data["figures"][1][1], clear_figure=False)
            st.markdown(
                """
                极端值实验：把噪声调高，再把树深度拉满。你会看到训练准确率可能升高，
                但边界开始破碎，这正是过拟合的可视化症状。
                """
            )

        with st.expander("控制台输出与工程解释", expanded=False):
            st.code(str(data["log"]), language="text")
            st.markdown("工程经验：真实项目排查模型前，先在小数据、低噪声、可视化边界上做 sanity check；如果玩具任务都学不会，优先查数据、标签、损失和优化器。")
    except Exception as exc:
        render_module_error("part5_toolbox/05_dataset_toys.py", exc)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_dataset_toys(samples=120, noise=0.1, save_artifacts=False)
    return bool(data["figures"]) and bool(data["scores"])


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_dataset_toys))
