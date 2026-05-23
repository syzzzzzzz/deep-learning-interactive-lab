"""
Classical machine learning algorithm visualizations.

Run:
    streamlit run part1_foundations/classical_ml.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import animation
from textwrap import dedent
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


st.set_page_config(
    page_title="经典机器学习算法可视化",
    layout="wide",
    initial_sidebar_state="expanded",
)


INK = "#172026"
MUTED = "#58646d"
TEAL = "#0f8b8d"
ROSE = "#c73e5b"
AMBER = "#d99a22"
GREEN = "#477b44"
VIOLET = "#5e4ae3"
BLUE = "#2d6cdf"
GRAY = "#9aa7ad"
PAPER = "#fbfaf6"
CLASS_COLORS = [TEAL, ROSE, AMBER, VIOLET, GREEN, BLUE]


st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(239,245,242,0.96) 100%),
            #fbfaf6;
        color: #172026;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }
    section[data-testid="stSidebar"] {
        background: #eef4f1;
        border-right: 1px solid #d7dde1;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.7rem;
    }
    .hero {
        border-bottom: 1px solid #d7dde1;
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3rem);
        margin: 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #58646d;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 980px;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.74);
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 0.9rem;
        color: #26343b;
        line-height: 1.65;
        margin: 0.4rem 0 0.9rem 0;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.4rem 0 0.9rem 0;
    }
    .mini-cell {
        background: rgba(255,255,255,0.74);
        border: 1px solid #d7dde1;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        min-height: 88px;
    }
    .mini-cell strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #172026;
    }
    .mini-cell span {
        color: #58646d;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    @media (max-width: 900px) {
        .mini-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def segmented(label: str, options: list[str], default: str) -> str:
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default)
        return value or default
    return st.radio(label, options, index=options.index(default), horizontal=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def concept_cards(cards: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="mini-cell"><strong>{title}</strong><span>{text}</span></div>'
        for title, text in cards
    )
    st.markdown(f'<div class="mini-grid">{body}</div>', unsafe_allow_html=True)


def safe_legend(ax: plt.Axes, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc=loc)


def render_learning_map() -> None:
    st.markdown(
        dedent(
            """
            #### 为什么先学经典机器学习？

            经典机器学习像深度学习的“低维剖面图”：它把**损失函数、正则化、决策边界、过拟合、特征空间、距离度量**这些核心思想用更小的模型展示出来。理解这些算法后，再看神经网络就不会只觉得它是一个黑盒，而能看见里面反复出现的同一批原则。

            > 互动：先任选一个算法，拖动右侧参数观察图形变化；再切换到另一个算法。重点比较：哪些模型在画直线，哪些模型在画弯曲边界，哪些模型根本没有标签也能找结构？
            """
        )
    )
    concept_cards(
        [
            ("监督学习", "线性回归、逻辑回归、决策树、SVM、KNN 都从带标签样本学习输入到输出的映射。"),
            ("无监督学习", "K-Means 不看标签，只根据样本之间的距离把数据自动分组。"),
            ("深度学习连接", "神经网络会把这些思想放进更大的可微系统：用梯度优化损失，用正则控制复杂度，用表示学习替代手工特征。"),
        ]
    )


def render_linear_regression_text(lr: float, reg: float, epochs: int) -> None:
    st.markdown(
        dedent(
            f"""
            **教材导读：**线性回归假设目标值可以由特征的加权和解释。生活化地说，它像用“面积、楼层、距离地铁”等因素给房价打分，每个因素有一个权重，权重越大说明这个因素越重要。左图中的散点是真实样本，半透明平面是模型学到的预测规则；右图是训练过程中损失如何下降。

            > 互动：把“学习率”从 {lr:.3f} 调到更小，再把“训练轮数”调大，观察右侧损失曲线是否下降得更慢但更稳。再把“L2 正则化系数”从 {reg:.2f} 调高，观察权重和平面是否更保守。
            """
        )
    )
    st.latex(r"\hat{y}=w_1x_1+w_2x_2+b")
    st.latex(r"L(w,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2+\lambda\|w\|_2^2")
    st.markdown(
        dedent(
            fr"""
            公式中，\(\hat{{y}}\) 是预测值，\(w_1,w_2\) 是特征权重，\(b\) 是偏置，\(\lambda\) 对应页面里的 **L2 正则化系数**。当前设置为学习率 **{lr:.3f}**、正则化 **{reg:.2f}**、训练 **{epochs}** 轮。

            **常见误区：**线性回归不是“只能画一条线”，在二维特征里它画的是平面；MSE 下降也不代表模型一定泛化好，正则化就是为了防止权重过度迎合训练噪声。工程上，它常被用作房价、销量、耗时等连续数值预测的强基线。
            """
        )
    )


def render_logistic_regression_text(class_sep: float, c_value: float) -> None:
    st.markdown(
        dedent(
            fr"""
            **教材导读：**逻辑回归解决的是分类问题。它先像线性回归一样算一个分数 \(z\)，再用 Sigmoid 把分数压成 0 到 1 的概率。左图颜色表示“属于正类”的概率，黑线是概率等于 0.5 的决策边界；右图说明为什么极大的正分数接近 1，极小的负分数接近 0。

            > 互动：把“类别可分程度”从 {class_sep:.2f} 调低，观察两类样本混在一起时边界为什么更难画；再把“正则化强度 C”调低，观察边界是否更保守。
            """
        )
    )
    st.latex(r"p(y=1\mid x)=\sigma(w^Tx+b)=\frac{1}{1+e^{-(w^Tx+b)}}")
    st.latex(r"\text{decision boundary}: \quad w^Tx+b=0")
    st.markdown(
        dedent(
            fr"""
            这里的 \(C\) 是正则化强度的反向写法：**C 越大，模型越愿意贴合训练集；C 越小，正则越强，边界越稳健**。当前类别可分程度为 **{class_sep:.2f}**，C 为 **{c_value:.2f}**。

            **常见误区：**逻辑回归名字里有“回归”，但它通常用于分类；Sigmoid 是非线性的，但二维图中的分类边界仍然是线性的。工程上，它常用于点击率预估、风控二分类、医学阳性概率等需要概率解释的场景。
            """
        )
    )


def render_decision_tree_text(max_depth: int, max_leaf_nodes: int, show_anim: bool) -> None:
    anim_text = "已开启" if show_anim else "未开启"
    st.markdown(
        dedent(
            f"""
            **教材导读：**决策树像一串“如果……那么……”的问题。每个节点选择一个特征阈值，把样本切成更纯的两边；切到叶子节点时，就给这片区域一个类别。左图展示特征空间被切成哪些区域，右图展示树结构中的问题顺序。

            > 互动：把“最大树深度”从 1 慢慢调到 {max_depth} 附近，观察边界如何从粗糙变细碎。再限制“最大叶子节点数”，思考为什么少量叶子会让模型更简单。
            """
        )
    )
    st.latex(r"Gini(D)=1-\sum_{c=1}^{C}p_c^2")
    st.markdown(
        dedent(
            f"""
            Gini 越小表示节点越纯。训练时，树会寻找能最大幅度降低不纯度的切分。当前最大深度是 **{max_depth}**，最大叶子节点数是 **{max_leaf_nodes}**，分裂过程动画 **{anim_text}**。

            **常见误区：**训练准确率高不等于树好；深度过大的树很容易把噪声也记住。工程上，单棵树可解释性强，适合做规则诊断；更强的随机森林、梯度提升树则是在许多树的基础上提升稳定性和精度。CART 决策树体系由 Breiman 等人在 1984 年系统化，是现代树模型的重要源头。
            """
        )
    )


def render_kmeans_text(k: int, radius: float, steps: int, inertia: float) -> None:
    st.markdown(
        dedent(
            f"""
            **教材导读：**K-Means 是无监督聚类算法。它不看“正确标签”，只假设同一簇里的样本应该离同一个中心更近。动画右侧展示两步反复发生：先把点分给最近中心，再把中心移动到本簇样本的平均位置。

            > 互动：把“K 值”从 2 调到 6，观察簇会从合并变成切碎；把“迭代步数”从 1 调到 {steps}，观察中心如何移动。注意：“邻域半径”只是辅助观察圆圈，不改变 K-Means 的训练结果。
            """
        )
    )
    st.latex(r"\min_{\mu_1,\ldots,\mu_K}\sum_{i=1}^{n}\min_{k}\|x_i-\mu_k\|_2^2")
    st.markdown(
        dedent(
            f"""
            目标函数就是让每个点到最近中心的平方距离总和尽量小，这个值在页面里叫 **Inertia**。当前 K 为 **{k}**，邻域半径为 **{radius:.2f}**，最终 Inertia 为 **{inertia:.2f}**。

            **常见误区：**K-Means 找到的是“球形、距离意义上的簇”，不是所有形状都能处理；K 值也不是模型自动知道的，需要结合业务、肘部法或轮廓系数判断。工程上，它常用于用户分群、图像颜色量化、向量粗聚类等场景。该思想可追溯到 MacQueen 1967 年对 k-means 的系统描述。
            """
        )
    )


def render_svm_text(kernel: str, c_value: float, gamma_value: float, support_count: int) -> None:
    st.markdown(
        dedent(
            f"""
            **教材导读：**SVM 的核心不是“随便找一条能分开的线”，而是找一条让两类样本离边界尽可能远的分割面。图中黑圈标出的点是支持向量，它们是最靠近边界、最能决定边界位置的样本。

            > 互动：切换“核函数”观察 linear、poly、rbf 的边界差异；把 C 调大，观察模型是否更努力贴住训练样本；把 gamma 调大，观察 RBF 边界是否更局部、更弯曲。
            """
        )
    )
    st.latex(r"\max \ \text{margin}, \quad \text{with soft penalty controlled by } C")
    st.markdown(
        dedent(
            f"""
            当前核函数是 **{kernel}**，C 为 **{c_value:.2f}**，gamma 为 **{gamma_value:.2f}**，支持向量数为 **{support_count}**。C 越大越不容忍错分；gamma 越大，单个样本影响范围越小，边界更容易局部扭曲。

            **常见误区：**支持向量不是异常点，而是决定间隔的关键样本；核函数也不是“把数据真的画到高维再训练”，而是用核技巧计算高维相似度。工程上，SVM 适合中小规模、特征质量较高的分类任务；在超大数据上训练成本会变重。Cortes 和 Vapnik 在 1995 年提出现代软间隔 SVM。
            """
        )
    )


def render_knn_text(dataset: str, k: int, query_x: float, query_y: float, pred: int) -> None:
    st.markdown(
        dedent(
            f"""
            **教材导读：**KNN 是“近朱者赤，近墨者黑”的算法。它几乎没有训练阶段，预测时才寻找离查询点最近的 K 个样本，然后让这些邻居投票。图中的星标是查询点，黑圈圈住的是参与投票的近邻范围。

            > 互动：拖动“查询点 x1 / x2”，观察星标移动后预测类别如何改变；把 K 从 1 调到 31，观察边界从锯齿状变平滑。思考：为什么 K 太小容易受噪声影响，K 太大又会忽略局部结构？
            """
        )
    )
    st.latex(r"\hat{y}=\operatorname{mode}\{y_i: x_i \in N_K(x)\}")
    st.markdown(
        dedent(
            fr"""
            公式中 \(N_K(x)\) 表示查询点 \(x\) 的 K 个最近邻，mode 表示多数投票。当前数据形状是 **{dataset}**，K 为 **{k}**，查询点为 **({query_x:.2f}, {query_y:.2f})**，预测类别是 **{pred}**。

            **常见误区：**KNN 简单不等于弱；当特征尺度合理、局部结构清晰时，它很直观有效。但它预测时要查邻居，数据大时会慢；特征维度很高时，“距离”也可能失去区分度。工程上常配合标准化、近似最近邻索引和向量检索使用。
            """
        )
    )


def new_figure(figsize: tuple[float, float] = (7.2, 4.8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.22)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#c8d0d5")
    return fig, ax


def render_matplotlib(fig: plt.Figure) -> None:
    try:
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


def styled_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: np.ndarray | None = None) -> None:
    if labels is None:
        ax.scatter(x[:, 0], x[:, 1], s=34, c=TEAL, edgecolor="white", linewidth=0.7, alpha=0.9)
        return
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=34,
            c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            label=f"类别 {int(cls)}",
        )


def make_grid(x: np.ndarray, margin: float = 0.65, points: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = x[:, 0].min() - margin, x[:, 0].max() + margin
    y_min, y_max = x[:, 1].min() - margin, x[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, points), np.linspace(y_min, y_max, points))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_classifier_boundary(
    model,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    *,
    support_vectors: np.ndarray | None = None,
) -> plt.Figure:
    xx, yy, grid = make_grid(x)
    pred = model.predict(grid).reshape(xx.shape)
    fig, ax = new_figure()
    ax.contourf(xx, yy, pred, levels=np.arange(pred.max() + 2) - 0.5, colors=CLASS_COLORS, alpha=0.15)
    styled_scatter(ax, x, y)
    if support_vectors is not None:
        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=118,
            facecolors="none",
            edgecolors=INK,
            linewidths=1.8,
            label="支持向量",
        )
    ax.set_title(title, color=INK, fontsize=13)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def linear_regression_data(seed: int, lr: float, reg: float, epochs: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.4, 2.4, size=(130, 2))
    y = 1.8 * x[:, 0] - 1.15 * x[:, 1] + 0.55 + rng.normal(0, 0.7, size=len(x))
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-8
    x_scaled = (x - x_mean) / x_std

    w = np.zeros(2)
    b = 0.0
    losses: list[float] = []
    path: list[np.ndarray] = []
    for _ in range(epochs):
        pred = x_scaled @ w + b
        err = pred - y
        grad_w = (x_scaled.T @ err) / len(x_scaled) + reg * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        losses.append(float(np.mean(err**2) + 0.5 * reg * np.sum(w**2)))
        path.append(w.copy())

    gx1, gx2 = np.meshgrid(np.linspace(-2.6, 2.6, 38), np.linspace(-2.6, 2.6, 38))
    grid_original = np.c_[gx1.ravel(), gx2.ravel()]
    grid_scaled = (grid_original - x_mean) / x_std
    gz = (grid_scaled @ w + b).reshape(gx1.shape)

    return {
        "x": x,
        "y": y,
        "w": w,
        "b": b,
        "losses": np.array(losses),
        "path": np.array(path),
        "gx1": gx1,
        "gx2": gx2,
        "gz": gz,
        "x_mean": x_mean,
        "x_std": x_std,
    }


def render_linear_regression(seed: int) -> None:
    st.subheader("线性回归：从误差最小化到拟合平面")
    concept_cards(
        [
            ("模型", "用一个平面 y = w1*x1 + w2*x2 + b 解释连续数值。"),
            ("学习率", "控制每次沿梯度方向移动多远；太大容易震荡，太小收敛慢。"),
            ("L2 正则化", "惩罚过大的权重，让拟合平面更平滑、更保守。"),
        ]
    )
    with st.sidebar:
        st.markdown("### 线性回归参数")
        lr = st.slider("学习率", 0.005, 0.30, 0.06, 0.005)
        reg = st.slider("L2 正则化系数", 0.0, 2.0, 0.15, 0.01)
        epochs = st.slider("训练轮数", 5, 220, 90, 5)
    render_linear_regression_text(lr, reg, epochs)

    data = linear_regression_data(seed, lr, reg, epochs)
    losses = np.asarray(data["losses"])
    c1, c2, c3 = st.columns(3)
    c1.metric("最终 MSE + 正则项", f"{losses[-1]:.3f}")
    c2.metric("权重 w1", f"{float(np.asarray(data['w'])[0]):.3f}")
    c3.metric("权重 w2", f"{float(np.asarray(data['w'])[1]):.3f}")

    left, right = st.columns([1.25, 1])
    with left:
        fig = plt.figure(figsize=(7.2, 5.2))
        fig.patch.set_facecolor(PAPER)
        ax = fig.add_subplot(111, projection="3d")
        x = np.asarray(data["x"])
        y = np.asarray(data["y"])
        ax.scatter(x[:, 0], x[:, 1], y, c=TEAL, s=24, alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.plot_surface(
            np.asarray(data["gx1"]),
            np.asarray(data["gx2"]),
            np.asarray(data["gz"]),
            color=ROSE,
            alpha=0.33,
            linewidth=0,
            antialiased=True,
        )
        ax.set_title("二维特征的线性拟合平面", color=INK)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        render_matplotlib(fig)
    with right:
        fig2, ax2 = new_figure((6.0, 3.2))
        ax2.plot(losses, color=ROSE, linewidth=2.4)
        ax2.set_title("训练损失随迭代下降", color=INK)
        ax2.set_xlabel("迭代轮数")
        ax2.set_ylabel("Loss")
        fig2.tight_layout()
        render_matplotlib(fig2)
        note("观察学习率时，重点看损失曲线是否平滑下降；观察正则化时，重点看平面是否被拉得更平、更不极端。")


@st.cache_data(show_spinner=False)
def logistic_data(seed: int, class_sep: float, regularization_c: float) -> dict[str, np.ndarray | float]:
    x, y = make_classification(
        n_samples=260,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=0.06,
        random_state=seed,
    )
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=regularization_c, max_iter=1000, random_state=seed)
    model.fit(x, y)
    return {
        "x": x,
        "y": y,
        "coef": model.coef_[0],
        "intercept": float(model.intercept_[0]),
        "accuracy": float(model.score(x, y)),
    }


def render_logistic_regression(seed: int) -> None:
    st.subheader("逻辑回归：S 型概率曲线和线性分类边界")
    concept_cards(
        [
            ("线性打分", "先计算 z = w*x + b，z 越大越偏向正类。"),
            ("Sigmoid", "把任意实数 z 压到 0 到 1，解释为正类概率。"),
            ("分类边界", "概率等于 0.5 的地方就是 w*x + b = 0。"),
        ]
    )
    with st.sidebar:
        st.markdown("### 逻辑回归参数")
        class_sep = st.slider("类别可分程度", 0.45, 2.4, 1.15, 0.05)
        c_value = st.slider("正则化强度 C", 0.05, 10.0, 1.0, 0.05)
    render_logistic_regression_text(class_sep, c_value)

    data = logistic_data(seed, class_sep, c_value)
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    coef = np.asarray(data["coef"])
    intercept = float(data["intercept"])

    c1, c2, c3 = st.columns(3)
    c1.metric("训练准确率", f"{float(data['accuracy']):.1%}")
    c2.metric("w1", f"{coef[0]:.3f}")
    c3.metric("w2", f"{coef[1]:.3f}")

    left, right = st.columns([1.15, 1])
    with left:
        xx, yy, grid = make_grid(x)
        z = grid @ coef + intercept
        prob = 1.0 / (1.0 + np.exp(-z))
        fig, ax = new_figure()
        ax.contourf(xx, yy, prob.reshape(xx.shape), levels=np.linspace(0, 1, 16), cmap="RdYlBu_r", alpha=0.35)
        ax.contour(xx, yy, prob.reshape(xx.shape), levels=[0.5], colors=INK, linewidths=2.2)
        styled_scatter(ax, x, y)
        ax.set_title("概率热力图与 P(y=1)=0.5 分类边界", color=INK)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        safe_legend(ax)
        fig.tight_layout()
        render_matplotlib(fig)
    with right:
        z_line = np.linspace(-8, 8, 320)
        sigmoid = 1 / (1 + np.exp(-z_line))
        fig2, ax2 = new_figure((6.0, 3.7))
        ax2.plot(z_line, sigmoid, color=ROSE, linewidth=2.8)
        ax2.axhline(0.5, color=INK, linestyle="--", linewidth=1.3)
        ax2.axvline(0, color=INK, linestyle="--", linewidth=1.3)
        ax2.set_title("Sigmoid 把线性打分变成概率", color=INK)
        ax2.set_xlabel("线性打分 z")
        ax2.set_ylabel("P(y=1)")
        fig2.tight_layout()
        render_matplotlib(fig2)
        note("逻辑回归的边界仍然是线性的；非线性来自概率压缩，不来自边界形状。")


@st.cache_data(show_spinner=False)
def tree_data(seed: int, max_depth: int, max_leaf_nodes: int) -> dict[str, np.ndarray | float | DecisionTreeClassifier]:
    x, y = make_moons(n_samples=300, noise=0.24, random_state=seed)
    model = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=seed)
    model.fit(x, y)
    return {
        "x": x,
        "y": y,
        "model": model,
        "accuracy": float(model.score(x, y)),
        "leaves": float(model.get_n_leaves()),
        "depth": float(model.get_depth()),
    }


def tree_animation_html(x: np.ndarray, y: np.ndarray, depths: list[int], seed: int) -> str:
    fig, ax = new_figure((6.2, 4.2))

    def update(frame: int) -> list:
        ax.clear()
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.22)
        for spine in ax.spines.values():
            spine.set_color("#c8d0d5")
        depth = depths[frame]
        model = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(x, y)
        xx, yy, grid = make_grid(x, points=180)
        pred = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], colors=[TEAL, ROSE], alpha=0.17)
        styled_scatter(ax, x, y)
        ax.set_title(f"分裂过程：当前最大深度 = {depth}", color=INK)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(depths), interval=850, blit=False)
    html = anim.to_jshtml()
    plt.close(fig)
    return html


def render_decision_tree(seed: int) -> None:
    st.subheader("决策树：递归切分特征空间")
    concept_cards(
        [
            ("分裂", "每个内部节点选择一个特征阈值，把样本分成更纯的两部分。"),
            ("树深度", "深度越大，边界越碎，越容易贴住训练集噪声。"),
            ("叶子节点", "每片最终区域对应一个预测类别或概率。"),
        ]
    )
    with st.sidebar:
        st.markdown("### 决策树参数")
        max_depth = st.slider("最大树深度", 1, 8, 4, 1)
        max_leaf_nodes = st.slider("最大叶子节点数", 2, 24, 10, 1)
        show_anim = st.toggle("显示分裂过程动画", value=True)
    render_decision_tree_text(max_depth, max_leaf_nodes, show_anim)

    data = tree_data(seed, max_depth, max_leaf_nodes)
    model = data["model"]
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    m1, m2, m3 = st.columns(3)
    m1.metric("训练准确率", f"{float(data['accuracy']):.1%}")
    m2.metric("实际深度", f"{int(data['depth'])}")
    m3.metric("实际叶子数", f"{int(data['leaves'])}")

    left, right = st.columns([1.15, 1])
    with left:
        render_matplotlib(plot_classifier_boundary(model, x, y, "决策树分类边界"))
    with right:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        fig.patch.set_facecolor(PAPER)
        plot_tree(
            model,
            ax=ax,
            max_depth=3,
            filled=True,
            rounded=True,
            impurity=False,
            feature_names=["x1", "x2"],
            class_names=["0", "1"],
            fontsize=8,
        )
        ax.set_title("树结构预览", color=INK)
        fig.tight_layout()
        render_matplotlib(fig)

    if show_anim:
        st.components.v1.html(tree_animation_html(x, y, list(range(1, max_depth + 1)), seed), height=470)
    note("决策树的动画展示了边界从粗到细的过程。深度和叶子数越受限制，边界越简单。")


def kmeans_steps(x: np.ndarray, k: int, seed: int, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    history: list[tuple[np.ndarray, np.ndarray]] = []
    labels = np.zeros(len(x), dtype=int)
    for _ in range(steps):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        history.append((labels.copy(), centers.copy()))
        new_centers = centers.copy()
        for cluster in range(k):
            mask = labels == cluster
            if mask.any():
                new_centers[cluster] = x[mask].mean(axis=0)
        centers = new_centers
    history.append((labels.copy(), centers.copy()))
    return history


@st.cache_data(show_spinner=False)
def kmeans_data(seed: int, k: int, steps: int) -> dict[str, np.ndarray | list[tuple[np.ndarray, np.ndarray]]]:
    x, _ = make_blobs(n_samples=320, centers=max(k, 3), cluster_std=0.68, random_state=seed)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    return {"x": x, "history": kmeans_steps(x, k, seed + 13, steps)}


def kmeans_animation_html(
    x: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
) -> str:
    fig, ax = new_figure((6.2, 4.2))

    def update(frame: int) -> list:
        ax.clear()
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.22)
        for spine in ax.spines.values():
            spine.set_color("#c8d0d5")
        labels, centers = history[frame]
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(
                x[mask, 0],
                x[mask, 1],
                s=28,
                c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
            )
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, c=INK, edgecolor="white", linewidth=1.0)
        for cx, cy in centers:
            circle = plt.Circle((cx, cy), radius, fill=False, color=INK, alpha=0.25, linewidth=1.5)
            ax.add_patch(circle)
        ax.set_xlim(x[:, 0].min() - 0.7, x[:, 0].max() + 0.7)
        ax.set_ylim(x[:, 1].min() - 0.7, x[:, 1].max() + 0.7)
        ax.set_title(f"K-Means 迭代过程：第 {frame + 1} 步", color=INK)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=850, blit=False)
    html = anim.to_jshtml()
    plt.close(fig)
    return html


def render_kmeans(seed: int) -> None:
    st.subheader("K-Means 聚类：分配样本和移动中心")
    concept_cards(
        [
            ("K 值", "预先指定要找几个簇；K 太小会合并结构，K 太大会切碎结构。"),
            ("迭代", "先把样本分配给最近中心，再把中心移动到簇均值。"),
            ("邻域半径", "图中圆圈帮助观察中心附近的覆盖范围，不改变 K-Means 本身。"),
        ]
    )
    with st.sidebar:
        st.markdown("### K-Means 参数")
        k = st.slider("K 值", 2, 6, 3, 1)
        radius = st.slider("邻域半径", 0.2, 2.0, 0.75, 0.05)
        steps = st.slider("迭代步数", 1, 12, 7, 1)

    data = kmeans_data(seed, k, steps)
    x = np.asarray(data["x"])
    history = data["history"]
    labels, centers = history[-1]
    inertia = float(np.sum((x - centers[labels]) ** 2))
    render_kmeans_text(k, radius, steps, inertia)
    m1, m2, m3 = st.columns(3)
    m1.metric("簇数量", str(k))
    m2.metric("最终惯性 Inertia", f"{inertia:.2f}")
    m3.metric("动画帧数", str(len(history)))

    left, right = st.columns([1.1, 1])
    with left:
        fig, ax = new_figure()
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(
                x[mask, 0],
                x[mask, 1],
                s=34,
                c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
                label=f"簇 {int(cls) + 1}",
            )
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=190, c=INK, edgecolor="white", linewidth=1.0, label="中心")
        for cx, cy in centers:
            ax.add_patch(plt.Circle((cx, cy), radius, fill=False, color=INK, alpha=0.25, linewidth=1.6))
        ax.set_title("最终聚类结果与中心邻域", color=INK)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        safe_legend(ax)
        fig.tight_layout()
        render_matplotlib(fig)
    with right:
        st.components.v1.html(kmeans_animation_html(x, history, radius), height=470)
    note("K-Means 没有类别标签，它只根据距离和均值找结构。动画里的中心移动，就是目标函数逐步下降的直观表现。")


@st.cache_data(show_spinner=False)
def svm_data(seed: int, kernel: str, c_value: float, gamma_value: float) -> dict[str, np.ndarray | float | SVC]:
    if kernel == "linear":
        x, y = make_classification(
            n_samples=260,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.15,
            flip_y=0.04,
            random_state=seed,
        )
    elif kernel == "poly":
        x, y = make_moons(n_samples=260, noise=0.18, random_state=seed)
    else:
        x, y = make_circles(n_samples=280, noise=0.08, factor=0.45, random_state=seed)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    model = SVC(kernel=kernel, C=c_value, gamma=gamma_value, degree=3)
    model.fit(x, y)
    return {
        "x": x,
        "y": y,
        "model": model,
        "accuracy": float(model.score(x, y)),
        "support_count": float(model.support_vectors_.shape[0]),
    }


def render_svm(seed: int) -> None:
    st.subheader("SVM：最大间隔、支持向量和核函数")
    concept_cards(
        [
            ("支持向量", "真正决定边界的是离间隔最近的一小部分样本。"),
            ("C 参数", "C 越大越不愿容忍错分，边界更贴训练集。"),
            ("核函数", "用线性、多项式或 RBF 在更高维空间构造可分边界。"),
        ]
    )
    with st.sidebar:
        st.markdown("### SVM 参数")
        kernel_label = st.selectbox("核函数", ["linear", "poly", "rbf"], index=2)
        c_value = st.slider("C", 0.05, 20.0, 2.0, 0.05)
        gamma_value = st.slider("gamma", 0.05, 5.0, 1.0, 0.05)

    data = svm_data(seed, kernel_label, c_value, gamma_value)
    model = data["model"]
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    render_svm_text(kernel_label, c_value, gamma_value, int(data["support_count"]))
    m1, m2, m3 = st.columns(3)
    m1.metric("训练准确率", f"{float(data['accuracy']):.1%}")
    m2.metric("支持向量数", f"{int(data['support_count'])}")
    m3.metric("核函数", kernel_label)

    fig = plot_classifier_boundary(model, x, y, f"SVM 分类边界：kernel = {kernel_label}", support_vectors=model.support_vectors_)
    render_matplotlib(fig)
    note("黑圈标出的样本是支持向量。切换核函数时，注意边界从直线变为弯曲曲线，但目标仍然是找到间隔尽量大的分割。")


@st.cache_data(show_spinner=False)
def knn_data(seed: int, k: int, dataset: str) -> dict[str, np.ndarray | float | KNeighborsClassifier | NearestNeighbors]:
    if dataset == "moons":
        x, y = make_moons(n_samples=280, noise=0.23, random_state=seed)
    else:
        x, y = make_blobs(n_samples=280, centers=3, cluster_std=1.05, random_state=seed)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x, y)
    nn = NearestNeighbors(n_neighbors=k).fit(x)
    return {"x": x, "y": y, "model": model, "nn": nn, "accuracy": float(model.score(x, y))}


def render_knn(seed: int) -> None:
    st.subheader("KNN：由邻居投票形成分类边界")
    concept_cards(
        [
            ("无显式训练", "KNN 主要存下训练样本，预测时再找最近邻。"),
            ("K 值", "K 小边界更敏感，K 大边界更平滑。"),
            ("局部性", "一个点的预测只取决于它附近的样本。"),
        ]
    )
    with st.sidebar:
        st.markdown("### KNN 参数")
        dataset = st.selectbox("数据形状", ["moons", "blobs"], index=0)
        k = st.slider("K 值", 1, 31, 9, 2)
        query_x = st.slider("查询点 x1", -3.0, 3.0, 0.15, 0.05)
        query_y = st.slider("查询点 x2", -3.0, 3.0, 0.10, 0.05)

    data = knn_data(seed, k, dataset)
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    model = data["model"]
    query = np.array([[query_x, query_y]])
    pred = int(model.predict(query)[0])
    distances, indices = data["nn"].kneighbors(query)
    render_knn_text(dataset, k, query_x, query_y, pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("训练准确率", f"{float(data['accuracy']):.1%}")
    m2.metric("查询点预测类别", str(pred))
    m3.metric("第 K 近邻距离", f"{float(distances[0, -1]):.2f}")

    xx, yy, grid = make_grid(x)
    pred_grid = model.predict(grid).reshape(xx.shape)
    fig, ax = new_figure()
    ax.contourf(xx, yy, pred_grid, levels=np.arange(pred_grid.max() + 2) - 0.5, colors=CLASS_COLORS, alpha=0.15)
    styled_scatter(ax, x, y)
    neighbor_points = x[indices[0]]
    ax.scatter(
        neighbor_points[:, 0],
        neighbor_points[:, 1],
        s=132,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.8,
        label=f"{k} 个近邻",
    )
    ax.scatter(query[:, 0], query[:, 1], marker="*", s=230, c=INK, edgecolor="white", linewidth=1.2, label="查询点")
    ax.add_patch(plt.Circle((query_x, query_y), float(distances[0, -1]), fill=False, color=INK, alpha=0.35, linewidth=1.6))
    ax.set_title("KNN 分类边界与查询点近邻", color=INK)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    safe_legend(ax)
    fig.tight_layout()
    render_matplotlib(fig)
    note("拖动查询点可以看到 KNN 的核心机制：星标点被圈内最近的 K 个样本投票决定类别。")


st.markdown(
    """
    <div class="hero">
      <h1>经典机器学习算法可视化</h1>
      <p>用 Streamlit 和 Matplotlib 把线性回归、逻辑回归、决策树、K-Means、SVM、KNN 放在同一个可调实验台里。每个场景都展示模型假设、关键参数和分类或拟合结果。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_learning_map()

with st.sidebar:
    st.header("全局设置")
    seed = int(st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1))
    st.caption("相同随机种子会复现相同数据和模型结果。")
    st.divider()


algorithm = segmented(
    "选择算法",
    ["线性回归", "逻辑回归", "决策树", "K-Means", "SVM", "KNN"],
    "线性回归",
)

if algorithm == "线性回归":
    render_linear_regression(seed)
elif algorithm == "逻辑回归":
    render_logistic_regression(seed)
elif algorithm == "决策树":
    render_decision_tree(seed)
elif algorithm == "K-Means":
    render_kmeans(seed)
elif algorithm == "SVM":
    render_svm(seed)
else:
    render_knn(seed)
