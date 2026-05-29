MODULE_TITLE = "特征图可视化"
MODULE_SUMMARY = "观察卷积层如何把边缘、纹理和局部形状逐步变成可分类的特征。"
MODULE_TAGS = ["CNN", "特征图", "可视化", "调试"]
MODULE_RELATED_TOPICS = ["卷积直觉", "经典 CNN 架构", "Grad-CAM 可视化", "梯度监控"]
PRACTICE_TARGET = "part6_universal_framework/neural_network_playground?example=cnn"

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


class TinyFeatureCNN(nn.Module):
    """A deterministic CNN used only for feature-map teaching."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 10, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(10, 12, 3, padding=1)
        self.relu3 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {"input": x}
        x = self.relu1(self.conv1(x))
        outputs["conv1"] = x
        x = self.relu2(self.conv2(x))
        outputs["conv2"] = x
        x = self.pool(x)
        outputs["pool"] = x
        x = self.relu3(self.conv3(x))
        outputs["conv3"] = x
        return outputs


def make_input_image(pattern: str, size: int = 64, noise: float = 0.04, seed: int = 7) -> np.ndarray:
    """Create a small synthetic image so the feature maps are reproducible."""

    size = clamp_int(size, 24, 128, "输入尺寸")
    noise = clamp_float(noise, 0.0, 0.35, "噪声强度")
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]

    if pattern == "几何图形":
        image[size // 8 : size // 2, size // 8 : size // 2] = 0.95
        circle = (yy - size * 0.68) ** 2 + (xx - size * 0.68) ** 2 < (size * 0.17) ** 2
        image[circle] = 0.75
        image[size // 3 : size // 3 + 3, :] = 0.55
        image[:, size // 2 : size // 2 + 3] = 0.65
    elif pattern == "斜线纹理":
        image[((xx + yy) % 14) < 5] = 0.85
        image[((xx - yy) % 19) < 3] = 0.45
    elif pattern == "中心亮斑":
        distance = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) ** 0.5
        image = np.exp(-(distance**2) / (2 * (size * 0.18) ** 2)).astype(np.float32)
    else:
        raise ValueError(f"未知输入模式：{pattern}")

    image += rng.normal(0.0, noise, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def _feature_stats(features: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer, tensor in features.items():
        values = tensor.detach().cpu()
        rows.append(
            {
                "层": layer,
                "shape": tuple(values.shape),
                "均值": round(float(values.mean()), 4),
                "标准差": round(float(values.std()), 4),
                "稀疏率(≈0)": round(float((values.abs() < 1e-5).float().mean()), 4),
            }
        )
    return rows


def _plot_feature_grid(features: torch.Tensor, title: str, max_channels: int) -> plt.Figure:
    channels = clamp_int(max_channels, 1, min(16, features.shape[1]), "最多通道数")
    maps = features[0, :channels].detach().cpu().numpy()
    cols = min(4, channels)
    rows = int(np.ceil(channels / cols))
    with safe_mpl_figure(figsize=(cols * 2.4, rows * 2.2)) as fig:
        axes = fig.subplots(rows, cols, squeeze=False)
        for idx, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if idx >= channels:
                continue
            data = maps[idx]
            ax.imshow(data, cmap="viridis")
            ax.set_title(f"通道 {idx}\nmean={data.mean():.2f}", fontsize=9)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_single_channel(features: torch.Tensor, layer: str, channel: int) -> plt.Figure:
    safe_channel = clamp_int(channel, 0, features.shape[1] - 1, "通道编号")
    data = features[0, safe_channel].detach().cpu().numpy()
    with safe_mpl_figure(figsize=(5.5, 4.8)) as fig:
        ax = fig.subplots()
        im = ax.imshow(data, cmap="magma")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{layer} 第 {safe_channel} 个通道响应", fontsize=12, fontweight="bold")
        ax.set_xlabel("宽度位置")
        ax.set_ylabel("高度位置")
        fig.tight_layout()
        return fig


def _plot_input(image: np.ndarray) -> plt.Figure:
    with safe_mpl_figure(figsize=(4.8, 4.2)) as fig:
        ax = fig.subplots()
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title("输入图像", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        return fig


def compute_feature_maps(
    pattern: str = "几何图形",
    observed_layer: str = "conv2",
    max_channels: int = 8,
    channel: int = 0,
    noise: float = 0.04,
    seed: int = 7,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute feature maps without touching Streamlit."""

    observed_layer = observed_layer if observed_layer in {"conv1", "conv2", "conv3"} else "conv2"
    artifacts: list[Path] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        image = make_input_image(pattern, noise=noise, seed=seed)
        model = TinyFeatureCNN().eval()
        x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            features = model(x)
        stats = _feature_stats(features)
        print(f"输入模式：{pattern}")
        print(f"观察层：{observed_layer}")
        print("读图提示：浅层通常响应边缘和方向，深层更稀疏、更像局部结构探测器。")
        for row in stats:
            print(f"{row['层']}: shape={row['shape']}, mean={row['均值']}, sparsity={row['稀疏率(≈0)']}")

    input_fig = _plot_input(image)
    grid_fig = _plot_feature_grid(features[observed_layer], f"{observed_layer} 特征图网格", max_channels)
    channel_fig = _plot_single_channel(features[observed_layer], observed_layer, channel)
    figures = [
        ("feature_input.png", input_fig),
        (f"feature_grid_{observed_layer}.png", grid_fig),
        (f"feature_channel_{observed_layer}.png", channel_fig),
    ]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {
        "log": log_buffer.getvalue(),
        "figures": figures,
        "artifacts": artifacts,
        "stats": stats,
        "channel_count": int(features[observed_layer].shape[1]),
    }


def _go_to_playground() -> None:
    import streamlit as st

    st.query_params["module"] = "part6_universal_framework/neural_network_playground"
    st.query_params["example"] = "cnn"
    st.rerun()


def render() -> None:
    """Render the refactored feature-map lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        st.info("特征图不是原图的复制，而是某个卷积通道对局部模式的响应强度。亮区表示这个通道在该位置更兴奋。")

        with st.sidebar:
            pattern = st.selectbox("输入模式", ["几何图形", "斜线纹理", "中心亮斑"], index=0)
            observed_layer = st.selectbox("观察层", ["conv1", "conv2", "conv3"], index=1)
            max_channels = st.slider("最多显示通道数", 1, 16, 8)
            channel = st.slider("单通道编号", 0, 15, 0)
            noise = st.slider("噪声强度", 0.0, 0.35, 0.04, 0.01)
            seed = st.number_input("随机种子", min_value=0, max_value=9999, value=7, step=1)
            if st.button("去实战：CNN 构建器", width="stretch"):
                _go_to_playground()

        data = compute_feature_maps(pattern, observed_layer, max_channels, channel, noise, int(seed), save_artifacts=True)
        if channel >= data["channel_count"]:
            st.warning(f"当前层只有 {data['channel_count']} 个通道，单通道编号已自动夹到安全范围。")

        left, right = st.columns([0.42, 0.58])
        with left:
            st.subheader("图怎么看")
            st.markdown(
                """
                - `conv1` 通常看边缘、方向和亮度突变。
                - `conv2` 会组合多个浅层边缘，开始形成角点、纹理和局部形状。
                - `conv3` 更稀疏，亮区更像“某种结构出现了”的信号。
                """
            )
            st.dataframe(data["stats"], width="stretch")
        with right:
            for title, (_, fig) in zip(("输入图像", "特征图网格", "单通道热力图"), data["figures"]):
                st.subheader(title)
                st.pyplot(fig, clear_figure=False)

        with st.expander("控制台输出与工程解释", expanded=False):
            st.code(str(data["log"]), language="text")
            st.markdown(
                """
                工程经验：如果所有层都像噪声，先检查输入归一化和模型是否训练过；如果深层全部为 0，
                再查 ReLU 死亡、学习率过大或 BatchNorm 统计异常。
                """
            )
    except Exception as exc:
        render_module_error("part2_cnn/02_feature_maps.py", exc)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_feature_maps(max_channels=4, channel=0, save_artifacts=False)
    return bool(data["figures"]) and bool(data["stats"])


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_feature_maps))
