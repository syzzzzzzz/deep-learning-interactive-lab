"""
GAN, Autoencoder, and VAE interactive teaching demos.

Run:
    streamlit run part4_transformer/gan_ae.py
or:
    python main.py part4_transformer/gan_ae
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "ink": "#172026",
    "muted": "#596772",
    "line": "#d8dee3",
    "blue": "#3268a8",
    "teal": "#0f8b8d",
    "rose": "#bf3f5b",
    "amber": "#c4871f",
    "green": "#3f7d58",
    "violet": "#7353ba",
    "paper": "#fbfaf6",
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="GAN、自编码器与 VAE",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(241,246,244,0.96) 100%), #fbfaf6;
        color: #172026;
    }
    .block-container { padding-top: 1.25rem; padding-bottom: 2.2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    section[data-testid="stSidebar"] {
        background: #eef4f2;
        border-right: 1px solid #d8dee3;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #d8dee3;
        border-radius: 8px;
        padding: 0.7rem;
    }
    .hero {
        border-bottom: 1px solid #d8dee3;
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 3vw, 3.15rem);
        line-height: 1.1;
        margin: 0;
    }
    .hero p {
        color: #596772;
        max-width: 980px;
        line-height: 1.75;
        margin: 0.45rem 0 0 0;
    }
    .note {
        border-left: 4px solid #0f8b8d;
        background: rgba(255,255,255,0.76);
        border-radius: 0 8px 8px 0;
        padding: 0.72rem 0.9rem;
        color: #26343b;
        line-height: 1.7;
        margin: 0.35rem 0 0.85rem 0;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 0.9rem 0;
    }
    .mini-card {
        background: rgba(255,255,255,0.8);
        border: 1px solid #d8dee3;
        border-radius: 8px;
        padding: 0.74rem 0.82rem;
        min-height: 114px;
    }
    .mini-card strong {
        display: block;
        color: #1f2d35;
        margin-bottom: 0.35rem;
    }
    .mini-card p {
        color: #596772;
        margin: 0;
        line-height: 1.62;
        font-size: 0.92rem;
    }
    @media (max-width: 1000px) {
        .mini-grid { grid-template-columns: 1fr; }
        .mini-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>GAN、自编码器与变分自编码器</h1>
            <p>
            这个页面把生成模型拆成四个可观察过程：GAN 中生成器和判别器的博弈、生成图像从噪声到结构的演变、
            自编码器的压缩重建路径，以及 VAE 如何在潜变量空间里采样并生成。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_concept_cards() -> None:
    cards = [
        ("GAN", "生成器 G 试图骗过判别器 D；判别器 D 试图区分真实样本和生成样本，二者在对抗中共同改进。"),
        ("生成演变", "训练早期像随机噪声，中期出现局部结构，后期才形成稳定样式。这里用可重复的 toy 图像模拟这个过程。"),
        ("自编码器", "编码器把输入压缩到瓶颈向量，解码器从瓶颈向量重建输入；重建误差迫使瓶颈保留关键信息。"),
        ("VAE", "编码器输出均值和方差，采样 z 后再解码；KL 约束让潜空间更连续，适合生成和插值。"),
    ]
    html = '<div class="mini-grid">'
    for title, body in cards:
        html += f'<div class="mini-card"><strong>{title}</strong><p>{body}</p></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def gan_state(frame: int, total_frames: int, difficulty: float) -> dict[str, np.ndarray | float]:
    progress = frame / max(total_frames - 1, 1)
    x = np.linspace(-4, 4, 500)
    real = 0.55 * normal_pdf(x, -0.9, 0.45) + 0.45 * normal_pdf(x, 1.15, 0.62)

    fake_mean_left = -2.65 + 1.75 * (1 - np.exp(-3.2 * progress))
    fake_mean_right = 2.75 - 1.55 * (1 - np.exp(-2.5 * progress))
    fake_std = 1.2 - 0.58 * progress
    mode_balance = 0.85 - 0.28 * progress
    fake = mode_balance * normal_pdf(x, fake_mean_left, fake_std)
    fake += (1 - mode_balance) * normal_pdf(x, fake_mean_right, fake_std * 0.9)

    real = real / np.trapz(real, x)
    fake = fake / np.trapz(fake, x)
    logit = np.log(real + 1e-5) - np.log(fake + 1e-5)
    discriminator = 1 / (1 + np.exp(-difficulty * logit))

    steps = np.arange(total_frames)
    p = steps / max(total_frames - 1, 1)
    d_loss = 1.55 - 0.82 * np.exp(-((p - 0.22) / 0.22) ** 2) + 0.22 * np.sin(8 * p) * np.exp(-1.2 * p)
    g_loss = 1.9 * np.exp(-1.9 * p) + 0.42 + 0.22 * np.sin(10 * p + 0.8) * np.exp(-0.7 * p)
    overlap = np.trapz(np.minimum(real, fake), x)
    return {
        "x": x,
        "real": real,
        "fake": fake,
        "discriminator": discriminator,
        "d_loss": d_loss,
        "g_loss": g_loss,
        "overlap": overlap,
        "progress": progress,
    }


def plot_gan_game(frame: int, total_frames: int, difficulty: float) -> plt.Figure:
    state = gan_state(frame, total_frames, difficulty)
    x = state["x"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    ax.fill_between(x, state["real"], color=PALETTE["teal"], alpha=0.25, label="真实数据 p_data(x)")
    ax.plot(x, state["real"], color=PALETTE["teal"], linewidth=2.5)
    ax.fill_between(x, state["fake"], color=PALETTE["rose"], alpha=0.22, label="生成分布 p_G(x)")
    ax.plot(x, state["fake"], color=PALETTE["rose"], linewidth=2.5)
    ax2 = ax.twinx()
    ax2.plot(x, state["discriminator"], color=PALETTE["amber"], linewidth=2, label="判别器 D(x)")
    ax2.axhline(0.5, color="#8b949e", linestyle="--", linewidth=1)
    ax.set_title("生成器与判别器的当前局面", fontweight="bold")
    ax.set_xlabel("样本空间 x")
    ax.set_ylabel("概率密度")
    ax2.set_ylabel("D(x)=真实概率")
    ax2.set_ylim(-0.04, 1.04)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    steps = np.arange(total_frames)
    ax.plot(steps, state["d_loss"], color=PALETTE["amber"], linewidth=2.2, label="D loss")
    ax.plot(steps, state["g_loss"], color=PALETTE["rose"], linewidth=2.2, label="G loss")
    ax.scatter([frame], [state["d_loss"][frame]], color=PALETTE["amber"], s=60, zorder=5)
    ax.scatter([frame], [state["g_loss"][frame]], color=PALETTE["rose"], s=60, zorder=5)
    ax.set_title("训练损失的交替拉扯", fontweight="bold")
    ax.set_xlabel("训练帧")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03,
        0.08,
        f"分布重叠度: {state['overlap']:.2f}\n进度: {state['progress']:.0%}",
        transform=ax.transAxes,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["line"], alpha=0.85),
    )

    fig.tight_layout()
    return fig


def make_target_pattern(kind: str, size: int = 18) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    if kind == "圆环":
        r = np.sqrt(x**2 + y**2)
        image = np.exp(-((r - 0.52) ** 2) / 0.018)
    elif kind == "斜线":
        image = np.exp(-((y - 0.7 * x) ** 2) / 0.035) * np.exp(-(x**2 + y**2) / 1.8)
    elif kind == "十字":
        image = np.exp(-(x**2) / 0.035) + np.exp(-(y**2) / 0.035)
    else:
        image = (
            np.exp(-((x + 0.25) ** 2 + (y + 0.05) ** 2) / 0.055)
            + np.exp(-((x - 0.28) ** 2 + (y + 0.04) ** 2) / 0.055)
            + np.exp(-((y + 0.43) ** 2) / 0.028) * np.exp(-(x**2) / 0.7)
            + 0.55 * np.exp(-((y - 0.45) ** 2) / 0.035) * np.exp(-(x**2) / 0.6)
        )
    image = image - image.min()
    return image / (image.max() + 1e-9)


def generated_image(seed: int, epoch: int, max_epoch: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = make_target_pattern(kind)
    noise = rng.random(target.shape)
    low_freq = rng.normal(size=(5, 5))
    low_freq = np.kron(low_freq, np.ones((4, 4)))[: target.shape[0], : target.shape[1]]
    low_freq = (low_freq - low_freq.min()) / (low_freq.max() - low_freq.min() + 1e-9)
    progress = epoch / max(max_epoch, 1)
    structure = progress**0.72
    image = structure * target + (1 - structure) * (0.65 * noise + 0.35 * low_freq)
    image += rng.normal(0, 0.24 * (1 - progress), target.shape)
    contrast = 0.75 + 1.6 * progress
    image = 1 / (1 + np.exp(-contrast * (image - image.mean())))
    return np.clip(image, 0, 1)


def plot_generation_grid(epoch: int, max_epoch: int, kind: str) -> plt.Figure:
    fig, axes = plt.subplots(3, 6, figsize=(12.5, 6.4))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(generated_image(i + 11, epoch, max_epoch, kind), cmap="magma", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"z{i + 1}", fontsize=9, color=PALETTE["muted"])
    fig.suptitle(f"生成图像演变: epoch {epoch}/{max_epoch}", fontsize=15, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    return fig


def add_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.8,
        edgecolor="white",
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color="white", fontweight="bold", fontsize=10)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#52616b") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            shrinkA=5,
            shrinkB=5,
        )
    )


def autoencoder_sample(noise_level: float, bottleneck: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean = make_target_pattern("十字", size=20)
    noisy = np.clip(clean + rng.normal(0, noise_level, clean.shape), 0, 1)
    compression = bottleneck / 16
    smooth = clean * (0.72 + 0.25 * compression) + noisy * (0.28 - 0.18 * compression)
    reconstruction = np.clip(smooth + rng.normal(0, 0.025 + 0.09 * (1 - compression), clean.shape), 0, 1)
    code = np.array([noisy.mean(), noisy.std(), noisy[:, :10].mean(), noisy[:10].mean()])[: max(2, bottleneck // 4)]
    return noisy, reconstruction, code


def plot_autoencoder(noise_level: float, bottleneck: int, seed: int) -> plt.Figure:
    noisy, reconstructed, code = autoencoder_sample(noise_level, bottleneck, seed)
    fig = plt.figure(figsize=(13.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    image_axes = [
        fig.add_axes([0.04, 0.31, 0.15, 0.34]),
        fig.add_axes([0.81, 0.31, 0.15, 0.34]),
    ]
    image_axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    image_axes[0].set_title("输入 x", fontsize=11, fontweight="bold")
    image_axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
    image_axes[1].set_title("重建 x_hat", fontsize=11, fontweight="bold")
    for image_ax in image_axes:
        image_ax.set_xticks([])
        image_ax.set_yticks([])

    add_box(ax, (3.0, 4.5), 1.6, 0.72, "Encoder\nLinear + ReLU", PALETTE["blue"])
    add_box(ax, (4.95, 4.15), 1.4, 1.42, f"Bottleneck\nz dim={bottleneck}", PALETTE["violet"])
    add_box(ax, (6.75, 4.5), 1.6, 0.72, "Decoder\nLinear + ReLU", PALETTE["green"])
    add_box(ax, (9.0, 4.5), 1.55, 0.72, "输出层\nSigmoid", PALETTE["teal"])
    add_arrow(ax, (2.3, 4.85), (3.0, 4.85))
    add_arrow(ax, (4.6, 4.85), (4.95, 4.85))
    add_arrow(ax, (6.35, 4.85), (6.75, 4.85))
    add_arrow(ax, (8.35, 4.85), (9.0, 4.85))
    add_arrow(ax, (10.55, 4.85), (11.3, 4.85))

    mse = float(((noisy - reconstructed) ** 2).mean())
    ax.text(5.65, 2.82, "训练目标", ha="center", fontsize=12, fontweight="bold", color=PALETTE["ink"])
    ax.text(5.65, 2.36, "min || x - x_hat ||²", ha="center", fontsize=13, color=PALETTE["rose"])
    ax.text(
        5.65,
        1.65,
        f"当前重建误差约 {mse:.4f}\n瓶颈越窄，越像强制模型只保留主要因素",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=PALETTE["line"], alpha=0.9),
    )

    for i, value in enumerate(code):
        ax.add_patch(plt.Rectangle((5.1 + i * 0.24, 3.45), 0.18, 0.32 + value * 0.55, color=PALETTE["amber"], alpha=0.75))
    ax.text(5.65, 3.24, "压缩后的特征码", ha="center", fontsize=9, color=PALETTE["muted"])

    fig.tight_layout()
    return fig


def plot_vae(mu_x: float, mu_y: float, logvar: float, epsilon_x: float, epsilon_y: float, beta: float) -> plt.Figure:
    sigma = float(np.exp(0.5 * logvar))
    z = np.array([mu_x, mu_y]) + sigma * np.array([epsilon_x, epsilon_y])
    recon_loss = 0.62 + 0.12 * np.linalg.norm(z - np.array([0.8, -0.25]))
    kl = -0.5 * (2 + 2 * logvar - mu_x**2 - mu_y**2 - 2 * np.exp(logvar))
    total = recon_loss + beta * kl

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.1, 0.9]})
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.axhline(0, color=PALETTE["line"], linewidth=1)
    ax.axvline(0, color=PALETTE["line"], linewidth=1)
    ax.add_patch(Ellipse((0, 0), 2, 2, fill=False, edgecolor=PALETTE["line"], linewidth=1.6, linestyle="--"))
    ax.add_patch(Ellipse((mu_x, mu_y), 2 * sigma, 2 * sigma, fill=True, facecolor=PALETTE["blue"], alpha=0.12, edgecolor=PALETTE["blue"], linewidth=2))
    ax.scatter([mu_x], [mu_y], color=PALETTE["blue"], s=80, label="均值 mu")
    ax.scatter([z[0]], [z[1]], color=PALETTE["rose"], s=90, label="采样 z = mu + sigma * eps")
    ax.arrow(mu_x, mu_y, z[0] - mu_x, z[1] - mu_y, color=PALETTE["rose"], length_includes_head=True, head_width=0.08)
    ax.set_title("VAE 潜变量采样", fontweight="bold")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    labels = ["重建项", f"beta * KL\n(beta={beta:.1f})", "总损失"]
    values = [recon_loss, beta * kl, total]
    colors = [PALETTE["teal"], PALETTE["amber"], PALETTE["rose"]]
    ax.bar(labels, values, color=colors, alpha=0.86)
    ax.set_title("VAE 目标函数", fontweight="bold")
    ax.set_ylabel("loss")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.5,
        0.94,
        "L = 重建误差 + beta * KL(q(z|x) || N(0, I))",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["line"], alpha=0.9),
    )
    fig.tight_layout()
    return fig


def render_gan_tab(total_frames: int, difficulty: float, delay: float) -> None:
    st.subheader("1. GAN 的生成器与判别器博弈")
    st.markdown(
        '<div class="note">左图展示真实分布、生成分布和判别器输出。生成分布越贴近真实分布，判别器越难保持 D(x) 接近 0 或 1，最终理想状态是 D(x) 接近 0.5。</div>',
        unsafe_allow_html=True,
    )
    frame = st.slider("训练帧", 0, total_frames - 1, min(18, total_frames - 1), key="gan_frame")
    placeholder = st.empty()
    placeholder.pyplot(plot_gan_game(frame, total_frames, difficulty), clear_figure=True)
    if st.button("播放 GAN 博弈动画"):
        for step in range(frame, total_frames):
            placeholder.pyplot(plot_gan_game(step, total_frames, difficulty), clear_figure=True)
            time.sleep(delay)


def render_generation_tab(max_epoch: int, delay: float) -> None:
    st.subheader("2. 训练过程中生成图像的演变")
    col1, col2 = st.columns([0.24, 0.76])
    with col1:
        kind = st.selectbox("目标结构", ["圆环", "斜线", "十字", "人脸轮廓"])
        epoch = st.slider("epoch", 0, max_epoch, max_epoch // 2, 1)
        st.markdown(
            '<div class="note">这不是调用真实 GAN 训练，而是用噪声到目标结构的可控混合模拟训练视觉效果，便于观察“从噪声到模式”的过程。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        placeholder = st.empty()
        placeholder.pyplot(plot_generation_grid(epoch, max_epoch, kind), clear_figure=True)
        if st.button("播放生成图像演变"):
            for step in range(epoch, max_epoch + 1):
                placeholder.pyplot(plot_generation_grid(step, max_epoch, kind), clear_figure=True)
                time.sleep(delay)


def render_autoencoder_tab(seed: int) -> None:
    st.subheader("3. 自编码器的编码器-解码器结构")
    col1, col2 = st.columns([0.23, 0.77])
    with col1:
        noise = st.slider("输入噪声", 0.0, 0.7, 0.25, 0.05)
        bottleneck = st.select_slider("瓶颈维度", options=[2, 4, 8, 16], value=4)
        st.metric("压缩比", f"{bottleneck}/400")
        st.markdown(
            '<div class="note">自编码器不需要人工标签。它用“自己重建自己”的任务学习低维表示，常用于降噪、压缩、异常检测和预训练。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(plot_autoencoder(noise, bottleneck, seed), clear_figure=True)


def render_vae_tab() -> None:
    st.subheader("4. VAE 的原理演示")
    col1, col2 = st.columns([0.24, 0.76])
    with col1:
        mu_x = st.slider("mu_x", -2.0, 2.0, 0.7, 0.1)
        mu_y = st.slider("mu_y", -2.0, 2.0, -0.4, 0.1)
        logvar = st.slider("log variance", -2.4, 1.2, -0.6, 0.1)
        epsilon_x = st.slider("epsilon_x", -2.0, 2.0, 0.4, 0.1)
        epsilon_y = st.slider("epsilon_y", -2.0, 2.0, -0.8, 0.1)
        beta = st.slider("KL 权重 beta", 0.0, 3.0, 1.0, 0.1)
        st.markdown(
            '<div class="note">VAE 的关键是重参数化技巧：随机性放在 epsilon 上，让 mu 和 sigma 仍然可以通过反向传播学习。</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.pyplot(plot_vae(mu_x, mu_y, logvar, epsilon_x, epsilon_y, beta), clear_figure=True)


def main() -> None:
    render_hero()
    render_concept_cards()

    st.sidebar.header("交互参数")
    total_frames = st.sidebar.slider("GAN 动画帧数", 24, 80, 48, 4)
    max_epoch = st.sidebar.slider("生成图像最大 epoch", 20, 120, 60, 5)
    difficulty = st.sidebar.slider("判别器敏感度", 0.4, 2.4, 1.2, 0.1)
    delay = st.sidebar.slider("动画间隔秒", 0.02, 0.25, 0.06, 0.01)
    seed = st.sidebar.slider("随机种子", 0, 99, 7)

    tabs = st.tabs(["GAN 博弈", "生成演变", "自编码器", "VAE"])
    with tabs[0]:
        render_gan_tab(total_frames, difficulty, delay)
    with tabs[1]:
        render_generation_tab(max_epoch, delay)
    with tabs[2]:
        render_autoencoder_tab(seed)
    with tabs[3]:
        render_vae_tab()


if __name__ == "__main__":
    main()
