"""Legacy Matplotlib lesson page rendering and artifact explanation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from components.artifact_runtime import (
    artifact_context,
    latest_run_dir as latest_artifact_run_dir,
    run_legacy_script,
)
from components.course_catalog import ModuleInfo


@dataclass(frozen=True)
class LegacyPageDeps:
    project_root: Path
    learning_guides: dict[str, list[tuple[str, str]]]
    css: Callable[[str], str]
    escape_html: Callable[[str], str]
    render_home_button: Callable[[], None]
    render_module_header: Callable[[ModuleInfo], None]
    render_module_card: Callable[[ModuleInfo], str]
    render_module_knowledge_nav: Callable[[ModuleInfo], None]


def latest_run_dir(deps: LegacyPageDeps, module: ModuleInfo) -> Path | None:
    return latest_artifact_run_dir(deps.project_root, module.target)


def read_text_preview(path: Path, max_lines: int = 140) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n\n# ... 省略 {len(lines) - max_lines} 行，完整源码在 {path.name}"
    return preview


def run_legacy_module(deps: LegacyPageDeps, module: ModuleInfo, timeout_seconds: int = 45) -> dict[str, object]:
    result = run_legacy_script(deps.project_root, module.target, module.path, timeout_seconds)
    return {
        "run_dir": result.run_dir,
        "return_code": result.return_code,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def artifact_explanation(module: ModuleInfo, image_path: Path, index: int) -> tuple[str, str, str]:
    name = image_path.stem.lower()
    rules = [
        (("classic_kernel", "kernels", "kernel", "卷积核"), "卷积核在寻找什么", "把它当作一副小眼镜：不同数字模板会放大边缘、模糊、锐化或方向纹理。先比较同一张输入图经过不同卷积核后，哪里变亮、哪里被压低。", "卷积层的第一步不是理解物体，而是稳定地提取局部模式。"),
        (("receptive", "field", "感受野"), "感受野为什么会变大", "看横轴的层数和蓝色区域：层数越深，一个输出位置能回看输入图像的范围越大。", "深层 CNN 能组合更远处的信息，所以后层更容易表达部件和整体形状。"),
        (("pool", "pooling", "池化"), "池化在丢掉什么、保留什么", "观察最大池化和平均池化的区别：最大池化保留最强响应，平均池化保留局部总体趋势。", "池化会牺牲一些精确位置，换来更小的特征图和更强的平移鲁棒性。"),
        (("attention_alignment", "alignment"), "注意力如何形成对齐", "亮格表示当前词更依赖另一个词。先找每一行最亮的格子，看模型把信息从哪里取回来。", "注意力不是魔法解释器，但它能显示信息路由：谁在看谁。"),
        (("attention_vs_no_attention",), "有注意力和无注意力差在哪", "对比两组结果：没有注意力时信息容易被压进单一向量；有注意力时，解码过程可以回看输入的不同位置。", "注意力缓解了长序列信息瓶颈，让模型按需取信息。"),
        (("mha", "multihead", "head_specialization"), "多头注意力为什么要分头", "不同小图代表不同注意力头。看它们是否关注局部、远距离、特殊位置或整体平均。", "多头机制让模型同时用几种视角读同一句话。"),
        (("gradient_descent", "loss_surface"), "梯度下降在往哪里走", "轨迹通常从高损失区域移动到低损失区域。看它是否平稳靠近谷底，还是震荡、绕圈或停住。", "学习率和优化器决定了模型更新是稳步下降，还是跳过好答案。"),
        (("gradient_flow", "grad", "gradient"), "梯度有没有顺利传回去", "看不同层的梯度大小。太接近 0 说明学不动，突然很大说明可能爆炸。", "训练不是只看准确率；梯度健康决定参数能不能被有效更新。"),
        (("decision", "boundary", "xor"), "模型画出了怎样的分界线", "背景或曲线表示模型把空间分成了几类。先看边界有没有贴合数据主结构，再看有没有过分追噪声点。", "非线性网络的价值就在于能画出直线画不出的边界。"),
        (("feature", "activation", "map"), "特征图亮起来代表什么", "亮的区域表示该通道对某种局部模式响应强。多个通道并排看，就是模型的多种视觉探测器。", "从边缘到纹理再到语义，特征会逐层变抽象。"),
        (("confusion", "matrix"), "模型最容易混淆哪些类", "看非对角线哪里颜色深：那里表示真实类别被错判成另一个类别。", "混淆矩阵比总准确率更能告诉你下一步该补数据还是改模型。"),
        (("cam", "gradcam", "heatmap"), "模型决策时盯着哪里", "热区表示对最终判断贡献更大的图像区域。看热区是否落在真正有用的目标上。", "可解释性图不能完全证明因果，但能帮你发现模型是否看偏了。"),
        (("hidden", "state", "rnn", "lstm", "gru"), "序列记忆如何流动", "沿时间方向看隐藏状态或门控值如何变化：哪里保留，哪里遗忘，哪里突然更新。", "RNN 类模型的核心不是单个输入，而是历史信息怎样被压缩和传递。"),
        (("training", "accuracy", "loss", "curve"), "训练过程是否健康", "损失应整体下降，验证指标不应和训练指标越拉越开。震荡、发散、早早停住都值得检查。", "训练曲线是调参时最先看的仪表盘。"),
    ]
    for keywords, title, body, takeaway in rules:
        if any(keyword in name for keyword in keywords):
            return title, body, takeaway

    if module.part_key == "part2":
        return ("这张 CNN 图在说明什么", "先看颜色或亮度最强的区域，再对照标题判断它是在展示卷积、池化、特征图还是架构结构。", "视觉模型通常先提局部纹理，再把局部模式组合成更高层的形状。")
    if module.part_key == "part3":
        return ("这张序列图在说明什么", "按时间顺序从左到右看，重点观察信息在哪里被保留、遗忘或重新加权。", "序列模型的难点是让早期信息在后面仍然可用。")
    if module.part_key == "part4":
        return ("这张 Transformer 图在说明什么", "如果是热图，就看每行最亮的格子；如果是结构图，就看数据从嵌入、注意力、MLP 到输出的路径。", "Transformer 的核心是信息路由：每个位置如何从其他位置取信息。")
    if module.part_key == "part5":
        return ("这张工具图在说明什么", "把它当成训练仪表盘：先找异常峰值、断崖式下降、长期不变或训练验证分离。", "工程调试要靠指标定位问题，而不是只凭最终分数猜。")
    return (
        f"图 {index + 1} 应该怎么看",
        "先读图标题，再看坐标轴、颜色深浅和曲线趋势。不要急着看源码，先问：它想展示哪个变量变化后，结果发生了什么变化？",
        "图像的作用是把抽象概念变成可观察现象；看懂趋势比记住每个数字更重要。",
    )


def render_legacy_results(deps: LegacyPageDeps, module: ModuleInfo, run_dir: Path) -> None:
    import streamlit as st

    e = deps.escape_html
    context = artifact_context(run_dir)
    return_code = context["return_code"]
    timed_out = bool(context["timed_out"])
    stdout = str(context["stdout"])
    stderr = str(context["stderr"])
    images = list(context["images"])

    if return_code == 0:
        st.success(f"运行完成，生成 {len(images)} 张图。")
    elif timed_out:
        st.warning("运行时间过长，已经停止。下面保留了已捕获的输出，方便判断卡在哪里。")
    else:
        st.error("脚本运行失败，但页面已经兜住错误；不会再白屏。")

    if images:
        st.subheader("运行生成的图")
        st.markdown(
            """
            <div class="lesson-note">
              读图顺序：先看标题和坐标轴，再找颜色最深、曲线突变或结构最密集的地方；
              最后回到问题本身，问它是在说明“模型看到了什么”“训练有没有学动”，还是“结构为什么这样设计”。
            </div>
            """,
            unsafe_allow_html=True,
        )
        for index, image_path in enumerate(images):
            title, body, takeaway = artifact_explanation(module, image_path, index)
            st.markdown(f"**{index + 1}. {title}**")
            left, right = st.columns([0.62, 0.38])
            with left:
                st.image(image_path.read_bytes(), caption=image_path.name, width="stretch")
            with right:
                st.markdown(
                    f"""
                    <div class="artifact-note">
                      <strong>这图看什么</strong>
                      <p>{e(body)}</p>
                      <strong>为什么重要</strong>
                      <p>{e(takeaway)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("这次运行没有生成图片。可以看下方输出，很多旧脚本主要通过控制台打印讲解。")

    with st.expander("控制台输出", expanded=return_code != 0):
        st.caption("这里保留脚本原来的逐步讲解、关键数字和公式推导。看图陌生时，先读这段输出通常更容易接上思路。")
        if stdout.strip():
            st.code(stdout[-12000:], language="text")
        else:
            st.caption("没有 stdout 输出。")

    if stderr.strip():
        with st.expander("错误与警告", expanded=True):
            st.code(stderr[-12000:], language="text")


def render_legacy_learning_guide(deps: LegacyPageDeps, module: ModuleInfo) -> None:
    import streamlit as st

    guide = deps.learning_guides.get(module.target)
    if not guide:
        return

    st.subheader("学习导读")
    st.markdown(
        """
        <div class="lesson-note">
          先把下面四张卡读完，再运行脚本。它们会告诉你该看哪张图、调哪些参数、遇到训练异常时先查哪里。
        </div>
        """,
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(guide), 2):
        cols = st.columns(2)
        for col, (title, body) in zip(cols, guide[row_start : row_start + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="artifact-note">
                      <strong>{deps.escape_html(title)}</strong>
                      <p>{deps.escape_html(body)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_legacy_module_page(deps: LegacyPageDeps, module: ModuleInfo) -> None:
    import streamlit as st

    st.set_page_config(
        page_title=f"{module.title} - 深度学习书库",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(deps.css("light"), unsafe_allow_html=True)
    from components.visual_system import render_visual_system

    render_visual_system("light")
    deps.render_home_button()
    deps.render_module_header(module)

    st.markdown(
        """
        <div class="lesson-note">
          这个模块来自早期教材脚本：它原本面向命令行和 Matplotlib 弹窗，不是原生网页。
          现在已改成安全教学页：你可以先读目标和源码，再点击运行；运行失败也只会显示错误，不会让整个网站白屏。
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.38, 0.62])
    with left:
        st.subheader("怎么学")
        st.markdown(
            """
            1. 先读模块目标，知道它想解释什么。
            2. 点击“生成 / 更新运行结果”，把旧脚本输出成网页里的图片和日志。
            3. 如果脚本失败，直接看错误区；这通常是缺数据、下载被阻止，或脚本本身还停留在示例状态。
            """
        )
        st.code(f"python main.py {module.short_target}", language="bash")
        run_clicked = st.button("生成 / 更新运行结果", width="stretch")
    with right:
        st.subheader("模块目标")
        st.write(module.summary)
        st.markdown(deps.render_module_card(module), unsafe_allow_html=True)

    render_legacy_learning_guide(deps, module)

    if run_clicked:
        with st.spinner("正在安全运行旧脚本，并把 Matplotlib 图保存为网页图片..."):
            result = run_legacy_module(deps, module)
        render_legacy_results(deps, module, result["run_dir"])
    else:
        latest = latest_run_dir(deps, module)
        if latest:
            st.subheader("上次运行结果")
            render_legacy_results(deps, module, latest)
        else:
            st.info("还没有运行结果。点击上面的按钮后，这里会显示图像、控制台讲解和错误信息。")

    with st.expander("查看源码片段", expanded=False):
        st.code(read_text_preview(module.path), language="python")
    deps.render_module_knowledge_nav(module)
