"""
Neural network Lego factory / central playground.

Run:
    streamlit run part6_universal_framework/neural_network_playground.py
or:
    python main.py part6/neural_network_playground
"""

from __future__ import annotations

import json
import traceback
from typing import Any
from urllib.parse import quote

try:
    import streamlit as st
except ModuleNotFoundError:  # quality_check loads this file without streamlit
    st = None  # type: ignore[assignment]

try:
    from components.visual_system import render_visual_system
except (ModuleNotFoundError, ImportError):
    render_visual_system = None  # type: ignore[assignment,misc]

from components.playground_core import (
    COMPONENT_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
    PRESETS,
    ShapeStep,
    clone_layers,
    format_shape,
    generate_code,
    infer_layer_shape,
    infer_shapes,
    parse_shape,
)
from components.playground_charts import make_attention_heatmap
from components.playground_charts import make_cnn_feature_map
from components.playground_charts import make_gradient_flow_chart
from components.playground_charts import make_loss_curve
from components.playground_charts import make_update_ratio_chart
from components.playground_forms import render_component_docs
from components.playground_forms import render_layer_form
from components.playground_forms import render_layer_list
from components.playground_forms import render_optimizer_controls
from components.playground_forms import render_preset_controls
from components.playground_forms import render_project_io
from components.playground_forms import render_shape_table
from components.playground_state import default_state
from components.playground_state import ensure_query_preset_loaded
from components.playground_state import clear_layers
from components.playground_state import clear_training_history
from components.playground_state import get_input_shape_text
from components.playground_state import get_layers
from components.playground_state import get_loss
from components.playground_state import get_training_history
from components.playground_state import set_input_shape_text
from components.playground_state import set_training_history

PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

from components.playground_training import PlaygroundTrainingHistory
from components.playground_training import run_playground_training


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def go_home() -> None:
    st.query_params.clear()
    _rerun()


def open_playground(example: str) -> None:
    st.query_params["module"] = PLAYGROUND_TARGET
    st.query_params["example"] = example
    _rerun()


def render_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #fbfcfb 0%, #eef5f2 100%);
            color: #172026;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; }
        .playground-hero {
            border-bottom: 1px solid #d7dde1;
            padding-bottom: 0.9rem;
            margin-bottom: 1rem;
        }
        .playground-hero h1 {
            margin: 0;
            font-size: clamp(2rem, 3vw, 3rem);
            letter-spacing: 0;
        }
        .playground-hero p {
            color: #58646d;
            max-width: 980px;
            line-height: 1.7;
            margin: 0.45rem 0 0;
        }
        .mini-note {
            border-left: 4px solid #0f8b8d;
            background: rgba(255,255,255,0.78);
            border-radius: 0 8px 8px 0;
            padding: 0.7rem 0.9rem;
            line-height: 1.65;
            color: #26343b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def module_url(target: str, **params: str) -> str:
    query = {"module": target}
    query.update(params)
    parts = [f"{quote(str(key), safe='')}={quote(str(value), safe='')}" for key, value in query.items()]
    return "/?" + "&".join(parts)


def render_integration_panel(input_shape: tuple[int, ...] | None, steps: list[ShapeStep]) -> None:
    final_shape = steps[-1].output_shape if steps and steps[-1].ok else None
    st.subheader("训练与章节联动")
    st.markdown(
        """
        <div class="mini-note">
        当前控制台已经能输出结构、shape 和代码。下一步要理解训练行为时，请顺着下面几个入口看：
        损失曲线回答“有没有学会”，梯度监控回答“为什么学不动”，特征图/注意力回答“中间表示在看什么”。
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].link_button("训练过程可视化", module_url("part6_universal_framework/training_demo"), width="stretch")
    cols[1].link_button("梯度监控", module_url("part5_toolbox/02_gradient_monitor"), width="stretch")
    if final_shape and len(final_shape) == 3:
        cols[2].link_button("CNN 特征图", module_url("part2_cnn/02_feature_maps"), width="stretch")
    else:
        cols[2].link_button("训练动态", module_url("part5_toolbox/03_training_dynamics"), width="stretch")
    if final_shape and len(final_shape) == 2:
        cols[3].link_button("注意力机制", module_url("part4_transformer/transformer_models"), width="stretch")
    else:
        cols[3].link_button("超参搜索", module_url("part5_toolbox/04_hyperparam_search"), width="stretch")

    if input_shape and final_shape:
        st.caption(f"当前结构：输入 {format_shape(input_shape)} → 输出 {format_shape(final_shape)}。")


def render_linked_training_panel(
    input_shape: tuple[int, ...] | None,
    layers: list[dict[str, Any]],
    steps: list[ShapeStep],
    loss_name: str,
    optimizer_name: str,
    optimizer_params: dict[str, Any],
) -> None:
    st.subheader("联动训练演示")
    st.markdown(
        """
        <div class="mini-note">
        这里把中央控制台搭出来的结构真正跑一小段训练：损失曲线回答“有没有学到东西”，
        梯度流回答“学习信号有没有传到每一层”，参数更新动画回答“每轮参数到底动了多少”，
        CNN 特征图和注意力热力图则把中间表示拉回到对应章节里看。
        </div>
        """,
        unsafe_allow_html=True,
    )
    if input_shape is None or not steps or any(not step.ok for step in steps):
        st.info("请先修复上方 shape 推导错误；只有结构能真实前向传播，联动训练才会启动。")
        return

    default_lr = float(optimizer_params.get("lr", 0.001))
    left, middle, right = st.columns(3)
    epochs = int(left.slider("联动训练 epoch", min_value=1, max_value=20, value=6, step=1))
    learning_rate = float(
        middle.number_input(
            "联动训练学习率",
            min_value=0.000001,
            max_value=1.0,
            value=max(default_lr, 0.000001),
            step=0.0005,
            format="%.6f",
        )
    )
    seed = int(right.number_input("联动训练随机种子", min_value=0, max_value=9999, value=7, step=1))

    config_key = json.dumps(
        {
            "input_shape": input_shape,
            "layers": layers,
            "loss": loss_name,
            "optimizer": optimizer_name,
            "optimizer_params": optimizer_params,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    if st.button("运行联动演示", width="stretch"):
        try:
            with st.spinner("正在执行真实轻量训练，并收集损失、梯度、特征图和注意力权重..."):
                history = run_playground_training(
                    input_shape,
                    layers,
                    steps,
                    loss_name,
                    optimizer_name,
                    optimizer_params,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    seed=seed,
                )
            set_training_history(history, config_key)
            st.success("联动训练完成。下面的图表已经和当前结构对齐。")
        except Exception as exc:
            clear_training_history()
            st.error(f"联动训练没有跑通：{exc}")
            with st.expander("错误详情", expanded=False):
                st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), language="text")

    history = get_training_history(config_key)
    if history is None:
        st.caption("点击“运行联动演示”后，这里会出现损失、梯度流、参数更新、CNN 特征图和注意力热力图。")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("训练类型", "真实轻量训练")
    metric_cols[1].metric("可视化补充", "教学模拟" if history.attention_is_simulated else "真实权重/激活")
    metric_cols[2].metric("损失函数", history.effective_loss)
    metric_cols[3].metric("最终损失", f"{history.losses[-1]:.4f}")
    for note in history.mode_notes:
        st.caption(f"• {note}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(make_loss_curve(history), width="stretch", config=PLOT_CONFIG)
        st.markdown(
            "> 请观察损失曲线是否整体下降；如果它完全不动，通常说明结构无法表达目标、学习率太小，或某一层把梯度截断了。"
        )
    with chart_right:
        st.plotly_chart(make_update_ratio_chart(history), width="stretch", config=PLOT_CONFIG)
        frame = st.slider(
            "参数更新动画帧",
            min_value=1,
            max_value=len(history.epochs),
            value=len(history.epochs),
            step=1,
        )
        ratio = history.update_ratios[frame - 1]
        st.progress(min(100, max(0, int(ratio * 20000))))
        st.markdown(
            f"> 第 {frame} 轮的真实参数更新幅度比是 **{ratio * 100:.4f}%**。如果这条柱长期接近 0，模型几乎没动；如果突然很高，学习率可能过激。"
        )

    st.plotly_chart(make_gradient_flow_chart(history), width="stretch", config=PLOT_CONFIG)
    st.markdown(
        "> 梯度流里每条线对应一层。前面层长期接近 0 是梯度消失的信号；某一层突然暴涨，则要检查学习率、归一化和残差连接。"
    )

    visual_left, visual_right = st.columns(2)
    with visual_left:
        if history.cnn_feature_maps is not None:
            st.plotly_chart(make_cnn_feature_map(history.cnn_feature_maps), width="stretch", config=PLOT_CONFIG)
            st.markdown(
                "> CNN 特征图的亮区表示该卷积核强烈响应的位置。请切换到 CNN 预设再运行，观察浅层卷积通常更像边缘、纹理和方向探测器。"
            )
        else:
            st.info("当前结构没有 Conv2d 层，所以不会生成 CNN 特征图。加载 CNN 预设后，这里会显示第一层卷积的真实激活。")
    with visual_right:
        if history.attention_heatmap is not None:
            st.plotly_chart(make_attention_heatmap(history), width="stretch", config=PLOT_CONFIG)
            st.markdown(
                "> 热力图每一行是一个 query token，每一列是被看的 key token。颜色越深，表示这一行 token 在汇聚信息时越依赖那一列 token。"
            )
        else:
            st.info("当前结构没有注意力或 Transformer 组件，所以不会生成注意力热力图。加载 Transformer 预设后再运行即可观察。")



def render_app() -> None:
    st.set_page_config(page_title="神经网络乐高工厂", layout="wide", initial_sidebar_state="auto")
    render_visual_system()
    render_style()
    default_state()
    ensure_query_preset_loaded()

    st.markdown(
        """
        <div class="playground-hero">
          <h1>神经网络乐高工厂 / 中央控制台</h1>
          <p>用基础组件搭网络，实时检查形状连接，并生成可运行的 PyTorch 模型代码。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.8, 0.2])
    with top_left:
        render_preset_controls()
    with top_right:
        st.write("")
        st.write("")
        if st.button("返回主界面", width="stretch"):
            go_home()

    builder_col, result_col = st.columns([0.95, 1.05], gap="large")

    with builder_col:
        st.subheader("模型构建器")
        input_shape_text = st.text_input(
            "输入形状（不含 batch 维）",
            value=get_input_shape_text(),
            help="例如 MNIST 图像为 (1, 28, 28)，展平后的向量为 (784,)，Transformer token 表示可写 (16, 64)。",
        )
        set_input_shape_text(input_shape_text)
        render_layer_form()
        st.divider()
        st.subheader("已添加层")
        render_layer_list()
        if st.button("清空层", width="stretch", disabled=not get_layers()):
            clear_layers()
            _rerun()
        st.divider()
        render_project_io()

    with result_col:
        st.subheader("训练配置")
        optimizer_name, optimizer_params = render_optimizer_controls()

        st.divider()
        st.subheader("形状推导")
        try:
            input_shape = parse_shape(get_input_shape_text())
            steps = infer_shapes(input_shape, get_layers())
            render_shape_table(steps)
        except Exception as exc:
            input_shape = (1, 28, 28)
            steps = []
            st.error(f"输入形状无法解析：{exc}")

        st.divider()
        st.subheader("PyTorch 代码")
        if steps and all(step.ok for step in steps):
            code = generate_code(
                input_shape,
                get_layers(),
                steps,
                get_loss(),
                optimizer_name,
                optimizer_params,
            )
            st.code(code, language="python")
        else:
            st.warning("修复形状错误后会生成完整 PyTorch 代码。")

        st.divider()
        render_integration_panel(input_shape if steps else None, steps)
        render_linked_training_panel(
            input_shape if steps else None,
            get_layers(),
            steps,
            get_loss(),
            optimizer_name,
            optimizer_params,
        )

    with st.expander("组件注册表", expanded=False):
        render_component_docs()


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _should_render_app() -> bool:
    return __name__ == "__main__" or _running_under_streamlit()


if _should_render_app():
    try:
        render_app()
    except Exception as error:
        try:
            st.error("中央控制台暂时无法运行。")
            st.warning("请返回主界面重新进入，或检查最近一次组件参数修改。")
            if st.button("返回主界面"):
                go_home()
            with st.expander("错误详情", expanded=False):
                st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)), language="text")
        except Exception:
            raise

