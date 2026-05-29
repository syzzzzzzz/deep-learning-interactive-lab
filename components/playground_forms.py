"""Streamlit form adapters for the neural network playground."""

from __future__ import annotations

from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # quality_check may import without pandas.
    pd = None  # type: ignore[assignment]

try:
    import streamlit as st
except ModuleNotFoundError:  # quality_check may import without Streamlit.
    st = None  # type: ignore[assignment]

from components.playground_core import COMPONENT_REGISTRY
from components.playground_core import LOSS_REGISTRY
from components.playground_core import OPTIMIZER_REGISTRY
from components.playground_core import PRESETS
from components.playground_core import ShapeStep
from components.playground_core import format_shape
from components.playground_state import export_project_config
from components.playground_state import import_project_config
from components.playground_state import load_preset

PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
def render_project_io() -> None:
    st.subheader("保存 / 加载结构")
    st.session_state.playground_project_name = st.text_input(
        "结构名称",
        value=st.session_state.playground_project_name,
        max_chars=80,
    )
    json_text = export_project_config()
    st.download_button(
        "下载模型结构 JSON",
        data=json_text.encode("utf-8"),
        file_name="neural_network_playground_config.json",
        mime="application/json",
        width="stretch",
    )
    with st.expander("查看 / 粘贴 JSON", expanded=False):
        edited = st.text_area("模型结构 JSON", value=json_text, height=260, key="playground_config_json")
        if st.button("从上方 JSON 加载", width="stretch"):
            try:
                import_project_config(edited)
                st.session_state.playground_last_import_error = ""
                st.success("模型结构已加载。")
                _rerun()
            except Exception as exc:
                st.session_state.playground_last_import_error = str(exc)
                st.error(f"加载失败：{exc}")
    uploaded = st.file_uploader("上传结构 JSON", type=["json"], accept_multiple_files=False)
    if uploaded is not None and st.button("加载上传文件", width="stretch"):
        try:
            import_project_config(uploaded.getvalue().decode("utf-8"))
            st.session_state.playground_last_import_error = ""
            st.success("上传结构已加载。")
            _rerun()
        except Exception as exc:
            st.session_state.playground_last_import_error = str(exc)
            st.error(f"上传文件加载失败：{exc}")



def render_component_docs() -> None:
    rows = [
        {
            "组件": item["name"],
            "类型": item["type"],
            "默认参数": ", ".join(f"{key}={value}" for key, value in item["params"].items()) or "-",
            "输入规则": item["input_rule"],
            "输出规则": item["output_rule"],
            "相关主题": item["related_topic"],
        }
        for item in COMPONENT_REGISTRY.values()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_layer_form() -> None:
    with st.form("add-layer-form", clear_on_submit=False):
        left, right = st.columns([0.9, 1.4])
        with left:
            component = st.selectbox("组件类型", list(COMPONENT_REGISTRY), index=0)
            info = COMPONENT_REGISTRY[component]
            st.caption(info["description"])
            st.caption(f"PyTorch: `{info['pytorch_code']}`")
        with right:
            params: dict[str, Any] = {}
            defaults = COMPONENT_REGISTRY[component]["params"]
            if not defaults:
                st.info("该组件没有可配置参数。")
            for name, default in defaults.items():
                key = f"new-{component}-{name}"
                if isinstance(default, bool):
                    params[name] = st.checkbox(name, value=default, key=key)
                elif isinstance(default, int):
                    if component == "Flatten" and name == "end_dim":
                        min_value = -1
                    elif name == "padding":
                        min_value = 0
                    else:
                        min_value = 1
                    params[name] = int(st.number_input(name, min_value=min_value, value=default, step=1, key=key))
                elif isinstance(default, float):
                    max_value = 0.9 if name in {"p", "dropout"} else None
                    step = 0.01 if name not in {"eps"} else 1e-6
                    params[name] = float(
                        st.number_input(
                            name,
                            min_value=0.0,
                            max_value=max_value,
                            value=default,
                            step=step,
                            format="%.8f",
                            key=key,
                        )
                    )
                else:
                    params[name] = st.text_input(name, value=str(default), key=key)
        submitted = st.form_submit_button("添加层", width="stretch")
        if submitted:
            st.session_state.playground_layers.append({"component": component, "params": params})
            st.success(f"已添加 {component}")
            _rerun()


def render_layer_list() -> None:
    layers = st.session_state.playground_layers
    if not layers:
        st.info("还没有添加任何层。可以先加载一个预设，或从上方表单添加。")
        return

    for index, layer in enumerate(layers):
        params = layer.get("params", {})
        params_text = ", ".join(f"{name}={value}" for name, value in params.items()) or "无参数"
        cols = st.columns([0.14, 0.52, 0.22, 0.12])
        cols[0].markdown(f"**{index + 1}**")
        cols[1].markdown(f"**{layer['component']}**  `{params_text}`")
        cols[2].caption(COMPONENT_REGISTRY[layer["component"]]["related_topic"])
        if cols[3].button("删除", key=f"delete-layer-{index}", width="stretch"):
            del st.session_state.playground_layers[index]
            _rerun()


def render_optimizer_controls() -> tuple[str, dict[str, Any]]:
    loss = st.selectbox(
        "损失函数",
        list(LOSS_REGISTRY),
        index=list(LOSS_REGISTRY).index(st.session_state.playground_loss),
    )
    st.session_state.playground_loss = loss

    optimizer = st.selectbox(
        "优化器",
        list(OPTIMIZER_REGISTRY),
        index=list(OPTIMIZER_REGISTRY).index(st.session_state.playground_optimizer),
    )
    st.session_state.playground_optimizer = optimizer

    params = st.session_state.playground_optimizer_params[optimizer]
    st.caption(OPTIMIZER_REGISTRY[optimizer]["description"])
    if optimizer == "SGD":
        params["lr"] = float(st.number_input("lr", min_value=0.0, value=float(params["lr"]), step=0.001, format="%.6f"))
        params["momentum"] = float(
            st.number_input("momentum", min_value=0.0, max_value=1.0, value=float(params["momentum"]), step=0.05)
        )
    elif optimizer == "Adam":
        params["lr"] = float(st.number_input("lr", min_value=0.0, value=float(params["lr"]), step=0.0005, format="%.6f"))
        beta1, beta2 = tuple(params["betas"])
        b1_col, b2_col = st.columns(2)
        params["betas"] = (
            float(b1_col.number_input("beta1", min_value=0.0, max_value=0.9999, value=float(beta1), step=0.01)),
            float(b2_col.number_input("beta2", min_value=0.0, max_value=0.9999, value=float(beta2), step=0.001, format="%.4f")),
        )
        params["eps"] = float(st.number_input("eps", min_value=0.0, value=float(params["eps"]), step=1e-8, format="%.10f"))
    return optimizer, params


def render_shape_table(steps: list[ShapeStep]) -> None:
    if not steps:
        st.info("添加层后会在这里显示逐层形状推导。")
        return
    rows = [
        {
            "#": step.index,
            "层": step.layer,
            "输入形状": format_shape(step.input_shape),
            "输出形状": format_shape(step.output_shape),
            "状态": "通过" if step.ok else "错误",
            "说明": step.message,
        }
        for step in steps
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    failed = next((step for step in steps if not step.ok), None)
    if failed:
        st.error(f"第 {failed.index} 层连接非法：{failed.message}")
    else:
        st.success(f"形状推导通过，最终输出形状为 {format_shape(steps[-1].output_shape)}。")


def render_preset_controls() -> None:
    st.subheader("示例模型")
    preset_keys = ("mlp", "cnn", "transformer", "residual_mlp")
    cols = st.columns(len(preset_keys))
    for col, key in zip(cols, preset_keys, strict=True):
        preset = PRESETS[key]
        if col.button(preset["title"], key=f"load-{key}", width="stretch"):
            load_preset(key)
            st.query_params["module"] = PLAYGROUND_TARGET
            st.query_params["example"] = key
            _rerun()
    current = st.session_state.playground_loaded_example
    if current in PRESETS and PRESETS[current].get("note"):
        st.info(PRESETS[current]["note"])


