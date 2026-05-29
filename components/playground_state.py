"""Session state and JSON project persistence for the neural network playground."""

from __future__ import annotations

import copy
import json
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # quality checks can import this module without Streamlit.
    st = None  # type: ignore[assignment]

from components.playground_core import COMPONENT_REGISTRY
from components.playground_core import LOSS_REGISTRY
from components.playground_core import OPTIMIZER_REGISTRY
from components.playground_core import PRESETS
from components.playground_core import clone_layers
from components.playground_core import format_shape
from components.playground_core import parse_shape


def _state(session_state: Any | None = None) -> Any:
    if session_state is not None:
        return session_state
    if st is None:
        raise RuntimeError("Streamlit is required when session_state is not provided.")
    return st.session_state


def _query_params(query_params: Any | None = None) -> Any:
    if query_params is not None:
        return query_params
    if st is None:
        raise RuntimeError("Streamlit is required when query_params is not provided.")
    return st.query_params


def default_state(session_state: Any | None = None) -> None:
    state = _state(session_state)
    if "playground_layers" not in state:
        state["playground_layers"] = clone_layers(PRESETS["mlp"]["layers"])
    if "playground_input_shape" not in state:
        state["playground_input_shape"] = format_shape(PRESETS["mlp"]["input_shape"])
    if "playground_loss" not in state:
        state["playground_loss"] = "CrossEntropy"
    if "playground_optimizer" not in state:
        state["playground_optimizer"] = "Adam"
    if "playground_optimizer_params" not in state:
        state["playground_optimizer_params"] = {
            name: copy.deepcopy(item["params"]) for name, item in OPTIMIZER_REGISTRY.items()
        }
    if "playground_loaded_example" not in state:
        state["playground_loaded_example"] = ""
    if "playground_project_name" not in state:
        state["playground_project_name"] = "我的神经网络结构"
    if "playground_last_import_error" not in state:
        state["playground_last_import_error"] = ""
    if "playground_training_history" not in state:
        state["playground_training_history"] = None
    if "playground_training_key" not in state:
        state["playground_training_key"] = ""


def load_preset(key: str, session_state: Any | None = None) -> None:
    state = _state(session_state)
    preset = PRESETS[key]
    state["playground_layers"] = clone_layers(preset["layers"])
    state["playground_input_shape"] = format_shape(preset["input_shape"])
    state["playground_loss"] = preset["loss"]
    state["playground_optimizer"] = preset["optimizer"]
    state["playground_loaded_example"] = key
    state["playground_project_name"] = preset["title"]
    state["playground_training_history"] = None
    state["playground_training_key"] = ""


def query_example(query_params: Any | None = None) -> str:
    params = _query_params(query_params)
    value = params.get("example", "")
    if isinstance(value, list):
        value = value[0] if value else ""
    value = str(value).strip().lower()
    return value if value in PRESETS else ""


def ensure_query_preset_loaded(
    session_state: Any | None = None,
    query_params: Any | None = None,
) -> None:
    state = _state(session_state)
    example = query_example(query_params)
    if example and state.get("playground_loaded_example", "") != example:
        load_preset(example, state)


def project_config(session_state: Any | None = None) -> dict[str, Any]:
    state = _state(session_state)
    return {
        "version": 2,
        "name": state["playground_project_name"],
        "input_shape": state["playground_input_shape"],
        "loss": state["playground_loss"],
        "optimizer": state["playground_optimizer"],
        "optimizer_params": copy.deepcopy(state["playground_optimizer_params"]),
        "layers": clone_layers(state["playground_layers"]),
    }


def export_project_config(session_state: Any | None = None) -> str:
    """Serialize the current playground model to a stable JSON document."""

    return json.dumps(project_config(session_state), ensure_ascii=False, indent=2)


def get_input_shape_text(session_state: Any | None = None) -> str:
    return str(_state(session_state)["playground_input_shape"])


def set_input_shape_text(value: str, session_state: Any | None = None) -> None:
    _state(session_state)["playground_input_shape"] = value


def get_layers(session_state: Any | None = None) -> list[dict[str, Any]]:
    return _state(session_state)["playground_layers"]


def get_loss(session_state: Any | None = None) -> str:
    return str(_state(session_state)["playground_loss"])


def clear_layers(session_state: Any | None = None) -> None:
    state = _state(session_state)
    state["playground_layers"] = []
    state["playground_loaded_example"] = ""
    clear_training_history(state)


def set_training_history(history: Any, config_key: str, session_state: Any | None = None) -> None:
    state = _state(session_state)
    state["playground_training_history"] = history
    state["playground_training_key"] = config_key


def get_training_history(config_key: str, session_state: Any | None = None) -> Any | None:
    state = _state(session_state)
    if state.get("playground_training_key", "") != config_key:
        return None
    return state.get("playground_training_history")


def clear_training_history(session_state: Any | None = None) -> None:
    state = _state(session_state)
    state["playground_training_history"] = None
    state["playground_training_key"] = ""


def import_project_config(raw_json: str, session_state: Any | None = None) -> None:
    """Load a saved playground model config after validating public fields."""

    state = _state(session_state)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 无法解析：{exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("配置文件必须是 JSON object。")
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError("配置缺少 layers 数组。")

    normalized_layers: list[dict[str, Any]] = []
    for index, layer in enumerate(layers, 1):
        if not isinstance(layer, dict):
            raise ValueError(f"第 {index} 层不是 object。")
        component = layer.get("component")
        if component not in COMPONENT_REGISTRY:
            raise ValueError(f"第 {index} 层组件 {component!r} 不在注册表中。")
        params = layer.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"第 {index} 层 params 必须是 object。")
        defaults = COMPONENT_REGISTRY[component]["params"]
        merged = copy.deepcopy(defaults)
        for key, value in params.items():
            if key in defaults:
                merged[key] = value
        normalized_layers.append({"component": component, "params": merged})

    shape_text = str(payload.get("input_shape", "(1, 28, 28)"))
    parse_shape(shape_text)
    loss = str(payload.get("loss", "CrossEntropy"))
    optimizer = str(payload.get("optimizer", "Adam"))
    if loss not in LOSS_REGISTRY:
        raise ValueError(f"未知损失函数：{loss}")
    if optimizer not in OPTIMIZER_REGISTRY:
        raise ValueError(f"未知优化器：{optimizer}")

    state["playground_project_name"] = str(payload.get("name", "导入的神经网络结构"))[:80]
    state["playground_input_shape"] = shape_text
    state["playground_layers"] = normalized_layers
    state["playground_loss"] = loss
    state["playground_optimizer"] = optimizer
    optimizer_params = payload.get("optimizer_params")
    if isinstance(optimizer_params, dict):
        current = copy.deepcopy(state["playground_optimizer_params"])
        for name, params in optimizer_params.items():
            if name in current and isinstance(params, dict):
                current[name].update(params)
        state["playground_optimizer_params"] = current
    state["playground_loaded_example"] = ""
    state["playground_training_history"] = None
    state["playground_training_key"] = ""
