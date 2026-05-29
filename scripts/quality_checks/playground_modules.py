from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def check_playground_modules(context: QualityCheckContext) -> None:
    """Verify the central playground keeps training logic behind a deep module."""

    page_text = context.read_text(Path("part6_universal_framework/neural_network_playground.py"))
    core_text = context.read_text(Path("components/playground_core.py"))
    state_text = context.read_text(Path("components/playground_state.py"))
    forms_text = context.read_text(Path("components/playground_forms.py"))
    training_text = context.read_text(Path("components/playground_training.py"))
    charts_text = context.read_text(Path("components/playground_charts.py"))
    failures: list[str] = []

    required_state_fragments = [
        "def default_state",
        "def load_preset",
        "def ensure_query_preset_loaded",
        "def project_config",
        "def export_project_config",
        "def import_project_config",
        "def get_input_shape_text",
        "def set_input_shape_text",
        "def get_layers",
        "def get_loss",
        "def clear_layers",
        "def set_training_history",
        "def get_training_history",
        "def clear_training_history",
        "from components.playground_core import PRESETS",
    ]
    for fragment in required_state_fragments:
        if fragment not in state_text:
            failures.append(f"components/playground_state.py 缺少项目状态 Module 契约：{fragment}")

    required_form_fragments = [
        "def render_project_io",
        "def render_component_docs",
        "def render_layer_form",
        "def render_layer_list",
        "def render_optimizer_controls",
        "def render_shape_table",
        "def render_preset_controls",
        "from components.playground_state import export_project_config",
        "from components.playground_state import import_project_config",
    ]
    for fragment in required_form_fragments:
        if fragment not in forms_text:
            failures.append(f"components/playground_forms.py 缺少表单 Adapter 契约：{fragment}")

    forbidden_core_fragments = [
        "st.session_state",
        "st.query_params",
        "def default_state",
        "def load_preset",
        "def ensure_query_preset_loaded",
        "def project_config",
        "def import_project_config",
    ]
    for fragment in forbidden_core_fragments:
        if fragment in core_text:
            failures.append(f"components/playground_core.py 仍混入页面状态实现：{fragment}")

    required_training_fragments = [
        "class PlaygroundTrainingHistory",
        "class PlaygroundBuiltModel",
        "def build_torch_layer",
        "def build_playground_model",
        "def run_playground_training",
        "from components.playground_core import ShapeStep",
        "from components.playground_core import infer_layer_shape",
    ]
    for fragment in required_training_fragments:
        if fragment not in training_text:
            failures.append(f"components/playground_training.py 缺少训练遥测 Module 契约：{fragment}")

    required_chart_fragments = [
        "def make_loss_curve",
        "def make_gradient_flow_chart",
        "def make_update_ratio_chart",
        "def make_cnn_feature_map",
        "def make_attention_heatmap",
        "from components.playground_training import PlaygroundTrainingHistory",
        "from components.playground_training import attention_token_labels",
    ]
    for fragment in required_chart_fragments:
        if fragment not in charts_text:
            failures.append(f"components/playground_charts.py 缺少图表 Adapter 契约：{fragment}")

    forbidden_training_fragments = [
        "plotly.graph_objects",
        "from plotly.subplots import make_subplots",
        "def make_loss_curve",
        "def make_attention_heatmap",
        "def project_config",
        "def export_project_config",
        "def import_project_config",
        "def render_project_io",
        "def render_component_docs",
        "def render_layer_form",
        "def render_layer_list",
        "def render_optimizer_controls",
        "def render_shape_table",
        "def render_preset_controls",
        "st.session_state",
    ]
    for fragment in forbidden_training_fragments:
        if fragment in training_text:
            failures.append(f"components/playground_training.py 仍混入图表 Adapter 实现：{fragment}")

    forbidden_page_fragments = [
        "class PlaygroundTrainingHistory",
        "class PlaygroundBuiltModel",
        "def build_torch_layer",
        "def build_playground_model",
        "def run_playground_training",
        "def make_loss_curve",
        "def make_gradient_flow_chart",
        "def make_update_ratio_chart",
        "def make_cnn_feature_map",
        "def make_attention_heatmap",
    ]
    for fragment in forbidden_page_fragments:
        if fragment in page_text:
            failures.append(f"neural_network_playground.py 仍保留训练联动实现细节：{fragment}")

    required_page_fragments = [
        "from components.playground_training import PlaygroundTrainingHistory",
        "from components.playground_training import run_playground_training",
        "from components.playground_charts import make_loss_curve",
        "from components.playground_charts import make_attention_heatmap",
        "from components.playground_state import default_state",
        "from components.playground_state import ensure_query_preset_loaded",
        "from components.playground_forms import render_layer_form",
        "from components.playground_forms import render_project_io",
    ]
    for fragment in required_page_fragments:
        if fragment not in page_text:
            failures.append(f"neural_network_playground.py 未从 playground_training 复用训练接口：{fragment}")

    if failures:
        raise QualityCheckFailure(
            "中央控制台训练联动 Module 检查失败：\n" + "\n".join(f"  - {item}" for item in failures)
        )
    print("[通过] 中央控制台训练联动 Module 检查：模型构建、训练遥测和图表已从页面 Adapter 中拆出")
