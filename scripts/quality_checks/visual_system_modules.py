from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def check_visual_system_modules(context: QualityCheckContext) -> None:
    """Verify the visual system is split into deep Modules behind a stable facade."""

    facade_path = Path("components/visual_system.py")
    facade_text = context.read_text(facade_path)
    token_text = context.read_text(Path("components/visual_tokens.py"))
    runtime_text = context.read_text(Path("components/visual_runtime.py"))
    primitive_text = context.read_text(Path("components/visual_primitives.py"))
    effect_text = context.read_text(Path("components/visual_effects.py"))
    gallery_text = context.read_text(Path("components/visual_gallery.py"))
    failures: list[str] = []

    required_tokens = ["BRAND_GOLD", "BRAND_INK", "BRAND_BORDER", "BRAND_SOFT", "NEON_BLUE", "NEON_GREEN"]
    for fragment in required_tokens:
        if fragment not in token_text:
            failures.append(f"components/visual_tokens.py 缺少设计 token：{fragment}")

    required_runtime = [
        "class _StreamlitProxy",
        "def _st",
        "def render_visual_system",
        "def render_particle_field",
        "def render_loading_bar",
        "prefers-reduced-motion",
        "focus-visible",
        ".vs-tooltip",
        ".vs-loading",
        ".vs-chart-note",
    ]
    for fragment in required_runtime:
        if fragment not in runtime_text:
            failures.append(f"components/visual_runtime.py 缺少运行时/CSS 契约：{fragment}")

    required_primitives = [
        "def render_tooltip_label",
        "def render_status_badge",
        "def render_motion_note",
        "def render_beginner_hint",
        "def render_neon_metric_card",
        "def render_concept_animation_shell",
        "def render_responsive_motion_grid",
        "def render_shape_flow",
        "def render_card",
        "def render_metric_card",
        "def render_neon_button",
        "def render_chart_container",
    ]
    for fragment in required_primitives:
        if fragment not in primitive_text:
            failures.append(f"components/visual_primitives.py 缺少基础教学 UI 契约：{fragment}")

    required_effects = [
        "def render_convolution_particle_flow",
        "def render_gradient_descent_landscape",
        "def render_attention_light_beams",
        "def render_backprop_current_flow",
        "def render_training_dashboard_gauges",
        "def render_cnn_layer_pipeline",
        "def render_gradient_monitor",
        "def render_training_dynamics_panel",
        "def render_advanced_conv_comparison",
        "def render_rnn_hidden_state_flow",
        "def render_transformer_attention_heatmap",
        "def render_central_console_assembly",
    ]
    for fragment in required_effects:
        if fragment not in effect_text:
            failures.append(f"components/visual_effects.py 缺少学科动效契约：{fragment}")

    if "def render_motion_gallery" not in gallery_text:
        failures.append("components/visual_gallery.py 缺少 motion gallery 契约：def render_motion_gallery")

    forbidden_facade_fragments = [
        "def render_visual_system",
        "def render_tooltip_label",
        "def render_convolution_particle_flow",
        "def render_motion_gallery",
        "class _StreamlitProxy",
        "<style>",
        "@keyframes",
    ]
    for fragment in forbidden_facade_fragments:
        if fragment in facade_text:
            failures.append(f"components/visual_system.py 仍保留视觉系统实现细节：{fragment}")

    required_facade_imports = [
        "from components.visual_tokens import",
        "from components.visual_runtime import render_visual_system",
        "from components.visual_primitives import render_tooltip_label",
        "from components.visual_effects import render_convolution_particle_flow",
        "from components.visual_gallery import render_motion_gallery",
    ]
    for fragment in required_facade_imports:
        if fragment not in facade_text:
            failures.append(f"components/visual_system.py 未作为稳定门面重导出：{fragment}")

    if len(facade_text.splitlines()) > 220:
        failures.append("components/visual_system.py 门面过厚，行数应保持在 220 行以内")

    if failures:
        raise QualityCheckFailure(
            "视觉系统 Module 深化检查失败：\n" + "\n".join(f"  - {item}" for item in failures)
        )
    print("[通过] 视觉系统 Module 深化检查：tokens、runtime、primitives、effects、gallery 已拆分并由 visual_system 门面重导出")
