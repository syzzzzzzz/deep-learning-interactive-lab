"""Stable facade for the shared UI/UX visual language and teaching motion system."""

from __future__ import annotations

from components.visual_tokens import BRAND_BORDER
from components.visual_tokens import BRAND_GOLD
from components.visual_tokens import BRAND_INK
from components.visual_tokens import BRAND_MUTED
from components.visual_tokens import BRAND_SOFT
from components.visual_tokens import LIGHT_BLUE
from components.visual_tokens import LIGHT_GREEN
from components.visual_tokens import LIGHT_PURPLE
from components.visual_tokens import NEON_BLUE
from components.visual_tokens import NEON_GREEN
from components.visual_tokens import NEON_PURPLE
from components.visual_runtime import render_loading_bar
from components.visual_runtime import render_particle_field
from components.visual_runtime import render_visual_system
from components.visual_primitives import render_beginner_hint
from components.visual_primitives import render_card
from components.visual_primitives import render_chart_container
from components.visual_primitives import render_concept_animation_shell
from components.visual_primitives import render_metric_card
from components.visual_primitives import render_motion_note
from components.visual_primitives import render_neon_button
from components.visual_primitives import render_neon_metric_card
from components.visual_primitives import render_responsive_motion_grid
from components.visual_primitives import render_shape_flow
from components.visual_primitives import render_status_badge
from components.visual_primitives import render_tooltip_label
from components.visual_effects import render_advanced_conv_comparison
from components.visual_effects import render_attention_light_beams
from components.visual_effects import render_backprop_current_flow
from components.visual_effects import render_central_console_assembly
from components.visual_effects import render_cnn_layer_pipeline
from components.visual_effects import render_convolution_particle_flow
from components.visual_effects import render_gradient_descent_landscape
from components.visual_effects import render_gradient_monitor
from components.visual_effects import render_rnn_hidden_state_flow
from components.visual_effects import render_training_curve_scanner
from components.visual_effects import render_training_dashboard_gauges
from components.visual_effects import render_training_dynamics_panel
from components.visual_effects import render_transformer_attention_heatmap
from components.visual_gallery import render_motion_gallery

__all__ = [
    "BRAND_BORDER",
    "BRAND_GOLD",
    "BRAND_INK",
    "BRAND_MUTED",
    "BRAND_SOFT",
    "LIGHT_BLUE",
    "LIGHT_GREEN",
    "LIGHT_PURPLE",
    "NEON_BLUE",
    "NEON_GREEN",
    "NEON_PURPLE",
    "render_advanced_conv_comparison",
    "render_attention_light_beams",
    "render_backprop_current_flow",
    "render_beginner_hint",
    "render_card",
    "render_central_console_assembly",
    "render_chart_container",
    "render_cnn_layer_pipeline",
    "render_concept_animation_shell",
    "render_convolution_particle_flow",
    "render_gradient_descent_landscape",
    "render_gradient_monitor",
    "render_loading_bar",
    "render_metric_card",
    "render_motion_gallery",
    "render_motion_note",
    "render_neon_button",
    "render_neon_metric_card",
    "render_particle_field",
    "render_responsive_motion_grid",
    "render_rnn_hidden_state_flow",
    "render_shape_flow",
    "render_status_badge",
    "render_tooltip_label",
    "render_training_curve_scanner",
    "render_training_dashboard_gauges",
    "render_training_dynamics_panel",
    "render_transformer_attention_heatmap",
    "render_visual_system",
]