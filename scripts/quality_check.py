"""
内容质量检查入口。

运行：
    python scripts/quality_check.py
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from scripts.quality_checks.common import QualityCheckContext, QualityCheckFailure
from scripts.quality_checks.artifacts import (
    check_artifact_runtime_behavior as check_artifact_runtime_behavior_domain,
    check_direct_script_artifact_redirection as check_artifact_redirection_domain,
    check_legacy_script_artifact_run as check_legacy_script_artifact_run_domain,
)
from scripts.quality_checks.artifacts import (
    check_root_runtime_artifacts_clean as check_root_artifacts_domain,
)
from scripts.quality_checks.course_catalog import check_course_catalog_module
from scripts.quality_checks.course_source import check_course_source_of_truth
from scripts.quality_checks.legacy_page import check_legacy_page_module
from scripts.quality_checks.legacy_runtime import check_legacy_runtime_module
from scripts.quality_checks.local_runtime import check_local_runtime_module
from scripts.quality_checks.playground_modules import check_playground_modules
from scripts.quality_checks.static_site import check_navigation_and_learning_ux
from scripts.quality_checks.streamlit_home import check_streamlit_home_module
from scripts.quality_checks.streamlit_shell import check_streamlit_shell_module
from scripts.quality_checks.visual_system_modules import check_visual_system_modules

PLACEHOLDER_PATTERNS = [
    "【知识点名称】",
    "【参数名】",
    "【可视化图名称】",
    "【对应可视化",
    "【调参控件",
]

FOCUS_FILES = [
    Path("part4_transformer/transformer_models.py"),
    Path("part1_foundations/classical_ml.py"),
    Path("part1_foundations/math_primer.py"),
    Path("part2_cnn/cnn_architectures.py"),
    Path("part2_cnn/advanced_cnn.py"),
    Path("part3_rnn/sequence_models.py"),
]

TEXT_SCAN_EXCLUDES = {
    Path("scripts/quality_check.py"),
}

LEGACY_PROTOCOL_FILES = [
    Path("part1_foundations/01_tensors_gradients.py"),
    Path("part1_foundations/02_activations_normalization.py"),
    Path("part1_foundations/03_datasets_optimizers.py"),
    Path("part2_cnn/01_convolution_visual.py"),
    Path("part2_cnn/02_feature_maps.py"),
    Path("part2_cnn/03_classic_architectures.py"),
    Path("part2_cnn/04_debug_panel.py"),
    Path("part2_cnn/05_mnist_toy.py"),
    Path("part2_cnn/06_modern_architectures.py"),
    Path("part2_cnn/07_advanced_convolution.py"),
    Path("part2_cnn/08_visualization_gradcam.py"),
    Path("part2_cnn/09_transfer_learning.py"),
    Path("part3_rnn/01_rnn_intuition.py"),
    Path("part3_rnn/02_hidden_states.py"),
    Path("part3_rnn/03_sequence_toys.py"),
    Path("part3_rnn/04_hyperparam_rnn.py"),
    Path("part3_rnn/05_seq2seq_attention.py"),
    Path("part3_rnn/06_text_classification.py"),
    Path("part3_rnn/07_advanced_training.py"),
    Path("part3_rnn/08_debug_problems.py"),
    Path("part4_transformer/01_attention_mechanism.py"),
    Path("part4_transformer/02_multihead_visual.py"),
    Path("part4_transformer/03_encoder_decoder.py"),
    Path("part4_transformer/04_minimal_transformer.py"),
    Path("part4_transformer/05_flash_attention.py"),
    Path("part4_transformer/06_debug_problems.py"),
    Path("part5_toolbox/01_feature_visualization.py"),
    Path("part5_toolbox/02_gradient_monitor.py"),
    Path("part5_toolbox/03_training_dynamics.py"),
    Path("part5_toolbox/04_hyperparam_search.py"),
    Path("part5_toolbox/05_dataset_toys.py"),
    Path("part6_universal_framework/01_unified_interface.py"),
    Path("part6_universal_framework/02_modular_structure.py"),
    Path("part6_universal_framework/03_full_project.py"),
    Path("part6_universal_framework/04_plugin_system.py"),
    Path("part6_universal_framework/05_one_click_training.py"),
    Path("part6_universal_framework/06_streamlit_demo.py"),
    Path("part6_universal_framework/07_project_template.py"),
]

STRICT_LEGACY_PROTOCOL_FILES = [
    Path("part1_foundations/01_tensors_gradients.py"),
    Path("part1_foundations/02_activations_normalization.py"),
    Path("part1_foundations/03_datasets_optimizers.py"),
    Path("part2_cnn/01_convolution_visual.py"),
    Path("part2_cnn/02_feature_maps.py"),
    Path("part2_cnn/03_classic_architectures.py"),
    Path("part2_cnn/04_debug_panel.py"),
    Path("part2_cnn/05_mnist_toy.py"),
    Path("part2_cnn/06_modern_architectures.py"),
    Path("part2_cnn/07_advanced_convolution.py"),
    Path("part2_cnn/08_visualization_gradcam.py"),
    Path("part2_cnn/09_transfer_learning.py"),
    Path("part3_rnn/01_rnn_intuition.py"),
    Path("part3_rnn/02_hidden_states.py"),
    Path("part3_rnn/03_sequence_toys.py"),
    Path("part3_rnn/04_hyperparam_rnn.py"),
    Path("part3_rnn/05_seq2seq_attention.py"),
    Path("part3_rnn/06_text_classification.py"),
    Path("part3_rnn/07_advanced_training.py"),
    Path("part3_rnn/08_debug_problems.py"),
    Path("part4_transformer/01_attention_mechanism.py"),
    Path("part4_transformer/02_multihead_visual.py"),
    Path("part4_transformer/03_encoder_decoder.py"),
    Path("part4_transformer/04_minimal_transformer.py"),
    Path("part4_transformer/05_flash_attention.py"),
    Path("part4_transformer/06_debug_problems.py"),
    Path("part5_toolbox/01_feature_visualization.py"),
    Path("part5_toolbox/02_gradient_monitor.py"),
    Path("part5_toolbox/03_training_dynamics.py"),
    Path("part5_toolbox/04_hyperparam_search.py"),
    Path("part5_toolbox/05_dataset_toys.py"),
    Path("part6_universal_framework/01_unified_interface.py"),
    Path("part6_universal_framework/02_modular_structure.py"),
    Path("part6_universal_framework/03_full_project.py"),
    Path("part6_universal_framework/04_plugin_system.py"),
    Path("part6_universal_framework/05_one_click_training.py"),
    Path("part6_universal_framework/06_streamlit_demo.py"),
    Path("part6_universal_framework/07_project_template.py"),
]

LEGACY_PROTOCOL_DEFERRED_FILES = []

FOCUS_ROUTES = [
    "part2_cnn/02_feature_maps",
    "part3_rnn/01_rnn_intuition",
    "part4_transformer/02_multihead_visual",
    "part5_toolbox/05_dataset_toys",
    "part4_transformer/transformer_models",
    "part1_foundations/classical_ml",
    "part1_foundations/math_primer",
    "part2_cnn/cnn_architectures",
    "part2_cnn/advanced_cnn",
    "part3_rnn/sequence_models",
    "part5_toolbox/01_feature_visualization",
    "part5_toolbox/02_gradient_monitor",
    "part5_toolbox/03_training_dynamics",
    "part5_toolbox/04_hyperparam_search",
    "part5_toolbox/data_training",
    "part6_universal_framework/01_unified_interface",
    "part6_universal_framework/03_full_project",
    "part6_universal_framework/04_plugin_system",
    "part6_universal_framework/05_one_click_training",
    "part6_universal_framework/07_project_template",
]

EXPECTED_CONTROL_REFERENCES = {
    Path("part4_transformer/transformer_models.py"): [
        "输入文本",
        "计算步骤",
        "注意力演示维度",
        "多头注意力锐度",
        "选择一个 query token",
    ],
    Path("part1_foundations/classical_ml.py"): [
        "学习率",
        "L2 正则化系数",
        "训练轮数",
        "类别可分程度",
        "正则化强度 C",
        "最大树深度",
        "最大叶子节点数",
        "K 值",
        "迭代步数",
        "核函数",
        "gamma",
        "查询点 x1",
        "查询点 x2",
    ],
    Path("part1_foundations/math_primer.py"): [
        "u_x",
        "u_y",
        "v_x",
        "v_y",
        "缩放系数 alpha",
        "观察点 x0",
        "割线步长 h",
        "当前 x",
        "当前 y",
        "先验 P(H)：样本真实为正的比例",
        "选择分布",
        "采样数量",
        "学习率",
        "动量",
    ],
    Path("part2_cnn/cnn_architectures.py"): [
        "选择网络",
        "查看内容",
        "前向传播步骤",
        "输入模式",
        "观察层级",
        "输入/输出维度不一致时使用 1x1 投影",
        "残差分支 F(x) 强度",
    ],
    Path("part2_cnn/advanced_cnn.py"): [
        "卷积类型",
        "输入尺寸 H=W",
        "输入通道数",
        "输出通道数",
        "卷积核大小",
        "步长 stride",
        "填充 padding",
        "空洞率 dilation",
        "分组数 groups",
        "转置卷积 output_padding",
        "池化核",
        "池化步长",
        "batch size",
        "通道数",
        "丢弃概率 p",
        "训练模式：开启 Dropout",
    ],
    Path("part3_rnn/sequence_models.py"): [
        "查看内容",
        "展开时间步",
        "当前时间步",
        "循环权重尺度",
        "隐藏单元数",
        "输入强度",
        "反向传播时间距离",
        "循环 Jacobian 尺度",
        "tanh 饱和程度",
        "遗忘门偏置",
        "输入门偏置",
        "输出门偏置",
        "更新门偏置",
        "重置门偏置",
        "序列长度",
        "双向合并方式",
        "模型类型",
        "窗口长度",
        "学习率",
        "temperature",
    ],
    Path("part5_toolbox/data_training.py"): [
        "样本数",
        "收入尺度压缩因子",
        "数值处理方式",
        "旋转角度",
        "裁剪保留比例",
        "缩放比例",
        "亮度",
        "对比度",
        "色彩饱和度",
        "网络深度",
        "每层宽度",
        "激活函数",
        "L1 强度",
        "L2 强度",
        "Dropout",
        "早停耐心",
        "最多训练轮数",
        "初始 / 峰值学习率",
        "最小学习率",
        "总 epoch",
        "StepLR 间隔",
        "StepLR 衰减系数",
        "Warmup epoch",
        "随机种子",
    ],
}

EXPECTED_CONTENT_REFERENCES = {
    Path("main.py"): [
        "LEGACY_LEARNING_GUIDES",
        "part6_universal_framework/03_full_project",
        "part6_universal_framework/05_one_click_training",
        "part6_universal_framework/07_project_template",
        "render_module_knowledge_nav",
        "render学习操作面板",
    ],
    Path("components/legacy_page.py"): [
        "学习导读",
    ],
    Path("components/streamlit_home.py"): [
        "柔和阅读界面",
    ],
    Path("components/streamlit_shell.py"): [
        "render_visual_system",
    ],
    Path("index.html"): [
        "assets/site.css",
        "assets/site.js",
        'id="app"',
        "data-drawer",
        "code-template",
    ],
    Path("assets/site.css"): [
        "--accent: #b08a4f",
        "--accent-dark: #2a2118",
        "--border: #e6ded2",
        "Noto Serif SC",
        "Noto Sans SC",
        "resize: vertical",
        "overflow: auto",
        "interactive-lab",
        "concept-demo",
        "demo-stage",
        "lesson-deep-dive",
        "dry-goods",
        "dry-goods-grid",
        "dry-goods-card",
        "knowledge-point-index",
        "zero-basics",
        "zero-basics-grid",
        "zero-basics-action",
        "starter-section",
        "student-route-card",
        "course-learning-actions",
        "course-console-cta",
        "course-topline",
        "mode-switcher",
        "drawer-onboarding",
        "console-purpose-strip",
        "progress-dashboard",
        "console-task-strip",
        "deep-dive-details",
        "three-minute-brief",
        "source-guide",
        "source-snippet",
        "learning-plan-mini",
        "zero-basics-case-strip",
        "three-minute-brief",
        "source-guide",
        "source-snippet",
        "learning-plan-mini",
        "zero-basics-case-strip",
        "lesson-outline",
        "deep-dive-grid",
        "lesson-code",
        "knowledge-columns",
        "practice-callout",
        "lab-grid",
        "central-console",
        "console-topline",
        "console-hero",
        "console-context",
        "console-grid",
        "console-panel",
        "console-dl",
        "console-chip-list",
        "console-steps",
        "console-workbench",
        "console-metrics",
        "console-result",
        "textarea[data-console-note]",
        "portfolio-panel",
        "tech-stack",
        "code-wall",
        "hardcore-labs",
        "llm-cookbook-section",
        "llm-roadmap-panel",
        "llm-track-grid",
        "llm-track-card",
        "llm-detail-list",
        "source-note",
        "hardcore-workbench-grid",
        "hardcore-lab-card",
        "hardcore-control-grid",
        "hardcore-stage",
        "hardcore-metrics",
        "hardcore-readout",
        "xai-board",
        "adversarial-board",
        "challenge-board",
        "case-board",
        "node-canvas",
        "canvas-node",
        "canvas-edge-layer",
        "canvas-readout",
        "training-event-bus",
        "event-log",
        "event-subscriber",
        "matrix-grid",
        "attention-row",
        "attention-chain",
        "attention-score-row",
        "attention-output-card",
        "progress-meter",
        "course-grid",
        "side-drawer",
        "@keyframes pageReveal",
        "@keyframes revealSoftly",
        "@keyframes tracePath",
        "motion-reveal",
        "motion-trace",
        "matrixSettle",
        "barFlow",
        "flowDraw",
        "scanWindow",
        "nodeRise",
        "memoryPulse",
        "readoutPulse",
        "prefers-reduced-motion",
    ],
    Path("assets/site.js"): [
        "const PARTS",
        "const MODULES",
        "renderCourse",
        "loadSource",
        "DOMAIN_PROFILES",
        "renderLessonBrief",
        "renderDryGoods",
        "renderKnowledgePointIndex",
        "renderZeroBasics",
        "renderConceptAnimation",
        "renderKnowledgeSections",
        "renderLessonDeepDiveShell",
        "parseLegacyMarkdown",
        "extractSourceTexts",
        "parseSourceLessonNotes",
        "renderSourceLessonNotes",
        "loadLessonNotes",
        "renderConceptStage",
        "wireConceptDemos",
        "renderInteractiveLab",
        "wireInteractiveLab",
        "renderMathGradientLab",
        "renderCnnFeatureLab",
        "renderAttentionLab",
        "renderSequenceMemoryLab",
        "renderTrainingDiagnosticsLab",
        "renderArchitectureFlowLab",
        "renderSystemsFlowLab",
        "BEGINNER_BLUEPRINTS",
        "DOMAIN_BLUEPRINTS",
        "legacyMarkdownCandidates",
        "deep_learning_book/",
        "把刚才的动画讲明白",
        "可点击学习目录",
        "先看 3 分钟版",
        "知识点硬核笔记",
        "知识点全量索引",
        "机制骨架",
        "必看变量",
        "源码抓手",
        "data-knowledge-points",
        "data-source-guide",
        "renderTeachingSourceGuide",
        "renderThreeMinuteBrief",
        "moduleLearningPlan",
        "scrollCourseTarget",
        "readLearningMode",
        "writeLearningMode",
        "wireTagFilters",
        "wireCourseModeSwitcher",
        "data-learning-mode",
        "data-tag-filter",
        "drawer-onboarding",
        "course-topline",
        "course-console-cta",
        "mode-switcher",
        "console-purpose-strip",
        "learning-plan-mini",
        "zero-basics-case-strip",
        "源码对照：只看和动画有关的几段",
        "这是什么？",
        "生活类比",
        "一句话直觉",
        "严谨定义",
        "图中每个元素代表什么",
        "颜色/亮度/方向/速度代表什么",
        "用户应该调哪个参数",
        "观察什么变化",
        "为什么会这样",
        "常见误区",
        "工程用途",
        "去中央控制台实战",
        'data-lab="math-gradient"',
        'data-lab="cnn-feature"',
        'data-lab="attention"',
        'data-lab="sequence-memory"',
        'data-lab="training-diagnostics"',
        'data-lab="architecture-flow"',
        'data-lab="systems-flow"',
        "hashchange",
        "fetch(module.sourcePath)",
        "drawer-search",
        "kickRouteMotion",
        "applyMotionReveal",
        "data-demo-control",
        "demo-stage",
        "data-attention-mechanism",
        "data-attention-scores",
        "data-attention-output",
        "Q·K 匹配分",
        "softmax 权重",
        "Value 贡献",
        "注意力锐度",
        "上下文噪声",
        "Query token",
        "data-lesson-notes",
        "legacyMarkdownPath",
        "pulseLabReadout",
        "IntersectionObserver",
        "motion-point",
        "motion-trace",
        "--cell-delay",
    ],
    Path("components/progress_tracker.py"): [
        "稍后复习",
        "今日推荐",
        "学习报告",
        "弱点分析",
        "章节完成标准",
        "错题",
        "实验记录",
        "render学习操作面板",
        "build_learning_report",
        "analyze_weaknesses",
        "recommend_today",
    ],
    Path("components/visual_system.py"): [
        "BRAND_GOLD",
        "#b08a4f",
        "#2a2118",
        "#e6ded2",
        "#f7f3ec",
        "JetBrains Mono",
        "Inter",
        "font-awesome",
        "render_particle_field",
        "render_convolution_particle_flow",
        "render_gradient_descent_landscape",
        "render_attention_light_beams",
        "render_backprop_current_flow",
        "render_training_dashboard_gauges",
        "render_tooltip_label",
        "render_motion_note",
        "render_neon_metric_card",
        "render_concept_animation_shell",
        "render_responsive_motion_grid",
        "render_shape_flow",
        "render_status_badge",
        "render_beginner_hint",
        "prefers-reduced-motion",
        "@media (max-width: 760px)",
        ":focus-visible",
        "tooltip",
        "loading",
        "图表说明",
        "render_motion_gallery",
        "scroll-behavior: smooth",
    ],
    Path("components/knowledge_graph.py"): [
        "canonical_node_keys",
        "practice_url",
        "render_legacy_book_reference",
        "掌握标准",
        "去实战目标",
        "前置知识",
        "相关知识",
        "后续推荐",
    ],
    Path("components/legacy_book.py"): [
        "deep_learning_book",
        "get_legacy_lesson",
        "render_legacy_book_reference",
        "下载旧教材 Markdown",
        "预览旧教材原文",
    ],
    Path("part5_toolbox/01_feature_visualization.py"): [
        "print_learning_guide",
        "学习导读",
        "工程坑案例",
        "进阶思考",
    ],
    Path("part5_toolbox/02_gradient_monitor.py"): [
        "print_learning_guide",
        "梯度爆炸",
        "梯度消失",
        "工程坑案例",
        "进阶思考",
    ],
    Path("part5_toolbox/03_training_dynamics.py"): [
        "print_learning_guide",
        "更新幅度比",
        "激活饱和率",
        "工程坑案例",
        "进阶思考",
    ],
    Path("part5_toolbox/04_hyperparam_search.py"): [
        "print_learning_guide",
        "LR Finder",
        "工程经验",
        "真实踩坑",
        "进阶思考",
    ],
    Path("part5_toolbox/data_training.py"): [
        "render_action",
        "图怎么看",
        "工程经验",
        "真实踩坑",
        "进阶思考",
    ],
    Path("part6_universal_framework/01_unified_interface.py"): [
        "print_learning_guide",
        "统一接口",
        "工程坑案例",
        "进阶思考",
    ],
    Path("part6_universal_framework/03_full_project.py"): [
        "print_learning_guide",
        "UniversalTrainer",
        "UniversalVisualizer",
        "工程坑案例",
        "复现实验",
    ],
    Path("part6_universal_framework/04_plugin_system.py"): [
        "print_learning_guide",
        "注册表",
        "配置模板",
        "工程坑案例",
        "进阶思考",
    ],
    Path("part6_universal_framework/05_one_click_training.py"): [
        "print_learning_guide",
        "best.pt",
        "training_log.csv",
        "config.json",
        "工程坑案例",
    ],
    Path("part6_universal_framework/07_project_template.py"): [
        "print_learning_guide",
        "训练入口",
        "评估脚本",
        "K-Fold",
        "工程坑案例",
    ],
    Path("part6_universal_framework/neural_network_playground.py"): [
        "LayerNorm",
        "ResidualBlock",
        "MultiheadAttention",
        "TransformerEncoder",
        "export_project_config",
        "import_project_config",
        "训练与章节联动",
        "联动训练演示",
        "真实轻量训练",
        "教学模拟",
        "make_loss_curve",
        "make_gradient_flow_chart",
        "make_update_ratio_chart",
        "make_cnn_feature_map",
        "make_attention_heatmap",
    ],
    Path("part1_foundations/01_tensors_gradients.py"): [
        "render_gradient_descent_landscape",
        "render_backprop_current_flow",
        "render_beginner_hint",
        "render_motion_note",
        "render_shape_flow",
    ],
    Path("part2_cnn/01_convolution_visual.py"): [
        "render_convolution_particle_flow",
        "render_beginner_hint",
        "render_motion_note",
        "render_shape_flow",
    ],
    Path("part4_transformer/transformer_models.py"): [
        "render_attention_light_beams",
        "render_beginner_hint",
        "render_motion_note",
        "render_shape_flow",
        "render_tooltip_label",
    ],
    Path("part6_universal_framework/training_demo.py"): [
        "render_training_dashboard_gauges",
        "render_beginner_hint",
        "render_motion_note",
        "render_neon_metric_card",
    ],
    Path("part7_interview/interview_quiz.py"): [
        "QUESTION_MODULE_MAP",
        "SIMULATION_SCRIPTS",
        "score_user_answer",
        "render_score_card",
        "render_simulation_panel",
        "persist_interview_record",
        "自动评分",
        "模拟面试流程",
        "错题档案",
    ],
    Path("part7_interview/networking.py"): [
        "进入网络专项刷题",
        "模型推理 API",
    ],
    Path("part7_interview/database_sql.py"): [
        "进入数据库专项刷题",
        "实验记录数据库",
    ],
    Path("part7_interview/data_structures.py"): [
        "进入算法专项刷题",
        "Transformer 注意力",
    ],
    Path("part7_interview/operating_system.py"): [
        "进入操作系统专项刷题",
        "DataLoader",
    ],
}

SMOKE_FUNCTIONS = {
    Path("part4_transformer/transformer_models.py"): [
        "render_overview",
        "render_self_attention",
        "render_multihead",
        "render_text_heatmap",
    ],
    Path("part1_foundations/classical_ml.py"): [
        "render_linear_regression",
        "render_logistic_regression",
        "render_decision_tree",
        "render_kmeans",
        "render_svm",
        "render_knn",
    ],
    Path("part1_foundations/math_primer.py"): [
        "render_linear_algebra",
        "render_calculus",
        "render_probability",
        "render_gradient_descent",
        "render_cheatsheet",
    ],
    Path("part2_cnn/cnn_architectures.py"): [
        "render_evolution_guide",
        "render_forward_guide",
        "render_feature_map_guide",
        "render_residual_guide",
        "render_inception_guide",
        "render_detection_segmentation_guide",
    ],
    Path("part2_cnn/advanced_cnn.py"): [
        "render_conv_experiment_guide",
        "render_conv_overview_guide",
        "render_pooling_guide",
        "render_bn_guide",
        "render_dropout_guide",
    ],
    Path("part3_rnn/sequence_models.py"): [
        "render_sequence_learning_map",
        "render_rnn_unroll_guide",
        "render_gradient_issue_guide",
        "render_lstm_guide",
        "render_gru_guide",
        "render_bidirectional_guide",
        "render_forecast_guide",
        "render_text_generation_guide",
    ],
}


class CheckFailure(Exception):
    pass


QUALITY_CONTEXT = QualityCheckContext(ROOT)


def project_files(suffixes: tuple[str, ...]) -> list[Path]:
    ignored_parts = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".streamlit_module_outputs",
        ".codex-ref-llm-cookbook",
        "legacy_book",
    }
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in ignored_parts for part in rel.parts):
            continue
        if path.suffix.lower() in suffixes:
            files.append(rel)
    return sorted(files)


def read_text(rel_path: Path) -> str:
    return QUALITY_CONTEXT.read_text(rel_path)


def read_visual_system_text() -> str:
    paths = [
        Path("components/visual_system.py"),
        Path("components/visual_tokens.py"),
        Path("components/visual_runtime.py"),
        Path("components/visual_primitives.py"),
        Path("components/visual_effects.py"),
        Path("components/visual_gallery.py"),
    ]
    return "\n".join(read_text(path) for path in paths if (ROOT / path).exists())


def run_domain_check(check) -> None:
    try:
        check(QUALITY_CONTEXT)
    except QualityCheckFailure as exc:
        raise CheckFailure(str(exc)) from exc


def parse_python(rel_path: Path) -> ast.Module:
    return ast.parse(read_text(rel_path), filename=str(ROOT / rel_path))


def literal_value(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def check_python_compile() -> None:
    failures: list[str] = []
    py_files = project_files((".py",))
    for rel_path in py_files:
        try:
            source = read_text(rel_path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", SyntaxWarning)
                compile(source, str(ROOT / rel_path), "exec")
        except Exception as exc:  # noqa: BLE001 - report all compile-time failures.
            failures.append(f"{rel_path}: {exc}")
    if failures:
        raise CheckFailure("Python 编译或 SyntaxWarning 检查失败：\n" + "\n".join(failures))
    print(f"[通过] Python 编译检查：{len(py_files)} 个文件")


def check_placeholders() -> None:
    failures: list[str] = []
    text_files = [path for path in project_files((".py", ".md", ".txt", ".bat")) if path not in TEXT_SCAN_EXCLUDES]
    generic_placeholder = re.compile(r"【[^】]{1,40}】")
    for rel_path in text_files:
        text = read_text(rel_path)
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern in text:
                failures.append(f"{rel_path}: 发现模板占位符 {pattern}")
        for match in generic_placeholder.finditer(text):
            failures.append(f"{rel_path}: 发现疑似模板占位符 {match.group(0)}")
    if failures:
        raise CheckFailure("模板占位符检查失败：\n" + "\n".join(failures))
    print(f"[通过] 模板占位符检查：{len(text_files)} 个文本文件")


def check_bracket_placeholders() -> None:
    failures: list[str] = []
    pattern = re.compile(r"【[^】\n]{1,80}】")
    py_files = [path for path in project_files((".py",)) if path not in TEXT_SCAN_EXCLUDES]
    for rel_path in py_files:
        text = read_text(rel_path)
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                failures.append(f"{rel_path}:{line_number}: 发现疑似模板占位符 {match.group(0)}")
    if failures:
        raise CheckFailure("残留模板占位符检查失败：\n" + "\n".join(failures))
    print(f"[通过] 残留模板占位符检查：【xxx】格式，{len(py_files)} 个 Python 文件")


def check_legacy_module_protocol_metadata() -> None:
    failures: list[str] = []
    for rel_path in LEGACY_PROTOCOL_FILES:
        tree = parse_python(rel_path)
        assignments: dict[str, object] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"MODULE_TITLE", "MODULE_SUMMARY", "MODULE_TAGS"}:
                        assignments[target.id] = literal_value(node.value)

        title = assignments.get("MODULE_TITLE")
        summary = assignments.get("MODULE_SUMMARY")
        tags = assignments.get("MODULE_TAGS")
        if not isinstance(title, str) or not title.strip():
            failures.append(f"{rel_path}: 缺少有效 MODULE_TITLE")
        if not isinstance(summary, str) or not summary.strip():
            failures.append(f"{rel_path}: 缺少有效 MODULE_SUMMARY")
        if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
            failures.append(f"{rel_path}: 缺少有效 MODULE_TAGS")
    if failures:
        raise CheckFailure("老脚本模块协议元数据检查失败：\n" + "\n".join(failures))
    print(f"[通过] 老脚本模块协议元数据检查：{len(LEGACY_PROTOCOL_FILES)} 个文件")


def has_top_level_name(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def top_level_assignment(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return literal_value(node.value)
    return None


def check_strict_legacy_module_protocol() -> None:
    """Ensure refactored old scripts expose stable render/compute/smoke hooks."""

    failures: list[str] = []
    for rel_path in STRICT_LEGACY_PROTOCOL_FILES:
        tree = parse_python(rel_path)
        funcs = function_names(tree)
        if not any(name.startswith("compute") for name in funcs):
            failures.append(f"{rel_path}: 缺少 compute*() 纯计算入口")
        if not has_top_level_name(tree, "render"):
            failures.append(f"{rel_path}: 缺少 render() 页面入口")
        if not has_top_level_name(tree, "smoke"):
            failures.append(f"{rel_path}: 缺少 smoke() 轻量自检入口")
        related = top_level_assignment(tree, "MODULE_RELATED_TOPICS")
        practice = top_level_assignment(tree, "PRACTICE_TARGET")
        if rel_path not in {Path("part1_foundations/01_tensors_gradients.py"), Path("part2_cnn/01_convolution_visual.py"), Path("part4_transformer/01_attention_mechanism.py")}:
            if not isinstance(related, list) or not related:
                failures.append(f"{rel_path}: 缺少 MODULE_RELATED_TOPICS 知识联动元数据")
            if not isinstance(practice, str) or not practice.strip():
                failures.append(f"{rel_path}: 缺少 PRACTICE_TARGET 实战目标")
    if failures:
        raise CheckFailure("严格老脚本协议检查失败：\n" + "\n".join(failures))
    print(f"[通过] 严格老脚本协议检查：{len(STRICT_LEGACY_PROTOCOL_FILES)} 个文件")


def legacy_manifest_python_targets() -> list[Path]:
    manifest_path = ROOT / "docs" / "legacy_book" / "manifest.json"
    if not manifest_path.exists():
        raise CheckFailure("旧教材协议审计失败：缺少 docs/legacy_book/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lessons = manifest.get("lessons", {})
    if not isinstance(lessons, dict):
        raise CheckFailure("旧教材协议审计失败：manifest.json 的 lessons 不是对象")

    targets: list[Path] = []
    for route, payload in lessons.items():
        if not isinstance(payload, dict):
            raise CheckFailure(f"旧教材协议审计失败：{route} 的 manifest 记录不是对象")
        source_path = payload.get("source_path")
        if not isinstance(source_path, str) or not source_path.endswith(".md"):
            raise CheckFailure(f"旧教材协议审计失败：{route} 缺少有效 source_path")
        targets.append(Path(source_path).with_suffix(".py"))
    return sorted(targets)


def check_legacy_protocol_audit() -> None:
    """Keep the old-book render/compute split status explicit and auditable."""

    failures: list[str] = []
    audit_path = ROOT / "docs" / "legacy_protocol_audit.md"
    if not audit_path.exists():
        failures.append("缺少 docs/legacy_protocol_audit.md")
        audit_text = ""
    else:
        audit_text = audit_path.read_text(encoding="utf-8")

    targets = legacy_manifest_python_targets()
    target_set = set(targets)
    strict_set = set(STRICT_LEGACY_PROTOCOL_FILES)
    deferred_set = set(LEGACY_PROTOCOL_DEFERRED_FILES)
    overlap = sorted(strict_set & deferred_set)
    missing_from_audit = sorted(target_set - strict_set - deferred_set)
    extra_in_audit = sorted((strict_set | deferred_set) - target_set)

    if overlap:
        failures.append("严格名单和待拆分名单存在重叠：\n" + "\n".join(str(path) for path in overlap))
    if missing_from_audit:
        failures.append("旧教材章节未被纳入协议审计：\n" + "\n".join(str(path) for path in missing_from_audit))
    if extra_in_audit:
        failures.append("协议审计名单包含非旧教材章节：\n" + "\n".join(str(path) for path in extra_in_audit))

    expected_markers = [
        f"legacy_lessons: {len(targets)}",
        f"strict_protocolized: {len(strict_set)}",
        f"deferred_legacy: {len(deferred_set)}",
    ]
    for marker in expected_markers:
        if marker not in audit_text:
            failures.append(f"docs/legacy_protocol_audit.md 缺少计数标记：{marker}")

    for rel_path in sorted(strict_set):
        if not (ROOT / rel_path).exists():
            failures.append(f"{rel_path}: 严格协议文件不存在")
        if str(rel_path).replace("\\", "/") not in audit_text:
            failures.append(f"docs/legacy_protocol_audit.md 未列出严格协议文件：{rel_path}")
    for rel_path in sorted(deferred_set):
        if not (ROOT / rel_path).exists():
            failures.append(f"{rel_path}: 待拆分旧脚本不存在")
        if str(rel_path).replace("\\", "/") not in audit_text:
            failures.append(f"docs/legacy_protocol_audit.md 未列出待拆分文件：{rel_path}")

    if failures:
        raise CheckFailure("旧教材 render/compute 协议审计失败：\n" + "\n".join(failures))
    print(
        "[通过] 旧教材 render/compute 协议审计："
        f"{len(strict_set)} 个严格协议化，{len(deferred_set)} 个待拆分，覆盖 {len(targets)} 个旧教材章节"
    )


def streamlit_control_labels(text: str) -> set[str]:
    label_patterns = [
        r"st(?:\.sidebar)?\.(?:slider|selectbox|select_slider|segmented_control|radio|text_input|text_area|number_input|toggle|checkbox|multiselect)\(\s*([\"'])(.*?)\1",
        r"\b\w+\.(?:slider|selectbox|select_slider|segmented_control|radio|text_input|text_area|number_input|toggle|checkbox|multiselect)\(\s*([\"'])(.*?)\1",
        r"segmented(?:_control)?\(\s*([\"'])(.*?)\1",
    ]
    labels: set[str] = set()
    for pattern in label_patterns:
        labels.update(match.group(2) for match in re.finditer(pattern, text, flags=re.S))
    return labels


def check_expected_controls() -> None:
    failures: list[str] = []
    for rel_path, expected_labels in EXPECTED_CONTROL_REFERENCES.items():
        text = read_text(rel_path)
        labels = streamlit_control_labels(text)
        for label in expected_labels:
            if label not in labels:
                failures.append(f"{rel_path}: 文案引用的控件不存在或未被识别：{label}")
    if failures:
        raise CheckFailure("重点控件引用检查失败：\n" + "\n".join(failures))
    print(f"[通过] 重点控件引用检查：{len(EXPECTED_CONTROL_REFERENCES)} 个页面")


def check_expected_content() -> None:
    failures: list[str] = []
    for rel_path, expected_fragments in EXPECTED_CONTENT_REFERENCES.items():
        text = read_text(rel_path)
        if rel_path == Path("part6_universal_framework/neural_network_playground.py"):
            text += read_text(Path("components/playground_core.py"))
            text += read_text(Path("components/playground_state.py"))
            text += read_text(Path("components/playground_forms.py"))
            text += read_text(Path("components/playground_training.py"))
        if rel_path == Path("components/visual_system.py"):
            text = read_visual_system_text()
        for fragment in expected_fragments:
            if fragment not in text:
                failures.append(f"{rel_path}: 缺少工程教学内容片段：{fragment}")
    if failures:
        raise CheckFailure("工程教学内容检查失败：\n" + "\n".join(failures))
    print(f"[通过] 工程教学内容检查：{len(EXPECTED_CONTENT_REFERENCES)} 个页面")


def check_static_html_site() -> None:
    failures: list[str] = []
    required_files = [
        Path("index.html"),
        Path("assets/site.css"),
        Path("assets/site.js"),
        Path("main.py"),
        Path("README.md"),
        Path("start_lab.bat"),
    ]
    for rel_path in required_files:
        if not (ROOT / rel_path).exists():
            failures.append(f"{rel_path}: 静态站文件不存在")

    if failures:
        raise CheckFailure("静态 HTML 站点检查失败：\n" + "\n".join(failures))

    html_text = read_text(Path("index.html"))
    css_text = read_text(Path("assets/site.css"))
    js_text = read_text(Path("assets/site.js"))
    main_text = read_text(Path("main.py"))
    streamlit_home_text = read_text(Path("components/streamlit_home.py"))
    local_runtime_text = read_text(Path("components/local_runtime.py"))
    readme_text = read_text(Path("README.md"))
    start_text = read_text(Path("start_lab.bat"))
    workflow_text = read_text(Path(".github/workflows/quality.yml"))

    required_html_fragments = [
        '<main id="app"',
        "assets/site.css",
        "assets/site.js",
        "data-drawer",
        "data-open-menu",
        "打开全站目录",
        "全站课程目录抽屉",
        'href="#home"',
        'href="#starter"',
        'href="#path"',
        'href="#courses"',
        'href="#hardcore-labs"',
        'href="#notes"',
    ]
    for fragment in required_html_fragments:
        if fragment not in html_text:
            failures.append(f"index.html: 缺少静态站入口片段 {fragment}")

    required_css_fragments = [
        "--bg: #ffffff",
        "--bg-soft: #f7f3ec",
        "--text: #171411",
        "--muted: #7c756c",
        "--border: #e6ded2",
        "--accent: #b08a4f",
        "--accent-dark: #2a2118",
        "Noto Serif SC",
        "Noto Sans SC",
        "resize: vertical",
        "overflow: auto",
        "interactive-lab",
        "concept-demo",
        "demo-stage",
        "lesson-deep-dive",
        "dry-goods",
        "dry-goods-grid",
        "dry-goods-card",
        "knowledge-point-index",
        "zero-basics",
        "zero-basics-grid",
        "zero-basics-action",
        "starter-section",
        "student-route-card",
        "course-learning-actions",
        "course-console-cta",
        "course-topline",
        "mode-switcher",
        "drawer-onboarding",
        "console-purpose-strip",
        "progress-dashboard",
        "console-task-strip",
        "deep-dive-details",
        "lesson-outline",
        "deep-dive-grid",
        "lesson-code",
        "knowledge-columns",
        "practice-callout",
        "lab-grid",
        "central-console",
        "console-topline",
        "console-hero",
        "console-context",
        "console-grid",
        "console-panel",
        "console-dl",
        "console-chip-list",
        "console-steps",
        "console-workbench",
        "console-metrics",
        "console-result",
        "textarea[data-console-note]",
        "hardcore-workbench-grid",
        "llm-cookbook-section",
        "llm-roadmap-panel",
        "llm-track-grid",
        "llm-track-card",
        "llm-detail-list",
        "source-note",
        "hardcore-lab-card",
        "hardcore-control-grid",
        "hardcore-stage",
        "hardcore-metrics",
        "hardcore-readout",
        "xai-board",
        "xai-cell",
        "adversarial-compare",
        "confidence-row",
        "challenge-gauge",
        "challenge-checklist",
        "case-pipeline",
        "artifact-checklist",
        "matrix-grid",
        "attention-row",
        "progress-meter",
        "@keyframes pageReveal",
        "@keyframes revealSoftly",
        "@keyframes tracePath",
        "motion-reveal",
        "motion-trace",
        "matrixSettle",
        "barFlow",
        "flowDraw",
        "scanWindow",
        "nodeRise",
        "memoryPulse",
        "readoutPulse",
        "prefers-reduced-motion",
    ]
    for fragment in required_css_fragments:
        if fragment not in css_text:
            failures.append(f"assets/site.css: 缺少高级学习视觉约束 {fragment}")

    forbidden_fragments = [
        "#00f0ff",
        "#b000ff",
        "#00ff88",
        "rgba(0,240,255",
        "rgba(176,0,255",
        "rgba(0,255,136",
    ]
    for fragment in forbidden_fragments:
        if fragment in css_text or fragment in js_text:
            failures.append(f"静态站不应再使用旧霓虹色片段 {fragment}")

    required_js_fragments = [
        "const PARTS",
        "const MODULES",
        "function renderHome",
        "function renderPortfolioSection",
        "function renderHardcoreLabsSection",
        "function renderHardcoreXaiLab",
        "function renderHardcoreAdversarialLab",
        "function renderHardcoreChallengeLab",
        "function renderHardcoreCaseLab",
        "function wireHardcoreLabs",
        "LLM_COOKBOOK_TRACKS",
        "function renderLLMDetailList",
        "function renderLLMStudyOrder",
        "function renderLLMCookbookBridge",
        "datawhalechina/llm-cookbook",
        "Prompt Engineering",
        "RAG 问答",
        "Agent & Tools",
        "Evaluation & Debugging",
        "核心直觉",
        "常见失败",
        "验收标准",
        "落地检查",
        "async function renderCourse",
        "DOMAIN_PROFILES",
        "function renderLessonBrief",
        "function renderDryGoods",
        "function renderKnowledgePointIndex",
        "function renderZeroBasics",
        "function renderConceptAnimation",
        "function renderKnowledgeSections",
        "function renderLessonDeepDiveShell",
        "SOURCE_LIBRARY",
        "CONTENT_CREDIBILITY",
        "function renderCredibilitySection",
        "data-content-credibility",
        "内容可信度",
        "本页参考来源",
        "Attention Is All You Need",
        "torch.nn.functional.scaled_dot_product_attention",
        "function parseLegacyMarkdown",
        "function extractSourceTexts",
        "function parseSourceLessonNotes",
        "function renderSourceLessonNotes",
        "async function loadLessonNotes",
        "function renderConceptStage",
        "function wireConceptDemos",
        "function renderInteractiveLab",
        "function wireInteractiveLab",
        "function consoleHref",
        "function renderCentralConsole",
        "function wireCentralConsole",
        "function renderNodeCanvas",
        "function wireNodeCanvas",
        "function renderTrainingEventBus",
        "function buildTrainingEvent",
        "function publishTrainingEvent",
        "function updateCanvasEdges",
        "function renderMathGradientLab",
        "function renderCnnFeatureLab",
        "function renderAttentionLab",
        "function renderSequenceMemoryLab",
        "function renderTrainingDiagnosticsLab",
        "function renderArchitectureFlowLab",
        "function renderSystemsFlowLab",
        "BEGINNER_BLUEPRINTS",
        "DOMAIN_BLUEPRINTS",
        "MODULE_TEACHING_NOTES",
        "知识点硬核笔记",
        "知识点全量索引",
        "机制骨架",
        "必看变量",
        "源码抓手",
        "data-knowledge-points",
        "data-source-guide",
        "renderTeachingSourceGuide",
        "renderThreeMinuteBrief",
        "moduleLearningPlan",
        "scrollCourseTarget",
        "learning-plan-mini",
        "zero-basics-case-strip",
        "可点击学习目录",
        "先看 3 分钟版",
        "源码对照：只看和动画有关的几段",
        "function legacyMarkdownCandidates",
        "deep_learning_book/",
        "把刚才的动画讲明白",
        "这是什么？",
        "生活类比",
        "一句话直觉",
        "严谨定义",
        "图中每个元素代表什么",
        "颜色/亮度/方向/速度代表什么",
        "用户应该调哪个参数",
        "观察什么变化",
        "为什么会这样",
        "常见误区",
        "工程用途",
        "去中央控制台实战",
        'data-lab="math-gradient"',
        'data-lab="cnn-feature"',
        'data-lab="attention"',
        'data-lab="sequence-memory"',
        'data-lab="training-diagnostics"',
        'data-lab="architecture-flow"',
        'data-lab="systems-flow"',
        "function route",
        "#course/",
        'hash.startsWith("#console/")',
        "data-central-console",
        "data-node-canvas",
        "data-node",
        "data-training-bus",
        "data-event-log",
        "data-console-result",
        "data-console-note",
        "学习成果档案",
        "深度实验室",
        "模型可解释性实验室",
        "对抗样本演示",
        "小型深度学习挑战",
        "端到端案例",
        "data-hardcore-lab",
        "data-hardcore-control",
        "data-hardcore-readout",
        "新手引导",
        "学习导航",
        "第 1 步：先动一个参数",
        "data-course-scroll",
        "data-mark-understood",
        "data-mark-review",
        "data-learning-mode",
        "data-tag-filter",
        "drawer-onboarding",
        "course-topline",
        "course-console-cta",
        "mode-switcher",
        "console-purpose-strip",
        "readLearningMode",
        "writeLearningMode",
        "wireTagFilters",
        "wireCourseModeSwitcher",
        "scrollToHashTarget",
        "返回首页",
        "window.addEventListener(\"hashchange\", route)",
        "fetch(module.sourcePath)",
        "kickRouteMotion",
        "applyMotionReveal",
        "data-demo-control",
        "demo-stage",
        "data-lesson-notes",
        "legacyMarkdownPath",
        "pulseLabReadout",
        "IntersectionObserver",
        "motion-point",
        "motion-trace",
        "--cell-delay",
    ]
    for fragment in required_js_fragments:
        if fragment not in js_text:
            failures.append(f"assets/site.js: 缺少前端路由/课程功能 {fragment}")

    for fragment in ("renderLessonCheckLab", 'data-lab="lesson-check"'):
        if fragment in js_text:
            failures.append(f"assets/site.js: 不应再保留旧的空学习检查 fallback {fragment}")
    for fragment in ("Course File", "Personal Learning System", "Central Practice Console"):
        if fragment in html_text or fragment in js_text:
            failures.append(f"静态站仍有开发者/旧导航文案残留：{fragment}")
    for fragment in (
        "return values.slice(",
        "return headings.slice(",
        "parsed.outline.length <",
        ".slice(0, 8)",
    ):
        if fragment in js_text:
            failures.append(f"assets/site.js: 知识点内容仍有旧截断逻辑 {fragment}")

    if "python -m http.server %PORT% --bind 127.0.0.1" not in start_text:
        failures.append("start_lab.bat: 默认启动命令仍未切换到静态 HTML 服务")
    if "set \"PORTS=8000 8001 8002 8003 4173 5173 5500\"" not in start_text:
        failures.append("start_lab.bat: 缺少备用端口池")
    if "s.bind(('127.0.0.1',int(sys.argv[1])))" not in start_text:
        failures.append("start_lab.bat: 缺少真实端口绑定探测")
    if "goto :serve" not in start_text:
        failures.append("start_lab.bat: 缺少端口成功后启动服务的降级逻辑")
    if "streamlit run" in start_text.lower():
        failures.append("start_lab.bat: 默认启动脚本仍包含 Streamlit 启动命令")
    if "run_static_site(args.port)" not in main_text:
        failures.append("main.py: 无参数默认入口没有启动静态 HTML 站")
    if "run_streamlit_app(Path(__file__).resolve())" in main_text:
        failures.append("main.py: 默认入口仍会启动 Streamlit 主页")
    if "DEFAULT_STATIC_PORTS = (8000, 8001, 8002, 8003, 4173, 5173, 5500)" not in local_runtime_text:
        failures.append("components/local_runtime.py: 缺少与 start_lab.bat 一致的静态站端口池")
    if "def render_streamlit_migration_notice" not in streamlit_home_text or "主站不在 Streamlit 里了" not in streamlit_home_text:
        failures.append("components/streamlit_home.py: streamlit run main.py 没有明确的静态站迁移提示")
    if "if not deps.query_module:" not in streamlit_home_text or "render_streamlit_migration_notice()" not in streamlit_home_text:
        failures.append("components/streamlit_home.py: Streamlit 无模块参数入口没有被迁移提示挡住")
    if "query_module=get_query_module()" not in main_text:
        failures.append("main.py: Streamlit 首页壳层未传入 query_module")
    if "python main.py --menu" not in readme_text:
        failures.append("README.md: 模块列表命令仍可能误导用户使用默认入口")
    if "不要用 `streamlit run main.py` 启动主站" not in readme_text:
        failures.append("README.md: 缺少 Streamlit 不再是主站入口的说明")
    if "学习成果档案" not in readme_text or "深度实验室" not in readme_text:
        failures.append("README.md: 缺少学习成果档案/深度实验室说明")
    if "拖拽式节点画布" not in readme_text or "训练事件总线" not in readme_text:
        failures.append("README.md: 缺少 #10 中央控制台节点画布/事件总线说明")
    if "Node 24" not in readme_text:
        failures.append("README.md: 缺少 GitHub Actions Node 24 兼容说明")
    if "actions/checkout@v6" not in workflow_text:
        failures.append(".github/workflows/quality.yml: checkout 仍未升级到 Node 24 兼容的 v6")
    if "actions/setup-python@v6" not in workflow_text:
        failures.append(".github/workflows/quality.yml: setup-python 仍未升级到 Node 24 兼容的 v6")
    deprecated_streamlit_width_pattern = re.compile(r"\buse_container_width\s*=")
    for rel_path in project_files((".py",)):
        if rel_path == Path("scripts/quality_check.py"):
            continue
        for line_no, line in enumerate(read_text(rel_path).splitlines(), start=1):
            if deprecated_streamlit_width_pattern.search(line):
                failures.append(f"{rel_path}:{line_no}: Streamlit 未来弃用项 use_container_width 尚未清理")

    node = shutil.which("node")
    if node:
        completed = subprocess.run(
            [node, "--check", str(ROOT / "assets" / "site.js")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if completed.returncode != 0:
            failures.append("assets/site.js: node --check 失败\n" + completed.stderr.strip())

    if failures:
        raise CheckFailure("静态 HTML 站点检查失败：\n" + "\n".join(failures))
    print("[通过] 静态 HTML 站点检查：首页、CSS、JS 路由、原生交互实验、返回入口、启动脚本和配色约束均正常")


def check_content_credibility_system() -> None:
    failures: list[str] = []
    source_catalog = read_text(Path("docs/references/source_catalog.md"))
    credibility_doc = read_text(Path("docs/content_credibility.md"))
    audit_matrix = read_text(Path("docs/content_audit_matrix.md"))
    attention_md = read_text(Path("deep_learning_book/part4_transformer/01_attention_mechanism.md"))
    attention_legacy_md = read_text(Path("docs/legacy_book/part4_transformer/01_attention_mechanism.md"))
    site_js = read_text(Path("assets/site.js"))
    site_css = read_text(Path("assets/site.css"))

    for fragment in (
        "Attention Is All You Need",
        "Neural Machine Translation by Jointly Learning to Align and Translate",
        "torch.nn.functional.scaled_dot_product_attention",
        "Dive into Deep Learning",
        "CS231n",
        "Long Short-Term Memory",
        "PostgreSQL Documentation",
        "Operating Systems: Three Easy Pieces",
        "不复制受版权保护书籍或非授权资源站正文",
    ):
        if fragment not in source_catalog:
            failures.append(f"docs/references/source_catalog.md: 缺少来源或版权边界 {fragment}")

    for fragment in ("已校对", "教学简化", "待复核", "强结论要有边界"):
        if fragment not in credibility_doc:
            failures.append(f"docs/content_credibility.md: 缺少可信度规范 {fragment}")
    for fragment in ("全站逐章内容校对矩阵", "注意力机制", "卷积直觉", "RNN 直觉", "自测刷题模式"):
        if fragment not in audit_matrix:
            failures.append(f"docs/content_audit_matrix.md: 缺少逐章校对记录 {fragment}")

    for rel_path, text in (
        ("deep_learning_book/part4_transformer/01_attention_mechanism.md", attention_md),
        ("docs/legacy_book/part4_transformer/01_attention_mechanism.md", attention_legacy_md),
    ):
        for fragment in ("内容可信度与来源", "已校对样板", "https://arxiv.org/abs/1706.03762", "https://arxiv.org/abs/1409.0473"):
            if fragment not in text:
                failures.append(f"{rel_path}: 注意力样板缺少来源尾注 {fragment}")

    a_level_modules = {
        "part1/math_primer": ["D2L", "DLBOOK", "MLCC", "PYTORCH_AUTOGRAD"],
        "part1/01_tensors_gradients": ["PYTORCH_AUTOGRAD", "D2L", "DLBOOK", "PYTORCH_NN"],
        "part2/01_convolution_visual": ["CS231N", "PYTORCH_NN", "LENET1998", "D2L"],
        "part3/01_rnn_intuition": ["D2L", "PYTORCH_RNN", "LSTM1997", "CHO2014"],
        "part4/02_multihead_visual": ["VAS2017", "PYTORCH_SDPA", "PYTORCH_TRANSFORMER", "D2L"],
    }
    for module_id, source_ids in a_level_modules.items():
        match = re.search(rf'"{re.escape(module_id)}":\s*\{{(.*?)\n  \}},', site_js, re.S)
        if not match:
            failures.append(f"assets/site.js: 缺少 A 级可信度模块 {module_id}")
            continue
        body = match.group(1)
        for fragment in ('level: "A"', "已校对", "boundaries", "sources"):
            if fragment not in body:
                failures.append(f"assets/site.js: {module_id} 可信度元数据缺少 {fragment}")
        for source_id in source_ids:
            if source_id not in body:
                failures.append(f"assets/site.js: {module_id} 缺少来源 {source_id}")

    for page_name in ("数学基础速查", "张量与梯度", "卷积直觉", "RNN 直觉", "多头注意力可视化"):
        if f"| {page_name} | A |" not in audit_matrix:
            failures.append(f"docs/content_audit_matrix.md: {page_name} 未标记为 A")

    checked_markdown_sources = {
        Path("deep_learning_book/part1_foundations/01_tensors_gradients.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://pytorch.org/docs/stable/notes/autograd.html",
            "https://www.deeplearningbook.org/",
        ],
        Path("docs/legacy_book/part1_foundations/01_tensors_gradients.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://pytorch.org/docs/stable/notes/autograd.html",
            "https://www.deeplearningbook.org/",
        ],
        Path("deep_learning_book/part2_cnn/01_convolution_visual.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://cs231n.github.io/convolutional-networks/",
            "https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html",
        ],
        Path("docs/legacy_book/part2_cnn/01_convolution_visual.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://cs231n.github.io/convolutional-networks/",
            "https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html",
        ],
        Path("deep_learning_book/part3_rnn/01_rnn_intuition.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://pytorch.org/docs/stable/nn.html#recurrent-layers",
            "https://arxiv.org/abs/1406.1078",
        ],
        Path("docs/legacy_book/part3_rnn/01_rnn_intuition.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://pytorch.org/docs/stable/nn.html#recurrent-layers",
            "https://arxiv.org/abs/1406.1078",
        ],
        Path("deep_learning_book/part4_transformer/02_multihead_visual.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://arxiv.org/abs/1706.03762",
            "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
        ],
        Path("docs/legacy_book/part4_transformer/02_multihead_visual.md"): [
            "内容可信度与来源",
            "可信度：已校对",
            "https://arxiv.org/abs/1706.03762",
            "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
        ],
    }
    for path, fragments in checked_markdown_sources.items():
        text = read_text(path)
        for fragment in fragments:
            if fragment not in text:
                failures.append(f"{path}: 主线章节来源尾注缺少 {fragment}")

    for fragment in ("SOURCE_LIBRARY", "CONTENT_CREDIBILITY", "CREDIBILITY_PROFILES", "credibilityProfileForModule", "renderCredibilitySection", "data-content-credibility"):
        if fragment not in site_js:
            failures.append(f"assets/site.js: 缺少可信度渲染能力 {fragment}")
    for source_id in ("CS231N", "LSTM1997", "GRADCAM2017", "MDN_HTTP", "POSTGRES", "OSTEP"):
        if source_id not in site_js:
            failures.append(f"assets/site.js: 来源库缺少 {source_id}")
    for fragment in (".credibility-section", ".credibility-badge.level-a", ".reference-list"):
        if fragment not in site_css:
            failures.append(f"assets/site.css: 缺少可信度样式 {fragment}")

    if failures:
        raise CheckFailure("内容可信度系统检查失败：\n" + "\n".join(failures))
    print("[通过] 内容可信度系统检查：来源库、可信度规范、注意力样板和前端来源区均已接入")


def check_zero_basics_teaching_notes() -> None:
    js_text = read_text(Path("assets/site.js"))
    failures: list[str] = []
    modules_match = re.search(r"const MODULES = \[(.*?)\]\.map", js_text, re.S)
    if not modules_match:
        raise CheckFailure("零基础十二问覆盖检查失败：assets/site.js 缺少 MODULES 清单")

    module_ids = [
        f"{match.group(1)}/{match.group(2)}"
        for match in re.finditer(
            r'\["(part\d+)",\s*"[^"]+",\s*"[^"]+",\s*"([^"]+)"',
            modules_match.group(1),
        )
    ]
    notes_match = re.search(r"const MODULE_TEACHING_NOTES = \{(.*?)\n\};\n\nfunction lessonDomain", js_text, re.S)
    if not notes_match:
        raise CheckFailure("零基础十二问覆盖检查失败：assets/site.js 缺少 MODULE_TEACHING_NOTES 块")
    notes_text = notes_match.group(1)

    note_ids = set(re.findall(r'"(part\d+/[^"]+)":\s*\{', notes_text))
    missing = [module_id for module_id in module_ids if module_id not in note_ids]
    if missing:
        failures.append("缺少 MODULE_TEACHING_NOTES 人工答案种子：" + "、".join(missing))

    required_fields = [
        "what",
        "analogy",
        "intuition",
        "variable",
        "elements",
        "controls",
        "observe",
        "why",
        "misconception",
        "engineering",
        "consoleTask",
    ]
    for module_id in module_ids:
        block_match = re.search(rf'"{re.escape(module_id)}":\s*\{{(.*?)\n\s*\}},', notes_text, re.S)
        if not block_match:
            continue
        block = block_match.group(1)
        for field in required_fields:
            if not re.search(rf"\b{field}\s*:", block):
                failures.append(f"{module_id}: MODULE_TEACHING_NOTES 缺少字段 {field}")

    if "const note = MODULE_TEACHING_NOTES[module.id] || {}" not in js_text:
        failures.append("renderZeroBasics 没有优先读取 MODULE_TEACHING_NOTES")
    if 'if (module.partKey === "part1") return "foundation";' not in js_text:
        failures.append("lessonDomain 没有把第一部分基础章节固定为 foundation，可能被训练关键词误分类")

    if failures:
        raise CheckFailure("零基础十二问覆盖检查失败：\n" + "\n".join(failures))
    print(f"[通过] 零基础十二问覆盖检查：{len(module_ids)} 个模块均有人工答案种子")


def check_playground_codegen() -> None:
    namespace = load_module_without_main(Path("part6_universal_framework/neural_network_playground.py"))
    registry = namespace["COMPONENT_REGISTRY"]
    presets = namespace["PRESETS"]
    infer_shapes = namespace["infer_shapes"]
    generate_code = namespace["generate_code"]

    required_components = {
        "Linear",
        "Conv2d",
        "MaxPool2d",
        "LayerNorm",
        "ResidualBlock",
        "MultiheadAttention",
        "TransformerEncoder",
        "Flatten",
    }
    missing_components = sorted(required_components - set(registry))
    failures: list[str] = []
    if missing_components:
        failures.append("组件注册表缺少：" + ", ".join(missing_components))

    required_presets = {"mlp", "cnn", "transformer", "residual_mlp"}
    missing_presets = sorted(required_presets - set(presets))
    if missing_presets:
        failures.append("预设模型缺少：" + ", ".join(missing_presets))

    for preset_name in sorted(required_presets & set(presets)):
        preset = presets[preset_name]
        steps = infer_shapes(tuple(preset["input_shape"]), preset["layers"])
        bad_steps = [step for step in steps if not step.ok]
        if bad_steps:
            failures.append(f"{preset_name}: shape 推导失败：{bad_steps[0].message}")
            continue
        code = generate_code(
            tuple(preset["input_shape"]),
            preset["layers"],
            steps,
            preset["loss"],
            preset["optimizer"],
            namespace["OPTIMIZER_REGISTRY"][preset["optimizer"]]["params"],
        )
        try:
            compile(code, f"<generated:{preset_name}>", "exec")
            exec_namespace: dict[str, object] = {}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exec(code, exec_namespace)
        except Exception as exc:  # noqa: BLE001 - generated code must be executable.
            failures.append(f"{preset_name}: 生成代码不可执行：{exc}")

    if failures:
        raise CheckFailure("中央控制台代码生成检查失败：\n" + "\n".join(failures))
    print(f"[通过] 中央控制台组件与代码生成检查：{len(required_presets)} 个预设")


def check_playground_training_linkage() -> None:
    namespace = load_module_without_main(Path("part6_universal_framework/neural_network_playground.py"))
    presets = namespace["PRESETS"]
    infer_shapes = namespace["infer_shapes"]
    run_playground_training = namespace["run_playground_training"]

    failures: list[str] = []
    mlp = presets["mlp"]
    mlp_steps = infer_shapes(tuple(mlp["input_shape"]), mlp["layers"])
    try:
        mlp_history = run_playground_training(
            tuple(mlp["input_shape"]),
            mlp["layers"],
            mlp_steps,
            mlp["loss"],
            mlp["optimizer"],
            namespace["OPTIMIZER_REGISTRY"][mlp["optimizer"]]["params"],
            epochs=2,
            seed=11,
            batch_size=4,
        )
        if len(mlp_history.losses) != 2:
            failures.append("MLP 联动训练没有记录 2 个 epoch 的损失")
        if not mlp_history.grad_norms or not any(any(value > 0 for value in values) for values in mlp_history.grad_norms.values()):
            failures.append("MLP 联动训练没有记录有效梯度流")
        if not mlp_history.update_ratios or not any(value > 0 for value in mlp_history.update_ratios):
            failures.append("MLP 联动训练没有记录参数更新幅度")
    except Exception as exc:  # noqa: BLE001 - report linkage failures together.
        failures.append(f"MLP 联动训练运行失败：{exc}")

    cnn = presets["cnn"]
    cnn_steps = infer_shapes(tuple(cnn["input_shape"]), cnn["layers"])
    try:
        cnn_history = run_playground_training(
            tuple(cnn["input_shape"]),
            cnn["layers"],
            cnn_steps,
            cnn["loss"],
            cnn["optimizer"],
            namespace["OPTIMIZER_REGISTRY"][cnn["optimizer"]]["params"],
            epochs=1,
            seed=13,
            batch_size=2,
        )
        if cnn_history.cnn_feature_maps is None:
            failures.append("CNN 联动训练没有生成卷积特征图")
    except Exception as exc:  # noqa: BLE001 - report linkage failures together.
        failures.append(f"CNN 联动训练运行失败：{exc}")

    transformer = presets["transformer"]
    transformer_steps = infer_shapes(tuple(transformer["input_shape"]), transformer["layers"])
    try:
        transformer_history = run_playground_training(
            tuple(transformer["input_shape"]),
            transformer["layers"],
            transformer_steps,
            transformer["loss"],
            transformer["optimizer"],
            namespace["OPTIMIZER_REGISTRY"][transformer["optimizer"]]["params"],
            epochs=1,
            seed=17,
            batch_size=2,
        )
        if transformer_history.attention_heatmap is None:
            failures.append("Transformer 联动训练没有生成注意力热力图")
    except Exception as exc:  # noqa: BLE001 - report linkage failures together.
        failures.append(f"Transformer 联动训练运行失败：{exc}")

    if failures:
        raise CheckFailure("中央控制台训练联动检查失败：\n" + "\n".join(failures))
    print("[通过] 中央控制台训练联动检查：损失、梯度、参数更新、CNN 特征图、注意力热力图均可生成")


def load_module(rel_path: Path) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return runpy.run_path(str(ROOT / rel_path), run_name="__main__")


def load_module_without_main(rel_path: Path) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return runpy.run_path(str(ROOT / rel_path), run_name="__quality_check__")


def runpy_no_bytecode(path: Path, run_name: str = "<run_path>") -> dict[str, object]:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return runpy.run_path(str(path), run_name=run_name)
    finally:
        sys.dont_write_bytecode = previous


def call_smoke_function(namespace: dict[str, object], name: str) -> None:
    func = namespace[name]
    if name == "render_overview":
        func(2)
    elif name == "render_self_attention":
        func("The cat sat on the mat because it was tired", 16, 7)
    elif name == "render_multihead":
        pack = namespace["compute_attention"](namespace["tokenize"]("The cat sat on the mat because it was tired"), 16, 7)
        func(pack, 1.4)
    elif name == "render_text_heatmap":
        pack = namespace["compute_attention"](namespace["tokenize"]("The cat sat on the mat because it was tired"), 16, 7)
        func(pack)
    elif name.startswith("render_") and name in {
        "render_linear_regression",
        "render_logistic_regression",
        "render_decision_tree",
        "render_kmeans",
        "render_svm",
        "render_knn",
    }:
        func(42)
    elif name == "render_forward_guide":
        arch = namespace["ARCHITECTURES"][0]
        func(arch, 1)
    elif name == "render_feature_map_guide":
        func("几何图形", "浅层：边缘/方向", (1, 6, 64, 64))
    elif name == "render_residual_guide":
        func(0.7, False)
    elif name == "render_inception_guide":
        func("亮斑目标", 4)
    elif name == "render_conv_experiment_guide":
        result_cls = namespace["ConvResult"]
        import torch

        result = result_cls(torch.zeros(1, 4, 32, 32), "smoke formula", 144, 3)
        func("标准卷积", 32, 4, 4, 3, 1, 1, 1, 2, 0, result)
    elif name == "render_pooling_guide":
        func(2, 2)
    elif name == "render_bn_guide":
        func(16, 6)
    elif name == "render_dropout_guide":
        func(0.35, True, 0.35)
    elif name == "render_rnn_unroll_guide":
        func(7, 4, 0.85, 8, 1.0)
    elif name == "render_gradient_issue_guide":
        func(60, 0.98, 0.12, {"effective_gain": 0.862, "final_norm": 1e-4})
    elif name == "render_lstm_guide":
        func(1.0, 0.0, 0.3, 1.1)
    elif name == "render_gru_guide":
        func(0.4, 0.0, 1.1)
    elif name == "render_bidirectional_guide":
        func(8, "concat")
    elif name == "render_forecast_guide":
        func("LSTM", 32, 32, 1, 0.003, 90, 0.08, 0.0, 0.0123)
    elif name == "render_text_generation_guide":
        func("唐诗风格小样本", 64, 260, 0.01, 50, 0.8)
    else:
        func()


def check_focus_smoke() -> None:
    failures: list[str] = []
    old_env = os.environ.copy()
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        for rel_path, function_names in SMOKE_FUNCTIONS.items():
            try:
                namespace = load_module(rel_path)
                for name in function_names:
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        call_smoke_function(namespace, name)
            except Exception as exc:  # noqa: BLE001 - smoke test should report every runtime failure.
                failures.append(f"{rel_path}: {exc}")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    if failures:
        raise CheckFailure("重点页面渲染 smoke 失败：\n" + "\n".join(failures))
    print(f"[通过] 重点页面渲染 smoke：{len(SMOKE_FUNCTIONS)} 个页面")


def check_strict_legacy_smoke() -> None:
    failures: list[str] = []
    old_env = os.environ.copy()
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        for rel_path in STRICT_LEGACY_PROTOCOL_FILES:
            try:
                namespace = load_module_without_main(rel_path)
                smoke = namespace.get("smoke")
                if not callable(smoke):
                    failures.append(f"{rel_path}: smoke 不是可调用对象")
                    continue
                result = smoke()
                if result is False:
                    failures.append(f"{rel_path}: smoke() 返回 False")
            except Exception as exc:  # noqa: BLE001 - smoke failures should be reported together.
                failures.append(f"{rel_path}: {exc}")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    if failures:
        raise CheckFailure("严格老脚本 smoke 失败：\n" + "\n".join(failures))
    print(f"[通过] 严格老脚本 smoke：{len(STRICT_LEGACY_PROTOCOL_FILES)} 个文件")


def check_main_routes() -> None:
    namespace = runpy_no_bytecode(ROOT / "main.py")
    routes = namespace["route_map"]()
    missing = [route for route in FOCUS_ROUTES if route not in routes]
    if missing:
        raise CheckFailure("主站路由检查失败，缺少：\n" + "\n".join(missing))
    print(f"[通过] 主站重点路由检查：{len(FOCUS_ROUTES)} 条路由")


def route_from_knowledge_url(url: str) -> str:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("module", [])
    return unquote(values[0]) if values else ""


def check_knowledge_graph_routes() -> None:
    main_namespace = runpy_no_bytecode(ROOT / "main.py")
    graph_namespace = runpy_no_bytecode(ROOT / "components" / "knowledge_graph.py")
    routes = main_namespace["route_map"]()
    modules = main_namespace["MODULES"]
    graph = graph_namespace["KNOWLEDGE_GRAPH"]
    module_url = graph_namespace["_module_url"]
    practice_url = graph_namespace["practice_url"]
    get_node = graph_namespace["get_node"]
    canonical_node_keys = graph_namespace["canonical_node_keys"]

    failures: list[str] = []
    required_routes = {module.short_target for module in modules}
    canonical_keys = set(canonical_node_keys())
    missing = sorted(required_routes - canonical_keys)
    if missing:
        failures.append("知识图谱缺少主站模块节点：\n" + "\n".join(missing))

    for key in sorted(canonical_keys):
        node = get_node(key)
        if node is None:
            failures.append(f"{key}: canonical_node_keys() 中存在，但 KNOWLEDGE_GRAPH 无法解析")
            continue
        route = route_from_knowledge_url(module_url(key))
        if route not in routes:
            failures.append(f"{key}: 理论页映射到 {route or '<empty>'}，但 main.py route_map 中没有对应路由")
        practice_route = route_from_knowledge_url(practice_url(key))
        if practice_route not in routes:
            failures.append(f"{key}: 实战页映射到 {practice_route or '<empty>'}，但 main.py route_map 中没有对应路由")
        for field_name in ("mastery_criteria", "practice_target", "practice_route", "route"):
            value = getattr(node, field_name, "")
            if not isinstance(value, str) or not value.strip():
                failures.append(f"{key}: 缺少 {field_name}")
        if not node.related:
            failures.append(f"{key}: 缺少相关知识 related")
        if not node.next_steps:
            failures.append(f"{key}: 缺少后续推荐 next_steps")
        for relation_name in ("prerequisites", "related", "next_steps"):
            for target in getattr(node, relation_name):
                if get_node(target) is None:
                    failures.append(f"{key}: {relation_name} 引用了不存在的节点 {target}")

    for key in sorted(graph):
        route = route_from_knowledge_url(module_url(key))
        if route not in routes:
            failures.append(f"{key}: 别名映射到 {route or '<empty>'}，但 main.py route_map 中没有对应路由")
    if failures:
        raise CheckFailure("知识图谱元数据完整性检查失败：\n" + "\n".join(failures))
    print(f"[通过] 知识图谱元数据完整性检查：{len(canonical_keys)} 个主站模块，{len(graph)} 个含别名节点")


def check_legacy_book_import() -> None:
    manifest_path = ROOT / "docs" / "legacy_book" / "manifest.json"
    if not manifest_path.exists():
        raise CheckFailure("旧教材迁移检查失败：缺少 docs/legacy_book/manifest.json")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed JSON should be reported.
        raise CheckFailure(f"旧教材迁移检查失败：manifest.json 无法解析：{exc}") from exc

    lessons = payload.get("lessons")
    failures: list[str] = []
    if not isinstance(lessons, dict):
        failures.append("manifest.json 缺少 lessons 对象")
        lessons = {}
    lesson_count = int(payload.get("lesson_count", 0) or 0)
    if lesson_count < 38 or len(lessons) < 38:
        failures.append(f"旧教材章节数量不足：lesson_count={lesson_count}, lessons={len(lessons)}")

    required_routes = [
        "part1/01_tensors_gradients",
        "part2/01_convolution_visual",
        "part3/05_seq2seq_attention",
        "part4/01_attention_mechanism",
        "part5/02_gradient_monitor",
        "part6/07_project_template",
    ]
    for route in required_routes:
        item = lessons.get(route)
        if not isinstance(item, dict):
            failures.append(f"旧教材缺少关键章节：{route}")
            continue
        local_path = item.get("local_path")
        title = item.get("title")
        outline = item.get("outline")
        if not isinstance(local_path, str) or not (ROOT / "docs" / "legacy_book" / local_path).exists():
            failures.append(f"{route}: local_path 不存在或文件缺失")
        if not isinstance(title, str) or not title.strip():
            failures.append(f"{route}: 缺少标题")
        if not isinstance(outline, list) or not outline:
            failures.append(f"{route}: 缺少章节大纲")

    namespace = load_module_without_main(Path("components/legacy_book.py"))
    get_legacy_lesson = namespace["get_legacy_lesson"]
    if get_legacy_lesson("part4_transformer/01_attention_mechanism") is None:
        failures.append("legacy_book.get_legacy_lesson 无法通过长目录路由找到注意力旧教材")

    if failures:
        raise CheckFailure("旧教材迁移检查失败：\n" + "\n".join(failures))
    print(f"[通过] 旧教材迁移检查：{len(lessons)} 个 Markdown 章节已导入 docs/legacy_book")


def check_learning_progress_system() -> None:
    namespace = load_module_without_main(Path("components/progress_tracker.py"))
    failures: list[str] = []

    statuses = tuple(namespace["PROGRESS_STATUSES"])
    for status in ("未学习", "已学习", "已掌握", "去实战", "稍后复习"):
        if status not in statuses:
            failures.append(f"PROGRESS_STATUSES 缺少 {status}")

    normalize = namespace["normalize_module_key"]
    if normalize("part4_transformer/01_attention_mechanism") != "part4/01_attention_mechanism":
        failures.append("normalize_module_key 无法把长目录路由规范化为知识图谱短路由")

    progress = {
        "part1/math_primer": "已掌握",
        "part1/01_tensors_gradients": "已学习",
        "part4/01_attention_mechanism": "稍后复习",
        "part5/04_hyperparam_search": "去实战",
    }
    profile = {
        "review_later": {
            "part4/01_attention_mechanism": {
                "reason": "softmax 权重解释还不稳",
                "priority": "高",
                "created_at": "2026-05-24 12:00",
            }
        },
        "records": [
            {
                "id": "r1",
                "module_key": "part4/01_attention_mechanism",
                "type": "错题",
                "title": "Q/K/V 混淆",
                "note": "把 key 当成数据库主键。",
                "reflection": "下次先画 query-key-value 三角色。",
                "created_at": "2026-05-24 12:01",
                "linked_nodes": ["part4/01_attention_mechanism"],
            },
            {
                "id": "r2",
                "module_key": "part5/04_hyperparam_search",
                "type": "实验记录",
                "title": "学习率过高",
                "note": "loss 震荡。",
                "reflection": "先降 3 倍。",
                "created_at": "2026-05-24 12:02",
                "linked_nodes": ["part5/04_hyperparam_search"],
            },
        ],
    }

    report = namespace["build_learning_report"](progress, profile)
    if report["review_count"] < 1:
        failures.append("学习报告没有统计稍后复习数量")
    if report["record_counts"].get("错题", 0) < 1:
        failures.append("学习报告没有统计错题记录")
    if report["record_counts"].get("实验记录", 0) < 1:
        failures.append("学习报告没有统计实验记录")

    today = namespace["recommend_today"](progress, profile)
    if not today or today.get("key") != "part4/01_attention_mechanism":
        failures.append("今日推荐没有优先选择高优先级稍后复习章节")

    weaknesses = namespace["analyze_weaknesses"](progress, profile, limit=3)
    if not weaknesses or weaknesses[0]["key"] != "part4/01_attention_mechanism":
        failures.append("弱点分析没有把错题和稍后复习联动到知识图谱节点")

    if failures:
        raise CheckFailure("学习进度系统检查失败：\n" + "\n".join(failures))
    print("[通过] 学习进度系统检查：稍后复习、今日推荐、学习报告、弱点分析、错题/实验记录联动正常")


def check_bagu_routes_placeholder() -> None:
    namespace = load_module_without_main(Path("part7_interview/interview_quiz.py"))
    failures: list[str] = []

    questions = namespace["QUESTIONS"]
    directions = {"网络", "数据库", "算法", "操作系统", "深度学习", "系统设计"}
    for direction in directions:
        count = sum(1 for item in questions if item.direction == direction)
        if count < 8:
            failures.append(f"{direction}: 题库数量不足 8 题，当前 {count} 题")

    score_user_answer = namespace["score_user_answer"]
    sample = next(item for item in questions if item.direction == "网络")
    strong = score_user_answer(sample, sample.answer + " 工程上还要结合模型推理 API 的延迟、监控和重试。")
    weak = score_user_answer(sample, "不知道")
    if strong["score"] <= weak["score"]:
        failures.append("自动评分没有区分强答案和弱答案")
    if strong["score"] < 70:
        failures.append("自动评分对标准答案给分过低")

    for name in ("QUESTION_MODULE_MAP", "SIMULATION_SCRIPTS", "SCORING_DIMENSIONS"):
        if name not in namespace:
            failures.append(f"面试训练缺少 {name}")

    scripts = namespace["SIMULATION_SCRIPTS"]
    if not all(len(flow) >= 5 for flow in scripts.values()):
        failures.append("模拟面试脚本没有覆盖至少 5 轮追问")

    route_files = {
        Path("part7_interview/networking.py"): "进入网络专项刷题",
        Path("part7_interview/database_sql.py"): "进入数据库专项刷题",
        Path("part7_interview/data_structures.py"): "进入算法专项刷题",
        Path("part7_interview/operating_system.py"): "进入操作系统专项刷题",
    }
    for rel_path, fragment in route_files.items():
        if fragment not in read_text(rel_path):
            failures.append(f"{rel_path}: 缺少专项刷题入口")

    if failures:
        raise CheckFailure("八股文训练营检查失败：\n" + "\n".join(failures))
    print("[通过] 八股文训练营检查：题库、自动评分、模拟面试、错题持久化和专项入口均已覆盖")


def check_static_navigation_entry() -> None:
    html_text = read_text(Path("index.html"))
    js_text = read_text(Path("assets/site.js"))
    css_text = read_text(Path("assets/site.css"))
    failures: list[str] = []

    if 'class="brand" href="#home"' not in html_text:
        failures.append("index.html: 品牌入口没有指向 #home")
    if 'href="#courses"' not in html_text or 'href="#path"' not in html_text:
        failures.append("index.html: 顶部导航缺少课程/路径锚点")
    if "返回首页" not in js_text:
        failures.append("assets/site.js: 课程页缺少可见的返回首页入口")
    if "hash.startsWith(\"#course/\")" not in js_text:
        failures.append("assets/site.js: 缺少 #course/ hash 路由处理")
    if "hash.startsWith(\"#console/\")" not in js_text:
        failures.append("assets/site.js: 缺少 #console/ 中央控制台路由处理")
    if "decodeURIComponent(location.hash" not in js_text:
        failures.append("assets/site.js: 路由没有解码 hash，中文/斜杠路由可能失效")
    if ".side-drawer.is-open" not in css_text:
        failures.append("assets/site.css: 目录抽屉缺少打开状态样式")
    if ".code-window" not in css_text or "resize: vertical" not in css_text:
        failures.append("assets/site.css: 源码窗口缺少可拉伸样式")

    if failures:
        raise CheckFailure("静态站导航入口检查失败：\n" + "\n".join(failures))
    print("[通过] 静态站导航入口检查：顶部导航、返回首页、目录抽屉和课程路由均正常")


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def is_guarded_main_if(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
    return "__name__" in test and "__main__" in test


def check_legacy_top_level_execution() -> None:
    failures: list[str] = []
    for rel_path in LEGACY_PROTOCOL_FILES:
        tree = parse_python(rel_path)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                failures.append(f"{rel_path}:{node.lineno}: 发现未包裹在 if __name__ 或 try/except 中的顶层调用")
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node_contains_call(node, "plt.show"):
                failures.append(f"{rel_path}:{node.lineno}: 发现未包裹在 if __name__ 或 try/except 中的顶层绘图执行")
    if failures:
        raise CheckFailure("老脚本顶层危险执行逻辑检查失败：\n" + "\n".join(failures))
    print(f"[通过] 老脚本顶层危险执行逻辑检查：{len(LEGACY_PROTOCOL_FILES)} 个文件")


def node_contains_call(node: ast.AST, expected: str) -> bool:
    return any(call_name(child) == expected for child in ast.walk(node) if isinstance(child, ast.Call))


def check_statement_list_for_unclosed_matplotlib(body: list[ast.stmt], rel_path: Path, failures: list[str]) -> None:
    for index, stmt in enumerate(body):
        if node_contains_call(stmt, "plt.show"):
            has_close_after = any(node_contains_call(next_stmt, "plt.close") for next_stmt in body[index + 1 :])
            if not has_close_after:
                failures.append(f"{rel_path}:{stmt.lineno}: plt.show() 后缺少 plt.close()")


def check_matplotlib_close_after_show() -> None:
    failures: list[str] = []
    runner_text = read_text(Path("legacy_runner.py"))
    if "plt.show = save_open_figures" not in runner_text or 'plt.close("all")' not in runner_text:
        failures.append("legacy_runner.py: 缺少 plt.show 统一替换或 plt.close(\"all\") 清理")

    legacy_roots = {
        "part1_foundations",
        "part2_cnn",
        "part3_rnn",
        "part4_transformer",
        "part5_toolbox",
        "part6_universal_framework",
    }
    scan_files = [
        rel_path
        for rel_path in project_files((".py",))
        if rel_path.parts[0] not in legacy_roots and rel_path != Path("legacy_runner.py")
    ]
    for rel_path in scan_files:
        tree = parse_python(rel_path)
        check_statement_list_for_unclosed_matplotlib(tree.body, rel_path, failures)
        for node in ast.walk(tree):
            for field_name in ("body", "orelse", "finalbody"):
                body = getattr(node, field_name, None)
                if isinstance(body, list) and all(isinstance(item, ast.stmt) for item in body):
                    check_statement_list_for_unclosed_matplotlib(body, rel_path, failures)
    if failures:
        raise CheckFailure("Matplotlib 图未关闭检查失败：\n" + "\n".join(failures))
    print(f"[通过] Matplotlib 图未关闭检查：legacy_runner 统一关闭旧脚本图像，额外扫描 {len(scan_files)} 个非旧脚本文件")


def check_visual_system_integration() -> None:
    """Check that visual_system.py keeps the shared teaching components on the
    restrained premium palette while legacy Streamlit pages still import it."""
    failures: list[str] = []

    vs_path = Path("components/visual_system.py")
    if not (ROOT / vs_path).exists():
        raise CheckFailure("视觉系统集成检查失败：components/visual_system.py 不存在")

    text = read_visual_system_text()

    # --- Premium learning palette ---
    required_palette = {
        "BRAND_GOLD": "#b08a4f",
        "BRAND_INK": "#2a2118",
        "BRAND_BORDER": "#e6ded2",
        "BRAND_SOFT": "#f7f3ec",
    }
    for var_name, hex_val in required_palette.items():
        if var_name not in text:
            failures.append(f"{vs_path}: 缺少高级学习视觉变量 {var_name}")
        if hex_val not in text:
            failures.append(f"{vs_path}: 缺少高级学习配色 {hex_val}")
    forbidden_neon_values = ["#00f0ff", "#b000ff", "#00ff88"]
    for hex_val in forbidden_neon_values:
        if hex_val in text:
            failures.append(f"{vs_path}: 不应继续使用旧霓虹色值 {hex_val}")

    # --- Core motion / effect components (must never be deleted) ---
    required_motion_components = [
        "render_visual_system",
        "render_particle_field",
        "render_convolution_particle_flow",
        "render_gradient_descent_landscape",
        "render_attention_light_beams",
        "render_backprop_current_flow",
        "render_training_dashboard_gauges",
        "render_motion_gallery",
    ]
    for func_name in required_motion_components:
        if f"def {func_name}" not in text:
            failures.append(f"{vs_path}: 核心动效组件 {func_name} 被删除或重命名")

    # --- Font assets ---
    if "font-awesome" not in text.lower():
        failures.append(f"{vs_path}: 缺少 Font Awesome 引用")
    for font in ("Inter", "JetBrains Mono"):
        if font not in text:
            failures.append(f"{vs_path}: 缺少字体 {font} 引用")

    required_p1_components = [
        "render_tooltip_label",
        "render_motion_note",
        "render_neon_metric_card",
        "render_concept_animation_shell",
        "render_responsive_motion_grid",
        "render_shape_flow",
        "render_status_badge",
        "render_beginner_hint",
    ]
    for func_name in required_p1_components:
        if f"def {func_name}" not in text:
            failures.append(f"{vs_path}: P1 通用视觉组件 {func_name} 缺失")

    required_css_fragments = [
        "prefers-reduced-motion",
        "@media (max-width: 760px)",
        ":focus-visible",
        ".vs-tooltip",
        ".vs-loading",
        ".vs-chart-note",
        "图表说明",
    ]
    for fragment in required_css_fragments:
        if fragment not in text:
            failures.append(f"{vs_path}: 缺少视觉系统 CSS/说明约束 {fragment}")

    # --- Core pages must import render_visual_system ---
    VISUAL_SYSTEM_IMPORT_PAGES = [
        Path("components/streamlit_shell.py"),
        Path("components/legacy_page.py"),
        Path("part2_cnn/03_classic_architectures.py"),
        Path("part2_cnn/advanced_cnn.py"),
        Path("part4_transformer/transformer_models.py"),
        Path("part6_universal_framework/training_demo.py"),
    ]
    for page_path in VISUAL_SYSTEM_IMPORT_PAGES:
        if not (ROOT / page_path).exists():
            failures.append(f"{page_path}: 文件不存在，无法检查视觉系统接入")
            continue
        page_text = read_text(page_path)
        if "from components.visual_system import" not in page_text:
            failures.append(f"{page_path}: 未 import 视觉系统组件")
        if "render_visual_system" not in page_text:
            failures.append(f"{page_path}: 未调用 render_visual_system")

    # --- Pages that use specific motion effects ---
    MOTION_EFFECT_PAGES = {
        Path("part1_foundations/01_tensors_gradients.py"): [
            "render_gradient_descent_landscape",
            "render_backprop_current_flow",
        ],
        Path("part2_cnn/01_convolution_visual.py"): [
            "render_convolution_particle_flow",
        ],
        Path("part2_cnn/03_classic_architectures.py"): [
            "render_cnn_layer_pipeline",
        ],
        Path("part2_cnn/advanced_cnn.py"): [
            "render_advanced_conv_comparison",
        ],
    }
    for page_path, effects in MOTION_EFFECT_PAGES.items():
        if not (ROOT / page_path).exists():
            failures.append(f"{page_path}: 文件不存在，无法检查动效接入")
            continue
        page_text = read_text(page_path)
        for effect in effects:
            if effect not in page_text:
                failures.append(f"{page_path}: 未引用动效组件 {effect}")

    if failures:
        raise CheckFailure("视觉系统集成检查失败：\n" + "\n".join(failures))
    print("[通过] 视觉系统集成检查：高级学习配色、动效组件、字体资产、页面接入均正常")


def run_checks(include_smoke: bool) -> None:
    check_python_compile()
    check_placeholders()
    check_bracket_placeholders()
    check_legacy_module_protocol_metadata()
    check_strict_legacy_module_protocol()
    check_legacy_protocol_audit()
    check_expected_controls()
    check_expected_content()
    check_static_html_site()
    check_content_credibility_system()
    check_zero_basics_teaching_notes()
    check_visual_system_integration()
    check_playground_codegen()
    check_playground_training_linkage()
    check_main_routes()
    check_knowledge_graph_routes()
    check_legacy_book_import()
    check_learning_progress_system()
    check_bagu_routes_placeholder()
    run_domain_check(check_course_catalog_module)
    run_domain_check(check_course_source_of_truth)
    run_domain_check(check_legacy_page_module)
    run_domain_check(check_legacy_runtime_module)
    run_domain_check(check_local_runtime_module)
    run_domain_check(check_streamlit_home_module)
    run_domain_check(check_streamlit_shell_module)
    run_domain_check(check_playground_modules)
    run_domain_check(check_visual_system_modules)
    run_domain_check(check_navigation_and_learning_ux)
    check_legacy_top_level_execution()
    check_matplotlib_close_after_show()
    run_domain_check(check_artifact_redirection_domain)
    run_domain_check(check_artifact_runtime_behavior_domain)
    run_domain_check(check_legacy_script_artifact_run_domain)
    run_domain_check(check_root_artifacts_domain)
    if include_smoke:
        check_strict_legacy_smoke()
        check_focus_smoke()
    print("[完成] 内容质量检查全部通过")


def main() -> int:
    parser = argparse.ArgumentParser(description="深度学习交互式网站内容质量检查")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="跳过重点页面渲染 smoke，只做静态检查。",
    )
    args = parser.parse_args()
    try:
        run_checks(include_smoke=not args.skip_smoke)
    except CheckFailure as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
