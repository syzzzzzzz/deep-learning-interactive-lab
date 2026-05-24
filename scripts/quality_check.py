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
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]

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
    Path("part1_foundations/03_datasets_optimizers.py"),
    Path("part2_cnn/01_convolution_visual.py"),
    Path("part2_cnn/02_feature_maps.py"),
    Path("part3_rnn/01_rnn_intuition.py"),
    Path("part4_transformer/01_attention_mechanism.py"),
    Path("part4_transformer/02_multihead_visual.py"),
    Path("part5_toolbox/05_dataset_toys.py"),
]

STRICT_LEGACY_PROTOCOL_FILES = [
    Path("part1_foundations/01_tensors_gradients.py"),
    Path("part2_cnn/01_convolution_visual.py"),
    Path("part2_cnn/02_feature_maps.py"),
    Path("part3_rnn/01_rnn_intuition.py"),
    Path("part4_transformer/01_attention_mechanism.py"),
    Path("part4_transformer/02_multihead_visual.py"),
    Path("part5_toolbox/05_dataset_toys.py"),
]

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
        "学习导读",
        "render_module_knowledge_nav",
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
    return (ROOT / rel_path).read_text(encoding="utf-8", errors="replace")


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
        for fragment in expected_fragments:
            if fragment not in text:
                failures.append(f"{rel_path}: 缺少工程教学内容片段：{fragment}")
    if failures:
        raise CheckFailure("工程教学内容检查失败：\n" + "\n".join(failures))
    print(f"[通过] 工程教学内容检查：{len(EXPECTED_CONTENT_REFERENCES)} 个页面")


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
    namespace = runpy.run_path(str(ROOT / "main.py"))
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
    main_namespace = runpy.run_path(str(ROOT / "main.py"))
    graph_namespace = runpy.run_path(str(ROOT / "components" / "knowledge_graph.py"))
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


def check_bagu_routes_placeholder() -> None:
    """Reserved check for future interview-route batches."""

    print("[通过] 八股文路由检查：接口已预留，当前批次未定义八股文路由表")


def check_back_to_home_entry() -> None:
    text = read_text(Path("main.py"))
    failures: list[str] = []
    if "def render_home_button" not in text:
        failures.append("main.py: 缺少统一返回主界面入口 render_home_button()")
    if "render_legacy_module_page" not in text or "render_home_button()" not in text:
        failures.append("main.py: 老脚本页面缺少统一返回主界面入口调用")
    if "runpy.run_path" in text and text.count("render_home_button()") < 3:
        failures.append("main.py: Streamlit 路由外壳缺少返回主界面入口调用")
    if failures:
        raise CheckFailure("返回主界面入口检查失败：\n" + "\n".join(failures))
    print("[通过] 返回主界面入口检查：main.py 已为路由页面提供统一返回入口")


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


def run_checks(include_smoke: bool) -> None:
    check_python_compile()
    check_placeholders()
    check_bracket_placeholders()
    check_legacy_module_protocol_metadata()
    check_strict_legacy_module_protocol()
    check_expected_controls()
    check_expected_content()
    check_playground_codegen()
    check_playground_training_linkage()
    check_main_routes()
    check_knowledge_graph_routes()
    check_legacy_book_import()
    check_bagu_routes_placeholder()
    check_back_to_home_entry()
    check_legacy_top_level_execution()
    check_matplotlib_close_after_show()
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
