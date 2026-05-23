"""
内容质量检查入口。

运行：
    python scripts/quality_check.py
"""

from __future__ import annotations

import argparse
import ast
import io
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
    Path("part3_rnn/01_rnn_intuition.py"),
    Path("part4_transformer/01_attention_mechanism.py"),
]

FOCUS_ROUTES = [
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


def load_module(rel_path: Path) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return runpy.run_path(str(ROOT / rel_path), run_name="__main__")


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
    graph = graph_namespace["KNOWLEDGE_GRAPH"]
    module_url = graph_namespace["_module_url"]

    failures: list[str] = []
    for key in sorted(graph):
        route = route_from_knowledge_url(module_url(key))
        if route not in routes:
            failures.append(f"{key}: 映射到 {route or '<empty>'}，但 main.py MODULES/route_map 中没有对应路由")
    if failures:
        raise CheckFailure("知识图谱元数据完整性检查失败：\n" + "\n".join(failures))
    print(f"[通过] 知识图谱元数据完整性检查：{len(graph)} 个节点")


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
    check_expected_controls()
    check_expected_content()
    check_main_routes()
    check_knowledge_graph_routes()
    check_bagu_routes_placeholder()
    check_back_to_home_entry()
    check_legacy_top_level_execution()
    check_matplotlib_close_after_show()
    if include_smoke:
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
