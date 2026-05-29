from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def check_legacy_page_module(context: QualityCheckContext) -> None:
    """Verify legacy script page rendering is outside main.py."""

    main_text = context.read_text(Path("main.py"))
    legacy_text = context.read_text(Path("components/legacy_page.py"))
    failures: list[str] = []

    required_legacy_fragments = [
        "class LegacyPageDeps",
        "def render_legacy_module_page",
        "def render_legacy_results",
        "def render_legacy_learning_guide",
        "def artifact_explanation",
        "def run_legacy_module",
        "def latest_run_dir",
        "artifact_context",
        "run_legacy_script",
        "生成 / 更新运行结果",
        "控制台输出",
        "这图看什么",
        "查看源码片段",
    ]
    for fragment in required_legacy_fragments:
        if fragment not in legacy_text:
            failures.append(f"components/legacy_page.py 缺少 legacy 页面契约：{fragment}")

    forbidden_main_fragments = [
        "artifact_context",
        "latest_artifact_run_dir",
        "run_legacy_script",
        "def latest_run_dir",
        "def read_text_preview",
        "def run_legacy_module",
        "def artifact_explanation",
        "def render_legacy_results",
        "def render_legacy_learning_guide",
        "生成 / 更新运行结果",
        "控制台输出",
        "这图看什么",
    ]
    for fragment in forbidden_main_fragments:
        if fragment in main_text:
            failures.append(f"main.py 仍保留 legacy 页面实现细节：{fragment}")

    required_main_fragments = [
        "from components.legacy_page import LegacyPageDeps",
        "render_legacy_page_shell(deps, module)",
        "learning_guides=LEGACY_LEARNING_GUIDES",
        "render_module_knowledge_nav=render_module_knowledge_nav",
    ]
    for fragment in required_main_fragments:
        if fragment not in main_text:
            failures.append(f"main.py 未正确装配 legacy_page：{fragment}")

    if failures:
        raise QualityCheckFailure("Legacy 页面模块检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] Legacy 页面模块检查：旧脚本运行、产物展示、图像解释和源码片段已由 legacy_page 承担")
