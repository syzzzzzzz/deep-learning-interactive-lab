from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def check_streamlit_shell_module(context: QualityCheckContext) -> None:
    """Verify shared Streamlit UI shell is outside main.py."""

    main_text = context.read_text(Path("main.py"))
    shell_text = context.read_text(Path("components/streamlit_shell.py"))
    failures: list[str] = []

    required_shell_fragments = [
        "def css",
        "def escape_html",
        "def module_href",
        "def render_home_button",
        "def render_module_header",
        "def render_module_card",
        "def render_route_error",
        "def render_streamlit_module_page",
        "def render_missing_module_page",
        "Noto Sans SC",
        "home-float",
        "module-hero",
        "traceback.format_exception",
    ]
    for fragment in required_shell_fragments:
        if fragment not in shell_text:
            failures.append(f"components/streamlit_shell.py 缺少 Streamlit 壳层契约：{fragment}")

    forbidden_main_fragments = [
        "def css(",
        "def e(",
        "home-float",
        "Noto Sans SC",
        "traceback.format_exception",
        "module-hero {",
    ]
    for fragment in forbidden_main_fragments:
        if fragment in main_text:
            failures.append(f"main.py 仍保留 Streamlit 壳层实现细节：{fragment}")

    required_main_fragments = [
        "from components.streamlit_shell import css",
        "from components.streamlit_shell import escape_html as e",
        "from components.streamlit_shell import module_href as shell_module_href",
        "return shell_module_href(module)",
        "return shell_render_module_card(module)",
        "render_streamlit_module_page_shell(module, render_module_knowledge_nav, render_module_header)",
        "render_missing_module_page_shell(query_module)",
    ]
    for fragment in required_main_fragments:
        if fragment not in main_text:
            failures.append(f"main.py 未正确装配 streamlit_shell：{fragment}")

    if failures:
        raise QualityCheckFailure("Streamlit 壳层模块检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] Streamlit 壳层模块检查：CSS、模块头、卡片、返回入口和错误页已由 streamlit_shell 承担")
