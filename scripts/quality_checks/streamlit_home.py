from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def check_streamlit_home_module(context: QualityCheckContext) -> None:
    """Verify legacy Streamlit home rendering is outside main.py."""

    main_text = context.read_text(Path("main.py"))
    home_text = context.read_text(Path("components/streamlit_home.py"))
    failures: list[str] = []

    required_home_fragments = [
        "class StreamlitHomeDeps",
        "def render_streamlit_home",
        "def render_streamlit_migration_notice",
        "def render_sidebar",
        "def render_progress_visual",
        "def render_course_card",
        "def render_feature_cards",
        "主站不在 Streamlit 里了",
        "运行质量检查",
        "home-progress-donut",
    ]
    for fragment in required_home_fragments:
        if fragment not in home_text:
            failures.append(f"components/streamlit_home.py 缺少首页渲染契约：{fragment}")

    forbidden_main_fragments = [
        "def render_sidebar",
        "def _render_progress_visual",
        "def render_streamlit_migration_notice",
        "def render_course_card",
        "def render_feature_cards",
        "Personal Learning System",
        "home-progress-donut",
        "运行质量检查",
    ]
    for fragment in forbidden_main_fragments:
        if fragment in main_text:
            failures.append(f"main.py 仍保留首页渲染实现细节：{fragment}")

    required_main_fragments = [
        "from components.streamlit_home import StreamlitHomeDeps",
        "render_streamlit_home_shell(deps)",
        "render_streamlit_module_page",
        "render_missing_module_page",
        "query_module=get_query_module()",
    ]
    for fragment in required_main_fragments:
        if fragment not in main_text:
            failures.append(f"main.py 未正确装配 streamlit_home：{fragment}")

    if failures:
        raise QualityCheckFailure("Streamlit 首页模块检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] Streamlit 首页模块检查：迁移提示、侧栏、进度图和首页 UI 已由 streamlit_home 承担")
