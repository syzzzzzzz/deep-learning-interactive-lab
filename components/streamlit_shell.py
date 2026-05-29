"""Shared Streamlit compatibility shell for legacy module pages."""

from __future__ import annotations

import html
import runpy
import traceback
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from components.course_catalog import ModuleInfo, PartInfo


def escape_html(value: str) -> str:
    return html.escape(str(value), quote=True)


def module_href(module: ModuleInfo) -> str:
    return f"/?module={quote(module.target, safe='')}"


def css(theme: str) -> str:
    dark = theme == "dark"
    colors = {
        "app_bg": "#171411" if dark else "#ffffff",
        "panel": "#211a14" if dark else "#ffffff",
        "panel_soft": "#2a2118" if dark else "#f7f3ec",
        "ink": "#f7f3ec" if dark else "#171411",
        "muted": "#c9beb0" if dark else "#7c756c",
        "line": "#4a3c2d" if dark else "#e6ded2",
        "sidebar": "#1b1510" if dark else "#f7f3ec",
        "accent": "#b08a4f",
        "accent_soft": "rgba(176,138,79,0.14)",
        "accent_dark": "#2a2118",
    }
    return f"""
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@500;700&family=Inter:wght@400;500;650;750&family=JetBrains+Mono:wght@400;600;700&display=swap");
    :root {{
        --app-bg: {colors["app_bg"]};
        --panel: {colors["panel"]};
        --panel-soft: {colors["panel_soft"]};
        --ink: {colors["ink"]};
        --muted: {colors["muted"]};
        --line: {colors["line"]};
        --sidebar: {colors["sidebar"]};
        --accent: {colors["accent"]};
        --accent-soft: {colors["accent_soft"]};
        --accent-dark: {colors["accent_dark"]};
    }}
    .stApp {{
        background: var(--app-bg);
        color: var(--ink);
    }}
    .block-container {{
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4.8rem;
    }}
    body, .stApp, [data-testid="stAppViewContainer"] {{
        font-family: "Noto Sans SC", Inter, "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    }}
    h1, h2, h3 {{
        font-family: "Noto Serif SC", Georgia, serif;
        font-weight: 500;
        line-height: 1.18;
        color: var(--ink);
        letter-spacing: 0;
    }}
    p, li, label, span {{
        color: inherit;
        letter-spacing: 0;
    }}
    .stMarkdown p, li {{
        line-height: 1.82;
    }}
    section[data-testid="stSidebar"] {{
        background: var(--sidebar);
        border-right: 1px solid var(--line);
    }}
    .module-hero {{
        border: 1px solid var(--line);
        background: var(--panel-soft);
        padding: clamp(1.2rem, 4vw, 2.4rem);
        margin-bottom: 1.4rem;
    }}
    .module-hero h1 {{
        font-size: clamp(2rem, 5vw, 3.6rem);
        margin: 0.55rem 0 0.8rem 0;
    }}
    .module-hero p, .path-line {{
        color: var(--muted);
        line-height: 1.78;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }}
    .module-card, .feature-card, .recommend-card, .stat-card, .artifact-note, .lesson-note {{
        display: block;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 0;
        padding: 1rem;
        color: var(--ink) !important;
        text-decoration: none !important;
    }}
    .module-card strong, .artifact-note strong {{
        display: block;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }}
    .module-card p, .artifact-note p, .lesson-note {{
        color: var(--muted);
        line-height: 1.78;
    }}
    .module-card-link:hover {{
        border-color: var(--accent);
        background: var(--accent-soft);
    }}
    .tag {{
        display: inline-flex;
        align-items: center;
        min-height: 1.8rem;
        border: 1px solid var(--line);
        color: var(--muted);
        padding: 0 0.55rem;
        margin: 0.35rem 0.35rem 0 0;
        font-size: 0.76rem;
    }}
    code, pre {{
        font-family: "JetBrains Mono", Consolas, monospace !important;
    }}
    [data-testid="stButton"] button, [data-testid="stLinkButton"] a {{
        border-radius: 0;
        border: 1px solid var(--accent-dark);
        font-weight: 700;
    }}
    @media (max-width: 760px) {{
        .block-container {{
            padding-top: 1.2rem;
        }}
        .grid {{
            grid-template-columns: 1fr;
        }}
        .module-hero h1 {{
            font-size: 2rem;
        }}
    }}
    </style>
    """


def render_module_header(
    module: ModuleInfo,
    parts: dict[str, PartInfo],
    module_kind: Callable[[ModuleInfo], str],
) -> None:
    import streamlit as st

    part = parts[module.part_key]
    tags = "".join(f'<span class="tag">{escape_html(tag)}</span>' for tag in module.tags)
    st.markdown(
        f"""
        <div class="module-hero">
          <div class="path-line">{escape_html(part.title)} / {escape_html(module.short_target)} / {escape_html(module_kind(module))}</div>
          <h1>{escape_html(module.title)}</h1>
          <p>{escape_html(module.summary)}</p>
          <div>{tags}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_button() -> None:
    import streamlit as st

    st.html(
        """
        <style>
        .home-float {
            position: fixed;
            right: 1.1rem;
            top: 0.75rem;
            z-index: 999999;
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            background: rgba(255,255,255,0.92);
            color: #2a2118 !important;
            border: 1px solid #e6ded2;
            border-radius: 0;
            padding: 0.48rem 0.72rem;
            text-decoration: none !important;
            font-size: 0.88rem;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(42,33,24,0.08);
            backdrop-filter: blur(10px);
        }
        .home-float:hover {
            background: #2a2118;
            border-color: #2a2118;
            color: #fff !important;
        }
        .stLinkButton a[href="/"],
        div[data-testid="stLinkButton"] a[href="/"] {
            display: none !important;
        }
        @media (max-width: 760px) {
            .home-float {
                right: 0.65rem;
                top: 0.55rem;
                padding: 0.42rem 0.58rem;
            }
        }
        </style>
        <a class="home-float" href="/" target="_self" aria-label="返回主界面">← 返回主界面</a>
        """
    )


def render_module_card(module: ModuleInfo) -> str:
    tags = "".join(f'<span class="tag">{escape_html(tag)}</span>' for tag in module.tags[:4])
    return (
        f'<a class="module-card module-card-link" href="{module_href(module)}" target="_self" aria-label="打开 {escape_html(module.title)}">'
        f"<strong>{escape_html(module.title)}</strong>"
        f"<p>{escape_html(module.summary)}</p>"
        f'<div class="path-line">{escape_html(module.short_target)}</div>'
        f"{tags}"
        "</a>"
    )


def render_route_error(module: ModuleInfo | None, error: BaseException) -> None:
    import streamlit as st

    title = module.title if module else "未知模块"
    st.error(f"{title} 打开失败，但主站已经兜住异常。")
    st.caption("下面是完整错误，方便继续修这个具体模块。")
    st.code("".join(traceback.format_exception(error)), language="text")


def render_streamlit_module_page(
    module: ModuleInfo,
    render_module_knowledge_nav: Callable[[ModuleInfo], None],
    render_header: Callable[[ModuleInfo], None],
) -> None:
    import streamlit as st

    try:
        runpy.run_path(str(module.path), run_name="__main__")
        from components.visual_system import render_visual_system

        render_visual_system("light")
        render_home_button()
        render_module_knowledge_nav(module)
    except Exception as exc:
        st.set_page_config(
            page_title=f"{module.title} - 打开失败",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        st.markdown(css("light"), unsafe_allow_html=True)
        from components.visual_system import render_visual_system

        render_visual_system("light")
        render_home_button()
        render_header(module)
        render_route_error(module, exc)


def render_missing_module_page(query_module: str) -> None:
    import streamlit as st

    st.set_page_config(
        page_title="模块不存在 - 深度学习书库",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(css("light"), unsafe_allow_html=True)
    from components.visual_system import render_visual_system

    render_visual_system("light")
    render_home_button()
    st.error(f"没有找到模块：{query_module}")
    st.link_button("返回首页", "/")
