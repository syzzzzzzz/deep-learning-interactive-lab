"""Legacy Streamlit home and navigation rendering."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from components.course_catalog import ModuleInfo, PartInfo, matching_modules, search_modules


@dataclass(frozen=True)
class StreamlitHomeDeps:
    project_root: Path
    parts: dict[str, PartInfo]
    catalog: tuple[ModuleInfo, ...]
    routes: dict[str, ModuleInfo]
    query_module: str | None
    is_streamlit_app: Callable[[Path], bool]
    render_legacy_module_page: Callable[[ModuleInfo], None]
    render_streamlit_module: Callable[[ModuleInfo], None]
    render_missing_module: Callable[[str], None]
    css: Callable[[str], str]
    module_href: Callable[[ModuleInfo], str]
    render_module_card: Callable[[ModuleInfo], str]
    daily_recommendation: Callable[[list[ModuleInfo]], tuple[str, str, ModuleInfo]]
    escape_html: Callable[[str], str]


def open_module(module: ModuleInfo) -> None:
    import streamlit as st

    st.query_params["module"] = module.target
    st.rerun()


def first_module_for_part(part_key: str, catalog: list[ModuleInfo]) -> ModuleInfo | None:
    for module in catalog:
        if module.part_key == part_key and module.path.exists():
            return module
    return None


def render_course_card(
    module: ModuleInfo,
    meta: str,
    image_url: str,
    module_href: Callable[[ModuleInfo], str],
    escape_html: Callable[[str], str],
) -> str:
    e = escape_html
    return (
        f'<a class="course-card" href="{module_href(module)}" target="_self" aria-label="打开 {e(module.title)}">'
        f'<div class="course-image"><img src="{e(image_url)}" alt="{e(module.title)}"></div>'
        '<div class="course-body">'
        f'<div class="course-meta">{e(meta)}</div>'
        f"<h3>{e(module.title)}</h3>"
        f"<p>{e(module.summary)}</p>"
        f'<div class="path-line">{e(module.short_target)}</div>'
        "</div>"
        "</a>"
    )


def render_feature_cards(escape_html: Callable[[str], str]) -> str:
    features = [
        ("课程", "按学习阶段和主题分组，目录中的新页面会自动进入侧栏。"),
        ("路径", "从数学、神经网络、视觉、序列到注意力机制，保持递进秩序。"),
        ("笔记", "每个知识点配有读图提示、实验记录和工程解释。"),
        ("进度", "柔和阅读界面承载学习统计、稍后复习和今日推荐。"),
    ]
    e = escape_html
    cards = "".join(
        '<div class="feature-card">'
        f"<strong>{e(title)}</strong>"
        f"<p>{e(text)}</p>"
        "</div>"
        for title, text in features
    )
    return f'<div class="feature-grid">{cards}</div>'


def render_sidebar(deps: StreamlitHomeDeps) -> None:
    import streamlit as st

    st.header("学习导航")
    st.caption("课程、路径、笔记与进度")

    query = st.text_input("全局搜索", placeholder="例如 Transformer / 梯度 / 部署 / 可视化")
    results = search_modules(query, list(deps.catalog))
    if query:
        st.caption(f"找到 {len(results)} 个结果")
        for module in results[:8]:
            if st.button(module.title, key=f"search-{module.target}", width="stretch"):
                open_module(module)
        st.divider()

    for part_key, part in deps.parts.items():
        part_modules = [module for module in deps.catalog if module.part_key == part_key]
        with st.expander(f"{part.emoji} {part.short_title} · {len(part_modules)}", expanded=part_key in {"part1", "part6"}):
            for module in part_modules:
                if st.button(module.title, key=f"nav-{module.target}", width="stretch"):
                    open_module(module)

    st.divider()
    st.caption("命令行示例")
    st.code("python main.py part6/frontier", language="bash")

    st.divider()
    if st.button("运行质量检查", width="stretch"):
        with st.spinner("正在运行 scripts/quality_check.py..."):
            completed = subprocess.run(
                [sys.executable, str(deps.project_root / "scripts" / "quality_check.py")],
                cwd=deps.project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
        if completed.returncode == 0:
            st.success("质量检查通过")
        else:
            st.error(f"质量检查失败，退出码 {completed.returncode}")
        output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        st.code(output or "没有输出", language="text")


def render_progress_visual(available: list[ModuleInfo], parts: dict[str, PartInfo]) -> None:
    import streamlit as st

    try:
        from components.progress_tracker import PROGRESS_STATUSES, _store, normalize_module_key

        store = _store()
    except Exception:
        return

    if not store:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    counts = {status: 0 for status in PROGRESS_STATUSES}
    for status in store.values():
        if status in counts:
            counts[status] += 1

    total_modules = len(available)
    studied = sum(1 for m in available if store.get(normalize_module_key(m.target), "未学习") != "未学习")
    if studied == 0:
        return

    status_colors = {
        "未学习": "#e6ded2",
        "已学习": "#b08a4f",
        "已掌握": "#2a2118",
        "去实战": "#8a6a37",
    }
    labels = list(PROGRESS_STATUSES)
    values = [counts.get(status, 0) for status in labels]
    colors = [status_colors.get(status, "#ccc") for status in labels]

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+value",
                textfont=dict(size=13),
                sort=False,
            )
        ]
    )
    fig_donut.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{studied}/{total_modules}", x=0.5, y=0.5, font=dict(size=22, color="#171411"), showarrow=False)],
    )

    part_data = []
    for part_key, part_info in parts.items():
        part_modules = [module for module in available if module.part_key == part_key]
        if not part_modules:
            continue
        done = sum(
            1
            for module in part_modules
            if store.get(normalize_module_key(module.target), "未学习") in ("已学习", "已掌握", "去实战", "稍后复习")
        )
        part_data.append((f"{part_info.emoji} {part_info.short_title}", done, len(part_modules)))

    if not part_data:
        return

    part_labels = [item[0] for item in part_data]
    part_done = [item[1] for item in part_data]
    part_total = [item[2] for item in part_data]
    part_pct = [done / total * 100 if total else 0 for done, total in zip(part_done, part_total)]

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            y=part_labels,
            x=part_pct,
            orientation="h",
            marker=dict(color="#b08a4f"),
            text=[f"{done}/{total}" for done, total in zip(part_done, part_total)],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        )
    )
    fig_bar.update_layout(
        xaxis=dict(range=[0, 105], ticksuffix="%", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=60, t=10, b=10),
        height=max(200, len(part_data) * 38 + 40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    left, right = st.columns([0.35, 0.65])
    with left:
        st.plotly_chart(fig_donut, width="stretch", key="home-progress-donut")
    with right:
        st.plotly_chart(fig_bar, width="stretch", key="home-progress-bar")


def render_streamlit_migration_notice() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="主站已迁移到静态 HTML",
        page_icon="HTML",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        .block-container {
          max-width: 760px;
          padding-top: 8rem;
          color: #171411;
        }
        .legacy-notice {
          border: 1px solid #e6ded2;
          background: #fff;
          padding: 32px;
        }
        .legacy-notice h1 {
          margin: 0 0 14px;
          font-family: Georgia, "Times New Roman", serif;
          font-weight: 500;
          color: #171411;
        }
        .legacy-notice p {
          color: #7c756c;
          line-height: 1.8;
        }
        .legacy-notice code {
          color: #2a2118;
        }
        </style>
        <div class="legacy-notice">
          <h1>主站不在 Streamlit 里了</h1>
          <p>
            当前窗口是旧版 Streamlit 兼容入口，只用于临时打开 legacy 模块。
            正式学习网站已经迁移到原生 HTML/CSS/JavaScript，避免旧页面切换时的卡顿、白屏和侧边栏壳子。
          </p>
          <p>请在项目根目录运行：</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("python main.py", language="powershell")
    st.caption("或者双击 start_lab.bat。启动后打开终端中打印的 http://127.0.0.1:端口 地址。")
    st.stop()


def render_streamlit_home(deps: StreamlitHomeDeps) -> None:
    import streamlit as st

    if not deps.query_module:
        render_streamlit_migration_notice()
        return

    module = deps.routes.get(deps.query_module)
    if module and module.path.exists() and module.path.resolve() != deps.project_root / "main.py":
        if not deps.is_streamlit_app(module.path):
            deps.render_legacy_module_page(module)
            return
        deps.render_streamlit_module(module)
        return

    deps.render_missing_module(deps.query_module)
