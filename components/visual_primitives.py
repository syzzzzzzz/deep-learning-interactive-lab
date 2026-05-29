"""Reusable teaching UI primitives for the visual system."""

from __future__ import annotations

from html import escape

from components.visual_runtime import _st
from components.visual_tokens import NEON_BLUE
from components.visual_tokens import NEON_GREEN
from components.visual_tokens import NEON_PURPLE
def render_tooltip_label(label: str, tooltip: str, *, icon: str = "fa-solid fa-circle-info") -> str:
    """Return a focusable label with a CSS tooltip for chart controls and terms."""

    return (
        '<span class="vs-tooltip-label">'
        f'<i class="{escape(icon)}"></i>'
        f'<span class="vs-tooltip" tabindex="0" role="tooltip" data-tip="{escape(tooltip)}">{escape(label)}</span>'
        "</span>"
    )


def render_status_badge(label: str, *, status: str = "info", icon: str | None = None) -> str:
    """Return a reusable neon status badge."""

    palette = {
        "success": (NEON_GREEN, icon or "fa-solid fa-check"),
        "warning": ("#ffd166", icon or "fa-solid fa-triangle-exclamation"),
        "danger": ("#ff4d6d", icon or "fa-solid fa-circle-exclamation"),
        "running": (NEON_BLUE, icon or "fa-solid fa-spinner"),
        "info": (NEON_PURPLE, icon or "fa-solid fa-circle-info"),
    }
    color, resolved_icon = palette.get(status, palette["info"])
    return (
        f'<span class="vs-status-badge" style="--badge-color:{color}">'
        f'<i class="{escape(resolved_icon)}"></i>{escape(label)}</span>'
    )


def render_motion_note(title: str, body: str, *, icon: str = "fa-solid fa-wave-square") -> None:
    """Explain what an animation is teaching, so motion stays pedagogical."""

    st = _st()
    st.markdown(
        f"""
        <div class="vs-motion-note">
          <strong><i class="{escape(icon)}"></i> {escape(title)}</strong><br>
          {escape(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_beginner_hint(title: str, body: str, *, action: str = "先看颜色最亮的地方，再看它为什么亮。") -> None:
    """Render a beginner-friendly hint block for zero-foundation readers."""

    st = _st()
    st.markdown(
        f"""
        <div class="vs-beginner-hint">
          <strong><i class="fa-solid fa-seedling"></i> {escape(title)}</strong><br>
          {escape(body)}<br>
          <span class="vs-chart-note"><strong>图表说明：</strong>{escape(action)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_neon_metric_card(
    label: str,
    value: str,
    *,
    caption: str = "",
    delta: str = "",
    icon: str = "fa-solid fa-chart-simple",
    accent: str = NEON_BLUE,
) -> None:
    """Metric card that works in dense dashboards without losing teaching context."""

    st = _st()
    delta_html = render_status_badge(delta, status="success" if not delta.startswith("-") else "warning") if delta else ""
    st.markdown(
        f"""
        <div class="vs-neon-metric" style="--metric-color:{accent}">
          <div class="vs-neon-metric-head">
            <span class="vs-neon-metric-label"><i class="{escape(icon)}"></i> {escape(label)}</span>
            {delta_html}
          </div>
          <span class="vs-neon-metric-value">{escape(value)}</span>
          <div class="vs-neon-metric-foot">
            <span class="vs-neon-metric-caption">{escape(caption)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_concept_animation_shell(
    title: str,
    body_html: str,
    *,
    subtitle: str = "",
    status: str = "教学动效",
    note: str = "图表说明：先观察流动方向，再观察颜色亮度，最后把它对应到公式里的变量。",
    icon: str = "fa-solid fa-diagram-project",
) -> None:
    """Wrap any concept animation with a consistent title, status, and explanation."""

    st = _st()
    st.markdown(
        f"""
        <section class="vs-concept-shell">
          <div class="vs-concept-shell-head">
            <div>
              <div class="vs-concept-shell-title"><i class="{escape(icon)}"></i>{escape(title)}</div>
              <div class="vs-neon-metric-caption">{escape(subtitle)}</div>
            </div>
            {render_status_badge(status, status="running")}
          </div>
          <div class="vs-concept-shell-body">
            {body_html}
            <div class="vs-chart-note"><strong>图表说明：</strong>{escape(note.replace("图表说明：", ""))}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_responsive_motion_grid(items: list[str], *, min_width: int = 240) -> None:
    """Render already-built HTML cards inside a responsive motion grid."""

    st = _st()
    safe_items = "".join(f'<div class="vs-motion-grid-item">{item}</div>' for item in items)
    st.markdown(
        f"""
        <div class="vs-motion-grid" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">
          {safe_items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shape_flow(steps: list[tuple[str, str]], *, title: str = "Shape Flow") -> None:
    """Show how tensor shapes transform through a model pipeline."""

    st = _st()
    nodes = []
    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, "#ffd166"]
    for idx, (name, shape) in enumerate(steps):
        color = colors[idx % len(colors)]
        nodes.append(
            f"""
            <div class="vs-shape-node" style="--node-color:{color}">
              <strong>{escape(name)}</strong>
              <code>{escape(shape)}</code>
            </div>
            """
        )
        if idx < len(steps) - 1:
            nodes.append('<i class="vs-shape-arrow fa-solid fa-arrow-right"></i>')
    render_concept_animation_shell(
        title,
        f'<div class="vs-shape-flow">{"".join(nodes)}</div>',
        subtitle="张量形状从左到右变化；每个箭头都代表一次真实的层计算。",
        status="Shape 检查",
        note="如果某一格的 batch、通道数或序列长度突然对不上，模型通常会在这里报 shape mismatch。",
        icon="fa-solid fa-cubes-stacked",
    )

def render_card(
    title: str,
    body: str,
    *,
    icon: str = "fa-solid fa-cube",
    accent: str = NEON_BLUE,
    footer: str = "",
) -> None:
    """渲染一个带发光边框的通用信息卡片。

    Parameters
    ----------
    title : str
        卡片标题。
    body : str
        卡片正文（支持 HTML）。
    icon : str
        Font Awesome 图标 class。
    accent : str
        主题强调色 hex。
    footer : str
        可选底部注释文本。
    """
    st = _st()
    footer_html = f'<div class="vs-card-footer">{escape(footer)}</div>' if footer else ""
    st.markdown(
        f"""
        <div class="vs-card vs-generic-card" style="--card-accent:{accent}">
          <div class="vs-card-header"><i class="{icon}"></i> {escape(title)}</div>
          <div class="vs-card-body">{body}</div>
          {footer_html}
        </div>
        <style>
        .vs-generic-card {{ padding:1rem; border-left:3px solid var(--card-accent) !important; }}
        .vs-generic-card .vs-card-header {{ font-weight:850; font-size:1.05rem; margin-bottom:.55rem;
            color:var(--vs-ink); display:flex; align-items:center; gap:.45rem; }}
        .vs-generic-card .vs-card-header i {{ color:var(--card-accent);
            filter:drop-shadow(0 0 8px var(--card-accent)); }}
        .vs-generic-card .vs-card-body {{ color:var(--vs-muted); line-height:1.65; }}
        .vs-generic-card .vs-card-footer {{ margin-top:.6rem; padding-top:.45rem;
            border-top:1px solid var(--vs-soft-line); font-size:.82rem; color:var(--vs-muted);
            font-style:italic; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    *,
    delta: str = "",
    icon: str = "fa-solid fa-chart-simple",
    accent: str = NEON_GREEN,
) -> None:
    """渲染单个数值指标卡片。

    Parameters
    ----------
    label : str
        指标名称。
    value : str
        当前值（字符串，方便格式化如 "98.2%"）。
    delta : str
        可选变化量（如 "+0.5%"），绿色为正，红色为负。
    icon : str
        Font Awesome 图标 class。
    accent : str
        主题强调色 hex。
    """
    st = _st()
    delta_html = ""
    if delta:
        positive = delta.startswith("+") or not delta.startswith("-")
        delta_color = NEON_GREEN if positive else "#ff4d6a"
        delta_html = (
            f'<span class="vs-metric-delta" style="color:{delta_color}">{escape(delta)}</span>'
        )
    st.markdown(
        f"""
        <div class="vs-card vs-metric-card" style="--mc-accent:{accent}">
          <div class="vs-metric-icon"><i class="{icon}"></i></div>
          <div class="vs-metric-content">
            <span class="vs-metric-label">{escape(label)}</span>
            <span class="vs-metric-value">{escape(value)}</span>
            {delta_html}
          </div>
        </div>
        <style>
        .vs-metric-card {{ display:flex; align-items:center; gap:.85rem; padding:.85rem 1rem; }}
        .vs-metric-icon {{ width:44px; height:44px; border-radius:10px; display:grid; place-items:center;
            background:var(--vs-stage-bg-strong); border:1px solid var(--vs-soft-line); flex-shrink:0; }}
        .vs-metric-icon i {{ font-size:1.2rem; color:var(--mc-accent);
            filter:drop-shadow(0 0 8px var(--mc-accent)); }}
        .vs-metric-content {{ display:flex; flex-direction:column; gap:.12rem; }}
        .vs-metric-label {{ font-size:.82rem; color:var(--vs-muted); letter-spacing:.03em; }}
        .vs-metric-value {{ font-family:"JetBrains Mono"; font-size:1.45rem; font-weight:700; color:var(--vs-ink); }}
        .vs-metric-delta {{ font-size:.82rem; font-family:"JetBrains Mono"; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_neon_button(label: str, key: str, *, icon: str = "", accent: str = NEON_BLUE) -> bool:
    """渲染一个霓虹风格按钮并返回是否被点击。

    Parameters
    ----------
    label : str
        按钮文字。
    key : str
        Streamlit widget key。
    icon : str
        可选 Font Awesome 图标 class（前置）。
    accent : str
        主题强调色 hex。

    Returns
    -------
    bool
        按钮是否被点击。
    """
    st = _st()
    prefix = f'<i class="{icon}" style="margin-right:.35rem"></i>' if icon else ""
    st.markdown(
        f"""
        <style>
        div[data-testid=\"stButton\"] > button[key=\"{key}\"] {{
            border-color:{accent} !important;
            box-shadow:0 0 18px {accent}44, inset 0 0 12px {accent}18 !important;
        }}
        div[data-testid=\"stButton\"] > button[key=\"{key}\"]:hover {{
            box-shadow:0 0 28px {accent}66 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.button(f"{prefix}{label}", key=key, width="stretch")


def render_chart_container(
    title: str,
    fig,
    *,
    icon: str = "fa-solid fa-chart-area",
    description: str = "",
) -> None:
    """将 Plotly 图表包裹在带标题的容器面板中。

    Parameters
    ----------
    title : str
        面板标题。
    fig : plotly.graph_objects.Figure
        Plotly 图表对象。
    icon : str
        Font Awesome 图标 class。
    description : str
        可选描述文字，显示在图表下方。
    """
    st = _st()
    desc_html = f'<p class="vs-chart-desc">{escape(description)}</p>' if description else ""
    st.markdown(
        f"""
        <div class="vs-card vs-chart-panel">
          <div class="vs-chart-title"><i class="{icon}"></i> {escape(title)}</div>
        </div>
        <style>
        .vs-chart-panel {{ padding:1rem; }}
        .vs-chart-title {{ font-weight:850; font-size:1.02rem; margin-bottom:.6rem; color:var(--vs-ink);
            display:flex; align-items:center; gap:.4rem; }}
        .vs-chart-title i {{ color:var(--vs-blue); filter:var(--vs-glow-filter); }}
        .vs-chart-desc {{ color:var(--vs-muted); font-size:.86rem; line-height:1.55; margin:.5rem 0 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width="stretch")
    if description:
        st.markdown(
            f'<p style="color:var(--vs-muted);font-size:.86rem;line-height:1.55;margin:-.3rem 0 .8rem;">{escape(description)}</p>',
            unsafe_allow_html=True,
        )
