"""全站 UI/UX 视觉语言与教学动效组件。"""

from __future__ import annotations

from html import escape
from math import cos, sin


NEON_BLUE = "#00f0ff"
NEON_PURPLE = "#b000ff"
NEON_GREEN = "#00ff88"


def _st():
    import streamlit as st

    return st


def render_visual_system(theme: str = "dark", *, particles: bool = True) -> None:
    """Inject the shared dark technical visual language and motion system."""

    st = _st()
    dark = theme == "dark"
    bg = "#070a12" if dark else "#f7fbfc"
    panel = "rgba(12,18,30,0.86)" if dark else "rgba(255,255,255,0.86)"
    panel_soft = "rgba(17,27,43,0.74)" if dark else "rgba(239,246,243,0.86)"
    ink = "#eaf7ff" if dark else "#172026"
    muted = "#9db2c7" if dark else "#596772"
    line = "rgba(0,240,255,0.22)" if dark else "rgba(15,139,141,0.18)"
    sidebar = "#090d17" if dark else "#eef4f2"
    code_bg = "#050812" if dark else "#101820"
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;750;850&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
        :root {{
            --vs-bg: {bg};
            --vs-panel: {panel};
            --vs-panel-soft: {panel_soft};
            --vs-ink: {ink};
            --vs-muted: {muted};
            --vs-line: {line};
            --vs-sidebar: {sidebar};
            --vs-blue: {NEON_BLUE};
            --vs-purple: {NEON_PURPLE};
            --vs-green: {NEON_GREEN};
            --vs-code-bg: {code_bg};
        }}
        html {{ scroll-behavior: smooth; }}
        body, .stApp, [data-testid="stAppViewContainer"] {{
            font-family: Inter, "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
        }}
        code, pre, .stCode, .stCode *, textarea, input {{
            font-family: "JetBrains Mono", Consolas, "Microsoft YaHei UI", monospace !important;
        }}
        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(0,240,255,0.16), transparent 30%),
                radial-gradient(circle at 88% 14%, rgba(176,0,255,0.16), transparent 28%),
                linear-gradient(180deg, rgba(7,10,18,0.96), rgba(10,17,28,0.98)),
                var(--vs-bg);
            color: var(--vs-ink);
            animation: vs-page-in 420ms ease both;
        }}
        .block-container {{ animation: vs-slide-in 460ms ease both; }}
        @keyframes vs-page-in {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        @keyframes vs-slide-in {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--vs-sidebar), rgba(9,13,23,0.94));
            border-right: 1px solid var(--vs-line);
        }}
        section[data-testid="stSidebar"] *, h1, h2, h3, h4, p, li, label, span {{
            color: var(--vs-ink);
            letter-spacing: 0;
        }}
        .stMarkdown, [data-testid="stMarkdownContainer"], .stDataFrame, .stTable {{
            color: var(--vs-ink);
        }}
        div[data-testid="stMetric"], .vs-card, .module-card, .feature-card, .hero-panel,
        .recommend-card, .stat-card, .artifact-note, .lesson-note {{
            background: var(--vs-panel) !important;
            border: 1px solid var(--vs-line) !important;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 18px 42px rgba(0,0,0,0.28), 0 0 30px rgba(0,240,255,0.05);
            backdrop-filter: blur(14px);
        }}
        .stButton > button, .stDownloadButton > button, .stLinkButton > a {{
            border-radius: 8px !important;
            border: 1px solid rgba(0,240,255,0.42) !important;
            background: linear-gradient(135deg, rgba(0,240,255,0.16), rgba(176,0,255,0.12)) !important;
            color: var(--vs-ink) !important;
            box-shadow: 0 0 16px rgba(0,240,255,0.12);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {{
            transform: translateY(-1px);
            border-color: var(--vs-green) !important;
            box-shadow: 0 0 22px rgba(0,255,136,0.22);
        }}
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, .stLinkButton > a:focus-visible,
        a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {{
            outline: 2px solid var(--vs-green) !important;
            outline-offset: 3px !important;
            box-shadow: 0 0 0 4px rgba(0,255,136,0.16), 0 0 24px rgba(0,255,136,0.24) !important;
        }}
        .tag, .demo-badge {{
            border-color: rgba(0,240,255,0.42) !important;
            background: rgba(0,240,255,0.10) !important;
            color: var(--vs-blue) !important;
            box-shadow: 0 0 14px rgba(0,240,255,0.10);
        }}
        .fa, .fa-solid, .fa-regular, .fa-brands {{
            color: var(--vs-blue);
            filter: drop-shadow(0 0 8px rgba(0,240,255,0.45));
        }}
        .vs-tooltip-label {{
            display: inline-flex;
            align-items: center;
            gap: .38rem;
            color: var(--vs-ink);
            font-weight: 760;
        }}
        .vs-tooltip {{
            position: relative;
            border-bottom: 1px dashed rgba(0,240,255,0.45);
            cursor: help;
        }}
        .vs-tooltip:focus-visible {{
            outline: 2px solid var(--vs-green);
            outline-offset: 4px;
            border-radius: 6px;
        }}
        .vs-tooltip:hover::after, .vs-tooltip:focus-visible::after {{
            content: attr(data-tip);
            position: absolute;
            z-index: 50;
            left: 0;
            top: 1.55rem;
            width: min(320px, 80vw);
            padding: 0.62rem 0.72rem;
            border-radius: 8px;
            background: #07101c;
            border: 1px solid rgba(0,240,255,0.38);
            color: #eaf7ff;
            box-shadow: 0 16px 38px rgba(0,0,0,0.38), 0 0 18px rgba(0,240,255,0.16);
            font-size: 0.86rem;
            line-height: 1.5;
        }}
        .vs-motion-note, .vs-beginner-hint, .vs-chart-note {{
            border: 1px solid rgba(0,240,255,0.24);
            background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(0,255,136,0.05));
            border-radius: 8px;
            padding: .82rem .92rem;
            margin: .65rem 0 .9rem;
            color: var(--vs-ink);
            line-height: 1.65;
        }}
        .vs-motion-note strong, .vs-beginner-hint strong, .vs-chart-note strong {{
            color: var(--vs-green);
        }}
        .vs-chart-note {{
            border-left: 3px solid var(--vs-blue);
        }}
        .vs-status-badge {{
            display: inline-flex;
            align-items: center;
            gap: .38rem;
            border: 1px solid color-mix(in srgb, var(--badge-color) 52%, transparent);
            background: color-mix(in srgb, var(--badge-color) 14%, transparent);
            color: var(--vs-ink);
            border-radius: 999px;
            padding: .26rem .58rem;
            font-size: .8rem;
            font-weight: 800;
            white-space: nowrap;
            box-shadow: 0 0 14px color-mix(in srgb, var(--badge-color) 18%, transparent);
        }}
        .vs-neon-metric {{
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid color-mix(in srgb, var(--metric-color) 42%, transparent);
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--metric-color) 18%, transparent), rgba(255,255,255,0.035)),
                var(--vs-panel);
            padding: .9rem .95rem;
            min-height: 116px;
            box-shadow: 0 18px 42px rgba(0,0,0,.24), 0 0 24px color-mix(in srgb, var(--metric-color) 14%, transparent);
        }}
        .vs-neon-metric::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--metric-color) 18%, transparent), transparent);
            transform: translateX(-100%);
            animation: vs-metric-sheen 3.2s ease-in-out infinite;
        }}
        @keyframes vs-metric-sheen {{
            0%, 46% {{ transform: translateX(-100%); opacity: 0; }}
            62% {{ opacity: 1; }}
            100% {{ transform: translateX(100%); opacity: 0; }}
        }}
        .vs-neon-metric-head, .vs-neon-metric-foot {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .6rem;
        }}
        .vs-neon-metric-label {{
            color: var(--vs-muted);
            font-size: .82rem;
            font-weight: 760;
        }}
        .vs-neon-metric-value {{
            position: relative;
            display: block;
            margin-top: .45rem;
            color: var(--metric-color);
            font-family: "JetBrains Mono", monospace;
            font-size: clamp(1.34rem, 2.2vw, 2rem);
            font-weight: 900;
            text-shadow: 0 0 16px color-mix(in srgb, var(--metric-color) 42%, transparent);
        }}
        .vs-neon-metric-caption {{
            color: var(--vs-muted);
            font-size: .78rem;
            line-height: 1.45;
        }}
        .vs-concept-shell {{
            border-radius: 8px;
            border: 1px solid rgba(0,240,255,.24);
            background: var(--vs-panel);
            overflow: hidden;
            margin: .85rem 0 1rem;
        }}
        .vs-concept-shell-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .75rem;
            padding: .82rem .95rem;
            border-bottom: 1px solid rgba(0,240,255,.16);
        }}
        .vs-concept-shell-title {{
            display: flex;
            align-items: center;
            gap: .45rem;
            font-weight: 880;
        }}
        .vs-concept-shell-body {{
            padding: .95rem;
        }}
        .vs-motion-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: .85rem;
            align-items: stretch;
            margin: .75rem 0 1rem;
        }}
        .vs-shape-flow {{
            display: flex;
            align-items: center;
            gap: .55rem;
            flex-wrap: wrap;
            margin: .65rem 0 .8rem;
        }}
        .vs-shape-node {{
            border: 1px solid color-mix(in srgb, var(--node-color) 48%, transparent);
            background: color-mix(in srgb, var(--node-color) 13%, rgba(255,255,255,.03));
            border-radius: 8px;
            padding: .5rem .62rem;
            min-width: 112px;
            box-shadow: 0 0 18px color-mix(in srgb, var(--node-color) 16%, transparent);
        }}
        .vs-shape-node strong {{
            display: block;
            color: var(--vs-ink);
            font-size: .82rem;
            margin-bottom: .2rem;
        }}
        .vs-shape-node code {{
            color: var(--node-color);
            background: rgba(0,0,0,.22);
            border-radius: 5px;
            padding: .08rem .32rem;
            font-size: .78rem;
        }}
        .vs-shape-arrow {{
            color: var(--vs-blue);
            filter: drop-shadow(0 0 8px rgba(0,240,255,.38));
            animation: vs-shape-pulse 1.8s ease-in-out infinite;
        }}
        @keyframes vs-shape-pulse {{
            0%, 100% {{ opacity: .45; transform: translateX(0); }}
            50% {{ opacity: 1; transform: translateX(3px); }}
        }}
        .vs-icon-row {{
            display: flex;
            gap: 0.62rem;
            flex-wrap: wrap;
            margin: 0.45rem 0 0.8rem;
        }}
        .vs-icon-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            border: 1px solid rgba(0,240,255,0.25);
            background: rgba(255,255,255,0.04);
            border-radius: 999px;
            padding: 0.32rem 0.62rem;
            font-size: 0.86rem;
        }}
        .vs-particle-field {{
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
            opacity: 0.46;
        }}
        .vs-particle-field span {{
            position: absolute;
            width: 4px;
            height: 4px;
            border-radius: 999px;
            background: var(--vs-blue);
            box-shadow: 0 0 12px var(--vs-blue);
            animation: vs-float-particle var(--duration) linear infinite;
            left: var(--x);
            top: var(--y);
        }}
        @keyframes vs-float-particle {{
            0% {{ transform: translate3d(0, 0, 0) scale(.7); opacity: .15; }}
            35% {{ opacity: .8; }}
            100% {{ transform: translate3d(var(--dx), -120px, 0) scale(1.15); opacity: 0; }}
        }}
        .vs-loading-strip {{
            height: 3px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.06);
            margin: 0.35rem 0 0.9rem;
        }}
        .vs-loading-strip::before {{
            content: "";
            display: block;
            width: 38%;
            height: 100%;
            background: linear-gradient(90deg, transparent, var(--vs-blue), var(--vs-green), transparent);
            animation: vs-loading 1.45s ease-in-out infinite;
        }}
        @keyframes vs-loading {{
            from {{ transform: translateX(-110%); }}
            to {{ transform: translateX(270%); }}
        }}
        .vs-loading {{
            position: relative;
        }}
        @media (max-width: 760px) {{
            .vs-particle-field {{ opacity: 0.24; }}
            .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
            .vs-concept-shell-head {{ align-items: flex-start; flex-direction: column; }}
            .vs-neon-metric {{ min-height: auto; }}
            .vs-shape-flow {{ align-items: stretch; }}
            .vs-shape-node {{ flex: 1 1 100%; min-width: 0; }}
            .vs-shape-arrow {{ transform: rotate(90deg); align-self: center; }}
            .vs-tooltip:hover::after, .vs-tooltip:focus-visible::after {{
                left: 50%;
                transform: translateX(-50%);
                width: min(280px, 88vw);
            }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            html {{ scroll-behavior: auto; }}
            *, *::before, *::after {{
                animation-duration: 0.001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
            }}
            .vs-particle-field, .vs-loading-strip::before {{
                display: none !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if particles:
        render_particle_field()


def render_particle_field(count: int = 28) -> None:
    st = _st()
    spans = []
    for idx in range(count):
        x = (idx * 37) % 100
        y = (idx * 53) % 100
        dx = round(sin(idx * 1.7) * 70, 2)
        duration = 8 + (idx % 7)
        color = [NEON_BLUE, NEON_PURPLE, NEON_GREEN][idx % 3]
        spans.append(
            f'<span style="--x:{x}%;--y:{y}%;--dx:{dx}px;--duration:{duration}s;background:{color};box-shadow:0 0 12px {color};"></span>'
        )
    st.markdown(f'<div class="vs-particle-field">{"".join(spans)}</div>', unsafe_allow_html=True)


def render_loading_bar(label: str = "神经网络粒子正在载入本章可视化") -> None:
    st = _st()
    st.markdown(
        f"""
        <div class="vs-icon-row vs-loading" aria-live="polite">
          <span class="vs-icon-pill"><i class="fa-solid fa-circle-nodes"></i>{escape(label)}</span>
        </div>
        <div class="vs-loading-strip"></div>
        """,
        unsafe_allow_html=True,
    )


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


def render_convolution_particle_flow() -> None:
    st = _st()
    particles = []
    for idx in range(36):
        row = idx // 6
        col = idx % 6
        delay = round((row + col) * 0.08, 2)
        particles.append(f'<i style="--r:{row};--c:{col};--delay:{delay}s"></i>')
    st.markdown(
        f"""
        <div class="vs-card vs-conv-flow" title="像素粒子流：输入像素经过卷积核加权汇聚，形成输出特征响应。">
          <div class="vs-panel-title"><i class="fa-solid fa-wand-magic-sparkles"></i> 卷积粒子流</div>
          <div class="vs-flow-stage vs-input">{''.join(particles)}</div>
          <div class="vs-kernel">3×3<br>Kernel</div>
          <div class="vs-flow-stage vs-output"><b></b><b></b><b></b><b></b><b></b><b></b><b></b><b></b><b></b></div>
          <p>左侧每个亮点是一块局部像素；它们穿过卷积核时被加权求和，右侧亮度代表输出特征图响应强弱。</p>
        </div>
        <style>
        .vs-conv-flow {{ padding: 1rem; position: relative; overflow: hidden; }}
        .vs-panel-title {{ font-weight: 850; margin-bottom: .7rem; color: var(--vs-ink); }}
        .vs-conv-flow {{ display: grid; grid-template-columns: 1fr 96px 1fr; gap: .9rem; align-items: center; }}
        .vs-conv-flow p {{ grid-column: 1 / -1; color: var(--vs-muted); margin: .45rem 0 0; line-height: 1.62; }}
        .vs-flow-stage {{ min-height: 170px; border: 1px solid rgba(0,240,255,.22); border-radius: 8px; position: relative; background: rgba(0,0,0,.18); }}
        .vs-input i {{ position:absolute; width:8px; height:8px; border-radius:50%; left:calc(10% + var(--c)*14%); top:calc(12% + var(--r)*13%); background:var(--vs-blue); box-shadow:0 0 12px var(--vs-blue); animation: vs-pixel-fly 2.8s ease-in-out infinite; animation-delay:var(--delay); }}
        @keyframes vs-pixel-fly {{ 0%,100%{{transform:translateX(0);opacity:.35}} 48%{{transform:translateX(132px);opacity:1}} 70%{{opacity:.25}} }}
        .vs-kernel {{ height: 96px; border-radius: 8px; display:grid; place-items:center; text-align:center; font-family:"JetBrains Mono"; color:#061018; background:linear-gradient(135deg,var(--vs-blue),var(--vs-green)); box-shadow:0 0 26px rgba(0,240,255,.36); font-weight:800; }}
        .vs-output {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; padding:18px; }}
        .vs-output b {{ border-radius:7px; background:rgba(0,255,136,.22); box-shadow:0 0 16px rgba(0,255,136,.24); animation: vs-feature-pulse 2.8s ease-in-out infinite; }}
        .vs-output b:nth-child(2n){{animation-delay:.25s}} .vs-output b:nth-child(3n){{animation-delay:.5s}}
        @keyframes vs-feature-pulse {{ 0%,100%{{opacity:.28;transform:scale(.96)}} 50%{{opacity:1;transform:scale(1.02)}} }}
        @media(max-width:800px){{.vs-conv-flow{{grid-template-columns:1fr}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_gradient_descent_landscape() -> None:
    st = _st()
    dots = []
    for idx in range(20):
        x = 7 + idx * 4.4
        y = 76 - 56 * (1 - (idx / 19)) ** 2 + sin(idx * 0.9) * 4
        dots.append(f'<span style="left:{x}%;top:{y}%;animation-delay:{idx*0.08:.2f}s"></span>')
    st.markdown(
        f"""
        <div class="vs-card vs-loss-landscape" title="3D 损失地形：小球沿负梯度方向移动，学习率越大每次跨得越远。">
          <div class="vs-panel-title"><i class="fa-solid fa-chart-line"></i> 梯度下降损失地形</div>
          <div class="vs-surface">{''.join(dots)}<em></em></div>
          <p>小球不是随便滚，它每一步都沿着局部最陡下降方向移动；如果学习率过大，会越过谷底并震荡。</p>
        </div>
        <style>
        .vs-loss-landscape{{padding:1rem}}
        .vs-surface{{height:220px;position:relative;border-radius:8px;overflow:hidden;border:1px solid rgba(0,240,255,.22);background:
          radial-gradient(ellipse at 50% 70%, rgba(0,255,136,.24), transparent 28%),
          repeating-radial-gradient(ellipse at 50% 72%, rgba(0,240,255,.22) 0 1px, transparent 1px 18px),
          linear-gradient(160deg, rgba(176,0,255,.22), rgba(0,240,255,.10)); transform: perspective(700px) rotateX(18deg);}}
        .vs-surface span{{position:absolute;width:8px;height:8px;border-radius:50%;background:var(--vs-green);box-shadow:0 0 16px var(--vs-green);opacity:.28;animation:vs-trail 2.6s ease infinite}}
        .vs-surface em{{position:absolute;width:18px;height:18px;border-radius:50%;left:82%;top:20%;background:var(--vs-blue);box-shadow:0 0 28px var(--vs-blue);animation:vs-ball 2.6s ease-in-out infinite}}
        @keyframes vs-ball{{0%{{left:7%;top:76%}}45%{{left:46%;top:45%}}100%{{left:89%;top:22%}}}}
        @keyframes vs-trail{{0%,100%{{opacity:.18}}50%{{opacity:1}}}}
        .vs-loss-landscape p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_attention_light_beams(tokens: list[str] | None = None, query_index: int = 0) -> None:
    st = _st()
    tokens = tokens or ["Query", "Key", "Value", "Context", "Output"]
    tokens = tokens[:7]
    query_index = max(0, min(query_index, len(tokens) - 1))
    token_html = "".join(
        f'<span class="{"active" if idx == query_index else ""}" style="--i:{idx}">{escape(token)}</span>'
        for idx, token in enumerate(tokens)
    )
    beams = "".join(f'<i style="--i:{idx};--w:{0.25 + (idx + 1) / (len(tokens) * 1.3):.2f}"></i>' for idx in range(len(tokens)))
    st.markdown(
        f"""
        <div class="vs-card vs-attention-beams" title="光线投射：Query 像手电筒，越亮的 Key 表示 softmax 权重越高。">
          <div class="vs-panel-title"><i class="fa-solid fa-bullseye"></i> Query 光线投射到 Key</div>
          <div class="vs-beam-stage"><div class="query-lamp">Q</div>{beams}<div class="token-row">{token_html}</div></div>
          <p>每条光线代表 Query 和某个 Key 的相似度。经过 softmax 后，亮光会变成更大的注意力权重，再去加权取回 Value。</p>
        </div>
        <style>
        .vs-attention-beams{{padding:1rem}}
        .vs-beam-stage{{height:210px;position:relative;border-radius:8px;border:1px solid rgba(0,240,255,.22);overflow:hidden;background:radial-gradient(circle at 12% 45%,rgba(0,240,255,.22),transparent 28%),rgba(0,0,0,.18)}}
        .query-lamp{{position:absolute;left:8%;top:42%;width:42px;height:42px;border-radius:50%;display:grid;place-items:center;background:var(--vs-blue);color:#061018;font-weight:900;box-shadow:0 0 32px var(--vs-blue);z-index:3}}
        .vs-beam-stage i{{position:absolute;left:13%;top:50%;width:70%;height:2px;background:linear-gradient(90deg,var(--vs-blue),rgba(0,255,136,var(--w)),transparent);transform-origin:left center;transform:rotate(calc(-24deg + var(--i)*8deg));opacity:var(--w);animation:vs-beam 1.8s ease-in-out infinite alternate}}
        @keyframes vs-beam{{from{{filter:brightness(.75)}}to{{filter:brightness(1.55)}}}}
        .token-row{{position:absolute;right:4%;top:18%;bottom:12%;display:flex;flex-direction:column;justify-content:space-between}}
        .token-row span{{border:1px solid rgba(0,240,255,.28);border-radius:8px;padding:.25rem .55rem;background:rgba(255,255,255,.06);font-family:"JetBrains Mono";}}
        .token-row span.active{{border-color:var(--vs-green);box-shadow:0 0 18px rgba(0,255,136,.22)}}
        .vs-attention-beams p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_backprop_current_flow() -> None:
    st = _st()
    layers = ["Loss", "Output", "Hidden 2", "Hidden 1", "Input"]
    layer_html = "".join(f"<span>{escape(layer)}</span>" for layer in layers)
    st.markdown(
        f"""
        <div class="vs-card vs-current-flow" title="反向传播电流：梯度从损失出发反向流过各层，亮度代表梯度大小。">
          <div class="vs-panel-title"><i class="fa-solid fa-bolt"></i> 反向传播电流</div>
          <div class="vs-circuit">{layer_html}<i></i><i></i><i></i><i></i></div>
          <p>亮线从损失端反向流动。某一段突然变暗，通常意味着梯度消失；突然刺眼，可能是梯度爆炸。</p>
        </div>
        <style>
        .vs-current-flow{{padding:1rem}}
        .vs-circuit{{height:150px;position:relative;display:flex;align-items:center;justify-content:space-between;border-radius:8px;border:1px solid rgba(0,240,255,.22);background:rgba(0,0,0,.18);padding:0 1rem}}
        .vs-circuit span{{position:relative;z-index:2;border:1px solid rgba(0,240,255,.28);border-radius:8px;padding:.45rem .6rem;background:#07101c;font-weight:750}}
        .vs-circuit i{{position:absolute;top:50%;height:4px;width:19%;background:linear-gradient(90deg,transparent,var(--vs-green),var(--vs-blue),transparent);box-shadow:0 0 18px var(--vs-green);animation:vs-current 1.4s linear infinite}}
        .vs-circuit i:nth-of-type(1){{left:76%;animation-delay:0s}}.vs-circuit i:nth-of-type(2){{left:55%;animation-delay:.18s}}.vs-circuit i:nth-of-type(3){{left:34%;animation-delay:.36s}}.vs-circuit i:nth-of-type(4){{left:13%;animation-delay:.54s}}
        @keyframes vs-current{{0%{{opacity:.12;transform:translateX(18px)}}50%{{opacity:1}}100%{{opacity:.12;transform:translateX(-18px)}}}}
        .vs-current-flow p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_training_dashboard_gauges(loss: float = 0.42, accuracy: float = 0.86, lr: float = 0.001, gpu: float = 0.72) -> None:
    st = _st()
    metrics = [
        ("Loss", loss, 1.0, NEON_PURPLE),
        ("Accuracy", accuracy, 1.0, NEON_GREEN),
        ("LR ×1000", lr * 1000, 2.0, NEON_BLUE),
        ("GPU", gpu, 1.0, "#ffd166"),
    ]
    cards = []
    for name, value, max_value, color in metrics:
        angle = -120 + min(1, max(0, value / max_value)) * 240
        cards.append(
            f"""
            <div class="vs-gauge" style="--angle:{angle:.1f}deg;--color:{color}">
              <div class="dial"><i></i></div><strong>{escape(name)}</strong><span>{value:.3g}</span>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="vs-card vs-dashboard" title="训练仪表盘：把 loss、accuracy、learning rate 和 GPU 利用率放在同一屏观察。">
          <div class="vs-panel-title"><i class="fa-solid fa-gauge-high"></i> 训练实时仪表盘</div>
          <div class="vs-gauge-grid">{''.join(cards)}</div>
          <p>仪表盘不是为了好看：loss 看优化是否下降，accuracy 看任务效果，学习率看步子大小，GPU 利用率看系统是否真的跑满。</p>
        </div>
        <style>
        .vs-dashboard{{padding:1rem}}
        .vs-gauge-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem}}
        .vs-gauge{{border:1px solid rgba(0,240,255,.22);border-radius:8px;padding:.8rem;text-align:center;background:rgba(0,0,0,.18)}}
        .dial{{width:96px;height:52px;margin:0 auto .45rem;position:relative;overflow:hidden;border-radius:96px 96px 0 0;background:conic-gradient(from 240deg,var(--color),rgba(255,255,255,.08) 240deg)}}
        .dial i{{position:absolute;width:42px;height:3px;background:var(--color);left:48px;bottom:5px;transform-origin:left center;transform:rotate(var(--angle));box-shadow:0 0 14px var(--color);animation:vs-needle 1.6s ease-in-out infinite alternate}}
        @keyframes vs-needle{{from{{filter:brightness(.75)}}to{{filter:brightness(1.35)}}}}
        .vs-gauge strong{{display:block}} .vs-gauge span{{font-family:"JetBrains Mono";color:var(--vs-muted)}}
        .vs-dashboard p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        @media(max-width:900px){{.vs-gauge-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 通用 UI 组件
# ──────────────────────────────────────────────


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
            border-top:1px solid rgba(255,255,255,.08); font-size:.82rem; color:var(--vs-muted);
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
            background:rgba(0,0,0,.22); border:1px solid rgba(255,255,255,.06); flex-shrink:0; }}
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
    return st.button(f"{prefix}{label}", key=key, use_container_width=True)


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
        .vs-chart-title i {{ color:var(--vs-blue); filter:drop-shadow(0 0 6px rgba(0,240,255,.45)); }}
        .vs-chart-desc {{ color:var(--vs-muted); font-size:.86rem; line-height:1.55; margin:.5rem 0 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    if description:
        st.markdown(
            f'<p style="color:var(--vs-muted);font-size:.86rem;line-height:1.55;margin:-.3rem 0 .8rem;">{escape(description)}</p>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# CNN 层级管线动效
# ──────────────────────────────────────────────


def render_cnn_layer_pipeline() -> None:
    """CNN 层级管线动效：展示数据从 Input 经 Conv→ReLU→Pool→FC 的完整前向流动。

    每个阶段用独立容器表示，中间有粒子流动的连接箭头，
    各阶段的特征图尺寸逐步缩小、通道数递增。
    """
    st = _st()
    stages = [
        ("Input", "224×224×3", NEON_BLUE, "fa-solid fa-image"),
        ("Conv", "112×112×64", NEON_PURPLE, "fa-solid fa-expand"),
        ("ReLU", "112×112×64", "#ff4d6a", "fa-solid fa-bolt"),
        ("Pool", "56×56×64", NEON_GREEN, "fa-solid fa-compress"),
        ("Conv", "56×56×128", NEON_PURPLE, "fa-solid fa-expand"),
        ("ReLU", "56×56×128", "#ff4d6a", "fa-solid fa-bolt"),
        ("Pool", "28×28×128", NEON_GREEN, "fa-solid fa-compress"),
        ("FC", "1×1×1000", "#ffd166", "fa-solid fa-brain"),
    ]
    stage_items = []
    for idx, (name, shape, color, icon) in enumerate(stages):
        anim_delay = round(idx * 0.18, 2)
        stage_items.append(
            f"""
            <div class="vs-pipe-stage" style="--stage-color:{color};--stage-delay:{anim_delay}s">
              <i class="{icon}"></i>
              <strong>{escape(name)}</strong>
              <code>{escape(shape)}</code>
            </div>
            """
        )
    # 箭头连接
    arrows = "".join(
        f'<div class="vs-pipe-arrow" style="--arr-delay:{round(idx * 0.18 + 0.09, 2)}s"><i class="fa-solid fa-chevron-right"></i></div>'
        for idx in range(len(stages) - 1)
    )
    # 数据粒子行
    data_dots = "".join(
        f'<span class="vs-pipe-dot" style="--dot-delay:{round(idx * 0.12, 2)}s;--dot-color:{stages[idx % len(stages)][2]}"></span>'
        for idx in range(24)
    )
    st.markdown(
        f"""
        <div class="vs-card vs-cnn-pipeline" title="CNN 前向管线：数据逐层经过卷积、激活、池化，最终到达全连接分类层。">
          <div class="vs-panel-title"><i class="fa-solid fa-layer-group"></i> CNN 层级管线</div>
          <div class="vs-pipe-row">
            {''.join(stage_items)}
          </div>
          <div class="vs-pipe-arrow-row">{arrows}</div>
          <div class="vs-pipe-data-track">{data_dots}</div>
          <p>数据从左到右流动：卷积提取局部特征 → ReLU 引入非线性 → 池化压缩空间维度 → 全连接层输出分类。特征图越来越小，但语义越来越丰富。</p>
        </div>
        <style>
        .vs-cnn-pipeline {{ padding:1rem; overflow:hidden; }}
        .vs-pipe-row {{ display:flex; gap:.35rem; flex-wrap:wrap; justify-content:center; margin-bottom:.45rem; }}
        .vs-pipe-stage {{ display:flex; flex-direction:column; align-items:center; gap:.2rem;
            padding:.55rem .5rem; border-radius:8px; min-width:78px;
            border:1px solid var(--stage-color); background:rgba(0,0,0,.22);
            animation:vs-pipe-stage-in .5s ease both; animation-delay:var(--stage-delay); }}
        .vs-pipe-stage i {{ color:var(--stage-color); font-size:1.1rem;
            filter:drop-shadow(0 0 8px var(--stage-color)); }}
        .vs-pipe-stage strong {{ font-size:.78rem; color:var(--vs-ink); }}
        .vs-pipe-stage code {{ font-size:.68rem; color:var(--vs-muted); background:rgba(255,255,255,.05);
            padding:.1rem .35rem; border-radius:4px; }}
        @keyframes vs-pipe-stage-in {{ from {{ opacity:0; transform:translateY(10px) scale(.92); }}
            to {{ opacity:1; transform:translateY(0) scale(1); }} }}
        .vs-pipe-arrow-row {{ display:flex; justify-content:center; gap:0; margin:.2rem 0; position:relative; }}
        .vs-pipe-arrow {{ color:var(--vs-blue); font-size:.72rem; opacity:.55;
            animation:vs-pipe-arrow-pulse 1.4s ease-in-out infinite; animation-delay:var(--arr-delay); }}
        @keyframes vs-pipe-arrow-pulse {{ 0%,100%{{ opacity:.3; transform:scale(.85); }}
            50%{{ opacity:1; transform:scale(1.15); }} }}
        .vs-pipe-data-track {{ height:8px; border-radius:4px; background:rgba(255,255,255,.05);
            position:relative; overflow:hidden; margin:.45rem 0; }}
        .vs-pipe-dot {{ position:absolute; width:6px; height:6px; border-radius:50%;
            top:1px; background:var(--dot-color); box-shadow:0 0 10px var(--dot-color);
            animation:vs-pipe-data-flow 3.2s linear infinite; animation-delay:var(--dot-delay); }}
        @keyframes vs-pipe-data-flow {{ from {{ left:-3%; opacity:0; }}
            10%{{ opacity:1; }} 90%{{ opacity:1; }} to {{ left:103%; opacity:0; }} }}
        .vs-cnn-pipeline p {{ color:var(--vs-muted); line-height:1.62; margin:.7rem 0 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 梯度监控仪表盘
# ──────────────────────────────────────────────


def render_gradient_monitor(mode: str = "normal") -> None:
    """梯度监控仪表盘：展示正常/消失/爆炸三种梯度状态的可视化对比。

    Parameters
    ----------
    mode : str
        梯度状态，可选 'normal' | 'vanishing' | 'exploding'。
        若传入 'all' 则同时展示三种状态对比。
    """
    st = _st()
    configs = {
        "normal": {
            "label": "正常梯度",
            "color": NEON_GREEN,
            "icon": "fa-solid fa-circle-check",
            "desc": "各层梯度量级相近，训练稳定收敛。",
            "bars": [0.85, 0.80, 0.78, 0.75, 0.72, 0.70, 0.68],
        },
        "vanishing": {
            "label": "梯度消失",
            "color": NEON_PURPLE,
            "icon": "fa-solid fa-ghost",
            "desc": "深层梯度趋近零，前面的层几乎无法学习。",
            "bars": [0.82, 0.50, 0.22, 0.08, 0.02, 0.005, 0.001],
        },
        "exploding": {
            "label": "梯度爆炸",
            "color": "#ff4d6a",
            "icon": "fa-solid fa-explosion",
            "desc": "梯度指数增长，参数更新剧烈震荡甚至 NaN。",
            "bars": [0.10, 0.35, 0.80, 1.50, 3.20, 7.50, 15.0],
        },
    }
    if mode == "all":
        modes_to_show = ["normal", "vanishing", "exploding"]
    else:
        modes_to_show = [mode]
    for m in modes_to_show:
        cfg = configs[m]
        layer_names = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6", "Layer 7"]
        max_val = max(cfg["bars"]) if max(cfg["bars"]) > 0 else 1.0
        bars_html = ""
        for idx, (lname, val) in enumerate(zip(layer_names, cfg["bars"])):
            pct = min(100, val / max_val * 100)
            delay = round(idx * 0.1, 2)
            bars_html += (
                f'<div class="vs-grad-bar-row">'
                f'<span class="vs-grad-bar-label">{escape(lname)}</span>'
                f'<div class="vs-grad-bar-track">'
                f'<div class="vs-grad-bar-fill" style="--bar-w:{pct:.1f}%;--bar-color:{cfg["color"]};--bar-delay:{delay}s"></div>'
                f'</div>'
                f'<span class="vs-grad-bar-val">{val:.3g}</span>'
                f'</div>'
            )
        st.markdown(
            f"""
            <div class="vs-card vs-grad-monitor" style="--gm-color:{cfg['color']}">
              <div class="vs-panel-title"><i class="{cfg['icon']}"></i> 梯度监控 — {escape(cfg['label'])}</div>
              <div class="vs-grad-bars">{bars_html}</div>
              <p>{escape(cfg['desc'])}</p>
            </div>
            <style>
            .vs-grad-monitor {{ padding:1rem; }}
            .vs-grad-bars {{ display:flex; flex-direction:column; gap:.4rem; }}
            .vs-grad-bar-row {{ display:flex; align-items:center; gap:.55rem; }}
            .vs-grad-bar-label {{ width:58px; font-size:.76rem; color:var(--vs-muted); text-align:right; flex-shrink:0; }}
            .vs-grad-bar-track {{ flex:1; height:14px; border-radius:7px; background:rgba(255,255,255,.06); overflow:hidden; }}
            .vs-grad-bar-fill {{ height:100%; border-radius:7px; width:0; background:var(--bar-color);
                box-shadow:0 0 12px var(--bar-color); animation:vs-grad-fill .7s ease forwards; animation-delay:var(--bar-delay); }}
            @keyframes vs-grad-fill {{ to {{ width:var(--bar-w); }} }}
            .vs-grad-bar-val {{ width:52px; font-size:.76rem; font-family:"JetBrains Mono"; color:var(--gm-color); flex-shrink:0; }}
            .vs-grad-monitor p {{ color:var(--vs-muted); line-height:1.62; margin:.7rem 0 0; }}
            </style>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# 训练动态监控面板
# ──────────────────────────────────────────────


def render_training_dynamics_panel() -> None:
    """训练动态监控面板：用 Plotly 绘制 loss、accuracy、learning rate、gradient norm 四条实时曲线。

    使用模拟数据演示，实际调用时可传入真实训练数据。
    """
    st = _st()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import math

    epochs = list(range(1, 51))
    # 模拟训练曲线
    loss_vals = [2.3 * math.exp(-0.06 * e) + 0.15 + 0.05 * math.sin(e * 0.8) for e in epochs]
    acc_vals = [1 - 0.85 * math.exp(-0.07 * e) + 0.02 * math.sin(e * 0.6) for e in epochs]
    lr_vals = [0.01 * (0.95 ** e) for e in epochs]
    grad_vals = [0.5 * math.exp(-0.02 * e) + 0.08 * abs(math.sin(e * 0.4)) for e in epochs]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Loss", "Accuracy", "Learning Rate", "Gradient Norm"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=loss_vals, line=dict(color=NEON_PURPLE, width=2), name="Loss"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=acc_vals, line=dict(color=NEON_GREEN, width=2), name="Accuracy"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=lr_vals, line=dict(color=NEON_BLUE, width=2), name="LR"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=grad_vals, line=dict(color="#ffd166", width=2), name="Grad Norm"),
        row=2, col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        font=dict(family="JetBrains Mono, Inter", color="#eaf7ff", size=11),
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
        height=420,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(family="Inter", size=13, color="#eaf7ff")

    st.markdown(
        """
        <div class="vs-card" style="padding:1rem;">
          <div class="vs-panel-title"><i class="fa-solid fa-heart-pulse"></i> 训练动态监控面板</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<p style="color:var(--vs-muted);font-size:.86rem;line-height:1.55;margin:-.3rem 0 .8rem;">'
        '四条曲线同步观察：Loss 应稳步下降，Accuracy 应逐步上升，LR 按策略衰减，Gradient Norm 反映各层梯度健康度。',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 训练曲线实时扫描光标动效
# ──────────────────────────────────────────────


def render_training_curve_scanner() -> None:
    """训练曲线实时扫描光标动效：一条 loss 下降曲线，发光竖线光标从左到右扫描。

    CSS 动画驱动，光标经过时显示当前 epoch 的 loss 值，风格与 render_gradient_descent_landscape() 统一。
    """
    st = _st()
    import math

    # 模拟 50 个 epoch 的 loss 曲线
    n_epochs = 50
    loss_points = []
    for e in range(n_epochs):
        loss = 2.3 * math.exp(-0.06 * (e + 1)) + 0.18 + 0.05 * math.sin(e * 0.8)
        loss_points.append(loss)

    # 生成 SVG 折线路径点
    svg_w, svg_h = 900, 240
    pad_l, pad_r, pad_t, pad_b = 40, 20, 25, 35
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    min_loss = min(loss_points) * 0.92
    max_loss = max(loss_points) * 1.06
    range_loss = max_loss - min_loss or 1

    points = []
    for i, loss in enumerate(loss_points):
        x = pad_l + (i / (n_epochs - 1)) * plot_w
        y = pad_t + (1 - (loss - min_loss) / range_loss) * plot_h
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    # 渐变填充区域
    fill_path = (
        f"M{points[0][0]:.1f},{pad_t + plot_h} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in points)
        + f" L{points[-1][0]:.1f},{pad_t + plot_h} Z"
    )

    # Y 轴刻度
    y_ticks_html = ""
    for i in range(5):
        val = min_loss + (range_loss * i / 4)
        yy = pad_t + (1 - i / 4) * plot_h
        y_ticks_html += (
            f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" '
            f'text-anchor="end" fill="rgba(157,178,199,.6)" font-size="10" '
            f'font-family="JetBrains Mono">{val:.2f}</text>'
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{svg_w - pad_r}" y2="{yy:.1f}" '
            f'stroke="rgba(0,240,255,.07)" stroke-width="1"/>'
        )

    # X 轴刻度
    x_ticks_html = ""
    for i in range(0, n_epochs, 10):
        x = pad_l + (i / (n_epochs - 1)) * plot_w
        x_ticks_html += (
            f'<text x="{x:.1f}" y="{svg_h - 6}" text-anchor="middle" '
            f'fill="rgba(157,178,199,.6)" font-size="10" '
            f'font-family="JetBrains Mono">{i + 1}</text>'
        )

    # 光标扫过的 epoch 值标注（关键点）
    key_epochs = [0, 9, 19, 29, 39, 49]
    labels_html = ""
    for idx in key_epochs:
        x, y = points[idx]
        loss = loss_points[idx]
        delay = round(3.5 * (idx / (n_epochs - 1)), 2)
        labels_html += (
            f'<g class="vs-scan-label" style="animation-delay:{delay}s">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{NEON_GREEN}" opacity="0"/>'
            f'<rect x="{x - 28:.1f}" y="{y - 28:.1f}" width="56" height="20" rx="5" '
            f'fill="rgba(0,255,136,.15)" stroke="{NEON_GREEN}" stroke-width="0.8" opacity="0"/>'
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" '
            f'fill="{NEON_GREEN}" font-size="10" font-weight="700" '
            f'font-family="JetBrains Mono" opacity="0">{loss:.3f}</text>'
            f'</g>'
        )

    st.markdown(
        f"""
        <div class="vs-card vs-curve-scanner" title="训练曲线扫描：光标从左到右逐 epoch 扫描 loss 下降过程。">
          <div class="vs-panel-title"><i class="fa-solid fa-wave-square"></i> 训练曲线实时扫描</div>
          <div class="vs-scanner-wrap">
            <svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="vs-scan-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="{NEON_PURPLE}" stop-opacity="0.25"/>
                  <stop offset="100%" stop-color="{NEON_PURPLE}" stop-opacity="0.02"/>
                </linearGradient>
                <linearGradient id="vs-scan-line" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="{NEON_PURPLE}" stop-opacity="0.6"/>
                  <stop offset="50%" stop-color="{NEON_BLUE}" stop-opacity="1"/>
                  <stop offset="100%" stop-color="{NEON_GREEN}" stop-opacity="0.8"/>
                </linearGradient>
                <filter id="vs-glow">
                  <feGaussianBlur stdDeviation="3" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="vs-glow-strong">
                  <feGaussianBlur stdDeviation="6" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>
              {y_ticks_html}
              {x_ticks_html}
              <!-- 填充区域 -->
              <path d="{fill_path}" fill="url(#vs-scan-fill)"/>
              <!-- 曲线 -->
              <polyline points="{polyline}" fill="none" stroke="url(#vs-scan-line)"
                stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" filter="url(#vs-glow)"/>
              <!-- 关键点标注 -->
              {labels_html}
              <!-- 扫描光标（竖线 + 光晕） -->
              <g class="vs-scan-cursor">
                <rect x="-1.5" y="{pad_t}" width="3" height="{plot_h}" rx="1.5"
                  fill="{NEON_BLUE}" filter="url(#vs-glow-strong)"/>
                <ellipse cx="0" cy="{pad_t + plot_h / 2:.1f}" rx="18" ry="{plot_h / 2:.1f}"
                  fill="{NEON_BLUE}" opacity="0.08"/>
              </g>
              <!-- 坐标轴标签 -->
              <text x="{svg_w / 2:.1f}" y="{svg_h - 0}" text-anchor="middle"
                fill="rgba(157,178,199,.55)" font-size="11" font-family="Inter">Epoch</text>
              <text x="10" y="{svg_h / 2:.1f}" text-anchor="middle"
                fill="rgba(157,178,199,.55)" font-size="11" font-family="Inter"
                transform="rotate(-90, 10, {svg_h / 2:.1f})">Loss</text>
            </svg>
          </div>
          <p>光标从 Epoch 1 扫描到 Epoch 50，直观展示 loss 随训练逐步收敛的过程。早期下降剧烈，后期趋于平稳——这是典型的指数衰减学习曲线。</p>
        </div>
        <style>
        .vs-curve-scanner{{padding:1rem;overflow:hidden}}
        .vs-scanner-wrap{{position:relative;border-radius:8px;overflow:hidden;
          border:1px solid rgba(0,240,255,.22);background:rgba(0,0,0,.22)}}
        .vs-scanner-wrap svg{{display:block;width:100%;height:auto}}
        /* 扫描光标动画：从左到右 */
        .vs-scan-cursor{{
          animation:vs-scan-sweep 3.5s ease-in-out infinite;
        }}
        @keyframes vs-scan-sweep{{
          0%{{transform:translateX({pad_l}px);opacity:.3}}
          5%{{opacity:1}}
          95%{{opacity:1}}
          100%{{transform:translateX({pad_l + plot_w}px);opacity:.3}}
        }}
        /* 关键点标注：光标到达时弹出 */
        .vs-scan-label{{opacity:0;animation:vs-label-pop 3.5s ease-in-out infinite}}
        @keyframes vs-label-pop{{
          0%,8%{{opacity:0;transform:scale(.85)}}
          12%,88%{{opacity:1;transform:scale(1)}}
          92%,100%{{opacity:0;transform:scale(.85)}}
        }}
        .vs-curve-scanner p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 高级卷积对比动效
# ──────────────────────────────────────────────


def render_advanced_conv_comparison() -> None:
    """高级卷积对比动效：展示普通卷积、扩张卷积、分组卷积、深度可分离卷积的区别。

    用 CSS 动画展示卷积核滑动效果，不同颜色区分各类型。
    """
    st = _st()
    conv_types = [
        ("标准卷积", "3×3, stride=1, padding=1", NEON_BLUE,
         "所有输入通道参与所有输出通道，参数量 = C_in × C_out × K × K",
         "standard"),
        ("扩张卷积", "3×3, dilation=2, 感受野=5×5", NEON_PURPLE,
         "在卷积核元素间插入空洞，在不增加参数的前提下扩大感受野",
         "dilated"),
        ("分组卷积", "3×3, groups=2, 参数减半", NEON_GREEN,
         "将输入通道分成若干组，每组独立卷积，大幅减少计算量",
         "grouped"),
        ("深度可分离", "DW 3×3 + PW 1×1, 参数≈1/9", "#ffd166",
         "先逐通道空间卷积，再逐点通道混合，MobileNet 的核心技巧",
         "depthwise"),
    ]

    stages_html = []
    for idx, (name, spec, color, desc, css_class) in enumerate(conv_types):
        delay = round(idx * 0.15, 2)
        # 生成 3×3 卷积核小格子
        kernel_cells = ""
        for r in range(3):
            for c in range(3):
                cell_opacity = 1.0
                if css_class == "dilated" and r == 1 and c == 1:
                    cell_opacity = 0.0  # 中心空洞
                elif css_class == "grouped":
                    group_id = (r + c) % 2
                    cell_opacity = 0.9 if group_id == 0 else 0.35
                elif css_class == "depthwise":
                    cell_opacity = 0.85
                kernel_cells += (
                    f'<span class="vs-ck-cell" style="--cr:{r};--cc:{c};'
                    f'opacity:{cell_opacity};--ck-color:{color}"></span>'
                )

        # 滑动光标
        slider = f'<span class="vs-ck-slider vs-ck-{css_class}" style="--ck-color:{color}"></span>'

        stages_html.append(
            f"""
            <div class="vs-conv-kind" style="--ck-color:{color};--ck-delay:{delay}s">
              <div class="vs-ck-header">
                <span class="vs-ck-name">{escape(name)}</span>
                <code>{escape(spec)}</code>
              </div>
              <div class="vs-ck-kernel vs-ck-grid-{css_class}">
                {kernel_cells}
                {slider}
              </div>
              <p class="vs-ck-desc">{escape(desc)}</p>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="vs-card vs-conv-compare" title="四种卷积类型对比：感受野、参数量、计算方式各不相同。">
          <div class="vs-panel-title"><i class="fa-solid fa-layer-group"></i> 高级卷积对比</div>
          <div class="vs-cc-grid">{''.join(stages_html)}</div>
          <p>标准卷积是基线；扩张卷积用空洞换感受野；分组卷积用分治换速度；深度可分离卷积用两步分解换极致轻量。</p>
        </div>
        <style>
        .vs-conv-compare{{padding:1rem;overflow:hidden}}
        .vs-cc-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem}}
        .vs-conv-kind{{border:1px solid rgba(0,240,255,.22);border-radius:8px;padding:.65rem;
            background:rgba(0,0,0,.18);animation:vs-ck-in .5s ease both;animation-delay:var(--ck-delay)}}
        @keyframes vs-ck-in{{from{{opacity:0;transform:translateY(10px) scale(.92)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}
        .vs-ck-header{{display:flex;align-items:center;gap:.35rem;margin-bottom:.5rem;flex-wrap:wrap}}
        .vs-ck-name{{font-weight:800;font-size:.82rem;color:var(--ck-color)}}
        .vs-ck-header code{{font-size:.65rem;color:var(--vs-muted);background:rgba(255,255,255,.05);
            padding:.1rem .35rem;border-radius:4px}}
        .vs-ck-kernel{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;width:72px;height:72px;
            position:relative;margin:0 auto .5rem}}
        .vs-ck-cell{{border-radius:4px;background:var(--ck-color);box-shadow:0 0 8px var(--ck-color);
            animation:vs-ck-cell-pulse 2.2s ease-in-out infinite;animation-delay:calc(var(--cr)*.12s + var(--cc)*.08s)}}
        @keyframes vs-ck-cell-pulse{{0%,100%{{opacity:.35;transform:scale(.88)}}50%{{opacity:1;transform:scale(1.05)}}}}
        .vs-ck-dilated .vs-ck-cell:nth-child(5){{opacity:0!important;background:transparent!important;box-shadow:none!important}}
        .vs-ck-slider{{position:absolute;width:100%;height:3px;background:var(--ck-color);border-radius:3px;
            top:50%;left:0;box-shadow:0 0 14px var(--ck-color);animation:vs-ck-slide 2.2s ease-in-out infinite}}
        @keyframes vs-ck-slide{{0%{{top:0;opacity:.4}}50%{{top:90%;opacity:1}}100%{{top:0;opacity:.4}}}}
        .vs-ck-desc{{color:var(--vs-muted);font-size:.72rem;line-height:1.45;margin:0}}
        .vs-conv-compare>p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        @media(max-width:900px){{.vs-cc-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
        @media(max-width:520px){{.vs-cc-grid{{grid-template-columns:1fr}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# RNN 隐藏状态传递动效
# ──────────────────────────────────────────────


def render_rnn_hidden_state_flow() -> None:
    """RNN 隐藏状态传递动效：展示时间步 t=0,1,2,3 的隐藏状态传递。

    用粒子流动效果表示信息传递，展示 h0→h1→h2→h3 的状态变化。
    """
    st = _st()
    time_steps = ["t=0", "t=1", "t=2", "t=3"]
    tokens = ["我", "喜欢", "深度", "学习"]
    hidden_states = ["h₀", "h₁", "h₂", "h₃"]

    # 每个时间步的隐藏状态强度（模拟信息积累）
    intensities = [0.35, 0.55, 0.75, 0.95]

    nodes = []
    for idx, (ts, tok, hid, intensity) in enumerate(zip(time_steps, tokens, hidden_states, intensities)):
        delay = round(idx * 0.22, 2)
        nodes.append(
            f"""
            <div class="vs-rnn-node" style="--rn-delay:{delay}s;--rn-int:{intensity}">
              <div class="vs-rnn-cell">
                <span class="vs-rnn-hidden">{escape(hid)}</span>
              </div>
              <span class="vs-rnn-token">{escape(tok)}</span>
              <span class="vs-rnn-ts">{escape(ts)}</span>
            </div>
            """
        )

    # 连接箭头和粒子流
    arrows = []
    particles = []
    for idx in range(len(time_steps) - 1):
        delay = round(idx * 0.22 + 0.11, 2)
        arrows.append(
            f'<div class="vs-rnn-arrow" style="--ra-delay:{delay}s"><i class="fa-solid fa-arrow-right"></i></div>'
        )
        # 粒子：从左节点飞向右节点
        for p in range(5):
            pd = round(idx * 0.22 + p * 0.06, 2)
            particles.append(
                f'<span class="vs-rnn-particle" style="--rp-delay:{pd}s;--rp-idx:{idx}"></span>'
            )

    st.markdown(
        f"""
        <div class="vs-card vs-rnn-flow" title="RNN 隐藏状态沿时间步传递：每个 h_t 既接收输入 x_t，也接收上一步的 h_(t-1)。">
          <div class="vs-panel-title"><i class="fa-solid fa-arrow-trend-up"></i> RNN 隐藏状态传递</div>
          <div class="vs-rnn-timeline">
            <div class="vs-rnn-nodes">{''.join(nodes)}</div>
            <div class="vs-rnn-arrows">{''.join(arrows)}</div>
            <div class="vs-rnn-particles">{''.join(particles)}</div>
          </div>
          <p>每个时间步的隐藏状态 h_t = σ(W_hh · h_(t-1) + W_xh · x_t + b)。信息沿时间轴累积，越靠后的状态承载越多上下文。</p>
        </div>
        <style>
        .vs-rnn-flow{{padding:1rem;overflow:hidden}}
        .vs-rnn-timeline{{position:relative;min-height:180px}}
        .vs-rnn-nodes{{display:flex;justify-content:space-between;align-items:center;position:relative;z-index:2}}
        .vs-rnn-node{{display:flex;flex-direction:column;align-items:center;gap:.3rem;
            animation:vs-rnn-in .5s ease both;animation-delay:var(--rn-delay)}}
        @keyframes vs-rnn-in{{from{{opacity:0;transform:translateY(14px) scale(.88)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}
        .vs-rnn-cell{{width:72px;height:72px;border-radius:14px;display:grid;place-items:center;
            border:2px solid rgba(176,0,255,.55);background:rgba(176,0,255,calc(var(--rn-int)*.22));
            box-shadow:0 0 24px rgba(176,0,255,calc(var(--rn-int)*.45));
            animation:vs-rnn-glow 2.4s ease-in-out infinite alternate;animation-delay:var(--rn-delay)}}
        @keyframes vs-rnn-glow{{from{{box-shadow:0 0 12px rgba(176,0,255,.15)}}to{{box-shadow:0 0 32px rgba(176,0,255,.55)}}}}
        .vs-rnn-hidden{{font-family:"JetBrains Mono";font-weight:700;font-size:1rem;color:var(--vs-ink);
            text-shadow:0 0 12px rgba(176,0,255,.6)}}
        .vs-rnn-token{{font-size:.85rem;color:var(--vs-blue);font-weight:700;
            background:rgba(0,240,255,.08);border:1px solid rgba(0,240,255,.3);border-radius:6px;
            padding:.15rem .5rem}}
        .vs-rnn-ts{{font-family:"JetBrains Mono";font-size:.7rem;color:var(--vs-muted)}}
        .vs-rnn-arrows{{position:absolute;top:50%;left:0;right:0;display:flex;justify-content:space-around;
            transform:translateY(-50%);z-index:1;pointer-events:none}}
        .vs-rnn-arrow{{color:var(--vs-purple);font-size:.9rem;opacity:.5;
            animation:vs-rnn-arrow-pulse 1.5s ease-in-out infinite;animation-delay:var(--ra-delay)}}
        @keyframes vs-rnn-arrow-pulse{{0%,100%{{opacity:.25;transform:scale(.8)}}50%{{opacity:1;transform:scale(1.2)}}}}
        .vs-rnn-particles{{position:absolute;inset:0;z-index:3;pointer-events:none}}
        .vs-rnn-particle{{position:absolute;width:5px;height:5px;border-radius:50%;top:45%;
            background:var(--vs-green);box-shadow:0 0 12px var(--vs-green);
            animation:vs-rnn-particle-fly 2.4s linear infinite;animation-delay:var(--rp-delay);
            left:calc(var(--rp-idx)*25% + 12%)}}
        @keyframes vs-rnn-particle-fly{{0%{{transform:translateX(0);opacity:0}}15%{{opacity:1}}85%{{opacity:1}}100%{{transform:translateX(calc(25vw * 0.8));opacity:0}}}}
        .vs-rnn-flow>p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        @media(max-width:600px){{.vs-rnn-cell{{width:56px;height:56px}}.vs-rnn-hidden{{font-size:.82rem}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Transformer 注意力热力图动效
# ──────────────────────────────────────────────


def render_transformer_attention_heatmap(
    tokens: list[str] | None = None,
    num_heads: int = 4,
) -> None:
    """Transformer 多头注意力热力图动效。

    展示多头注意力的权重分布，用热力图展示 Q·K^T 的注意力分数，
    支持动态切换注意力头。

    Parameters
    ----------
    tokens : list[str] | None
        输入 token 列表，默认 ["The", "cat", "sat", "on", "mat"]。
    num_heads : int
        注意力头数量，默认 4。
    """
    st = _st()
    tokens = tokens or ["The", "cat", "sat", "on", "mat"]
    tokens = tokens[:6]  # 最多 6 个 token
    n = len(tokens)

    # 为每个头生成模拟注意力权重矩阵
    import math

    head_configs = [
        ("头 1 — 局部关注", NEON_BLUE, lambda i, j, n: max(0.05, 1.0 - abs(i - j) * 0.35)),
        ("头 2 — 全局均等", NEON_PURPLE, lambda i, j, n: 0.4 + 0.2 * math.sin(i + j)),
        ("头 3 — 对角强", NEON_GREEN, lambda i, j, n: 0.9 if i == j else 0.1 + 0.08 * (i + j)),
        ("头 4 — 起始聚焦", "#ffd166", lambda i, j, n: max(0.05, 1.2 - j * 0.25)),
    ]

    # 生成热力图单元格
    def make_heatmap_cells(weight_fn, color, head_idx):
        cells = ""
        for i in range(n):
            for j in range(n):
                w = weight_fn(i, j, n)
                w = max(0.0, min(1.0, w))
                opacity = round(w * 0.85 + 0.1, 2)
                delay = round((i * n + j) * 0.03, 2)
                cells += (
                    f'<span class="vs-attn-cell" '
                    f'style="--ac-color:{color};--ac-opacity:{opacity};--ac-delay:{delay}s" '
                    f'title="{escape(tokens[i])}→{escape(tokens[j])}: {w:.2f}">'
                    f'{w:.1f}</span>'
                )
        return cells

    # 构建所有头的热力图
    heads_html = []
    for h_idx in range(num_heads):
        label, color, fn = head_configs[h_idx % len(head_configs)]
        cells = make_heatmap_cells(fn, color, h_idx)
        # 行标签
        row_labels = "".join(
            f'<span class="vs-attn-rlbl">{escape(t)}</span>' for t in tokens
        )
        # 列标签
        col_labels = "".join(
            f'<span class="vs-attn-clbl">{escape(t)}</span>' for t in tokens
        )
        active_class = "vs-head-active" if h_idx == 0 else ""
        heads_html.append(
            f"""
            <div class="vs-attn-head {active_class}" data-head="{h_idx}"
                 style="--head-color:{color};--head-idx:{h_idx}">
              <div class="vs-attn-head-label" style="color:{color}">
                <i class="fa-solid fa-brain"></i> {escape(label)}
              </div>
              <div class="vs-attn-grid-wrap">
                <div class="vs-attn-row-labels">{row_labels}</div>
                <div class="vs-attn-grid" style="grid-template-columns:repeat({n},1fr)">
                  {col_labels}
                  {cells}
                </div>
              </div>
            </div>
            """
        )

    # 切换按钮的 HTML（用模板避免多层转义）
    btns = ""
    for h in range(num_heads):
        color = head_configs[h % len(head_configs)][1]
        # onclick 用 data 属性 + 简化 JS，避免嵌套引号地狱
        btns += (
            f'<button class="vs-head-btn" data-bh="{h}" '
            f'style="--btn-color:{color}" '
            f'onclick="vsSwitchHead({h})">'
            f'Head {h + 1}</button>'
        )

    st.markdown(
        f"""
        <div class="vs-card vs-attn-heatmap" title="多头注意力热力图：颜色越深表示注意力权重越高。">
          <div class="vs-panel-title"><i class="fa-solid fa-fire-flame-curved"></i> Transformer 多头注意力热力图</div>
          <div class="vs-head-btns">
            <span class="vs-head-btn-label">切换注意力头:</span>
            {btns}
          </div>
          <div class="vs-attn-heads">
            {''.join(heads_html)}
          </div>
          <p>Q·K<sup>T</sup> / √d_k 的注意力分数经 softmax 后得到权重矩阵。不同头学到不同的注意力模式：有的看局部上下文，有的捕捉长距离依赖。</p>
        </div>
        <style>
        .vs-attn-heatmap{{padding:1rem;overflow:hidden}}
        .vs-head-btns{{display:flex;align-items:center;gap:.4rem;margin-bottom:.7rem;flex-wrap:wrap}}
        .vs-head-btn-label{{font-size:.82rem;color:var(--vs-muted);margin-right:.3rem}}
        .vs-head-btn{{border:1px solid var(--btn-color);border-radius:6px;padding:.3rem .7rem;
            background:rgba(0,0,0,.18);color:var(--vs-ink);font-size:.78rem;font-weight:700;
            cursor:pointer;transition:all .2s ease;font-family:Inter,sans-serif}}
        .vs-head-btn:hover{{background:rgba(255,255,255,.08);box-shadow:0 0 12px var(--btn-color)}}
        .vs-head-btn.vs-btn-active{{background:var(--btn-color);color:#061018;
            box-shadow:0 0 18px var(--btn-color)}}
        .vs-attn-heads{{min-height:240px}}
        .vs-attn-head{{display:none;animation:vs-attn-fade-in .35s ease both}}
        .vs-attn-head.vs-head-active{{display:block}}
        @keyframes vs-attn-fade-in{{from{{opacity:0;transform:scale(.96)}}to{{opacity:1;transform:scale(1)}}}}
        .vs-attn-head-label{{font-weight:800;font-size:.92rem;margin-bottom:.45rem;
            display:flex;align-items:center;gap:.35rem}}
        .vs-attn-head-label i{{filter:drop-shadow(0 0 6px var(--head-color))}}
        .vs-attn-grid-wrap{{display:flex;gap:.35rem;align-items:flex-start}}
        .vs-attn-row-labels{{display:flex;flex-direction:column;gap:2px;padding-top:22px}}
        .vs-attn-rlbl{{height:30px;display:grid;place-items:center;font-size:.68rem;
            color:var(--vs-muted);font-family:"JetBrains Mono"}}
        .vs-attn-grid{{display:grid;gap:2px;position:relative}}
        .vs-attn-clbl{{height:18px;display:grid;place-items:center;font-size:.62rem;
            color:var(--vs-muted);font-family:"JetBrains Mono"}}
        .vs-attn-cell{{width:52px;height:30px;border-radius:4px;display:grid;place-items:center;
            font-family:"JetBrains Mono";font-size:.62rem;color:rgba(255,255,255,.85);
            background:rgba(0,0,0,.22);border:1px solid rgba(255,255,255,.04);
            animation:vs-attn-cell-in .4s ease both;animation-delay:var(--ac-delay);
            transition:background .3s ease,box-shadow .3s ease}}
        .vs-attn-cell:hover{{box-shadow:0 0 16px var(--ac-color);z-index:2;transform:scale(1.08)}}
        @keyframes vs-attn-cell-in{{from{{opacity:0;transform:scale(.8)}}to{{opacity:var(--ac-opacity);transform:scale(1)}}}}
        /* 用伪元素实现背景热力效果 */
        .vs-attn-cell::before{{content:"";position:absolute;inset:0;border-radius:4px;
            background:var(--ac-color);opacity:calc(var(--ac-opacity)*.45)}}
        .vs-attn-cell{{position:relative}}
        .vs-attn-heatmap>p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        </style>
        <script>
        function vsSwitchHead(idx) {{
            document.querySelectorAll('.vs-attn-head').forEach(function(e) {{
                e.classList.remove('vs-head-active');
            }});
            var target = document.querySelector('.vs-attn-head[data-head="' + idx + '"]');
            if (target) target.classList.add('vs-head-active');
            document.querySelectorAll('.vs-head-btn').forEach(function(e) {{
                e.classList.remove('vs-btn-active');
            }});
            var btn = document.querySelector('.vs-head-btn[data-bh="' + idx + '"]');
            if (btn) btn.classList.add('vs-btn-active');
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 中央控制台模型拼装动效
# ──────────────────────────────────────────────


def render_central_console_assembly() -> None:
    """中央控制台模型拼装动效：展示用户拖拽拼装神经网络的过程。

    用 CSS 动画依次展示 Input → Hidden 1 → Hidden 2 → Output 四个层的拼装过程，
    每个层用不同颜色方块表示并带发光效果，同时展示数据在网络中流动的粒子动画
    以及模型配置参数的实时显示。
    """
    st = _st()

    layers = [
        ("输入层", "Input", "784 neurons", NEON_BLUE, "fa-solid fa-arrow-right-to-bracket"),
        ("隐藏层 1", "Hidden 1", "256 neurons", NEON_PURPLE, "fa-solid fa-layer-group"),
        ("隐藏层 2", "Hidden 2", "128 neurons", NEON_GREEN, "fa-solid fa-layer-group"),
        ("输出层", "Output", "10 neurons", "#ffd166", "fa-solid fa-arrow-right-from-bracket"),
    ]

    layer_blocks = []
    for idx, (label, short, detail, color, icon) in enumerate(layers):
        delay = round(idx * 0.6, 2)
        layer_blocks.append(
            f"""
            <div class="vs-asm-layer" style="--asm-color:{color};--asm-delay:{delay}s">
              <div class="vs-asm-block">
                <i class="{icon}"></i>
                <strong>{escape(short)}</strong>
                <code>{escape(detail)}</code>
              </div>
              <span class="vs-asm-label">{escape(label)}</span>
            </div>
            """
        )

    # 连接箭头
    arrows = "".join(
        f'<div class="vs-asm-arrow" style="--asa-delay:{round(idx * 0.6 + 0.3, 2)}s">'
        f'<i class="fa-solid fa-link"></i></div>'
        for idx in range(len(layers) - 1)
    )

    # 数据流动粒子
    flow_particles = "".join(
        f'<span class="vs-asm-particle" style="--asp-delay:{round(idx * 0.18, 2)}s;'
        f'--asp-color:{layers[idx % len(layers)][3]}"></span>'
        for idx in range(20)
    )

    # 拼装进度指示（模拟拖拽拼装过程）
    progress_segments = "".join(
        f'<div class="vs-asm-seg" style="--seg-color:{color};--seg-delay:{round(idx * 0.6 + 0.45, 2)}s"></div>'
        for idx, (_, _, _, color, _) in enumerate(layers)
    )

    # 模型配置参数
    config_params = [
        ("Loss", "CrossEntropy", NEON_PURPLE),
        ("Optimizer", "Adam (lr=0.001)", NEON_BLUE),
        ("Batch Size", "64", NEON_GREEN),
        ("Epochs", "20", "#ffd166"),
    ]
    config_items = "".join(
        f'<div class="vs-asm-param" style="--param-color:{color}">'
        f'<span class="vs-asm-param-key">{escape(key)}</span>'
        f'<span class="vs-asm-param-val">{escape(val)}</span></div>'
        for key, val, color in config_params
    )

    st.markdown(
        f"""
        <div class="vs-card vs-console-assembly" title="中央控制台：拖拽拼装神经网络，从输入层到输出层逐步组装完成。">
          <div class="vs-panel-title"><i class="fa-solid fa-gears"></i> 中央控制台 — 模型拼装</div>
          <div class="vs-asm-progress">{progress_segments}</div>
          <div class="vs-asm-stage">
            <div class="vs-asm-layers">{''.join(layer_blocks)}</div>
            <div class="vs-asm-arrows">{arrows}</div>
            <div class="vs-asm-flow">{flow_particles}</div>
          </div>
          <div class="vs-asm-config">
            <div class="vs-asm-config-title"><i class="fa-solid fa-sliders"></i> 模型配置</div>
            <div class="vs-asm-params">{config_items}</div>
          </div>
          <p>拖拽各层模块依次拼装：输入层接收数据 → 隐藏层提取特征 → 输出层生成预测。右侧实时显示当前模型配置参数。</p>
        </div>
        <style>
        .vs-console-assembly{{padding:1rem;overflow:hidden}}
        .vs-asm-progress{{display:flex;gap:4px;margin-bottom:.7rem}}
        .vs-asm-seg{{flex:1;height:5px;border-radius:3px;background:var(--seg-color);
            box-shadow:0 0 10px var(--seg-color);opacity:0;
            animation:vs-asm-seg-in .4s ease forwards;animation-delay:var(--seg-delay)}}
        @keyframes vs-asm-seg-in{{to{{opacity:1}}}}
        .vs-asm-stage{{position:relative;min-height:160px;border:1px solid rgba(0,240,255,.22);
            border-radius:10px;background:rgba(0,0,0,.18);padding:1.2rem .8rem;overflow:hidden}}
        .vs-asm-layers{{display:flex;justify-content:space-between;align-items:center;position:relative;z-index:2}}
        .vs-asm-layer{{display:flex;flex-direction:column;align-items:center;gap:.3rem;
            animation:vs-asm-drop .65s cubic-bezier(.34,1.56,.64,1) both;animation-delay:var(--asm-delay)}}
        @keyframes vs-asm-drop{{from{{opacity:0;transform:translateY(-30px) scale(.7) rotate(-6deg)}}
            to{{opacity:1;transform:translateY(0) scale(1) rotate(0deg)}}}}
        .vs-asm-block{{width:90px;height:80px;border-radius:10px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:.15rem;
            border:2px solid var(--asm-color);background:rgba(0,0,0,.35);
            box-shadow:0 0 22px rgba(0,0,0,.3),0 0 18px color-mix(in srgb,var(--asm-color) 35%,transparent);
            animation:vs-asm-glow 2.2s ease-in-out infinite alternate;animation-delay:var(--asm-delay)}}
        @keyframes vs-asm-glow{{from{{box-shadow:0 0 10px rgba(0,0,0,.3),0 0 8px color-mix(in srgb,var(--asm-color) 20%,transparent)}}
            to{{box-shadow:0 0 14px rgba(0,0,0,.2),0 0 30px color-mix(in srgb,var(--asm-color) 55%,transparent)}}}}
        .vs-asm-block i{{color:var(--asm-color);font-size:1rem;filter:drop-shadow(0 0 6px var(--asm-color))}}
        .vs-asm-block strong{{font-size:.78rem;color:var(--vs-ink);font-weight:800}}
        .vs-asm-block code{{font-size:.65rem;color:var(--vs-muted);background:rgba(255,255,255,.06);
            padding:.1rem .35rem;border-radius:4px}}
        .vs-asm-label{{font-size:.72rem;color:var(--asm-color);font-weight:700;
            text-shadow:0 0 8px color-mix(in srgb,var(--asm-color) 50%,transparent)}}
        .vs-asm-arrows{{position:absolute;top:50%;left:0;right:0;display:flex;justify-content:space-around;
            transform:translateY(-50%);z-index:1;pointer-events:none}}
        .vs-asm-arrow{{color:var(--vs-blue);font-size:.85rem;opacity:.5;
            animation:vs-asm-arrow-pulse 1.6s ease-in-out infinite;animation-delay:var(--asa-delay)}}
        @keyframes vs-asm-arrow-pulse{{0%,100%{{opacity:.2;transform:scale(.75)}}50%{{opacity:1;transform:scale(1.2)}}}}
        .vs-asm-flow{{position:absolute;inset:0;z-index:3;pointer-events:none}}
        .vs-asm-particle{{position:absolute;width:5px;height:5px;border-radius:50%;top:48%;
            background:var(--asp-color);box-shadow:0 0 10px var(--asp-color);
            animation:vs-asm-particle-fly 3s linear infinite;animation-delay:var(--asp-delay)}}
        @keyframes vs-asm-particle-fly{{0%{{left:-2%;opacity:0}}8%{{opacity:1}}92%{{opacity:1}}100%{{left:102%;opacity:0}}}}
        .vs-asm-config{{margin-top:.75rem;padding:.65rem .8rem;border:1px solid rgba(0,240,255,.18);
            border-radius:8px;background:rgba(0,0,0,.14)}}
        .vs-asm-config-title{{font-weight:800;font-size:.85rem;color:var(--vs-ink);margin-bottom:.45rem;
            display:flex;align-items:center;gap:.35rem}}
        .vs-asm-config-title i{{color:var(--vs-blue);filter:drop-shadow(0 0 6px var(--vs-blue))}}
        .vs-asm-params{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.5rem}}
        .vs-asm-param{{display:flex;flex-direction:column;align-items:center;gap:.15rem;
            padding:.35rem .4rem;border-radius:6px;border:1px solid color-mix(in srgb,var(--param-color) 30%,transparent);
            background:rgba(0,0,0,.18);animation:vs-asm-param-in .5s ease both;
            animation-delay:calc(var(--asm-delay, 0s) + .5s)}}
        .vs-asm-param-key{{font-size:.68rem;color:var(--vs-muted);text-transform:uppercase;letter-spacing:.04em}}
        .vs-asm-param-val{{font-family:"JetBrains Mono";font-size:.78rem;font-weight:700;
            color:var(--param-color);text-shadow:0 0 6px color-mix(in srgb,var(--param-color) 40%,transparent)}}
        @keyframes vs-asm-param-in{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
        .vs-console-assembly>p{{color:var(--vs-muted);line-height:1.62;margin:.7rem 0 0}}
        @media(max-width:700px){{
            .vs-asm-block{{width:68px;height:62px}}
            .vs-asm-block strong{{font-size:.68rem}}
            .vs-asm-params{{grid-template-columns:repeat(2,minmax(0,1fr))}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 运动画廊（展示所有动效）
# ──────────────────────────────────────────────


def render_motion_gallery() -> None:
    """展示所有教学动效组件的画廊，含新增组件演示。"""
    st = _st()
    st.subheader("核心教学动效")
    render_loading_bar("页面动效服务于观察：每一种发光都对应一个可解释的学习信号")

    # ── 通用组件区 ──
    st.markdown("### 通用 UI 组件")
    st.markdown(
        render_tooltip_label("什么是教学动效？", "动效必须对应一个可解释的学习信号，例如方向、强度、权重或状态变化。"),
        unsafe_allow_html=True,
    )
    render_motion_note(
        "动效观察顺序",
        "先看数据从哪里来，再看它流向哪里；最后把颜色亮度对应到权重、梯度或激活强度。",
    )
    render_beginner_hint(
        "小白先抓三个词",
        "方向表示信息流，颜色表示强弱，停顿表示当前步骤的关键节点。",
        action="看到图时不要急着看公式，先用一句话说出“谁影响了谁”。",
    )
    c0a, c0b, c0c = st.columns(3)
    with c0a:
        render_neon_metric_card("Accuracy", "98.2%", delta="+0.5%", icon="fa-solid fa-bullseye", accent=NEON_GREEN, caption="越高不一定越好，还要看验证集。")
    with c0b:
        render_neon_metric_card("Loss", "0.042", delta="-0.008", icon="fa-solid fa-fire", accent=NEON_PURPLE, caption="下降表示优化正在找到更低误差。")
    with c0c:
        render_neon_metric_card("Epoch", "127/200", icon="fa-solid fa-rotate", accent=NEON_BLUE, caption="训练轮次只说明看过数据几遍。")
    render_responsive_motion_grid(
        [
            f'<div class="vs-card" style="padding:.9rem">{render_status_badge("运行中", status="running")}<p>用于标记正在刷新或正在训练的教学面板。</p></div>',
            f'<div class="vs-card" style="padding:.9rem">{render_status_badge("已通过", status="success")}<p>用于标记 smoke、质量门或实验完成状态。</p></div>',
            f'<div class="vs-card" style="padding:.9rem">{render_status_badge("需复查", status="warning")}<p>用于提醒参数极端、图像异常或理解薄弱点。</p></div>',
        ]
    )
    render_shape_flow(
        [
            ("输入图片", "B×1×28×28"),
            ("卷积特征", "B×16×24×24"),
            ("展平", "B×9216"),
            ("分类输出", "B×10"),
        ],
        title="CNN 张量形状流",
    )
    render_card(
        "快速入门",
        "<b>Step 1</b> 导入 PyTorch → <b>Step 2</b> 定义模型 → <b>Step 3</b> 训练循环。就这么简单。",
        icon="fa-solid fa-rocket",
        accent=NEON_GREEN,
        footer="适用于所有章节的最小可运行示例",
    )

    # ── CNN 管线 ──
    st.markdown("### CNN 层级管线")
    render_cnn_layer_pipeline()

    # ── 梯度监控 ──
    st.markdown("### 梯度监控仪表盘")
    render_gradient_monitor("all")

    # ── 训练动态面板 ──
    st.markdown("### 训练动态监控面板")
    render_training_dynamics_panel()

    # ── 高级卷积对比 ──
    st.markdown("### 高级卷积对比")
    render_advanced_conv_comparison()

    # ── RNN 隐藏状态传递 ──
    st.markdown("### RNN 隐藏状态传递")
    render_rnn_hidden_state_flow()

    # ── 中央控制台模型拼装 ──
    st.markdown("### 中央控制台模型拼装")
    render_central_console_assembly()

    # ── Transformer 注意力热力图 ──
    st.markdown("### Transformer 注意力热力图")
    render_transformer_attention_heatmap()

    # ── 原有动效 ──
    st.markdown("### 交互式教学动效")
    c1, c2 = st.columns(2)
    with c1:
        render_convolution_particle_flow()
        render_attention_light_beams(["I", "love", "deep", "learning", "because", "it", "works"], 3)
    with c2:
        render_gradient_descent_landscape()
        render_training_dashboard_gauges()
    render_backprop_current_flow()
