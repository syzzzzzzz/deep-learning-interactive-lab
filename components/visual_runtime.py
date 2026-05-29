"""Streamlit CSS runtime for the shared teaching visual system."""

from __future__ import annotations

from html import escape
from math import sin
from textwrap import dedent

from components.visual_tokens import BRAND_GOLD
from components.visual_tokens import BRAND_INK
from components.visual_tokens import BRAND_MUTED
from components.visual_tokens import BRAND_SOFT
from components.visual_tokens import LIGHT_BLUE
from components.visual_tokens import LIGHT_GREEN
from components.visual_tokens import LIGHT_PURPLE
from components.visual_tokens import NEON_BLUE
from components.visual_tokens import NEON_GREEN
from components.visual_tokens import NEON_PURPLE
class _StreamlitProxy:
    """Keep raw HTML out of Markdown's parser while preserving st.* ergonomics."""

    def __init__(self, st):
        self._st = st

    def __getattr__(self, name: str):
        return getattr(self._st, name)

    def markdown(self, body, unsafe_allow_html: bool = False, **kwargs):
        if not unsafe_allow_html:
            return self._st.markdown(body, unsafe_allow_html=False, **kwargs)

        html = dedent(str(body)).strip()
        if hasattr(self._st, "html"):
            width = kwargs.pop("width", "stretch")
            allow_js = "<script" in html.lower()
            return self._st.html(html, width=width, unsafe_allow_javascript=allow_js)
        return self._st.markdown(html, unsafe_allow_html=True, **kwargs)


def _st():
    import streamlit as st

    return _StreamlitProxy(st)


def render_visual_system(theme: str = "light", *, particles: bool | None = None) -> None:
    """Inject the shared teaching visual language and motion system."""

    st = _st()
    dark = theme == "dark"
    if particles is None:
        particles = dark
    bg = "#171411" if dark else "#ffffff"
    panel = "rgba(33,26,20,0.88)" if dark else "rgba(255,255,255,0.94)"
    panel_soft = "rgba(42,33,24,0.76)" if dark else "rgba(247,243,236,0.92)"
    ink = "#f7f3ec" if dark else "#171411"
    muted = "#c9beb0" if dark else BRAND_MUTED
    line = "rgba(176,138,79,0.30)" if dark else "rgba(230,222,210,0.92)"
    sidebar = "#1b1510" if dark else BRAND_SOFT
    code_bg = "#211a14" if dark else BRAND_SOFT
    vs_blue = NEON_BLUE if dark else LIGHT_BLUE
    vs_purple = NEON_PURPLE if dark else LIGHT_PURPLE
    vs_green = NEON_GREEN if dark else LIGHT_GREEN
    app_background = (
        """
                linear-gradient(180deg, rgba(23,20,17,0.96), rgba(42,33,24,0.98)),
                radial-gradient(circle at 12% 8%, rgba(176,138,79,0.10), transparent 32%),
                var(--vs-bg)
        """
        if dark
        else """
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,243,236,0.88)),
                var(--vs-bg)
        """
    )
    sidebar_background = (
        "linear-gradient(180deg, var(--vs-sidebar), rgba(9,13,23,0.94))"
        if dark
        else "linear-gradient(180deg, #f8f7f1, var(--vs-sidebar))"
    )
    card_shadow = (
        "0 0 0 1px rgba(255,255,255,0.03), 0 18px 42px rgba(0,0,0,0.24), 0 0 24px rgba(176,138,79,0.08)"
        if dark
        else "none"
    )
    button_border = "rgba(176,138,79,0.50)" if dark else "rgba(42,33,24,0.92)"
    button_background = (
        "linear-gradient(135deg, rgba(176,138,79,0.18), rgba(42,33,24,0.18))"
        if dark
        else "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,243,236,0.94))"
    )
    button_shadow = "0 0 16px rgba(176,138,79,0.10)" if dark else "none"
    button_hover_shadow = "0 0 22px rgba(176,138,79,0.18)" if dark else "0 8px 18px rgba(42,33,24,0.10)"
    icon_filter = "drop-shadow(0 0 8px rgba(176,138,79,0.32))" if dark else "none"
    tooltip_bg = "#211a14" if dark else "#ffffff"
    tooltip_color = "#f7f3ec" if dark else "#171411"
    tooltip_shadow = "0 16px 38px rgba(0,0,0,0.38), 0 0 18px rgba(176,138,79,0.14)" if dark else "0 12px 28px rgba(42,33,24,0.12)"
    surface_bg = "rgba(0,0,0,.18)" if dark else "rgba(255,253,247,.72)"
    stage_bg = "rgba(0,0,0,.18)" if dark else "rgba(255,253,247,.78)"
    stage_bg_strong = "rgba(0,0,0,.22)" if dark else "rgba(243,247,241,.92)"
    stage_card_bg = "#211a14" if dark else "#fffdf7"
    code_chip_bg = "rgba(255,255,255,.05)" if dark else "rgba(23,32,38,.06)"
    track_bg = "rgba(255,255,255,.06)" if dark else "rgba(176,138,79,.10)"
    soft_line = "rgba(255,255,255,.08)" if dark else "rgba(23,32,38,.10)"
    cell_ink = "rgba(255,255,255,.85)" if dark else "#171411"
    stage_shadow = (
        "0 0 18px color-mix(in srgb, var(--vs-blue) 14%, transparent)"
        if dark
        else "0 8px 20px rgba(23,32,38,.06)"
    )
    glow_filter = "drop-shadow(0 0 8px rgba(176,138,79,.30))" if dark else "none"
    shell_border = "rgba(176,138,79,.26)" if dark else "rgba(176,138,79,.20)"
    shell_head_border = "rgba(176,138,79,.18)" if dark else "rgba(230,222,210,.92)"
    icon_pill_bg = "rgba(255,255,255,0.04)" if dark else "rgba(255,255,255,0.72)"
    metric_sheen_display = "block" if dark else "none"
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;750;850&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;750;850&family=JetBrains+Mono:wght@400;600;700&display=swap");
        @import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css");
        :root {{
            --vs-bg: {bg};
            --vs-panel: {panel};
            --vs-panel-soft: {panel_soft};
            --vs-ink: {ink};
            --vs-muted: {muted};
            --vs-line: {line};
            --vs-sidebar: {sidebar};
            --vs-blue: {vs_blue};
            --vs-purple: {vs_purple};
            --vs-green: {vs_green};
            --vs-code-bg: {code_bg};
            --vs-stage-bg: {stage_bg};
            --vs-stage-bg-strong: {stage_bg_strong};
            --vs-stage-card-bg: {stage_card_bg};
            --vs-code-chip-bg: {code_chip_bg};
            --vs-track-bg: {track_bg};
            --vs-soft-line: {soft_line};
            --vs-cell-ink: {cell_ink};
            --vs-stage-shadow: {stage_shadow};
            --vs-glow-filter: {glow_filter};
        }}
        html {{ scroll-behavior: smooth; }}
        body, .stApp, [data-testid="stAppViewContainer"] {{
            font-family: Inter, "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
        }}
        code, pre, .stCode, .stCode *, textarea, input {{
            font-family: "JetBrains Mono", Consolas, "Microsoft YaHei UI", monospace !important;
        }}
        :not(pre) > code {{
            background: var(--vs-code-bg);
            color: var(--vs-ink);
            border: 1px solid var(--vs-soft-line);
            border-radius: 5px;
            padding: .08rem .28rem;
        }}
        pre, .stCode {{
            background: var(--vs-code-bg) !important;
            color: var(--vs-ink) !important;
            border: 1px solid var(--vs-soft-line);
            border-radius: 8px;
            max-width: 100% !important;
            min-width: 0 !important;
            max-height: min(70vh, 680px);
            overflow: auto !important;
            resize: vertical;
            white-space: pre !important;
        }}
        .stCode pre, .stCode code, pre code {{
            white-space: pre !important;
            overflow-wrap: normal !important;
            word-break: normal !important;
        }}
        .stCode code {{
            display: inline-block;
            min-width: max-content;
        }}
        .stApp {{
            background: {app_background};
            color: var(--vs-ink);
            animation: none;
        }}
        .block-container {{ animation: none; }}
        @keyframes vs-page-in {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        @keyframes vs-slide-in {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        section[data-testid="stSidebar"] {{
            background: {sidebar_background};
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
            box-shadow: {card_shadow};
            backdrop-filter: blur(8px);
        }}
        .vs-card, .vs-card *, .vs-concept-shell, .vs-motion-grid, .vs-motion-grid-item {{
            box-sizing: border-box;
            min-width: 0;
        }}
        .vs-card {{
            max-width: 100%;
            overflow: hidden;
        }}
        .stButton > button, .stDownloadButton > button, .stLinkButton > a {{
            border-radius: 8px !important;
            border: 1px solid {button_border} !important;
            background: {button_background} !important;
            color: var(--vs-ink) !important;
            box-shadow: {button_shadow};
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {{
            transform: translateY(-1px);
            border-color: var(--vs-green) !important;
            box-shadow: {button_hover_shadow};
        }}
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, .stLinkButton > a:focus-visible,
        a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {{
            outline: 2px solid var(--vs-green) !important;
            outline-offset: 3px !important;
            box-shadow: 0 0 0 4px rgba(176,138,79,0.16), 0 0 24px rgba(176,138,79,0.22) !important;
        }}
        .tag, .demo-badge {{
            border-color: color-mix(in srgb, var(--vs-blue) 42%, transparent) !important;
            background: color-mix(in srgb, var(--vs-blue) 10%, transparent) !important;
            color: var(--vs-blue) !important;
            box-shadow: none;
        }}
        .fa, .fa-solid, .fa-regular, .fa-brands {{
            color: var(--vs-blue);
            filter: {icon_filter};
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
            border-bottom: 1px dashed color-mix(in srgb, var(--vs-blue) 45%, transparent);
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
            background: {tooltip_bg};
            border: 1px solid color-mix(in srgb, var(--vs-blue) 38%, transparent);
            color: {tooltip_color};
            box-shadow: {tooltip_shadow};
            font-size: 0.86rem;
            line-height: 1.5;
        }}
        .vs-motion-note, .vs-beginner-hint, .vs-chart-note {{
            border: 1px solid color-mix(in srgb, var(--vs-blue) 24%, transparent);
            background: linear-gradient(135deg, color-mix(in srgb, var(--vs-blue) 7%, transparent), color-mix(in srgb, var(--vs-green) 5%, transparent));
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
            box-shadow: var(--vs-stage-shadow), 0 0 18px color-mix(in srgb, var(--metric-color) 10%, transparent);
        }}
        .vs-neon-metric::before {{
            content: "";
            display: {metric_sheen_display};
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
            border: 1px solid {shell_border};
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
            border-bottom: 1px solid {shell_head_border};
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
        .vs-motion-grid-item {{
            min-width: 0;
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
            background: var(--vs-code-chip-bg);
            border-radius: 5px;
            padding: .08rem .32rem;
            font-size: .78rem;
        }}
        .vs-shape-arrow {{
            color: var(--vs-blue);
            filter: var(--vs-glow-filter);
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
            border: 1px solid color-mix(in srgb, var(--vs-blue) 25%, transparent);
            background: {icon_pill_bg};
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
            background: var(--vs-track-bg);
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
            .vs-motion-grid {{ grid-template-columns: 1fr !important; }}
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
