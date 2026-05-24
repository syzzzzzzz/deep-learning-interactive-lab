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
        .vs-tooltip {{
            position: relative;
            border-bottom: 1px dashed rgba(0,240,255,0.45);
            cursor: help;
        }}
        .vs-tooltip:hover::after {{
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
        @media (max-width: 760px) {{
            .vs-particle-field {{ opacity: 0.24; }}
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
        <div class="vs-icon-row">
          <span class="vs-icon-pill"><i class="fa-solid fa-circle-nodes"></i>{escape(label)}</span>
        </div>
        <div class="vs-loading-strip"></div>
        """,
        unsafe_allow_html=True,
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
# 运动画廊（展示所有动效）
# ──────────────────────────────────────────────


def render_motion_gallery() -> None:
    """展示所有教学动效组件的画廊，含新增组件演示。"""
    st = _st()
    st.subheader("核心教学动效")
    render_loading_bar("页面动效服务于观察：每一种发光都对应一个可解释的学习信号")

    # ── 通用组件区 ──
    st.markdown("### 通用 UI 组件")
    c0a, c0b, c0c = st.columns(3)
    with c0a:
        render_metric_card("Accuracy", "98.2%", delta="+0.5%", icon="fa-solid fa-bullseye", accent=NEON_GREEN)
    with c0b:
        render_metric_card("Loss", "0.042", delta="-0.008", icon="fa-solid fa-fire", accent=NEON_PURPLE)
    with c0c:
        render_metric_card("Epoch", "127/200", icon="fa-solid fa-rotate", accent=NEON_BLUE)
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
