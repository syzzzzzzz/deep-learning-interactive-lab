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


def render_motion_gallery() -> None:
    st = _st()
    st.subheader("核心教学动效")
    render_loading_bar("页面动效服务于观察：每一种发光都对应一个可解释的学习信号")
    c1, c2 = st.columns(2)
    with c1:
        render_convolution_particle_flow()
        render_attention_light_beams(["I", "love", "deep", "learning", "because", "it", "works"], 3)
    with c2:
        render_gradient_descent_landscape()
        render_training_dashboard_gauges()
    render_backprop_current_flow()
