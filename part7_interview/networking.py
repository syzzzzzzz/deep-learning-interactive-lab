"""
CS interview page: computer networking.

Run:
    streamlit run part7_interview/networking.py
or:
    python main.py part7/networking
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pandas as pd
import streamlit as st

from components.visual_system import render_visual_system


T = TypeVar("T")

st.set_page_config(page_title="计算机网络面试训练", layout="wide", initial_sidebar_state="expanded")


def css() -> str:
    return """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1180px; }
    .stApp { background: #f7f8f4; color: #172026; }
    h1, h2, h3, p, li, label, span { letter-spacing: 0; }
    .hero { border-bottom: 1px solid #d8dee3; padding-bottom: 1rem; margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: clamp(2rem, 3vw, 3.1rem); line-height: 1.1; }
    .hero p { color: #596772; line-height: 1.7; max-width: 920px; }
    .note { border-left: 4px solid #0f8b8d; background: rgba(255,255,255,.78); border-radius: 0 8px 8px 0; padding: .74rem .9rem; line-height: 1.68; margin: .4rem 0 .9rem; }
    .step { background: rgba(255,255,255,.82); border: 1px solid #d8dee3; border-radius: 8px; padding: .72rem .82rem; min-height: 104px; line-height: 1.55; }
    .arrow { text-align: center; font-weight: 800; color: #0f8b8d; padding-top: 2.25rem; }
    .flow { background: #172026; color: #f7fbfc; border-radius: 8px; padding: .82rem 1rem; font-family: Consolas, "Courier New", monospace; line-height: 1.72; white-space: pre-wrap; }
    .small { color: #596772; font-size: .92rem; line-height: 1.58; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    /* TCP 三次握手动态箭头 */
    .tcp-wrap { display: flex; align-items: flex-start; justify-content: center; gap: 0; margin: 1.2rem 0; position: relative; }
    .tcp-side { width: 120px; text-align: center; }
    .tcp-side .tcp-box { background: rgba(255,255,255,.82); border: 2px solid #0f8b8d; border-radius: 10px; padding: .7rem .5rem; font-weight: 700; }
    .tcp-arrows { flex: 1; max-width: 380px; position: relative; min-height: 220px; }
    .tcp-arrow-row { display: flex; align-items: center; justify-content: center; margin: 18px 0; position: relative; }
    .tcp-arrow-line { flex: 1; height: 3px; position: relative; margin: 0 6px; }
    .tcp-arrow-line::after { content: ''; position: absolute; right: -2px; top: -5px; border: solid #0f8b8d; border-width: 0 3px 3px 0; display: inline-block; padding: 5px; transform: rotate(-45deg); }
    .tcp-arrow-label { background: #0f8b8d; color: #fff; font-size: .82rem; font-weight: 700; padding: 3px 12px; border-radius: 12px; white-space: nowrap; }
    .tcp-arrow-rev .tcp-arrow-line::after { right: auto; left: -2px; transform: rotate(135deg); }
    @keyframes tcp-slide { 0% { transform: translateX(-100%); opacity: 0; } 100% { transform: translateX(0); opacity: 1; } }
    @keyframes tcp-slide-rev { 0% { transform: translateX(100%); opacity: 0; } 100% { transform: translateX(0); opacity: 1; } }
    .tcp-s1 { animation: tcp-slide .5s ease both; }
    .tcp-s2 { animation: tcp-slide-rev .5s ease .6s both; }
    .tcp-s3 { animation: tcp-slide .5s ease 1.2s both; }
    .tcp-glow { animation: tcp-glow-pulse 2s ease-in-out infinite alternate; }
    @keyframes tcp-glow-pulse { 0% { box-shadow: 0 0 6px rgba(15,139,141,0.3); } 100% { box-shadow: 0 0 18px rgba(15,139,141,0.7); } }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("计算机网络页面执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="networking-back-home", use_container_width=True):
        st.query_params.clear()
        st.rerun()


def handshake_diagram() -> None:
    st.subheader("TCP 三次握手")
    st.markdown(
        '''
        <div class="tcp-wrap">
          <div class="tcp-side">
            <div class="tcp-box tcp-glow">🖥️ 客户端</div>
          </div>
          <div class="tcp-arrows">
            <div class="tcp-arrow-row tcp-s1">
              <div class="tcp-arrow-label">SYN=1, seq=x</div>
              <div class="tcp-arrow-line" style="background:linear-gradient(90deg,#0f8b8d,#00f0ff);"></div>
            </div>
            <div class="tcp-arrow-row tcp-arrow-rev tcp-s2">
              <div class="tcp-arrow-line" style="background:linear-gradient(270deg,#0f8b8d,#00f0ff);"></div>
              <div class="tcp-arrow-label">SYN=1 ACK=1, seq=y ack=x+1</div>
            </div>
            <div class="tcp-arrow-row tcp-s3">
              <div class="tcp-arrow-label">ACK=1, ack=y+1</div>
              <div class="tcp-arrow-line" style="background:linear-gradient(90deg,#0f8b8d,#00ff88);"></div>
            </div>
            <div style="text-align:center;margin-top:6px;animation:tcp-slide .5s ease 1.8s both;">
              <span style="background:#00ff88;color:#172026;font-weight:800;padding:4px 16px;border-radius:12px;">✅ ESTABLISHED</span>
            </div>
          </div>
          <div class="tcp-side">
            <div class="tcp-box tcp-glow">🗄️ 服务端</div>
          </div>
        </div>
        <div style="text-align:center;color:#596772;font-size:.88rem;margin-top:.4rem;">逐帧动画：SYN → SYN-ACK → ACK，三次交互确认双向通信能力</div>
        ''',
        unsafe_allow_html=True,
    )


def wave_diagram() -> None:
    st.subheader("TCP 四次挥手")
    rows = [
        ("客户端 → 服务端", "FIN=1, seq=u", "客户端不再发送数据，进入 FIN_WAIT_1。"),
        ("服务端 → 客户端", "ACK=1, ack=u+1", "服务端确认关闭请求，但可能还有数据要发。"),
        ("服务端 → 客户端", "FIN=1, seq=v", "服务端数据发送完毕，请求关闭反向通道。"),
        ("客户端 → 服务端", "ACK=1, ack=v+1", "客户端进入 TIME_WAIT，等待旧报文自然消失。"),
    ]
    cols = st.columns(4)
    for col, (title, packet, detail) in zip(cols, rows):
        with col:
            st.markdown(f'<div class="step"><strong>{title}</strong><br>{packet}<br><span class="small">{detail}</span></div>', unsafe_allow_html=True)


def protocol_interaction() -> None:
    st.subheader("交互：协议场景与 HTTP/HTTPS 差异")
    left, right = st.columns([0.45, 0.55])
    with left:
        scenario = st.selectbox(
            "选择协议场景",
            ["网页浏览", "登录支付", "模型推理 API", "文件下载", "内网健康检查"],
            key="networking-scenario",
        )
        secure = st.toggle("切换为 HTTPS", value=True, key="networking-https-toggle")
    with right:
        if secure:
            st.success(f"{scenario}：HTTPS 在 HTTP 语义外包了一层 TLS，提供身份认证、加密传输和完整性校验。")
            st.code("ClientHello -> ServerHello/Certificate -> Key Exchange -> Encrypted HTTP", language="text")
        else:
            st.warning(f"{scenario}：HTTP 明文传输，适合低风险内网或本地调试；公网登录、支付、推理请求不应明文。")
            st.code("TCP connection -> HTTP request -> HTTP response", language="text")


def main() -> None:
    render_visual_system("dark")
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>计算机网络面试训练</h1>
          <p>把 TCP、HTTP/HTTPS、DNS 和模型部署接口串起来：面试不只背流程，还要能说明每一步解决什么工程问题。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    protocol_interaction()
    handshake_diagram()
    wave_diagram()

    st.subheader("HTTP 与 HTTPS 对比")
    st.table(
        pd.DataFrame(
            [
                ["传输内容", "明文 HTTP 报文", "HTTP 报文经过 TLS 加密"],
                ["默认端口", "80", "443"],
                ["身份认证", "无内建认证", "通过证书链验证服务端身份"],
                ["完整性", "容易被篡改且难发现", "TLS MAC/AEAD 能发现篡改"],
                ["典型场景", "本地调试、可信内网", "登录、支付、API、模型服务公网调用"],
            ],
            columns=["维度", "HTTP", "HTTPS"],
        )
    )

    st.subheader("DNS 解析流程")
    st.markdown(
        """
        <div class="flow">1. 浏览器检查自身 DNS 缓存
2. 操作系统检查 hosts 和本地缓存
3. 向本地递归 DNS 服务器查询
4. 递归服务器依次询问根域名服务器、顶级域名服务器、权威域名服务器
5. 返回域名对应 IP，按 TTL 缓存
6. 浏览器拿到 IP 后发起 TCP/TLS/HTTP 请求</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("高频问答区")
    with st.expander("为什么 TCP 需要三次握手？"):
        st.write("核心是同时确认双方的发送能力和接收能力，并同步初始序列号。两次握手只能让服务端确认客户端到服务端方向可达，无法确认服务端到客户端方向的确认能被客户端收到，也更容易受历史连接请求干扰。")
    with st.expander("HTTPS 为什么安全？"):
        st.write("HTTPS 依赖 TLS：证书链解决“我连的是谁”，密钥协商解决“怎么安全地生成会话密钥”，对称加密保护内容，完整性校验防止中间人篡改。")
    with st.expander("浏览器输入 URL 后发生了什么？"):
        st.write("先解析 URL 和缓存，再 DNS 查询得到 IP，建立 TCP 连接；如果是 HTTPS 还要做 TLS 握手；随后发送 HTTP 请求，服务端返回响应，浏览器解析 HTML、CSS、JS，构建 DOM/CSSOM、布局、绘制，并继续请求子资源。")

    st.subheader("与深度学习的连接")
    st.markdown(
        """
        <div class="note">
        模型部署 API 服务本质上也是网络服务：客户端把特征、图片或文本通过 HTTP/gRPC 发给推理服务，
        服务端在负载均衡、TLS、超时、重试和限流约束下完成推理。面试里可以把“网络通信”落到真实问题：
        推理接口延迟、请求体大小、连接复用、HTTPS 证书、DNS 故障和跨区域调用。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("进入网络专项刷题", "/?module=part7%2Finterview_quiz", width="stretch")
    render_back_home()


if __name__ == "__main__":
    safe_run(main)
