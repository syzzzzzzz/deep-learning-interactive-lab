"""可复用 Streamlit 教学组件。

示例：
if __name__ == "__main__":
    import streamlit as st
    render导读卡片(
        "张量与梯度",
        "用一个最小例子理解张量、自动求导和反向传播。",
        "入门",
        ["Python 基础", "线性代数初步"],
        ["张量", "梯度", "自动求导"],
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape


def _st():
    import streamlit as st

    return st


def _safe_list(values: Sequence[str] | None) -> list[str]:
    return [str(value) for value in values or [] if str(value).strip()]


def _show_widget_error(name: str, error: BaseException) -> None:
    st = _st()
    st.warning(f"{name} 暂时无法显示，请继续阅读正文。")
    with st.expander("查看组件错误详情", expanded=False):
        st.code(str(error), language="text")


def render_back_to_home() -> None:
    """渲染返回主界面的按钮，并清空地址栏查询参数。"""
    try:
        st = _st()
        if st.button("返回主界面", key="teaching-back-home", width="stretch"):
            st.query_params.clear()
            st.rerun()
    except Exception as error:
        _show_widget_error("返回主界面按钮", error)


def render导读卡片(
    title: str,
    description: str,
    difficulty: str,
    prerequisites: Sequence[str] | None,
    tags: Sequence[str] | None,
) -> None:
    """渲染页面顶部导读信息。"""
    try:
        st = _st()
        prerequisite_text = "、".join(_safe_list(prerequisites)) or "无需额外前置知识"
        tag_html = "".join(f'<span class="teaching-tag">{escape(tag)}</span>' for tag in _safe_list(tags))
        st.markdown(
            f"""
            <div class="teaching-guide-card" style="border:1px solid #d9e2ec;border-radius:8px;padding:18px 20px;margin:8px 0 18px;background:#f8fafc;">
              <div style="font-size:13px;color:#52606d;margin-bottom:6px;">学习导读 · {escape(str(difficulty))}</div>
              <h2 style="margin:0 0 8px 0;">{escape(str(title))}</h2>
              <p style="margin:0 0 12px 0;color:#334e68;">{escape(str(description))}</p>
              <p style="margin:0 0 10px 0;"><strong>前置知识：</strong>{escape(prerequisite_text)}</p>
              <div style="display:flex;gap:8px;flex-wrap:wrap;">{tag_html}</div>
            </div>
            <style>
            .teaching-guide-card .teaching-tag {{
                display:inline-block;padding:3px 8px;border:1px solid #bcccdc;border-radius:999px;
                background:#ffffff;color:#334e68;font-size:12px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as error:
        _show_widget_error("导读卡片", error)


def render_error_card(title: str, error_msg: str, suggestions: Sequence[str] | None) -> None:
    """渲染中文错误摘要卡片，错误详情默认折叠。"""
    try:
        st = _st()
        st.error(str(title))
        suggestion_items = _safe_list(suggestions)
        if suggestion_items:
            st.markdown("**建议排查：**")
            for item in suggestion_items:
                st.markdown(f"- {item}")
        with st.expander("查看错误详情", expanded=False):
            st.code(str(error_msg), language="text")
    except Exception as error:
        _show_widget_error("错误摘要卡片", error)


def render知识点推荐(related_topics: Sequence[Mapping[str, str] | str] | None) -> None:
    """渲染相关知识点推荐块。"""
    try:
        st = _st()
        topics = list(related_topics or [])
        if not topics:
            st.info("暂时没有相关知识点推荐。")
            return
        st.subheader("相关知识点")
        for index, topic in enumerate(topics, 1):
            if isinstance(topic, Mapping):
                title = str(topic.get("title") or topic.get("name") or f"知识点 {index}")
                description = str(topic.get("description") or "建议作为延伸阅读。")
            else:
                title = str(topic)
                description = "建议作为延伸阅读。"
            st.markdown(f"**{index}. {title}**")
            st.caption(description)
    except Exception as error:
        _show_widget_error("相关知识点推荐", error)


def render公式块(latex_str: str, explanation: str = "") -> None:
    """渲染数学公式块。"""
    try:
        st = _st()
        st.markdown("**公式理解**")
        st.latex(str(latex_str))
        if explanation:
            st.info(str(explanation))
    except Exception as error:
        _show_widget_error("数学公式块", error)


def render参数实验表(params_dict: Mapping[str, object] | None) -> None:
    """渲染参数实验表。"""
    try:
        st = _st()
        rows = [{"参数": str(key), "取值": str(value)} for key, value in (params_dict or {}).items()]
        if not rows:
            st.info("暂无参数实验记录。")
            return
        st.markdown("**参数实验表**")
        st.table(rows)
    except Exception as error:
        _show_widget_error("参数实验表", error)


def render误区卡片(misconception: str, correct_explanation: str) -> None:
    """渲染常见误区卡片。"""
    try:
        st = _st()
        st.warning(f"常见误区：{misconception}")
        st.success(f"正确理解：{correct_explanation}")
    except Exception as error:
        _show_widget_error("常见误区卡片", error)


def render工程经验卡片(tip_title: str, tip_content: str) -> None:
    """渲染工程经验卡片。"""
    try:
        st = _st()
        st.info(f"工程经验：{tip_title}\n\n{tip_content}")
    except Exception as error:
        _show_widget_error("工程经验卡片", error)


def render_source_code(code: str, language: str = "python") -> None:
    """渲染源码展示块。"""
    try:
        st = _st()
        st.markdown("**源码展示**")
        st.code(str(code), language=str(language or "text"))
    except Exception as error:
        _show_widget_error("源码展示块", error)


def render_console_output(output_text: str) -> None:
    """渲染控制台输出块。"""
    try:
        st = _st()
        st.markdown("**控制台输出**")
        st.code(str(output_text), language="text")
    except Exception as error:
        _show_widget_error("控制台输出块", error)


if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="教学组件示例", layout="wide")
    render导读卡片("教学组件示例", "这是组件库的最小演示页面。", "入门", ["Streamlit 基础"], ["组件", "教学"])
    render公式块(r"y = wx + b", "线性层把输入按权重加权后再加偏置。")
    render参数实验表({"学习率": 0.001, "批大小": 32, "优化器": "Adam"})
    render误区卡片("损失下降就一定泛化更好", "还需要同时观察验证集指标和过拟合迹象。")
    render工程经验卡片("先跑通最小闭环", "先用小数据确认训练、评估和保存流程都可执行，再扩大实验规模。")
