"""基于 Streamlit session_state 的轻量学习进度。"""

from __future__ import annotations

from collections import Counter


PROGRESS_STATUSES = ("未学习", "已学习", "已掌握", "去实战")
_STATE_KEY = "learning_progress"


def _st():
    import streamlit as st

    return st


def _store() -> dict[str, str]:
    st = _st()
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = {}
    return st.session_state[_STATE_KEY]


def get_progress(module_key: str) -> str:
    try:
        return _store().get(module_key, "未学习")
    except Exception:
        return "未学习"


def set_progress(module_key: str, status: str) -> None:
    try:
        if status not in PROGRESS_STATUSES:
            status = "未学习"
        _store()[module_key] = status
    except Exception as error:
        st = _st()
        st.warning("学习进度暂时无法保存。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")


def render进度标记(module_key: str) -> None:
    """渲染可点击的进度标记按钮。"""
    try:
        st = _st()
        current = get_progress(module_key)
        st.markdown(f"**当前进度：{current}**")
        cols = st.columns(len(PROGRESS_STATUSES))
        for index, status in enumerate(PROGRESS_STATUSES):
            label = f"设为{status}" if status != current else f"当前：{status}"
            if cols[index].button(label, key=f"progress-{module_key}-{status}", width="stretch"):
                set_progress(module_key, status)
                st.rerun()
    except Exception as error:
        st = _st()
        st.warning("进度标记暂时无法显示。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")


def render进度总览() -> None:
    """渲染所有模块的进度概览。"""
    try:
        st = _st()
        store = _store()
        graph = {}
        try:
            from components.knowledge_graph import KNOWLEDGE_GRAPH

            graph = KNOWLEDGE_GRAPH
            module_keys = list(KNOWLEDGE_GRAPH)
        except Exception:
            module_keys = sorted(store)
        rows = [{"模块": graph[key].title if key in graph else key, "进度": store.get(key, "未学习")} for key in module_keys]
        counts = Counter(row["进度"] for row in rows)
        cols = st.columns(len(PROGRESS_STATUSES))
        for index, status in enumerate(PROGRESS_STATUSES):
            cols[index].metric(status, counts.get(status, 0))

        if not module_keys:
            st.info("还没有记录学习进度。打开任意模块后，可以把它标记为已学习、已掌握或去实战。")
            return

        st.dataframe(rows, width="stretch", hide_index=True)
    except Exception as error:
        st = _st()
        st.warning("学习进度总览暂时无法显示。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")
