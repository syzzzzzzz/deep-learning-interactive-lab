"""Shared Streamlit error boundary helpers for lesson modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def _return_home_button() -> None:
    import streamlit as st

    if st.button("返回主界面"):
        st.query_params.clear()
        st.rerun()


def render_module_error(module_name: str, exception: Exception) -> None:
    """Render a Chinese module error page with troubleshooting guidance."""
    import streamlit as st

    st.error(f"{module_name} 执行出错")
    st.error(f"错误摘要：{type(exception).__name__}: {exception}")
    st.info(
        "可能原因：缺少依赖、运行环境不完整、数据或参数不符合预期，"
        "也可能是老脚本在当前 Streamlit/Matplotlib 环境下不兼容。"
    )
    st.warning(
        "排查建议：请先返回主界面重新进入；如果问题仍然存在，"
        "检查终端日志、依赖安装状态和本模块最近的代码改动。"
    )
    _return_home_button()


def safe_execute(func: Callable[..., T], *args: object, **kwargs: object) -> T | None:
    """Execute a module function safely and render an error page on failure."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        module_name = getattr(func, "__module__", "当前模块")
        render_module_error(module_name, exc)
        return None
