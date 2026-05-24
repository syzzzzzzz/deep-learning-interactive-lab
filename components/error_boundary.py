"""Shared Streamlit error boundary helpers for lesson modules.

Provides Chinese-language error pages with:
- Smart diagnosis of common failure modes (missing packages, CUDA, etc.)
- Collapsible full traceback for debugging
- Retry button for transient errors
- Suggested fixes based on exception type
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Common error patterns → human-readable diagnosis
# ---------------------------------------------------------------------------

_ERROR_DIAGNOSIS: list[tuple[str, str, str]] = [
    # (substring in repr/error, title, suggestion)
    (
        "No module named",
        "缺少依赖包",
        "请在终端执行 `pip install -r requirements.txt` 安装缺失的依赖，然后重启页面。",
    ),
    (
        "CUDA",
        "GPU / CUDA 问题",
        "当前环境可能没有可用的 GPU。本项目的教学演示已尽量兼容 CPU 模式，"
        "请检查 PyTorch 安装是否匹配你的 CUDA 版本，或尝试在 CPU 上运行。",
    ),
    (
        "CUDA out of memory",
        "显存不足",
        "请减小 batch size 或输入尺寸，或关闭其他占用显存的程序后重试。",
    ),
    (
        "RuntimeError: Expected all tensors to be on the same device",
        "设备不一致",
        "部分张量在 CPU 而另一部分在 GPU。请检查模型和输入数据是否在同一设备上。",
    ),
    (
        "FileNotFoundError",
        "文件未找到",
        "可能是数据文件缺失或路径不正确。请确认项目目录结构完整，必要时重新下载数据集。",
    ),
    (
        "ConnectionError",
        "网络连接失败",
        "加载在线资源时网络超时。请检查网络连接后重试。",
    ),
    (
        "ImportError",
        "导入失败",
        "某个依赖包版本不兼容或未安装。请执行 `pip install -r requirements.txt` 更新依赖。",
    ),
    (
        "ModuleNotFoundError",
        "模块未找到",
        "项目内部模块缺失。请确认项目目录完整，没有被误删的文件。",
    ),
    (
        "KeyError",
        "键值错误",
        "可能是 Streamlit 会话状态或数据字典中缺少预期的键。尝试返回主界面重新进入。",
    ),
    (
        "AttributeError",
        "属性错误",
        "某个对象缺少预期的方法或属性，可能是版本不兼容或数据类型不符合预期。",
    ),
    (
        "TypeError: unhashable type",
        "不可哈希类型",
        "尝试用不可哈希的对象（如列表、字典）作为字典键或集合元素。",
    ),
    (
        "ValueError: operands could not be broadcast",
        "广播维度不匹配",
        "NumPy / PyTorch 张量形状不兼容。请检查输入数据维度和模型参数。",
    ),
]


def _diagnose_error(exception: Exception) -> tuple[str, str] | None:
    """Match an exception against known patterns; return (title, suggestion)."""
    err_text = f"{type(exception).__name__}: {exception}"
    for pattern, title, suggestion in _ERROR_DIAGNOSIS:
        if pattern.lower() in err_text.lower():
            return title, suggestion
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _return_home_button() -> None:
    import streamlit as st

    if st.button("🏠 返回主界面"):
        st.query_params.clear()
        st.rerun()


def render_module_error(module_name: str, exception: Exception) -> None:
    """Render a Chinese module error page with troubleshooting guidance.

    Features:
    - Smart diagnosis of the root cause
    - Actionable suggestion with specific fix
    - Collapsible full traceback for developers
    - Retry button for transient errors
    """
    import streamlit as st

    # --- Error header ---
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a0a0a 0%, #2a1215 100%);
            border: 1px solid #5c2a2a;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">
                ⚠️ 模块执行出错
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.error(f"**{module_name}** 执行过程中遇到异常")

    # --- Smart diagnosis ---
    diagnosis = _diagnose_error(exception)
    if diagnosis:
        title, suggestion = diagnosis
        st.warning(f"**🔍 诊断：{title}**\n\n{suggestion}")
    else:
        st.info(
            "这可能是运行环境不完整、数据/参数不符合预期，"
            "或老脚本在当前 Streamlit/Matplotlib 版本下不兼容导致的。"
        )

    # --- Quick actions ---
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 重新加载模块", key="_retry_err"):
            st.rerun()

    with col2:
        _return_home_button()

    with col3:
        if st.button("📋 复制错误信息", key="_copy_err"):
            err_text = (
                f"模块: {module_name}\n"
                f"错误: {type(exception).__name__}: {exception}\n"
                f"诊断: {diagnosis[0] if diagnosis else '未知'}"
            )
            st.code(err_text, language="text")
            st.toast("错误信息已显示，可手动复制", icon="📋")

    # --- Collapsible full traceback ---
    with st.expander("🔧 完整错误堆栈（点击展开，用于调试）", expanded=False):
        tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        st.code(tb, language="python", line_numbers=True)

    # --- General troubleshooting ---
    with st.expander("💡 通用排查建议", expanded=False):
        st.markdown(
            """
            1. **返回主界面重新进入** — 很多临时问题可以通过重新加载解决。
            2. **检查终端日志** — 启动 Streamlit 的终端窗口通常有更详细的错误信息。
            3. **确认依赖完整** — 运行 `pip install -r requirements.txt`。
            4. **检查 Python 版本** — 本项目建议 Python 3.10+。
            5. **清除缓存** — 删除项目下的 `__pycache__` 和 `.streamlit` 缓存目录后重试。
            """
        )


def safe_execute(func: Callable[..., T], *args: object, **kwargs: object) -> T | None:
    """Execute a module function safely and render an error page on failure.

    Usage::

        from components.error_boundary import safe_execute
        safe_execute(main)

    If *main* raises, an interactive error page is shown instead of a raw traceback.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        module_name = getattr(func, "__module__", None) or getattr(func, "__qualname__", "当前模块")
        render_module_error(module_name, exc)
        return None
