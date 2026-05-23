"""
CS interview page: data structures and algorithms.

Run:
    streamlit run part7_interview/data_structures.py
or:
    python main.py part7/data_structures
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TypeVar

import streamlit as st


T = TypeVar("T")

st.set_page_config(page_title="数据结构与算法面试训练", layout="wide", initial_sidebar_state="expanded")


def css() -> str:
    return """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1180px; }
    .stApp { background: #f7f8f4; color: #172026; }
    h1, h2, h3, p, li, label, span { letter-spacing: 0; }
    .hero { border-bottom: 1px solid #d8dee3; padding-bottom: 1rem; margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: clamp(2rem, 3vw, 3.1rem); }
    .hero p { color: #596772; line-height: 1.7; max-width: 920px; }
    .note { border-left: 4px solid #0f8b8d; background: rgba(255,255,255,.78); border-radius: 0 8px 8px 0; padding: .74rem .9rem; line-height: 1.68; margin: .4rem 0 .9rem; }
    .viz, .codebox { background: #172026; color: #f7fbfc; border-radius: 8px; padding: .82rem 1rem; font-family: Consolas, "Courier New", monospace; line-height: 1.72; white-space: pre-wrap; }
    .mini { background: rgba(255,255,255,.82); border: 1px solid #d8dee3; border-radius: 8px; padding: .75rem .85rem; min-height: 116px; line-height: 1.6; }
    .bar { display: inline-block; background: #0f8b8d; color: #fff; border-radius: 6px 6px 0 0; margin-right: 6px; width: 42px; text-align: center; vertical-align: bottom; font-weight: 800; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("数据结构与算法页面执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="ds-back-home", use_container_width=True):
        st.query_params.clear()
        st.rerun()


def reset_sort_state(kind: str) -> None:
    st.session_state["ds_sort_kind"] = kind
    st.session_state["ds_array"] = [6, 3, 8, 2, 5, 1]
    st.session_state["ds_i"] = 0
    st.session_state["ds_j"] = 0
    st.session_state["ds_stack"] = [(0, 5)]
    st.session_state["ds_done"] = False
    st.session_state["ds_message"] = "准备开始"


def ensure_sort_state(kind: str) -> None:
    if st.session_state.get("ds_sort_kind") != kind or "ds_array" not in st.session_state:
        reset_sort_state(kind)


def bubble_step() -> None:
    arr = st.session_state["ds_array"]
    n = len(arr)
    i = st.session_state["ds_i"]
    j = st.session_state["ds_j"]
    if i >= n - 1:
        st.session_state["ds_done"] = True
        st.session_state["ds_message"] = "冒泡排序完成"
        return
    if arr[j] > arr[j + 1]:
        arr[j], arr[j + 1] = arr[j + 1], arr[j]
        st.session_state["ds_message"] = f"比较 a[{j}] 和 a[{j + 1}]，发生交换"
    else:
        st.session_state["ds_message"] = f"比较 a[{j}] 和 a[{j + 1}]，无需交换"
    j += 1
    if j >= n - 1 - i:
        i += 1
        j = 0
    st.session_state["ds_i"] = i
    st.session_state["ds_j"] = j


def quick_step() -> None:
    arr = st.session_state["ds_array"]
    stack = st.session_state["ds_stack"]
    if not stack:
        st.session_state["ds_done"] = True
        st.session_state["ds_message"] = "快速排序完成"
        return
    lo, hi = stack.pop()
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    if lo < i - 1:
        stack.append((lo, i - 1))
    if i + 1 < hi:
        stack.append((i + 1, hi))
    st.session_state["ds_message"] = f"以 {pivot} 为枢轴，放到下标 {i}；左右子区间入栈"


def render_array_bars(values: list[int]) -> None:
    html = "".join(
        f'<span class="bar" style="height:{value * 22}px; padding-top:4px">{value}</span>'
        for value in values
    )
    st.markdown(f"<div style='height:190px; display:flex; align-items:flex-end;'>{html}</div>", unsafe_allow_html=True)


def graph_order(kind: str) -> list[str]:
    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": ["F"],
        "F": [],
    }
    if kind == "BFS":
        seen = {"A"}
        q: deque[str] = deque(["A"])
        order = []
        while q:
            node = q.popleft()
            order.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return order
    seen = set()
    order = []

    def dfs(node: str) -> None:
        seen.add(node)
        order.append(node)
        for nxt in graph[node]:
            if nxt not in seen:
                dfs(nxt)

    dfs("A")
    return order


def main() -> None:
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>数据结构与算法面试训练</h1>
          <p>用可视化把数组、链表、树、排序和图搜索连起来：面试回答要能同时讲结构、复杂度和工程适用场景。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("基础数据结构可视化")
    cols = st.columns(5)
    visuals = [
        ("数组", "[0] 4 | [1] 8 | [2] 15 | [3] 16", "连续内存，按下标 O(1) 访问。"),
        ("链表", "head -> 4 -> 8 -> 15 -> None", "插入删除灵活，随机访问 O(n)。"),
        ("栈", "top\n[ 9 ]\n[ 5 ]\n[ 2 ]", "后进先出，常见于调用栈和括号匹配。"),
        ("队列", "front <- 2 <- 5 <- 9 <- rear", "先进先出，常见于 BFS 和任务队列。"),
        ("树", "    A\n   / \\\n  B   C\n / \\   \\\nD   E   F", "层级结构，搜索、索引和表达式解析常用。"),
    ]
    for col, (name, art, desc) in zip(cols, visuals):
        with col:
            st.markdown(f'<div class="mini"><strong>{name}</strong><br><pre>{art}</pre><span>{desc}</span></div>', unsafe_allow_html=True)

    st.subheader("排序算法动画")
    left, right = st.columns([0.34, 0.66])
    with left:
        algorithm = st.selectbox("选择算法", ["冒泡排序", "快速排序"], key="ds-algorithm")
        ensure_sort_state(algorithm)
        if st.button("单步执行", key="ds-step", use_container_width=True):
            if algorithm == "冒泡排序":
                bubble_step()
            else:
                quick_step()
        if st.button("重置数组", key="ds-reset", use_container_width=True):
            reset_sort_state(algorithm)
        complexity = {
            "冒泡排序": ("O(n^2)", "O(1)"),
            "快速排序": ("平均 O(n log n)，最坏 O(n^2)", "平均 O(log n)，最坏 O(n)"),
        }[algorithm]
        st.metric("时间复杂度", complexity[0])
        st.metric("空间复杂度", complexity[1])
    with right:
        render_array_bars(st.session_state["ds_array"])
        st.info(st.session_state["ds_message"])

    st.subheader("BFS / DFS 可视化")
    search_kind = st.radio("选择图搜索", ["BFS", "DFS"], horizontal=True, key="ds-search")
    st.markdown(
        """
        <div class="viz">      A
     / \\
    B   C
   / \\   \\
  D   E -> F</div>
        """,
        unsafe_allow_html=True,
    )
    st.success(f"{search_kind} 访问顺序：{' -> '.join(graph_order(search_kind))}")

    st.subheader("高频问答区")
    with st.expander("数组和链表的区别？"):
        st.write("数组连续存储，支持 O(1) 随机访问，但中间插入删除通常要搬移元素；链表节点离散存储，插入删除只改指针，但访问第 k 个元素需要从头遍历。")
    with st.expander("哈希表冲突怎么解决？"):
        st.write("常见方法有链地址法和开放寻址法。工程上还要关注负载因子、扩容、哈希函数质量，以及极端冲突下退化为链表或树结构的成本。")
    with st.expander("快排为什么平均 O(n log n)？"):
        st.write("如果枢轴大致把数组分成两半，每层 partition 总成本 O(n)，递归深度约 O(log n)，所以平均 O(n log n)。最坏情况是每次只划掉一个元素，深度退化到 O(n)。")

    st.subheader("与深度学习的连接")
    st.markdown(
        """
        <div class="note">
        张量是高维数组，图计算框架用 DAG 表示算子依赖；Beam Search、采样和检索会用到搜索算法；
        Transformer 注意力的朴素复杂度是 O(n^2)，理解数据结构和复杂度才能判断长上下文优化为什么重要。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_back_home()


if __name__ == "__main__":
    safe_run(main)

