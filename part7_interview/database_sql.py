"""
CS interview page: database and SQL.

Run:
    streamlit run part7_interview/database_sql.py
or:
    python main.py part7/database_sql
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pandas as pd
import streamlit as st

from components.visual_system import render_visual_system


T = TypeVar("T")

st.set_page_config(page_title="数据库 SQL 面试训练", layout="wide", initial_sidebar_state="auto")


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
    .tree, .flow { background: #172026; color: #f7fbfc; border-radius: 8px; padding: .82rem 1rem; font-family: Consolas, "Courier New", monospace; line-height: 1.72; white-space: pre-wrap; }
    .step { background: rgba(255,255,255,.82); border: 1px solid #d8dee3; border-radius: 8px; padding: .72rem .82rem; min-height: 92px; line-height: 1.55; }
    .small { color: #596772; font-size: .92rem; line-height: 1.58; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    /* B+树查询路径高亮 */
    .bptree-container { display: flex; flex-direction: column; align-items: center; margin: 1rem 0; font-family: Consolas, "Courier New", monospace; }
    .bptree-level { display: flex; gap: 12px; justify-content: center; margin: 6px 0; }
    .bptree-node { background: rgba(255,255,255,.82); border: 2px solid #d8dee3; border-radius: 8px; padding: 6px 14px; font-weight: 700; font-size: .92rem; transition: all .3s; min-width: 60px; text-align: center; }
    .bptree-connector { display: flex; justify-content: center; gap: 12px; margin: 0; height: 20px; }
    .bptree-line { width: 2px; height: 20px; background: #d8dee3; }
    @keyframes bptree-highlight { 0% { background: rgba(255,255,255,.82); border-color: #d8dee3; box-shadow: none; transform: scale(1); } 100% { background: #0f8b8d; color: #fff; border-color: #0f8b8d; box-shadow: 0 0 16px rgba(15,139,141,0.5); transform: scale(1.08); } }
    .bptree-active { animation: bptree-highlight .5s ease forwards; }
    .bptree-found { background: #00ff88 !important; color: #172026 !important; border-color: #00ff88 !important; box-shadow: 0 0 14px rgba(0,255,136,0.5) !important; }
    .bptree-connector-active .bptree-line { background: #0f8b8d; box-shadow: 0 0 8px rgba(15,139,141,0.4); }
    .bptree-leaf-link { display: flex; gap: 0; justify-content: center; margin-top: 4px; }
    .bptree-leaf-link .bptree-node { border-radius: 0; min-width: 50px; font-size: .84rem; padding: 4px 10px; }
    .bptree-leaf-link .bptree-node:first-child { border-radius: 8px 0 0 8px; }
    .bptree-leaf-link .bptree-node:last-child { border-radius: 0 8px 8px 0; }
    @keyframes bptree-scan { 0% { background: rgba(255,255,255,.82); color: #172026; } 50% { background: #0f8b8d; color: #fff; } 100% { background: rgba(255,255,255,.82); color: #172026; } }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("数据库 SQL 页面执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="database-back-home", width="stretch"):
        st.query_params.clear()
        st.rerun()


def estimate_cost(sql: str, use_index: bool) -> pd.DataFrame:
    rows = 1_000_000
    selectivity = 0.01 if "where" in sql.lower() else 1.0
    full_pages = rows // 100
    index_height = 3
    matched_rows = max(1, int(rows * selectivity))
    index_pages = index_height + matched_rows // 120
    if not use_index:
        index_pages = full_pages
    return pd.DataFrame(
        [
            ["全表扫描", full_pages, rows, "逐页读取整张表，适合小表或返回大部分数据"],
            ["索引扫描" if use_index else "无可用索引", index_pages, matched_rows if use_index else rows, "先走 B+ 树定位，再回表或覆盖索引返回结果"],
        ],
        columns=["访问方式", "估算读取页数", "估算检查行数", "说明"],
    )


def main() -> None:
    render_visual_system("light")
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>数据库 SQL 面试训练</h1>
          <p>用 SELECT 执行流程、B+ 树索引和慢查询排查，把“会写 SQL”提升到“能解释数据库为什么这么跑”。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("交互：SQL 与扫描成本")
    left, right = st.columns([0.42, 0.58])
    with left:
        sql = st.text_input("输入简单 SQL", "SELECT * FROM experiments WHERE run_id = 42", key="db-sql")
        use_index = st.checkbox("使用 run_id 索引", value=True, key="db-use-index")
        st.caption("示例按 100 万行表粗略估算，重点看数量级差异。")
    with right:
        st.table(estimate_cost(sql, use_index))

    st.subheader("SELECT 查询执行流程")
    cols = st.columns(5)
    steps = [
        ("1. 解析", "词法/语法分析，检查 SQL 是否合法。"),
        ("2. 预处理", "解析表、列、权限和视图。"),
        ("3. 优化", "选择索引、连接顺序和访问路径。"),
        ("4. 执行", "按执行计划访问存储引擎。"),
        ("5. 返回", "过滤、排序、聚合后返回结果集。"),
    ]
    for col, (title, body) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step"><strong>{title}</strong><br><span class="small">{body}</span></div>', unsafe_allow_html=True)

    st.subheader("B+ 树索引直观图")
    search_val = st.selectbox("模拟查询 key", [5, 18, 35, 48, 70, 90], key="db-bptree-search")
    # 定义查询路径: root -> internal -> leaf
    paths = {
        5:  {"root": 0, "internal": 0, "leaf": 0},
        18: {"root": 0, "internal": 0, "leaf": 1},
        35: {"root": 0, "internal": 1, "leaf": 0},
        48: {"root": 0, "internal": 1, "leaf": 1},
        70: {"root": 0, "internal": 2, "leaf": 0},
        90: {"root": 0, "internal": 2, "leaf": 1},
    }
    p = paths[search_val]
    root_cls = "bptree-node bptree-active"
    inodes = ["bptree-node"] * 3
    inodes[p["internal"]] = "bptree-node bptree-active"
    leaves = [["bptree-node"] * 2 for _ in range(3)]
    leaves[p["internal"]][p["leaf"]] = "bptree-node bptree-found"
    conn_cls = ["bptree-connector"] * 3
    conn_cls[p["internal"]] = "bptree-connector bptree-connector-active"
    st.markdown(f'''
    <div class="bptree-container" style="animation-delay:0s;">
      <div class="bptree-level">
        <div class="{root_cls}" style="animation-delay:.1s;">30 | 60</div>
      </div>
      <div class="bptree-connector {conn_cls[0]} {conn_cls[1]} {conn_cls[2]}">
        <div class="bptree-line" style="margin:0 42px;"></div>
        <div class="bptree-line" style="margin:0 42px;"></div>
        <div class="bptree-line" style="margin:0 42px;"></div>
      </div>
      <div class="bptree-level">
        <div class="{inodes[0]}" style="animation-delay:.4s;">5 | 18</div>
        <div class="{inodes[1]}" style="animation-delay:.4s;">35 | 48</div>
        <div class="{inodes[2]}" style="animation-delay:.4s;">70 | 90</div>
      </div>
      <div class="bptree-connector">
        <div class="bptree-line" style="margin:0 18px;"></div><div class="bptree-line" style="margin:0 18px;"></div>
        <div class="bptree-line" style="margin:0 18px;"></div><div class="bptree-line" style="margin:0 18px;"></div>
        <div class="bptree-line" style="margin:0 18px;"></div><div class="bptree-line" style="margin:0 18px;"></div>
      </div>
      <div class="bptree-leaf-link">
        <div class="{leaves[0][0]}">5</div>
        <div class="{leaves[0][1]}">18</div>
        <div class="{leaves[1][0]}">35</div>
        <div class="{leaves[1][1]}">48</div>
        <div class="{leaves[2][0]}">70</div>
        <div class="{leaves[2][1]}">90</div>
      </div>
    </div>
    <div style="text-align:center;color:#596772;font-size:.88rem;">绿框 = 查询 key={search_val} 的最终定位，高亮路径 = 从根到叶的查找过程</div>
    ''', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tree">                 [ 30 | 60 ]
                  /    |    \\
        [  5 | 18 ]  [ 35 | 48 ]  [ 70 | 90 ]
          |    |        |    |        |    |
        rows rows     rows rows     rows rows

叶子节点按 key 有序串联：5 -> 18 -> 35 -> 48 -> 70 -> 90
范围查询可以先定位起点，再沿叶子链表顺序扫描。</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("索引为什么能加速查询")
    st.markdown(
        """
        <div class="note">
        全表扫描像从头翻完整本书；B+ 树索引像目录。索引把查找从 O(n) 的逐行比较，变成接近 O(log n) 的树高定位。
        B+ 树分叉大、树高低，数据库通常只需少量页读取就能定位到目标叶子页；如果查询字段都在索引里，还能覆盖索引，避免回表。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("高频问答区")
    with st.expander("索引为什么不是越多越好？"):
        st.write("索引会占磁盘和内存；INSERT、UPDATE、DELETE 时还要维护索引页，降低写入速度。索引太多也会增加优化器选择成本，并可能让低选择性索引误导执行计划。")
    with st.expander("聚簇索引和非聚簇索引区别？"):
        st.write("聚簇索引的叶子节点直接存放整行数据，数据物理组织按主键顺序聚集；非聚簇索引叶子节点通常存索引列和主键值，需要再按主键回表取完整行。InnoDB 主键索引就是聚簇索引。")
    with st.expander("SQL 慢查询怎么排查？"):
        st.write("先用慢查询日志定位 SQL，再 EXPLAIN 看访问类型、索引、扫描行数、是否 filesort/temporary；之后检查 WHERE 条件选择性、联合索引最左前缀、回表次数、锁等待、数据量和缓存命中。")

    st.subheader("与深度学习的连接")
    st.markdown(
        """
        <div class="note">
        实验记录数据库保存 run_id、超参数、指标和模型路径；样本元数据表管理标注、来源、切分和质量状态；
        训练日志存储支撑曲线回放、模型对比和回滚。索引设计会直接影响实验检索、数据集筛选和在线特征查询延迟。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("进入数据库专项刷题", "/?module=part7%2Finterview_quiz", width="stretch")
    render_back_home()


if __name__ == "__main__":
    safe_run(main)
