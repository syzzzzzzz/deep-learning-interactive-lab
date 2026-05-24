"""
CS interview page: operating systems.

Run:
    streamlit run part7_interview/operating_system.py
or:
    python main.py part7/operating_system
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
import streamlit as st

from components.visual_system import render_visual_system


T = TypeVar("T")

st.set_page_config(page_title="操作系统面试训练", layout="wide", initial_sidebar_state="expanded")


@dataclass(frozen=True)
class Job:
    name: str
    arrive: int
    burst: int


JOBS = [Job("P1", 0, 7), Job("P2", 2, 4), Job("P3", 4, 1), Job("P4", 5, 4)]


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
    .diagram { background: #172026; color: #f7fbfc; border-radius: 8px; padding: .82rem 1rem; font-family: Consolas, "Courier New", monospace; line-height: 1.72; white-space: pre-wrap; }
    .pill { display: inline-block; border: 1px solid #d8dee3; background: rgba(255,255,255,.84); border-radius: 8px; padding: .62rem .75rem; margin: .22rem; min-width: 9rem; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("操作系统页面执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="os-back-home", use_container_width=True):
        st.query_params.clear()
        st.rerun()


def fcfs(jobs: list[Job]) -> list[tuple[str, int, int]]:
    timeline = []
    now = 0
    for job in sorted(jobs, key=lambda item: item.arrive):
        now = max(now, job.arrive)
        start = now
        now += job.burst
        timeline.append((job.name, start, now))
    return timeline


def sjf(jobs: list[Job]) -> list[tuple[str, int, int]]:
    remaining = jobs[:]
    timeline = []
    now = 0
    while remaining:
        available = [job for job in remaining if job.arrive <= now]
        if not available:
            now = min(job.arrive for job in remaining)
            available = [job for job in remaining if job.arrive <= now]
        job = min(available, key=lambda item: item.burst)
        remaining.remove(job)
        start = now
        now += job.burst
        timeline.append((job.name, start, now))
    return timeline


def rr(jobs: list[Job], quantum: int) -> list[tuple[str, int, int]]:
    remaining = {job.name: job.burst for job in jobs}
    arrived = sorted(jobs, key=lambda item: item.arrive)
    ready: list[Job] = []
    timeline: list[tuple[str, int, int]] = []
    now = 0
    index = 0
    while any(value > 0 for value in remaining.values()):
        while index < len(arrived) and arrived[index].arrive <= now:
            ready.append(arrived[index])
            index += 1
        if not ready:
            now = arrived[index].arrive
            continue
        job = ready.pop(0)
        run = min(quantum, remaining[job.name])
        start = now
        now += run
        remaining[job.name] -= run
        timeline.append((job.name, start, now))
        while index < len(arrived) and arrived[index].arrive <= now:
            ready.append(arrived[index])
            index += 1
        if remaining[job.name] > 0:
            ready.append(job)
    return timeline


def metrics(timeline: list[tuple[str, int, int]], jobs: list[Job]) -> pd.DataFrame:
    finish = {}
    for name, _, end in timeline:
        finish[name] = end
    rows = []
    for job in jobs:
        turnaround = finish[job.name] - job.arrive
        waiting = turnaround - job.burst
        rows.append([job.name, job.arrive, job.burst, finish[job.name], waiting, turnaround])
    return pd.DataFrame(rows, columns=["进程", "到达时间", "运行时间", "完成时间", "等待时间", "周转时间"])


def render_timeline(timeline: list[tuple[str, int, int]]) -> None:
    parts = [f"| {name} {start}-{end} " for name, start, end in timeline]
    st.markdown(f'<div class="diagram">{"".join(parts)}|</div>', unsafe_allow_html=True)


def main() -> None:
    render_visual_system("dark")
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>操作系统面试训练</h1>
          <p>围绕进程线程、调度、虚拟内存和死锁，把 OS 基础落到训练性能、DataLoader 和 GPU 资源管理上。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("进程与线程对比")
    st.table(
        pd.DataFrame(
            [
                ["资源拥有", "拥有独立地址空间和系统资源", "共享所属进程的地址空间和资源"],
                ["创建/切换成本", "较高", "较低"],
                ["隔离性", "强，一个进程崩溃通常不影响其他进程", "弱，一个线程崩溃可能影响整个进程"],
                ["通信方式", "管道、消息队列、共享内存、Socket", "共享变量、锁、条件变量"],
                ["典型场景", "浏览器多进程、Python 多进程 DataLoader", "Web 服务器线程池、后台异步任务"],
            ],
            columns=["维度", "进程", "线程"],
        )
    )

    st.subheader("进程调度算法演示")
    left, right = st.columns([0.34, 0.66])
    with left:
        algorithm = st.selectbox("选择调度算法", ["FCFS", "SJF", "Round Robin"], key="os-scheduler")
        quantum = st.slider("调整时间片大小", 1, 6, 2, key="os-quantum")
        st.table(pd.DataFrame([[j.name, j.arrive, j.burst] for j in JOBS], columns=["进程", "到达", "运行"]))
    if algorithm == "FCFS":
        timeline = fcfs(JOBS)
    elif algorithm == "SJF":
        timeline = sjf(JOBS)
    else:
        timeline = rr(JOBS, quantum)
    with right:
        render_timeline(timeline)
        df = metrics(timeline, JOBS)
        st.table(df)
        st.metric("平均等待时间", f"{df['等待时间'].mean():.2f}")
        st.metric("平均周转时间", f"{df['周转时间'].mean():.2f}")

    st.subheader("虚拟内存和分页机制图解")
    st.markdown(
        """
        <div class="diagram">进程虚拟地址: [ 页号 VPN | 页内偏移 offset ]
                       |
                       v
                 页表 / TLB 查询
                       |
        物理地址: [ 物理页框 PFN | 页内偏移 offset ]

如果页表项不存在或无权限 -> 缺页异常 / 保护异常
如果页面在磁盘上 -> OS 调页，更新页表，再恢复执行</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("死锁四个必要条件")
    st.markdown(
        """
        <span class="pill"><strong>互斥</strong><br>资源一次只能被一个执行流占用</span>
        <span class="pill"><strong>占有且等待</strong><br>拿着已有资源继续等待新资源</span>
        <span class="pill"><strong>不可抢占</strong><br>资源不能被强行剥夺</span>
        <span class="pill"><strong>循环等待</strong><br>A 等 B，B 等 C，C 又等 A</span>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("高频问答区")
    with st.expander("进程和线程区别？"):
        st.write("进程是资源分配单位，线程是 CPU 调度单位。进程隔离更强、通信更重；线程共享内存、切换更轻，但同步和崩溃影响更复杂。")
    with st.expander("什么是上下文切换？"):
        st.write("CPU 从一个执行流切到另一个执行流时，需要保存当前寄存器、程序计数器、栈等状态，再恢复目标执行流状态。切换过多会带来调度和缓存失效成本。")
    with st.expander("死锁如何避免？"):
        st.write("破坏四个必要条件之一：统一加锁顺序破坏循环等待；一次性申请资源破坏占有且等待；设置超时和回滚；或用银行家算法在分配前判断安全状态。")
    with st.expander("虚拟内存为什么存在？"):
        st.write("它给进程提供连续、隔离、可保护的地址空间，并允许按需加载、换页和共享映射。应用不需要直接管理物理内存碎片，OS 可以更灵活地调度内存。")

    st.subheader("与深度学习的连接")
    st.markdown(
        """
        <div class="note">
        多进程 DataLoader 依赖进程调度和 IPC；GPU 训练会涉及 CPU/GPU 任务队列、显存分配和 kernel 调度；
        大 batch、预取、pin memory、共享内存不足、上下文切换过多都会影响训练吞吐。OS 基础能帮你定位“模型没变但训练变慢”的工程问题。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("进入操作系统专项刷题", "/?module=part7%2Finterview_quiz", width="stretch")
    render_back_home()


if __name__ == "__main__":
    safe_run(main)
