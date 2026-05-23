"""
CS interview quiz mode.

Run:
    streamlit run part7_interview/interview_quiz.py
or:
    python main.py part7/interview_quiz
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
import streamlit as st


T = TypeVar("T")

st.set_page_config(page_title="CS 面试刷题模式", layout="wide", initial_sidebar_state="expanded")


@dataclass(frozen=True)
class QuizItem:
    qid: str
    direction: str
    difficulty: str
    question: str
    answer: str
    follow_up: str
    trap: str
    application: str


QUESTIONS: tuple[QuizItem, ...] = (
    QuizItem("net-01", "网络", "基础", "TCP 和 UDP 的区别是什么？", "TCP 面向连接、可靠、有序、拥塞控制，适合文件、网页、接口调用；UDP 无连接、尽力而为、开销小，适合直播、游戏、DNS 等可容忍丢包或自行保证可靠性的场景。", "如果让你设计一个推理服务接口，会选 TCP 还是 UDP？为什么？", "只说 TCP 可靠、UDP 不可靠，不讲连接、顺序、拥塞控制和业务取舍。", "模型推理 API 通常走 HTTP/gRPC over TCP，因为需要可靠返回结果和清晰错误语义。"),
    QuizItem("net-02", "网络", "高频", "为什么 TCP 需要三次握手？", "三次握手用于确认双方收发能力并同步初始序列号。两次握手无法让服务端确认客户端已经收到自己的 SYN+ACK，也更容易被历史连接请求干扰。", "如果只有两次握手，会出现什么异常连接？", "把答案背成“防止资源浪费”，但说不清双向能力确认和旧报文问题。", "服务端推理接口要避免半开连接堆积，连接建立和超时配置会影响吞吐。"),
    QuizItem("net-03", "网络", "进阶", "HTTPS 的安全性来自哪里？", "HTTPS 来自 TLS：证书链验证身份，密钥协商生成会话密钥，对称加密保护内容，完整性校验防篡改。", "证书过期或域名不匹配时浏览器为什么报警？", "误以为 HTTPS 只是在 HTTP 上做一次简单加密，忽略身份认证。", "公网模型服务传输用户输入、图片或日志时，TLS 是基本安全边界。"),
    QuizItem("net-04", "网络", "大厂追问", "浏览器输入 URL 后发生了什么？", "URL 解析、缓存检查、DNS 查询、TCP 建连、TLS 握手、HTTP 请求响应、浏览器解析渲染，并加载 CSS/JS/图片等子资源。", "如果首屏很慢，你会从哪些网络指标排查？", "只说 DNS、TCP、HTTP 三步，漏掉缓存、TLS、渲染和子资源。", "推理平台控制台慢，常用 DNS、TTFB、连接复用和 CDN 缓存指标定位。"),
    QuizItem("net-05", "网络", "高频", "TIME_WAIT 为什么存在？", "TIME_WAIT 让主动关闭方等待足够时间，确保最后 ACK 可重传，并让旧连接报文在网络中自然消失，避免污染新连接。", "大量 TIME_WAIT 一定是坏事吗？怎么优化？", "把 TIME_WAIT 当成连接泄漏，不区分主动关闭和协议保护。", "高 QPS 推理网关要关注连接池、keep-alive 和端口耗尽。"),
    QuizItem("db-01", "数据库", "基础", "SELECT 查询大致如何执行？", "SQL 先解析和预处理，再由优化器选择执行计划，执行器按计划访问存储引擎，最后做过滤、排序、聚合并返回结果。", "优化器会根据什么选择索引？", "以为 SQL 一定按书写顺序逐行执行。", "实验平台查询 run 记录时，执行计划决定列表页响应速度。"),
    QuizItem("db-02", "数据库", "高频", "索引为什么能加速查询？", "索引用 B+ 树或哈希等结构把全表扫描变成快速定位。B+ 树分叉大、树高低，范围查询还能沿叶子链表顺序扫描。", "什么情况下索引反而不一定会被使用？", "只说“索引像目录”，不讲减少扫描行数和页读取。", "样本元数据按 dataset_id、label、split 建索引，可快速筛训练集。"),
    QuizItem("db-03", "数据库", "高频", "索引为什么不是越多越好？", "索引占空间，也会拖慢写入和更新，因为数据变更时要维护索引页。索引过多还会增加优化器选择成本。", "一个表应该怎么设计联合索引？", "只从查询角度看索引，忽略写入和维护成本。", "训练日志高频写入表不适合盲目给每个指标列都建索引。"),
    QuizItem("db-04", "数据库", "进阶", "聚簇索引和非聚簇索引有什么区别？", "聚簇索引叶子节点存整行数据，表数据按索引组织；非聚簇索引叶子节点存索引列和主键或行指针，查完整行可能需要回表。", "InnoDB 为什么推荐使用自增主键？", "把聚簇索引理解成“唯一索引”，混淆概念。", "实验记录表用稳定主键能减少页分裂，查询详情也更直接。"),
    QuizItem("db-05", "数据库", "大厂追问", "SQL 慢查询怎么排查？", "先定位慢 SQL，再 EXPLAIN 看访问类型、索引命中、扫描行数、排序临时表；再查索引设计、数据分布、锁等待、网络和缓存。", "EXPLAIN 里 type=ALL、Using filesort 分别说明什么？", "只回答“加索引”，不先定位瓶颈。", "模型管理后台慢查询常来自实验表数据暴涨、条件低选择性或排序未命中索引。"),
    QuizItem("algo-01", "算法", "基础", "数组和链表的区别？", "数组连续存储，随机访问 O(1)，插入删除可能 O(n)；链表离散存储，插入删除改指针较方便，但查找第 k 个节点 O(n)。", "CPU 缓存对数组和链表性能有什么影响？", "只背复杂度，不说连续内存和缓存友好性。", "张量底层通常依赖连续数组布局，影响矩阵计算效率。"),
    QuizItem("algo-02", "算法", "高频", "哈希表冲突怎么解决？", "常见有链地址法、开放寻址法；工程上还要控制负载因子、扩容策略和哈希函数质量。", "最坏情况下哈希表为什么会退化？", "以为哈希表任何操作永远 O(1)。", "特征字典、词表和缓存都依赖哈希表，冲突会影响吞吐。"),
    QuizItem("algo-03", "算法", "高频", "快排为什么平均 O(n log n)？", "partition 每层总成本 O(n)，如果枢轴平均能较均衡切分，递归深度 O(log n)，总成本 O(n log n)。", "最坏 O(n^2) 怎么避免？", "只记结论，不会从递归树解释。", "大规模样本排序、top-k 和分桶采样都要理解分治代价。"),
    QuizItem("algo-04", "算法", "进阶", "BFS 和 DFS 的区别？", "BFS 按层扩展，适合最短路径和层序遍历；DFS 沿路径深入，适合回溯、连通性和拓扑相关问题。两者都常用 visited 防重复。", "无权图最短路径为什么用 BFS？", "不区分队列和栈/递归的访问顺序。", "Beam Search、图计算和依赖分析都能看到搜索算法影子。"),
    QuizItem("algo-05", "算法", "大厂追问", "如何分析一个算法的时间和空间复杂度？", "先找输入规模 n，再数核心操作随 n 增长的阶；递归看递归树或主定理，额外空间看辅助结构、递归栈和输出是否计入。", "两个嵌套循环一定是 O(n^2) 吗？", "机械套循环层数，不看循环边界和数据规模。", "注意力 O(n^2)、KV cache 空间、batch 维度都会影响模型能否部署。"),
    QuizItem("os-01", "操作系统", "基础", "进程和线程有什么区别？", "进程是资源分配单位，有独立地址空间；线程是 CPU 调度单位，共享进程资源。进程隔离强但通信重，线程轻但同步复杂。", "Python 多线程为什么不适合 CPU 密集任务？", "只说进程大线程小，不讲地址空间和调度。", "PyTorch DataLoader 常用多进程绕开 GIL 并提升数据读取吞吐。"),
    QuizItem("os-02", "操作系统", "高频", "什么是上下文切换？", "CPU 保存当前执行流状态，再恢复另一个执行流状态，包括寄存器、程序计数器、栈等。频繁切换会带来调度和缓存失效成本。", "怎么观察上下文切换过多？", "只说“切换进程”，忽略线程切换和缓存影响。", "训练吞吐异常时，过多 worker、锁竞争和小任务碎片化都会导致切换成本上升。"),
    QuizItem("os-03", "操作系统", "高频", "死锁的四个必要条件是什么？", "互斥、占有且等待、不可抢占、循环等待。处理死锁可以预防、避免、检测和恢复。", "实际工程里如何用加锁顺序避免死锁？", "只背四个词，不会解释每个条件。", "多线程数据预处理、日志写入和缓存更新都可能出现锁顺序问题。"),
    QuizItem("os-04", "操作系统", "进阶", "虚拟内存为什么存在？", "虚拟内存提供连续、隔离、可保护的地址空间，并支持按需加载、换页和共享映射，让 OS 更灵活管理物理内存。", "缺页异常一定是错误吗？", "把虚拟内存理解成单纯扩大内存。", "大模型训练会同时受 CPU 内存、共享内存、显存和 mmap 数据集影响。"),
    QuizItem("os-05", "操作系统", "大厂追问", "FCFS、SJF、Round Robin 有什么取舍？", "FCFS 简单但短任务可能被长任务阻塞；SJF 平均等待时间低但需要估计运行时间；RR 公平、适合交互系统，但时间片过小会增加切换成本。", "时间片设置过大或过小分别会怎样？", "只背算法名称，不会联系等待时间和周转时间。", "训练集预处理任务、推理请求队列和多租户 GPU 调度都涉及公平性与吞吐权衡。"),
    QuizItem("dl-01", "深度学习", "基础", "过拟合是什么，怎么缓解？", "过拟合是训练集表现好但验证/测试集表现差。可用更多数据、数据增强、正则化、Dropout、早停、降低模型复杂度等缓解。", "如何区分过拟合和数据分布漂移？", "只看训练准确率，不看验证曲线。", "实验记录数据库会保存训练/验证曲线，用来判断是否过拟合。"),
    QuizItem("dl-02", "深度学习", "高频", "BatchNorm 和 LayerNorm 的区别？", "BatchNorm 通常按 batch 统计归一化，依赖 batch 分布；LayerNorm 在单样本特征维度归一化，更适合 Transformer 和变长序列。", "小 batch 时 BatchNorm 为什么可能不稳定？", "只说都是归一化，不讲统计维度。", "训练视觉模型和语言模型时，归一化选择会影响稳定性和部署一致性。"),
    QuizItem("dl-03", "深度学习", "高频", "为什么 Transformer 注意力复杂度是 O(n^2)？", "自注意力要计算每个 token 与其他 token 的相关性，形成 n x n 注意力矩阵，所以时间和显存都随序列长度平方增长。", "KV cache 能优化训练还是推理？", "只记 O(n^2)，不会说明注意力矩阵来源。", "长上下文推理和 RAG 文档拼接会直接受注意力复杂度限制。"),
    QuizItem("dl-04", "深度学习", "进阶", "模型部署时如何降低推理延迟？", "可从模型压缩、量化、批处理、缓存、并发、编译优化、硬件选择、网络连接复用和输入输出裁剪入手。", "动态 batching 会带来什么副作用？", "只谈模型结构，不看网络、队列和序列化开销。", "推理 API 延迟由 DNS/TLS/网关、排队、预处理、GPU 执行和后处理共同决定。"),
    QuizItem("dl-05", "深度学习", "大厂追问", "如何排查训练 loss 不下降？", "先确认数据和标签，再检查模型前向、损失函数、学习率、梯度是否为 0/NaN、参数是否更新、归一化和初始化是否合理，最后缩小到小数据集过拟合实验。", "为什么先做小数据集过拟合实验？", "一上来盲目换模型或调大训练轮数。", "训练日志、梯度监控和实验配置记录能让排查从猜测变成证据链。"),
)


def css() -> str:
    return """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1180px; }
    .stApp { background: #f7f8f4; color: #172026; }
    h1, h2, h3, p, li, label, span { letter-spacing: 0; }
    .hero { border-bottom: 1px solid #d8dee3; padding-bottom: 1rem; margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: clamp(2rem, 3vw, 3.1rem); }
    .hero p { color: #596772; line-height: 1.7; max-width: 920px; }
    .question { background: rgba(255,255,255,.86); border: 1px solid #d8dee3; border-radius: 8px; padding: 1rem 1.1rem; line-height: 1.7; margin: .5rem 0 1rem; }
    .tag { display: inline-block; border: 1px solid #c7d1d5; border-radius: 999px; padding: .16rem .55rem; margin-right: .35rem; color: #42515a; background: #fff; font-size: .84rem; }
    .note { border-left: 4px solid #0f8b8d; background: rgba(255,255,255,.78); border-radius: 0 8px 8px 0; padding: .74rem .9rem; line-height: 1.68; margin: .4rem 0 .9rem; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("面试刷题模式执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="quiz-back-home", use_container_width=True):
        st.query_params.clear()
        st.rerun()


def ensure_state() -> None:
    st.session_state.setdefault("interview_current_qid", "")
    st.session_state.setdefault("interview_wrong_book", [])
    st.session_state.setdefault("interview_later_book", [])
    st.session_state.setdefault("interview_answered_count", 0)
    st.session_state.setdefault("interview_correct_count", 0)
    st.session_state.setdefault("interview_user_answer_visible", False)
    st.session_state.setdefault("interview_user_answer", "")


def filtered_questions(direction: str, difficulty: str) -> list[QuizItem]:
    return [
        item
        for item in QUESTIONS
        if (direction == "全部" or item.direction == direction)
        and (difficulty == "全部" or item.difficulty == difficulty)
    ]


def pick_question(candidates: list[QuizItem]) -> None:
    if not candidates:
        st.session_state["interview_current_qid"] = ""
        return
    st.session_state["interview_current_qid"] = random.choice(candidates).qid
    st.session_state["interview_user_answer_visible"] = False
    st.session_state["interview_user_answer"] = ""


def current_question(candidates: list[QuizItem]) -> QuizItem | None:
    qid = st.session_state.get("interview_current_qid", "")
    by_id = {item.qid: item for item in candidates}
    if qid in by_id:
        return by_id[qid]
    if candidates:
        st.session_state["interview_current_qid"] = candidates[0].qid
        return candidates[0]
    return None


def add_unique(state_key: str, item: QuizItem) -> None:
    rows = list(st.session_state.get(state_key, []))
    if item.qid not in [row["qid"] for row in rows]:
        rows.append(
            {
                "qid": item.qid,
                "方向": item.direction,
                "难度": item.difficulty,
                "题目": item.question,
                "标准答案": item.answer,
            }
        )
    st.session_state[state_key] = rows


def main() -> None:
    ensure_state()
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>CS 面试刷题模式</h1>
          <p>随机出题、按方向和难度筛选，并把错题与稍后复习题留在本轮会话中，适合高频八股的快速口述训练。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.3, 0.7])
    with left:
        direction = st.selectbox("按方向筛选", ["全部", "网络", "数据库", "算法", "操作系统", "深度学习"], key="quiz-direction")
        difficulty = st.selectbox("按难度筛选", ["全部", "基础", "高频", "进阶", "大厂追问"], key="quiz-difficulty")
        candidates = filtered_questions(direction, difficulty)
        st.metric("当前题库数量", len(candidates))

        # 本轮会话统计
        answered = st.session_state.get("interview_answered_count", 0)
        correct = st.session_state.get("interview_correct_count", 0)
        if answered > 0:
            accuracy = correct / answered * 100
            st.metric("本轮正确率", f"{accuracy:.0f}%")
            st.caption(f"已答 {answered} 题，答对 {correct} 题")

        if st.button("随机出题", key="quiz-random", use_container_width=True):
            pick_question(candidates)
        if st.button("清空错题本", key="quiz-clear", use_container_width=True):
            st.session_state["interview_wrong_book"] = []
            st.session_state["interview_later_book"] = []
            st.session_state["interview_answered_count"] = 0
            st.session_state["interview_correct_count"] = 0

    with right:
        candidates = filtered_questions(direction, difficulty)
        item = current_question(candidates)
        if item is None:
            st.warning("当前筛选条件下没有题目，请放宽方向或难度。")
        else:
            st.markdown(
                f"""
                <div class="question">
                  <span class="tag">{item.direction}</span>
                  <span class="tag">{item.difficulty}</span>
                  <h3>{item.question}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 先自己作答模式
            if not st.session_state.get("interview_user_answer_visible", False):
                user_answer = st.text_area(
                    "先自己想一想，口述或写下你的答案：",
                    value=st.session_state.get("interview_user_answer", ""),
                    height=120,
                    key="quiz-user-input",
                    placeholder="写出你的理解，尽量覆盖关键点、边界条件和工程取舍...",
                )
                st.session_state["interview_user_answer"] = user_answer
                if st.button("提交并查看标准答案", key="quiz-show-answer", use_container_width=True):
                    st.session_state["interview_user_answer_visible"] = True
                    st.rerun()
            else:
                # 显示用户答案
                user_answer = st.session_state.get("interview_user_answer", "")
                if user_answer.strip():
                    st.markdown("**你的答案：**")
                    st.info(user_answer)

            # 标准答案和追问（仅在提交后显示）
            if st.session_state.get("interview_user_answer_visible", False):
                with st.expander("标准答案", expanded=True):
                    st.write(item.answer)
                with st.expander("面试官追问"):
                    st.write(item.follow_up)
                with st.expander("初学者易错点"):
                    st.write(item.trap)
                with st.expander("项目中的真实应用"):
                    st.write(item.application)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("我答对了", key="quiz-right", use_container_width=True):
                    st.session_state["interview_answered_count"] = st.session_state.get("interview_answered_count", 0) + 1
                    st.session_state["interview_correct_count"] = st.session_state.get("interview_correct_count", 0) + 1
                    pick_question(candidates)
            with c2:
                if st.button("我答错了", key="quiz-wrong", use_container_width=True):
                    st.session_state["interview_answered_count"] = st.session_state.get("interview_answered_count", 0) + 1
                    add_unique("interview_wrong_book", item)
                    pick_question(candidates)
            with c3:
                if st.button("稍后复习", key="quiz-later", use_container_width=True):
                    add_unique("interview_later_book", item)
                    pick_question(candidates)

    st.subheader("错题本")
    wrong = st.session_state.get("interview_wrong_book", [])
    later = st.session_state.get("interview_later_book", [])
    tabs = st.tabs([f"答错 {len(wrong)}", f"稍后复习 {len(later)}"])
    with tabs[0]:
        if wrong:
            st.dataframe(pd.DataFrame(wrong), use_container_width=True, hide_index=True)
        else:
            st.info("本轮还没有答错题。")
    with tabs[1]:
        if later:
            st.dataframe(pd.DataFrame(later), use_container_width=True, hide_index=True)
        else:
            st.info("本轮还没有标记稍后复习。")

    st.markdown(
        """
        <div class="note">
        训练建议：先在输入框里写下你的答案，再提交对照标准答案。真正面试时要主动给出边界条件、工程取舍和排查路径，
        不要只背一句定义。对比自己的答案和标准答案，找出遗漏的关键点。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_back_home()


if __name__ == "__main__":
    safe_run(main)

