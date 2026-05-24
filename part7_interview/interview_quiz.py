"""
CS interview quiz mode.

Run:
    streamlit run part7_interview/interview_quiz.py
or:
    python main.py part7/interview_quiz
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


QUESTION_MODULE_MAP = {
    "网络": "part7/networking",
    "数据库": "part7/database_sql",
    "算法": "part7/data_structures",
    "操作系统": "part7/operating_system",
    "深度学习": "part6/neural_network_playground",
    "系统设计": "part7/interview_quiz",
}

SCORING_DIMENSIONS = (
    ("定义准确", "能说清核心概念，不只背关键词。", 30),
    ("机制完整", "覆盖关键步骤、边界条件和取舍。", 30),
    ("工程落地", "能联系真实系统或深度学习工程场景。", 25),
    ("表达结构", "答案有层次，先结论再展开。", 15),
)

SIMULATION_SCRIPTS = {
    "后端基础岗": ("网络", "数据库", "算法", "操作系统", "系统设计"),
    "算法工程岗": ("深度学习", "算法", "操作系统", "系统设计", "数据库"),
    "MLOps/平台岗": ("系统设计", "操作系统", "网络", "数据库", "深度学习"),
}


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
    QuizItem("net-06", "网络", "高频", "HTTP 状态码 301、302、304、401、403、502 分别表示什么？", "301 是永久重定向，302 是临时重定向，304 表示协商缓存命中，401 表示未认证，403 表示已认证但无权限，502 常见于网关收到上游异常响应。", "401 和 403 在登录态失效、权限不足时如何区分处理？", "只背 200、404、500，不能解释状态码和缓存、鉴权、网关的关系。", "模型服务控制台常用状态码区分前端缓存、用户权限、网关和推理后端故障。"),
    QuizItem("net-07", "网络", "进阶", "HTTP/1.1、HTTP/2 和 HTTP/3 的核心差异是什么？", "HTTP/1.1 以文本协议和连接复用为主，但容易有队头阻塞；HTTP/2 使用二进制分帧和多路复用；HTTP/3 基于 QUIC/UDP，把连接迁移、握手延迟和传输层队头阻塞做了改进。", "HTTP/2 已经多路复用，为什么仍可能受 TCP 队头阻塞影响？", "把 HTTP/2 简单理解成更快版本，不讲二进制分帧、多路复用和传输层差异。", "流式推理、SSE 和大文件下载会受到连接复用、队头阻塞和代理协议支持影响。"),
    QuizItem("net-08", "网络", "大厂追问", "DNS 解析慢或异常时怎么排查？", "先区分本地缓存、递归解析器、权威 DNS、网络链路和域名配置问题；再看 TTL、CNAME 链路、解析地域、劫持、超时重试和备用解析策略。", "为什么过长的 CNAME 链路会增加首包延迟？", "只会说清缓存或换 DNS，不会沿解析链路定位。", "公网推理 API 首次访问慢时，DNS、TLS 和连接建立往往和模型计算一样需要被拆开测量。"),
    QuizItem("db-06", "数据库", "高频", "事务的 ACID 分别是什么？", "原子性保证事务内操作要么全成要么全不成，一致性保证约束不被破坏，隔离性控制并发事务相互影响，持久性保证提交后数据可靠保存。", "隔离性和一致性为什么不是一回事？", "只背四个英文单词，不会解释每项解决的问题。", "实验配置、训练任务状态和计费记录需要事务保证状态变更不会半成功。"),
    QuizItem("db-07", "数据库", "进阶", "MySQL 常见事务隔离级别解决了哪些并发问题？", "读未提交可能脏读；读已提交避免脏读但可能不可重复读；可重复读避免不可重复读，MySQL InnoDB 通过 MVCC 和间隙锁处理部分幻读；串行化最严格但并发最低。", "MVCC 为什么能让读写并发更高？", "把隔离级别背成列表，不讲脏读、不可重复读、幻读和实现代价。", "多人同时更新训练任务、审批模型版本时，隔离级别会影响一致性和吞吐。"),
    QuizItem("db-08", "数据库", "大厂追问", "数据库主从复制延迟会带来什么问题，怎么处理？", "复制延迟会导致读到旧数据、状态回退和幂等判断错误。处理方式包括读写同源、关键路径读主库、延迟监控、半同步复制、版本号校验和业务补偿。", "为什么刚写完立刻读从库可能读不到？", "只说加主从能提升性能，不考虑一致性窗口。", "训练任务提交后立刻查询状态，如果读到从库旧数据，用户会看到任务消失或状态倒退。"),
    QuizItem("algo-06", "算法", "高频", "二分查找最容易错在哪里？", "关键是维护搜索区间语义，明确左右边界是闭区间还是半开区间，并保证循环能收敛；找第一个、最后一个满足条件的位置时要配合单调谓词更新答案。", "如何用二分查找找到第一个大于等于 target 的位置？", "只会写普通命中 target 的版本，边界、重复元素和插入位置容易错。", "阈值搜索、学习率扫描和日志时间范围定位都可抽象成单调条件上的二分。"),
    QuizItem("algo-07", "算法", "进阶", "动态规划题通常怎么识别和建模？", "如果问题有重叠子问题和最优子结构，就考虑 DP。建模时先定义状态含义，再写状态转移、初始化、遍历顺序和答案位置，必要时做空间优化。", "为什么有些 DP 需要倒序遍历？", "一上来套模板，不先定义状态，导致转移式没有语义。", "序列对齐、最短编辑距离和部分解码算法都能看到动态规划思想。"),
    QuizItem("algo-08", "算法", "大厂追问", "Top-K 问题有哪些常见解法？", "小规模可排序；数据流或大规模可用大小为 k 的堆；值域有限可计数；需要平均线性时间可用快速选择；分布式场景先局部 Top-K 再合并。", "海量数据放不进内存时怎么做 Top-K？", "只会全量排序，忽略时间、空间和数据规模。", "推荐召回、相似样本检索和日志热点分析都常把问题化成 Top-K。"),
    QuizItem("os-06", "操作系统", "高频", "进程间通信有哪些方式？", "常见方式包括管道、消息队列、共享内存、信号量、信号、Socket 和文件。共享内存快但需要同步，Socket 适合跨机器，消息队列解耦但有额外开销。", "共享内存为什么还需要锁或信号量？", "只罗列 IPC 名称，不会比较适用场景和同步成本。", "多进程数据加载、推理 worker 通信和本地服务编排都涉及 IPC 选择。"),
    QuizItem("os-07", "操作系统", "进阶", "分页和分段有什么区别？", "分页按固定大小页管理物理内存，利于离散分配和换页；分段按逻辑模块划分，便于表达代码、数据、栈等语义但会有外部碎片。现代系统通常以分页为主，也保留段式保护的概念。", "页表为什么可能很大，怎么优化？", "把页和段都理解成切内存，不讲固定大小、逻辑语义和碎片类型。", "mmap 数据集、显存分页和大页优化都会影响训练数据读取和内存效率。"),
    QuizItem("os-08", "操作系统", "大厂追问", "高 CPU 使用率但吞吐不高时怎么排查？", "先区分用户态、内核态、IO wait 和上下文切换，再看热点函数、锁竞争、GC、系统调用、线程数和队列积压；需要结合 top、perf、日志和压测指标定位。", "CPU 高一定说明计算资源不够吗？", "看到 CPU 高就直接扩容，不拆用户态计算、内核开销和等待。", "推理服务 CPU 可能耗在序列化、图片预处理、日志、锁竞争或网络栈，而不是模型本身。"),
    QuizItem("dl-06", "深度学习", "高频", "梯度消失和梯度爆炸是什么，怎么缓解？", "梯度消失是反向传播时梯度越来越小，前层难以学习；梯度爆炸是梯度过大导致更新不稳定。可用合适初始化、归一化、残差连接、门控结构、梯度裁剪和学习率调整缓解。", "为什么残差连接有助于深层网络训练？", "只说调小学习率，不区分消失和爆炸，也不看梯度统计。", "训练平台的梯度监控能直接暴露梯度范数异常和更新比例异常。"),
    QuizItem("dl-07", "深度学习", "进阶", "混合精度训练为什么能提速，风险是什么？", "混合精度用 FP16/BF16 降低显存和带宽压力，并利用硬件张量核心加速；风险包括数值下溢、溢出和不稳定，通常配合 loss scaling、保留 FP32 主权重或选择 BF16。", "BF16 和 FP16 在动态范围上有什么差异？", "只把混合精度理解成把所有张量改成半精度。", "大模型训练常靠混合精度扩大 batch 或降低显存占用，但要监控 NaN 和 loss spike。"),
    QuizItem("dl-08", "深度学习", "大厂追问", "如何判断一个模型线上效果变差是模型问题还是数据问题？", "先对比线上输入分布、标签或反馈延迟、特征缺失、预处理版本、训练数据覆盖、模型版本和服务日志；再用离线回放、分桶评估、A/B 实验和漂移指标建立证据链。", "为什么只看整体准确率可能掩盖问题？", "一看到指标下降就回滚模型，不排查数据漂移、特征链路和流量变化。", "线上推理系统要把模型版本、数据版本、预处理配置和监控指标打通，才能定位退化来源。"),
    # ── 扩充题库：每个方向新增 4 题 ──
    QuizItem("net-09", "网络", "基础", "Cookie、Session 和 Token 的区别是什么？", "Cookie 存在浏览器端，随请求自动发送；Session 在服务端存储用户状态，靠 Cookie 中的 Session ID 关联；Token（如 JWT）是自包含的凭证，服务端无状态验证，适合分布式和跨服务认证。", "为什么 JWT 适合微服务架构但不适合存大量用户数据？", "把 Token 和 Session 混为一谈，或者不区分存储位置和验证方式。", "推理网关常用 JWT 做服务鉴权，避免每次请求都查数据库。"),
    QuizItem("net-10", "网络", "进阶", "WebSocket 和 HTTP 长轮询有什么区别？", "WebSocket 是全双工持久连接，服务端可主动推送；HTTP 长轮询是客户端发起请求后服务端挂起直到有数据才响应，本质还是半双工的请求-响应模式。WebSocket 延迟更低、开销更小，但需要额外的握手和心跳维护。", "推理服务的流式输出用 SSE 还是 WebSocket 更合适？", "只说 WebSocket 更快，不讲全双工、协议升级和心跳机制。", "大模型推理的流式 token 输出常用 SSE 或 WebSocket，选择取决于是否需要双向通信。"),
    QuizItem("net-11", "网络", "大厂追问", "CDN 的工作原理是什么，为什么能加速？", "CDN 通过在全球部署边缘节点缓存静态资源，用户请求被 DNS 动态导向最近节点，减少网络跳数和延迟。回源机制保证缓存过期时从源站更新，分层缓存降低源站压力。", "CDN 缓存命中率低时应该怎么排查？", "只说 CDN 就是加速，不讲 DNS 调度、边缘节点、回源和缓存策略。", "模型文件、前端资源和 API 网关都可以利用 CDN 降低全球访问延迟。"),
    QuizItem("net-12", "网络", "高频", "负载均衡有哪些常见算法？", "轮询按顺序分配、加权轮询按权重分配、最少连接优先选当前连接最少的服务器、IP 哈希保证同一客户端打到同一后端、一致性哈希适合有缓存的分布式场景。L4 负载均衡工作在传输层，L7 工作在应用层可做更细粒度路由。", "为什么微服务网关通常用 L7 负载均衡？", "只说轮询，不讲加权、最少连接、会话保持和分层区别。", "推理集群的负载均衡直接影响 GPU 利用率和请求延迟，需要结合队列深度和批处理策略。"),
    QuizItem("db-09", "数据库", "基础", "SQL 注入是什么，怎么防御？", "SQL 注入是攻击者通过输入恶意 SQL 片段改变查询逻辑，获取或篡改数据。防御手段包括参数化查询、预编译语句、ORM、输入校验和最小权限原则。", "ORM 为什么能天然防 SQL 注入？", "只说转义特殊字符，不讲参数化查询和预编译的原理。", "实验平台的查询接口如果不做参数化，攻击者可通过筛选条件注入恶意 SQL。"),
    QuizItem("db-10", "数据库", "进阶", "分库分表的策略和常见问题是什么？", "水平拆分按行分到多个表或库，常见分片键有用户 ID、时间范围；垂直拆分按业务拆不同表或库。常见问题包括跨分片查询、分布式事务、数据迁移、扩容再平衡和全局唯一 ID。", "为什么分片键选择很重要？", "一上来就分库分表，不先评估单表性能瓶颈。", "实验记录、训练日志按时间或模型 ID 分表，可控制单表大小并提升查询效率。"),
    QuizItem("db-11", "数据库", "高频", "Redis 和 MySQL 有什么区别，各适合什么场景？", "Redis 是内存键值存储，读写微秒级，支持丰富数据结构，适合缓存、会话、排行榜和分布式锁；MySQL 是关系型数据库，支持复杂 SQL、事务和持久化，适合结构化业务数据。两者常配合使用，Redis 做缓存层加速 MySQL 查询。", "缓存和数据库的数据一致性怎么保证？", "把 Redis 当成 MySQL 的替代品，或者不考虑持久化和一致性。", "模型配置、推理缓存和用户会话常用 Redis 加速，训练元数据和实验记录存 MySQL。"),
    QuizItem("db-12", "数据库", "大厂追问", "数据库连接池为什么重要，怎么配置？", "连接池复用已建立的数据库连接，避免每次请求都做 TCP 握手和认证，显著降低延迟和资源消耗。配置要点包括最大连接数、最小空闲连接、连接超时、空闲回收和健康检查。", "连接池设置过大有什么问题？", "每次查询都新建连接，不理解连接池的复用机制。", "推理服务高频查数据库时，连接池大小要匹配并发量，过大会耗尽数据库连接配额。"),
    QuizItem("algo-09", "算法", "基础", "栈和队列的区别和应用场景？", "栈是后进先出（LIFO），适合函数调用栈、括号匹配、表达式求值和 DFS；队列是先进先出（FIFO），适合任务调度、BFS、消息队列和缓冲区。两者都可以用数组或链表实现。", "为什么 BFS 用队列而 DFS 用栈？", "只说一个先进先出一个后进先出，不联系实际应用场景。", "推理任务队列、梯度累积和模型推理流水线都涉及队列和栈的选择。"),
    QuizItem("algo-10", "算法", "高频", "递归和迭代有什么区别，什么时候用哪个？", "递归代码简洁、适合树和分治问题，但有调用栈开销和栈溢出风险；迭代用循环显式管理状态，空间效率更高。尾递归优化可以把部分递归转成迭代，但 Python 不支持尾递归优化。", "递归转迭代通常需要什么数据结构？", "只说递归慢迭代快，不讲调用栈和显式状态管理。", "模型的前向计算图遍历、特征金字塔构建都涉及递归或迭代的选择。"),
    QuizItem("algo-11", "算法", "进阶", "图的最短路径有哪些算法？", "Dijkstra 适合非负权图，时间 O((V+E)logV)；Bellman-Ford 支持负权边但更慢；Floyd-Warshall 求所有点对最短路径 O(V³)；BFS 适合无权图。选择取决于图的规模、边权特性和需要求解的问题。", "Dijkstra 为什么不能处理负权边？", "只背算法名称，不理解适用条件和时间复杂度。", "注意力路径分析、计算图优化和依赖拓扑排序都用到图算法。"),
    QuizItem("algo-12", "算法", "大厂追问", "设计一个 LRU 缓存需要什么数据结构？", "用哈希表实现 O(1) 查找，用双向链表维护访问顺序实现 O(1) 插入和删除。get 时把节点移到链表头，put 超容量时淘汰链表尾。Python 的 OrderedDict 内置了 LRU 语义。", "如果要支持过期时间怎么办？", "只用数组或只用链表，做不到 O(1) 的 get 和 put。", "推理结果缓存、KV Cache 管理和特征缓存都需要 LRU 或类似的淘汰策略。"),
    QuizItem("os-09", "操作系统", "基础", "协程和线程有什么区别？", "协程是用户态的轻量级执行单元，由程序自身调度，切换成本低；线程由 OS 内核调度，切换涉及上下文保存和恢复。协程适合 IO 密集型异步任务，线程适合需要真正并行的场景。", "Python 的 asyncio 和多线程有什么区别？", "把协程理解成更轻的线程，不讲用户态调度和 IO 多路复用。", "异步推理服务、流式数据处理和并发 API 调用常用协程提升吞吐。"),
    QuizItem("os-10", "操作系统", "高频", "什么是系统调用，和库函数有什么区别？", "系统调用是用户程序请求内核服务的接口，涉及用户态到内核态的切换，如文件读写、进程创建；库函数在用户态执行，如 printf 内部最终调用 write 系统调用。系统调用开销大但提供硬件保护和资源隔离。", "为什么频繁系统调用会降低性能？", "把系统调用和普通函数调用混为一谈。", "模型推理中的 IO 操作、日志写入和进程间通信都会触发系统调用，需要控制频率。"),
    QuizItem("os-11", "操作系统", "进阶", "什么是信号（Signal），常见信号有哪些？", "信号是 OS 通知进程发生了异步事件的机制。SIGKILL 强制终止、SIGTERM 优雅终止、SIGINT 是 Ctrl+C、SIGSEGV 是段错误、SIGCHLD 通知子进程退出。进程可以捕获、忽略或执行默认处理。", "SIGKILL 为什么不能被捕获？", "只说信号是中断，不讲信号的产生、传递和处理方式。", "训练进程被 OOM Killer 终止就是 SIGKILL，优雅退出需要捕获 SIGTERM 做 checkpoint 保存。"),
    QuizItem("os-12", "操作系统", "大厂追问", "容器和虚拟机有什么区别？", "虚拟机通过 Hypervisor 模拟完整硬件和 OS，隔离强但资源开销大；容器共享宿主机内核，通过 namespace 隔离进程/网络/文件系统，cgroup 限制资源，启动快、开销小但隔离性弱于虚拟机。", "容器逃逸是怎么发生的？", "把容器理解成轻量虚拟机，不讲共享内核和 namespace。", "模型训练和推理部署常用 Docker 容器，GPU 直通和资源隔离是关键配置。"),
    QuizItem("dl-09", "深度学习", "基础", "什么是学习率预热（Warmup），为什么需要？", "训练初期参数随机初始化，大学习率容易导致梯度爆炸或跑偏。Warmup 先用小学习率线性增长到目标值，让模型稳定进入合理优化区域后再正常衰减。常配合余弦退火或线性衰减使用。", "为什么 Transformer 训练几乎必须用 Warmup？", "直接用大学习率开始训练，不理解初期参数不稳定的危害。", "大模型训练的 Warmup 步数和学习率调度是关键超参，影响最终收敛质量。"),
    QuizItem("dl-10", "深度学习", "高频", "什么是知识蒸馏，怎么用？", "知识蒸馏是用大模型（教师）的软标签指导小模型（学生）学习，软标签包含类别间关系信息，比硬标签更丰富。学生模型通过 KL 散度损失对齐教师输出分布，通常能学到比直接训练更好的泛化能力。", "温度参数 T 在蒸馏中起什么作用？", "只说把大模型变小，不讲软标签、温度缩放和 KL 散度。", "推理部署时常用蒸馏压缩大模型，在保持效果的同时降低延迟和显存占用。"),
    QuizItem("dl-11", "深度学习", "进阶", "LoRA 微调的原理是什么？", "LoRA 在预训练权重旁插入低秩矩阵 A 和 B，只训练这两个小矩阵而冻结原始权重。更新量 ΔW = BA，其中秩 r 远小于原始维度。这样大幅减少可训练参数和显存占用，同时保持接近全量微调的效果。", "LoRA 的秩 r 怎么选？", "把 LoRA 理解成简单地减少参数，不讲低秩分解和旁路结构。", "LoRA 是大模型微调的标准方案，可以为不同任务维护不同的小矩阵而共享大模型权重。"),
    QuizItem("dl-12", "深度学习", "大厂追问", "如何设计一个模型的 A/B 测试？", "先确定评估指标和最小可检测效果，再做流量分割保证随机性和样本量，实验组和对照组同时运行，用统计显著性检验判断差异是否可靠。还要注意新奇效应、长期指标和互斥实验设计。", "为什么不能只看一天的数据就下结论？", "直接对比不同时间段的指标，忽略流量分配和统计检验。", "模型上线前的 A/B 测试是验证效果的金标准，需要和监控、回滚机制配合。"),
    QuizItem("sd-01", "系统设计", "基础", "什么是 CAP 定理？", "CAP 指分布式系统最多同时满足一致性（Consistency）、可用性（Availability）和分区容错性（Partition tolerance）中的两个。网络分区不可避免，所以实际选择是 CP（强一致但可能拒绝请求）或 AP（高可用但可能读到旧数据）。", "为什么大多数系统选择 AP 而不是 CP？", "把 CAP 理解成三选二的静态配置，不理解分区是故障场景而非常态。", "推荐系统和特征存储通常选 AP 最终一致，模型注册中心可能选 CP 保证版本唯一。"),
    QuizItem("sd-02", "系统设计", "基础", "负载均衡有哪些策略？", "常见策略有轮询、加权轮询、最少连接数、一致性哈希和 IP 哈希。四层负载均衡基于 IP/端口转发，速度快；七层负载均衡能解析 HTTP 内容，支持按 URL、Header 分流。一致性哈希在增减节点时只迁移少量数据。", "一致性哈希为什么需要虚拟节点？", "只说轮询分发，不讲健康检查、会话粘性和后端权重。", "推理服务多实例部署时，负载均衡策略影响延迟均匀性和 GPU 利用率。"),
    QuizItem("sd-03", "系统设计", "高频", "消息队列解决什么问题？", "消息队列实现异步解耦和削峰填谷。生产者把消息发到队列就返回，消费者按自己的速率处理。好处包括：服务解耦、流量削峰、广播分发、重试和死信处理。常见实现有 Kafka、RabbitMQ、Redis Stream。", "Kafka 和 RabbitMQ 的核心区别是什么？", "只说解耦和异步，不讲消费模型、持久化和吞吐差异。", "模型推理请求接入消息队列可以平滑突发流量，训练日志采集常用 Kafka 做高吞吐管道。"),
    QuizItem("sd-04", "系统设计", "高频", "什么是缓存穿透、击穿和雪崩？", "穿透是查询不存在的数据，缓存永远不命中，请求直达数据库；击穿是热点 key 过期瞬间大量并发请求打到数据库；雪崩是大量 key 同时过期或缓存服务宕机，请求全部回落到数据库。解决方案分别是布隆过滤器/空值缓存、互斥锁/不过期、随机过期时间/多级缓存。", "缓存和数据库的双写一致性怎么保证？", "只知道加缓存，不区分三种异常场景和各自对策。", "推理结果缓存、模型配置缓存和用户画像缓存都需要考虑这些边界场景。"),
    QuizItem("sd-05", "系统设计", "高频", "数据库分库分表的策略和代价是什么？", "垂直分库按业务拆，水平分表按某字段哈希或范围拆分。好处是单表数据量可控、查询快；代价是跨分片查询复杂、分布式事务困难、扩容需要数据迁移。分片键选择直接影响数据分布均匀性和查询路由效率。", "分库分表后怎么做全局唯一 ID？", "只说拆表提速，不讲跨分片 JOIN、事务和扩容问题。", "实验记录表按用户 ID 分片后，查某个用户所有实验很快，但跨用户统计需要聚合。"),
    QuizItem("sd-06", "系统设计", "高频", "什么是服务熔断和降级？", "熔断是当下游服务错误率超过阈值时自动切断请求，防止故障蔓延，类似电路保险丝。降级是在系统压力大时主动关闭非核心功能，保证核心链路可用。熔断器通常有关闭、打开、半开三种状态，半开状态放少量请求探测下游恢复情况。", "熔断和限流有什么区别？", "把熔断理解成简单的超时重试，不讲状态机和故障探测。", "推理服务依赖的模型加载、特征获取和后处理链路都需要熔断保护，避免级联故障。"),
    QuizItem("sd-07", "系统设计", "进阶", "如何设计一个短链接服务？", "核心是长短映射：用自增 ID 或分布式 ID 生成器得到唯一数字，再 Base62 编码成短码。存储用 MySQL 或 Redis 做映射表。读多写少场景适合加缓存。302 重定向比 301 灵活，可以统计点击量。自增 ID 要避免可预测性带来的安全风险。", "为什么用 Base62 而不是 Base64？", "不考虑分布式 ID 生成和高并发写入的瓶颈。", "短链接服务是系统设计经典题，涉及缓存、数据库选型、分布式 ID 和重定向策略。"),
    QuizItem("sd-08", "系统设计", "进阶", "如何设计一个推荐系统的在线架构？", "典型分三层：召回从海量候选中快速筛出几百个，排序用精排模型打分选出几十个，重排考虑多样性和业务规则输出最终列表。召回常用双塔模型和向量检索，排序用特征丰富的精排模型。特征服务要低延迟，通常用 Redis 或特征平台。", "召回和排序为什么要分开？", "直接对全量候选做精排，不理解计算量和延迟的矛盾。", "推荐系统架构是深度学习工程化的核心场景，召回/排序/重排分层是行业标准。"),
    QuizItem("sd-09", "系统设计", "进阶", "分布式训练中数据并行和模型并行有什么区别？", "数据并行把数据切分到多个 GPU，每个 GPU 持有完整模型副本，梯度通过 AllReduce 同步；模型并行把模型切分到多个 GPU，每个 GPU 只存一部分参数。流水线并行是模型并行的变体，把不同层放到不同 GPU。大模型训练通常混合使用。", "AllReduce 的通信开销怎么优化？", "只说多 GPU 一起训练，不区分数据切分和模型切分。", "千亿参数模型训练必须用混合并行，通信拓扑和梯度压缩直接影响训练效率。"),
    QuizItem("sd-10", "系统设计", "大厂追问", "如何设计一个模型推理服务平台？", "核心组件包括：模型仓库管理版本和灰度，调度器根据请求特征路由到合适模型和 GPU 实例，推理引擎（TensorRT/vLLM）做高效计算，监控收集延迟、吞吐和 GPU 利用率。还要支持 A/B 流量分配、自动扩缩容和优雅发布。", "怎么处理推理服务的长尾延迟？", "只关注平均延迟，不考虑 P99 和排队效应。", "推理平台是 MLOps 的核心产品，涉及模型管理、资源调度、流量控制和可观测性。"),
    QuizItem("sd-11", "系统设计", "大厂追问", "什么是幂等性，为什么 API 设计需要它？", "幂等是指同一请求执行多次和执行一次效果相同。GET、PUT、DELETE 天然幂等，POST 不幂等需要通过唯一请求 ID 或去重表实现。网络超时重试时幂等性避免重复创建订单或重复扣款。", "怎么给一个 POST 接口实现幂等？", "只说重试安全，不讲去重机制和状态机。", "模型训练任务提交、推理请求重试和配置变更都需要幂等设计。"),
    QuizItem("sd-12", "系统设计", "大厂追问", "如何设计一个实时特征计算平台？", "离线特征用批处理（Spark）预先计算存入特征库，在线特征用流处理（Flink）实时更新。特征存储统一管理版本和一致性，避免训练和推理用不同特征。关键挑战是低延迟读取、特征穿越（避免用到未来数据）和特征复用。", "训练和推理的特征不一致会怎样？", "只关注模型效果，不考虑特征工程的线上线下一致性。", "特征平台是推荐和风控系统的核心基础设施，特征穿越是常见 bug 来源。"),
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
    st.session_state.setdefault("interview_question_start_time", 0.0)
    st.session_state.setdefault("interview_answer_times", [])
    st.session_state.setdefault("interview_results", [])
    st.session_state.setdefault("interview_auto_score", None)
    st.session_state.setdefault("interview_auto_saved_qid", "")
    st.session_state.setdefault("interview_simulation_index", 0)
    st.session_state.setdefault("interview_simulation_scores", [])
    st.session_state.setdefault("interview_simulation_role", "算法工程岗")


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
    st.session_state["interview_auto_score"] = None
    st.session_state["interview_auto_saved_qid"] = ""
    st.session_state["interview_question_start_time"] = time.time()


def current_question(candidates: list[QuizItem]) -> QuizItem | None:
    qid = st.session_state.get("interview_current_qid", "")
    by_id = {item.qid: item for item in candidates}
    if qid in by_id:
        return by_id[qid]
    if candidates:
        st.session_state["interview_current_qid"] = candidates[0].qid
        # 首次加载时也记录开始时间
        if st.session_state.get("interview_question_start_time", 0.0) == 0.0:
            st.session_state["interview_question_start_time"] = time.time()
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


def persist_interview_record(item: QuizItem, outcome: str, user_answer: str, score: dict[str, object] | None = None) -> None:
    """把面试错题和复习记录写入全站学习档案。"""

    try:
        from components.progress_tracker import add_learning_record, mark_review_later

        module_key = QUESTION_MODULE_MAP.get(item.direction, "part7/interview_quiz")
        title = f"{item.direction}｜{item.question}"
        score_text = ""
        if score:
            score_text = f"\n自动评分：{score.get('score', 0)} 分；命中要点：{', '.join(score.get('matched', [])) or '暂无'}"
        note = (
            f"结果：{outcome}\n"
            f"题目：{item.question}\n"
            f"我的答案：{user_answer.strip() or '未填写'}\n"
            f"标准答案：{item.answer}\n"
            f"面试官追问：{item.follow_up}"
            f"{score_text}"
        )
        reflection = f"下一次回答要避开：{item.trap}。工程连接：{item.application}"
        record_type = "错题" if outcome in {"答错", "稍后复习", "自动评分偏低"} else "学习笔记"
        add_learning_record(module_key, record_type, title, note, reflection)
        if outcome == "稍后复习":
            mark_review_later(module_key, reason=f"面试题待复习：{item.question}", priority="高")
    except Exception:
        pass


def record_interview_result(item: QuizItem, correct: bool) -> None:
    """记录本轮每次判题结果，用于后续统计分析。"""
    answer_times = st.session_state.get("interview_answer_times", [])
    elapsed = 0.0
    if st.session_state.get("interview_user_answer_visible", False) and answer_times:
        elapsed = answer_times[-1]
    if elapsed <= 0:
        start = st.session_state.get("interview_question_start_time", 0.0)
        if start > 0:
            elapsed = time.time() - start
            st.session_state["interview_answer_times"].append(elapsed)

    results = list(st.session_state.get("interview_results", []))
    results.append(
        {
            "qid": item.qid,
            "direction": item.direction,
            "difficulty": item.difficulty,
            "correct": correct,
            "elapsed": float(elapsed),
        }
    )
    st.session_state["interview_results"] = results


def answer_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[，。；：、！？,.!?;:()\[\]{}<>\"'“”‘’/\\|-]", " ", text.lower())
    raw = {token.strip() for token in cleaned.split() if len(token.strip()) >= 2}
    chinese_chunks = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    # 对中文长句做短片段切分，避免必须整句完全命中。
    chunks: set[str] = set()
    for chunk in chinese_chunks:
        if len(chunk) <= 4:
            chunks.add(chunk)
        else:
            chunks.update(chunk[index : index + 2] for index in range(0, len(chunk) - 1))
            chunks.update(chunk[index : index + 3] for index in range(0, len(chunk) - 2))
    return raw | chunks


def score_user_answer(item: QuizItem, user_answer: str) -> dict[str, object]:
    """用关键词覆盖、工程连接和结构信号给口述答案做轻量自动评分。"""

    user_tokens = answer_tokens(user_answer)
    reference_tokens = answer_tokens(item.answer)
    engineering_tokens = answer_tokens(item.application)
    essential_tokens = [
        token
        for token, count in Counter(reference_tokens).items()
        if len(token) >= 2 and count >= 1
    ][:28]
    matched = [token for token in essential_tokens if token in user_tokens]
    coverage = len(matched) / max(1, len(essential_tokens))

    structure_hits = sum(marker in user_answer for marker in ("首先", "其次", "最后", "因为", "所以", "但是", "取舍", "如果"))
    engineering_hits = sum(term in user_answer for term in ("项目", "工程", "服务", "训练", "推理", "线上", "延迟", "吞吐", "日志", "监控"))
    engineering_coverage = len([token for token in engineering_tokens if token in user_tokens]) / max(1, len(engineering_tokens))
    length_score = min(1.0, len(user_answer.strip()) / 120)

    dimension_scores = {
        "定义准确": min(30, round(coverage * 38)),
        "机制完整": min(30, round((coverage * 0.8 + length_score * 0.2) * 33)),
        "工程落地": min(25, 8 + engineering_hits * 5 + round(engineering_coverage * 8)) if engineering_hits else round(coverage * 12),
        "表达结构": min(15, 5 + structure_hits * 3) if structure_hits else round(length_score * 8),
    }
    total = int(sum(dimension_scores.values()))
    missing = [token for token in essential_tokens if token not in user_tokens][:8]
    if not user_answer.strip():
        total = 0
        dimension_scores = {name: 0 for name, _, _ in SCORING_DIMENSIONS}
        missing = essential_tokens[:8]
    return {
        "score": min(100, total),
        "dimension_scores": dimension_scores,
        "matched": matched[:10],
        "missing": missing,
        "advice": score_advice(total),
    }


def score_advice(score: int) -> str:
    if score >= 85:
        return "表达已经接近面试可用，下一步练追问和极端场景。"
    if score >= 70:
        return "主干正确，但还需要补边界条件、复杂度或工程取舍。"
    if score >= 50:
        return "有部分关键词，但答案结构不稳，建议按“定义-机制-问题-工程场景”重答一遍。"
    return "当前更像碎片记忆，建议先看标准答案，再把遗漏要点写进错题本。"


def render_score_card(score: dict[str, object]) -> None:
    st.markdown("**自动评分**")
    st.progress(int(score["score"]) / 100)
    st.metric("综合分", f"{score['score']} / 100")
    rows = [
        {"维度": name, "得分": score["dimension_scores"].get(name, 0), "满分": full, "评分依据": desc}
        for name, desc, full in SCORING_DIMENSIONS
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    matched = "、".join(score.get("matched", [])) or "暂无明显命中"
    missing = "、".join(score.get("missing", [])) or "暂无明显遗漏"
    st.caption(f"命中要点：{matched}")
    st.caption(f"建议补充：{missing}")
    st.info(str(score["advice"]))


def extract_key_points(answer: str) -> list[str]:
    """从标准答案中提取关键要点。"""
    parts = [s.strip() for s in answer.replace("；", "。").replace("；", "。").split("。") if s.strip()]
    return parts[:5]


def format_elapsed(seconds: float) -> str:
    """格式化用时显示。"""
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes} 分 {secs} 秒"


def render_question_bank_overview() -> None:
    """展示题库总览：方向×难度分布热力图和覆盖统计。"""
    st.subheader("📊 题库总览")
    st.caption("查看全部题目的方向和难度分布，找到薄弱环节有针对性地练习。")

    direction_order = ["网络", "数据库", "算法", "操作系统", "深度学习", "系统设计"]
    difficulty_order = ["基础", "高频", "进阶", "大厂追问"]

    # 构建方向×难度矩阵
    matrix = {}
    for d in direction_order:
        matrix[d] = {}
        for diff in difficulty_order:
            matrix[d][diff] = sum(
                1 for q in QUESTIONS if q.direction == d and q.difficulty == diff
            )

    # 热力图
    z_values = [[matrix[d][diff] for diff in difficulty_order] for d in direction_order]
    text_values = [[str(v) if v > 0 else "" for v in row] for row in z_values]

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=difficulty_order,
            y=direction_order,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#172026"},
            colorscale=[
                [0.0, "#f7f8f4"],
                [0.2, "#d4ede8"],
                [0.5, "#7ecac0"],
                [1.0, "#0f8b8d"],
            ],
            showscale=True,
            colorbar=dict(title="题数", len=0.6),
            hovertemplate="%{y} · %{x}<br>题数: %{z}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": False})

    # 方向汇总条形图 + 难度汇总
    col_chart, col_stats = st.columns([0.6, 0.4])
    with col_chart:
        dir_counts = [sum(matrix[d].values()) for d in direction_order]
        fig_bar = go.Figure(
            data=go.Bar(
                y=direction_order,
                x=dir_counts,
                orientation="h",
                marker_color=["#0f8b8d", "#3268a8", "#c4871f", "#3f7d58", "#7353ba", "#bf3f5b"],
                text=dir_counts,
                textposition="auto",
                hovertemplate="%{y}: %{x} 题<extra></extra>",
            )
        )
        fig_bar.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="题数",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_stats:
        total = len(QUESTIONS)
        st.metric("题库总量", f"{total} 题")
        st.metric("覆盖方向", f"{len(direction_order)} 个")
        st.metric("难度梯度", f"{len(difficulty_order)} 级")

        # 覆盖最薄弱的方向
        min_dir = min(direction_order, key=lambda d: sum(matrix[d].values()))
        min_count = sum(matrix[min_dir].values())
        st.caption(f"题量最少方向：{min_dir}（{min_count} 题）")

        # 难度分布
        diff_totals = {diff: sum(matrix[d][diff] for d in direction_order) for diff in difficulty_order}
        diff_text = " · ".join(f"{diff} {diff_totals[diff]}" for diff in difficulty_order)
        st.caption(f"难度分布：{diff_text}")

    st.markdown("---")


def render_practice_analysis() -> None:
    st.subheader("本轮练题统计分析")
    results = st.session_state.get("interview_results", [])
    if not results:
        st.info("本轮还没有答题记录。点击“我答对了”或“我答错了”后，这里会展示方向、难度和用时分析。")
        return

    df = pd.DataFrame(results)
    chart_config = {"displayModeBar": False, "responsive": True}
    colors = {
        "答对": "#0f8b8d",
        "答错": "#bf3f5b",
        "总数": "#3268a8",
        "正确率": "#c4871f",
        "用时": "#3268a8",
    }

    direction_order = ["网络", "数据库", "算法", "操作系统", "深度学习", "系统设计"]
    direction_summary = (
        df.groupby("direction", as_index=False)
        .agg(答对=("correct", "sum"), 总数=("qid", "count"))
        .assign(答错=lambda data: data["总数"] - data["答对"])
    )
    direction_summary = (
        direction_summary.set_index("direction")
        .reindex(direction_order, fill_value=0)
        .reset_index()
        .rename(columns={"index": "direction"})
    )
    direction_summary[["答对", "总数", "答错"]] = direction_summary[["答对", "总数", "答错"]].astype(int)

    fig_direction = go.Figure()
    for name in ["答对", "答错", "总数"]:
        fig_direction.add_trace(
            go.Bar(
                y=direction_summary["direction"],
                x=direction_summary[name],
                name=name,
                orientation="h",
                marker_color=colors[name],
                text=direction_summary[name],
                textposition="auto",
            )
        )
    fig_direction.update_layout(
        height=320,
        barmode="group",
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="题数",
        yaxis_title="方向",
    )
    st.plotly_chart(fig_direction, use_container_width=True, config=chart_config)

    difficulty_order = ["基础", "高频", "进阶", "大厂追问"]
    difficulty_summary = (
        df.groupby("difficulty", as_index=False)
        .agg(答对=("correct", "sum"), 总数=("qid", "count"))
    )
    difficulty_summary = (
        difficulty_summary.set_index("difficulty")
        .reindex(difficulty_order, fill_value=0)
        .reset_index()
        .rename(columns={"index": "difficulty"})
    )
    difficulty_summary[["答对", "总数"]] = difficulty_summary[["答对", "总数"]].astype(int)
    difficulty_summary["正确率"] = difficulty_summary.apply(
        lambda row: row["答对"] / row["总数"] * 100 if row["总数"] else 0,
        axis=1,
    )

    fig_difficulty = px.bar(
        difficulty_summary,
        x="正确率",
        y="difficulty",
        orientation="h",
        text=difficulty_summary["正确率"].map(lambda value: f"{value:.0f}%"),
        color_discrete_sequence=[colors["正确率"]],
    )
    fig_difficulty.update_traces(hovertemplate="%{y}<br>正确率 %{x:.1f}%<extra></extra>")
    fig_difficulty.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="正确率",
        yaxis_title="难度",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        showlegend=False,
    )
    st.plotly_chart(fig_difficulty, use_container_width=True, config=chart_config)

    time_df = df.reset_index().rename(columns={"index": "题序"})
    time_df["题序"] = time_df["题序"] + 1
    fig_time = px.line(
        time_df,
        x="题序",
        y="elapsed",
        markers=True,
        color_discrete_sequence=[colors["用时"]],
        hover_data={"qid": True, "direction": True, "difficulty": True, "elapsed": ":.1f"},
    )
    fig_time.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="答题顺序",
        yaxis_title="用时（秒）",
        showlegend=False,
    )
    st.plotly_chart(fig_time, use_container_width=True, config=chart_config)

    answered_directions = direction_summary[direction_summary["总数"] > 0].copy()
    answered_directions["正确率"] = answered_directions["答对"] / answered_directions["总数"] * 100
    weakest = answered_directions.sort_values(["正确率", "总数"], ascending=[True, False]).iloc[0]
    st.info(
        f"薄弱方向提示：当前正确率最低的是 {weakest['direction']}，正确率 {weakest['正确率']:.0f}% "
        f"（答对 {int(weakest['答对'])} / 共 {int(weakest['总数'])} 题）。建议下一轮优先筛选该方向，"
        "先复盘错题本里的遗漏点，再连续练 3-5 道同方向题巩固表达。"
    )


def simulation_candidates(role: str) -> list[QuizItem]:
    directions = SIMULATION_SCRIPTS[role]
    selected: list[QuizItem] = []
    for direction in directions:
        pool = [item for item in QUESTIONS if item.direction == direction and item.difficulty in {"高频", "进阶", "大厂追问"}]
        if not pool:
            pool = [item for item in QUESTIONS if item.direction == direction]
        if pool:
            selected.append(pool[date_seed(len(selected)) % len(pool)])
    return selected


def date_seed(offset: int) -> int:
    return int(time.strftime("%Y%m%d")) + offset * 17


def start_simulation(role: str) -> None:
    st.session_state["interview_simulation_role"] = role
    st.session_state["interview_simulation_index"] = 0
    st.session_state["interview_simulation_scores"] = []
    candidates = simulation_candidates(role)
    if candidates:
        st.session_state["interview_current_qid"] = candidates[0].qid
        st.session_state["interview_user_answer_visible"] = False
        st.session_state["interview_user_answer"] = ""
        st.session_state["interview_auto_score"] = None
        st.session_state["interview_auto_saved_qid"] = ""
        st.session_state["interview_question_start_time"] = time.time()


def render_simulation_panel() -> None:
    st.subheader("模拟面试流程")
    role = st.selectbox("选择岗位脚本", list(SIMULATION_SCRIPTS), key="simulation-role-select")
    candidates = simulation_candidates(role)
    index = st.session_state.get("interview_simulation_index", 0)
    scores = st.session_state.get("interview_simulation_scores", [])
    cols = st.columns(3)
    if cols[0].button("开始模拟面试", key="simulation-start", use_container_width=True):
        start_simulation(role)
        st.rerun()
    cols[1].metric("当前轮次", f"{min(index + 1, len(candidates))} / {len(candidates)}")
    avg_score = sum(scores) / len(scores) if scores else 0
    cols[2].metric("模拟均分", f"{avg_score:.0f}")

    st.caption("流程按岗位自动串联 5 个方向：先回答，再自动评分；低于 70 分会自动写入错题档案。")
    if scores and len(scores) >= len(candidates):
        st.success("本轮模拟面试完成。建议复盘低分题，再按同岗位脚本重来一轮。")
    st.markdown("面试路径：" + " → ".join(SIMULATION_SCRIPTS[role]))


def advance_simulation_if_needed(item: QuizItem, score: dict[str, object]) -> bool:
    role = st.session_state.get("interview_simulation_role", "算法工程岗")
    candidates = simulation_candidates(role)
    qids = [candidate.qid for candidate in candidates]
    if item.qid not in qids:
        return False
    scores = list(st.session_state.get("interview_simulation_scores", []))
    scores.append(int(score["score"]))
    st.session_state["interview_simulation_scores"] = scores
    next_index = qids.index(item.qid) + 1
    st.session_state["interview_simulation_index"] = next_index
    if next_index < len(candidates):
        st.session_state["interview_current_qid"] = candidates[next_index].qid
        st.session_state["interview_user_answer_visible"] = False
        st.session_state["interview_user_answer"] = ""
        st.session_state["interview_auto_score"] = None
        st.session_state["interview_auto_saved_qid"] = ""
        st.session_state["interview_question_start_time"] = time.time()
    return True


def main() -> None:
    ensure_state()
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>CS 面试刷题模式</h1>
          <p>随机出题、按方向和难度筛选，支持先自己作答再对照标准答案，附带要点自查清单和答题计时，适合高频八股的快速口述训练。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.link_button("← 返回主界面", "/", width="small")
    st.markdown("")

    render_simulation_panel()
    render_question_bank_overview()

    left, right = st.columns([0.3, 0.7])
    with left:
        direction = st.selectbox("按方向筛选", ["全部", "网络", "数据库", "算法", "操作系统", "深度学习", "系统设计"], key="quiz-direction")
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

        # 平均用时统计
        answer_times = st.session_state.get("interview_answer_times", [])
        if answer_times:
            avg_time = sum(answer_times) / len(answer_times)
            st.metric("平均用时", format_elapsed(avg_time))
            st.caption(f"共作答 {len(answer_times)} 次")

        if st.button("随机出题", key="quiz-random", use_container_width=True):
            pick_question(candidates)
        if st.button("清空错题本", key="quiz-clear", use_container_width=True):
            st.session_state["interview_wrong_book"] = []
            st.session_state["interview_later_book"] = []
            st.session_state["interview_answered_count"] = 0
            st.session_state["interview_correct_count"] = 0
            st.session_state["interview_answer_times"] = []
            st.session_state["interview_results"] = []
            st.session_state["interview_auto_score"] = None
            st.session_state["interview_auto_saved_qid"] = ""
            st.session_state["interview_simulation_scores"] = []
            st.session_state["interview_simulation_index"] = 0

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
                    score = score_user_answer(item, user_answer)
                    st.session_state["interview_auto_score"] = score
                    if int(score["score"]) < 70 and st.session_state.get("interview_auto_saved_qid") != item.qid:
                        add_unique("interview_wrong_book", item)
                        persist_interview_record(item, "自动评分偏低", user_answer, score)
                        st.session_state["interview_auto_saved_qid"] = item.qid
                    # 记录用时
                    start = st.session_state.get("interview_question_start_time", 0.0)
                    if start > 0:
                        elapsed = time.time() - start
                        st.session_state["interview_answer_times"].append(elapsed)
                    st.rerun()
            else:
                # 显示用户答案
                user_answer = st.session_state.get("interview_user_answer", "")
                if user_answer.strip():
                    st.markdown("**你的答案：**")
                    st.info(user_answer)

            # 标准答案和追问（仅在提交后显示）
            if st.session_state.get("interview_user_answer_visible", False):
                # 显示用时
                answer_times = st.session_state.get("interview_answer_times", [])
                if answer_times:
                    last_time = answer_times[-1]
                    st.caption(f"⏱️ 用时 {format_elapsed(last_time)}")

                score = st.session_state.get("interview_auto_score")
                if score:
                    render_score_card(score)
                    if int(score["score"]) < 70:
                        st.warning("自动评分低于 70 分，建议点击“我答错了”把它写入错题档案。")

                with st.expander("标准答案", expanded=True):
                    st.write(item.answer)

                # 要点自查清单
                key_points = extract_key_points(item.answer)
                if key_points:
                    st.markdown("**📋 要点自查清单**")
                    for i, point in enumerate(key_points, 1):
                        st.markdown(f"- [ ] {point}")

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
                    record_interview_result(item, True)
                    score = st.session_state.get("interview_auto_score") or score_user_answer(item, st.session_state.get("interview_user_answer", ""))
                    persist_interview_record(item, "答对", st.session_state.get("interview_user_answer", ""), score)
                    if not advance_simulation_if_needed(item, score):
                        pick_question(candidates)
                    st.rerun()
            with c2:
                if st.button("我答错了", key="quiz-wrong", use_container_width=True):
                    st.session_state["interview_answered_count"] = st.session_state.get("interview_answered_count", 0) + 1
                    record_interview_result(item, False)
                    add_unique("interview_wrong_book", item)
                    score = st.session_state.get("interview_auto_score") or score_user_answer(item, st.session_state.get("interview_user_answer", ""))
                    persist_interview_record(item, "答错", st.session_state.get("interview_user_answer", ""), score)
                    if not advance_simulation_if_needed(item, score):
                        pick_question(candidates)
                    st.rerun()
            with c3:
                if st.button("稍后复习", key="quiz-later", use_container_width=True):
                    add_unique("interview_later_book", item)
                    score = st.session_state.get("interview_auto_score") or score_user_answer(item, st.session_state.get("interview_user_answer", ""))
                    persist_interview_record(item, "稍后复习", st.session_state.get("interview_user_answer", ""), score)
                    if not advance_simulation_if_needed(item, score):
                        pick_question(candidates)
                    st.rerun()

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

    st.caption("错题和稍后复习会同时写入全站学习档案，回到主界面学习报告可以看到长期记录。")

    st.markdown(
        """
        <div class="note">
        训练建议：先在输入框里写下你的答案，再提交对照标准答案。真正面试时要主动给出边界条件、工程取舍和排查路径，
        不要只背一句定义。对比自己的答案和标准答案，找出遗漏的关键点。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_practice_analysis()

    # 知识图谱导航：展示推荐学习路径
    try:
        from components.knowledge_graph import KNOWLEDGE_GRAPH

        st.subheader("📐 推荐学习路径")
        st.caption("面试八股覆盖的知识点在深度学习书库中都有对应的交互式讲解模块，点击可跳转学习。")
        path_keys = [
            "math_primer",
            "tensors_gradients",
            "neural_network_basics",
            "convolution_visual",
            "cnn_architectures",
            "rnn_intuition",
            "sequence_models",
            "attention_mechanism",
            "transformer_models",
            "training_dynamics",
            "gradient_monitor",
        ]
        cols = st.columns(4)
        for idx, key in enumerate(path_keys):
            node = KNOWLEDGE_GRAPH.get(key)
            if not node:
                continue
            with cols[idx % 4]:
                st.markdown(f"**{idx + 1}. {node.title}**")
                st.caption(f"{node.description[:40]}…")
    except Exception:
        pass

    render_back_home()


if __name__ == "__main__":
    safe_run(main)
