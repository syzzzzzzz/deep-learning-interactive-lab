"""
CS interview page: system design.

Run:
    streamlit run part7_interview/system_design.py
or:
    python main.py part7/system_design
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import streamlit as st


T = TypeVar("T")

st.set_page_config(page_title="系统设计面试训练", layout="wide", initial_sidebar_state="expanded")


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
    .kv-table { width: 100%; border-collapse: collapse; margin: .6rem 0; }
    .kv-table th, .kv-table td { padding: .5rem .7rem; border: 1px solid #d8dee3; text-align: left; }
    .kv-table th { background: #eef1f4; font-weight: 700; }
    .kv-table td { background: rgba(255,255,255,.75); }
    .layer-box { border: 2px solid #0f8b8d; border-radius: 8px; padding: .6rem .8rem; text-align: center; background: rgba(15,139,141,.06); margin: .3rem 0; }
    .layer-box strong { display: block; font-size: 1.05rem; }
    </style>
    """


def safe_run(func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        st.error("系统设计页面执行出错，已进入兜底视图。")
        st.warning("请返回主界面后重新进入；如果仍然失败，请查看下方错误信息。")
        with st.expander("错误详情", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        render_back_home()
        return None


def render_back_home() -> None:
    if st.button("返回主界面", key="sd-back-home", use_container_width=True):
        st.query_params.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Section 1: CAP 定理交互
# ---------------------------------------------------------------------------

def cap_section() -> None:
    st.subheader("CAP 定理交互理解")
    st.markdown(
        '<div class="note">CAP 定理：分布式系统在网络分区（Partition）发生时，最多只能同时保证一致性（Consistency）和可用性（Availability）中的一个。</div>',
        unsafe_allow_html=True,
    )
    choice = st.radio(
        "网络分区发生时，你的系统优先保什么？",
        ["一致性（CP）— 拒绝不一致的请求", "可用性（AP）— 允许短暂读到旧数据"],
        key="cap-choice",
        horizontal=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="step"><strong>一致性（C）</strong><br>每次读都能读到最新写入的值。<br><span class="small">代价：分区时可能拒绝请求，牺牲可用性。</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="step"><strong>可用性（A）</strong><br>每个请求都能收到响应（不保证是最新的）。<br><span class="small">代价：分区时可能读到旧数据，牺牲一致性。</span></div>', unsafe_allow_html=True)

    if "CP" in choice:
        st.success(
            "**CP 选择**：拒绝不一致请求，保证数据正确。"
            " 适用场景：银行转账、库存扣减、模型注册中心（版本必须唯一）。"
            " 典型系统：ZooKeeper、etcd、HBase。"
        )
    else:
        st.info(
            "**AP 选择**：优先响应，允许短暂不一致（最终一致性）。"
            " 适用场景：社交 Feed、推荐系统、CDN、用户画像缓存。"
            " 典型系统：Cassandra、DynamoDB、CouchDB。"
        )
    st.markdown(
        '<div class="small">面试要点：不要把 CAP 理解为"三选二的静态配置"。分区是故障场景而非常态，正常运行时 C 和 A 都能满足。关键是分区发生时的取舍策略。</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 2: 分层架构交互
# ---------------------------------------------------------------------------

def layered_architecture() -> None:
    st.subheader("系统设计通用分层架构")
    st.markdown("面试中几乎所有系统设计题都可以用以下分层框架拆解：")
    layers = [
        ("🌐 接入层", "客户端 + 负载均衡", "Nginx/LB 做流量分发、限流、TLS 终结"),
        ("⚙️ 服务层", "业务逻辑 + API Gateway", "微服务拆分、认证鉴权、请求路由"),
        ("📊 数据层", "缓存 + 数据库 + 消息队列", "Redis 缓存、MySQL/PostgreSQL 持久化、Kafka 异步"),
        ("💾 存储层", "对象存储 + 文件系统", "S3/OSS 存图片模型、HDFS 存训练数据"),
        ("📈 监控层", "日志 + 指标 + 链路追踪", "Prometheus + Grafana + Jaeger"),
    ]
    for emoji_title, subtitle, detail in layers:
        st.markdown(
            f'<div class="layer-box"><strong>{emoji_title}</strong>{subtitle}<br><span class="small">{detail}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="note">面试技巧：先画分层架构，再逐层细化。面试官最想看的是你的拆解思路，而不是直接背一个现成方案。</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 3: 缓存三兄弟交互
# ---------------------------------------------------------------------------

def cache_problems() -> None:
    st.subheader("缓存穿透、击穿、雪崩")
    problem = st.selectbox(
        "选择要了解的缓存问题",
        ["缓存穿透", "缓存击穿", "缓存雪崩"],
        key="cache-problem",
    )
    if problem == "缓存穿透":
        st.markdown(
            '<div class="step"><strong>缓存穿透</strong><br>'
            '查询的数据在缓存和数据库中都不存在，请求每次都穿透缓存直达数据库。<br><br>'
            '<strong>攻击场景：</strong>恶意用户用随机 ID 大量请求不存在的资源。<br>'
            '<strong>解决方案：</strong><br>'
            '① 布隆过滤器：在缓存前加一层，快速判断 key 是否可能存在<br>'
            '② 空值缓存：对不存在的 key 也缓存一个空值，设置较短 TTL<br>'
            '③ 参数校验：拦截明显非法的请求参数</div>',
            unsafe_allow_html=True,
        )
    elif problem == "缓存击穿":
        st.markdown(
            '<div class="step"><strong>缓存击穿</strong><br>'
            '某个热点 key 过期瞬间，大量并发请求同时打到数据库。<br><br>'
            '<strong>触发条件：</strong>热点数据过期 + 高并发<br>'
            '<strong>解决方案：</strong><br>'
            '① 互斥锁（singleflight / mutex）：只放一个请求去查 DB，其他等待<br>'
            '② 逻辑过期：缓存不设 TTL，由业务层判断是否需要异步刷新<br>'
            '③ 热点数据永不过期 + 后台定时刷新</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="step"><strong>缓存雪崩</strong><br>'
            '大量 key 同时过期，或缓存服务宕机，请求全部回落到数据库。<br><br>'
            '<strong>触发条件：</strong>批量 key 同时过期 / Redis 整体故障<br>'
            '<strong>解决方案：</strong><br>'
            '① 随机过期时间：TTL = base + random(0, jitter)，打散过期时间<br>'
            '② 多级缓存：L1 本地缓存 + L2 Redis，L2 挂了还有 L1 兜底<br>'
            '③ 熔断降级：缓存不可用时直接拒绝非核心请求，保护数据库<br>'
            '④ Redis 高可用：哨兵或集群模式避免单点故障</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 4: 消息队列交互
# ---------------------------------------------------------------------------

def message_queue_section() -> None:
    st.subheader("消息队列：异步解耦与削峰填谷")
    mq = st.selectbox(
        "选择消息队列",
        ["Kafka", "RabbitMQ", "Redis Stream"],
        key="mq-choice",
    )
    descriptions = {
        "Kafka": {
            "模型": "发布-订阅，基于分区的日志追加",
            "吞吐": "极高（百万级 TPS），适合大数据管道",
            "持久化": "消息持久化到磁盘，支持回溯消费",
            "场景": "日志采集、流处理、事件溯源、训练指标管道",
        },
        "RabbitMQ": {
            "模型": "AMQP 协议，支持复杂路由（exchange + binding）",
            "吞吐": "中等（万级 TPS），适合业务消息",
            "持久化": "支持消息持久化和确认机制",
            "场景": "任务队列、订单处理、邮件通知、异步推理请求",
        },
        "Redis Stream": {
            "模型": "轻量级消息队列，基于 Redis 数据结构",
            "吞吐": "高（十万级 TPS），但受限于内存",
            "持久化": "依赖 Redis RDB/AOF 持久化",
            "场景": "实时通知、轻量任务分发、小规模事件驱动",
        },
    }
    info = descriptions[mq]
    for k, v in info.items():
        st.markdown(f"**{k}**：{v}")

    st.markdown(
        '<div class="note">面试要点：Kafka vs RabbitMQ 的核心区别——Kafka 是日志模型（消费者自己管理 offset），RabbitMQ 是队列模型（消息被消费后删除）。选型要看吞吐、消息语义和运维成本。</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 5: 推荐系统架构
# ---------------------------------------------------------------------------

def recommendation_architecture() -> None:
    st.subheader("推荐系统在线架构")
    st.markdown("典型推荐系统分为三层：召回 → 排序 → 重排")
    cols = st.columns([1, 0.3, 1, 0.3, 1])
    with cols[0]:
        st.markdown(
            '<div class="step"><strong>🔍 召回层</strong><br>'
            '从百万候选中快速筛出几百个<br><br>'
            '<span class="small">• 双塔模型（user/item embedding）<br>'
            '• 向量检索（FAISS / Milvus）<br>'
            '• 多路召回（协同过滤 + 热门 + 关注）</span></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            '<div class="step"><strong>📊 排序层</strong><br>'
            '精排模型对候选打分，选出 Top N<br><br>'
            '<span class="small">• 特征丰富的精排模型<br>'
            '• DeepFM / DCN / 多任务模型<br>'
            '• 特征服务低延迟读取</span></div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown(
            '<div class="step"><strong>🎯 重排层</strong><br>'
            '考虑多样性和业务规则<br><br>'
            '<span class="small">• 去重、打散同类目<br>'
            '• 业务干预（置顶/过滤）<br>'
            '• 最终列表输出</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="note">面试追问：为什么召回和排序要分开？——因为对全量候选做精排计算量太大，召回负责"粗筛降低候选集"，排序负责"精排选出最优"。两层的模型复杂度和延迟要求完全不同。</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 6: 分布式训练架构
# ---------------------------------------------------------------------------

def distributed_training() -> None:
    st.subheader("分布式训练：数据并行 vs 模型并行")
    mode = st.radio(
        "选择并行策略",
        ["数据并行", "模型并行", "流水线并行", "混合并行"],
        key="dist-train-mode",
        horizontal=True,
    )
    if mode == "数据并行":
        st.markdown(
            '<div class="step"><strong>数据并行（Data Parallelism）</strong><br>'
            '每个 GPU 持有完整模型副本，数据被切分到各 GPU 上。<br><br>'
            '① 各 GPU 独立前向 + 反向，计算局部梯度<br>'
            '② 通过 AllReduce 同步梯度（Ring AllReduce 最常用）<br>'
            '③ 各 GPU 用同步后的梯度更新参数<br><br>'
            '<strong>优点：</strong>实现简单，加速比接近线性<br>'
            '<strong>瓶颈：</strong>通信开销随 GPU 数量增加，单卡显存要装下整个模型</div>',
            unsafe_allow_html=True,
        )
    elif mode == "模型并行":
        st.markdown(
            '<div class="step"><strong>模型并行（Tensor Parallelism）</strong><br>'
            '把模型的某一层切分到多个 GPU 上。<br><br>'
            '① 矩阵乘法按列或行切分到不同 GPU<br>'
            '② 每次前向需要 GPU 间通信（AllReduce / AllGather）<br>'
            '③ 适合单层参数量超大的场景（如 LLM 的 Attention + FFN）<br><br>'
            '<strong>优点：</strong>突破单卡显存限制<br>'
            '<strong>瓶颈：</strong>层内通信频繁，需要高带宽互联（NVLink）</div>',
            unsafe_allow_html=True,
        )
    elif mode == "流水线并行":
        st.markdown(
            '<div class="step"><strong>流水线并行（Pipeline Parallelism）</strong><br>'
            '把模型的不同层放到不同 GPU 上，形成流水线。<br><br>'
            '① GPU 0 放 Layer 0-3，GPU 1 放 Layer 4-7，依此类推<br>'
            '② 数据按 micro-batch 切分，各 GPU 交替计算<br>'
            '③ GPipe / 1F1B 调度减少气泡率<br><br>'
            '<strong>优点：</strong>通信量小（只传激活值），适合跨节点<br>'
            '<strong>瓶颈：</strong>流水线气泡导致 GPU 空闲</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="step"><strong>混合并行（Hybrid Parallelism）</strong><br>'
            '实际大模型训练通常组合使用多种并行策略。<br><br>'
            '• 3D 并行 = 数据并行 + 张量并行 + 流水线并行<br>'
            '• ZeRO（DeepSpeed）：把优化器状态、梯度、参数分片到各 GPU<br>'
            '• FSDP（PyTorch）：全切分数据并行，自动管理分片<br><br>'
            '<strong>典型配置：</strong>千亿参数模型用 8 卡张量并行 + 64 路流水线 + 16 路数据并行<br>'
            '<strong>面试要点：</strong>通信拓扑和梯度压缩直接影响训练效率</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 7: 模型推理服务设计
# ---------------------------------------------------------------------------

def inference_service() -> None:
    st.subheader("模型推理服务平台设计")
    st.markdown("系统设计高频题：设计一个支持多模型、多版本的推理服务平台。")
    components = [
        ("模型仓库", "管理模型版本、权重文件和配置，支持灰度发布和回滚"),
        ("调度器", "根据请求特征（模型名、输入大小、SLA）路由到合适实例"),
        ("推理引擎", "TensorRT / vLLM / Triton，做高效计算和批处理"),
        ("特征服务", "Redis / 特征平台，提供低延迟特征读取"),
        ("监控系统", "延迟、吞吐、GPU 利用率、错误率、队列深度"),
        ("流量管理", "A/B 分流、金丝雀发布、自动扩缩容"),
    ]
    for name, desc in components:
        st.markdown(f"**{name}**：{desc}")

    st.subheader("推理优化交互")
    optimize = st.selectbox(
        "选择优化手段",
        ["模型量化", "动态批处理", "KV Cache", "算子融合", "模型蒸馏"],
        key="infer-optimize",
    )
    optimizations = {
        "模型量化": "将 FP32/FP16 权重量化为 INT8/INT4，减少显存占用和计算量。代价是精度损失，需要校准数据集评估影响。GPTQ/AWQ 是大模型常用量化方案。",
        "动态批处理": "将多个请求合并为一个 batch 执行，提高 GPU 利用率。vLLM 的 PagedAttention 支持动态插入和抢占请求，减少排队延迟。",
        "KV Cache": "自回归生成时缓存已计算的 Key/Value，避免重复计算。是 LLM 推理的基础优化，显存占用随序列长度线性增长。",
        "算子融合": "将多个小算子合并为一个大 kernel，减少 GPU kernel launch 和内存读写开销。TensorRT 和 TorchScript 都支持自动融合。",
        "模型蒸馏": "用大模型（教师）的输出训练小模型（学生），保留大部分精度的同时大幅提升推理速度。适合部署到资源受限的边缘设备。",
    }
    st.markdown(f'<div class="note">{optimizations[optimize]}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 8: 高频问答
# ---------------------------------------------------------------------------

def faq_section() -> None:
    st.subheader("高频问答区")
    with st.expander("什么是幂等性？为什么 API 设计需要它？"):
        st.write(
            "幂等是指同一请求执行多次和执行一次效果相同。GET、PUT、DELETE 天然幂等，POST 不幂等需要通过唯一请求 ID 或去重表实现。"
            "网络超时重试时幂等性避免重复创建订单或重复扣款。"
            "实现方式：客户端生成唯一 request_id，服务端用去重表或 Redis SETNX 判断是否已处理。"
        )
    with st.expander("熔断、降级和限流有什么区别？"):
        st.write(
            "**熔断**：当下游服务错误率超过阈值时自动切断请求，防止故障蔓延。熔断器有关闭→打开→半开三种状态。\n\n"
            "**降级**：系统压力大时主动关闭非核心功能，保证核心链路可用。比如推荐服务挂了就返回热门兜底列表。\n\n"
            "**限流**：控制进入系统的请求速率，防止过载。常用算法有令牌桶和漏桶。"
        )
    with st.expander("如何设计一个短链接服务？"):
        st.write(
            "核心是长短映射：用自增 ID 或分布式 ID 生成器得到唯一数字，再 Base62 编码成短码。\n\n"
            "存储用 MySQL 或 Redis 做映射表。读多写少场景适合加缓存。302 重定向比 301 灵活，可以统计点击量。\n\n"
            "为什么用 Base62 而不是 Base64？因为 Base64 包含 + 和 /，在 URL 中需要转义；Base62 只用 [0-9a-zA-Z]，更干净。"
        )
    with st.expander("数据库分库分表怎么做？代价是什么？"):
        st.write(
            "垂直分库按业务拆（用户库、订单库），水平分表按某字段哈希或范围拆分。\n\n"
            "好处：单表数据量可控、查询快。\n"
            "代价：跨分片查询复杂、分布式事务困难、扩容需要数据迁移。\n\n"
            "分片键选择直接影响数据分布均匀性和查询路由效率。全局唯一 ID 可用 Snowflake 或 UUID。"
        )
    with st.expander("如何设计一个实时特征计算平台？"):
        st.write(
            "离线特征用批处理（Spark）预先计算存入特征库，在线特征用流处理（Flink）实时更新。\n\n"
            "特征存储统一管理版本和一致性，避免训练和推理用不同特征。\n\n"
            "关键挑战：低延迟读取、特征穿越（避免用到未来数据）、特征复用。"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>系统设计面试训练</h1>
          <p>从 CAP 定理到推理服务平台，用分层架构和交互实验建立系统设计直觉：不只背方案，还要能说清每一步的设计权衡。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    layered_architecture()
    cap_section()
    cache_problems()
    message_queue_section()
    recommendation_architecture()
    distributed_training()
    inference_service()
    faq_section()

    st.subheader("与深度学习的连接")
    st.markdown(
        """
        <div class="note">
        系统设计是深度学习工程师从"会训练模型"到"能把模型交付给用户"的关键能力跨越。
        推理平台、特征服务、模型仓库、分布式训练——每一个都需要理解缓存、消息队列、负载均衡和监控。
        面试中把系统设计和深度学习场景结合，是脱颖而出的核心策略。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("进入系统设计专项刷题", "/?module=part7%2Finterview_quiz&direction=系统设计", width="stretch")
    render_back_home()


if __name__ == "__main__":
    safe_run(main)
