#!/usr/bin/env python3
"""
Deep Learning Book home page and launcher.

Run:
    python main.py                         # start the static HTML learning site
    python main.py --menu
    python main.py part4/transformer_models
    streamlit run main.py                  # legacy notice only
"""

from __future__ import annotations

import argparse
import runpy
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs

from components.artifact_runtime import ARTIFACT_ROOT_NAME
from components.course_catalog import (
    ModuleInfo,
    build_module_catalog,
    build_route_map,
    daily_recommendation as recommend_daily_module,
    list_part_modules,
)
from components.course_manifest import PARTS
from components.course_manifest import REGISTERED_MODULES as MODULES
from components.local_runtime import (
    DEFAULT_STATIC_PORTS,
    DEFAULT_STREAMLIT_PORT,
    choose_static_port,
    port_is_available,
    running_under_streamlit,
    run_static_site as run_local_static_site,
    run_streamlit_app as run_local_streamlit_app,
)
from components.legacy_page import LegacyPageDeps
from components.legacy_page import render_legacy_module_page as render_legacy_page_shell
from components.streamlit_home import StreamlitHomeDeps
from components.streamlit_home import render_streamlit_home as render_streamlit_home_shell
from components.streamlit_shell import css
from components.streamlit_shell import escape_html as e
from components.streamlit_shell import module_href as shell_module_href
from components.streamlit_shell import render_home_button
from components.streamlit_shell import render_missing_module_page as render_missing_module_page_shell
from components.streamlit_shell import render_module_card as shell_render_module_card
from components.streamlit_shell import render_module_header as render_module_header_shell
from components.streamlit_shell import render_streamlit_module_page as render_streamlit_module_page_shell


BASE_DIR = Path(__file__).resolve().parent
LEGACY_OUTPUT_DIR = BASE_DIR / ARTIFACT_ROOT_NAME




KNOWLEDGE_POINTS = [
    ("梯度下降", "先看损失曲线，再看梯度范数。很多训练问题不是模型不够大，而是优化过程已经失稳。", "part1/03_datasets_optimizers"),
    ("卷积核", "卷积核不是只会找边缘。浅层偏纹理，深层偏语义，特征图越往后越依赖训练目标。", "part2/02_feature_maps"),
    ("残差连接", "残差结构让网络学习修正量，而不是每层都重新表达完整映射，是深层网络可训练的关键机制之一。", "part2/06_modern_architectures"),
    ("隐藏状态", "RNN 的隐藏状态是压缩后的历史摘要；长期依赖难学，正是因为这个摘要需要跨很多步保持可用。", "part3/02_hidden_states"),
    ("注意力", "注意力权重不是解释的全部，但它提供了观察信息路由的入口：谁在问、问谁、取回什么。", "part4/01_attention_mechanism"),
    ("位置编码", "Transformer 没有天然顺序感，位置编码把序列顺序注入表示空间，才让模型分清相同词在不同位置的角色。", "part4/transformer_models"),
    ("梯度监控", "梯度直方图和范数曲线能提前暴露训练问题，比只看最终准确率更适合调试。", "part5/02_gradient_monitor"),
    ("实验记录", "超参搜索的价值不只是找到最好结果，更是留下可复盘的失败样本。", "part5/04_hyperparam_search"),
    ("统一接口", "项目变复杂后，真正降低成本的是稳定边界：数据、模型、训练、评估分别可替换。", "part6/01_unified_interface"),
    ("学习路径", "先补最短缺口，再做完整项目。路径规划的目标不是学最多，而是让下一步最有杠杆。", "part6/learning_path"),
]


@lru_cache(maxsize=1)
def module_catalog() -> tuple[ModuleInfo, ...]:
    return build_module_catalog(BASE_DIR, PARTS, MODULES)


@lru_cache(maxsize=1)
def cached_route_map() -> dict[str, ModuleInfo]:
    return build_route_map(module_catalog())


def route_map(modules: tuple[ModuleInfo, ...] | list[ModuleInfo] | None = None) -> dict[str, ModuleInfo]:
    if modules is None:
        return cached_route_map().copy()
    return build_route_map(tuple(modules))


def configure_plotting() -> None:
    try:
        import sitecustomize
    except Exception:
        return

    configure = getattr(sitecustomize, "_configure_matplotlib", None)
    if callable(configure):
        configure()


@lru_cache(maxsize=None)
def is_streamlit_app(module_path: Path) -> bool:
    if not module_path.exists():
        return False
    text = module_path.read_text(encoding="utf-8", errors="ignore")
    return "import streamlit as st" in text


def list_modules(part_name: str) -> list[str]:
    return list_part_modules(BASE_DIR, PARTS, part_name)


def run_static_site(port: int | None = None) -> None:
    run_local_static_site(BASE_DIR, port)


def run_streamlit_app(module_path: Path, port: int = DEFAULT_STREAMLIT_PORT) -> None:
    run_local_streamlit_app(BASE_DIR, module_path, port)


def run_module(target: str) -> None:
    normalized = target.strip().replace("\\", "/")
    if normalized in PARTS:
        show_part_picker(normalized)
        return
    if "/" not in normalized:
        print(f"未知目标: {target}")
        return

    routes = route_map()
    module = routes.get(normalized)
    if module:
        module_path = module.path
    else:
        part, module_name = normalized.split("/", 1)
        part_info = PARTS.get(part)
        part_dir = part_info.directory if part_info else part
        module_path = BASE_DIR / part_dir / f"{module_name}.py"

    if not module_path.exists():
        print(f"文件不存在: {module_path}")
        return

    if is_streamlit_app(module_path):
        run_streamlit_app(module_path)
        return

    print(f"运行: {module_path.relative_to(BASE_DIR)}")
    print("=" * 60)
    configure_plotting()
    runpy.run_path(str(module_path), run_name="__main__")


def show_menu() -> None:
    print("=" * 60)
    print("  深度学习书库 - 模块菜单")
    print("=" * 60)
    for key, part in PARTS.items():
        modules = list_modules(key)
        print(f"\n{key} ({part.title}, {part.directory}) - {len(modules)} 个模块")
        for module in modules:
            print(f"  - {module}")
    print("\n用法: python main.py <part>/<module>")
    print("示例: python main.py part6/frontier")


def show_part_picker(part: str) -> None:
    modules = list_modules(part)
    if not modules:
        return
    print(f"\n{part} 的模块")
    for index, module in enumerate(modules, 1):
        print(f"  {index}. {module}")
    choice = input("\n选择模块编号: ").strip()
    if choice.isdigit() and 0 < int(choice) <= len(modules):
        run_module(f"{part}/{modules[int(choice) - 1]}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--menu", action="store_true", help="显示命令行模块菜单")
    parser.add_argument("--port", type=int, help="静态 HTML 本地端口；默认自动选择 8000/8001/4173 等可用端口")
    parser.add_argument("target", nargs="?", help="可选模块，例如 part6/frontier")
    return parser.parse_args(argv)


LEGACY_LEARNING_GUIDES: dict[str, list[tuple[str, str]]] = {
    "part5_toolbox/01_feature_visualization": [
        (
            "图怎么看",
            "特征图每个小格是一条通道：亮区表示这个通道对某种边缘、纹理或局部形状反应强；卷积核图看权重模式，激活最大化图看“什么输入会让某个通道最兴奋”。",
        ),
        (
            "参数怎么想",
            "`max_channels` 决定一次看多少通道，`layer_name` 决定观察浅层边缘还是深层语义，`channel_idx` 决定激活最大化优化哪个探测器。",
        ),
        (
            "工程坑",
            "我见过最常见的误读是把特征图当成原图热力图。正确做法是先确认层名和通道，再问它是否稳定响应同一种模式；如果每个通道都像噪声，通常要查输入归一化、模型是否训练过、hook 是否挂错层。",
        ),
        (
            "进阶思考",
            "同一张输入经过 `conv1` 和 `conv2` 后，哪些通道更稀疏？如果只看最后分类结果，你会漏掉哪些模型已经学偏的线索？",
        ),
    ],
    "part5_toolbox/02_gradient_monitor": [
        (
            "图怎么看",
            "折线图看梯度随训练步数是否平稳，条形图看最后一刻哪一层异常，热力图看异常是突然出现还是长期存在；红色接近消失，黄色或极高值接近爆炸。",
        ),
        (
            "排查顺序",
            "先看 `loss` 是否发散，再看梯度 `max` 是否越过爆炸阈值，最后看前几层 `mean` 是否长期小于 `1e-6`。三者同时看，才能区分学习率过高、初始化不当和激活饱和。",
        ),
        (
            "工程坑",
            "梯度爆炸的症状通常是 loss 突然变成 NaN 或曲线大幅震荡；梯度消失的症状是训练 loss 很慢、早期层梯度接近 0。先尝试把学习率降到原来的 1/3，再加 `clip_grad_norm_(..., 1.0)`，不要一上来重写模型。",
        ),
        (
            "进阶思考",
            "如果只有最后一层梯度很大，是标签/损失设置更可疑，还是整网学习率更可疑？如果只有前几层梯度消失，残差连接和归一化层各能解决哪一段问题？",
        ),
    ],
    "part5_toolbox/03_training_dynamics": [
        (
            "图怎么看",
            "权重分布图看均值是否持续偏移、标准差是否膨胀；激活饱和图看 ReLU 死亡比例或 Tanh/Sigmoid 饱和比例；更新幅度图看 `lr * grad_norm / weight_norm` 是否在 `1e-4` 到 `1e-2` 的健康带内。",
        ),
        (
            "联合诊断",
            "loss 降而验证不升，多半是过拟合或数据问题；loss 不降且更新比小，多半学习率太低；loss 震荡且更新比大，多半学习率太高。训练动态要和学习率曲线一起读。",
        ),
        (
            "工程坑",
            "只盯准确率很容易晚发现问题。我踩过的典型坑是准确率暂时上涨，但激活饱和率已经超过 50%，两天后模型迁移到新数据立刻崩。看到饱和率变红，要先查初始化、激活函数和输入尺度。",
        ),
        (
            "进阶思考",
            "为什么更新比比单独的梯度范数更可靠？如果某一层权重很小但梯度正常，它的更新比会怎样提醒你？",
        ),
    ],
    "part5_toolbox/04_hyperparam_search": [
        (
            "图怎么看",
            "LR Finder 重点看对数坐标下 loss 下降最快的区间；调度策略图看学习率何时变大、何时退火；敏感性图看哪个参数改变后验证分数波动最大。",
        ),
        (
            "搜索空间",
            "学习率优先用对数尺度搜索，例如 `1e-4`、`3e-4`、`1e-3`、`3e-3`；dropout 先看 `0.0` 到 `0.5`；hidden size 先做 2 到 3 个量级点，不要一开始铺满网格。",
        ),
        (
            "工程坑",
            "超参搜索最大的坑不是慢，而是用测试集选参数。正确流程是训练集训练、验证集选择、测试集只最终报告；早停也必须看验证集，否则会把噪声当成能力。",
        ),
        (
            "进阶思考",
            "如果最优点周围一圈配置都很差，你会信这个最优点吗？随机搜索和网格搜索在高维参数空间里为什么表现不同？",
        ),
    ],
    "part6_universal_framework/01_unified_interface": [
        (
            "抽象边界",
            "`TensorDatasetWrapper` 管数据形状和归一化，`TrainableMixin.fit` 管训练流程，`MLP/SimpleCNN` 管模型结构。统一接口的价值是让数据、模型、训练三件事能替换，但边界仍然清楚。",
        ),
        (
            "默认值经验",
            "`batch_size=32/64`、`lr=1e-3`、`patience=10`、`grad_clip=1.0` 是多数小实验的稳妥起点；生产项目再根据验证曲线微调，而不是把所有参数都暴露给新手。",
        ),
        (
            "工程坑",
            "过度抽象会把错误藏起来：如果 `.fit()` 里自动做了太多事，初学者会不知道优化器、调度器和早停在哪里生效。遇到异常时先打印 config、数据 batch shape、loss 和 lr。",
        ),
        (
            "进阶思考",
            "哪些东西应该统一成接口，哪些东西应该保留在具体模型里？如果任务从分类变成回归，应该改 task、loss，还是改模型 forward？",
        ),
    ],
    "part6_universal_framework/04_plugin_system": [
        (
            "图怎么看",
            "注册表不是神秘容器，本质是名字到类的映射。`register_model`、`register_dataset`、`register_task` 分别把模型、数据、任务挂到同一个可查询目录里。",
        ),
        (
            "扩展流程",
            "新增组件时先写类，再用装饰器注册，最后通过配置里的 `name` 构建。这样切换模型只改配置，不改训练主循环。",
        ),
        (
            "工程坑",
            "插件系统最容易踩的是名称冲突和默认参数失控。生产中要检查重复注册、记录最终合并后的 config，并让插件加载失败时给出明确错误，而不是静默跳过。",
        ),
        (
            "进阶思考",
            "插件让扩展更快，但也让系统更难追踪。你会把数据增强也做成插件吗？哪些组件变化频繁，值得注册化？",
        ),
    ],
    "part6_universal_framework/03_full_project": [
        (
            "完整闭环",
            "`UniversalTrainer` 管训练、验证、调度、早停和最优模型保存；`UniversalVisualizer` 管结构摘要、参数分布、预测样例和错误分析。",
        ),
        (
            "产物边界",
            "一次可复现实验至少要有 config、checkpoint、history、训练曲线和最终评估。只留下一个模型权重，后面很难解释结果从哪里来。",
        ),
        (
            "工程坑",
            "完整项目最常见的问题不是少写模型，而是训练、验证、测试边界混乱。评估函数必须只评估，不应偷偷更新参数或改动随机种子。",
        ),
        (
            "进阶思考",
            "如果验证 loss 改善但业务指标变差，你会先查 metric_fn、数据切分，还是模型结构？为什么 evaluate 不应该调用 optimizer.step()？",
        ),
    ],
    "part6_universal_framework/05_one_click_training": [
        (
            "流程闭环",
            "一键训练不是只有 `runner.run()`，而是配置、设备、模型、数据、loss、optimizer、scheduler、checkpoint、日志和最终评估的完整流水线。",
        ),
        (
            "产物怎么看",
            "`training_log.csv` 看每轮指标，`best.pt` 保存最优权重，`config.json` 固化复现实验，`training_curves.png` 把 loss、metric、lr 放到同一张诊断图里。",
        ),
        (
            "工程坑",
            "我见过最贵的坑是只保存最后一轮模型，没有保存最佳验证集模型。训练后期过拟合时，最后一轮可能比第 7 轮差很多；所以默认保存 best checkpoint，并记录 monitor 指标。",
        ),
        (
            "进阶思考",
            "如果训练中断，恢复实验需要哪些文件？为什么日志、配置和 checkpoint 必须放在同一个实验目录下？",
        ),
    ],
    "part6_universal_framework/07_project_template": [
        (
            "项目目录",
            "训练入口负责读 config、设 seed、构建组件和启动 runner；评估脚本只加载 checkpoint 做验证；K-Fold 和 ensemble 是比赛或高风险评估中的复现工具。",
        ),
        (
            "复现流程",
            "一个可复现项目至少要保存 config、随机种子、数据切分、代码版本、checkpoint、训练日志和最终指标。少一个，后面就很难解释为什么这次结果变了。",
        ),
        (
            "工程坑",
            "模板最大的风险是复制后不删占位逻辑。比如 `TODO` 评估数据、`...` dataloader 必须在真实项目里补全，否则会形成“看起来完整、实际不可复现”的假工程。",
        ),
        (
            "进阶思考",
            "为什么训练脚本和评估脚本要分开？如果线上指标和离线验证集冲突，你会优先检查数据切分、指标定义，还是模型结构？",
        ),
    ],
}


def module_kind(module: ModuleInfo) -> str:
    if is_streamlit_app(module.path):
        return "交互页面"
    return "经典脚本"


def module_href(module: ModuleInfo) -> str:
    return shell_module_href(module)


def render_module_header(module: ModuleInfo) -> None:
    render_module_header_shell(module, PARTS, module_kind)


def render_module_card(module: ModuleInfo) -> str:
    return shell_render_module_card(module)


def render_module_knowledge_nav(module: ModuleInfo) -> None:
    """把每个章节页接到全站知识图谱：前置、相关、下一步和去实战。"""

    try:
        from components.progress_tracker import render学习操作面板

        render学习操作面板(module.short_target)
    except Exception as exc:
        st = __import__("streamlit")
        st.warning("学习进度与复盘面板暂时无法显示，但当前章节内容不受影响。")
        with st.expander("查看学习进度错误", expanded=False):
            st.code(str(exc), language="text")

    try:
        from components.knowledge_graph import render知识图谱导航

        render知识图谱导航(module.short_target)
    except Exception as exc:
        st = __import__("streamlit")
        st.warning("知识图谱导航暂时无法显示，但当前章节内容不受影响。")
        with st.expander("查看知识图谱错误", expanded=False):
            st.code(str(exc), language="text")


def render_legacy_module_page(module: ModuleInfo) -> None:
    deps = LegacyPageDeps(
        project_root=BASE_DIR,
        learning_guides=LEGACY_LEARNING_GUIDES,
        css=css,
        escape_html=e,
        render_home_button=render_home_button,
        render_module_header=render_module_header,
        render_module_card=render_module_card,
        render_module_knowledge_nav=render_module_knowledge_nav,
    )
    render_legacy_page_shell(deps, module)


def render_route_error(module: ModuleInfo | None, error: BaseException) -> None:
    from components.streamlit_shell import render_route_error as render_error

    render_error(module, error)


def render_streamlit_module_page(module: ModuleInfo) -> None:
    render_streamlit_module_page_shell(module, render_module_knowledge_nav, render_module_header)


def render_missing_module_page(query_module: str) -> None:
    render_missing_module_page_shell(query_module)


def daily_recommendation(catalog: list[ModuleInfo]) -> tuple[str, str, ModuleInfo]:
    return recommend_daily_module(catalog, KNOWLEDGE_POINTS)


def get_query_module() -> str | None:
    try:
        import streamlit as st

        value = st.query_params.get("module")
        if isinstance(value, list):
            return value[0] if value else None
        if value:
            return str(value)
    except Exception:
        pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
    except Exception:
        return None
    if ctx is None:
        return None

    query_string = getattr(ctx, "query_string", "")
    values = parse_qs(query_string).get("module", [])
    return values[0] if values else None


def render_streamlit_home() -> None:
    catalog = module_catalog()
    deps = StreamlitHomeDeps(
        project_root=BASE_DIR,
        parts=PARTS,
        catalog=catalog,
        routes=route_map(catalog),
        query_module=get_query_module(),
        is_streamlit_app=is_streamlit_app,
        render_legacy_module_page=render_legacy_module_page,
        render_streamlit_module=render_streamlit_module_page,
        render_missing_module=render_missing_module_page,
        css=css,
        module_href=module_href,
        render_module_card=render_module_card,
        daily_recommendation=daily_recommendation,
        escape_html=e,
    )
    render_streamlit_home_shell(deps)


def main(argv: list[str] | None = None) -> int:
    if running_under_streamlit():
        render_streamlit_home()
        return 0

    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.menu:
        show_menu()
        return 0
    if args.target:
        run_module(args.target)
        return 0

    run_static_site(args.port)
    return 0


if __name__ == "__main__":
    if running_under_streamlit():
        main()
    else:
        raise SystemExit(main())
