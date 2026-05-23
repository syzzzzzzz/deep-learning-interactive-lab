#!/usr/bin/env python3
"""
Deep Learning Book home page and launcher.

Run:
    python main.py
    streamlit run main.py
    python main.py --menu
    python main.py part4/transformer_models
"""

from __future__ import annotations

import argparse
import hashlib
import html
import inspect
import random
import runpy
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, quote


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8501
LEGACY_OUTPUT_DIR = BASE_DIR / ".streamlit_module_outputs"


@dataclass(frozen=True)
class PartInfo:
    key: str
    directory: str
    emoji: str
    title: str
    short_title: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.title}"


@dataclass(frozen=True)
class ModuleInfo:
    part_key: str
    part_dir: str
    title: str
    module: str
    summary: str
    level: str
    tags: tuple[str, ...]
    priority: int = 50

    @property
    def path(self) -> Path:
        return BASE_DIR / self.part_dir / f"{self.module}.py"

    @property
    def target(self) -> str:
        return f"{self.part_dir}/{self.module}"

    @property
    def short_target(self) -> str:
        return f"{self.part_key}/{self.module}"


PARTS: dict[str, PartInfo] = {
    "part1": PartInfo(
        "part1",
        "part1_foundations",
        "🧱",
        "第一部分 基础",
        "基础",
        "张量、梯度、经典机器学习、数学基础和神经网络入门。",
    ),
    "part2": PartInfo(
        "part2",
        "part2_cnn",
        "👁️",
        "第二部分 CNN",
        "CNN",
        "卷积、特征图、现代视觉架构、调试、迁移学习和可视化。",
    ),
    "part3": PartInfo(
        "part3",
        "part3_rnn",
        "🔁",
        "第三部分 RNN",
        "RNN",
        "序列建模、隐藏状态、注意力、文本分类和高级训练技巧。",
    ),
    "part4": PartInfo(
        "part4",
        "part4_transformer",
        "⚡",
        "第四部分 Transformer",
        "Transformer",
        "自注意力、多头注意力、编码器解码器、生成模型和图神经网络。",
    ),
    "part5": PartInfo(
        "part5",
        "part5_toolbox",
        "🧰",
        "第五部分 工具箱",
        "工具箱",
        "数据训练、特征可视化、训练监控、超参搜索、部署和测验系统。",
    ),
    "part6": PartInfo(
        "part6",
        "part6_universal_framework",
        "🚀",
        "第六部分 统一框架与前沿",
        "框架与前沿",
        "统一接口、模块化工程、项目模板、学习路径、术语表和前沿方向。",
    ),
}


MODULES: list[ModuleInfo] = [
    ModuleInfo("part1", "part1_foundations", "张量与梯度", "01_tensors_gradients", "用可视化理解张量、自动求导和梯度传播。", "入门", ("基础", "张量", "梯度"), 1),
    ModuleInfo("part1", "part1_foundations", "激活与归一化", "02_activations_normalization", "比较常见激活函数、归一化方法和训练稳定性。", "入门", ("基础", "激活函数", "归一化"), 2),
    ModuleInfo("part1", "part1_foundations", "数据集与优化器", "03_datasets_optimizers", "理解数据划分、批训练、SGD、Adam 和优化曲线。", "入门", ("基础", "数据", "优化"), 3),
    ModuleInfo("part1", "part1_foundations", "数学基础速查", "math_primer", "线性代数、微积分、概率论和梯度下降的交互式速查。", "入门", ("基础", "数学", "可视化"), 4),
    ModuleInfo("part1", "part1_foundations", "机器学习基础", "machine_learning_basics", "监督学习、损失函数、泛化、评估和模型选择。", "入门", ("基础", "机器学习"), 5),
    ModuleInfo("part1", "part1_foundations", "神经网络基础", "neural_network_basics", "从感知机到多层网络，理解反向传播和非线性表达。", "入门", ("基础", "神经网络"), 6),
    ModuleInfo("part1", "part1_foundations", "经典机器学习", "classical_ml", "用传统模型建立深度学习前的基线意识。", "入门", ("基础", "模型", "基线"), 7),
    ModuleInfo("part2", "part2_cnn", "卷积直觉", "01_convolution_visual", "用滑窗、卷积核和边缘检测建立 CNN 直觉。", "进阶", ("视觉", "CNN", "卷积"), 10),
    ModuleInfo("part2", "part2_cnn", "特征图可视化", "02_feature_maps", "观察卷积层如何从局部纹理逐步形成抽象特征。", "进阶", ("视觉", "CNN", "可视化"), 11),
    ModuleInfo("part2", "part2_cnn", "经典 CNN 架构", "03_classic_architectures", "梳理 LeNet、AlexNet、VGG、GoogLeNet 和 ResNet。", "进阶", ("视觉", "CNN", "架构"), 12),
    ModuleInfo("part2", "part2_cnn", "CNN 调试面板", "04_debug_panel", "定位卷积模型训练中的过拟合、梯度和数据问题。", "工程", ("视觉", "调试", "训练"), 13),
    ModuleInfo("part2", "part2_cnn", "MNIST 玩具实验", "05_mnist_toy", "用小型手写数字实验串起数据、模型、训练和评估。", "实验", ("视觉", "CNN", "实验"), 14),
    ModuleInfo("part2", "part2_cnn", "现代 CNN 架构", "06_modern_architectures", "理解残差、深度可分离卷积和高效视觉网络。", "进阶", ("视觉", "CNN", "架构"), 15),
    ModuleInfo("part2", "part2_cnn", "高级卷积技术", "07_advanced_convolution", "扩张卷积、转置卷积、分组卷积和感受野分析。", "进阶", ("视觉", "CNN", "卷积"), 16),
    ModuleInfo("part2", "part2_cnn", "Grad-CAM 可视化", "08_visualization_gradcam", "用热力图解释 CNN 决策关注区域。", "实验", ("视觉", "解释性", "可视化"), 17),
    ModuleInfo("part2", "part2_cnn", "迁移学习", "09_transfer_learning", "复用预训练模型完成小数据任务。", "工程", ("视觉", "迁移学习"), 18),
    ModuleInfo("part2", "part2_cnn", "CNN 架构实验", "cnn_architectures", "对比经典卷积网络的结构与特征提取方式。", "进阶", ("视觉", "CNN"), 19),
    ModuleInfo("part2", "part2_cnn", "高级 CNN", "advanced_cnn", "现代卷积技巧、残差思想和视觉模型设计。", "进阶", ("视觉", "CNN"), 20),
    ModuleInfo("part3", "part3_rnn", "RNN 直觉", "01_rnn_intuition", "从循环状态理解序列信息如何流动。", "进阶", ("序列", "RNN"), 30),
    ModuleInfo("part3", "part3_rnn", "隐藏状态", "02_hidden_states", "观察隐藏状态、门控结构和长期依赖。", "进阶", ("序列", "RNN", "可视化"), 31),
    ModuleInfo("part3", "part3_rnn", "序列玩具任务", "03_sequence_toys", "用可控任务理解记忆、预测和序列泛化。", "实验", ("序列", "RNN", "实验"), 32),
    ModuleInfo("part3", "part3_rnn", "RNN 超参实验", "04_hyperparam_rnn", "比较学习率、隐藏维度、层数和截断反传。", "实验", ("序列", "训练", "超参数"), 33),
    ModuleInfo("part3", "part3_rnn", "Seq2Seq 与注意力", "05_seq2seq_attention", "理解编码器解码器和注意力对齐。", "核心", ("序列", "注意力", "NLP"), 34),
    ModuleInfo("part3", "part3_rnn", "文本分类", "06_text_classification", "用序列模型完成文本表示和分类。", "实验", ("序列", "NLP", "分类"), 35),
    ModuleInfo("part3", "part3_rnn", "高级训练技巧", "07_advanced_training", "处理梯度裁剪、Teacher Forcing、正则化和训练稳定性。", "工程", ("序列", "训练", "调试"), 36),
    ModuleInfo("part3", "part3_rnn", "RNN 调试问题", "08_debug_problems", "定位序列模型中的梯度、数据和评估问题。", "工程", ("序列", "调试"), 37),
    ModuleInfo("part3", "part3_rnn", "序列模型", "sequence_models", "RNN、LSTM、GRU 与序列任务的基本范式。", "进阶", ("序列", "NLP"), 38),
    ModuleInfo("part4", "part4_transformer", "注意力机制", "01_attention_mechanism", "从查询、键、值理解注意力权重。", "核心", ("Transformer", "注意力"), 40),
    ModuleInfo("part4", "part4_transformer", "多头注意力可视化", "02_multihead_visual", "观察不同注意力头如何捕获互补关系。", "核心", ("Transformer", "注意力", "可视化"), 41),
    ModuleInfo("part4", "part4_transformer", "编码器与解码器", "03_encoder_decoder", "拆解 Transformer 编码器、解码器和掩码机制。", "核心", ("Transformer", "NLP"), 42),
    ModuleInfo("part4", "part4_transformer", "最小 Transformer", "04_minimal_transformer", "用精简实现串起嵌入、注意力、MLP 和残差。", "核心", ("Transformer", "实现"), 43),
    ModuleInfo("part4", "part4_transformer", "Flash Attention", "05_flash_attention", "理解高效注意力的内存访问与计算优化。", "前沿", ("Transformer", "性能", "注意力"), 44),
    ModuleInfo("part4", "part4_transformer", "Transformer 调试", "06_debug_problems", "分析大模型训练中的掩码、位置编码和梯度问题。", "工程", ("Transformer", "调试"), 45),
    ModuleInfo("part4", "part4_transformer", "Transformer 架构", "transformer_models", "可视化拆解自注意力、多头、位置编码和 BERT/GPT。", "核心", ("Transformer", "NLP"), 46),
    ModuleInfo("part4", "part4_transformer", "GAN 与自编码器", "gan_ae", "理解生成模型、潜空间和重构学习。", "进阶", ("生成模型", "表征"), 47),
    ModuleInfo("part4", "part4_transformer", "图神经网络", "gnn_intro", "从节点、边和消息传递理解图学习。", "进阶", ("GNN", "结构数据"), 48),
    ModuleInfo("part5", "part5_toolbox", "特征可视化", "01_feature_visualization", "观察特征、激活、嵌入和决策边界。", "工程", ("工具", "可视化", "解释性"), 50),
    ModuleInfo("part5", "part5_toolbox", "梯度监控", "02_gradient_monitor", "监控梯度范数、爆炸、消失和训练健康度。", "工程", ("工具", "梯度", "调试"), 51),
    ModuleInfo("part5", "part5_toolbox", "训练动态", "03_training_dynamics", "用曲线和指标追踪模型如何学习。", "工程", ("训练", "监控"), 52),
    ModuleInfo("part5", "part5_toolbox", "超参搜索", "04_hyperparam_search", "比较网格搜索、随机搜索和实验记录。", "工程", ("超参数", "工具"), 53),
    ModuleInfo("part5", "part5_toolbox", "玩具数据集", "05_dataset_toys", "用小数据集快速验证模型直觉。", "实验", ("数据", "实验"), 54),
    ModuleInfo("part5", "part5_toolbox", "数据与训练", "data_training", "数据管线、训练循环、指标与调试。", "工程", ("训练", "数据"), 55),
    ModuleInfo("part5", "part5_toolbox", "案例研究", "case_studies", "用完整案例串联建模、调参和诊断流程。", "工程", ("案例", "实践"), 56),
    ModuleInfo("part5", "part5_toolbox", "部署工具", "deployment_tools", "模型导出、服务化、推理和工程落地。", "工程", ("部署", "工具"), 57),
    ModuleInfo("part5", "part5_toolbox", "练习题与测验", "quiz_system", "覆盖机器学习基础、CNN、RNN、Transformer 和 GAN 的交互式测验。", "复习", ("测验", "复习"), 58),
    ModuleInfo("part5", "part5_toolbox", "调参实战挑战", "tuning_challenge", "在真实约束下练习学习率、正则、模型规模和实验记录决策。", "实验", ("调参", "实验", "诊断"), 59),
    ModuleInfo("part6", "part6_universal_framework", "统一接口", "01_unified_interface", "把模型、数据和任务抽象成统一可扩展接口。", "工程", ("框架", "架构"), 60),
    ModuleInfo("part6", "part6_universal_framework", "模块化结构", "02_modular_structure", "拆分配置、数据、模型、训练和评估边界。", "工程", ("框架", "模块化"), 61),
    ModuleInfo("part6", "part6_universal_framework", "完整项目骨架", "03_full_project", "组织可复用的深度学习项目目录和执行流程。", "工程", ("项目", "架构"), 62),
    ModuleInfo("part6", "part6_universal_framework", "插件系统", "04_plugin_system", "用注册表和插件扩展任务、模型与工具。", "工程", ("框架", "插件"), 63),
    ModuleInfo("part6", "part6_universal_framework", "一键训练", "05_one_click_training", "从配置到训练、评估和产物保存的一键流程。", "工程", ("训练", "自动化"), 64),
    ModuleInfo("part6", "part6_universal_framework", "可视化实验台", "06_streamlit_demo", "用 Streamlit 交互观察边界、卷积和注意力。", "核心", ("实验", "可视化", "Streamlit"), 65),
    ModuleInfo("part6", "part6_universal_framework", "项目模板", "07_project_template", "训练脚本、评估脚本、K-Fold 和集成预测模板。", "工程", ("项目", "模板"), 66),
    ModuleInfo("part6", "part6_universal_framework", "强化学习入门", "reinforcement_learning", "强化学习概念、多臂老虎机、Q-Learning 和纯 Python 环境 demo。", "核心", ("RL", "强化学习", "实验"), 67),
    ModuleInfo("part6", "part6_universal_framework", "学习路径推荐", "learning_path", "入门测评、个性化路径、知识图谱、进度追踪和下一步推荐。", "核心", ("路径", "知识图谱", "测评"), 68),
    ModuleInfo("part6", "part6_universal_framework", "深度学习术语表", "glossary", "集中检索常见概念、缩写和相关模块。", "复习", ("术语", "搜索", "复习"), 69),
    ModuleInfo("part6", "part6_universal_framework", "前沿方向", "frontier", "LLM、AGI、多模态、自监督、XAI、安全与对齐。", "前沿", ("LLM", "AGI", "安全"), 70),
    ModuleInfo("part6", "part6_universal_framework", "经典论文解读实验室", "paper_reading_lab", "用时间线、机制图和最小复现清单读懂经典深度学习论文。", "进阶", ("论文", "可视化", "复现"), 71),
]


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


def fallback_title(module_name: str) -> str:
    title = module_name
    for prefix in [f"{index:02d}_" for index in range(1, 100)]:
        if title.startswith(prefix):
            title = title[len(prefix):]
            break
    return title.replace("_", " ").strip().title()


@lru_cache(maxsize=1)
def module_catalog() -> tuple[ModuleInfo, ...]:
    registered = {(module.part_dir, module.module): module for module in MODULES}
    modules: list[ModuleInfo] = []

    for part_index, part in enumerate(PARTS.values(), 1):
        full_dir = BASE_DIR / part.directory
        if not full_dir.is_dir():
            continue

        for path in sorted(full_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue

            key = (part.directory, path.stem)
            if key in registered:
                modules.append(registered[key])
                continue

            modules.append(
                ModuleInfo(
                    part_key=part.key,
                    part_dir=part.directory,
                    title=fallback_title(path.stem),
                    module=path.stem,
                    summary=f"自动发现页面：{part.directory}/{path.name}",
                    level="模块",
                    tags=(part.short_title, "自动注册"),
                    priority=part_index * 100 + len(modules),
                )
            )

    part_order = {key: index for index, key in enumerate(PARTS)}
    return tuple(sorted(modules, key=lambda item: (part_order[item.part_key], item.priority, item.module)))


def build_route_map(modules: tuple[ModuleInfo, ...]) -> dict[str, ModuleInfo]:
    routes: dict[str, ModuleInfo] = {}
    for module in modules:
        routes[module.target] = module
        routes[module.short_target] = module
    return routes


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
    part = PARTS.get(part_name)
    part_dir = part.directory if part else part_name
    full_dir = BASE_DIR / part_dir
    if not full_dir.is_dir():
        print(f"目录不存在: {full_dir}")
        return []
    return sorted(p.stem for p in full_dir.iterdir() if p.suffix == ".py" and p.name != "__init__.py")


def run_streamlit_app(module_path: Path, port: int = DEFAULT_PORT) -> None:
    print(f"启动 Streamlit 页面: {module_path.relative_to(BASE_DIR)}")
    print(f"地址: http://127.0.0.1:{port}")
    print("请保持这个终端窗口打开；关闭窗口后页面会断开。")
    print("=" * 60)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(module_path),
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        check=False,
    )


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
    parser.add_argument("target", nargs="?", help="可选模块，例如 part6/frontier")
    return parser.parse_args(argv)


def e(value: str) -> str:
    return html.escape(value, quote=True)


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


def safe_output_root(module: ModuleInfo) -> Path:
    digest = hashlib.sha1(module.target.encode("utf-8")).hexdigest()[:12]
    return LEGACY_OUTPUT_DIR / digest


def latest_run_dir(module: ModuleInfo) -> Path | None:
    root = safe_output_root(module)
    if not root.is_dir():
        return None
    runs = [path for path in root.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime, default=None)


def read_text_preview(path: Path, max_lines: int = 140) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n\n# ... 省略 {len(lines) - max_lines} 行，完整源码在 {path.name}"
    return preview


def run_legacy_module(module: ModuleInfo, timeout_seconds: int = 45) -> dict[str, object]:
    root = safe_output_root(module)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    env = os_environ_utf8()
    command = [
        sys.executable,
        str(BASE_DIR / "legacy_runner.py"),
        str(BASE_DIR),
        str(module.path),
        str(run_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = (exc.stdout or "")
        stderr = (exc.stderr or "") + f"\n运行超过 {timeout_seconds} 秒，已停止。"

    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")
    (run_dir / "status.txt").write_text(
        f"return_code={return_code}\ntimed_out={timed_out}\n",
        encoding="utf-8",
    )
    return {
        "run_dir": run_dir,
        "return_code": return_code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def read_run_text(run_dir: Path, name: str) -> str:
    path = run_dir / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def run_status(run_dir: Path) -> tuple[int | None, bool]:
    text = read_run_text(run_dir, "status.txt")
    return_code: int | None = None
    timed_out = False
    for line in text.splitlines():
        if line.startswith("return_code="):
            try:
                return_code = int(line.split("=", 1)[1])
            except ValueError:
                return_code = None
        elif line.startswith("timed_out="):
            timed_out = line.split("=", 1)[1].strip().lower() == "true"
    return return_code, timed_out


def image_artifacts(run_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(run_dir.glob(pattern))
    unique: list[Path] = []
    seen_hashes: set[str] = set()
    for image_path in sorted(images, key=lambda path: (path.name.startswith("figure_"), path.name)):
        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique.append(image_path)
    return unique


def module_href(module: ModuleInfo) -> str:
    return f"/?module={quote(module.target, safe='')}"


def first_module_for_part(part_key: str, catalog: list[ModuleInfo]) -> ModuleInfo | None:
    for module in catalog:
        if module.part_key == part_key and module.path.exists():
            return module
    return None


def artifact_explanation(module: ModuleInfo, image_path: Path, index: int) -> tuple[str, str, str]:
    name = image_path.stem.lower()
    rules = [
        (("classic_kernel", "kernels", "kernel", "卷积核"), "卷积核在寻找什么", "把它当作一副小眼镜：不同数字模板会放大边缘、模糊、锐化或方向纹理。先比较同一张输入图经过不同卷积核后，哪里变亮、哪里被压低。", "卷积层的第一步不是理解物体，而是稳定地提取局部模式。"),
        (("receptive", "field", "感受野"), "感受野为什么会变大", "看横轴的层数和蓝色区域：层数越深，一个输出位置能回看输入图像的范围越大。", "深层 CNN 能组合更远处的信息，所以后层更容易表达部件和整体形状。"),
        (("pool", "pooling", "池化"), "池化在丢掉什么、保留什么", "观察最大池化和平均池化的区别：最大池化保留最强响应，平均池化保留局部总体趋势。", "池化会牺牲一些精确位置，换来更小的特征图和更强的平移鲁棒性。"),
        (("attention_alignment", "alignment"), "注意力如何形成对齐", "亮格表示当前词更依赖另一个词。先找每一行最亮的格子，看模型把信息从哪里取回来。", "注意力不是魔法解释器，但它能显示信息路由：谁在看谁。"),
        (("attention_vs_no_attention",), "有注意力和无注意力差在哪", "对比两组结果：没有注意力时信息容易被压进单一向量；有注意力时，解码过程可以回看输入的不同位置。", "注意力缓解了长序列信息瓶颈，让模型按需取信息。"),
        (("mha", "multihead", "head_specialization"), "多头注意力为什么要分头", "不同小图代表不同注意力头。看它们是否关注局部、远距离、特殊位置或整体平均。", "多头机制让模型同时用几种视角读同一句话。"),
        (("gradient_descent", "loss_surface"), "梯度下降在往哪里走", "轨迹通常从高损失区域移动到低损失区域。看它是否平稳靠近谷底，还是震荡、绕圈或停住。", "学习率和优化器决定了模型更新是稳步下降，还是跳过好答案。"),
        (("gradient_flow", "grad", "gradient"), "梯度有没有顺利传回去", "看不同层的梯度大小。太接近 0 说明学不动，突然很大说明可能爆炸。", "训练不是只看准确率；梯度健康决定参数能不能被有效更新。"),
        (("decision", "boundary", "xor"), "模型画出了怎样的分界线", "背景或曲线表示模型把空间分成了几类。先看边界有没有贴合数据主结构，再看有没有过分追噪声点。", "非线性网络的价值就在于能画出直线画不出的边界。"),
        (("feature", "activation", "map"), "特征图亮起来代表什么", "亮的区域表示该通道对某种局部模式响应强。多个通道并排看，就是模型的多种视觉探测器。", "从边缘到纹理再到语义，特征会逐层变抽象。"),
        (("confusion", "matrix"), "模型最容易混淆哪些类", "看非对角线哪里颜色深：那里表示真实类别被错判成另一个类别。", "混淆矩阵比总准确率更能告诉你下一步该补数据还是改模型。"),
        (("cam", "gradcam", "heatmap"), "模型决策时盯着哪里", "热区表示对最终判断贡献更大的图像区域。看热区是否落在真正有用的目标上。", "可解释性图不能完全证明因果，但能帮你发现模型是否看偏了。"),
        (("hidden", "state", "rnn", "lstm", "gru"), "序列记忆如何流动", "沿时间方向看隐藏状态或门控值如何变化：哪里保留，哪里遗忘，哪里突然更新。", "RNN 类模型的核心不是单个输入，而是历史信息怎样被压缩和传递。"),
        (("training", "accuracy", "loss", "curve"), "训练过程是否健康", "损失应整体下降，验证指标不应和训练指标越拉越开。震荡、发散、早早停住都值得检查。", "训练曲线是调参时最先看的仪表盘。"),
    ]
    for keywords, title, body, takeaway in rules:
        if any(keyword in name for keyword in keywords):
            return title, body, takeaway

    if module.part_key == "part2":
        return (
            "这张 CNN 图在说明什么",
            "先看颜色或亮度最强的区域，再对照标题判断它是在展示卷积、池化、特征图还是架构结构。",
            "视觉模型通常先提局部纹理，再把局部模式组合成更高层的形状。",
        )
    if module.part_key == "part3":
        return (
            "这张序列图在说明什么",
            "按时间顺序从左到右看，重点观察信息在哪里被保留、遗忘或重新加权。",
            "序列模型的难点是让早期信息在后面仍然可用。",
        )
    if module.part_key == "part4":
        return (
            "这张 Transformer 图在说明什么",
            "如果是热图，就看每行最亮的格子；如果是结构图，就看数据从嵌入、注意力、MLP 到输出的路径。",
            "Transformer 的核心是信息路由：每个位置如何从其他位置取信息。",
        )
    if module.part_key == "part5":
        return (
            "这张工具图在说明什么",
            "把它当成训练仪表盘：先找异常峰值、断崖式下降、长期不变或训练验证分离。",
            "工程调试要靠指标定位问题，而不是只凭最终分数猜。",
        )
    return (
        f"图 {index + 1} 应该怎么看",
        "先读图标题，再看坐标轴、颜色深浅和曲线趋势。不要急着看源码，先问：它想展示哪个变量变化后，结果发生了什么变化？",
        "图像的作用是把抽象概念变成可观察现象；看懂趋势比记住每个数字更重要。",
    )


def render_module_header(module: ModuleInfo) -> None:
    part = PARTS[module.part_key]
    tags = "".join(f'<span class="tag">{e(tag)}</span>' for tag in module.tags)
    st = __import__("streamlit")
    st.markdown(
        f"""
        <div class="module-hero">
          <div class="path-line">{e(part.title)} / {e(module.short_target)} / {e(module_kind(module))}</div>
          <h1>{e(module.title)}</h1>
          <p>{e(module.summary)}</p>
          <div>{tags}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_button() -> None:
    st = __import__("streamlit")
    st.markdown(
        """
        <style>
        .home-float {
            position: fixed;
            right: 1.1rem;
            top: 0.75rem;
            z-index: 999999;
            background: #172026;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 8px;
            padding: 0.48rem 0.72rem;
            text-decoration: none !important;
            font-size: 0.88rem;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        }
        .home-float:hover {
            background: #0f8b8d;
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            .home-float {
                right: 0.65rem;
                top: 0.55rem;
                padding: 0.42rem 0.58rem;
            }
        }
        </style>
        <a class="home-float" href="/" target="_self">返回主界面</a>
        """,
        unsafe_allow_html=True,
    )


def render_legacy_results(module: ModuleInfo, run_dir: Path) -> None:
    import streamlit as st

    return_code, timed_out = run_status(run_dir)
    stdout = read_run_text(run_dir, "stdout.txt")
    stderr = read_run_text(run_dir, "stderr.txt")
    images = image_artifacts(run_dir)

    if return_code == 0:
        st.success(f"运行完成，生成 {len(images)} 张图。")
    elif timed_out:
        st.warning("运行时间过长，已经停止。下面保留了已捕获的输出，方便判断卡在哪里。")
    else:
        st.error("脚本运行失败，但页面已经兜住错误；不会再白屏。")

    if images:
        st.subheader("运行生成的图")
        st.markdown(
            """
            <div class="lesson-note">
              读图顺序：先看标题和坐标轴，再找颜色最深、曲线突变或结构最密集的地方；
              最后回到问题本身，问它是在说明“模型看到了什么”“训练有没有学动”，还是“结构为什么这样设计”。
            </div>
            """,
            unsafe_allow_html=True,
        )
        for index, image_path in enumerate(images):
            title, body, takeaway = artifact_explanation(module, image_path, index)
            st.markdown(f"**{index + 1}. {title}**")
            left, right = st.columns([0.62, 0.38])
            with left:
                st.image(image_path.read_bytes(), caption=image_path.name, width="stretch")
            with right:
                st.markdown(
                    f"""
                    <div class="artifact-note">
                      <strong>这图看什么</strong>
                      <p>{e(body)}</p>
                      <strong>为什么重要</strong>
                      <p>{e(takeaway)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("这次运行没有生成图片。可以看下方输出，很多旧脚本主要通过控制台打印讲解。")

    with st.expander("控制台输出", expanded=return_code != 0):
        st.caption("这里保留脚本原来的逐步讲解、关键数字和公式推导。看图陌生时，先读这段输出通常更容易接上思路。")
        if stdout.strip():
            st.code(stdout[-12000:], language="text")
        else:
            st.caption("没有 stdout 输出。")

    if stderr.strip():
        with st.expander("错误与警告", expanded=True):
            st.code(stderr[-12000:], language="text")


def render_legacy_learning_guide(module: ModuleInfo) -> None:
    import streamlit as st

    guide = LEGACY_LEARNING_GUIDES.get(module.target)
    if not guide:
        return

    st.subheader("学习导读")
    st.markdown(
        """
        <div class="lesson-note">
          先把下面四张卡读完，再运行脚本。它们会告诉你该看哪张图、调哪些参数、遇到训练异常时先查哪里。
        </div>
        """,
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(guide), 2):
        cols = st.columns(2)
        for col, (title, body) in zip(cols, guide[row_start:row_start + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="artifact-note">
                      <strong>{e(title)}</strong>
                      <p>{e(body)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_legacy_module_page(module: ModuleInfo) -> None:
    import streamlit as st

    st.set_page_config(
        page_title=f"{module.title} - 深度学习书库",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(css("light"), unsafe_allow_html=True)
    render_home_button()
    render_module_header(module)

    st.markdown(
        """
        <div class="lesson-note">
          这个模块来自早期教材脚本：它原本面向命令行和 Matplotlib 弹窗，不是原生网页。
          现在已改成安全教学页：你可以先读目标和源码，再点击运行；运行失败也只会显示错误，不会让整个网站白屏。
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.38, 0.62])
    with left:
        st.subheader("怎么学")
        st.markdown(
            """
            1. 先读模块目标，知道它想解释什么。
            2. 点击“生成 / 更新运行结果”，把旧脚本输出成网页里的图片和日志。
            3. 如果脚本失败，直接看错误区；这通常是缺数据、下载被阻止，或脚本本身还停留在示例状态。
            """
        )
        st.code(f"python main.py {module.short_target}", language="bash")
        run_clicked = st.button("生成 / 更新运行结果", width="stretch")
    with right:
        st.subheader("模块目标")
        st.write(module.summary)
        render_module_card_html = render_module_card(module)
        st.markdown(render_module_card_html, unsafe_allow_html=True)

    render_legacy_learning_guide(module)

    if run_clicked:
        with st.spinner("正在安全运行旧脚本，并把 Matplotlib 图保存为网页图片..."):
            result = run_legacy_module(module)
        render_legacy_results(module, result["run_dir"])
    else:
        latest = latest_run_dir(module)
        if latest:
            st.subheader("上次运行结果")
            render_legacy_results(module, latest)
        else:
            st.info("还没有运行结果。点击上面的按钮后，这里会显示图像、控制台讲解和错误信息。")

    with st.expander("查看源码片段", expanded=False):
        st.code(read_text_preview(module.path), language="python")


def render_route_error(module: ModuleInfo | None, error: BaseException) -> None:
    import streamlit as st

    title = module.title if module else "未知模块"
    st.error(f"{title} 打开失败，但主站已经兜住异常。")
    st.caption("下面是完整错误，方便继续修这个具体模块。")
    st.code("".join(traceback.format_exception(error)), language="text")


def os_environ_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    return env


def css(theme: str) -> str:
    dark = theme == "dark"
    if dark:
        colors = {
            "app_bg": "#111418",
            "app_overlay": "rgba(17,20,24,0.96)",
            "panel": "#181d22",
            "panel_soft": "#1e252b",
            "ink": "#edf1f4",
            "muted": "#a9b4bd",
            "line": "#303941",
            "sidebar": "#151a1f",
            "accent": "#23a6a8",
            "accent_soft": "rgba(35,166,168,0.14)",
            "warm": "#d7a441",
            "rose": "#dc6f87",
        }
    else:
        colors = {
            "app_bg": "#f7f5ef",
            "app_overlay": "rgba(255,255,255,0.88)",
            "panel": "#ffffff",
            "panel_soft": "#eef4f2",
            "ink": "#172026",
            "muted": "#596772",
            "line": "#d8dee3",
            "sidebar": "#eef4f2",
            "accent": "#0f8b8d",
            "accent_soft": "rgba(15,139,141,0.10)",
            "warm": "#b9801d",
            "rose": "#bf3f5b",
        }
    return f"""
    <style>
    :root {{
        --app-bg: {colors["app_bg"]};
        --app-overlay: {colors["app_overlay"]};
        --panel: {colors["panel"]};
        --panel-soft: {colors["panel_soft"]};
        --ink: {colors["ink"]};
        --muted: {colors["muted"]};
        --line: {colors["line"]};
        --sidebar: {colors["sidebar"]};
        --accent: {colors["accent"]};
        --accent-soft: {colors["accent_soft"]};
        --warm: {colors["warm"]};
        --rose: {colors["rose"]};
    }}
    .stApp {{
        background:
            linear-gradient(180deg, var(--app-overlay) 0%, var(--panel-soft) 100%),
            var(--app-bg);
        color: var(--ink);
    }}
    .block-container {{
        max-width: 1280px;
        padding-top: 1.1rem;
        padding-bottom: 2.4rem;
    }}
    h1, h2, h3, p, li, label, span {{
        letter-spacing: 0;
    }}
    h1, h2, h3, [data-testid="stMarkdownContainer"] strong {{
        color: var(--ink);
    }}
    section[data-testid="stSidebar"] {{
        background: var(--sidebar);
        border-right: 1px solid var(--line);
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--ink);
    }}
    .hero {{
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
        gap: 1rem;
        align-items: stretch;
        border-bottom: 1px solid var(--line);
        padding: 0.35rem 0 1rem 0;
        margin-bottom: 1rem;
    }}
    .hero h1 {{
        font-size: clamp(2.2rem, 3.4vw, 3.7rem);
        line-height: 1.08;
        margin: 0;
    }}
    .hero p {{
        color: var(--muted);
        max-width: 920px;
        line-height: 1.75;
        margin: 0.55rem 0 0 0;
        font-size: 1.02rem;
    }}
    .hero-panel, .feature-card, .module-card, .recommend-card, .stat-card {{
        background: color-mix(in srgb, var(--panel) 88%, transparent);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
    }}
    .hero-panel {{
        padding: 0.95rem;
        min-height: 100%;
    }}
    .hero-panel .kicker, .recommend-card .kicker {{
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}
    .hero-panel strong {{
        display: block;
        font-size: 1.08rem;
        margin-bottom: 0.25rem;
    }}
    .grid, .feature-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.7rem 0 1rem 0;
    }}
    .feature-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .feature-card, .module-card, .recommend-card, .stat-card {{
        padding: 0.82rem 0.92rem;
    }}
    .feature-card strong, .module-card strong, .stat-card strong {{
        display: block;
        margin-bottom: 0.34rem;
    }}
    .feature-card p, .module-card p, .recommend-card p, .stat-card p {{
        color: var(--muted);
        margin: 0;
        line-height: 1.6;
        font-size: 0.92rem;
    }}
    .tag {{
        display: inline-block;
        margin: 0.45rem 0.32rem 0 0;
        padding: 0.14rem 0.44rem;
        border: 1px solid color-mix(in srgb, var(--accent) 34%, transparent);
        border-radius: 999px;
        color: var(--accent);
        font-size: 0.78rem;
        background: var(--accent-soft);
    }}
    .path-line {{
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.38rem;
    }}
    .recommend-card {{
        border-left: 4px solid var(--warm);
        border-radius: 0 8px 8px 0;
        margin: 0.45rem 0 0.9rem 0;
    }}
    .module-hero {{
        border-bottom: 1px solid var(--line);
        padding: 0.25rem 0 0.95rem 0;
        margin-bottom: 0.9rem;
    }}
    .module-hero h1 {{
        font-size: clamp(1.9rem, 2.8vw, 3rem);
        line-height: 1.1;
        margin: 0;
    }}
    .module-hero p {{
        color: var(--muted);
        line-height: 1.75;
        max-width: 920px;
        margin: 0.5rem 0 0 0;
    }}
    .lesson-note {{
        border-left: 4px solid var(--accent);
        background: color-mix(in srgb, var(--panel) 82%, transparent);
        padding: 0.75rem 0.9rem;
        margin: 0.55rem 0 0.9rem 0;
        border-radius: 0 8px 8px 0;
        color: var(--muted);
        line-height: 1.68;
    }}
    .artifact-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.5rem 0 1rem 0;
    }}
    .artifact-note {{
        background: color-mix(in srgb, var(--panel) 86%, transparent);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.78rem 0.88rem;
        margin-bottom: 0.85rem;
    }}
    .artifact-note strong {{
        display: block;
        color: var(--ink);
        margin: 0.15rem 0 0.25rem 0;
    }}
    .artifact-note p {{
        color: var(--muted);
        line-height: 1.66;
        margin: 0 0 0.62rem 0;
        font-size: 0.92rem;
    }}
    .module-card-link, .stat-card-link {{
        display: block;
        color: inherit !important;
        text-decoration: none !important;
        cursor: pointer;
        transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }}
    .module-card-link:hover, .stat-card-link:hover {{
        transform: translateY(-1px);
        border-color: color-mix(in srgb, var(--accent) 55%, var(--line));
        box-shadow: 0 14px 34px rgba(0,0,0,0.10);
    }}
    .module-card-link:focus, .stat-card-link:focus {{
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }}
    .search-result {{
        padding: 0.55rem 0;
        border-bottom: 1px solid var(--line);
    }}
    div[data-testid="stMetric"] {{
        background: color-mix(in srgb, var(--panel) 90%, transparent);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
    }}
    .stButton > button {{
        border-radius: 8px;
        min-height: 2.4rem;
        font-weight: 700;
    }}
    @media (max-width: 1080px) {{
        .hero, .grid, .feature-grid, .artifact-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """


def render_module_card(module: ModuleInfo) -> str:
    tags = "".join(f'<span class="tag">{e(tag)}</span>' for tag in module.tags[:4])
    return (
        f'<a class="module-card module-card-link" href="{module_href(module)}" target="_self" aria-label="打开 {e(module.title)}">'
        f"<strong>{e(module.title)}</strong>"
        f"<p>{e(module.summary)}</p>"
        f'<div class="path-line">{e(module.short_target)}</div>'
        f"{tags}"
        "</a>"
    )


@lru_cache(maxsize=1)
def render_feature_cards() -> str:
    features = [
        ("🧭 六大模块导航", "按学习阶段和主题分组，新增页面会自动注册到侧边栏。"),
        ("🔎 全局搜索", "同时检索标题、摘要、标签和文件路径，适合快速定位知识点。"),
        ("🎲 今日推荐", "每天随机抽取一个知识点，给复习和探索一个明确入口。"),
        ("🌗 主题切换", "亮色适合阅读，暗色适合长时间实验和演示。"),
    ]
    cards = "".join(
        '<div class="feature-card">'
        f"<strong>{e(title)}</strong>"
        f"<p>{e(text)}</p>"
        "</div>"
        for title, text in features
    )
    return f'<div class="feature-grid">{cards}</div>'


def search_modules(query: str, modules: list[ModuleInfo]) -> list[ModuleInfo]:
    terms = [term.lower() for term in query.strip().split() if term.strip()]
    if not terms:
        return []

    scored: list[tuple[int, ModuleInfo]] = []
    for module in modules:
        haystack = " ".join(
            [
                module.title,
                module.summary,
                module.level,
                module.target,
                module.short_target,
                " ".join(module.tags),
            ]
        ).lower()
        if not all(term in haystack for term in terms):
            continue
        score = 0
        for term in terms:
            if term in module.title.lower():
                score += 8
            if term in " ".join(module.tags).lower():
                score += 5
            if term in module.summary.lower():
                score += 3
            if term in module.target.lower():
                score += 2
        scored.append((score * 100 - module.priority, module))

    scored.sort(reverse=True, key=lambda item: item[0])
    return [module for _, module in scored]


def matching_modules(level: str, interest: str, catalog: list[ModuleInfo]) -> list[ModuleInfo]:
    if level == "刚入门":
        keys = {"part1", "part2"}
    elif level == "已有基础":
        keys = {"part2", "part3", "part4", "part5"}
    else:
        keys = {"part4", "part5", "part6"}

    interest_map = {
        "看懂深度学习整体": {"基础", "神经网络", "实验", "Transformer"},
        "计算机视觉": {"视觉", "CNN", "表征", "解释性"},
        "自然语言与大模型": {"NLP", "Transformer", "LLM", "AGI", "注意力"},
        "工程落地": {"训练", "数据", "部署", "工具", "框架", "架构", "项目"},
        "前沿研究": {"LLM", "AGI", "安全", "生成模型", "GNN", "表征", "性能"},
    }
    wanted = interest_map[interest]
    scored: list[tuple[int, ModuleInfo]] = []
    for module in catalog:
        score = 0
        if module.part_key in keys:
            score += 3
        score += len(set(module.tags) & wanted) * 4
        if module.level in {"核心", "前沿"} and interest in {"自然语言与大模型", "前沿研究"}:
            score += 2
        if score > 0 and module.path.exists():
            scored.append((score * 100 - module.priority, module))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [module for _, module in scored[:6]]


def daily_recommendation(catalog: list[ModuleInfo]) -> tuple[str, str, ModuleInfo]:
    seed_text = f"{date.today().isoformat()}|deep-learning-book"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    title, text, target = rng.choice(KNOWLEDGE_POINTS)
    routes = route_map(catalog)
    module = routes.get(target) or rng.choice(catalog)
    return title, text, module


def open_module(module: ModuleInfo) -> None:
    import streamlit as st

    st.query_params["module"] = module.target
    st.rerun()


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


def render_sidebar(catalog: list[ModuleInfo]) -> None:
    import streamlit as st

    st.header("🧭 学习导航")
    st.caption("六大模块分组；目录中的新页面会自动进入这里。")

    query = st.text_input("🔎 全局搜索", placeholder="例如 Transformer / 梯度 / 部署 / 可视化")
    results = search_modules(query, catalog)
    if query:
        st.caption(f"找到 {len(results)} 个结果")
        for module in results[:8]:
            if st.button(module.title, key=f"search-{module.target}", width="stretch"):
                open_module(module)
        st.divider()

    for part_key, part in PARTS.items():
        part_modules = [module for module in catalog if module.part_key == part_key]
        with st.expander(f"{part.emoji} {part.short_title} · {len(part_modules)}", expanded=part_key in {"part1", "part6"}):
            for module in part_modules:
                label = f"{module.title}"
                if st.button(label, key=f"nav-{module.target}", width="stretch"):
                    open_module(module)

    st.divider()
    st.caption("命令行示例")
    st.code("python main.py part6/frontier", language="bash")


def render_streamlit_home() -> None:
    import streamlit as st

    catalog = module_catalog()
    routes = route_map(catalog)
    query_module = get_query_module()
    if query_module:
        module = routes.get(query_module)
        if module and module.path.exists() and module.path.resolve() != Path(__file__).resolve():
            if not is_streamlit_app(module.path):
                render_legacy_module_page(module)
                return

            try:
                runpy.run_path(str(module.path), run_name="__main__")
                render_home_button()
            except Exception as exc:
                st.set_page_config(
                    page_title=f"{module.title} - 打开失败",
                    page_icon="🧠",
                    layout="wide",
                    initial_sidebar_state="collapsed",
                )
                st.markdown(css("light"), unsafe_allow_html=True)
                render_home_button()
                render_module_header(module)
                render_route_error(module, exc)
            return
        st.set_page_config(
            page_title="模块不存在 - 深度学习书库",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(css("light"), unsafe_allow_html=True)
        render_home_button()
        st.error(f"没有找到模块：{query_module}")
        st.link_button("返回首页", "/")
        return

    st.set_page_config(
        page_title="深度学习书库",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        dark_mode = st.toggle("🌗 暗色模式", value=False)
        render_sidebar(catalog)

    st.markdown(css("dark" if dark_mode else "light"), unsafe_allow_html=True)

    available = [module for module in catalog if module.path.exists()]
    tags = sorted({tag for module in available for tag in module.tags})
    levels = sorted({module.level for module in available})
    today_title, today_text, today_module = daily_recommendation(available)

    st.markdown(
        f"""
        <div class="hero">
          <div>
            <h1>深度学习书库</h1>
            <p>
              一个用 Streamlit 组织的交互式深度学习学习台：从数学、神经网络和 CNN/RNN，
              一路到 Transformer、训练工具、工程框架与前沿方向。主页现在承担导航、搜索、
              推荐和学习统计四件事，帮助你快速进入下一块知识。
            </p>
          </div>
          <div class="hero-panel">
            <div class="kicker">今日推荐</div>
            <strong>{e(today_title)}</strong>
            <p>{e(today_text)}</p>
            <div class="path-line">推荐入口：{e(today_module.short_target)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(f"打开今日推荐：{today_module.title}", width="stretch"):
        open_module(today_module)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("六大模块", len(PARTS))
    c2.metric("导航页面", len(available))
    c3.metric("知识标签", len(tags))
    c4.metric("学习层级", len(levels))

    st.subheader("功能亮点")
    st.markdown(render_feature_cards(), unsafe_allow_html=True)

    st.subheader("学习路径推荐")
    left, right = st.columns([0.38, 0.62])
    with left:
        level = st.segmented_control("当前水平", ["刚入门", "已有基础", "准备进阶"], default="已有基础")
        interest = st.selectbox(
            "学习目标",
            ["看懂深度学习整体", "计算机视觉", "自然语言与大模型", "工程落地", "前沿研究"],
            index=2,
        )
        st.markdown(
            """
            <div class="recommend-card">
              <div class="kicker">路径逻辑</div>
              <p>先补必要概念，再进入核心实验，最后用工程工具和前沿主题巩固迁移能力。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        recommended = matching_modules(level or "已有基础", interest, available)
        for index, module in enumerate(recommended, 1):
            cols = st.columns([0.72, 0.28])
            with cols[0]:
                st.markdown(f"**{index}. {module.title}**  `{module.short_target}`")
                st.caption(module.summary)
            with cols[1]:
                if st.button("打开", key=f"rec-{module.target}", width="stretch"):
                    open_module(module)

    st.subheader("核心与前沿模块")
    core = [module for module in available if module.level in {"核心", "前沿"}]
    st.markdown('<div class="grid">' + "".join(render_module_card(module) for module in core) + "</div>", unsafe_allow_html=True)

    st.subheader("全书模块概览")
    tabs = st.tabs([part.label for part in PARTS.values()])
    for tab, (part_key, part) in zip(tabs, PARTS.items()):
        with tab:
            st.write(part.description)
            part_modules = [module for module in available if module.part_key == part_key]
            st.markdown(
                '<div class="grid">' + "".join(render_module_card(module) for module in part_modules) + "</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("学习统计")
    part_lines = []
    for part_key, part in PARTS.items():
        count = len([module for module in available if module.part_key == part_key])
        first_module = first_module_for_part(part_key, available)
        href = module_href(first_module) if first_module else "/"
        part_lines.append(
            f'<a class="stat-card stat-card-link" href="{href}" target="_self" aria-label="打开 {e(part.short_title)} 章节">'
            f"<strong>{part.emoji} {e(part.short_title)}</strong>"
            f"<p>{count} 个页面 · {e(part.description)}</p>"
            "</a>"
        )
    st.markdown('<div class="grid">' + "".join(part_lines) + "</div>", unsafe_allow_html=True)


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:
            return True
    except Exception:
        pass

    streamlit_frames = ("streamlit\\testing", "streamlit\\runtime\\scriptrunner")
    return any(
        any(marker in frame.filename.replace("/", "\\") for marker in streamlit_frames)
        for frame in inspect.stack()
    )


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

    run_streamlit_app(Path(__file__).resolve())
    return 0


if __name__ == "__main__":
    if running_under_streamlit():
        main()
    else:
        raise SystemExit(main())
