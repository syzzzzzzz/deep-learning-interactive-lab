from __future__ import annotations

import runpy
import sys
from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def _load(path: Path) -> dict[str, object]:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return runpy.run_path(str(path))
    finally:
        sys.dont_write_bytecode = previous


def check_course_source_of_truth(context: QualityCheckContext) -> None:
    """Verify the course manifest is the single source for parts and modules."""

    main_text = context.read_text(Path("main.py"))
    graph_text = context.read_text(Path("components/knowledge_graph.py"))
    manifest_text = context.read_text(Path("components/course_manifest.py"))
    failures: list[str] = []

    required_manifest = [
        "PARTS: dict[str, PartInfo]",
        "MODULES: list[ModuleInfo]",
        "REGISTERED_MODULES: tuple[ModuleInfo, ...]",
    ]
    for fragment in required_manifest:
        if fragment not in manifest_text:
            failures.append(f"components/course_manifest.py 缺少课程事实契约：{fragment}")

    forbidden_main = [
        "PARTS: dict[str, PartInfo]",
        "MODULES: list[ModuleInfo]",
        "PartInfo(",
        "ModuleInfo(",
    ]
    for fragment in forbidden_main:
        if fragment in main_text:
            failures.append(f"main.py 仍手写课程事实：{fragment}")

    required_main = [
        "from components.course_manifest import PARTS",
        "from components.course_manifest import REGISTERED_MODULES as MODULES",
    ]
    for fragment in required_main:
        if fragment not in main_text:
            failures.append(f"main.py 未从 course_manifest 读取课程事实：{fragment}")

    required_graph = [
        "from components.course_manifest import REGISTERED_MODULES",
        "def build_module_seeds_from_course_manifest",
        "LEGACY_MODULE_SEED_OVERRIDES",
        "MODULE_SEEDS: tuple[ModuleSeed, ...] = build_module_seeds_from_course_manifest",
    ]
    for fragment in required_graph:
        if fragment not in graph_text:
            failures.append(f"knowledge_graph.py 未从课程事实派生节点：{fragment}")

    manifest = _load(context.root / "components" / "course_manifest.py")
    graph = _load(context.root / "components" / "knowledge_graph.py")
    registered = manifest["REGISTERED_MODULES"]
    graph_keys = graph["canonical_node_keys"]()
    manifest_keys = [module.short_target for module in registered]  # type: ignore[attr-defined]
    if graph_keys != manifest_keys:
        failures.append("知识图谱 canonical_node_keys 与 course_manifest.REGISTERED_MODULES 顺序/覆盖不一致")

    if failures:
        raise QualityCheckFailure("课程目录 Source of Truth 检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] 课程目录 Source of Truth 检查：主站、知识图谱和质量门均从 course_manifest 派生课程事实")
