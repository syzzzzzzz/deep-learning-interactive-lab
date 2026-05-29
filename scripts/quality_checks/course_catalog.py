from __future__ import annotations

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

from .common import QualityCheckContext, QualityCheckFailure


def load_main_without_bytecode(project_root: Path) -> SimpleNamespace:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        namespace = runpy.run_path(str(project_root / "main.py"))
    finally:
        sys.dont_write_bytecode = previous
    return SimpleNamespace(**namespace)


def check_course_catalog_module(context: QualityCheckContext) -> None:
    """Verify the course catalog module owns routing and recommendation logic."""

    from components.course_catalog import (
        ModuleInfo,
        build_module_catalog,
        build_route_map,
        daily_recommendation,
        list_part_modules,
        matching_modules,
        search_modules,
    )
    main = load_main_without_bytecode(context.root)

    failures: list[str] = []
    main_text = context.read_text(Path("main.py"))
    catalog_text = context.read_text(Path("components/course_catalog.py"))

    required_exports = [
        "class PartInfo",
        "class ModuleInfo",
        "def build_module_catalog",
        "def build_route_map",
        "def list_part_modules",
        "def search_modules",
        "def matching_modules",
        "def daily_recommendation",
    ]
    for fragment in required_exports:
        if fragment not in catalog_text:
            failures.append(f"components/course_catalog.py 缺少课程目录契约：{fragment}")
    for old_fragment in [
        "def search_modules(",
        "def matching_modules(",
        "def fallback_title(",
        "from dataclasses import dataclass",
    ]:
        if old_fragment in main_text:
            failures.append(f"main.py 仍保留课程目录实现细节：{old_fragment}")

    catalog = list(main.module_catalog())
    routes = main.route_map(catalog)
    if len(catalog) < 60:
        failures.append(f"课程目录数量异常：当前仅 {len(catalog)} 个模块")
    for target in ["part1/math_primer", "part1_foundations/math_primer"]:
        if target not in routes:
            failures.append(f"路由别名缺失：{target}")
    math_module = routes.get("part1/math_primer")
    if math_module and math_module.path.name != "math_primer.py":
        failures.append("part1/math_primer 未指向数学基础速查模块")

    discovered = build_module_catalog(main.BASE_DIR, main.PARTS, main.MODULES)
    if [module.target for module in discovered] != [module.target for module in catalog]:
        failures.append("build_module_catalog 与 main.module_catalog 结果不一致")
    if list_part_modules(main.BASE_DIR, main.PARTS, "part1") and "math_primer" not in list_part_modules(
        main.BASE_DIR, main.PARTS, "part1"
    ):
        failures.append("list_part_modules 未列出 part1/math_primer")
    if not search_modules("Transformer 注意力", catalog):
        failures.append("search_modules 无法检索 Transformer 注意力")
    if not matching_modules("刚入门", "看懂深度学习整体", catalog):
        failures.append("matching_modules 未给零基础路径返回推荐")

    title, text, module = daily_recommendation(catalog, main.KNOWLEDGE_POINTS)
    if not title or not text or not isinstance(module, ModuleInfo):
        failures.append("daily_recommendation 未返回完整推荐结果")
    if not build_route_map(catalog).get(module.target):
        failures.append("daily_recommendation 返回了不在路由表中的模块")

    if failures:
        raise QualityCheckFailure("课程目录模块检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] 课程目录模块检查：目录发现、路由别名、搜索和推荐均由 course_catalog 承担")
