from __future__ import annotations

import runpy
import sys
import importlib
from pathlib import Path
from typing import Any

from .common import QualityCheckContext, QualityCheckFailure


def _load(path: Path) -> dict[str, Any]:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return runpy.run_path(str(path))
    finally:
        sys.dont_write_bytecode = previous


def check_legacy_runtime_module(context: QualityCheckContext) -> None:
    """Verify legacy runtime behavior has one implementation and thin adapters."""

    runtime_text = context.read_text(Path("components/legacy_runtime.py"))
    lesson_text = context.read_text(Path("components/lesson_runtime.py"))
    protocol_text = context.read_text(Path("components/legacy_protocol.py"))
    failures: list[str] = []

    required_runtime_functions = [
        "def running_under_streamlit",
        "def configure_stdio",
        "def clamp_int",
        "def clamp_float",
        "def normalize_legacy_payload",
        "def run_cli",
        "def run_or_render",
        "def legacy_entrypoint",
    ]
    for fragment in required_runtime_functions:
        if fragment not in runtime_text:
            failures.append(f"components/legacy_runtime.py 缺少统一 runtime 实现：{fragment}")

    forbidden_lesson_defs = [
        "def running_under_streamlit",
        "def configure_stdio",
        "def clamp_int",
        "def clamp_float",
        "def run_cli",
        "def run_or_render",
    ]
    for fragment in forbidden_lesson_defs:
        if fragment in lesson_text:
            failures.append(f"components/lesson_runtime.py 仍然保留重复实现：{fragment}")

    required_lesson_facade = [
        "from components.legacy_runtime import clamp_float",
        "from components.legacy_runtime import clamp_int",
        "from components.legacy_runtime import run_cli",
        "from components.legacy_runtime import run_or_render",
        "from components.legacy_runtime import running_under_streamlit",
        "__all__",
    ]
    for fragment in required_lesson_facade:
        if fragment not in lesson_text:
            failures.append(f"components/lesson_runtime.py 兼容 facade 缺少导出：{fragment}")

    required_protocol_imports = [
        "from components.legacy_runtime import run_cli, run_or_render, running_under_streamlit",
    ]
    for fragment in required_protocol_imports:
        if fragment not in protocol_text:
            failures.append(f"components/legacy_protocol.py 没有直接依赖统一 runtime：{fragment}")
    if "from components.lesson_runtime import" in protocol_text:
        failures.append("components/legacy_protocol.py 仍然通过 lesson_runtime 间接拿 runtime")
    if "def run_or_render" in protocol_text:
        failures.append("components/legacy_protocol.py 仍然重复定义 run_or_render")

    runtime = _load(context.root / "components" / "legacy_runtime.py")
    if runtime["clamp_int"](99, 1, 3, "x") != 3:
        failures.append("legacy_runtime.clamp_int 没有正确裁剪上界")
    if runtime["clamp_float"](-2.0, 0.2, 0.8, "x") != 0.2:
        failures.append("legacy_runtime.clamp_float 没有正确裁剪下界")
    normalized = runtime["normalize_legacy_payload"]({"rows": [1]})
    for key in ("log", "figures", "artifacts", "rows", "notes"):
        if key not in normalized:
            failures.append(f"legacy_runtime.normalize_legacy_payload 缺少标准字段：{key}")
    runtime_module = importlib.import_module("components.legacy_runtime")
    lesson_module = importlib.import_module("components.lesson_runtime")
    if lesson_module.run_cli is not runtime_module.run_cli:
        failures.append("lesson_runtime.run_cli 不是 legacy_runtime.run_cli 的兼容导出")
    if lesson_module.run_or_render is not runtime_module.run_or_render:
        failures.append("lesson_runtime.run_or_render 不是 legacy_runtime.run_or_render 的兼容导出")

    if failures:
        raise QualityCheckFailure("旧脚本 runtime 统一检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] 旧脚本 runtime 统一检查：实现集中在 legacy_runtime，lesson_runtime 保持兼容外壳，legacy_protocol 直接消费统一 runtime")
