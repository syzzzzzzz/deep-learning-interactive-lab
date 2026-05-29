"""Unified runtime for legacy teaching lessons.

This module is the shared seam for old scripts, whether they use a
LegacyLessonSpec page adapter or a hand-written Streamlit render function.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any


def running_under_streamlit() -> bool:
    """Return True when the current script is executed by Streamlit."""

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def configure_stdio() -> None:
    """Use UTF-8 streams so Chinese console output is readable on Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def clamp_int(value: int, minimum: int, maximum: int, name: str) -> int:
    """Validate an integer parameter and clamp it to a safe teaching range."""

    if minimum > maximum:
        raise ValueError(f"{name} 的最小值不能大于最大值")
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def clamp_float(value: float, minimum: float, maximum: float, name: str) -> float:
    """Validate a float parameter and clamp it to a safe teaching range."""

    if minimum > maximum:
        raise ValueError(f"{name} 的最小值不能大于最大值")
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def normalize_legacy_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Fill optional payload fields so downstream runners see one shape."""

    normalized = dict(result)
    normalized.setdefault("log", "")
    normalized.setdefault("figures", [])
    normalized.setdefault("artifacts", [])
    normalized.setdefault("rows", [])
    normalized.setdefault("notes", [])
    return normalized


def run_cli(compute: Callable[..., dict[str, Any]], **kwargs: Any) -> int:
    """Run a compute function from the command line and print useful artifacts."""

    configure_stdio()
    try:
        result = normalize_legacy_payload(compute(save_artifacts=True, **kwargs))
        log = str(result.get("log", "")).strip()
        if log:
            print(log)
        for path in result.get("artifacts", []):
            print(f"图像已保存: {path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - command line should expose full traceback.
        traceback.print_exception(exc)
        return 1


def run_or_render(compute: Callable[..., dict[str, Any]], render: Callable[[], None]) -> int | None:
    """Use Streamlit render in app mode and CLI compute in direct script mode."""

    if running_under_streamlit():
        render()
        return None
    return run_cli(compute)


def legacy_entrypoint(compute: Callable[..., dict[str, Any]], render: Callable[[], None]) -> None:
    """Raise SystemExit only for CLI mode, keeping Streamlit runs alive."""

    result = run_or_render(compute, render)
    if result is not None:
        raise SystemExit(result)
