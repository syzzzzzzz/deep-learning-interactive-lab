"""Small runtime helpers for refactored teaching lessons."""

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


def run_cli(compute: Callable[..., dict[str, Any]], **kwargs: Any) -> int:
    """Run a compute function from the command line and print useful artifacts."""

    configure_stdio()
    try:
        result = compute(save_artifacts=True, **kwargs)
        log = str(result.get("log", "")).strip()
        if log:
            print(log)
        for path in result.get("artifacts", []):
            print(f"图像已保存: {path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - command line should expose full traceback.
        traceback.print_exception(exc)
        return 1
