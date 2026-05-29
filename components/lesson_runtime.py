"""Compatibility facade for older teaching scripts.

The runtime implementation lives in :mod:`components.legacy_runtime`.  Keep
this module thin so old imports continue to work while new code depends on the
single runtime source directly.
"""

from __future__ import annotations

from components.legacy_runtime import clamp_float
from components.legacy_runtime import clamp_int
from components.legacy_runtime import configure_stdio
from components.legacy_runtime import legacy_entrypoint
from components.legacy_runtime import normalize_legacy_payload
from components.legacy_runtime import run_cli
from components.legacy_runtime import run_or_render
from components.legacy_runtime import running_under_streamlit

__all__ = [
    "clamp_float",
    "clamp_int",
    "configure_stdio",
    "legacy_entrypoint",
    "normalize_legacy_payload",
    "run_cli",
    "run_or_render",
    "running_under_streamlit",
]
