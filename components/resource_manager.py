"""Shared resource helpers for Streamlit teaching modules."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator


OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".streamlit_module_outputs"


@contextmanager
def safe_mpl_figure(*args: object, **kwargs: object) -> Iterator[object]:
    """Create a Matplotlib figure and close it automatically."""

    import matplotlib.pyplot as plt

    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


def get_artifact_path(filename: str) -> Path:
    """Return a path inside the unified Streamlit module output directory."""

    clean_name = Path(filename).name
    if not clean_name:
        raise ValueError("filename must not be empty")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / clean_name


def clean_old_artifacts(max_age_hours: int = 24) -> int:
    """Delete files in the artifact directory older than max_age_hours."""

    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    if not OUTPUT_DIR.exists():
        return 0

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    removed = 0
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified < cutoff:
            path.unlink()
            removed += 1
    return removed
