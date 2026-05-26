"""
Project-wide runtime polish.

Python imports sitecustomize automatically when this directory is on sys.path.
That lets the teaching scripts share readable matplotlib defaults without
modifying every generated lesson file.
"""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT_ROOT / ".streamlit_module_outputs"
ROOT_ARTIFACT_RUN_DIR = ARTIFACT_ROOT / "direct_script_runs"
ARTIFACT_DIR_ENV = "DL_BOOK_ARTIFACT_DIR"


sys.dont_write_bytecode = True


def _redirect_artifact_path(path_like: object) -> object:
    """Route simple root-level runtime artifacts into the shared output dir."""

    if not isinstance(path_like, (str, os.PathLike)):
        return path_like
    path = Path(path_like)
    if path.is_absolute() or path.parent != Path("."):
        return path_like
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".csv", ".pt", ".pth", ".ckpt", ".log"}:
        return path_like
    target_dir = Path(os.environ.get(ARTIFACT_DIR_ENV, ROOT_ARTIFACT_RUN_DIR))
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / path.name


def _cleanup_root_pycache() -> None:
    cache_dir = PROJECT_ROOT / "__pycache__"
    if not cache_dir.exists():
        return
    try:
        for cache_file in cache_dir.glob("sitecustomize*.pyc"):
            cache_file.unlink(missing_ok=True)
        if not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    except OSError:
        pass


def _mode_writes_file(mode: object) -> bool:
    if not isinstance(mode, str):
        return False
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _configure_file_output_redirection() -> None:
    if getattr(builtins.open, "_deep_learning_book_redirects_artifacts", False):
        return

    original_open = builtins.open
    original_path_open = Path.open
    original_path_write_text = Path.write_text
    original_path_write_bytes = Path.write_bytes

    def redirected_open(file, mode="r", *args, **kwargs):
        target = _redirect_artifact_path(file) if _mode_writes_file(mode) else file
        return original_open(target, mode, *args, **kwargs)

    def redirected_path_open(self, mode="r", *args, **kwargs):
        target = _redirect_artifact_path(self) if _mode_writes_file(mode) else self
        if target is self:
            return original_path_open(self, mode, *args, **kwargs)
        return original_open(target, mode, *args, **kwargs)

    def redirected_path_write_text(self, data, *args, **kwargs):
        target = _redirect_artifact_path(self)
        if target is self:
            return original_path_write_text(self, data, *args, **kwargs)
        return Path(target).write_text(data, *args, **kwargs)

    def redirected_path_write_bytes(self, data, *args, **kwargs):
        target = _redirect_artifact_path(self)
        if target is self:
            return original_path_write_bytes(self, data, *args, **kwargs)
        return Path(target).write_bytes(data, *args, **kwargs)

    redirected_open._deep_learning_book_redirects_artifacts = True
    redirected_path_open._deep_learning_book_redirects_artifacts = True
    redirected_path_write_text._deep_learning_book_redirects_artifacts = True
    redirected_path_write_bytes._deep_learning_book_redirects_artifacts = True
    builtins.open = redirected_open
    Path.open = redirected_path_open
    Path.write_text = redirected_path_write_text
    Path.write_bytes = redirected_path_write_bytes


def _configure_matplotlib() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib import font_manager
    except Exception:
        return

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "#fbfaf6",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#c9d2d6",
            "axes.labelcolor": "#172026",
            "xtick.color": "#34444d",
            "ytick.color": "#34444d",
            "grid.color": "#cfd8dc",
            "grid.alpha": 0.32,
            "lines.linewidth": 2.2,
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )

    if getattr(plt.show, "_deep_learning_book_closes_figures", False):
        return

    original_show = plt.show
    original_pyplot_savefig = plt.savefig
    original_figure_savefig = Figure.savefig

    def redirected_pyplot_savefig(fname, *args, **kwargs):
        return original_pyplot_savefig(_redirect_artifact_path(fname), *args, **kwargs)

    def redirected_figure_savefig(self, fname, *args, **kwargs):
        return original_figure_savefig(self, _redirect_artifact_path(fname), *args, **kwargs)

    def show_and_close(*args, **kwargs):
        try:
            return original_show(*args, **kwargs)
        finally:
            plt.close("all")

    redirected_pyplot_savefig._deep_learning_book_redirects_artifacts = True
    redirected_figure_savefig._deep_learning_book_redirects_artifacts = True
    show_and_close._deep_learning_book_closes_figures = True
    plt.savefig = redirected_pyplot_savefig
    Figure.savefig = redirected_figure_savefig
    plt.show = show_and_close


def _configure_torch_save() -> None:
    try:
        import torch
    except Exception:
        return

    if getattr(torch.save, "_deep_learning_book_redirects_artifacts", False):
        return

    original_save = torch.save

    def redirected_torch_save(obj, f, *args, **kwargs):
        return original_save(obj, _redirect_artifact_path(f), *args, **kwargs)

    redirected_torch_save._deep_learning_book_redirects_artifacts = True
    torch.save = redirected_torch_save


def _configure_pandas_csv() -> None:
    try:
        import pandas as pd
    except Exception:
        return

    if getattr(pd.DataFrame.to_csv, "_deep_learning_book_redirects_artifacts", False):
        return

    original_to_csv = pd.DataFrame.to_csv

    def redirected_to_csv(self, path_or_buf=None, *args, **kwargs):
        return original_to_csv(self, _redirect_artifact_path(path_or_buf), *args, **kwargs)

    redirected_to_csv._deep_learning_book_redirects_artifacts = True
    pd.DataFrame.to_csv = redirected_to_csv


_configure_file_output_redirection()
_configure_matplotlib()
_configure_torch_save()
_configure_pandas_csv()
_cleanup_root_pycache()
