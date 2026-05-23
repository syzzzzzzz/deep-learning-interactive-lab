"""
Project-wide runtime polish.

Python imports sitecustomize automatically when this directory is on sys.path.
That lets the teaching scripts share readable matplotlib defaults without
modifying every generated lesson file.
"""

from __future__ import annotations


def _configure_matplotlib() -> None:
    try:
        import matplotlib.pyplot as plt
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

    def show_and_close(*args, **kwargs):
        try:
            return original_show(*args, **kwargs)
        finally:
            plt.close("all")

    show_and_close._deep_learning_book_closes_figures = True
    plt.show = show_and_close


_configure_matplotlib()
