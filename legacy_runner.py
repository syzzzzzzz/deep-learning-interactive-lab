"""Run legacy Matplotlib lessons safely for the Streamlit shell."""

from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def configure_matplotlib(output_dir: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

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
    plt.rcParams["axes.unicode_minus"] = False

    figure_index = {"value": 0}

    def save_open_figures(*_args, **_kwargs) -> None:
        figure_numbers = list(plt.get_fignums())
        for figure_number in figure_numbers:
            figure_index["value"] += 1
            figure = plt.figure(figure_number)
            figure.savefig(
                output_dir / f"figure_{figure_index['value']:02d}.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close("all")

    plt.show = save_open_figures


def main() -> int:
    configure_stdio()
    if len(sys.argv) != 4:
        print("Usage: python legacy_runner.py <base_dir> <script_path> <output_dir>", file=sys.stderr)
        return 2

    base_dir = Path(sys.argv[1]).resolve()
    script_path = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not script_path.is_file():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 2
    if not str(script_path).lower().startswith(str(base_dir).lower()):
        print(f"Refusing to run a script outside project: {script_path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(base_dir))
    os.chdir(output_dir)
    configure_matplotlib(output_dir)

    try:
        runpy.run_path(str(script_path), run_name="__main__")
        import matplotlib.pyplot as plt

        plt.show()
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
