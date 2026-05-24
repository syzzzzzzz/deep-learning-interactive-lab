"""Import the old Markdown teaching book into this Streamlit project.

The source project is expected to live next to this repository:
    ../deep_learning_book

The importer copies Markdown files into docs/legacy_book and writes a stable
manifest used by the knowledge graph. It deliberately avoids absolute paths so
the GitHub project remains self-contained after the import.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "deep_learning_book"
DESTINATION = ROOT / "docs" / "legacy_book"

FOLDER_TO_SHORT_PART = {
    "part1_foundations": "part1",
    "part2_cnn": "part2",
    "part3_rnn": "part3",
    "part4_transformer": "part4",
    "part5_toolbox": "part5",
    "part6_universal_framework": "part6",
}


def route_for(path: Path, source_root: Path) -> str | None:
    rel = path.relative_to(source_root)
    if len(rel.parts) != 2:
        return None
    folder, filename = rel.parts
    short_part = FOLDER_TO_SHORT_PART.get(folder)
    if short_part is None or path.suffix.lower() != ".md":
        return None
    return f"{short_part}/{path.stem}"


def title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def outline_from(text: str, limit: int = 14) -> list[str]:
    outline: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{2,3})\s+(.+)$", line.strip())
        if not match:
            continue
        title = match.group(2).strip()
        if title:
            outline.append(title)
        if len(outline) >= limit:
            break
    return outline


def lesson_record(path: Path, source_root: Path, destination_root: Path) -> tuple[str, dict[str, Any]] | None:
    route = route_for(path, source_root)
    if route is None:
        return None
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(source_root)
    dest_rel = rel.as_posix()
    return route, {
        "route": route,
        "source_path": dest_rel,
        "local_path": dest_rel,
        "title": title_from(text, path.stem),
        "outline": outline_from(text),
        "char_count": len(text),
        "line_count": text.count("\n") + 1,
        "code_block_count": text.count("```") // 2,
        "source_folder": rel.parts[0],
        "source_stem": path.stem,
    }


def import_legacy_book(source_root: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"旧教材目录不存在：{source_root}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    lessons: dict[str, dict[str, Any]] = {}

    for path in sorted(source_root.rglob("*.md")):
        record = lesson_record(path, source_root, DESTINATION)
        if record is None:
            continue
        route, payload = record
        target = DESTINATION / payload["local_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        lessons[route] = payload

    manifest = {
        "source_project": "deep_learning_book",
        "format_version": 1,
        "lesson_count": len(lessons),
        "lessons": dict(sorted(lessons.items())),
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    manifest = import_legacy_book()
    print(f"已导入旧教材 Markdown：{manifest['lesson_count']} 个章节 -> {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
