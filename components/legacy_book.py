"""Render references to the migrated Markdown version of the old book."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BOOK_DIR = ROOT / "docs" / "legacy_book"
MANIFEST_PATH = LEGACY_BOOK_DIR / "manifest.json"

PART_FOLDER_TO_SHORT = {
    "part1_foundations": "part1",
    "part2_cnn": "part2",
    "part3_rnn": "part3",
    "part4_transformer": "part4",
    "part5_toolbox": "part5",
    "part6_universal_framework": "part6",
}


@dataclass(frozen=True)
class LegacyLesson:
    route: str
    title: str
    local_path: str
    outline: list[str]
    char_count: int
    line_count: int
    code_block_count: int

    @property
    def file_path(self) -> Path:
        return LEGACY_BOOK_DIR / self.local_path


def _short_route(route: str) -> str:
    cleaned = route.strip().replace("\\", "/")
    parts = cleaned.split("/")
    if len(parts) != 2:
        return cleaned
    folder, module = parts
    return f"{PART_FOLDER_TO_SHORT.get(folder, folder)}/{module}"


@lru_cache(maxsize=1)
def legacy_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"lesson_count": 0, "lessons": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def get_legacy_lesson(route: str) -> LegacyLesson | None:
    lessons = legacy_manifest().get("lessons", {})
    payload = lessons.get(_short_route(route))
    if not isinstance(payload, dict):
        return None
    return LegacyLesson(
        route=str(payload["route"]),
        title=str(payload["title"]),
        local_path=str(payload["local_path"]),
        outline=[str(item) for item in payload.get("outline", [])],
        char_count=int(payload.get("char_count", 0)),
        line_count=int(payload.get("line_count", 0)),
        code_block_count=int(payload.get("code_block_count", 0)),
    )


def read_legacy_markdown(lesson: LegacyLesson, max_chars: int | None = None) -> str:
    text = lesson.file_path.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n\n……（原稿较长，已截断；可用下载按钮查看完整 Markdown。）"
    return text


def render_legacy_book_reference(route: str) -> None:
    """Show the migrated old-book source outline for a graph node."""

    lesson = get_legacy_lesson(route)
    if lesson is None:
        return

    st = __import__("streamlit")
    st.markdown("**旧教材原稿（已从 `deep_learning_book` 移植）**")
    st.caption(
        f"{lesson.title}｜{lesson.line_count} 行｜约 {lesson.char_count} 字符｜"
        f"{lesson.code_block_count} 个代码块"
    )
    outline = lesson.outline[:8]
    if outline:
        st.markdown("".join(f"- {item}\n" for item in outline))

    text = read_legacy_markdown(lesson)
    cols = st.columns([0.34, 0.66])
    cols[0].download_button(
        "下载旧教材 Markdown",
        data=text.encode("utf-8"),
        file_name=Path(lesson.local_path).name,
        mime="text/markdown",
        width="stretch",
    )
    with cols[1].expander("预览旧教材原文", expanded=False):
        st.text_area(
            "原文预览",
            value=read_legacy_markdown(lesson, max_chars=6000),
            height=340,
            disabled=True,
            label_visibility="collapsed",
        )
