from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class QualityCheckFailure(Exception):
    """Raised when a domain quality check fails."""


@dataclass(frozen=True)
class QualityCheckContext:
    root: Path

    def read_text(self, rel_path: Path) -> str:
        return (self.root / rel_path).read_text(encoding="utf-8", errors="replace")

    def exists(self, rel_path: Path) -> bool:
        return (self.root / rel_path).exists()

