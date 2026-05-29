"""Course catalog, routing aliases, and learning recommendations."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PartInfo:
    key: str
    directory: str
    emoji: str
    title: str
    short_title: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.title}"


@dataclass(frozen=True)
class ModuleInfo:
    part_key: str
    part_dir: str
    title: str
    module: str
    summary: str
    level: str
    tags: tuple[str, ...]
    priority: int = 50

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.part_dir / f"{self.module}.py"

    @property
    def target(self) -> str:
        return f"{self.part_dir}/{self.module}"

    @property
    def short_target(self) -> str:
        return f"{self.part_key}/{self.module}"


def fallback_title(module_name: str) -> str:
    title = module_name
    for prefix in [f"{index:02d}_" for index in range(1, 100)]:
        if title.startswith(prefix):
            title = title[len(prefix) :]
            break
    return title.replace("_", " ").strip().title()


def build_module_catalog(
    project_root: Path,
    parts: dict[str, PartInfo],
    registered_modules: list[ModuleInfo] | tuple[ModuleInfo, ...],
) -> tuple[ModuleInfo, ...]:
    registered = {(module.part_dir, module.module): module for module in registered_modules}
    modules: list[ModuleInfo] = []

    for part_index, part in enumerate(parts.values(), 1):
        full_dir = project_root / part.directory
        if not full_dir.is_dir():
            continue

        for path in sorted(full_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue

            key = (part.directory, path.stem)
            if key in registered:
                modules.append(registered[key])
                continue

            modules.append(
                ModuleInfo(
                    part_key=part.key,
                    part_dir=part.directory,
                    title=fallback_title(path.stem),
                    module=path.stem,
                    summary=f"自动发现页面：{part.directory}/{path.name}",
                    level="模块",
                    tags=(part.short_title, "自动注册"),
                    priority=part_index * 100 + len(modules),
                )
            )

    part_order = {key: index for index, key in enumerate(parts)}
    return tuple(sorted(modules, key=lambda item: (part_order[item.part_key], item.priority, item.module)))


def build_route_map(modules: tuple[ModuleInfo, ...] | list[ModuleInfo]) -> dict[str, ModuleInfo]:
    routes: dict[str, ModuleInfo] = {}
    for module in modules:
        routes[module.target] = module
        routes[module.short_target] = module
    return routes


def list_part_modules(project_root: Path, parts: dict[str, PartInfo], part_name: str) -> list[str]:
    part = parts.get(part_name)
    part_dir = part.directory if part else part_name
    full_dir = project_root / part_dir
    if not full_dir.is_dir():
        print(f"目录不存在: {full_dir}")
        return []
    return sorted(path.stem for path in full_dir.iterdir() if path.suffix == ".py" and path.name != "__init__.py")


def search_modules(query: str, modules: list[ModuleInfo]) -> list[ModuleInfo]:
    terms = [term.lower() for term in query.strip().split() if term.strip()]
    if not terms:
        return []

    scored: list[tuple[int, ModuleInfo]] = []
    for module in modules:
        haystack = " ".join(
            [
                module.title,
                module.summary,
                module.level,
                module.target,
                module.short_target,
                " ".join(module.tags),
            ]
        ).lower()
        if not all(term in haystack for term in terms):
            continue
        score = 0
        for term in terms:
            if term in module.title.lower():
                score += 8
            if term in " ".join(module.tags).lower():
                score += 5
            if term in module.summary.lower():
                score += 3
            if term in module.target.lower():
                score += 2
        scored.append((score * 100 - module.priority, module))

    scored.sort(reverse=True, key=lambda item: item[0])
    return [module for _, module in scored]


def matching_modules(level: str, interest: str, catalog: list[ModuleInfo]) -> list[ModuleInfo]:
    if level == "刚入门":
        keys = {"part1", "part2"}
    elif level == "已有基础":
        keys = {"part2", "part3", "part4", "part5"}
    else:
        keys = {"part4", "part5", "part6"}

    interest_map = {
        "看懂深度学习整体": {"基础", "神经网络", "实验", "Transformer"},
        "计算机视觉": {"视觉", "CNN", "表征", "解释性"},
        "自然语言与大模型": {"NLP", "Transformer", "LLM", "AGI", "注意力"},
        "工程落地": {"训练", "数据", "部署", "工具", "框架", "架构", "项目"},
        "前沿研究": {"LLM", "AGI", "安全", "生成模型", "GNN", "表征", "性能"},
    }
    wanted = interest_map[interest]
    scored: list[tuple[int, ModuleInfo]] = []
    for module in catalog:
        score = 0
        if module.part_key in keys:
            score += 3
        score += len(set(module.tags) & wanted) * 4
        if module.level in {"核心", "前沿"} and interest in {"自然语言与大模型", "前沿研究"}:
            score += 2
        if score > 0 and module.path.exists():
            scored.append((score * 100 - module.priority, module))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [module for _, module in scored[:6]]


def daily_recommendation(
    catalog: list[ModuleInfo],
    knowledge_points: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...],
) -> tuple[str, str, ModuleInfo]:
    seed_text = f"{date.today().isoformat()}|deep-learning-book"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    title, text, target = rng.choice(tuple(knowledge_points))
    routes = build_route_map(catalog)
    module = routes.get(target) or rng.choice(catalog)
    return title, text, module
