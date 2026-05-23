"""Unified module metadata protocol for learning pages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class LearningModule(Protocol):
    """Minimum protocol implemented by a renderable learning module."""

    title: str
    summary: str
    tags: list[str] | tuple[str, ...]

    def render(self) -> None:
        """Render the module in Streamlit."""


@dataclass(frozen=True)
class ModuleMetadata:
    title: str
    summary: str
    tags: tuple[str, ...]


F = TypeVar("F", bound=Callable[..., Any])


def metadata_from_module(module: ModuleType | dict[str, Any]) -> ModuleMetadata:
    """Read MODULE_* metadata from a module object or namespace."""

    if isinstance(module, dict):
        getter = module.get
    else:
        getter = lambda name, default=None: getattr(module, name, default)

    title = getter("MODULE_TITLE") or getter("title")
    summary = getter("MODULE_SUMMARY") or getter("summary")
    tags = getter("MODULE_TAGS") or getter("tags")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("module metadata requires a non-empty title")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("module metadata requires a non-empty summary")
    if not _valid_tags(tags):
        raise ValueError("module metadata requires tags as a non-empty string list/tuple")

    return ModuleMetadata(title=title.strip(), summary=summary.strip(), tags=tuple(str(tag).strip() for tag in tags))


def _valid_tags(tags: object) -> bool:
    if not isinstance(tags, (list, tuple)):
        return False
    return bool(tags) and all(isinstance(tag, str) and tag.strip() for tag in tags)


def validate_module_protocol(module: ModuleType | dict[str, Any]) -> ModuleMetadata:
    """Validate minimum metadata plus a callable render entry."""

    metadata = metadata_from_module(module)
    render = module.get("render") if isinstance(module, dict) else getattr(module, "render", None)
    if not callable(render):
        raise ValueError("module protocol requires a callable render() entry")
    return metadata


def register_module(*, title: str, summary: str, tags: Iterable[str]) -> Callable[[F], F]:
    """Attach MODULE_* metadata to a render function."""

    clean_tags = tuple(str(tag).strip() for tag in tags if str(tag).strip())
    if not title.strip() or not summary.strip() or not clean_tags:
        raise ValueError("register_module requires title, summary, and at least one tag")

    def decorator(func: F) -> F:
        setattr(func, "MODULE_TITLE", title.strip())
        setattr(func, "MODULE_SUMMARY", summary.strip())
        setattr(func, "MODULE_TAGS", list(clean_tags))
        return func

    return decorator
