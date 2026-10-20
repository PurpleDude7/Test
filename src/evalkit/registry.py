"""A tiny name -> class registry with entry-point discovery for third-party plugins."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from evalkit.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps names to classes.

    Built-ins register themselves via :meth:`register`. Third-party packages can expose
    implementations through the ``entry_point_group`` entry-point group; those are loaded
    lazily the first time a lookup misses.
    """

    def __init__(self, kind: str, entry_point_group: str) -> None:
        self._kind = kind
        self._group = entry_point_group
        self._items: dict[str, type[T]] = {}
        self._entry_points_loaded = False

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            for name in names:
                existing = self._items.get(name)
                if existing is not None and existing is not cls:
                    raise RegistryError(f"{self._kind} {name!r} is already registered")
                self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            self._load_entry_points()
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(self.names()) or "<none>"
            raise RegistryError(f"unknown {self._kind} {name!r} (known: {known})") from None

    def names(self) -> list[str]:
        self._load_entry_points()
        return sorted(self._items)

    def _load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in entry_points(group=self._group):
            if ep.name not in self._items:
                self._items[ep.name] = ep.load()
