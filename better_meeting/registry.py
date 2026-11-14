"""Named providers of one kind, behind one interface.

The module of a built-in provider is imported when the provider is first used,
so optional dependencies stay optional.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from .utils import BmError


class Registry[T]:
    """Providers of one kind (``"ASR backend"``, ``"OCR backend"``, ...) by name.

    A factory is a zero-argument callable that returns the provider, or a
    ``"package.module:ClassName"`` string that is imported on first use.
    """

    def __init__(self, kind: str, builtin: dict[str, str]) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[[], T] | str] = dict(builtin)
        self._instances: dict[str, T] = {}

    def register(self, name: str, factory: Callable[[], T] | str) -> None:
        """Add or replace a provider."""
        self._factories[name] = factory
        self._instances.pop(name, None)

    def names(self) -> list[str]:
        return list(self._factories)

    def check(self, name: str) -> None:
        if name not in self._factories:
            known = ", ".join(self._factories)
            raise BmError(f"unknown {self._kind} {name!r}; available: {known}")

    def create(self, name: str) -> T:
        """A new provider instance."""
        self.check(name)
        factory = self._factories[name]
        if isinstance(factory, str):
            module, _, attr = factory.partition(":")
            factory = getattr(importlib.import_module(module), attr)
        return factory()

    def get(self, name: str) -> T:
        """The provider instance for ``name``, created once."""
        if name not in self._instances:
            self._instances[name] = self.create(name)
        return self._instances[name]
