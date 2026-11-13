"""The provider registry shared by the ASR and OCR packages."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from better_meeting.registry import Registry
from better_meeting.utils import BmError


def test_a_provider_is_created_once_and_again_after_it_is_replaced():
    registry: Registry[list] = Registry("thing", {})
    registry.register("a", list)
    first = registry.get("a")
    assert registry.get("a") is first
    assert registry.create("a") is not first
    registry.register("a", lambda: [1])
    assert registry.get("a") == [1]


def test_a_module_path_is_imported_on_first_use():
    registry: Registry[type] = Registry("thing", {"path": "pathlib:PurePosixPath"})
    assert registry.names() == ["path"]
    assert registry.create("path") == PurePosixPath()


def test_an_unknown_name_lists_the_known_ones():
    registry: Registry[list] = Registry("thing", {"a": "x:y", "b": "x:y"})
    with pytest.raises(BmError, match=r"unknown thing 'c'; available: a, b"):
        registry.get("c")
