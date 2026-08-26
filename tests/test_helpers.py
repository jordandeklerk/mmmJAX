"""Tests for shared test helpers."""

import pytest

from .helpers import importorskip

MISSING_MODULE = "_mmmjax_missing_test_dependency"


def test_importorskip_returns_module() -> None:
    assert importorskip("pytest") is pytest


def test_importorskip_skips_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MMMJAX_REQUIRE_ALL_DEPS", raising=False)

    with pytest.raises(pytest.skip.Exception, match="could not import"):
        importorskip(MISSING_MODULE)


def test_importorskip_requires_dependency_in_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMMJAX_REQUIRE_ALL_DEPS", "1")

    with pytest.raises(ModuleNotFoundError, match=MISSING_MODULE):
        importorskip(MISSING_MODULE)
