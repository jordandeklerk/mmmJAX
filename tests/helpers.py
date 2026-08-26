"""Shared helpers for the mmmJAX test suite."""

import importlib
import os
import warnings
from types import ModuleType

import pytest


def importorskip(modname: str, reason: str | None = None) -> ModuleType:
    """Import and return an optional dependency or skip its tests.

    Defining ``MMMJAX_REQUIRE_ALL_DEPS`` turns a missing dependency into an
    import error instead of a skip. This lets comprehensive CI environments
    ensure that every optional dependency is installed and working.

    Parameters
    ----------
    modname : str
        Name of the module to import.
    reason : str, optional
        Message shown when the module cannot be imported.

    Returns
    -------
    ModuleType
        The imported module.
    """
    __tracebackhide__ = True
    compile(modname, "", "eval")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            if "MMMJAX_REQUIRE_ALL_DEPS" in os.environ:
                raise
            if reason is None:
                reason = f"could not import {modname!r}: {exc}"
            pytest.skip(reason, allow_module_level=True)
