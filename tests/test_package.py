"""Initialization tests for the mmmJAX package."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import mmmjax


def test_package_imports():
    assert mmmjax is not None


def test_version_is_consistent():
    assert isinstance(mmmjax.__version__, str)
    assert mmmjax.__version__
    assert mmmjax.__version__ == version("mmmjax")

    pixi_config = tomllib.loads(Path("pixi.toml").read_text())
    assert mmmjax.__version__ == pixi_config["workspace"]["version"]
