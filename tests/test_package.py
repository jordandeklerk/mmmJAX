"""Initialization tests for the mmmJAX package."""

import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import mmmjax


def test_package_imports():
    assert mmmjax is not None


def test_package_imports_on_matplotlib_without_rc_key_type():
    # matplotlib added RcKeyType in 3.11, so removing it stands in for the older releases the package allows
    script = "import matplotlib.typing\nvars(matplotlib.typing).pop('RcKeyType', None)\nimport mmmjax"

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stdout + result.stderr


def test_version_is_consistent():
    assert isinstance(mmmjax.__version__, str)
    assert mmmjax.__version__
    assert mmmjax.__version__ == version("mmmjax")

    pixi_config = tomllib.loads(Path("pixi.toml").read_text())
    assert mmmjax.__version__ == pixi_config["workspace"]["version"]
