"""Add an example with its figure to the API page of each plot function.

Each file in ``api/examples`` holds one plot function's example. The build runs it against the stored
fit of the ten-channel brand from the Plotting guide and adds its code and figure to the function's
Examples section, so the figures on the API pages match the guide and are redrawn on every build.
"""

import ast
import functools
import runpy
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import plotnine as pn


def setup(app):
    """Register the handler that adds examples to plot function docstrings."""
    # Run before napoleon so it turns the added section into an example box like any other.
    app.connect("autodoc-process-docstring", _add_example, priority=400)
    # Drawing goes through pyplot's shared state.
    metadata = {"parallel_read_safe": False, "parallel_write_safe": True}
    return metadata


def _add_example(app, what, name, obj, options, lines):
    """Append a plot function's example and the figure its code draws to its docstring."""
    function = name.rpartition(".")[2]
    source = Path(app.srcdir, "api", "examples", f"{function}.py")
    # autosummary reads each docstring for its tables too, so only the function's own page draws.
    if what != "function" or app.env.docname != f"api/generated/{name}" or not source.exists():
        return
    code = source.read_text().rstrip()
    _draw(code, Path(app.srcdir, "api", "generated", "examples", f"{function}.png"), app.srcdir)
    indented = [f"   {line}" if line else "" for line in code.splitlines()]
    lines.extend(
        [
            "",
            "Examples",
            "--------",
            "The code continues the ten-channel brand from :doc:`Plotting </user_guide/plotting>`.",
            "",
            ".. code-block:: python",
            "",
            *indented,
            "",
            f".. image:: /api/generated/examples/{function}.png",
            f"   :alt: The figure {function} draws for the ten-channel brand",
        ]
    )


def _draw(code, path, srcdir):
    """Run an example and save the plot its last expression returns."""
    namespace = dict(_brand(srcdir))
    *body, last = ast.parse(code).body
    exec(compile(ast.Module(body=body, type_ignores=[]), path.name, "exec"), namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The same white panels with left and bottom axes as the guide pages.
    panels = {
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": ".33",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    }
    with plt.style.context(az.style.get("arviz-darkgrid", backend="matplotlib")), plt.rc_context(panels):
        plot = eval(compile(ast.Expression(last.value), path.name, "eval"), namespace)
        if isinstance(plot, pn.ggplot):
            plot.save(path, dpi=200, verbose=False)
        else:
            plot.savefig(path, dpi=200)
    plt.close("all")


@functools.cache
def _brand(srcdir):
    """Build the brand model and load its stored fit once per build."""
    guide = Path(srcdir, "user_guide")
    sys.path.insert(0, str(guide))
    from prerun import brand_results

    brand = runpy.run_path(str(guide / "prerun" / "brand_model.py"))
    results = brand_results(brand["model"])
    namespace = {"mj": brand["mj"], "model": brand["model"], "priors": brand["priors"], "results": results}
    return namespace
