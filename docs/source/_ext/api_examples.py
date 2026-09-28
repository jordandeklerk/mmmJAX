"""Add an example with its figure to the API page of each plot function.

Each file in ``api/examples`` holds one plot function's example. The build runs it against the stored
fit of the ten-channel brand from A first model and adds its code and figure to the function's
Examples section, so the figures on the API pages match the guide and are redrawn on every build.
An example that optimizes a budget, computes response curves, or samples the priors gets the result
the guide pages store in place of running that line, so the build never samples or optimizes.
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
    _draw(function, code, Path(app.srcdir, "api", "generated", "examples", f"{function}.png"), app.srcdir)
    indented = [f"   {line}" if line else "" for line in code.splitlines()]
    lines.extend(
        [
            "",
            "Examples",
            "--------",
            "The code continues the ten-channel brand from :doc:`A first model </user_guide/first_model>`.",
            "",
            ".. code-block:: python",
            "",
            *indented,
            "",
            f".. image:: /api/generated/examples/{function}.png",
            f"   :alt: The figure {function} draws for the ten-channel brand",
        ]
    )


def _draw(function, code, path, srcdir):
    """Run an example and save the plot its last expression returns."""
    namespace = dict(_brand(srcdir))
    provided = _stored(function, namespace)
    namespace |= provided
    *body, last = ast.parse(code).body
    kept = [statement for statement in body if not _assigns(statement, provided)]
    exec(compile(ast.Module(body=kept, type_ignores=[]), path.name, "exec"), namespace)
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
    sys.path.insert(0, str(srcdir))
    from prerun import first_model_results

    brand = runpy.run_path(str(Path(srcdir, "prerun", "first_model.py")))
    results = first_model_results(brand["model"])
    namespace = {"mj": brand["mj"], "model": brand["model"], "priors": brand["priors"], "results": results}
    return namespace


def _stored(function, namespace):
    """Return the stored results that stand in for an example's modeling calls, by the names it assigns."""
    from prerun import first_model_curves, first_model_limited_plan, first_model_prior_results

    model, results, priors = namespace["model"], namespace["results"], namespace["priors"]
    # Each stored result comes from the same call its example shows.
    match function:
        case "plot_budget_response" | "plot_budget_spend":
            provided = {"plan": first_model_limited_plan(model, results)}
        case "plot_prior_posterior":
            provided = {"prior_results": first_model_prior_results(model, priors)}
        case "plot_response_curves":
            provided = {"curves": first_model_curves(model, results)}
        case _:
            provided = {}
    return provided


def _assigns(statement, names):
    """Return whether a statement assigns only names among the given ones."""
    targets = statement.targets if isinstance(statement, ast.Assign) else []
    assigned = [target.id for target in targets if isinstance(target, ast.Name)]
    covered = bool(assigned) and all(name in names for name in assigned)
    return covered
