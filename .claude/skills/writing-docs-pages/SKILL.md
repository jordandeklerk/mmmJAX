---
name: writing-docs-pages
description: Adds, edits, and verifies pages in the mmmJAX Sphinx docs under docs/source, including the executed MyST-NB User Guide pages, their stored fits, the numbers their prose quotes, and their figures. Use when the user asks to write, edit, restructure, rename, or check docs, a docs page, a User Guide, Getting Started, or Examples page, a toctree, or the docs build and its warnings, or when a change to a model, a stored fit, or library code could leave a page's outputs, numbers, or plots stale.
---

# Writing docs pages

The Prose section of `.claude/CLAUDE.md` and the rules below govern every page, and the sections after them are the procedure. Run commands from the repository root.

## Rules

- User Guide pages are MyST-NB pages executed during the build, and no page samples. A sampling cell is tagged `skip-execution` and its fit loads from a stored file, as Stored fits describes. After changing a `prerun/*.py` model or a stored fit, delete `docs/_build/.jupyter_cache`.
- Every number in the prose must match an executed output. Escape dollar amounts as `\$`, lead with runnable code, and use no Markdown tables.
- Print `az.summary` tables in full, and check convergence with `az.plot_rank(..., thin=True)` next to trace plots.
- A model's first page shows its generative model in full, covering the data transformations, the model equation, the media transformation, and the priors. Variants show only what changes.

## Place the page

Add the page to the toctree in its section's `index.md`, since a page in no toctree fails the build (`toc.not_included`), and keep section toctrees visible. The User Guide toctree runs in reading order, and the intro paragraph of `user_guide/index.md` names the first four pages, so update it when a page lands among them. Examples hold the in-depth tutorials with more complex models, and an Examples page adds a toctree to `examples/index.md` when it has none.

## Page skeleton

A User Guide page built on the first model opens like this, with the hidden cell after the intro and before the first `##`.

````markdown
---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Title in sentence case

An intro paragraph that names the model it uses, such as the model from [A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)

import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = ".33"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["date.converter"] = "concise"  # only with date axes
```
````

Without the front matter the cells never run. A page has one H1 and skips no heading level (`myst.header`). MyST-NB runs each page from its own folder, so `prerun` loads only from `user_guide/`, and for a page elsewhere ask the user where its setup should live. `example_data.md` and `data.md` come before the model and never load it or a fit, and the frame they simulate must come from the same `simulate_data` call as `prerun/first_model.py`. `first_model.md` shows its setup instead of hiding it, and its cells before the sampling cell must equal `prerun/first_model.py`, so change the two together. The ten-channel brand of `plotting.md` and `custom_plots.md` lives in `prerun/brand_model.py`. `plotting.md` gives its math and runs the whole file in one `hide-input` cell that must equal it, and `custom_plots.md` loads it with `%run prerun/brand_model.py` and `brand_results(model)`.

## Stored fits

- The hidden cell after a visible `skip-execution` cell runs `x = stored("x", lambda: <the visible call>, groups=[...])`. The call keeps the visible cell's seed and settings and adds at most a quiet flag, such as NumPyro's `progress_bar=False`. A fit of several statements goes in a `def fit_name()` inside the hidden cell, as in `user_guide/inference.md`.
- List in `groups` only what the page reads. A group left out raises a KeyError in the build. `continue_sampling` needs `sampling_state`, which is why `first_model.nc` keeps every group.
- `stored` loads an existing `.nc` without looking at the call, so delete a fit's file after changing its call or its model. A change to `prerun/first_model.py` or the simulated data can make every `prerun/*.nc` stale, so find them with `grep -n "stored(" docs/source/user_guide/*.md` and confirm with the user before refitting them all.
- A missing file is sampled and saved by whichever run reaches it first, the draft below or the build. Read the Docs builds from a clean checkout, so tell the user each new `.nc` must be committed with its page.

## Cells

- End each cell in one expression or a `print`, rounded to what the prose quotes, as in `round(float(x), 2)` or `.to_series().round(3)`. Print ArviZ text reports such as `az.compare`, but leave `az.loo(results)` bare.
- The setup cell keeps `arviz-darkgrid` for its colors and fonts and swaps its gray panels for white ones with left and bottom axes, which the style otherwise hides. Pages name curves by color, as in the blue curves of `priors.md`, so a palette change means rereading the prose.
- Prefer mmmJAX's plotting functions wherever one draws what the page needs, and see `user_guide/plotting.md`. A plotnine plot such as `mj.plot_fit(...)` ends its cell as the last expression, since a `ggplot` displays itself at 12 by 7 inches.
- End other plot cells with `plt.show()`, including mmmJAX's ArviZ wrappers such as `mj.plot_rank`, which already size their figures at 12 by 7. ArviZ 1.3 ignores the rc figure size. It draws multi-panel plots 24 inches wide or more and single-panel plots as 12 by 4 strips, so give every direct ArviZ plot `figure_kwargs={"figsize": (12, 7)}`, or `(12, 9)` with `col_wrap=2` for rank plots. Hand-made figures use `plt.subplots(layout="constrained")` and `legend(frameon=False)`. Keep one idea per figure.
- Model code defines its parameters in a `parameters = {...}` block and passes that name to `mj.Model`, never an inline dict in the call.
- Setup code a reader may want but need not read goes in one cell tagged `hide-input`, which folds it behind a Show code line styled in `custom.css`. The prompts are set in `conf.py`, so don't tag cells `hide-output`, whose toggle would read the same.
- `_static/css/custom.css` styles DataTree, Dataset, and data frame outputs. Any other HTML output is unstyled, so check it in the screenshot.
- A cell meant to fail needs `:tags: [raises-exception]` and a hidden `%xmode minimal` cell earlier on the page, as in `user_guide/functions.md`. Any other error stops the build. Stderr is dropped, so warnings and progress bars never show.
- Code cells are not linted. Write them in ruff style by hand, with two blank lines after a top-level `def`, and keep them plain. The code box shows about 110 characters before it scrolls sideways, so wrap lines well before ruff's 120.
- A new import needs its package in both pixi's `[feature.docs.dependencies]` (or `pypi-dependencies`) and the `doc` extra in `pyproject.toml`, which Read the Docs installs.

Use callout boxes sparingly, as `:::{admonition} Title` with `:class:` set to `note`, `warning`, or `important`. Put model logic in `$$` displays rather than long inline math. No User Guide model passes `media_history`.

## Draft numbers before building

Candidate numbers come from the stored fits in seconds.

```bash
(cd docs/source/user_guide && PYTHONDONTWRITEBYTECODE=1 ../../../.pixi/envs/docs/bin/python - <<'EOF'
exec(open("prerun/first_model.py").read())
from prerun import first_model_results, stored
results = first_model_results(model)
# the page's cells, printing what the prose will quote
EOF
)
```

## Build

The execution cache keys each page on its own cells, so also delete `docs/_build/.jupyter_cache` after changing library code a page calls. `pixi run -e docs docs-clean` wipes the whole build. Build through pixi so the kernel is the docs environment.

```bash
pixi run -e docs docs 2>&1 | grep -E "(WARNING|ERROR):|sphinx-err|build (succeeded|finished)"
```

A cold cache runs every page, close to two minutes of kernel time, so use a long timeout or run it in the background. A failing cell stops the build, and its traceback is in the `sphinx-err-*.log` file the output names. Link the Data page as `data.md`, because `[...](data)` also matches an API page and `mmmjax.Model.data` (`myst.xref_ambiguous`). Link other guide pages by name, as in `[A first model](first_model)`, and other sections by relative path, as in `../getting_started/installation`. Intersphinx covers only Python, JAX, and NumPy, so link ArviZ and xarray by URL to avoid `myst.xref_missing`.

## Verify

Read the outputs in `docs/_build/html/user_guide/<page>.html`, since the notebooks in `docs/_build/jupyter_execute` hold no outputs in cache mode. The build never deletes old HTML, so delete a renamed page's old file from `docs/_build/html`.

1. A role such as `{func}` that finds no target renders as plain code without a warning. A resolved one sits inside an `<a>` tag, so `grep -o '.\{2\}<code class="xref' <page>.html | grep -v '">'` prints only the unresolved ones.
2. Match every number in the prose to an output and recompute derived ones, such as differences and percentages, from the printed values. After a fit or the model changes, grep the other pages for the numbers it moved, since pages quote each other (recovery repeats the TV retention from the sampling page, and example_data carries the true ROIs and shares).
3. Read each figure the page's `<img>` tags point to in `docs/_build/html/_images`, and get its pixel size from `file`. About 1200 pixels wide is 12 inches at 100 dpi, and a height near 390 means a single-panel ArviZ plot is missing `figure_kwargs`. Look for grid lines, overlapping labels or legends, and squeezed panels.
4. Screenshot the page into the scratchpad in light mode (`preferredColorScheme=1`) and dark mode (`0`) and read both. A taller window reaches further down a long page, and `--user-data-dir` makes Chrome hang.

   ```bash
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1400,2400 --blink-settings=preferredColorScheme=1 --screenshot=<scratchpad>/light.png "file://$PWD/docs/_build/html/user_guide/<page>.html"
   ```

5. Read the page's prose against the Prose rules in `.claude/CLAUDE.md` and the avoid-ai-writing skill, leaving out code cells, math, and front matter.
