# mmmJAX

mmmJAX is a JAX library for Bayesian marketing mix models written as Stan-style blocks. The package is in `mmmjax/`, tests are in `tests/`, and the Sphinx docs are in `docs/source/`.

## Commands

- `pixi run lint` runs the pre-commit hooks, and `pixi run typecheck` runs mypy and ty on `mmmjax`.
- When running tests, run only the files that cover a change, found by grepping `tests/` for the changed names, never the whole suite. A public export change also needs `tests/distributions/test_public_api.py` and `tests/test_package.py`. A numeric change also needs a run with `JAX_ENABLE_X64=true`, since CI tests float64.
- `pixi run -e docs docs` builds the docs with warnings as errors, and `pixi run -e docs docs-open` builds and serves them.
- Work on a branch. Pre-commit refuses commits to `main`.

## Code

- Modules run in this order: docstring, imports, `__all__`, the types signatures need, the public API in `__all__` order, then private helpers in call order.
- No module-level constants such as sentinels, name strings, or TypeVars. Use a StrEnum, a literal where it is used, or PEP 695 generics.
- Compute and name each value on its own line and return the names, not inline expressions.
- The user states every prior. Never add helpers that choose distributions, scales, or parameterizations.
- Docstrings follow numpydoc.

## Prose

Prose means docs pages, the README, docstrings, and code comments.

- IMPORTANT: apply the [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing) skill to every piece of prose you write or edit. Treat its detector's findings as signals, since "features" as a noun and low vocabulary diversity on long technical pages are known false positives.
- Write brief narrative paragraphs. No bulleted or numbered lists, no colons that introduce a clause, no em dashes, and no summaries ending in ", <verb>ing". Returns sections may use `- **name** — description` bullets.
- Comments explain why in one or two lines.
