# mmmJAX docstrings

The templates below cover a numerical primitive, a workflow or analysis function, and a class, and the last section holds the rules every docstring follows.

Sphinx renders numpydoc through napoleon with `autodoc_typehints = "none"`, so the docstring is the only place a reader sees types. Start from the template for the kind of object and replace each `<placeholder>`. Text outside the angle brackets, including section underlines, type spellings, and default forms, is the wording new code uses.

## Numerical primitive

A jit-safe array transform such as `fourier_features` or `hill_saturation`.

```python
def <name>(<input>: ArrayLike, *, <count>: int) -> jax.Array:
    r"""<Apply or Build> <what> <to or for which inputs>.

    For <symbols and their domains, as in :math:`x \geq 0`>, the <quantity>
    is

    .. math::

        <formula>.

    <Edge values, the shape of the curve, and what the caller supplies
    separately.>

    Parameters
    ----------
    <input> : array_like
        <Constraints first, as in "Finite, nonnegative exposures">. <Units and
        layout, as in ``(time, group, channel)``.>
    <count> : int
        <Meaning>. <Static note, as in "Keep this argument static when using
        ``jax.jit``".>

    Returns
    -------
    jax.Array
        <Values> with shape ``<shape>``. <Dtype floor, as in "Floating dtype
        is at least float32".> <What invalid inputs give, as in "Invalid
        numeric inputs give ``nan``".>

    Examples
    --------
    Start with <data>.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import <name>

        In [2]: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "<column>": [<v1>, <v2>, <v3>],
           ...: })

    <What the call does.>

    .. ipython::

        In [3]: <output> = <name>(frame["<column>"].to_numpy(), <count>=<n>)

        In [4]: frame.with_columns(
           ...:     pl.Series("<column>_<output>", np.asarray(<output>)),
           ...: )
    """
```

- Drop the lead-in and `.. math::` block when there is no formula, and keep the `r` prefix only when the docstring holds a backslash. The display ends in a period, or in a comma before a "where ..." paragraph, and a `cases` block puts that mark inside its last case.
- Examples use the executed `.. ipython::` directive and never `>>>` or `.. code-block:: python`. Each must run, because the docs build executes it. Each later block gets its own lead-in and continues the prompt numbers. From `In [10]` on, continuation lines use `   ....:` with four dots, because the directive silently skips a `   ...:` line there.
- Adstock, saturation, media, seasonality, HSGP, and scaling examples start from a tiny polars frame with a `"week"` column. Several channels use `frame.select("<a>", "<b>").to_numpy()` and read `<output>[:, 0]`. Other primitives use small `jnp.array` inputs or `simulate_data`.
- Distribution functions copy the summaries, Parameters, Returns, and one-cell Examples of `distributions/_normal.py`, and the scalar sum adds a `grad` cell. A `_logcdf` or `_logsf` lead-in puts the conversion to probabilities in its own sentence.

## Workflow or analysis function

A function that takes a `Model` and its draws, as `response_curves` and `media_metrics` do.

```python
def <name>(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    group: Literal["prior", "posterior"] = "posterior",
) -> xr.Dataset:
    """<Verb> <what the function returns> <by channel or scenario>.

    <One paragraph saying what is rerun or discarded, which inputs stay
    fixed, and how uncertainty carries through the draws.>

    Parameters
    ----------
    model : Model
        <What the model must hold>.
    results : xarray.DataTree
        Results containing the model's constrained draws in the selected group.
    quantity : str
        <The seven-line entry from ``media_metrics``, copied word for word.>
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. <When to choose ``"prior"``.>
        No sampling is performed. Compare groups using separate calls.

    Returns
    -------
    xarray.Dataset
        <Noun> with chain and draw labels and a ``group`` attribute
        identifying the parameter draws. <Units of the fields.>

        - **<field>** — <Capitalized description with no final period>
        - **<field>**, **<field>** — <Description the fields share>

        <Short paragraph on axes or ``nan`` behavior.>
    """
```

- It has one body paragraph, no math, and no Examples when it needs a fit. The `results`, `quantity`, and `group` entries are shared word for word by the analysis functions that take `group`.
- Field bullets use the dash form of `fit_scaling` and `check_prior`. No analysis function has it yet, so `media_metrics` is a model for the body and Parameters only. Named `name : type` Returns entries are only for a tuple, as in `hsgp_basis`.
- A host-side preparation function such as `fit_scaling` shares this body and these bullets, opens its Returns with "<Noun> with the following fields.", and adds the primitive's Examples when it runs on small inline data.

## Class

```python
class <Name>:
    """<Verb> <what the object does or stores>.

    <How the workflow uses the object, in one to three short paragraphs.>

    Parameters
    ----------
    <lower> : float, default 0.0
        <Meaning>.
    <upper> : float, optional
        <Meaning>. Defaults to <computed value>.
    """
```

- Constructor arguments go in signature order in the class docstring, since Sphinx never renders an `__init__` docstring. A custom `__init__` gets one line naming what it normalizes, copies, or validates, and why when the reason is not obvious, as in `Real.__init__`.
- The summary starts with a verb, and parameterization classes such as `Real` use a noun phrase.
- A class the library builds and returns, such as `Scaling` or `PreparedData`, lists its public fields under Attributes instead of Parameters, in declaration order with no default markers. Its body names the factory with a `:func:` role. Attribute types name the stored type (`dict of str to jax.Array`), and Parameters name what they accept (`mapping of str to array_like`). No class has both sections.
- Class Examples appear only where construction is the lesson, as in `SpendConstraint`.
- A property's summary starts with "Return", or is a noun phrase on parameterization classes (`Real.position_shape`). Most are one line. One whose value needs a type, unit, or None note adds a Returns section, as `Model.scaling` does.

## Every docstring

- Wrap text near 79 columns as the templates do, because ruff format never rewraps it.
- Other type spellings are `callable`, `bool`, `tuple of int`, `sequence of str`, `mapping of str to array_like`, and `dataframe-like or PreparedData`. `optional` marks a `None` default, and its entry says "Defaults to <value>" when the function computes one (`channels`, `budget`) or "Omit to <effect>" when leaving it out changes the behavior (`new_data`, `by`). Every other default is written `default X`, including `default ()`, and never `default=X`.
- The sections in use are Parameters, Returns, Examples, and Attributes. No docstring uses See Also or References, and the Raises section in `optimize_budget` and the Notes in `_binomial.py` are outliers.
- Body paragraphs read like a methods section, and details about one argument go in its Parameters entry. Keep Returns sections short, and never restore longer text from an earlier version of a docstring.
- Names in running text use double backticks. Sphinx roles (`:func:`, `:class:`, `:meth:`) are rare and unqualified, since the API pages set `.. currentmodule:: mmmjax`.
- Besides ", <verb>ing", no sentence ends in a comma plus a trailing clause such as ", which", ", with", ", or None while", ", empty when", or ", then". Split it into two sentences. "A, B, and C" lists are fine.
- Private helpers in the core modules get a one-line docstring stating their intent or the constraint they keep, as in `"""Broadcast a scalar or per-channel setting to one value per channel."""`. Private functions under `distributions/` mostly have none.
- Some older docstrings break these rules, so never cite them as precedent. They include the sentence bullets in `sampling.py`, `response.py`, `contribution.py`, and `optimization.py`, the colon bullets in `DataScaling.transform`, and the trailing clauses in `sampling.py`, `data.py`, `synthetic.py`, and the distribution `_logcdf` and `_logsf` lead-ins. `adstock.py` restates literal defaults, gathers its static notes on `normalize`, and names the group axis `geo`. The `import mmmjax as mm` Examples in `media.py` and `import mmmjax as mj` in `distributions/_distribution.py`, the missing helper docstrings in `adstock.py` and `parameters.py`, the missing factory role in `SyntheticData`, and the missing `Data.__init__` docstring are older too.
