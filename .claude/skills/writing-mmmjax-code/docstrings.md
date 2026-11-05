# mmmJAX docstrings

## Contents

- Numerical primitive, the template for a jit-safe array transform
- Workflow or analysis function, the template for a function that takes a `Model` and its draws
- Class, the template for a public class
- Every docstring, the wrapping, type spellings, default forms, and section rules every docstring follows

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

    See Also
    --------
    <sibling> : <Short imperative phrase>.

    Notes
    -----
    <Edge cases, numerical behavior, or a derivation, when there are any.>

    Examples
    --------
    <What the call does and what the output shows, in a sentence or two.>

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import <name>

        In [2]: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "<column>": [<v1>, <v2>, <v3>],
           ...: })

        In [3]: <output> = <name>(frame["<column>"].to_numpy(), <count>=<n>)

        In [4]: frame.with_columns(
           ...:     pl.Series("<column>_<output>", np.asarray(<output>)),
           ...: )
    """
```

- Drop the lead-in and `.. math::` block when there is no formula, and keep the `r` prefix only when the docstring holds a backslash. The display ends in a period, or in a comma before a "where ..." paragraph, and a `cases` block puts that mark inside its last case.
- Examples use the executed `.. ipython::` directive and never `>>>` or `.. code-block:: python`. Each must run, because the docs build executes it. An example is one block under a lead-in of one or two sentences, with no commentary between its cells. A second block is only for a separate idea, as the `vmap` call in `student_t` is, and it gets its own lead-in and continues the prompt numbers. From `In [10]` on, continuation lines use `   ....:` with four dots, because the directive silently skips a `   ...:` line there.
- Adstock, saturation, media, seasonality, and scaling examples start from a tiny polars frame with a `"week"` column so the output sits beside its input. Several channels use `frame.select("<a>", "<b>").to_numpy()` and read `<output>[:, 0]`. Other primitives, HSGP included, use small lists, `jnp.array` inputs, or `simulate_data`.
- Distribution functions copy the summaries, Parameters, and Returns of `distributions/_normal.py`. Only a family's scalar sum and `_rng` carry Examples, as there. The pointwise, CDF, and survival functions and the `_logit` and `_log` parameterizations would repeat those examples under another name, so they have none.

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

    <A brief paragraph saying what is rerun or discarded and which inputs
    stay fixed.>

    <A brief paragraph saying that ``quantity`` is recomputed for each
    scenario and draw in original outcome units, how uncertainty carries
    through the draws, and which axes the output keeps.>

    Parameters
    ----------
    model : Model
        <What the model must hold>.
    results : xarray.DataTree
        Results containing the model's constrained draws in the selected group.
    quantity : str
        Key returned by ``transformed_parameters`` holding the expected outcome
        on the scale the likelihood uses, such as ``"mu"``.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. <When to choose ``"prior"``.>

    Returns
    -------
    xarray.Dataset
        <Noun> with chain and draw labels and a ``group`` attribute
        identifying the parameter draws. <Units of the fields.>

        - **<field>** — <Capitalized description with no final period>
        - **<field>**, **<field>** — <Description the fields share>

    See Also
    --------
    <name> : <Short imperative phrase>.

    Notes
    -----
    <A brief paragraph on the errors raised and on reading any
    normalization inside the blocks from ``reference``, ending with a link
    to :doc:`/user_guide/scenarios`.>
    """
```

- Its body has no math and no Examples when it needs a fit. A plot function's example goes in `docs/source/api/examples/<name>.py` instead, which the docs build runs against the brand fit and adds to the end of its docstring, after any See Also and Notes, with the figure.
- A plotting function takes an analysis output or results, returns `plotnine.ggplot` or `arviz_plots.PlotCollection`, and shares the `ci_prob` entry of `plot_media_metrics` word for word. Its See Also names the analysis function it draws from, and how the figure sizes itself or narrows what it shows goes in Notes. An ArviZ wrapper documents `**kwargs` as further keywords for the wrapped `arviz_plots` function. A plot that could outgrow its figure takes `channels`. A plot of one curve per channel gives each channel a panel and takes `combine` to draw up to five in one, as `plot_response_curves` and `plot_adstock` do. A bar chart shows every channel by default and sizes itself through `_bar_layout`, which widens it so notebooks scroll it sideways and tilts and shortens its labels, and any other plot shows the largest channels up to a readable limit and names what it left out in a caption or title. A plot that gives groups panels takes `coords` and `n_groups=3` after `channels` and names the groups it leaves out in its caption, and a plot of parameter draws takes the output of `sample_prior` as `prior` and draws it beside the posterior. The `results`, `quantity`, and `group` entries are shared word for word by the analysis functions that take `group`.
- Field bullets use the dash form, as `media_metrics` and `prepare_hsgp` show. Named `name : type` Returns entries are only for a tuple, as in `hsgp_basis`.
- A host-side preparation function such as `prepare_hsgp` shares this body, describes its result in one line because that class lists its fields under Attributes, and adds the primitive's Examples when it runs on small inline data.

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

    See Also
    --------
    <name> : <Short imperative phrase>.
    """
```

- Constructor arguments go in signature order in the class docstring, since Sphinx never renders an `__init__` docstring. A custom `__init__` gets one line naming what it normalizes, copies, or validates, and why when the reason is not obvious, as in `Real.__init__`.
- The summary starts with a verb, and parameterization classes such as `Real` use a noun phrase.
- A class the library builds and returns, such as `Scaling` or `PreparedData`, lists its public fields under Attributes instead of Parameters, in declaration order with no default markers. Its body names the factory with a `:func:` role. Attribute types name the stored type (`dict of str to jax.Array`), and Parameters name what they accept (`mapping of str to array_like`). No class has both sections.
- Class Examples appear only where construction is the lesson, as in `Model`, `Prior`, and `SpendConstraint`.
- A property's summary starts with "Return", or is a noun phrase on parameterization classes (`Real.position_shape`). Most are one line. One whose value needs a type, unit, or None note adds a Returns section, as `Model.scaling` does.

## Every docstring

- Wrap text near 79 columns as the templates do, because ruff format never rewraps it.
- Other type spellings are `callable`, `bool`, `tuple of int`, `sequence of str`, `mapping of str to array_like`, and `dataframe-like or PreparedData`. `optional` marks a `None` default, and its entry says "Defaults to <value>" when the function computes one (`channels`, `budget`) or "Omit to <effect>" when leaving it out changes the behavior (`new_data`, `by`). Every other default is written `default X`, including `default ()`, and never `default=X`.
- The sections in use are Parameters or Attributes, Returns, See Also, Notes, and Examples, in that order. No docstring uses Raises or References, so errors and warnings go in Notes, as in `optimize_budget`, and a paper is cited in the text, as in `prepare_hsgp`.
- See Also names up to four public objects a reader is likely to want next, such as sibling transforms, the analysis behind a plot, or the next step of the workflow. Each entry is `name : <Short imperative phrase>.` on one line, and a name reads the same in every docstring that lists it. Distribution families and parameterization classes have no See Also, since their API pages already group them.
- Notes holds what a reader can skip until something surprises them, such as errors and warnings, edge cases, numerical behavior, derivations, plot layout, and the rule about reading from `reference`. It follows the prose rules of the body and may hold math. A docstring with nothing of that kind has no Notes, and a one-sentence error can stay in a short body, as in `select_channels`.
- Examples show what the summary, Parameters, and Returns cannot, such as realistic inputs and their layout, the visible effect of a transform, or how the object works with the rest of the API. Leave them out when they would repeat a sibling's or a factory's example, as the methods of `HSGPApproximation` and `Scaling` would, or when the call needs a fit.
- A Parameters entry says only what the argument is and what it requires, such as its meaning, units, shape, allowed values, and whether it must stay static. Leave out what other functions later do with the result, such as which ArviZ function reads a group, and describe outputs under Returns.
- A docstring describes what the code does on its own terms and never cites or compares with another package, as in "like Meridian's chart" or "as PyMC-Marketing does". Naming a library the code uses, accepts, or returns is part of that description, such as a polars or pandas frame, `jax.vmap` for more draws, ArviZ's `stats.ci_prob` setting, or a returned `xarray.Dataset`.
- Body text reads like the prose of the docs pages, in brief narrative paragraphs that flow from one to the next, even where its tone is closer to a methods section. Each paragraph holds one idea in two to four sentences, such as what the code computes, which inputs it holds fixed, how uncertainty carries through the draws, or what a plot shows. A body that covers several ideas gives each its own paragraph instead of running them together in one block, as `contributions`, `contribution_coefficient`, and `plot_media_metrics` do. The plotting docstrings set the spacing, tone, and length every public docstring follows, and the body describes what the reader gets instead of issuing clipped commands.
- A body holds what a reader needs at the call, usually one to three paragraphs and rarely more than about 120 words outside its math. Anything a formula, a Parameters entry, or the example already shows goes. The longer explanation behind a rule belongs on a guide page, linked with an absolute `:doc:` path, as in "See :doc:`/user_guide/data` for the names prepared data supplies."
- Details about one argument go in its Parameters entry, in one or two sentences, and behavior of the function as a whole goes in the body. Keep Returns sections to the type line, a one-line description, and any field bullets, with axis and `nan` behavior in the body or Notes, and never restore longer text from an earlier version of a docstring.
- Names in running text use double backticks. Sphinx roles (`:func:`, `:class:`, `:meth:`) are rare and unqualified, since the API pages set `.. currentmodule:: mmmjax`, and `:doc:` links stay few.
- The summary line never contains a comma, not even in a list, so write it as one clause, as in "Prepare a dataframe for modeling while keeping its observation labels." This covers module docstrings and the one-line docstrings of private helpers too.
- Elsewhere a comma is fine wherever the sentence needs one, so never drop or reword one away for its own sake. Only a clause tacked onto the end of a sentence with ", <verb>ing", ", which", ", with", or ", then" gets split into its own sentence.
- Private helpers in the core modules get a one-line docstring stating their intent or the constraint they keep, as in `"""Broadcast a scalar or per-channel setting to one value per channel."""`. Private functions under `distributions/` mostly have none.
