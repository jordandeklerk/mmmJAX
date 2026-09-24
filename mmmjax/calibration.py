"""Coefficients implied by priors on returns and contributions."""

from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

__all__ = ["contribution_coefficient", "roi_coefficient"]

type _Effects = Literal["lognormal", "normal"]


def contribution_coefficient(
    contribution: ArrayLike,
    response: ArrayLike,
    *,
    outcome_scale: ArrayLike = 1.0,
    deviations: ArrayLike | None = None,
    effects: _Effects = "lognormal",
) -> jax.Array:
    r"""Return the coefficients that give each channel a chosen total contribution.

    Stan-style models place priors on quantities with business meaning, such
    as the revenue a channel added over the training window, and derive the
    coefficient that produces it. For transformed exposures :math:`h_{tgm}`
    over periods :math:`t`, optional groups :math:`g`, and channels :math:`m`,
    with outcome scale :math:`\sigma_g` and target contribution :math:`C_m`,
    the weighted exposure totals are

    .. math::

        A_{gm} = \sigma_g \sum_t h_{tgm}.

    Without deviations one coefficient per channel satisfies
    :math:`\beta_m = C_m / \sum_g A_{gm}`. With deviations
    :math:`\delta_{gm}` the coefficients vary by group around a shared center
    :math:`\beta_m` that the function solves for. Log-normal effects use
    :math:`\beta_{gm} = \exp(\beta_m + \delta_{gm})` with
    :math:`\beta_m = \log C_m - \log \sum_g A_{gm} e^{\delta_{gm}}`, and normal
    effects use :math:`\beta_{gm} = \beta_m + \delta_{gm}` with
    :math:`\beta_m = (C_m - \sum_g A_{gm} \delta_{gm}) / \sum_g A_{gm}`. In
    every case :math:`\sum_g A_{gm} \beta_{gm} = C_m`.

    Call this inside ``transformed_parameters`` with the training exposures
    and spending read from ``reference``, so scenario analyses that change the
    current inputs leave the coefficients fixed. The coefficients are derived,
    so they receive no prior and no change of variables adjustment.

    Parameters
    ----------
    contribution : array_like
        Target total contribution per channel in outcome units, as a scalar or
        one value per channel. Log-normal effects need positive values.
    response : array_like
        Transformed exposures with time first, an optional group axis, and
        channels last, on the scale the model multiplies by the coefficients.
        Cover the periods the contribution refers to, and subtract carryover
        from earlier history when the contribution should exclude it. For a
        marginal identity pass the difference between two exposure scenarios
        instead.
    outcome_scale : array_like, default 1.0
        Factor that restores the model's outcome scale to outcome units, as a
        scalar or one value per group. Pass ``outcome_scaling.scale`` when the
        outcome is scaled.
    deviations : array_like, optional
        Centered random effects with shape ``(groups, channels)``, already
        multiplied by their dispersion, such as ``eta * dev`` for a
        noncentered hierarchy. Requires a group axis in ``response``.
    effects : {"lognormal", "normal"}, default "lognormal"
        How deviations enter. Log-normal effects keep every coefficient
        positive and normal effects leave the sign free. Ignored without
        deviations.

    Returns
    -------
    jax.Array
        Coefficients with one value per channel, or one per group and channel
        when deviations are supplied. A channel whose weighted exposure total is
        zero has no finite coefficient.

    Examples
    --------
    Two channels are observed over three weeks in two regions, and each
    region's coefficient is a log-normal deviation from a shared center.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import contribution_coefficient
           ...: response = jnp.array([[[0.5, 1.0], [0.2, 0.8]],
           ...:                       [[0.6, 0.9], [0.3, 0.7]],
           ...:                       [[0.4, 1.1], [0.1, 0.9]]])
           ...: deviations = 0.2 * jnp.array([[1.0, -0.5], [-1.0, 0.5]])
           ...: coefficient = contribution_coefficient(
           ...:     jnp.array([300.0, 500.0]), response,
           ...:     outcome_scale=jnp.array([120.0, 80.0]), deviations=deviations,
           ...: )
           ...: coefficient

    The weighted contributions recover the targets.

    .. ipython::

        In [2]: jnp.einsum("tgm,g,gm->m", response, jnp.array([120.0, 80.0]), coefficient)
    """
    values = _response_array(response)
    n_channels = values.shape[-1]
    target = _channel_vector(contribution, n_channels, "contribution")
    grouped = values.ndim == 3
    factor = _scale_factor(outcome_scale, values)
    if effects not in ("lognormal", "normal"):
        raise ValueError("effects must be 'lognormal' or 'normal'")

    exposure = jnp.sum(values, axis=0)
    weighted = exposure * (jnp.reshape(factor, (-1, 1)) if grouped else factor)
    if deviations is None:
        total = jnp.sum(weighted, axis=0) if grouped else weighted
        shared = target / total
        return shared

    offsets = _deviation_matrix(deviations, values)
    if effects == "lognormal":
        log_total = logsumexp(offsets + jnp.log(weighted), axis=0)
        center = jnp.log(target) - log_total
        coefficient = jnp.exp(center + offsets)
        return coefficient
    total = jnp.sum(weighted, axis=0)
    center = (target - jnp.sum(weighted * offsets, axis=0)) / total
    coefficient = center + offsets
    return coefficient


def roi_coefficient(
    roi: ArrayLike,
    response: ArrayLike,
    spend: ArrayLike,
    *,
    outcome_scale: ArrayLike = 1.0,
    deviations: ArrayLike | None = None,
    effects: _Effects = "lognormal",
) -> jax.Array:
    r"""Return the coefficients that give each channel a chosen return on spending.

    Return on investment :math:`r_m` is the total contribution of a channel
    divided by its total spending :math:`S_m`, so the target contribution is
    :math:`C_m = r_m S_m` and the coefficients follow from
    :func:`contribution_coefficient`. A marginal return uses the same
    identity with the response difference between reference and increased
    exposures and the spending increment as ``spend``.

    Parameters
    ----------
    roi : array_like
        Return per unit of spending over the training window, as a scalar or
        one value per channel. Log-normal effects need positive values.
    response : array_like
        Transformed exposures with time first, an optional group axis, and
        channels last, on the scale the model multiplies by the coefficients,
        over the same periods as ``spend``. When the exposures include earlier
        history, pass the response to exposure inside the window, the full
        response minus the response with window exposure removed, so carryover
        from earlier weeks is not credited to the channel.
    spend : array_like
        Spending in original units, either totaled per channel or with time
        and optional group axes that are summed here.
    outcome_scale : array_like, default 1.0
        Factor that restores the model's outcome scale to outcome units, as a
        scalar or one value per group.
    deviations : array_like, optional
        Centered random effects with shape ``(groups, channels)``, already
        multiplied by their dispersion. Requires a group axis in ``response``.
    effects : {"lognormal", "normal"}, default "lognormal"
        How deviations enter, as for :func:`contribution_coefficient`.

    Returns
    -------
    jax.Array
        Coefficients with one value per channel, or one per group and channel
        when deviations are supplied.

    Examples
    --------
    A national model has two channels with three weeks of transformed
    exposures and spending, and its outcome is standardized with scale 40.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import roi_coefficient
           ...: response = jnp.array([[0.5, 1.0], [0.6, 0.9], [0.4, 1.1]])
           ...: spend = jnp.array([[100.0, 50.0], [120.0, 40.0], [80.0, 60.0]])
           ...: coefficient = roi_coefficient(jnp.array([2.0, 0.5]), response, spend, outcome_scale=40.0)
           ...: coefficient

    Each channel's contribution over the window equals its ROI times its
    total spending.

    .. ipython::

        In [2]: 40.0 * response.sum(axis=0) * coefficient / spend.sum(axis=0)
    """
    values = _response_array(response)
    spending = jnp.asarray(spend)
    if spending.ndim == 0 or spending.shape[-1] != values.shape[-1]:
        raise ValueError(
            "spend must end in the channel axis of response, totaled per channel or with time and group axes"
        )
    total_spend = jnp.sum(spending, axis=tuple(range(spending.ndim - 1)))
    contribution = _channel_vector(roi, values.shape[-1], "roi") * total_spend
    coefficient = contribution_coefficient(
        contribution, values, outcome_scale=outcome_scale, deviations=deviations, effects=effects
    )
    return coefficient


def _response_array(response: ArrayLike) -> jax.Array:
    """Require time and channel axes with an optional group axis between them."""
    values = jnp.asarray(response)
    if values.ndim not in (2, 3):
        raise ValueError(f"response must have shape (time, channel) or (time, group, channel), got {values.ndim} axes")
    return values


def _channel_vector(value: ArrayLike, n_channels: int, name: str) -> jax.Array:
    """Broadcast a scalar or per-channel setting to one value per channel."""
    array = jnp.asarray(value)
    if array.shape not in ((), (n_channels,)):
        raise ValueError(f"{name} must be a scalar or have one value per channel, got shape {array.shape}")
    vector = jnp.broadcast_to(array, (n_channels,))
    return vector


def _scale_factor(outcome_scale: ArrayLike, values: jax.Array) -> jax.Array:
    """Accept a scalar scale or one value per group when the response has groups."""
    array = jnp.asarray(outcome_scale)
    if array.size == 1:
        array = jnp.reshape(array, ())
    if array.ndim == 0:
        return array
    if values.ndim == 3 and array.shape == (values.shape[1],):
        return array
    raise ValueError(f"outcome_scale must be a scalar or have one value per group, got shape {array.shape}")


def _deviation_matrix(deviations: ArrayLike, values: jax.Array) -> jax.Array:
    """Require deviations shaped by group and channel to match a grouped response."""
    if values.ndim != 3:
        raise ValueError("deviations require response to have a group axis")
    array = jnp.asarray(deviations)
    if array.shape != values.shape[1:]:
        raise ValueError(f"deviations must have shape {values.shape[1:]}, got {array.shape}")
    return array
