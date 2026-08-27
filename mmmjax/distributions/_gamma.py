"""Gamma distribution functions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def gamma_logpdf(
    value: ArrayLike,
    shape: ArrayLike,
    rate: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Gamma log density elementwise.

    For value :math:`x > 0`, shape :math:`\alpha > 0`, and rate
    :math:`\beta > 0`, the log density is

    .. math::

        \log p(x \mid \alpha, \beta)
        = \alpha\log(\beta)
          - \log\Gamma(\alpha)
          + (\alpha - 1)\log(x)
          - \beta x.

    At :math:`x = 0`, the log density is :math:`+\infty` when
    :math:`\alpha < 1`, :math:`\log(\beta)` when :math:`\alpha = 1`, and
    :math:`-\infty` when :math:`\alpha > 1`.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Negative values and positive infinity produce ``-inf``. A nonpositive
        or nonfinite shape or rate produces ``nan``.
    """
    value_array, shape_array, rate_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("rate", rate),
    )

    at_boundary = value_array == 0
    outside_support = value_array < 0
    positive_infinity = jnp.isposinf(value_array)
    # Keep boundary and unsupported values out of undefined logarithms
    safe_value = jnp.where(
        at_boundary | outside_support | positive_infinity,
        jnp.ones_like(value_array),
        value_array,
    )

    interior_log_density = (
        shape_array * jnp.log(rate_array)
        - gammaln(shape_array)
        + xlogy(shape_array - 1, safe_value)
        - rate_array * safe_value
    )
    boundary_log_density = jnp.where(
        shape_array < 1,
        jnp.inf,
        jnp.where(shape_array == 1, jnp.log(rate_array), -jnp.inf),
    )
    supported_log_density = jnp.where(at_boundary, boundary_log_density, interior_log_density)
    supported_log_density = jnp.where(
        outside_support | positive_infinity,
        -jnp.inf,
        supported_log_density,
    )

    valid_parameters = jnp.isfinite(shape_array) & (shape_array > 0) & jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def gamma(
    value: ArrayLike,
    shape: ArrayLike,
    rate: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Gamma log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(gamma_logpdf(value, shape, rate))

    shape_array = jnp.asarray(shape)
    rate_array = jnp.asarray(rate)
    valid_parameters = jnp.all(jnp.isfinite(shape_array) & (shape_array > 0)) & jnp.all(
        jnp.isfinite(rate_array) & (rate_array > 0)
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def gamma_rng(
    key: jax.Array,
    shape: ArrayLike,
    rate: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Gamma distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite shape or rate produces ``nan``.
    """
    shape_array, rate_array = _promote_inexact(("shape", shape), ("rate", rate))
    output_shape = _random_shape(sample_shape, shape_array, rate_array)

    valid_shape = jnp.isfinite(shape_array) & (shape_array > 0)
    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    # Keep invalid shapes out of JAX's rejection sampler
    safe_shape = jnp.where(valid_shape, shape_array, jnp.ones_like(shape_array))
    safe_rate = jnp.where(valid_rate, rate_array, jnp.ones_like(rate_array))

    # Scale in log space so tiny unit-rate draws can be rescaled before they underflow
    log_unit_rate_samples = jax.random.loggamma(
        key,
        safe_shape,
        shape=output_shape,
        dtype=shape_array.dtype,
    )
    samples = jnp.exp(log_unit_rate_samples - jnp.log(safe_rate))

    return jnp.where(valid_shape & valid_rate, samples, jnp.nan)
