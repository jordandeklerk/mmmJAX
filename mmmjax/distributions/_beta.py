"""Beta distribution functions."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import beta as beta_distribution
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def beta_logpdf(
    value: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Beta log density elementwise.

    For value :math:`x \in (0, 1)` and shape parameters :math:`\alpha > 0`
    and :math:`\beta > 0`, the log density is

    .. math::

        \log p(x \mid \alpha, \beta)
        = (\alpha - 1)\log(x)
          + (\beta - 1)\log(1 - x)
          - \log\mathrm{B}(\alpha, \beta).

    At :math:`x = 0`, the log density is :math:`+\infty` when
    :math:`\alpha < 1`, :math:`\log(\beta)` when :math:`\alpha = 1`, and
    :math:`-\infty` when :math:`\alpha > 1`. At :math:`x = 1`, the log
    density is :math:`+\infty` when :math:`\beta < 1`, :math:`\log(\alpha)`
    when :math:`\beta = 1`, and :math:`-\infty` when :math:`\beta > 1`.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values outside ``[0, 1]`` produce ``-inf``. A nonpositive or
        nonfinite shape parameter produces ``nan``.
    """
    value_array, alpha_array, beta_array = _promote_inexact(
        ("value", value),
        ("alpha", alpha),
        ("beta", beta),
    )
    at_lower_boundary = value_array == 0
    at_upper_boundary = value_array == 1
    outside_support = (value_array < 0) | (value_array > 1) | jnp.isinf(value_array)
    # Keep endpoints and unsupported values out of undefined logarithms
    safe_value = jnp.where(
        at_lower_boundary | at_upper_boundary | outside_support,
        jnp.full_like(value_array, 0.5),
        value_array,
    )
    interior_log_density = beta_distribution.logpdf(safe_value, alpha_array, beta_array)
    lower_boundary_log_density = jnp.where(
        alpha_array < 1,
        jnp.inf,
        jnp.where(alpha_array == 1, jnp.log(beta_array), -jnp.inf),
    )
    upper_boundary_log_density = jnp.where(
        beta_array < 1,
        jnp.inf,
        jnp.where(beta_array == 1, jnp.log(alpha_array), -jnp.inf),
    )
    supported_log_density = jnp.where(
        at_lower_boundary,
        lower_boundary_log_density,
        jnp.where(at_upper_boundary, upper_boundary_log_density, interior_log_density),
    )
    supported_log_density = jnp.where(outside_support, -jnp.inf, supported_log_density)
    valid_parameters = jnp.isfinite(alpha_array) & (alpha_array > 0) & jnp.isfinite(beta_array) & (beta_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def beta(
    value: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Beta log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(beta_logpdf(value, alpha, beta))
    alpha_array = jnp.asarray(alpha)
    beta_array = jnp.asarray(beta)
    valid_parameters = jnp.all(jnp.isfinite(alpha_array) & (alpha_array > 0)) & jnp.all(
        jnp.isfinite(beta_array) & (beta_array > 0)
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def beta_rng(
    key: jax.Array,
    alpha: ArrayLike,
    beta: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Beta distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite shape parameter produces ``nan``.
    """
    alpha_array, beta_array = _promote_inexact(("alpha", alpha), ("beta", beta))
    output_shape = _random_shape(sample_shape, alpha_array, beta_array)
    valid_alpha = jnp.isfinite(alpha_array) & (alpha_array > 0)
    valid_beta = jnp.isfinite(beta_array) & (beta_array > 0)
    # Keep invalid shapes out of JAX's rejection sampler
    safe_alpha = jnp.where(valid_alpha, alpha_array, jnp.ones_like(alpha_array))
    safe_beta = jnp.where(valid_beta, beta_array, jnp.ones_like(beta_array))
    samples = jax.random.beta(
        key,
        safe_alpha,
        safe_beta,
        shape=output_shape,
        dtype=alpha_array.dtype,
    )
    return jnp.where(valid_alpha & valid_beta, samples, jnp.nan)
