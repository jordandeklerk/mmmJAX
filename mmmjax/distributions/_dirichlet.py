"""Dirichlet distribution functions."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, xlogy
from jax.typing import ArrayLike

from mmmjax.distributions._utils import (
    _asymptotic_gamma_shape_log_derivative,
    _asymptotic_gamma_shape_normalizer,
    _gamma_shape_log_derivative,
    _gamma_shape_normalizer,
    _promote_inexact,
    _random_shape,
    _stable_log_ratio,
    _weighted_log_ratio_deviance,
)


def dirichlet_logpdf(
    value: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Dirichlet log density along the final axis.

    For a :math:`K`-component simplex :math:`\mathbf{x}` and positive
    concentration vector :math:`\boldsymbol{\alpha}`, the log density is

    .. math::

        \log p(\mathbf{x} \mid \boldsymbol{\alpha})
        = \log\Gamma\!\left(\sum_{i=1}^{K}\alpha_i\right)
          - \sum_{i=1}^{K}\log\Gamma(\alpha_i)
          + \sum_{i=1}^{K}(\alpha_i - 1)\log(x_i).

    Parameters
    ----------
    value
        Simplex values. The final axis contains the components and every
        leading axis is a batch dimension.
    concentration
        Positive concentration parameters. The final axis must match the
        event size of ``value`` and leading axes are broadcast as batches.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast batch shape. Values
        outside the simplex produce ``-inf`` and a ``nan`` value produces
        ``nan``. A nonpositive or nonfinite concentration produces ``nan``.

        At a zero-valued component, the corresponding density term is
        ``inf``, zero, or ``-inf`` when its concentration is below, equal to,
        or above one. Opposing infinite terms produce ``nan`` because the
        multivariate boundary limit depends on the path of approach.
    """
    value_array, concentration_array = _promote_inexact(
        ("value", value),
        ("concentration", concentration),
    )
    if value_array.ndim == 0:
        raise ValueError("value must include a final Dirichlet event axis, got shape ()")
    if concentration_array.ndim == 0:
        raise ValueError("concentration must include a final Dirichlet event axis, got shape ()")
    if value_array.shape[-1] == 0 or concentration_array.shape[-1] == 0:
        raise ValueError(
            "Dirichlet event size must be positive, "
            f"got value.shape={value_array.shape} and concentration.shape={concentration_array.shape}"
        )
    if value_array.shape[-1] != concentration_array.shape[-1]:
        raise ValueError(
            "value and concentration must have the same final event size, "
            f"got {value_array.shape[-1]} and {concentration_array.shape[-1]}"
        )

    try:
        batch_shape = jnp.broadcast_shapes(value_array.shape[:-1], concentration_array.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "Dirichlet batch shapes must be broadcastable, "
            f"got value.shape={value_array.shape} and concentration.shape={concentration_array.shape}"
        ) from exc

    event_size = value_array.shape[-1]
    output_shape = (*batch_shape, event_size)
    value_array = jnp.broadcast_to(value_array, output_shape)
    concentration_array = jnp.broadcast_to(concentration_array, output_shape)

    valid_concentration = jnp.all(jnp.isfinite(concentration_array) & (concentration_array > 0), axis=-1)
    finite_nonnegative_value = jnp.all(jnp.isfinite(value_array) & (value_array >= 0), axis=-1)
    tolerance = jnp.asarray(
        1e-8 if jax.dtypes.itemsize_bits(value_array.dtype) == 64 else 1e-6,
        dtype=value_array.dtype,
    )
    on_simplex = finite_nonnegative_value & (jnp.abs(jnp.sum(value_array, axis=-1) - 1) <= tolerance)

    safe_event = valid_concentration & on_simplex
    safe_concentration = jnp.where(safe_event[..., None], concentration_array, jnp.ones_like(concentration_array))
    safe_value = jnp.where(
        safe_event[..., None],
        value_array,
        jnp.full_like(value_array, 1 / event_size),
    )

    log_density = _dirichlet_logpdf(safe_value, safe_concentration)

    supported_log_density = jnp.where(on_simplex, log_density, -jnp.inf)
    supported_log_density = jnp.where(jnp.any(jnp.isnan(value_array), axis=-1), jnp.nan, supported_log_density)
    return jnp.where(valid_concentration, supported_log_density, jnp.nan)


def dirichlet(
    value: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Dirichlet log densities.

    Parameters
    ----------
    value
        Simplex values with event components along the final axis.
    concentration
        Positive concentration parameters with the same final event size.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every broadcast batch dimension.
    """
    return jnp.sum(dirichlet_logpdf(value, concentration))


def dirichlet_rng(
    key: jax.Array,
    concentration: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Dirichlet distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    concentration
        Positive concentration parameters. The final axis contains the
        Dirichlet event and every leading axis is a batch dimension.
    sample_shape
        Independent sample dimensions prepended to the concentration batch
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random simplex values with shape
        ``sample_shape + concentration.shape``. An event containing a
        nonpositive or nonfinite concentration produces ``nan``.
    """
    (concentration_array,) = _promote_inexact(("concentration", concentration))
    if concentration_array.ndim == 0:
        raise ValueError("concentration must include a final Dirichlet event axis, got shape ()")
    if concentration_array.shape[-1] == 0:
        raise ValueError(f"Dirichlet event size must be positive, got concentration.shape={concentration_array.shape}")

    output_batch_shape = _random_shape(sample_shape, concentration_array[..., 0])
    valid_concentration = jnp.all(
        jnp.isfinite(concentration_array) & (concentration_array > 0),
        axis=-1,
    )
    # Keep invalid events out of JAX's Gamma sampler
    safe_concentration = jnp.where(
        valid_concentration[..., None],
        concentration_array,
        jnp.ones_like(concentration_array),
    )

    samples = jax.random.dirichlet(
        key,
        safe_concentration,
        shape=output_batch_shape,
        dtype=concentration_array.dtype,
    )
    return jnp.where(valid_concentration[..., None], samples, jnp.nan)


def _dirichlet_logpdf(
    value: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    event_size = value.shape[-1]
    concentration_sum = jnp.sum(concentration, axis=-1)
    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value.dtype)
    has_boundary = jnp.any(value == 0, axis=-1)
    uses_stable_formula = has_boundary | ~jnp.isfinite(concentration_sum) | (concentration_sum >= asymptotic_threshold)

    def evaluate_mixed_batch(_: None) -> jax.Array:
        uniform_value = jnp.full_like(value, 1 / event_size)

        # vmap can turn the conditional into a select, so keep both inactive sides finite
        stable_value = jnp.where(uses_stable_formula[..., None], value, uniform_value)
        stable_concentration = jnp.where(
            uses_stable_formula[..., None],
            concentration,
            jnp.ones_like(concentration),
        )
        stable_log_density = _stable_dirichlet_logpdf(
            stable_value,
            stable_concentration,
        )

        standard_value = jnp.where(uses_stable_formula[..., None], uniform_value, value)
        standard_concentration = jnp.where(
            uses_stable_formula[..., None],
            jnp.ones_like(concentration),
            concentration,
        )
        standard_log_density = _standard_dirichlet_logpdf(
            standard_value,
            standard_concentration,
        )
        return jnp.where(
            uses_stable_formula,
            stable_log_density,
            standard_log_density,
        )

    return cast(
        jax.Array,
        jax.lax.cond(
            jnp.any(uses_stable_formula),
            evaluate_mixed_batch,
            lambda _: _standard_dirichlet_logpdf(value, concentration),
            operand=None,
        ),
    )


def _standard_dirichlet_logpdf(
    value: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    concentration_sum = jnp.sum(concentration, axis=-1)
    return (
        gammaln(concentration_sum)
        - jnp.sum(gammaln(concentration), axis=-1)
        + jnp.sum(xlogy(concentration - 1, value), axis=-1)
    )


@jax.custom_jvp
def _stable_dirichlet_logpdf(
    value: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    (
        mean,
        largest_concentration,
        scaled_sum,
        concentration_sum,
        log_concentration_sum,
        inverse_concentration_sum,
        exact_sum_region,
    ) = _dirichlet_mean_and_total(concentration)

    exact_concentration_sum = jnp.where(
        exact_sum_region,
        concentration_sum,
        jnp.ones_like(concentration_sum),
    )
    sum_normalizer = jnp.where(
        exact_sum_region,
        _gamma_shape_normalizer(exact_concentration_sum),
        _asymptotic_gamma_shape_normalizer(
            log_concentration_sum,
            inverse_concentration_sum,
        ),
    )
    normalizer = jnp.sum(_gamma_shape_normalizer(concentration), axis=-1) - sum_normalizer

    positive_value = value > 0
    safe_value = jnp.where(positive_value, value, mean)
    log_safe_value = jnp.log(safe_value)
    log_concentration = jnp.log(concentration)
    raw_log_ratio = log_safe_value + log_concentration_sum[..., None] - log_concentration
    log_ratio, _, _ = _stable_log_ratio(safe_value, mean, raw_log_ratio)

    # Simplex deviations sum to zero, so removing their linear terms avoids cancellation near the mean
    linear_deviation = largest_concentration * (scaled_sum * (value - mean))
    density_deviation = _weighted_log_ratio_deviance(
        concentration,
        log_ratio,
        linear_deviation,
    )
    interior_contribution = density_deviation - log_safe_value
    boundary_contribution = jnp.where(
        concentration < 1,
        jnp.inf,
        jnp.where(
            concentration == 1,
            1 + log_concentration_sum[..., None] - log_concentration,
            -jnp.inf,
        ),
    )
    component_contribution = jnp.where(
        positive_value,
        interior_contribution,
        boundary_contribution,
    )
    return normalizer + jnp.sum(component_contribution, axis=-1)


@_stable_dirichlet_logpdf.defjvp
def _stable_dirichlet_logpdf_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, concentration = primals
    value_tangent, concentration_tangent = tangents
    log_density = _stable_dirichlet_logpdf(value, concentration)

    (
        mean,
        largest_concentration,
        scaled_sum,
        concentration_sum,
        log_concentration_sum,
        inverse_concentration_sum,
        exact_sum_region,
    ) = _dirichlet_mean_and_total(concentration)
    positive_value = value > 0
    interior_event = jnp.all(positive_value, axis=-1)
    safe_value = jnp.where(positive_value, value, mean)
    exact_concentration_sum = jnp.where(
        exact_sum_region,
        concentration_sum,
        jnp.ones_like(concentration_sum),
    )
    sum_log_derivative = jnp.where(
        exact_sum_region,
        _gamma_shape_log_derivative(exact_concentration_sum),
        _asymptotic_gamma_shape_log_derivative(inverse_concentration_sum),
    )

    raw_log_ratio = jnp.log(safe_value) + log_concentration_sum[..., None] - jnp.log(concentration)
    log_ratio, _, _ = _stable_log_ratio(safe_value, mean, raw_log_ratio)
    concentration_derivative = log_ratio + _gamma_shape_log_derivative(concentration) - sum_log_derivative[..., None]

    scaled_value_tangent = value_tangent / safe_value
    # Accumulating before restoring the largest scale avoids cancellation along simplex tangents
    centered_value_tangent = jnp.sum(
        scaled_sum * ((mean - safe_value) / safe_value) * value_tangent,
        axis=-1,
    )
    normal_value_tangent = jnp.squeeze(scaled_sum, axis=-1) * jnp.sum(
        value_tangent,
        axis=-1,
    )
    value_tangent_contribution = jnp.squeeze(largest_concentration, axis=-1) * (
        centered_value_tangent + normal_value_tangent
    ) - jnp.sum(scaled_value_tangent, axis=-1)
    # The centered shape derivative avoids subtracting nearly equal digamma values
    concentration_tangent_contribution = jnp.sum(
        concentration_derivative * concentration_tangent,
        axis=-1,
    )
    interior_log_density_tangent = value_tangent_contribution + concentration_tangent_contribution

    finite_boundary = jnp.all(positive_value | (concentration == 1), axis=-1)
    # Undefined boundary derivatives use the same zero convention as the scalar kernels
    boundary_value_derivative = jnp.where(
        finite_boundary[..., None] & positive_value,
        (concentration - 1) / safe_value,
        jnp.zeros_like(value),
    )
    positive_concentration = jnp.where(
        positive_value,
        concentration,
        jnp.zeros_like(concentration),
    )
    (
        positive_mean,
        _,
        _,
        positive_concentration_sum,
        log_positive_concentration_sum,
        inverse_positive_concentration_sum,
        exact_positive_sum_region,
    ) = _dirichlet_mean_and_total(positive_concentration)
    exact_positive_concentration_sum = jnp.where(
        exact_positive_sum_region,
        positive_concentration_sum,
        jnp.ones_like(positive_concentration_sum),
    )
    positive_sum_log_derivative = jnp.where(
        exact_positive_sum_region,
        _gamma_shape_log_derivative(exact_positive_concentration_sum),
        _asymptotic_gamma_shape_log_derivative(inverse_positive_concentration_sum),
    )
    boundary_safe_value = jnp.where(positive_value, value, jnp.ones_like(value))
    boundary_safe_mean = jnp.where(positive_value, positive_mean, jnp.ones_like(positive_mean))
    boundary_raw_log_ratio = (
        jnp.log(boundary_safe_value) + log_positive_concentration_sum[..., None] - jnp.log(concentration)
    )
    boundary_log_ratio, _, _ = _stable_log_ratio(
        boundary_safe_value,
        boundary_safe_mean,
        boundary_raw_log_ratio,
    )
    finite_boundary_concentration_derivative = (
        boundary_log_ratio + _gamma_shape_log_derivative(concentration) - positive_sum_log_derivative[..., None]
    )

    zero_count = jnp.sum(~positive_value, axis=-1)
    offsets = jnp.arange(value.shape[-1])
    float_offsets = offsets.astype(value.dtype)
    # The digamma recurrence retains zero-component corrections that fall below the total's ULP
    recurrence_terms = inverse_positive_concentration_sum[..., None] / (
        1 + float_offsets * inverse_positive_concentration_sum[..., None]
    )
    boundary_correction = jnp.sum(
        jnp.where(
            offsets < zero_count[..., None],
            recurrence_terms,
            jnp.zeros_like(recurrence_terms),
        ),
        axis=-1,
    )
    positive_boundary_concentration_derivative = jnp.where(
        finite_boundary[..., None],
        finite_boundary_concentration_derivative + boundary_correction[..., None],
        jnp.zeros_like(concentration_derivative),
    )
    boundary_concentration_derivative = jnp.where(
        positive_value,
        positive_boundary_concentration_derivative,
        jnp.zeros_like(concentration),
    )
    boundary_log_density_tangent = jnp.sum(
        boundary_value_derivative * value_tangent + boundary_concentration_derivative * concentration_tangent,
        axis=-1,
    )
    return log_density, jnp.where(
        interior_event,
        interior_log_density_tangent,
        boundary_log_density_tangent,
    )


def _dirichlet_mean_and_total(
    concentration: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # Scaling by the largest component keeps the total representable near the dtype limit
    largest_concentration = jnp.max(concentration, axis=-1, keepdims=True)
    reciprocal_limit = jnp.asarray(
        1 / np.finfo(concentration.dtype).tiny,
        dtype=concentration.dtype,
    )
    uses_log_scaling = largest_concentration > reciprocal_limit
    direct_largest = jnp.where(
        uses_log_scaling,
        jnp.ones_like(largest_concentration),
        largest_concentration,
    )
    direct_scaled_concentration = concentration / direct_largest
    log_scaled_concentration = jnp.exp(jnp.log(concentration) - jnp.log(largest_concentration))
    scaled_concentration = jnp.where(
        uses_log_scaling,
        log_scaled_concentration,
        direct_scaled_concentration,
    )
    scaled_sum = jnp.sum(scaled_concentration, axis=-1, keepdims=True)
    mean = scaled_concentration / scaled_sum

    concentration_sum = jnp.sum(concentration, axis=-1)
    log_concentration_sum = jnp.squeeze(
        jnp.log(largest_concentration) + jnp.log(scaled_sum),
        axis=-1,
    )
    inverse_concentration_sum = jnp.squeeze(
        (1 / largest_concentration) / scaled_sum,
        axis=-1,
    )

    dtype_bits = jax.dtypes.itemsize_bits(concentration.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=concentration.dtype)
    exact_sum_region = jnp.isfinite(concentration_sum) & (concentration_sum < asymptotic_threshold)
    return (
        mean,
        largest_concentration,
        scaled_sum,
        concentration_sum,
        log_concentration_sum,
        inverse_concentration_sum,
        exact_sum_region,
    )
