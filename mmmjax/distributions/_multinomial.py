"""Multinomial distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import (
    _as_real_array,
    _gamma_shape_normalizer,
    _is_valid_simplex,
    _promote_inexact,
    _random_shape,
    _stable_log_ratio,
    _weighted_log_ratio_deviance,
)


def multinomial_logpmf(
    value: ArrayLike,
    probabilities: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Multinomial log probability mass along the final axis.

    For category counts :math:`\mathbf{x} = (x_1, \ldots, x_K)`, total
    count :math:`N = \sum_{i=1}^K x_i`, and probability vector
    :math:`\mathbf{p}` on the simplex, the log probability mass is

    .. math::

        \log p(\mathbf{x} \mid \mathbf{p})
        = \log\Gamma(N + 1)
          - \sum_{i=1}^K \log\Gamma(x_i + 1)
          + \sum_{i=1}^K x_i \log(p_i).

    Parameters
    ----------
    value
        Nonnegative category counts. The final axis contains the categories,
        every leading axis is a batch dimension, and the total count is
        inferred from each event.
    probabilities
        Category probabilities. The final axis contains the categories and
        every leading axis is a batch dimension.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast batch shape.
        A count event outside the nonnegative integer support produces
        ``-inf`` and an event containing ``nan`` produces ``nan``. A
        probability vector outside the simplex produces ``nan``.
    """
    count, total, probability_array, supported, has_nan, batch_shape = _prepare_multinomial_inputs(
        value,
        probabilities,
        parameter_name="probabilities",
    )
    valid_probability = _is_valid_simplex(probability_array)
    event_size = probability_array.shape[-1]
    safe_probability = jnp.where(
        valid_probability[..., None],
        probability_array,
        jnp.full_like(probability_array, 1 / event_size),
    )
    probability_sum, probability_sum_error = _compensated_event_sum(safe_probability)

    output_shape = (*batch_shape, event_size)
    count = jnp.broadcast_to(count, output_shape)
    total = jnp.broadcast_to(total, batch_shape)
    safe_probability = jnp.broadcast_to(safe_probability, output_shape)
    probability_sum = jnp.broadcast_to(probability_sum, batch_shape)
    probability_sum_error = jnp.broadcast_to(probability_sum_error, batch_shape)
    valid_probability = jnp.broadcast_to(valid_probability, batch_shape)
    supported = jnp.broadcast_to(supported, batch_shape)
    has_nan = jnp.broadcast_to(has_nan, batch_shape)

    # A zero count makes its logarithmic term zero even when its probability is zero
    log_probability = jnp.log(
        jnp.where(count > 0, safe_probability, jnp.ones_like(safe_probability)),
    )
    total_float = jnp.asarray(total, dtype=safe_probability.dtype)
    normalization_correction = total_float * (probability_sum - 1) + total_float * probability_sum_error
    log_mass = _multinomial_log_mass(
        count,
        total,
        safe_probability,
        log_probability,
        normalization_correction,
    )

    log_mass = jnp.where(supported, log_mass, -jnp.inf)
    log_mass = jnp.where(has_nan, jnp.nan, log_mass)
    return jnp.where(valid_probability, log_mass, jnp.nan)


def multinomial(
    value: ArrayLike,
    probabilities: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Multinomial log probability masses.

    Parameters
    ----------
    value
        Nonnegative category counts with categories along the final axis.
    probabilities
        Category probabilities with categories along the final axis.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        broadcast batch dimension.
    """
    return jnp.sum(multinomial_logpmf(value, probabilities))


def multinomial_rng(
    key: jax.Array,
    probabilities: ArrayLike,
    trials: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw Multinomial outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    probabilities
        Category probabilities. The final axis contains the categories and
        every leading axis is a batch dimension. Each vector must be finite,
        nonnegative, and sum to one.
    trials
        Total counts assigned across the categories. Values must be finite
        nonnegative integers.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer category counts with shape ``sample_shape + broadcast_shape
        + (category_count,)``.
    """
    trials_array = _as_real_array("trials", trials)
    (probability_array,) = _promote_inexact(("probabilities", probabilities))
    _validate_multinomial_event_axis(probability_array, parameter_name="probabilities")
    return _draw_multinomial(
        key,
        trials_array,
        probability_array,
        sample_shape=sample_shape,
    )


def multinomial_logit_logpmf(
    value: ArrayLike,
    logits: ArrayLike,
) -> jax.Array:
    r"""Evaluate the logit-parameterized Multinomial log probability mass.

    For category counts :math:`\mathbf{x}`, total count
    :math:`N = \sum_{i=1}^K x_i`, and unnormalized log probabilities
    :math:`\boldsymbol{\eta}`, the log probability mass is

    .. math::

        \log p(\mathbf{x} \mid \boldsymbol{\eta})
        = \log\Gamma(N + 1)
          - \sum_{i=1}^K \log\Gamma(x_i + 1)
          + \sum_{i=1}^K x_i
            \left(\eta_i - \log\!\sum_{j=1}^K e^{\eta_j}\right).

    Parameters
    ----------
    value
        Nonnegative category counts. The final axis contains the categories,
        every leading axis is a batch dimension, and the total count is
        inferred from each event.
    logits
        Unnormalized category log probabilities. The final axis contains the
        categories and every leading axis is a batch dimension. A ``-inf``
        logit masks that category when another category has finite weight.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast batch shape.
        A count event outside the nonnegative integer support produces
        ``-inf`` and an event containing ``nan`` produces ``nan``. A logit
        event containing ``nan`` or ``+inf``, or containing no finite logit,
        produces ``nan``.
    """
    count, total, logits_array, supported, has_nan, batch_shape = _prepare_multinomial_inputs(
        value,
        logits,
        parameter_name="logits",
    )
    has_undefined_logit = jnp.any(
        jnp.isnan(logits_array) | jnp.isposinf(logits_array),
        axis=-1,
    )
    has_finite_logit = jnp.any(jnp.isfinite(logits_array), axis=-1)
    valid_logits = ~has_undefined_logit & has_finite_logit
    safe_logits = jnp.where(
        valid_logits[..., None],
        logits_array,
        jnp.zeros_like(logits_array),
    )
    log_probability = jax.nn.log_softmax(safe_logits, axis=-1)
    probability = jnp.exp(log_probability)

    event_size = logits_array.shape[-1]
    output_shape = (*batch_shape, event_size)
    count = jnp.broadcast_to(count, output_shape)
    total = jnp.broadcast_to(total, batch_shape)
    probability = jnp.broadcast_to(probability, output_shape)
    log_probability = jnp.broadcast_to(log_probability, output_shape)
    valid_logits = jnp.broadcast_to(valid_logits, batch_shape)
    supported = jnp.broadcast_to(supported, batch_shape)
    has_nan = jnp.broadcast_to(has_nan, batch_shape)

    log_mass = _multinomial_log_mass(
        count,
        total,
        probability,
        log_probability,
        jnp.zeros(batch_shape, dtype=probability.dtype),
    )

    log_mass = jnp.where(supported, log_mass, -jnp.inf)
    log_mass = jnp.where(has_nan, jnp.nan, log_mass)
    return jnp.where(valid_logits, log_mass, jnp.nan)


def multinomial_logit(
    value: ArrayLike,
    logits: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of logit-parameterized Multinomial log masses.

    Parameters
    ----------
    value
        Nonnegative category counts with categories along the final axis.
    logits
        Unnormalized category log probabilities with categories along the
        final axis.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        broadcast batch dimension.
    """
    return jnp.sum(multinomial_logit_logpmf(value, logits))


def multinomial_logit_rng(
    key: jax.Array,
    logits: ArrayLike,
    trials: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw logit-parameterized Multinomial outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    logits
        Unnormalized category log probabilities. The final axis contains the
        categories and every leading axis is a batch dimension. A ``-inf``
        logit is never drawn when another category has finite weight. Each
        vector must contain at least one finite value and must not contain
        ``nan`` or ``+inf``.
    trials
        Total counts assigned across the categories. Values must be finite
        nonnegative integers.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer category counts with shape ``sample_shape + broadcast_shape
        + (category_count,)``.
    """
    trials_array = _as_real_array("trials", trials)
    (logits_array,) = _promote_inexact(("logits", logits))
    _validate_multinomial_event_axis(logits_array, parameter_name="logits")
    return _draw_multinomial(
        key,
        trials_array,
        jax.nn.softmax(logits_array, axis=-1),
        sample_shape=sample_shape,
    )


def _multinomial_log_mass(
    count: jax.Array,
    total: jax.Array,
    probability: jax.Array,
    log_probability: jax.Array,
    normalization_correction: jax.Array,
) -> jax.Array:
    count_float = jnp.asarray(count, dtype=probability.dtype)
    total_float = jnp.asarray(total, dtype=probability.dtype)
    has_counts = total > 0
    safe_total = jnp.where(has_counts, total_float, jnp.ones_like(total_float))

    positive_count = count > 0
    safe_count = jnp.where(positive_count, count_float, jnp.ones_like(count_float))
    log_safe_count = jnp.log(safe_count)
    # Loader's deviance form keeps large factorial and probability terms from cancelling
    log_normalizer = (
        jnp.sum(
            jnp.where(
                positive_count,
                _gamma_shape_normalizer(safe_count) - log_safe_count,
                jnp.zeros_like(safe_count),
            ),
            axis=-1,
        )
        - _gamma_shape_normalizer(safe_total)
        + jnp.log(safe_total)
    )

    expected_count = safe_total[..., None] * probability
    raw_log_ratio = jnp.log(safe_total)[..., None] + log_probability - log_safe_count
    use_stable_ratio = (
        positive_count & (expected_count > 0) & jnp.isfinite(expected_count) & jnp.isfinite(raw_log_ratio)
    )
    stable_log_ratio, _, _ = _stable_log_ratio(
        jnp.where(use_stable_ratio, expected_count, safe_count),
        jnp.where(use_stable_ratio, safe_count, jnp.ones_like(safe_count)),
        jnp.where(use_stable_ratio, raw_log_ratio, jnp.zeros_like(raw_log_ratio)),
    )
    log_ratio = jnp.where(use_stable_ratio, stable_log_ratio, raw_log_ratio)

    count_contribution = _weighted_log_ratio_deviance(
        safe_count,
        jnp.where(positive_count, log_ratio, jnp.zeros_like(log_ratio)),
        jnp.where(
            positive_count,
            expected_count - count_float,
            jnp.zeros_like(count_float),
        ),
    )
    component_contribution = jnp.where(
        positive_count,
        count_contribution,
        -expected_count,
    )
    log_mass = log_normalizer + jnp.sum(component_contribution, axis=-1) + normalization_correction
    return jnp.where(has_counts, log_mass, jnp.zeros_like(log_mass))


def _draw_multinomial(
    key: jax.Array,
    trials: jax.Array,
    probabilities: jax.Array,
    *,
    sample_shape: tuple[int, ...],
) -> jax.Array:
    event_size = probabilities.shape[-1]
    output_batch_shape = _random_shape(
        sample_shape,
        trials,
        probabilities[..., 0],
    )
    output_shape = (*output_batch_shape, event_size)
    category = jnp.arange(event_size)
    largest_category = jnp.argmax(probabilities, axis=-1, keepdims=True)
    category_order = jnp.where(
        category == largest_category,
        event_size - 1,
        jnp.where(category == event_size - 1, largest_category, category),
    )
    ordered_probabilities = jnp.take_along_axis(
        probabilities,
        category_order,
        axis=-1,
    )

    # Sampling the largest category last keeps float32 rounding from erasing smaller categories
    ordered_counts = jax.random.multinomial(
        key,
        jnp.asarray(trials, dtype=probabilities.dtype),
        ordered_probabilities,
        shape=output_shape,
        dtype=probabilities.dtype,
    )
    category_order = jnp.broadcast_to(category_order, output_shape)
    counts = jnp.take_along_axis(ordered_counts, category_order, axis=-1)
    return counts.astype(jnp.int32)


def _validate_multinomial_event_axis(parameters: jax.Array, *, parameter_name: str) -> None:
    if parameters.ndim == 0:
        raise ValueError(f"{parameter_name} must include a final Multinomial event axis, got shape ()")
    if parameters.shape[-1] == 0:
        raise ValueError(f"Multinomial event size must be positive, got {parameter_name}.shape={parameters.shape}")


def _compensated_event_sum(value: jax.Array) -> tuple[jax.Array, jax.Array]:
    def add_component(
        carry: tuple[jax.Array, jax.Array],
        component: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        high, low = carry
        # TwoSum recovers each addition's rounding error without requiring float64 mode
        total = high + component
        virtual_component = total - high
        rounding_error = (high - (total - virtual_component)) + (component - virtual_component)
        return (total, low + rounding_error), None

    # lax.scan keeps the recurrence compiled instead of unrolling every category
    zero = jnp.zeros(value.shape[:-1], dtype=value.dtype)
    (high, low), _ = jax.lax.scan(
        add_component,
        (zero, zero),
        jnp.moveaxis(value, -1, 0),
    )
    return high, low


def _prepare_multinomial_inputs(
    value: ArrayLike,
    parameters: ArrayLike,
    *,
    parameter_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple[int, ...]]:
    value_array = _as_real_array("value", value)
    (parameter_array,) = _promote_inexact((parameter_name, parameters))
    if value_array.ndim == 0:
        raise ValueError("value must include a final Multinomial event axis, got shape ()")
    if parameter_array.ndim == 0:
        raise ValueError(f"{parameter_name} must include a final Multinomial event axis, got shape ()")
    if value_array.shape[-1] == 0 or parameter_array.shape[-1] == 0:
        raise ValueError(
            "Multinomial event size must be positive, "
            f"got value.shape={value_array.shape} and {parameter_name}.shape={parameter_array.shape}"
        )
    if value_array.shape[-1] != parameter_array.shape[-1]:
        raise ValueError(
            f"value and {parameter_name} must have the same final event size, "
            f"got {value_array.shape[-1]} and {parameter_array.shape[-1]}"
        )

    try:
        batch_shape = jnp.broadcast_shapes(value_array.shape[:-1], parameter_array.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "Multinomial batch shapes must be broadcastable, "
            f"got value.shape={value_array.shape} and {parameter_name}.shape={parameter_array.shape}"
        ) from exc

    count_dtype = jax.dtypes.canonicalize_dtype(jnp.int64)
    count_bits = jax.dtypes.itemsize_bits(count_dtype)
    count_limit = 2 ** (count_bits - 1)
    integer_count = jnp.asarray(value_array, dtype=count_dtype)

    if jnp.issubdtype(value_array.dtype, jnp.floating):
        range_dtype = jnp.promote_types(value_array.dtype, jnp.float32)
        count_in_range = jnp.asarray(value_array, dtype=range_dtype) < jnp.asarray(
            count_limit,
            dtype=range_dtype,
        )
        exact_count = (
            count_in_range
            & jnp.isfinite(value_array)
            & (value_array == jnp.asarray(integer_count, dtype=value_array.dtype))
        )
    elif jnp.issubdtype(value_array.dtype, jnp.unsignedinteger):
        if jax.dtypes.itemsize_bits(value_array.dtype) >= count_bits:
            exact_count = value_array < jnp.asarray(count_limit, dtype=value_array.dtype)
        else:
            exact_count = jnp.ones_like(value_array, dtype=jnp.bool_)
    else:
        exact_count = jnp.ones_like(value_array, dtype=jnp.bool_)

    supported_component = exact_count & (integer_count >= 0)
    safe_component = jnp.where(
        supported_component,
        integer_count,
        jnp.zeros_like(integer_count),
    )
    reverse_total = jnp.cumsum(safe_component[..., ::-1], axis=-1)[..., ::-1]
    total_fits = jnp.all(reverse_total >= 0, axis=-1)
    supported = jnp.all(supported_component, axis=-1) & total_fits
    count = jnp.where(
        supported[..., None],
        safe_component,
        jnp.zeros_like(safe_component),
    )
    total = jnp.where(supported, reverse_total[..., 0], jnp.zeros_like(reverse_total[..., 0]))
    has_nan = jnp.any(jnp.isnan(value_array), axis=-1)
    return count, total, parameter_array, supported, has_nan, batch_shape
