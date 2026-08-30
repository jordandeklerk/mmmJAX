"""Gamma distribution functions."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, hyp1f1
from jax.scipy.stats import gamma as gamma_distribution
from jax.typing import ArrayLike

from mmmjax.distributions._utils import (
    _gamma_shape_log_derivative,
    _gamma_shape_normalizer,
    _promote_inexact,
    _random_shape,
    _stable_log_ratio,
    _weighted_log_ratio_deviance,
)


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
    return _gamma_logpdf(value_array, shape_array, rate_array)


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
    log_densities = gamma_logpdf(value, shape, rate)
    log_density = jnp.sum(log_densities)

    # Only empty results need a separate check because no element can carry nan into the sum
    if log_densities.size:
        return log_density

    shape_array = jnp.asarray(shape)
    rate_array = jnp.asarray(rate)
    valid_parameters = jnp.all(jnp.isfinite(shape_array) & (shape_array > 0)) & jnp.all(
        jnp.isfinite(rate_array) & (rate_array > 0)
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def gamma_logcdf(
    value: ArrayLike,
    shape: ArrayLike,
    rate: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Gamma log cumulative distribution function elementwise.

    For value :math:`x \geq 0`, shape :math:`\alpha > 0`, and rate
    :math:`\beta > 0`, the log cumulative probability is

    .. math::

        \log F(x \mid \alpha, \beta)
        = \log P(\alpha, \beta x),

    where :math:`P` is the regularized lower incomplete Gamma function.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonpositive or nonfinite shape or rate produces ``nan``.
    """
    value_array, shape_array, rate_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("rate", rate),
    )
    return _gamma_log_probability(value_array, shape_array, rate_array, upper_tail=False)


def gamma_logsf(
    value: ArrayLike,
    shape: ArrayLike,
    rate: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Gamma log survival function elementwise.

    For value :math:`x \geq 0`, shape :math:`\alpha > 0`, and rate
    :math:`\beta > 0`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid \alpha, \beta)
        = \log Q(\alpha, \beta x),

    where :math:`Q` is the regularized upper incomplete Gamma function.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonpositive or nonfinite shape or rate produces ``nan``.
    """
    value_array, shape_array, rate_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("rate", rate),
    )
    return _gamma_log_probability(value_array, shape_array, rate_array, upper_tail=True)


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


@jax.custom_jvp
def _gamma_logpdf(
    value: jax.Array,
    shape: jax.Array,
    rate: jax.Array,
) -> jax.Array:
    ordinary_shape_limit = jnp.asarray(8, dtype=value.dtype)
    uses_ordinary_parameters = jnp.all(jnp.isfinite(shape) & (shape > 0) & (shape < ordinary_shape_limit)) & jnp.all(
        jnp.isfinite(rate) & (rate > 0)
    )

    # Large shapes skip direct terms that would only be discarded by the accuracy check
    return cast(
        jax.Array,
        jax.lax.cond(
            uses_ordinary_parameters,
            _ordinary_gamma_logpdf,
            _stable_gamma_logpdf,
            value,
            shape,
            rate,
        ),
    )


def _ordinary_gamma_logpdf(
    value: jax.Array,
    shape: jax.Array,
    rate: jax.Array,
) -> jax.Array:
    standard_log_density, standard_term_magnitude = _standard_gamma_terms(value, shape, rate)
    uses_standard_formula = _uses_standard_gamma_formula(
        value,
        standard_log_density,
        standard_term_magnitude,
    )

    # Ordinary homogeneous batches do not need the heavier stability calculation
    return cast(
        jax.Array,
        jax.lax.cond(
            uses_standard_formula,
            lambda _: standard_log_density,
            lambda _: _stable_gamma_logpdf(value, shape, rate),
            operand=None,
        ),
    )


@_gamma_logpdf.defjvp
def _gamma_logpdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    log_density = _gamma_logpdf(*primals)

    # The robust rule preserves derivatives near the mode and at extreme scales
    _, log_density_tangent = jax.jvp(_stable_gamma_logpdf, primals, tangents)
    return log_density, log_density_tangent


@jax.custom_jvp
def _stable_gamma_logpdf(
    value: jax.Array,
    shape: jax.Array,
    rate: jax.Array,
) -> jax.Array:
    valid_value = jnp.isfinite(value) & (value > 0)
    valid_shape = jnp.isfinite(shape) & (shape > 0)
    valid_rate = jnp.isfinite(rate) & (rate > 0)

    # Sanitizing every candidate branch keeps invalid inputs out of JAX derivatives
    safe_value = jnp.where(valid_value, value, jnp.ones_like(value))
    safe_shape = jnp.where(valid_shape, shape, jnp.ones_like(shape))
    safe_rate = jnp.where(valid_rate, rate, jnp.ones_like(rate))

    log_ratio, density_deviation, _, _, has_density_deviation = _gamma_ratio_terms(
        safe_value,
        safe_shape,
        safe_rate,
    )
    density_contribution = _weighted_log_ratio_deviance(
        safe_shape,
        log_ratio,
        jnp.where(has_density_deviation, density_deviation, jnp.inf),
    )
    interior_log_density = _gamma_shape_normalizer(safe_shape) + density_contribution - jnp.log(safe_value)

    boundary_log_density = jnp.where(
        safe_shape < 1,
        jnp.inf,
        jnp.where(safe_shape == 1, jnp.log(safe_rate), -jnp.inf),
    )
    supported_log_density = jnp.where(
        value == 0,
        boundary_log_density,
        jnp.where(valid_value, interior_log_density, jnp.where(jnp.isnan(value), jnp.nan, -jnp.inf)),
    )
    return jnp.where(valid_shape & valid_rate, supported_log_density, jnp.nan)


@_stable_gamma_logpdf.defjvp
def _stable_gamma_logpdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, shape, rate = primals
    value_tangent, shape_tangent, rate_tangent = tangents
    log_density = _stable_gamma_logpdf(value, shape, rate)

    valid_value = jnp.isfinite(value) & (value > 0)
    valid_shape = jnp.isfinite(shape) & (shape > 0)
    valid_rate = jnp.isfinite(rate) & (rate > 0)
    safe_value = jnp.where(valid_value, value, jnp.ones_like(value))
    safe_shape = jnp.where(valid_shape, shape, jnp.ones_like(shape))
    safe_rate = jnp.where(valid_rate, rate, jnp.ones_like(rate))

    log_ratio, density_deviation, ratio_deviation, near_unit_ratio, _ = _gamma_ratio_terms(
        safe_value,
        safe_shape,
        safe_rate,
    )
    centered_value_derivative = -(density_deviation + 1) / safe_value
    ordinary_value_derivative = (safe_shape - 1) / safe_value - safe_rate
    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value.dtype)
    # The direct form avoids cancellation when an ordinary shape is close to one
    value_derivative = jnp.where(
        safe_shape >= asymptotic_threshold,
        centered_value_derivative,
        ordinary_value_derivative,
    )
    shape_derivative = log_ratio + _gamma_shape_log_derivative(safe_shape)

    direct_rate_derivative = safe_shape / safe_rate - safe_value
    near_ratio_deviation = jnp.where(
        near_unit_ratio,
        ratio_deviation,
        jnp.zeros_like(ratio_deviation),
    )
    near_rate_derivative = -safe_value * near_ratio_deviation / (1 + near_ratio_deviation)
    rate_derivative = jnp.where(
        near_unit_ratio,
        near_rate_derivative,
        direct_rate_derivative,
    )

    value_derivative = jnp.where(valid_value, value_derivative, jnp.zeros_like(value_derivative))
    shape_derivative = jnp.where(valid_value, shape_derivative, jnp.zeros_like(shape_derivative))
    boundary_rate_derivative = jnp.where(
        (value == 0) & (safe_shape == 1),
        1 / safe_rate,
        jnp.zeros_like(rate_derivative),
    )
    rate_derivative = jnp.where(valid_value, rate_derivative, boundary_rate_derivative)

    defined_derivatives = valid_shape & valid_rate & ~jnp.isnan(value)
    value_derivative = jnp.where(defined_derivatives, value_derivative, jnp.nan)
    shape_derivative = jnp.where(defined_derivatives, shape_derivative, jnp.nan)
    rate_derivative = jnp.where(defined_derivatives, rate_derivative, jnp.nan)

    log_density_tangent = (
        value_derivative * value_tangent + shape_derivative * shape_tangent + rate_derivative * rate_tangent
    )
    return log_density, log_density_tangent


def _gamma_ratio_terms(
    value: jax.Array,
    shape: jax.Array,
    rate: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    scaled_value = rate * value
    product_density_deviation = scaled_value - shape
    raw_log_ratio = jnp.log(rate) + jnp.log(value) - jnp.log(shape)
    product_log_ratio, product_ratio_deviation, valid_product_ratio = _stable_log_ratio(
        scaled_value,
        shape,
        raw_log_ratio,
    )

    rate_first_ratio = (rate / shape) * value
    rate_first_log_ratio, rate_first_deviation, valid_rate_first_ratio = _stable_log_ratio(
        rate_first_ratio,
        jnp.ones_like(rate_first_ratio),
        raw_log_ratio,
    )
    value_first_ratio = (value / shape) * rate
    value_first_log_ratio, value_first_deviation, valid_value_first_ratio = _stable_log_ratio(
        value_first_ratio,
        jnp.ones_like(value_first_ratio),
        raw_log_ratio,
    )

    # Dividing equal shape and rate values first preserves displacements near the mode
    use_rate_first_ratio = valid_rate_first_ratio & ((rate == shape) | ~valid_product_ratio)
    log_ratio = jnp.where(
        use_rate_first_ratio,
        rate_first_log_ratio,
        jnp.where(valid_product_ratio, product_log_ratio, value_first_log_ratio),
    )
    ratio_deviation = jnp.where(
        use_rate_first_ratio,
        rate_first_deviation,
        jnp.where(valid_product_ratio, product_ratio_deviation, value_first_deviation),
    )
    has_finite_ratio = valid_product_ratio | valid_rate_first_ratio | valid_value_first_ratio

    valid_product_density_deviation = jnp.isfinite(product_density_deviation)
    fallback_density_deviation = shape * ratio_deviation
    valid_fallback_density_deviation = has_finite_ratio & jnp.isfinite(fallback_density_deviation)
    use_rate_first_density_deviation = use_rate_first_ratio & valid_fallback_density_deviation
    has_density_deviation = valid_product_density_deviation | valid_fallback_density_deviation
    density_deviation = jnp.where(
        use_rate_first_density_deviation,
        fallback_density_deviation,
        jnp.where(
            valid_product_density_deviation,
            product_density_deviation,
            fallback_density_deviation,
        ),
    )
    near_unit_ratio = has_finite_ratio & (jnp.abs(ratio_deviation) < 0.5)
    return log_ratio, density_deviation, ratio_deviation, near_unit_ratio, has_density_deviation


def _standard_gamma_terms(
    value: jax.Array,
    shape: jax.Array,
    rate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    log_rate = jnp.log(rate)
    shape_normalizer = gammaln(shape)
    shape_contribution = (shape - 1) * (log_rate + jnp.log(value))
    rate_contribution = rate * value
    log_density = log_rate - shape_normalizer + shape_contribution - rate_contribution
    term_magnitude = (
        jnp.abs(log_rate) + jnp.abs(shape_normalizer) + jnp.abs(shape_contribution) + jnp.abs(rate_contribution)
    )
    return log_density, term_magnitude


def _uses_standard_gamma_formula(
    value: jax.Array,
    log_density: jax.Array,
    term_magnitude: jax.Array,
) -> jax.Array:
    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    accuracy_tolerance = jnp.asarray(1e-14 if dtype_bits == 64 else 3e-6, dtype=value.dtype)
    machine_epsilon = jnp.spacing(jnp.asarray(1, dtype=value.dtype))
    # Keep the direct sum within the numerical accuracy expected for each dtype
    roundoff_bound = 4 * machine_epsilon * term_magnitude
    cancellation_safe = roundoff_bound <= accuracy_tolerance * (1 + jnp.abs(log_density))
    valid_inputs = jnp.isfinite(value) & (value > 0) & jnp.isfinite(log_density) & cancellation_safe
    return jnp.all(valid_inputs)


def _gamma_log_probability(
    value: jax.Array,
    shape: jax.Array,
    rate: jax.Array,
    *,
    upper_tail: bool,
    log_scaled_value: jax.Array | None = None,
) -> jax.Array:
    valid_shape = jnp.isfinite(shape) & (shape > 0)
    valid_rate = jnp.isfinite(rate) & (rate > 0)
    valid_parameters = valid_shape & valid_rate

    safe_shape = jnp.where(valid_shape, shape, jnp.ones_like(shape))
    safe_rate = jnp.where(valid_rate, rate, jnp.ones_like(rate))
    scaled_value = safe_rate * value

    if log_scaled_value is None:
        positive_finite_value = jnp.isfinite(value) & (value > 0)
        safe_value = jnp.where(positive_finite_value, value, jnp.ones_like(value))
        interior_log_scaled_value = jnp.log(safe_rate) + jnp.log(safe_value)
        log_scaled_value = jnp.where(
            positive_finite_value,
            interior_log_scaled_value,
            -jnp.inf,
        )

    if upper_tail:
        log_probability = _gamma_logsf_kernel(
            scaled_value,
            safe_shape,
            log_scaled_value,
        )
    else:
        log_probability = _gamma_logcdf_kernel(
            scaled_value,
            safe_shape,
            log_scaled_value,
        )

    return jnp.where(valid_parameters, log_probability, jnp.nan)


def _gamma_logcdf_kernel(
    scaled_value: jax.Array,
    shape: jax.Array,
    log_scaled_value: jax.Array,
) -> jax.Array:
    safe_log_scaled_value = jnp.where(
        jnp.isfinite(log_scaled_value),
        log_scaled_value,
        jnp.zeros_like(log_scaled_value),
    )
    log_series_prefix = shape * safe_log_scaled_value - scaled_value - gammaln(shape + 1)
    minimum_normal = jnp.asarray(
        np.finfo(scaled_value.dtype).tiny,
        dtype=scaled_value.dtype,
    )
    use_log_series = (
        jnp.isfinite(log_scaled_value)
        & jnp.isfinite(scaled_value)
        & (scaled_value >= 0)
        & (scaled_value < 100)
        & (shape > scaled_value)
        & (log_series_prefix < jnp.log(minimum_normal))
    )

    def recover_underflow(_: None) -> jax.Array:
        direct_value = jnp.where(use_log_series, jnp.ones_like(scaled_value), scaled_value)
        direct_shape = jnp.where(use_log_series, jnp.ones_like(shape), shape)
        safe_direct_log_probability = gamma_distribution.logcdf(direct_value, direct_shape)

        series_value = jnp.where(use_log_series, scaled_value, jnp.zeros_like(scaled_value))
        series_shape = jnp.where(use_log_series, shape, jnp.ones_like(shape))
        series_log_value = jnp.where(
            use_log_series,
            log_scaled_value,
            jnp.zeros_like(log_scaled_value),
        )
        series_log_probability = (
            series_shape * series_log_value
            - series_value
            - gammaln(series_shape + 1)
            + jnp.log(
                hyp1f1(
                    jnp.ones_like(series_shape),
                    series_shape + 1,
                    series_value,
                )
            )
        )
        return jnp.where(
            use_log_series,
            series_log_probability,
            safe_direct_log_probability,
        )

    # Avoid paying for the hypergeometric series on ordinary probabilities
    return cast(
        jax.Array,
        jax.lax.cond(
            jnp.any(use_log_series),
            recover_underflow,
            lambda _: gamma_distribution.logcdf(scaled_value, shape),
            operand=None,
        ),
    )


def _gamma_logsf_kernel(
    scaled_value: jax.Array,
    shape: jax.Array,
    log_scaled_value: jax.Array,
) -> jax.Array:
    continued_fraction_region = jnp.isfinite(scaled_value) & jnp.isfinite(log_scaled_value) & (scaled_value > shape + 1)
    prefix_value = jnp.where(
        continued_fraction_region,
        scaled_value,
        jnp.ones_like(scaled_value),
    )
    prefix_shape = jnp.where(
        continued_fraction_region,
        shape,
        jnp.ones_like(shape),
    )
    prefix_log_value = jnp.where(
        continued_fraction_region,
        log_scaled_value,
        jnp.zeros_like(log_scaled_value),
    )
    log_prefactor = prefix_shape * prefix_log_value - prefix_value - gammaln(prefix_shape)
    first_fraction_log_probability = log_prefactor - jnp.log(prefix_value + 1 - prefix_shape)
    minimum_normal = jnp.asarray(
        np.finfo(scaled_value.dtype).tiny,
        dtype=scaled_value.dtype,
    )
    use_continued_fraction = continued_fraction_region & (first_fraction_log_probability < jnp.log(minimum_normal))

    def recover_underflow(_: None) -> jax.Array:
        direct_value = jnp.where(
            use_continued_fraction,
            jnp.ones_like(scaled_value),
            scaled_value,
        )
        direct_shape = jnp.where(
            use_continued_fraction,
            jnp.ones_like(shape),
            shape,
        )
        safe_direct_log_probability = gamma_distribution.logsf(direct_value, direct_shape)

        fraction_value = jnp.where(
            use_continued_fraction,
            scaled_value,
            jnp.ones_like(scaled_value),
        )
        fraction_shape = jnp.where(
            use_continued_fraction,
            shape,
            jnp.ones_like(shape),
        )
        fraction_log_value = jnp.where(
            use_continued_fraction,
            log_scaled_value,
            jnp.zeros_like(log_scaled_value),
        )
        log_prefactor = (
            _gamma_logpdf(
                fraction_value,
                fraction_shape,
                jnp.ones_like(fraction_value),
            )
            + fraction_log_value
        )
        continued_fraction_log_probability = log_prefactor + jnp.log(
            _gamma_upper_tail_fraction(fraction_shape, fraction_value)
        )
        return jnp.where(
            use_continued_fraction,
            continued_fraction_log_probability,
            safe_direct_log_probability,
        )

    # The continued fraction only runs when native JAX loses the upper tail
    return cast(
        jax.Array,
        jax.lax.cond(
            jnp.any(use_continued_fraction),
            recover_underflow,
            lambda _: gamma_distribution.logsf(scaled_value, shape),
            operand=None,
        ),
    )


def _gamma_upper_tail_fraction(
    shape: jax.Array,
    scaled_value: jax.Array,
) -> jax.Array:
    minimum_normal = jnp.asarray(
        np.finfo(scaled_value.dtype).tiny,
        dtype=scaled_value.dtype,
    )
    denominator = scaled_value + 1 - shape
    d_term = 1 / denominator
    c_term = jnp.ones_like(scaled_value) / minimum_normal
    fraction = d_term
    active = jnp.ones_like(scaled_value, dtype=jnp.bool_)
    precision = jnp.spacing(jnp.ones((), dtype=scaled_value.dtype))

    def update_fraction(
        index: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        current_denominator, current_c, current_d, current_fraction, current_active = state
        iteration = jnp.asarray(index, dtype=scaled_value.dtype)
        numerator = -iteration * (iteration - shape)
        next_denominator = current_denominator + 2

        next_d = numerator * current_d + next_denominator
        next_d = jnp.where(
            jnp.abs(next_d) < minimum_normal,
            jnp.copysign(minimum_normal, next_d),
            next_d,
        )
        next_c = next_denominator + numerator / current_c
        next_c = jnp.where(
            jnp.abs(next_c) < minimum_normal,
            jnp.copysign(minimum_normal, next_c),
            next_c,
        )
        next_d = 1 / next_d
        multiplier = next_c * next_d
        fraction_candidate = current_fraction * multiplier
        next_active = current_active & (jnp.abs(multiplier - 1) > precision)

        return (
            jnp.where(current_active, next_denominator, current_denominator),
            jnp.where(current_active, next_c, current_c),
            jnp.where(current_active, next_d, current_d),
            jnp.where(current_active, fraction_candidate, current_fraction),
            next_active,
        )

    # More precision gets more room to converge without slowing the ordinary JAX path
    iteration_count = 4 * jax.dtypes.itemsize_bits(scaled_value.dtype)
    initial_state = (
        denominator,
        c_term,
        d_term,
        fraction,
        active,
    )
    return cast(
        jax.Array,
        jax.lax.fori_loop(
            1,
            iteration_count + 1,
            update_fraction,
            initial_state,
        )[3],
    )
