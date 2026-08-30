"""Provide JAX reference kernels for distribution benchmarks."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import erf, erfc

Kernel = Callable[..., jax.Array]


@dataclass(frozen=True)
class JaxReference:
    """Store equivalent JAX functions for one distribution."""

    logpdf: Kernel
    rng: Kernel
    logcdf: Kernel | None = None
    logsf: Kernel | None = None

    def density(self, *arguments: jax.Array) -> jax.Array:
        """Sum the elementwise log density over every broadcast value."""
        return jnp.sum(self.logpdf(*arguments))


def _exponential_logpdf(value: jax.Array, rate: jax.Array) -> jax.Array:
    return stats.expon.logpdf(value, loc=0, scale=1 / rate)


def _exponential_logcdf(value: jax.Array, rate: jax.Array) -> jax.Array:
    return jax.nn.log1mexp(rate * value)


def _exponential_logsf(value: jax.Array, rate: jax.Array) -> jax.Array:
    return -rate * value


def _gamma_logpdf(value: jax.Array, shape: jax.Array, rate: jax.Array) -> jax.Array:
    return stats.gamma.logpdf(value, shape, loc=0, scale=1 / rate)


def _gamma_logcdf(value: jax.Array, shape: jax.Array, rate: jax.Array) -> jax.Array:
    return stats.gamma.logcdf(rate * value, shape)


def _gamma_logsf(value: jax.Array, shape: jax.Array, rate: jax.Array) -> jax.Array:
    return stats.gamma.logsf(rate * value, shape)


def _half_normal_logpdf(value: jax.Array, scale: jax.Array) -> jax.Array:
    log_two = jnp.asarray(math.log(2), dtype=value.dtype)
    log_density = stats.norm.logpdf(value, loc=0, scale=scale) + log_two
    return jnp.where(value < 0, -jnp.inf, log_density)


def _half_normal_logcdf(value: jax.Array, scale: jax.Array) -> jax.Array:
    sqrt_two = jnp.sqrt(jnp.asarray(2, dtype=value.dtype))
    return jnp.log(erf((value / scale) / sqrt_two))


def _half_normal_logsf(value: jax.Array, scale: jax.Array) -> jax.Array:
    sqrt_two = jnp.sqrt(jnp.asarray(2, dtype=value.dtype))
    return jnp.log(erfc((value / scale) / sqrt_two))


def _inverse_gamma_logpdf(value: jax.Array, shape: jax.Array, scale: jax.Array) -> jax.Array:
    supported = jnp.isfinite(value) & (value > 0)
    safe_value = jnp.where(supported, value, jnp.ones_like(value))
    log_density = stats.gamma.logpdf(
        1 / safe_value,
        shape,
        loc=0,
        scale=1 / scale,
    ) - 2 * jnp.log(safe_value)
    return jnp.where(supported, log_density, jnp.where(jnp.isnan(value), jnp.nan, -jnp.inf))


def _inverse_gamma_logcdf(value: jax.Array, shape: jax.Array, scale: jax.Array) -> jax.Array:
    return stats.gamma.logsf(scale / value, shape)


def _inverse_gamma_logsf(value: jax.Array, shape: jax.Array, scale: jax.Array) -> jax.Array:
    return stats.gamma.logcdf(scale / value, shape)


def _lognormal_logpdf(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    outside_support = value <= 0
    safe_value = jnp.where(outside_support, jnp.ones_like(value), value)
    log_value = jnp.log(safe_value)
    log_density = stats.norm.logpdf(log_value, loc=location, scale=scale) - log_value
    supported_log_density = jnp.where(outside_support, -jnp.inf, log_density)
    return jnp.where(jnp.isnan(value), jnp.nan, supported_log_density)


def _lognormal_logcdf(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return stats.norm.logcdf(jnp.log(value), loc=location, scale=scale)


def _lognormal_logsf(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return stats.norm.logsf(jnp.log(value), loc=location, scale=scale)


def _uniform_logpdf(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    return stats.uniform.logpdf(value, loc=lower, scale=upper - lower)


def _beta_rng(
    key: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, alpha, beta)
    return jax.random.beta(key, alpha, beta, shape=shape, dtype=dtype)


def _exponential_rng(
    key: jax.Array,
    rate: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, rate)
    return jax.random.exponential(key, shape=shape, dtype=dtype) / rate


def _gamma_rng(
    key: jax.Array,
    distribution_shape: jax.Array,
    rate: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, distribution_shape, rate)
    return jax.random.gamma(key, distribution_shape, shape=shape, dtype=dtype) / rate


def _half_normal_rng(
    key: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, scale)
    return jnp.abs(scale * jax.random.normal(key, shape=shape, dtype=dtype))


def _inverse_gamma_rng(
    key: jax.Array,
    distribution_shape: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, distribution_shape, scale)
    return scale / jax.random.gamma(key, distribution_shape, shape=shape, dtype=dtype)


def _laplace_rng(
    key: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, location, scale)
    return location + scale * jax.random.laplace(key, shape=shape, dtype=dtype)


def _lognormal_rng(
    key: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, location, scale)
    # JAX's lognormal sampler has no location parameter, so use the canonical Normal transform
    standard_normal = jax.random.normal(key, shape=shape, dtype=dtype)
    return jnp.exp(location + scale * standard_normal)


def _normal_rng(
    key: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, location, scale)
    return location + scale * jax.random.normal(key, shape=shape, dtype=dtype)


def _student_t_rng(
    key: jax.Array,
    degrees_of_freedom: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, degrees_of_freedom, location, scale)
    return location + scale * jax.random.t(
        key,
        degrees_of_freedom,
        shape=shape,
        dtype=dtype,
    )


def _uniform_rng(
    key: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, lower, upper)
    return jax.random.uniform(
        key,
        shape=shape,
        dtype=dtype,
        minval=lower,
        maxval=upper,
    )


def _random_metadata(
    sample_shape: tuple[int, ...],
    *parameters: jax.Array,
) -> tuple[tuple[int, ...], jnp.dtype]:
    parameter_shape = jnp.broadcast_shapes(*(parameter.shape for parameter in parameters))
    return sample_shape + parameter_shape, jnp.result_type(*parameters)


JAX_REFERENCES: dict[str, JaxReference] = {
    "beta": JaxReference(stats.beta.logpdf, _beta_rng),
    "exponential": JaxReference(
        _exponential_logpdf,
        _exponential_rng,
        logcdf=_exponential_logcdf,
        logsf=_exponential_logsf,
    ),
    "gamma": JaxReference(
        _gamma_logpdf,
        _gamma_rng,
        logcdf=_gamma_logcdf,
        logsf=_gamma_logsf,
    ),
    "half_normal": JaxReference(
        _half_normal_logpdf,
        _half_normal_rng,
        logcdf=_half_normal_logcdf,
        logsf=_half_normal_logsf,
    ),
    "inverse_gamma": JaxReference(
        _inverse_gamma_logpdf,
        _inverse_gamma_rng,
        logcdf=_inverse_gamma_logcdf,
        logsf=_inverse_gamma_logsf,
    ),
    "laplace": JaxReference(stats.laplace.logpdf, _laplace_rng),
    "lognormal": JaxReference(
        _lognormal_logpdf,
        _lognormal_rng,
        logcdf=_lognormal_logcdf,
        logsf=_lognormal_logsf,
    ),
    "normal": JaxReference(
        stats.norm.logpdf,
        _normal_rng,
        logcdf=stats.norm.logcdf,
        logsf=stats.norm.logsf,
    ),
    "student_t": JaxReference(stats.t.logpdf, _student_t_rng),
    "uniform": JaxReference(_uniform_logpdf, _uniform_rng),
}
