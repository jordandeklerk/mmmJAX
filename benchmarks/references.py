"""Provide JAX reference kernels for distribution benchmarks."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import erf, erfc, gammaln

Kernel = Callable[..., jax.Array]


@dataclass(frozen=True)
class JaxReference:
    """Store equivalent JAX functions for one distribution."""

    elementwise_log_probability: Kernel
    rng: Kernel | None
    logcdf: Kernel | None = None
    logsf: Kernel | None = None

    def summed_log_probability(self, *arguments: jax.Array) -> jax.Array:
        """Sum the elementwise log probabilities over every broadcast value."""
        return jnp.sum(self.elementwise_log_probability(*arguments))


def _bernoulli_logit_logpmf(value: jax.Array, logits: jax.Array) -> jax.Array:
    signed_logits = jnp.where(value == 1, logits, -logits)
    return jax.nn.log_sigmoid(signed_logits)


def _binomial_logit_logpmf(
    value: jax.Array,
    trials: jax.Array,
    logits: jax.Array,
) -> jax.Array:
    log_coefficient = gammaln(trials + 1) - gammaln(value + 1) - gammaln(trials - value + 1)
    return log_coefficient + value * jax.nn.log_sigmoid(logits) + (trials - value) * jax.nn.log_sigmoid(-logits)


def _categorical_logpmf(value: jax.Array, probabilities: jax.Array) -> jax.Array:
    return _select_categorical_log_probability(value, jnp.log(probabilities))


def _categorical_logit_logpmf(value: jax.Array, logits: jax.Array) -> jax.Array:
    return _select_categorical_log_probability(value, jax.nn.log_softmax(logits, axis=-1))


def _dirichlet_logpdf(value: jax.Array, concentration: jax.Array) -> jax.Array:
    if concentration.ndim == 0:
        raise ValueError("Dirichlet reference concentration must include a final event axis")

    sample_ndim = value.ndim - concentration.ndim
    if sample_ndim < 0 or value.shape[sample_ndim:] != concentration.shape:
        raise ValueError(
            "Dirichlet reference values must have shape sample_shape + concentration.shape, "
            f"got value.shape={value.shape} and concentration.shape={concentration.shape}"
        )

    if concentration.ndim == 1:
        return stats.dirichlet.logpdf(jnp.moveaxis(value, -1, 0), concentration)

    batch_shape = concentration.shape[:-1]
    sample_shape = value.shape[:sample_ndim]
    event_axis = value.ndim - 1
    batch_axes = tuple(range(sample_ndim, event_axis))
    sample_axes = tuple(range(sample_ndim))

    # JAX handles shared samples but needs vmap for each batched concentration vector
    event_first = jnp.transpose(value, (*batch_axes, event_axis, *sample_axes))
    batched_values = event_first.reshape(math.prod(batch_shape), concentration.shape[-1], *sample_shape)
    batched_concentration = concentration.reshape(math.prod(batch_shape), concentration.shape[-1])
    log_density = jax.vmap(stats.dirichlet.logpdf)(batched_values, batched_concentration)

    batch_first_shape = (*batch_shape, *sample_shape)
    batch_first_log_density = log_density.reshape(batch_first_shape)
    output_axes = (*range(len(batch_shape), len(batch_first_shape)), *range(len(batch_shape)))
    return jnp.transpose(batch_first_log_density, output_axes)


def _multinomial_logpmf(value: jax.Array, probabilities: jax.Array) -> jax.Array:
    batch_shape = jnp.broadcast_shapes(value.shape[:-1], probabilities.shape[:-1])
    event_size = probabilities.shape[-1]
    event_shape = (*batch_shape, event_size)
    count = jnp.broadcast_to(value, event_shape)
    probability = jnp.broadcast_to(probabilities, event_shape)
    total = jnp.sum(count, axis=-1)

    # vmap keeps JAX's total-count check eventwise while adding mmmJAX's batch contract
    log_mass = jax.vmap(stats.multinomial.logpmf)(
        count.reshape(-1, event_size),
        total.reshape(-1),
        probability.reshape(-1, event_size),
    )
    return log_mass.reshape(batch_shape)


def _multinomial_logit_logpmf(value: jax.Array, logits: jax.Array) -> jax.Array:
    return _multinomial_logpmf(value, jax.nn.softmax(logits, axis=-1))


def _negative_binomial_logpmf(
    value: jax.Array,
    mean: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    probability = concentration / (concentration + mean)
    return stats.nbinom.logpmf(value, concentration, probability)


def _negative_binomial_log_logpmf(
    value: jax.Array,
    log_mean: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    probability = jax.nn.sigmoid(jnp.log(concentration) - log_mean)
    return stats.nbinom.logpmf(value, concentration, probability)


def _poisson_log_logpmf(value: jax.Array, log_rate: jax.Array) -> jax.Array:
    # Staying on the log scale avoids an exp/log round trip in the native JAX baseline
    return value * log_rate - jnp.exp(log_rate) - gammaln(value + 1)


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


def _laplace_logcdf(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return jnp.log(stats.laplace.cdf(value, loc=location, scale=scale))


def _laplace_logsf(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    # Reflecting the CDF avoids cancellation from computing one minus the CDF
    return jnp.log(stats.laplace.cdf(-value, loc=-location, scale=scale))


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


def _uniform_logcdf(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    return jnp.log(stats.uniform.cdf(value, loc=lower, scale=upper - lower))


def _uniform_logsf(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    # Reflecting the CDF avoids cancellation from computing one minus the CDF
    return jnp.log(stats.uniform.cdf(-value, loc=-upper, scale=upper - lower))


def _truncated_normal_logpdf(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    standardized_lower = (lower - location) / scale
    standardized_upper = (upper - location) / scale
    return cast(
        jax.Array,
        stats.truncnorm.logpdf(
            value,
            standardized_lower,
            standardized_upper,
            loc=location,
            scale=scale,
        ),
    )


def _bernoulli_rng(
    key: jax.Array,
    probability: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, _ = _random_metadata(sample_shape, probability)
    return jax.random.bernoulli(key, probability, shape=shape, mode="high").astype(jnp.int32)


def _bernoulli_logit_rng(
    key: jax.Array,
    logits: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, _ = _random_metadata(sample_shape, logits)
    categorical_logits = jnp.stack((jnp.zeros_like(logits), logits), axis=-1)
    return jax.random.categorical(key, categorical_logits, shape=shape, mode="high").astype(jnp.int32)


def _binomial_rng(
    key: jax.Array,
    trials: jax.Array,
    probability: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, trials, probability)
    return jax.random.binomial(key, trials, probability, shape=shape, dtype=dtype).astype(jnp.int32)


def _binomial_logit_rng(
    key: jax.Array,
    trials: jax.Array,
    logits: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, trials, logits)
    rare_probability = jnp.exp(jax.nn.log_sigmoid(-jnp.abs(logits)))
    rare_outcomes = jax.random.binomial(key, trials, rare_probability, shape=shape, dtype=dtype)
    return jnp.where(logits > 0, trials - rare_outcomes, rare_outcomes).astype(jnp.int32)


def _categorical_rng(
    key: jax.Array,
    probabilities: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    return _categorical_logit_rng(
        key,
        jnp.log(probabilities),
        sample_shape=sample_shape,
    )


def _categorical_logit_rng(
    key: jax.Array,
    logits: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape = sample_shape + logits.shape[:-1]
    return jax.random.categorical(key, logits, shape=shape, mode="high").astype(jnp.int32)


def _dirichlet_rng(
    key: jax.Array,
    concentration: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    return jax.random.dirichlet(
        key,
        concentration,
        shape=sample_shape + concentration.shape[:-1],
        dtype=concentration.dtype,
    )


def _multinomial_rng(
    key: jax.Array,
    probabilities: jax.Array,
    trials: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    batch_shape = jnp.broadcast_shapes(trials.shape, probabilities.shape[:-1])
    output_shape = (*sample_shape, *batch_shape, probabilities.shape[-1])
    return jnp.asarray(
        jax.random.multinomial(
            key,
            trials,
            probabilities,
            shape=output_shape,
            dtype=probabilities.dtype,
        ),
        dtype=jnp.int32,
    )


def _multinomial_logit_rng(
    key: jax.Array,
    logits: jax.Array,
    trials: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    return _multinomial_rng(
        key,
        jax.nn.softmax(logits, axis=-1),
        trials,
        sample_shape=sample_shape,
    )


def _negative_binomial_rng(
    key: jax.Array,
    mean: jax.Array,
    concentration: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, mean, concentration)
    gamma_key, poisson_key = jax.random.split(key)
    latent_rate = jax.random.gamma(
        gamma_key,
        concentration,
        shape=shape,
        dtype=dtype,
    ) * (mean / concentration)
    return jax.random.poisson(
        poisson_key,
        latent_rate,
        shape=shape,
        dtype=jnp.int32,
    )


def _negative_binomial_log_rng(
    key: jax.Array,
    log_mean: jax.Array,
    concentration: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    return _negative_binomial_rng(
        key,
        jnp.exp(log_mean),
        concentration,
        sample_shape=sample_shape,
    )


def _poisson_rng(
    key: jax.Array,
    rate: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, _ = _random_metadata(sample_shape, rate)
    return jax.random.poisson(key, rate, shape=shape, dtype=jnp.int32)


def _poisson_log_rng(
    key: jax.Array,
    log_rate: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    return _poisson_rng(key, jnp.exp(log_rate), sample_shape=sample_shape)


def _beta_rng(
    key: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, alpha, beta)
    return jax.random.beta(key, alpha, beta, shape=shape, dtype=dtype)


def _cauchy_rng(
    key: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    shape, dtype = _random_metadata(sample_shape, location, scale)
    return location + scale * jax.random.cauchy(key, shape=shape, dtype=dtype)


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


def _select_categorical_log_probability(value: jax.Array, log_probabilities: jax.Array) -> jax.Array:
    batch_shape = jnp.broadcast_shapes(value.shape, log_probabilities.shape[:-1])
    event_size = log_probabilities.shape[-1]
    value = jnp.broadcast_to(value, batch_shape).astype(jnp.int32)
    log_probabilities = jnp.broadcast_to(log_probabilities, (*batch_shape, event_size))
    return jnp.take_along_axis(log_probabilities, value[..., None], axis=-1)[..., 0]


JAX_REFERENCES: dict[str, JaxReference] = {
    "bernoulli": JaxReference(stats.bernoulli.logpmf, _bernoulli_rng),
    "bernoulli_logit": JaxReference(_bernoulli_logit_logpmf, _bernoulli_logit_rng),
    "beta": JaxReference(stats.beta.logpdf, _beta_rng),
    "binomial": JaxReference(stats.binom.logpmf, _binomial_rng),
    "binomial_logit": JaxReference(_binomial_logit_logpmf, _binomial_logit_rng),
    "categorical": JaxReference(_categorical_logpmf, _categorical_rng),
    "categorical_logit": JaxReference(_categorical_logit_logpmf, _categorical_logit_rng),
    "dirichlet": JaxReference(_dirichlet_logpdf, _dirichlet_rng),
    "multinomial": JaxReference(_multinomial_logpmf, _multinomial_rng),
    "multinomial_logit": JaxReference(_multinomial_logit_logpmf, _multinomial_logit_rng),
    "negative_binomial": JaxReference(_negative_binomial_logpmf, _negative_binomial_rng),
    "negative_binomial_log": JaxReference(
        _negative_binomial_log_logpmf,
        _negative_binomial_log_rng,
    ),
    "poisson": JaxReference(stats.poisson.logpmf, _poisson_rng),
    "poisson_log": JaxReference(_poisson_log_logpmf, _poisson_log_rng),
    "cauchy": JaxReference(stats.cauchy.logpdf, _cauchy_rng),
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
    "laplace": JaxReference(
        stats.laplace.logpdf,
        _laplace_rng,
        logcdf=_laplace_logcdf,
        logsf=_laplace_logsf,
    ),
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
    "truncated_normal": JaxReference(_truncated_normal_logpdf, None),
    "uniform": JaxReference(
        _uniform_logpdf,
        _uniform_rng,
        logcdf=_uniform_logcdf,
        logsf=_uniform_logsf,
    ),
}
