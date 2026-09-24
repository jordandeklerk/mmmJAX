"""Tests for reusable probability terms and their sampling shapes."""

from dataclasses import FrozenInstanceError
from functools import wraps
from inspect import Parameter, signature

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

import mmmjax
import mmmjax.distributions as dist
from mmmjax import Data, Model, Positive, Real, custom_distribution, prepare_data, sample_prior
from mmmjax.distributions._distribution import _bind_distribution, _get_distribution_spec
from mmmjax.distributions._normal import normal as module_normal
from mmmjax.distributions._normal import normal_logpdf as module_normal_logpdf
from mmmjax.distributions._normal import normal_rng as module_normal_rng
from mmmjax.inference.priors import Prior


def test_module_distribution_binding_preserves_public_callable_and_jit_behavior():
    assert module_normal is dist.normal is mmmjax.normal
    assert module_normal_logpdf is dist.normal_logpdf is mmmjax.normal_logpdf
    assert module_normal_rng is dist.normal_rng is mmmjax.normal_rng
    parameters = signature(module_normal, follow_wrapped=False).parameters
    assert tuple(parameters) == ("value", "location", "scale")
    assert all(parameter.kind is Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())
    assert all(parameter.default is Parameter.empty for parameter in parameters.values())
    assert module_normal.__module__ == "mmmjax.distributions._normal"
    binding = _get_distribution_spec(module_normal)
    assert binding is module_normal._mmmjax_distribution
    assert binding.density is module_normal
    assert binding.logpdf is module_normal_logpdf
    assert binding.rng is module_normal_rng

    values = jnp.array([-0.5, 0.75, 2.0])
    location, scale = 0.25, 1.5
    value, gradient = jax.jit(jax.value_and_grad(module_normal))(values, location, scale)
    expected = tfd.Normal(location, scale).log_prob(values).sum()
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    np.testing.assert_allclose(gradient, -(values - location) / scale**2, rtol=1e-6)
    np.testing.assert_allclose(Prior(module_normal, location=location, scale=scale)(values), value)


@pytest.mark.parametrize(
    ("distribution", "parameter_events", "event_ndims", "dimension_parameter"),
    [
        (dist.normal, {"location": 0, "scale": 0}, 0, None),
        (dist.gamma, {"shape": 0, "rate": 0}, 0, None),
        (dist.dirichlet, {"concentration": 1}, 1, None),
        (dist.multivariate_normal, {"location": 1, "scale_tril": 2}, 1, None),
        (dist.categorical, {"probabilities": 1}, 0, None),
        (dist.categorical_logit, {"logits": 1}, 0, None),
        (dist.multinomial, {"trials": 0, "probabilities": 1}, 1, None),
        (dist.multinomial_logit, {"trials": 0, "logits": 1}, 1, None),
        (dist.lkj_cholesky, {"concentration": 0}, 2, "dimension"),
    ],
)
def test_distribution_bindings_keep_parameter_and_value_event_axes_distinct(
    distribution, parameter_events, event_ndims, dimension_parameter
):
    binding = _get_distribution_spec(distribution)
    assert binding.density is distribution
    assert dict(binding.parameter_events) == parameter_events
    assert binding.event_ndims == event_ndims
    assert binding.dimension_parameter == dimension_parameter


def test_distribution_binding_metadata_is_immutable():
    binding = _get_distribution_spec(module_normal)
    with pytest.raises(FrozenInstanceError):
        binding.event_ndims = 1


def test_wrapper_cannot_reuse_a_copied_distribution_binding():
    @wraps(module_normal)
    def wrapper(value, location, scale):
        return module_normal(value, location, scale)

    assert wrapper._mmmjax_distribution is module_normal._mmmjax_distribution
    with pytest.raises(TypeError, match="distribution"):
        _get_distribution_spec(wrapper)
    with pytest.raises(TypeError, match="distribution"):
        Prior(wrapper, location=0.0, scale=1.0)


def test_local_distribution_binding_requires_no_central_registry():
    def local_density(value, center, spread):
        return dist.normal(value, location=center, scale=spread)

    def local_logpdf(value, center, spread):
        return dist.normal_logpdf(value, location=center, scale=spread)

    def local_rng(key, center, spread, *, sample_shape=()):
        return dist.normal_rng(key, location=center, scale=spread, sample_shape=sample_shape)

    original_signature = signature(local_density)
    with pytest.raises(TypeError, match="distribution"):
        Prior(local_density, center=0.0, spread=1.0)
    _bind_distribution(local_density, local_logpdf, local_rng, tfd.Normal, center="loc", spread="scale")
    assert signature(local_density) == original_signature
    assert dict(_get_distribution_spec(local_density).parameter_events) == {"center": 0, "spread": 0}

    prior = Prior(local_density, center=jnp.array([0.0, 2.0]), spread=0.5)
    values = jnp.array([-0.5, 2.25])
    expected = dist.normal_logpdf(values, location=jnp.array([0.0, 2.0]), scale=0.5)
    np.testing.assert_allclose(jax.jit(prior)(values), expected.sum())
    np.testing.assert_allclose(prior.logpdf(values), expected)
    assert prior.event_ndims == 0

    key = jax.random.key(72)
    samples = jax.jit(lambda key: prior._sample(key, (3, 2), jnp.float32))(key)
    expected_samples = local_rng(
        key,
        center=jnp.broadcast_to(jnp.array([0.0, 2.0], dtype=jnp.float32), (3, 2)),
        spread=jnp.full((3, 2), 0.5, dtype=jnp.float32),
    )
    np.testing.assert_allclose(samples, expected_samples, rtol=1e-6, atol=1e-7)


def test_prior_snapshots_settings_and_preserves_density_gradients():
    location = np.array([1.0, 2.0], dtype=np.float32)
    scale = [0.5, 1.5]
    prior = Prior(dist.normal, location=location, scale=scale)
    location[:] = 100
    scale[0] = 100
    values = jnp.array([1.25, 1.0])

    expected = dist.normal_logpdf(values, jnp.array([1.0, 2.0]), jnp.array([0.5, 1.5]))
    np.testing.assert_allclose(prior.logpdf(values), expected)
    np.testing.assert_allclose(jax.jit(prior)(values), expected.sum())
    np.testing.assert_allclose(jax.grad(prior)(values), -(values - jnp.array([1.0, 2.0])) / jnp.array([0.5, 1.5]) ** 2)
    with pytest.raises(FrozenInstanceError):
        prior._parameters = ()


@pytest.mark.parametrize(
    ("distribution", "parameters", "value"),
    [
        (dist.normal, {"location": 0.0, "scale": 1.0}, 0.5),
        (dist.half_normal, {"scale": 2.0}, 0.5),
        (dist.lognormal, {"location": 0.2, "scale": 0.9}, 1.2),
        (dist.uniform, {"lower": 0.0, "upper": 1.0}, 0.5),
        (dist.truncated_normal, {"location": 0.8, "scale": 0.8, "lower": 0.1, "upper": 10.0}, 1.0),
        (dist.beta, {"alpha": 2.0, "beta": 3.0}, 0.5),
        (dist.gamma, {"shape": 2.0, "rate": 3.0}, 0.5),
        (dist.inverse_gamma, {"shape": 2.0, "scale": 3.0}, 0.5),
        (dist.exponential, {"rate": 2.0}, 0.5),
        (dist.cauchy, {"location": 0.0, "scale": 1.0}, 0.5),
        (dist.laplace, {"location": 0.0, "scale": 1.0}, 0.5),
        (dist.student_t, {"degrees_of_freedom": 4.0, "location": 0.0, "scale": 1.0}, 0.5),
        (dist.bernoulli, {"probability": 0.7}, 1),
        (dist.bernoulli_logit, {"logits": 0.7}, 1),
        (dist.binomial, {"trials": 10, "probability": 0.7}, 3),
        (dist.binomial_logit, {"trials": 10, "logits": 0.7}, 3),
        (dist.poisson, {"rate": 2.0}, 3),
        (dist.poisson_log, {"log_rate": 0.5}, 3),
        (dist.negative_binomial, {"mean": 3.0, "concentration": 2.0}, 3),
        (dist.negative_binomial_log, {"log_mean": 0.5, "concentration": 2.0}, 3),
    ],
)
def test_registered_scalar_families_use_existing_density_and_rng(distribution, parameters, value):
    prior = Prior(distribution, **parameters)
    key = jax.random.key(4)
    expected = distribution(value, **parameters)
    np.testing.assert_allclose(prior(value), expected)
    np.testing.assert_allclose(prior.logpdf(value), expected)
    random_function = getattr(dist, f"{distribution.__name__}_rng")
    np.testing.assert_array_equal(
        prior.sample(key, sample_shape=(4,)), random_function(key, **parameters, sample_shape=(4,))
    )
    assert prior.event_ndims == 0


def test_declared_shape_broadcasts_settings_without_duplicating_axes_or_draws():
    prior = Prior(dist.normal, location=jnp.array([0.0, 10.0]), scale=1.0)
    samples = jax.jit(lambda key: prior._sample(key, (4, 2), jnp.float32))(jax.random.key(2))
    assert samples.shape == (4, 2)
    assert samples.dtype == jnp.float32
    assert np.unique(np.asarray(samples[:, 0])).size == 4
    assert np.all(np.asarray(samples[:, 1]) > 5)
    assert prior.sample(jax.random.key(2), sample_shape=(3,)).shape == (3, 2)


def test_declared_shape_uses_requested_precision_for_rng_parameters():
    with jax.enable_x64(True):
        prior = Prior(dist.normal, location=0.0, scale=1.0)
        samples = prior._sample(jax.random.key(6), (3,), jnp.float64)
        expected = dist.normal_rng(
            jax.random.key(6), location=jnp.zeros(3, dtype=jnp.float64), scale=jnp.ones(3, dtype=jnp.float64)
        )
        np.testing.assert_array_equal(samples, expected)
        assert samples.dtype == jnp.float64


@pytest.mark.parametrize(
    ("distribution", "parameters"),
    [
        (dist.dirichlet, {"concentration": jnp.array([2.0, 3.0, 4.0])}),
        (dist.multivariate_normal, {"location": jnp.zeros(3), "scale_tril": jnp.eye(3)}),
    ],
)
def test_vector_events_keep_batch_axes_and_full_declared_shape(distribution, parameters):
    prior = Prior(distribution, **parameters)
    samples = prior._sample(jax.random.key(8), (2, 4, 3), jnp.float32)
    assert prior.event_ndims == 1
    assert samples.shape == (2, 4, 3)
    assert prior.logpdf(samples).shape == (2, 4)
    assert prior(samples).shape == ()
    assert prior.sample(jax.random.key(8), sample_shape=(5,)).shape == (5, 3)
    if distribution is dist.dirichlet:
        np.testing.assert_allclose(samples.sum(-1), 1.0, rtol=1e-6)


def test_multivariate_normal_broadcasts_different_parameter_batches():
    prior = Prior(
        dist.multivariate_normal,
        location=jnp.zeros((4, 3)),
        scale_tril=jnp.broadcast_to(jnp.eye(3), (2, 1, 3, 3)),
    )
    samples = prior._sample(jax.random.key(10), (2, 4, 3), jnp.float32)
    assert samples.shape == (2, 4, 3)
    assert prior.logpdf(samples).shape == (2, 4)


def test_lkj_dimension_is_inferred_from_declaration_or_bound_for_direct_sampling():
    prior = Prior(dist.lkj_cholesky, concentration=jnp.array([1.0, 2.0]))
    samples = prior._sample(jax.random.key(11), (4, 2, 3, 3), jnp.float32)
    assert samples.shape == (4, 2, 3, 3)
    assert prior.logpdf(samples).shape == (4, 2)
    assert prior.event_ndims == 2
    np.testing.assert_allclose(jnp.sum(samples**2, axis=-1), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="requires dimension"):
        prior.sample(jax.random.key(11))

    bound = Prior(dist.lkj_cholesky, concentration=2.0, dimension=3)
    assert bound.sample(jax.random.key(11), sample_shape=(2,)).shape == (2, 3, 3)
    with pytest.raises(ValueError, match="event shape"):
        bound(jnp.eye(2))
    with pytest.raises(ValueError, match="event shape"):
        bound._validate_shape((2, 2))


@pytest.mark.parametrize(
    ("distribution", "parameters"),
    [(dist.categorical, {"probabilities": [0.2, 0.8]}), (dist.categorical_logit, {"logits": [-1.0, 1.0]})],
)
def test_categorical_parameter_event_axis_is_not_a_value_axis(distribution, parameters):
    prior = Prior(distribution, **parameters)
    samples = prior._sample(jax.random.key(12), (3, 4), jnp.int32)
    assert prior.event_ndims == 0
    assert samples.shape == (3, 4)
    assert prior.logpdf(samples).shape == (3, 4)
    assert np.isin(samples, [0, 1]).all()


@pytest.mark.parametrize(
    ("distribution", "parameters"),
    [(dist.multinomial, {"probabilities": [0.2, 0.8]}), (dist.multinomial_logit, {"logits": [-1.0, 1.0]})],
)
def test_multinomial_density_and_draws_share_bound_trial_counts(distribution, parameters):
    prior = Prior(distribution, **parameters, trials=jnp.array([3, 10]))
    samples = prior._sample(jax.random.key(13), (4, 2, 2), jnp.int32)
    assert samples.shape == (4, 2, 2)
    np.testing.assert_array_equal(samples.sum(-1), np.broadcast_to([3, 10], (4, 2)))
    assert prior.logpdf(samples).shape == (4, 2)
    np.testing.assert_allclose(prior(samples), distribution(samples, **parameters))
    wrong_counts = jnp.array([[1, 3], [2, 8]])
    assert np.isneginf(prior.logpdf(wrong_counts)[0])
    assert np.isfinite(prior.logpdf(wrong_counts)[1])
    assert np.isneginf(prior(wrong_counts))
    assert np.isnan(prior.logpdf(jnp.array([[jnp.nan, 3.0], [2.0, 8.0]]))[0])

    invalid_trials = Prior(distribution, **parameters, trials=-1)
    assert np.isnan(invalid_trials.logpdf(jnp.array([1, 2])))
    invalid_parameters = {name: [jnp.nan, 0.0] for name in parameters}
    invalid = Prior(distribution, **invalid_parameters, trials=3)
    assert np.isnan(invalid.logpdf(jnp.array([2, 2])))


@pytest.mark.parametrize("distribution", [lambda value: value, dist.normal_logpdf, "normal", None, []])
def test_unknown_distribution_functions_are_rejected(distribution):
    with pytest.raises(TypeError, match="exported by mmmjax or returned by custom_distribution"):
        Prior(distribution)


@pytest.mark.parametrize(
    ("distribution", "parameters", "message"),
    [
        (dist.normal, {"location": 0.0}, "Missing parameters.*scale"),
        (dist.normal, {"location": 0.0, "scale": 1.0, "sd": 1.0}, "Unknown parameters.*sd"),
        (dist.normal, {"location": 0.0, "scale": 1.0, "sample_shape": (2,)}, "Unknown parameters.*sample_shape"),
        (dist.multinomial, {"probabilities": [0.2, 0.8]}, "Missing parameters.*trials"),
        (dist.normal, {"location": "bad", "scale": 1.0}, "real numeric"),
        (dist.lkj_cholesky, {"concentration": 1.0, "dimension": True}, "positive integer"),
    ],
)
def test_invalid_parameter_names_and_types_are_rejected(distribution, parameters, message):
    with pytest.raises(TypeError, match=message):
        Prior(distribution, **parameters)


@pytest.mark.parametrize(
    ("distribution", "parameters", "shape"),
    [
        (dist.normal, {"location": [0.0, 1.0], "scale": 1.0}, ()),
        (dist.normal, {"location": [0.0, 1.0], "scale": 1.0}, (3,)),
        (dist.dirichlet, {"concentration": [1.0, 2.0]}, (3,)),
        (dist.dirichlet, {"concentration": [1.0, 2.0]}, ()),
        (dist.lkj_cholesky, {"concentration": 1.0}, (2, 3)),
        (dist.lkj_cholesky, {"concentration": 1.0}, (0, 0)),
    ],
)
def test_incompatible_declared_shapes_are_rejected(distribution, parameters, shape):
    prior = Prior(distribution, **parameters)
    with pytest.raises(ValueError, match=r"shape|event"):
        prior._validate_shape(shape)


@pytest.mark.parametrize(
    ("distribution", "parameters"),
    [
        (dist.normal, {"location": jnp.zeros(2), "scale": jnp.ones(3)}),
        (dist.dirichlet, {"concentration": 1.0}),
        (dist.categorical, {"probabilities": []}),
        (dist.multivariate_normal, {"location": jnp.zeros(2), "scale_tril": jnp.eye(3)}),
        (dist.multivariate_normal, {"location": jnp.zeros(2), "scale_tril": jnp.ones((2, 3))}),
    ],
)
def test_invalid_bound_batch_or_event_shapes_are_rejected(distribution, parameters):
    with pytest.raises(ValueError, match=r"batch shape|event|square matrix"):
        Prior(distribution, **parameters)


def half_cauchy_logpdf(value, scale):
    standardized = value / scale
    density = jnp.log(2.0 / jnp.pi) - jnp.log(scale) - jnp.log1p(standardized**2)
    return jnp.where(value < 0, -jnp.inf, density)


def half_cauchy_rng(key, scale, *, sample_shape=()):
    shape = sample_shape + jnp.shape(jnp.asarray(scale))
    return jnp.abs(scale * jax.random.cauchy(key, shape))


def rng_without_sample_shape(key, scale):
    return jnp.abs(scale * jax.random.cauchy(key))


def rng_with_another_setting(key, width, *, sample_shape=()):
    return jnp.abs(width * jax.random.cauchy(key, sample_shape))


def rng_with_missing_default(key, scale, *, sample_shape=None):
    shape = () if sample_shape is None else sample_shape
    return jnp.abs(scale * jax.random.cauchy(key, shape))


def test_custom_distribution_returns_a_summed_density_carrying_prior_metadata():
    half_cauchy = custom_distribution(half_cauchy_logpdf, half_cauchy_rng)
    values = jnp.array([0.5, 2.0, 4.0])

    assert half_cauchy.__name__ == "half_cauchy"
    assert tuple(signature(half_cauchy).parameters) == ("value", "scale")
    np.testing.assert_allclose(half_cauchy(values, 2.0), half_cauchy_logpdf(values, 2.0).sum(), rtol=1e-6)
    np.testing.assert_allclose(half_cauchy(values, scale=2.0), half_cauchy(values, 2.0))
    np.testing.assert_allclose(half_cauchy(value=values, scale=2.0), half_cauchy(values, 2.0))
    gradient = jax.jit(jax.grad(half_cauchy))(values, 2.0)
    expected = jax.grad(lambda value: half_cauchy_logpdf(value, 2.0).sum())(values)
    np.testing.assert_allclose(gradient, expected, rtol=1e-6)
    binding = _get_distribution_spec(half_cauchy)
    assert binding.logpdf is half_cauchy_logpdf
    assert binding.rng is half_cauchy_rng
    assert binding.parameter_events == (("scale", 0),)
    assert binding.event_ndims == 0
    named = custom_distribution(half_cauchy_logpdf, half_cauchy_rng, name="folded_cauchy")
    assert named.__name__ == "folded_cauchy"


def test_prior_binds_user_defined_distribution_settings_for_density_and_draws():
    half_cauchy = custom_distribution(half_cauchy_logpdf, half_cauchy_rng)
    prior = Prior(half_cauchy, scale=2.0)
    values = jnp.array([0.5, 2.0, 4.0])
    key = jax.random.key(3)

    assert prior.event_ndims == 0
    np.testing.assert_allclose(prior(values), half_cauchy_logpdf(values, 2.0).sum(), rtol=1e-6)
    np.testing.assert_allclose(prior.logpdf(values), half_cauchy_logpdf(values, 2.0), rtol=1e-6)
    draws = prior.sample(key, sample_shape=(4,))
    assert draws.shape == (4,)
    np.testing.assert_array_equal(draws, half_cauchy_rng(key, 2.0, sample_shape=(4,)))
    declared = prior._sample(key, (3,), jnp.float32)
    assert declared.shape == (3,)
    assert declared.dtype == jnp.float32
    np.testing.assert_allclose(declared, half_cauchy_rng(key, jnp.full((3,), 2.0, dtype=jnp.float32)), rtol=1e-6)


def test_sample_prior_draws_user_defined_priors_in_declared_shapes():
    half_cauchy = custom_distribution(half_cauchy_logpdf, half_cauchy_rng)
    model = Model(parameters={"scale": Positive((2,))}, log_density=lambda data, scale: half_cauchy(scale, 1.0))

    results = sample_prior(model, {"scale": Prior(half_cauchy, scale=1.0)}, draws=6, seed=4, generate=False)

    draws = results["prior"]["scale"].values
    assert draws.shape == (1, 6, 2)
    assert np.isfinite(draws).all()
    assert (draws > 0).all()


def test_custom_distribution_supports_vector_events_with_vector_settings():
    def isotropic_logpdf(value, location, scale):
        standardized = (value - location) / jnp.asarray(scale)[..., None]
        return jnp.sum(
            -0.5 * standardized**2 - jnp.log(jnp.asarray(scale))[..., None] - 0.5 * jnp.log(2 * jnp.pi), axis=-1
        )

    def isotropic_rng(key, location, scale, *, sample_shape=()):
        shape = sample_shape + jnp.shape(jnp.asarray(location))
        return location + jnp.asarray(scale)[..., None] * jax.random.normal(key, shape)

    isotropic = custom_distribution(
        isotropic_logpdf, isotropic_rng, event_ndims=1, parameter_event_ndims={"location": 1}
    )
    prior = Prior(isotropic, location=jnp.zeros(3), scale=2.0)
    values = jnp.arange(6.0).reshape(2, 3)

    assert prior.event_ndims == 1
    assert prior.logpdf(values).shape == (2,)
    np.testing.assert_allclose(prior.logpdf(values), isotropic_logpdf(values, jnp.zeros(3), 2.0), rtol=1e-6)
    np.testing.assert_allclose(prior(values), isotropic_logpdf(values, jnp.zeros(3), 2.0).sum(), rtol=1e-6)
    assert prior.sample(jax.random.key(0), sample_shape=(4,)).shape == (4, 3)
    assert prior._sample(jax.random.key(0), (5, 3), jnp.float32).shape == (5, 3)
    with pytest.raises(ValueError, match="event shape"):
        prior._sample(jax.random.key(0), (5, 2), jnp.float32)


@pytest.mark.parametrize(
    ("arguments", "options", "error", "message"),
    [
        ((half_cauchy_logpdf, rng_without_sample_shape), {}, TypeError, "sample_shape"),
        ((half_cauchy_logpdf, rng_with_missing_default), {}, TypeError, "default of"),
        (("density", half_cauchy_rng), {}, TypeError, "callable"),
        ((half_cauchy_logpdf, rng_with_another_setting), {}, TypeError, "same names"),
        ((half_cauchy_logpdf, "draws"), {}, TypeError, "callable"),
        ((half_cauchy_logpdf, half_cauchy_rng), {"name": 3}, TypeError, "name must be a string"),
        ((half_cauchy_logpdf, half_cauchy_rng), {"event_ndims": 2}, ValueError, "event_ndims must be 0"),
        ((half_cauchy_logpdf, half_cauchy_rng), {"parameter_event_ndims": {"width": 1}}, ValueError, "does not take"),
        ((half_cauchy_logpdf, half_cauchy_rng), {"parameter_event_ndims": {"scale": -1}}, ValueError, "nonnegative"),
        ((half_cauchy_logpdf, half_cauchy_rng), {"event_ndims": 1}, ValueError, "vector distribution needs"),
    ],
)
def test_custom_distribution_rejects_inconsistent_functions_and_metadata(arguments, options, error, message):
    with pytest.raises(error, match=message):
        custom_distribution(*arguments, **options)


def test_prior_reports_user_defined_setting_names_and_unregistered_functions():
    half_cauchy = custom_distribution(half_cauchy_logpdf, half_cauchy_rng)
    with pytest.raises(TypeError, match=r"Missing parameters for half_cauchy.*scale"):
        Prior(half_cauchy)
    with pytest.raises(TypeError, match=r"Unknown parameters for half_cauchy.*width"):
        Prior(half_cauchy, scale=1.0, width=2.0)
    with pytest.raises(TypeError, match="returned by custom_distribution"):
        Prior(half_cauchy_logpdf, scale=1.0)


def shifted_lognormal_logpdf(value, shift, location, scale):
    excess = value - shift
    safe = jnp.where(excess > 0, excess, 1.0)
    standardized = (jnp.log(safe) - location) / scale
    density = -jnp.log(safe) - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * standardized**2
    return jnp.where(excess > 0, density, -jnp.inf)


def shifted_lognormal_rng(key, shift, location, scale, *, sample_shape=()):
    batch_shape = jnp.broadcast_shapes(jnp.shape(shift), jnp.shape(location), jnp.shape(scale))
    draws = jax.random.normal(key, sample_shape + batch_shape)
    return shift + jnp.exp(location + scale * draws)


def mixture_logpdf(value, weight, location1, location2, scale):
    def component(location):
        standardized = (value - location) / scale
        return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    first = jnp.log(weight) + component(location1)
    second = jnp.log1p(-weight) + component(location2)
    return jnp.logaddexp(first, second)


def mixture_rng(key, weight, location1, location2, scale, *, sample_shape=()):
    batch_shape = jnp.broadcast_shapes(jnp.shape(weight), jnp.shape(location1), jnp.shape(location2), jnp.shape(scale))
    shape = sample_shape + batch_shape
    choose_key, noise_key = jax.random.split(key)
    first = jax.random.bernoulli(choose_key, weight, shape)
    location = jnp.where(first, location1, location2)
    return location + scale * jax.random.normal(noise_key, shape)


def logistic_logpdf(value, location, scale):
    standardized = (value - location) / scale
    return -standardized - jnp.log(scale) - 2.0 * jax.nn.softplus(-standardized)


def logistic_rng(key, location, scale, *, sample_shape=()):
    batch_shape = jnp.broadcast_shapes(jnp.shape(location), jnp.shape(scale))
    uniform = jax.random.uniform(key, sample_shape + batch_shape, minval=1e-6, maxval=1.0 - 1e-6)
    return location + scale * (jnp.log(uniform) - jnp.log1p(-uniform))


def test_custom_distribution_settings_with_mixed_batch_shapes_draw_per_channel_priors():
    frame = pl.DataFrame(
        {"week": [1, 2, 3], "sales": [1.0, 2.0, 3.0], "video": [4.0, 1.0, 3.0], "search": [2.0, 2.0, 1.0]}
    )
    data = prepare_data(frame, time="week", outcome="sales", media=["video", "search"])
    shifted_lognormal = custom_distribution(shifted_lognormal_logpdf, shifted_lognormal_rng)
    location = jnp.array([0.0, 0.7])
    scale = jnp.array([0.3, 0.5])
    prior = Prior(shifted_lognormal, shift=0.1, location=location, scale=scale)
    model = Model(
        parameters={"roi": Positive(dims="channel")},
        log_density=lambda media, roi: shifted_lognormal(roi, 0.1, location, scale),
        data=Data(data),
    )

    results = sample_prior(model, {"roi": prior}, draws=400, seed=11, generate=False)
    draws = results["prior"]["roi"]

    assert draws.dims == ("chain", "draw", "channel")
    assert list(draws["channel"].values) == ["video", "search"]
    values = draws.values
    assert (values > 0.1).all()
    logs = np.log(values - 0.1).reshape(-1, 2)
    np.testing.assert_allclose(logs.mean(axis=0), location, atol=0.1)
    np.testing.assert_allclose(logs.std(axis=0), scale, atol=0.1)
    np.testing.assert_allclose(
        prior.logpdf(values[0, :5]), shifted_lognormal_logpdf(values[0, :5], 0.1, location, scale), rtol=1e-6
    )
    assert prior.sample(jax.random.key(0), sample_shape=(4,)).shape == (4, 2)
    assert prior._sample(jax.random.key(0), (2,), jnp.float32).shape == (2,)


def test_custom_mixture_density_matches_component_sum_and_draw_frequencies():
    mixture = custom_distribution(mixture_logpdf, mixture_rng, name="normal_mixture")
    prior = Prior(mixture, weight=0.8, location1=-3.0, location2=3.0, scale=0.5)
    values = jnp.linspace(-5.0, 5.0, 11)

    first = 0.8 * jnp.exp(dist.normal_logpdf(values, -3.0, 0.5))
    second = 0.2 * jnp.exp(dist.normal_logpdf(values, 3.0, 0.5))
    expected = jnp.log(first + second)
    np.testing.assert_allclose(prior.logpdf(values), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(prior(values), expected.sum(), rtol=1e-5)
    gradient = jax.jit(jax.grad(mixture))(values, 0.8, -3.0, 3.0, 0.5)
    assert np.isfinite(gradient).all()
    draws = prior.sample(jax.random.key(1), sample_shape=(4000,))
    assert draws.shape == (4000,)
    np.testing.assert_allclose(np.mean(draws < 0), 0.8, atol=0.03)


def test_custom_distribution_serves_as_a_likelihood_with_parameter_settings():
    frame = pl.DataFrame({"week": [1, 2, 3, 4], "sales": [1.0, 2.5, 2.0, 3.5], "video": [1.0, 2.0, 1.5, 3.0]})
    data = prepare_data(frame, time="week", outcome="sales", media=["video"])
    logistic = custom_distribution(logistic_logpdf, logistic_rng)

    def transformed_parameters(media, coefficient):
        return {"mu": coefficient * media[:, 0]}

    def log_density(outcome, mu, coefficient, scale):
        target = dist.normal(coefficient, 0.0, 1.0) + dist.half_normal(scale, 1.0)
        target += logistic(outcome, mu, scale)
        return target

    def generated_quantities(key, outcome, mu, scale):
        return {
            "predictive": {"replicated": logistic_rng(key, mu, scale)},
            "log_likelihood": {"pointwise": logistic_logpdf(outcome, mu, scale)},
        }

    model = Model(
        parameters={"coefficient": Real(), "scale": Positive()},
        log_density=log_density,
        data=Data(data),
        transformed_parameters=transformed_parameters,
        generated_quantities=generated_quantities,
    )
    position = model.initialize_random(jax.random.key(2))
    parameters = model.constrain(position)

    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    assert np.isfinite(value)
    assert all(np.isfinite(entry).all() for entry in gradient.values())
    mu = parameters["coefficient"] * data.arrays["media"][:, 0]
    expected = (
        dist.normal(parameters["coefficient"], 0.0, 1.0)
        + dist.half_normal(parameters["scale"], 1.0)
        + logistic_logpdf(data.arrays["outcome"], mu, parameters["scale"]).sum()
    )
    np.testing.assert_allclose(model.log_prob(parameters), expected, rtol=1e-5)
    outputs = jax.jit(model.generate_quantities)(jax.random.key(3), parameters, model.data)
    assert outputs["predictive"]["replicated"].shape == (4,)
    np.testing.assert_allclose(
        outputs["log_likelihood"]["pointwise"],
        logistic_logpdf(data.arrays["outcome"], mu, parameters["scale"]),
        rtol=1e-6,
    )


def test_custom_distribution_draws_honor_the_requested_precision():
    half_cauchy = custom_distribution(half_cauchy_logpdf, half_cauchy_rng)
    with jax.enable_x64(True):
        # Settings are copied at construction, so the prior is built where they can be float64.
        prior = Prior(half_cauchy, scale=1.5)
        wide = prior._sample(jax.random.key(6), (3,), jnp.float64)
        assert wide.dtype == jnp.float64
        np.testing.assert_allclose(prior.logpdf(wide), half_cauchy_logpdf(wide, 1.5), rtol=1e-12)
    narrow = prior._sample(jax.random.key(6), (3,), jnp.float32)
    assert narrow.dtype == jnp.float32
    assert (narrow > 0).all()
