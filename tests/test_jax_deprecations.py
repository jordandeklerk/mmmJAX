"""Tests for deprecated JAX API use in public functions."""

import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import (
    Model,
    Positive,
    Prior,
    bernoulli_logit_logpmf,
    bernoulli_logit_rng,
    bernoulli_logpmf,
    bernoulli_rng,
    beta_logpdf,
    beta_rng,
    binomial_logit_logpmf,
    binomial_logit_rng,
    binomial_logpmf,
    binomial_rng,
    categorical_logit_logpmf,
    categorical_logit_rng,
    categorical_logpmf,
    categorical_rng,
    cauchy_logpdf,
    cauchy_rng,
    contribution_coefficient,
    delayed_adstock,
    dirichlet_logpdf,
    dirichlet_rng,
    exponential_logpdf,
    exponential_rng,
    fit_media_scaling,
    fit_scaling,
    fourier_features,
    gamma_logpdf,
    gamma_rng,
    generate_quantities,
    geometric_adstock,
    half_normal,
    half_normal_logpdf,
    half_normal_rng,
    hill_saturation,
    hsgp_basis,
    hsgp_weights,
    inverse_gamma_logpdf,
    inverse_gamma_rng,
    laplace_logpdf,
    laplace_rng,
    lkj_cholesky,
    lkj_cholesky_logpdf,
    lkj_cholesky_rng,
    log_saturation,
    logistic_saturation,
    lognormal_logpdf,
    lognormal_rng,
    media_response,
    multinomial_logit_logpmf,
    multinomial_logit_rng,
    multinomial_logpmf,
    multinomial_rng,
    multivariate_normal_logpdf,
    multivariate_normal_rng,
    negative_binomial_log_logpmf,
    negative_binomial_log_rng,
    negative_binomial_logpmf,
    negative_binomial_rng,
    normal,
    normal_logpdf,
    normal_rng,
    poisson_log_logpmf,
    poisson_log_rng,
    poisson_logpmf,
    poisson_rng,
    prepare_data,
    prepare_hsgp,
    reach_frequency_response,
    roi_coefficient,
    root_saturation,
    student_t_logpdf,
    student_t_rng,
    truncated_normal_logpdf,
    truncated_normal_rng,
    uniform_logpdf,
    uniform_rng,
    weibull_cdf_adstock,
    weibull_pdf_adstock,
)
from mmmjax._results import _collect_results

# Later marks take precedence. The pinned tfp-nightly still calls deprecated JAX
# APIs, so only those two messages and only from its own modules are ignored.
pytestmark = [
    pytest.mark.filterwarnings("error::DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:jax.core.pytype_aval_mappings is deprecated:DeprecationWarning:tensorflow_probability"
    ),
    pytest.mark.filterwarnings(
        "ignore:shape requires ndarray or scalar arguments:DeprecationWarning:tensorflow_probability"
    ),
]


@pytest.fixture(autouse=True)
def _reset_deprecated_attribute_warnings():
    # JAX caches each deprecated module attribute after its first warning, so an
    # access earlier in the process would otherwise pass here without a warning
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] != "jax":
            continue
        clear = getattr(getattr(module, "__getattr__", None), "cache_clear", None)
        if clear is not None:
            clear()


# jnp.shape and jnp.ndim warn on Python sequences but accept NumPy arrays silently,
# so every array argument below is a list to catch mmmjax calling them on user input
@pytest.mark.parametrize(
    ("density", "arguments"),
    [
        (bernoulli_logpmf, ([0, 1], [0.4, 0.6])),
        (bernoulli_logit_logpmf, ([0, 1], [0.2, -0.2])),
        (beta_logpdf, ([0.25, 0.75], [2.0, 2.5], [3.0, 3.5])),
        (binomial_logpmf, ([1, 2], [5, 6], [0.4, 0.5])),
        (binomial_logit_logpmf, ([1, 2], [5, 6], [0.2, -0.2])),
        (categorical_logpmf, ([0, 1], [0.4, 0.6])),
        (categorical_logit_logpmf, ([0, 1], [0.2, -0.2])),
        (cauchy_logpdf, ([-0.5, 0.5], [0.0, 0.1], [1.0, 2.0])),
        (dirichlet_logpdf, ([0.4, 0.6], [2.0, 3.0])),
        (exponential_logpdf, ([0.5, 1.5], [2.0, 3.0])),
        (gamma_logpdf, ([0.5, 1.5], [2.0, 2.5], [3.0, 3.5])),
        (half_normal_logpdf, ([0.5, 1.5], [2.0, 2.5])),
        (inverse_gamma_logpdf, ([0.5, 1.5], [2.0, 2.5], [3.0, 3.5])),
        (laplace_logpdf, ([-0.5, 0.5], [0.0, 0.1], [1.0, 2.0])),
        (lkj_cholesky_logpdf, ([[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0])),
        (lognormal_logpdf, ([0.5, 1.5], [0.0, 0.1], [1.0, 2.0])),
        (multinomial_logpmf, ([2, 3], [0.4, 0.6])),
        (multinomial_logit_logpmf, ([2, 3], [0.2, -0.2])),
        (multivariate_normal_logpdf, ([-0.5, 0.5], [0.0, 0.1], [[1.0, 0.0], [0.5, 1.0]])),
        (negative_binomial_logpmf, ([1, 2], [2.0, 2.5], [1.5, 2.0])),
        (negative_binomial_log_logpmf, ([1, 2], [0.5, 0.6], [1.5, 2.0])),
        (normal_logpdf, ([-0.5, 0.5], [0.0, 0.1], [1.0, 2.0])),
        (poisson_logpmf, ([1, 2], [2.0, 2.5])),
        (poisson_log_logpmf, ([1, 2], [0.5, 0.6])),
        (student_t_logpdf, ([-0.5, 0.5], [5.0, 6.0], [0.0, 0.1], [1.0, 2.0])),
        (truncated_normal_logpdf, ([-0.5, 0.5], [0.0, 0.1], [1.0, 2.0], [-1.0, -2.0], [1.0, 2.0])),
        (uniform_logpdf, ([0.25, 0.75], [0.0, 0.1], [1.0, 2.0])),
    ],
)
def test_distribution_densities_avoid_deprecated_jax_apis(density, arguments):
    result = density(*arguments)

    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    ("rng", "arguments"),
    [
        (bernoulli_rng, ([0.4, 0.6],)),
        (bernoulli_logit_rng, ([0.2, -0.2],)),
        (beta_rng, ([2.0, 2.5], [3.0, 3.5])),
        (binomial_rng, ([5, 6], [0.4, 0.5])),
        (binomial_logit_rng, ([5, 6], [0.2, -0.2])),
        (categorical_rng, ([0.4, 0.6],)),
        (categorical_logit_rng, ([0.2, -0.2],)),
        (cauchy_rng, ([0.0, 0.1], [1.0, 2.0])),
        (dirichlet_rng, ([2.0, 3.0],)),
        (exponential_rng, ([2.0, 3.0],)),
        (gamma_rng, ([2.0, 2.5], [3.0, 3.5])),
        (half_normal_rng, ([2.0, 2.5],)),
        (inverse_gamma_rng, ([2.0, 2.5], [3.0, 3.5])),
        (laplace_rng, ([0.0, 0.1], [1.0, 2.0])),
        (lkj_cholesky_rng, (2, [2.0, 3.0])),
        (lognormal_rng, ([0.0, 0.1], [1.0, 2.0])),
        (multinomial_rng, ([0.4, 0.6], [5, 6])),
        (multinomial_logit_rng, ([0.2, -0.2], [5, 6])),
        (multivariate_normal_rng, ([0.0, 0.1], [[1.0, 0.0], [0.5, 1.0]])),
        (negative_binomial_rng, ([2.0, 2.5], [1.5, 2.0])),
        (negative_binomial_log_rng, ([0.5, 0.6], [1.5, 2.0])),
        (normal_rng, ([0.0, 0.1], [1.0, 2.0])),
        (poisson_rng, ([2.0, 2.5],)),
        (poisson_log_rng, ([0.5, 0.6],)),
        (student_t_rng, ([5.0, 6.0], [0.0, 0.1], [1.0, 2.0])),
        (truncated_normal_rng, ([0.0, 0.1], [1.0, 2.0], [-1.0, -2.0], [1.0, 2.0])),
        (uniform_rng, ([0.0, 0.1], [1.0, 2.0])),
    ],
)
def test_distribution_rngs_avoid_deprecated_jax_apis(rng, arguments):
    key = jax.random.key(0)

    result = rng(key, *arguments, sample_shape=(3,))

    assert result.shape[0] == 3
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(partial(geometric_adstock, alpha=[0.5, 0.6], max_lag=2), id="geometric_adstock"),
        pytest.param(partial(delayed_adstock, alpha=[0.5, 0.6], theta=[1.0, 0.0], max_lag=2), id="delayed_adstock"),
        pytest.param(
            partial(weibull_cdf_adstock, shape=[1.5, 2.0], scale=[1.0, 2.0], max_lag=2), id="weibull_cdf_adstock"
        ),
        pytest.param(
            partial(weibull_pdf_adstock, shape=[1.5, 2.0], scale=[1.0, 2.0], max_lag=2), id="weibull_pdf_adstock"
        ),
        pytest.param(partial(hill_saturation, half_saturation=[1.0, 2.0], slope=[2.0, 1.5]), id="hill_saturation"),
        pytest.param(partial(logistic_saturation, half_saturation=[1.0, 2.0]), id="logistic_saturation"),
        pytest.param(partial(root_saturation, exponent=[0.5, 0.7]), id="root_saturation"),
        pytest.param(log_saturation, id="log_saturation"),
        pytest.param(
            partial(
                media_response,
                adstock=partial(geometric_adstock, alpha=[0.5, 0.6], max_lag=2),
                saturation=partial(hill_saturation, half_saturation=[1.0, 2.0], slope=[2.0, 1.5]),
                n_periods=3,
            ),
            id="media_response",
        ),
        pytest.param(
            partial(
                reach_frequency_response,
                frequency=[[1.5, 1.0], [2.0, 1.5], [1.0, 3.0], [1.5, 2.0]],
                adstock=partial(geometric_adstock, alpha=[0.5, 0.6], max_lag=2),
                saturation=partial(hill_saturation, half_saturation=[1.0, 2.0], slope=[2.0, 1.5]),
            ),
            id="reach_frequency_response",
        ),
    ],
)
def test_media_transforms_avoid_deprecated_jax_apis(transform):
    media = [[1.0, 0.0], [2.0, 1.0], [0.0, 3.0], [1.0, 1.0]]

    result = transform(media)

    assert jnp.all(jnp.isfinite(result))


def test_fourier_features_avoid_deprecated_jax_apis():
    time = [0.0, 1.0, 2.0, 3.0]

    result = fourier_features(time, period=52.0, order=2)

    assert result.shape == (4, 4)
    assert jnp.all(jnp.isfinite(result))


def test_hsgp_helpers_avoid_deprecated_jax_apis():
    time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    approximation = prepare_hsgp([0.0, 5.0], length_scale_range=[1.0, 4.0])
    basis, frequencies = hsgp_basis(
        time,
        center=approximation.center,
        boundary=approximation.boundary,
        n_basis=approximation.n_basis,
    )
    weights = hsgp_weights(frequencies, length_scale=[1.0, 2.0], amplitude=[0.5, 1.0])

    assert basis.shape == (6, approximation.n_basis)
    assert weights.shape == (2, approximation.n_basis)
    assert jnp.all(jnp.isfinite(basis))
    assert jnp.all(jnp.isfinite(weights))


def test_calibration_coefficients_avoid_deprecated_jax_apis():
    response = [[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]]
    spend = [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
    expected_contribution = [2.0 / 3.0, 0.75]
    expected_roi = [2.0, 2.0]

    contribution = contribution_coefficient([2.0, 3.0], response)
    roi = roi_coefficient([1.5, 2.0], response, spend)

    np.testing.assert_allclose(contribution, expected_contribution, rtol=2e-6, atol=0)
    np.testing.assert_allclose(roi, expected_roi, rtol=2e-6, atol=0)


def test_scaling_avoids_deprecated_jax_apis():
    values = [[1.0, 2.0], [3.0, 6.0]]

    scaling = fit_scaling(values)
    media_scaling = fit_media_scaling(values)
    restored = scaling.inverse_transform(scaling.transform(values))
    media_restored = media_scaling.inverse_transform(media_scaling.transform(values))

    np.testing.assert_allclose(restored, values, rtol=2e-6, atol=0)
    np.testing.assert_allclose(media_restored, values, rtol=2e-6, atol=0)


def test_prior_list_inputs_avoid_deprecated_jax_apis():
    factor = [[1.0, 0.0], [0.0, 1.0]]
    location_prior = Prior(normal, location=[0.0, 1.0], scale=[1.0, 2.0])
    correlation_prior = Prior(lkj_cholesky, concentration=2.0, dimension=2)
    # Two by two LKJ(2) correlations follow p(r) = 3 (1 - r**2) / 4
    expected = np.log(0.75)

    density = location_prior([0.5, 0.5])
    draws = location_prior.sample(jax.random.key(0), sample_shape=(3,))
    eager = correlation_prior.logpdf(factor)
    compiled = jax.jit(lambda first, second: correlation_prior.logpdf([first, second]))(*jnp.eye(2))

    assert jnp.isfinite(density)
    assert draws.shape == (3, 2)
    for result in (eager, compiled):
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


def test_model_workflow_avoids_deprecated_jax_apis():
    frame = pl.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "sales": [2.0, 3.0, 2.5, 4.0],
            "search": [1.0, 0.0, 2.0, 1.0],
            "video": [0.0, 1.0, 1.0, 2.0],
        }
    )
    coefficient_prior = Prior(half_normal, scale=[1.0, 1.0])
    noise_prior = Prior(half_normal, scale=1.0)
    position = {"coefficient": jnp.zeros(2), "noise": jnp.zeros(())}
    draws = _collect_results(
        {"coefficient": np.ones((1, 2, 2), dtype=np.float32), "noise": np.ones((1, 2), dtype=np.float32)},
        dims={"coefficient": ("channel",)},
        coords={"channel": ["search", "video"]},
    )

    def expected_outcome(media, coefficient):
        response = hill_saturation(geometric_adstock(media, 0.5, max_lag=1), 1.0, 2.0)
        mean = response @ coefficient
        return mean

    def log_density(outcome, media, coefficient, noise):
        mean = expected_outcome(media, coefficient)
        density = coefficient_prior(coefficient) + noise_prior(noise) + normal(outcome, mean, noise)
        return density

    def generate(key, outcome, media, coefficient, noise):
        mean = expected_outcome(media, coefficient)
        quantities = {
            "predictive": {"outcome": normal_rng(key, mean, noise)},
            "log_likelihood": {"outcome": normal_logpdf(outcome, mean, noise)},
        }
        return quantities

    data = prepare_data(frame, time="week", outcome="sales", media=["search", "video"])
    model = Model(
        data,
        parameters={"coefficient": Positive((2,)), "noise": Positive()},
        log_density=log_density,
        generated_quantities=generate,
        dims={"coefficient": "channel"},
    )
    density = model.log_density(position, model.data)
    compiled = jax.jit(model.log_density)(position, model.data)
    gradient = jax.grad(model.log_density)(position, model.data)
    quantities = model.generate_quantities(jax.random.key(0), model.constrain(position), model.data)
    evaluated = generate_quantities(model, draws)

    assert jnp.isfinite(density)
    np.testing.assert_allclose(compiled, density, rtol=2e-6, atol=0)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient))
    assert quantities["predictive"]["outcome"].shape == (4,)
    assert quantities["log_likelihood"]["outcome"].shape == (4,)
    assert evaluated["posterior_predictive"]["outcome"].shape == (1, 2, 4)
