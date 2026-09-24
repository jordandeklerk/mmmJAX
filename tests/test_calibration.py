"""Tests for coefficients implied by return and contribution priors."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from jax.scipy.special import logsumexp

from mmmjax import (
    Data,
    Model,
    Positive,
    Real,
    contribution_coefficient,
    fit_data_scaling,
    geometric_adstock,
    media_metrics,
    prepare_data,
    roi_coefficient,
)
from mmmjax._results import _collect_results


def _grouped_case():
    keys = jax.random.split(jax.random.key(0), 3)
    response = jax.random.uniform(keys[0], (5, 3, 4), minval=0.1, maxval=2.0)
    spend = jax.random.uniform(keys[1], (5, 3, 4), minval=1.0, maxval=5.0)
    deviations = 0.3 * jax.random.normal(keys[2], (3, 4))
    roi = jnp.array([0.5, 1.5, 2.0, 3.0])
    scale = jnp.array([1.5, 0.75, 2.0])
    return response, spend, roi, scale, deviations


def _weighted_total(response, scale, coefficient):
    if coefficient.ndim == 2:
        return jnp.einsum("tgm,g,gm->m", response, scale, coefficient)
    return jnp.einsum("tgm,g,m->m", response, scale, coefficient)


def test_roi_coefficient_national_matches_closed_form_and_identity():
    response = jnp.array([[0.5, 1.0], [0.6, 0.9], [0.4, 1.1]])
    spend = jnp.array([[100.0, 50.0], [120.0, 40.0], [80.0, 60.0]])
    roi = jnp.array([2.0, 0.5])

    coefficient = roi_coefficient(roi, response, spend, outcome_scale=40.0)

    assert coefficient.shape == (2,)
    expected = roi * spend.sum(axis=0) / (40.0 * response.sum(axis=0))
    np.testing.assert_allclose(coefficient, expected, rtol=1e-6)
    np.testing.assert_allclose(40.0 * response.sum(axis=0) * coefficient, roi * spend.sum(axis=0), rtol=1e-6)
    np.testing.assert_allclose(roi_coefficient(1.0, response, spend), spend.sum(axis=0) / response.sum(axis=0))


def test_roi_coefficient_grouped_without_deviations_shares_one_coefficient():
    response, spend, roi, scale, _ = _grouped_case()

    coefficient = roi_coefficient(roi, response, spend, outcome_scale=scale)

    assert coefficient.shape == (4,)
    np.testing.assert_allclose(_weighted_total(response, scale, coefficient), roi * spend.sum(axis=(0, 1)), rtol=1e-5)


@pytest.mark.parametrize("effects", ["lognormal", "normal"])
def test_roi_coefficient_deviations_vary_by_group_around_a_shared_center(effects):
    response, spend, roi, scale, deviations = _grouped_case()

    coefficient = roi_coefficient(roi, response, spend, outcome_scale=scale, deviations=deviations, effects=effects)

    assert coefficient.shape == (3, 4)
    np.testing.assert_allclose(_weighted_total(response, scale, coefficient), roi * spend.sum(axis=(0, 1)), rtol=1e-5)
    center = jnp.log(coefficient) - deviations if effects == "lognormal" else coefficient - deviations
    np.testing.assert_allclose(center, jnp.broadcast_to(center[0], center.shape), rtol=1e-5, atol=1e-6)
    if effects == "lognormal":
        assert (coefficient > 0).all()


def test_roi_coefficient_reproduces_the_meridian_log_center_formula():
    response, spend, roi, scale, deviations = _grouped_case()

    coefficient = roi_coefficient(roi, response, spend, outcome_scale=scale, deviations=deviations)

    # The same identity written out as Meridian's model does, without logsumexp.
    a_gm = jnp.sum(response, axis=0) * scale[:, None]
    beta_m = jnp.log(roi * spend.sum(axis=(0, 1))) - jnp.log(jnp.sum(a_gm * jnp.exp(deviations), axis=0))
    np.testing.assert_allclose(coefficient, jnp.exp(beta_m + deviations), rtol=1e-5)


def test_roi_coefficient_accepts_totaled_spend_and_marginal_increments():
    response, spend, roi, scale, deviations = _grouped_case()

    full = roi_coefficient(roi, response, spend, outcome_scale=scale, deviations=deviations)
    totaled = roi_coefficient(roi, response, spend.sum(axis=(0, 1)), outcome_scale=scale, deviations=deviations)
    np.testing.assert_allclose(totaled, full, rtol=1e-6)

    # A marginal return uses the response difference from a one percent exposure increase.
    increment = response * 1.01 - response
    marginal = roi_coefficient(roi, increment, 0.01 * spend, outcome_scale=scale, deviations=deviations)
    np.testing.assert_allclose(
        _weighted_total(increment, scale, marginal), roi * 0.01 * spend.sum(axis=(0, 1)), rtol=1e-4
    )


def test_contribution_coefficient_expresses_contribution_shares():
    response, _, _, scale, deviations = _grouped_case()
    total_outcome = 5000.0
    shares = jnp.array([0.05, 0.1, 0.02, 0.08])

    coefficient = contribution_coefficient(shares * total_outcome, response, outcome_scale=scale, deviations=deviations)

    np.testing.assert_allclose(_weighted_total(response, scale, coefficient), shares * total_outcome, rtol=1e-5)
    shared = contribution_coefficient(shares * total_outcome, response, outcome_scale=scale)
    np.testing.assert_allclose(_weighted_total(response, scale, shared), shares * total_outcome, rtol=1e-5)


def test_coefficients_compile_and_differentiate():
    response, spend, roi, scale, deviations = _grouped_case()

    def coefficients(returns):
        return roi_coefficient(returns, response, spend, outcome_scale=scale, deviations=deviations)

    compiled = jax.jit(coefficients)(roi)
    np.testing.assert_allclose(compiled, coefficients(roi), rtol=1e-6)
    # Log-normal coefficients scale linearly with the return, so the derivative is the ratio.
    gradient = jax.jacobian(coefficients)(roi)
    expected = jnp.einsum("gm,mn->gmn", compiled / roi, jnp.eye(4))
    np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=1e-6)


def test_coefficients_hold_the_identity_tightly_in_float64():
    with jax.enable_x64(True):
        response, spend, roi, scale, deviations = _grouped_case()
        for effects in ("lognormal", "normal"):
            coefficient = roi_coefficient(
                roi, response, spend, outcome_scale=scale, deviations=deviations, effects=effects
            )
            assert coefficient.dtype == jnp.float64
            np.testing.assert_allclose(
                _weighted_total(response, scale, coefficient), roi * spend.sum(axis=(0, 1)), rtol=1e-12
            )


def test_zero_exposure_leaves_no_finite_coefficient():
    response = jnp.array([[0.5, 0.0], [0.6, 0.0]])
    spend = jnp.array([[100.0, 50.0], [120.0, 40.0]])

    coefficient = roi_coefficient(jnp.array([2.0, 0.5]), response, spend)

    assert np.isfinite(coefficient[0])
    assert not np.isfinite(coefficient[1])


def test_lognormal_coefficients_and_gradients_match_the_unguarded_log_on_positive_exposures():
    response, _, _, scale, deviations = _grouped_case()
    contribution = jnp.array([40.0, 90.0, 25.0, 60.0])

    # Every group sees every channel here, so the guarded log must equal the plain log of each weighted total.
    def unguarded(values):
        weighted = jnp.sum(values, axis=0) * scale[:, None]
        center = jnp.log(contribution) - logsumexp(deviations + jnp.log(weighted), axis=0)
        coefficient = jnp.exp(center + deviations)
        return coefficient

    def coefficients(values):
        return contribution_coefficient(contribution, values, outcome_scale=scale, deviations=deviations)

    expected = unguarded(response)
    expected_gradient = jax.jacobian(unguarded)(response)

    eager = (coefficients(response), jax.jacobian(coefficients)(response))
    compiled = (jax.jit(coefficients)(response), jax.jit(jax.jacobian(coefficients))(response))

    for result, gradient in (eager, compiled):
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-7)


def test_lognormal_coefficients_keep_finite_gradients_when_a_group_never_sees_a_channel():
    response, spend, roi, scale, deviations = _grouped_case()
    # The second channel never runs in the first region, while its total across regions stays positive.
    media = response.at[:, 0, 1].set(0.0)

    def log_coefficients(retention):
        carried = geometric_adstock(media, retention, max_lag=2)
        coefficient = roi_coefficient(roi, carried, spend, outcome_scale=scale, deviations=deviations)
        log_coefficient = jnp.log(coefficient)
        return log_coefficient

    # Meridian's form sums exposures before taking the log, so it stays smooth where one region's total is zero.
    def meridian(retention):
        carried = geometric_adstock(media, retention, max_lag=2)
        a_gm = jnp.sum(carried, axis=0) * scale[:, None]
        beta_m = jnp.log(roi * spend.sum(axis=(0, 1))) - jnp.log(jnp.sum(a_gm * jnp.exp(deviations), axis=0))
        log_coefficient = beta_m + deviations
        return log_coefficient

    retention = jnp.asarray(0.5)
    expected = meridian(retention)
    expected_gradient = jax.jacobian(meridian)(retention)

    eager = (log_coefficients(retention), jax.jacobian(log_coefficients)(retention))
    compiled = (jax.jit(log_coefficients)(retention), jax.jit(jax.jacobian(log_coefficients))(retention))

    for result, gradient in (eager, compiled):
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-4, atol=1e-6)


def test_lognormal_coefficient_jacobians_match_meridian_where_a_group_never_sees_a_channel():
    response, _, _, scale, deviations = _grouped_case()
    contribution = jnp.array([40.0, 90.0, 25.0, 60.0])
    # The first region has no outcome scale, and the second channel also never runs in the second region.
    media = response.at[:, 1, 1].set(0.0)
    factor = scale.at[0].set(0.0)

    def coefficients(values, outcome_scale):
        return contribution_coefficient(contribution, values, outcome_scale=outcome_scale, deviations=deviations)

    # The zero totals have nonzero partials in Meridian's form, since it sums exposures before taking the log.
    def meridian(values, outcome_scale):
        a_gm = jnp.sum(values, axis=0) * outcome_scale[:, None]
        beta_m = jnp.log(contribution) - jnp.log(jnp.sum(a_gm * jnp.exp(deviations), axis=0))
        coefficient = jnp.exp(beta_m + deviations)
        return coefficient

    expected = jax.jacobian(meridian, argnums=(0, 1))(media, factor)

    reverse = jax.jacrev(coefficients, argnums=(0, 1))(media, factor)
    forward = jax.jacfwd(coefficients, argnums=(0, 1))(media, factor)
    compiled = jax.jit(jax.jacrev(coefficients, argnums=(0, 1)))(media, factor)

    for result in (reverse, forward, compiled):
        for derivative, expected_derivative in zip(result, expected, strict=True):
            assert np.isfinite(derivative).all()
            np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("arguments", "options", "message"),
    [
        ((jnp.ones(2), jnp.ones(4), jnp.ones(4)), {}, "response must have shape"),
        ((jnp.ones(2), jnp.ones((2, 2, 2, 2)), jnp.ones(2)), {}, "response must have shape"),
        ((jnp.ones(3), jnp.ones((4, 2)), jnp.ones(2)), {}, "roi must be a scalar or have one value per channel"),
        ((jnp.ones(2), jnp.ones((4, 2)), jnp.ones(3)), {}, "spend must end in the channel axis"),
        ((jnp.ones(2), jnp.ones((4, 2)), 5.0), {}, "spend must end in the channel axis"),
        (
            (jnp.ones(2), jnp.ones((4, 3, 2)), jnp.ones(2)),
            {"outcome_scale": jnp.ones(2)},
            "outcome_scale must be a scalar",
        ),
        (
            (jnp.ones(2), jnp.ones((4, 2)), jnp.ones(2)),
            {"outcome_scale": jnp.ones(3)},
            "outcome_scale must be a scalar",
        ),
        ((jnp.ones(2), jnp.ones((4, 2)), jnp.ones(2)), {"deviations": jnp.ones((1, 2))}, "group axis"),
        (
            (jnp.ones(2), jnp.ones((4, 3, 2)), jnp.ones(2)),
            {"deviations": jnp.ones((2, 3))},
            "deviations must have shape",
        ),
        ((jnp.ones(2), jnp.ones((4, 3, 2)), jnp.ones(2)), {"effects": "student"}, "effects must be"),
    ],
)
def test_roi_coefficient_rejects_inconsistent_shapes_and_settings(arguments, options, message):
    with pytest.raises(ValueError, match=message):
        roi_coefficient(*arguments, **options)


def test_contribution_coefficient_rejects_mismatched_contribution_shape():
    with pytest.raises(ValueError, match="contribution must be a scalar or have one value per channel"):
        contribution_coefficient(jnp.ones(3), jnp.ones((4, 2)))


def test_roi_coefficient_in_a_model_reproduces_declared_returns_through_media_metrics():
    frame = pl.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "sales": [10.0, 12.0, 9.0, 14.0],
            "video": [4.0, 1.0, 3.0, 2.0],
            "search": [2.0, 2.0, 1.0, 3.0],
            "video_cost": [2.0, 0.5, 1.5, 1.0],
            "search_cost": [1.0, 1.0, 0.5, 1.5],
        }
    )
    data = prepare_data(
        frame,
        time="week",
        outcome="sales",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Video", "Search"],
    )
    scaling = fit_data_scaling(data, scale_outcome=True)

    def transformed(media, reference, outcome_scaling, roi, intercept):
        coefficient = roi_coefficient(roi, reference.media, reference.spend, outcome_scale=outcome_scaling.scale)
        return {"mu": intercept + media @ coefficient, "coefficient": coefficient}

    model = Model(
        parameters={"roi": Positive(dims="channel"), "intercept": Real()},
        log_density=lambda mu: 0.0,
        data=Data(data, scaling=scaling),
        transformed_parameters=transformed,
    )
    draws = np.array([[[0.5, 2.0], [1.5, 0.8]]], dtype=np.float32)
    results = _collect_results(
        {"roi": draws, "intercept": np.zeros((1, 2), dtype=np.float32)}, data=data, dims={"roi": ("channel",)}
    )

    metrics = media_metrics(model, results, quantity="mu")
    np.testing.assert_allclose(metrics["roi"].values, draws, rtol=1e-4)

    # The identity is anchored on the training arrays, so changed exposures leave it alone.
    parameters = {"roi": jnp.asarray(draws[0, 0]), "intercept": jnp.asarray(0.0)}
    scenario = model.prepare_data(frame.with_columns(pl.col("video") * 3.0))
    np.testing.assert_array_equal(
        model.evaluate(parameters, scenario)["coefficient"], model.evaluate(parameters)["coefficient"]
    )


@pytest.mark.parametrize("window_only", [True, False])
def test_roi_coefficient_with_media_history_matches_media_metrics_only_for_the_window_response(window_only):
    frame = pl.DataFrame(
        {
            "week": [0, 1, 2, 3, 4],
            "sales": [8.0, 10.0, 12.0, 9.0, 14.0],
            "video": [6.0, 4.0, 1.0, 3.0, 2.0],
            "search": [3.0, 2.0, 2.0, 1.0, 3.0],
            "video_cost": [3.0, 2.0, 0.5, 1.5, 1.0],
            "search_cost": [1.5, 1.0, 1.0, 0.5, 1.5],
        }
    )
    data = prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        outcome="sales",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Video", "Search"],
        media_history=frame.filter(pl.col("week") == 0),
    )

    def transformed(media, reference, outcome_scaling, n_periods, roi, intercept):
        def carryover(exposure, periods):
            carried = geometric_adstock(exposure, alpha=0.6, max_lag=2)
            recent = carried[-periods:]
            return recent

        weeks = reference.n_periods
        response = carryover(reference.media, weeks)
        if window_only:
            response = response - carryover(reference.media.at[-weeks:].set(0.0), weeks)
        coefficient = roi_coefficient(roi, response, reference.spend, outcome_scale=outcome_scaling.scale)
        mu = intercept + carryover(media, n_periods) @ coefficient
        return {"mu": mu}

    model = Model(
        parameters={"roi": Positive(dims="channel"), "intercept": Real()},
        log_density=lambda mu: 0.0,
        data=Data(data, scaling=fit_data_scaling(data, scale_outcome=True)),
        transformed_parameters=transformed,
    )
    draws = np.array([[[0.5, 2.0], [1.5, 0.8]]], dtype=np.float32)
    results = _collect_results(
        {"roi": draws, "intercept": np.zeros((1, 2), dtype=np.float32)}, data=data, dims={"roi": ("channel",)}
    )

    measured = media_metrics(model, results, quantity="mu")["roi"].values

    # Crediting the history's carryover to the channel understates the measured return.
    if window_only:
        np.testing.assert_allclose(measured, draws, rtol=1e-4)
    else:
        assert (measured < draws * 0.99).all()
