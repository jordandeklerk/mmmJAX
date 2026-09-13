"""Tests for prepared carryover and saturation media contributions."""

from dataclasses import FrozenInstanceError, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

import mmmjax
from mmmjax import Interval, Positive, prepare_data
from mmmjax.adstock import delayed_adstock, geometric_adstock, weibull_cdf_adstock, weibull_pdf_adstock
from mmmjax.media import MediaEffect, ReachFrequencyEffect
from mmmjax.saturation import hill_saturation, log_saturation, logistic_saturation, root_saturation

_SUFFIXES = ("coefficient", "retention", "half_saturation", "slope")
_ADSTOCK_ROLES = {
    geometric_adstock: ("retention",),
    delayed_adstock: ("retention", "delay"),
    weibull_pdf_adstock: ("adstock_shape", "adstock_scale"),
    weibull_cdf_adstock: ("adstock_shape", "adstock_scale"),
}
_SATURATION_ROLES = {
    hill_saturation: ("half_saturation", "slope"),
    logistic_saturation: ("half_saturation",),
    root_saturation: ("exponent",),
    log_saturation: (),
}
_ALL_SUFFIXES = (*_SUFFIXES, "delay", "adstock_shape", "adstock_scale", "exponent")
_SELECTION_CASES = [
    pytest.param(geometric_adstock, hill_saturation, True, id="geometric-hill"),
    pytest.param(delayed_adstock, logistic_saturation, False, id="delayed-logistic"),
    pytest.param(weibull_pdf_adstock, root_saturation, True, id="weibull-pdf-root"),
    pytest.param(weibull_cdf_adstock, log_saturation, False, id="weibull-cdf-log"),
    pytest.param(geometric_adstock, log_saturation, False, id="geometric-log"),
    pytest.param(delayed_adstock, root_saturation, True, id="delayed-root"),
    pytest.param(weibull_pdf_adstock, hill_saturation, False, id="weibull-pdf-hill"),
    pytest.param(weibull_cdf_adstock, logistic_saturation, True, id="weibull-cdf-logistic"),
]


@pytest.fixture
def media_values():
    return np.array(
        [
            [[8, 1], [2, 4]],
            [[4, 2], [8, 1]],
            [[1, 4], [2, 2]],
            [[16, 8], [4, 1]],
            [[8, 2], [4, 4]],
        ],
        dtype=float,
    )


@pytest.fixture
def frequency_values():
    return np.array(
        [
            [[1, 3], [2, 4]],
            [[3, 2], [4, 1]],
            [[2, 4], [1, 3]],
            [[4, 1], [3, 2]],
            [[1, 2], [2, 4]],
        ],
        dtype=float,
    )


def _data(media, *, n_periods=3, start=0, reverse=False, observed=True):
    grouped = media.ndim == 3
    group_order = [1, 0] if grouped and reverse else ([0, 1] if grouped else [0])
    rows = []
    for time in range(len(media)):
        for group in group_order:
            exposure = media[time, group] if grouped else media[time]
            row = {
                "time": time + start,
                "video": exposure[0],
                "search": exposure[1],
                "sales": 0.5 + time / 10,
                "video_cost": exposure[0] / 2,
                "search_cost": exposure[1] / 2,
            }
            if grouped:
                row["region"] = ("west", "east")[group]
            rows.append(row)
    history_rows = (len(media) - n_periods) * len(group_order)
    channels = ["search", "video"] if reverse else ["video", "search"]
    return prepare_data(
        pl.DataFrame(rows[history_rows:]),
        time="time",
        groups=["region"] if grouped else [],
        media=channels,
        channels=channels,
        outcome="sales" if observed else None,
        spend=[channel + "_cost" for channel in channels] if observed else None,
        media_history=pl.DataFrame(rows[:history_rows]) if history_rows else None,
    )


def _parameters(*, grouped=False, name="paid"):
    values = ([[0.7, 1.3], [1.1, 0.5]] if grouped else [0.7, 1.3], [0.25, 0.6], [2.5, 4.0], [1.2, 0.8])
    return {f"{name}_{suffix}": jnp.asarray(value) for suffix, value in zip(_SUFFIXES, values, strict=True)}


def _reference(media, parameters, *, n_periods, max_lag=3, normalize=True, adstock_first=True):
    coefficient, retention, half, slope = (
        np.asarray(parameters[f"paid_{suffix}"], dtype=np.float64) for suffix in _SUFFIXES
    )
    source = np.asarray(media, dtype=np.float64)
    weights = retention ** np.arange(max_lag + 1)[:, None]
    if normalize:
        weights /= weights.sum(axis=0)

    def saturate(value):
        return value**slope / (value**slope + half**slope)

    if not adstock_first:
        source = saturate(source)
    carried = np.zeros_like(source)
    for time in range(len(source)):
        for lag in range(min(time, max_lag) + 1):
            carried[time] += weights[lag] * source[time - lag]
    response = saturate(carried) if adstock_first else carried
    return response[-n_periods:] * coefficient


def _selected_parameters(adstock, saturation, *, grouped=False, max_lag=3):
    values = {
        "coefficient": [[0.7, 1.3], [1.1, 0.5]] if grouped else [0.7, 1.3],
        "retention": [0.25, 0.6],
        "delay": [0.7, 1.4],
        "adstock_shape": [1.4, 2.3],
        "adstock_scale": [2.4, 3.1],
        "half_saturation": [2.5, 4.0],
        "slope": [1.2, 0.8],
        "exponent": [0.4, 0.7],
    }
    roles = ("coefficient", *_ADSTOCK_ROLES[adstock], *_SATURATION_ROLES[saturation])
    return {"paid_" + role: jnp.asarray(values[role]) for role in roles if role != "delay" or max_lag > 0}


def _selected_reference(
    media, parameters, *, adstock, saturation, n_periods, max_lag=3, normalize=True, adstock_first=True
):
    def carry(exposures):
        if adstock is geometric_adstock:
            values = {"alpha": parameters["paid_retention"]}
        elif adstock is delayed_adstock:
            values = {"alpha": parameters["paid_retention"], "theta": parameters.get("paid_delay", 0.0)}
        else:
            values = {"shape": parameters["paid_adstock_shape"], "scale": parameters["paid_adstock_scale"]}
        return adstock(exposures, **values, max_lag=max_lag, normalize=normalize)

    def saturate(exposures):
        values = [parameters["paid_" + role] for role in _SATURATION_ROLES[saturation]]
        return saturation(exposures, *values)

    response = saturate(carry(media)) if adstock_first else carry(saturate(media))
    return response[-n_periods:] * parameters["paid_coefficient"]


def test_media_effect_export_defaults_and_frozen_preparation_metadata(media_values):
    spec = MediaEffect(max_lag=0)
    prepared = spec._prepare(_data(media_values[:, 0]))

    assert mmmjax.MediaEffect is MediaEffect
    assert "MediaEffect" in mmmjax.__all__
    assert spec.name == "paid_media"
    assert spec.adstock is geometric_adstock and spec.saturation is hill_saturation
    assert spec.normalize is True and spec.adstock_first is True and spec.group_specific_coefficients is False
    assert not hasattr(spec, "automatic_priors")
    assert all(not hasattr(spec, suffix + "_prior") for suffix in _ALL_SUFFIXES)
    assert not hasattr(prepared, "log_prior")
    assert set(prepared.parameters) == {"paid_media_" + suffix for suffix in _SUFFIXES}
    assert jax.tree_util.tree_leaves(prepared) == []
    with pytest.raises(FrozenInstanceError):
        spec.max_lag = 2
    attribute = fields(prepared)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(prepared, attribute, getattr(prepared, attribute))


@pytest.mark.parametrize("adstock_first,normalize", [(True, True), (False, True), (True, False), (False, False)])
def test_component_matches_finite_lag_hill_reference_with_full_history(adstock_first, normalize, media_values):
    prepared = MediaEffect(max_lag=3, name="paid", adstock_first=adstock_first, normalize=normalize)._prepare(
        _data(media_values)
    )
    parameters = _parameters()
    actual = jax.jit(lambda component, media, values: component.apply(media, values))(
        prepared, jnp.asarray(media_values), parameters
    )
    expected = _reference(media_values, parameters, n_periods=3, adstock_first=adstock_first, normalize=normalize)
    without_history = _reference(
        media_values[-3:], parameters, n_periods=3, adstock_first=adstock_first, normalize=normalize
    )

    assert actual.shape == (3, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    assert not np.allclose(expected[0], without_history[0])


@pytest.mark.parametrize("group_specific_coefficients", [False, True])
def test_only_coefficient_shapes_change_across_groups(group_specific_coefficients, media_values):
    prepared = MediaEffect(max_lag=3, name="paid", group_specific_coefficients=group_specific_coefficients)._prepare(
        _data(media_values)
    )
    parameters = _parameters(grouped=group_specific_coefficients)
    for suffix in _SUFFIXES:
        declaration = prepared.parameters["paid_" + suffix]
        assert declaration.shape == ((2, 2) if suffix == "coefficient" and group_specific_coefficients else (2,))
        assert isinstance(declaration, Interval if suffix == "retention" else Positive)
    retention = prepared.parameters["paid_retention"]
    assert retention.lower == 0 and retention.upper == 1
    np.testing.assert_allclose(
        prepared.apply(media_values, parameters),
        _reference(media_values, parameters, n_periods=3),
        rtol=4e-6,
        atol=2e-6,
    )


def test_national_response_and_zero_lag_keep_channel_contributions_separate(media_values):
    media = media_values[:, 0]
    prepared = MediaEffect(max_lag=0, name="paid")._prepare(_data(media))
    parameters = _parameters()
    response = prepared.apply(media, parameters)

    assert response.shape == (3, 2)
    np.testing.assert_allclose(response, _reference(media, parameters, n_periods=3, max_lag=0), rtol=4e-6, atol=2e-6)


def test_component_gradients_match_finite_differences_and_depend_on_historical_media(media_values):
    prepared = MediaEffect(max_lag=3, name="paid")._prepare(_data(media_values))
    parameters = _parameters()

    def objective(media, values):
        return prepared.apply(media, values).sum()

    media_gradient, gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(jnp.asarray(media_values), parameters)
    values64 = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}
    for name, values in values64.items():
        expected = []
        for delta in np.eye(len(values)) * 1e-4:
            plus = _reference(media_values, {**values64, name: values + delta}, n_periods=3).sum()
            minus = _reference(media_values, {**values64, name: values - delta}, n_periods=3).sum()
            expected.append((plus - minus) / 2e-4)
        assert np.all(np.abs(expected) > 1e-5)
        np.testing.assert_allclose(gradients[name], expected, rtol=3e-5, atol=3e-6)
    shift = np.zeros_like(media_values)
    shift[0, 0, 0] = 1e-4
    expected_history = (
        _reference(media_values + shift, values64, n_periods=3).sum()
        - _reference(media_values - shift, values64, n_periods=3).sum()
    ) / 2e-4
    assert expected_history > 0
    np.testing.assert_allclose(media_gradient[0, 0, 0], expected_history, rtol=3e-5, atol=3e-6)


def test_vmap_uses_dynamic_media_and_parameter_draws(media_values):
    prepared = MediaEffect(max_lag=3, name="paid")._prepare(_data(media_values))
    draws = jax.tree.map(lambda value: jnp.stack((value, value * 1.1)), _parameters())
    apply = jax.jit(jax.vmap(lambda component, media, values: component.apply(media, values), in_axes=(None, None, 0)))
    for multiplier in (1.0, 1.7):
        actual = apply(prepared, jnp.asarray(media_values * multiplier), draws)
        assert actual.shape == (2, 3, 2, 2)
        for index in range(2):
            expected = _reference(
                media_values * multiplier, {name: value[index] for name, value in draws.items()}, n_periods=3
            )
            np.testing.assert_allclose(actual[index], expected, rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize("adstock_first", [True, False])
def test_sparse_exposures_have_finite_parameter_gradients_below_unit_slope(adstock_first):
    media = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    prepared = MediaEffect(max_lag=2, name="paid", adstock_first=adstock_first)._prepare(_data(media))
    parameters = {
        "paid_coefficient": jnp.ones(2),
        "paid_retention": jnp.full(2, 0.3),
        "paid_half_saturation": jnp.ones(2),
        "paid_slope": jnp.full(2, 0.5),
    }
    value, gradients = jax.jit(jax.value_and_grad(lambda values: prepared.apply(media, values).sum()))(parameters)
    values64 = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}
    assert np.isfinite(value)
    for name, values in values64.items():
        expected = []
        for delta in np.eye(2) * 1e-4:
            plus = _reference(
                media, {**values64, name: values + delta}, n_periods=3, max_lag=2, adstock_first=adstock_first
            ).sum()
            minus = _reference(
                media, {**values64, name: values - delta}, n_periods=3, max_lag=2, adstock_first=adstock_first
            ).sum()
            expected.append((plus - minus) / 2e-4)
        np.testing.assert_allclose(gradients[name], expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_array_equal(gradients[name][1], 0.0)


def test_zero_exposure_derivative_convention_preserves_unit_slope_derivatives():
    media = jnp.zeros((1, 2))
    prepared = MediaEffect(max_lag=0, name="paid")._prepare(_data(np.asarray(media), n_periods=1))
    parameters = _parameters()
    derivative = jax.jit(jax.grad(lambda exposures, values: prepared.apply(exposures, values).sum()))

    for slope in (0.5, 1.0, 2.0):
        values = {**parameters, "paid_slope": jnp.full(2, slope)}
        expected = (
            np.asarray(parameters["paid_coefficient"]) / np.asarray(parameters["paid_half_saturation"])
            if slope == 1
            else np.zeros(2)
        )
        np.testing.assert_allclose(derivative(media, values), expected[None, :], rtol=3e-6, atol=2e-6)


def test_zero_exposures_keep_invalid_saturation_parameters_nan():
    media = jnp.zeros((1, 2))
    prepared = MediaEffect(max_lag=0, name="paid")._prepare(_data(np.asarray(media), n_periods=1))
    parameters = {**_parameters(), "paid_slope": jnp.full(2, 0.5)}
    apply = jax.jit(lambda exposures, values: prepared.apply(exposures, values))

    for role in ("paid_half_saturation", "paid_slope"):
        for invalid in (0.0, -1.0, np.nan, np.inf):
            values = {**parameters, role: jnp.array([invalid, parameters[role][1]])}
            response = apply(media, values)
            assert np.isnan(response[0, 0])
            np.testing.assert_array_equal(response[0, 1], 0.0)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_media_declarations_and_response_follow_prepared_precision(enable_x64, media_values):
    data = _data(media_values)
    spec = MediaEffect(max_lag=3, name="paid", group_specific_coefficients=True)
    with jax.enable_x64(enable_x64):
        prepared = spec._prepare(data)
        parameters = _parameters(grouped=True)
        dtype = np.dtype("float64" if enable_x64 else "float32")
        response = jax.jit(prepared.apply)(jnp.asarray(media_values), parameters)
        assert response.dtype == prepared.dtype == dtype
        for name, declaration in prepared.parameters.items():
            assert declaration.dtype == dtype
            assert declaration.shape == parameters[name].shape
        np.testing.assert_allclose(response, _reference(media_values, parameters, n_periods=3), rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize("option", ["automatic_priors", *(suffix + "_prior" for suffix in _ALL_SUFFIXES)])
def test_media_rejects_removed_prior_options(option):
    with pytest.raises(TypeError, match=option):
        MediaEffect(max_lag=3, **{option: False if option == "automatic_priors" else lambda value: -value})


def test_media_rejects_old_group_specific_keyword():
    with pytest.raises(TypeError, match="group_specific"):
        MediaEffect(max_lag=3, group_specific=True)


def test_forecasts_require_alignment_and_reuse_coefficients_with_changed_history_and_periods(media_values):
    training = _data(media_values)
    with jax.enable_x64(False):
        prepared = MediaEffect(max_lag=3, name="paid", group_specific_coefficients=True)._prepare(training)
        parameters = _parameters(grouped=True)
    for media, periods in ((media_values[:4] * 0.7, 1), (media_values[:2] * 1.2, 2)):
        future_data = _data(media, n_periods=periods, start=6, reverse=True, observed=False)
        with pytest.raises(ValueError, match=r"align|order|group|channel|column"):
            prepared.for_data(future_data)
        aligned = future_data._align_to(training)
        with jax.enable_x64(True):
            future = prepared.for_data(aligned)
        for name in prepared.parameters:
            assert future.parameters[name].dtype == prepared.parameters[name].dtype
            assert future.parameters[name].shape == prepared.parameters[name].shape
        actual = jax.jit(lambda component, exposures, values: component.apply(exposures, values))(
            future, jnp.asarray(aligned.arrays["media"], dtype=jnp.float32), parameters
        )
        assert actual.shape == (periods, 2, 2)
        np.testing.assert_allclose(actual, _reference(media, parameters, n_periods=periods), rtol=4e-6, atol=2e-6)
        assert "outcome" not in aligned.arrays and "spend" not in aligned.arrays


def test_configuration_validates_required_lag_names_and_booleans():
    with pytest.raises(TypeError, match="max_lag"):
        MediaEffect()
    for lag in (True, 1.5, np.int64(2)):
        with pytest.raises(TypeError, match="max_lag"):
            MediaEffect(max_lag=lag)
    with pytest.raises(ValueError, match="max_lag"):
        MediaEffect(max_lag=-1)
    for option in ("normalize", "adstock_first", "group_specific_coefficients"):
        with pytest.raises(TypeError, match=option):
            MediaEffect(max_lag=2, **{option: 1})
    for name in ("", "not-valid", "class"):
        with pytest.raises(ValueError, match="name"):
            MediaEffect(max_lag=2, name=name)


def test_preparation_requires_media_and_consistent_forecast_identities(media_values):
    prepared = MediaEffect(max_lag=2, name="paid")._prepare(_data(media_values))
    with pytest.raises(TypeError, match=r"data|PreparedData"):
        MediaEffect(max_lag=2)._prepare({"media": media_values})
    missing = prepare_data(pl.DataFrame({"time": [0, 1], "sales": [0.3, 0.7]}), time="time", outcome="sales")
    with pytest.raises(ValueError, match="media"):
        MediaEffect(max_lag=2)._prepare(missing)
    with pytest.raises(ValueError, match=r"group_specific_coefficients|group"):
        MediaEffect(max_lag=2, group_specific_coefficients=True)._prepare(_data(media_values[:, 0]))
    future = _data(media_values, start=6, observed=False)
    for invalid in (
        replace(future, time_column="new_time"),
        replace(future, group_columns=("country",)),
        replace(future, group_values=(("west",), ("central",))),
        replace(future, channels=("other", "search")),
        replace(future, columns={"media": ("other", "search")}),
    ):
        with pytest.raises(ValueError):
            prepared.for_data(invalid)


def test_apply_validates_media_and_parameter_shapes(media_values):
    prepared = MediaEffect(max_lag=2, name="paid")._prepare(_data(media_values))
    parameters = _parameters()
    for shape in ((), (5, 2), (5, 3, 2), (5, 2, 3), (2, 2, 2), (6, 2, 2)):
        with pytest.raises(ValueError, match=r"media|shape|period"):
            prepared.apply(jnp.ones(shape), parameters)
    for suffix in _SUFFIXES:
        malformed = {**parameters, "paid_" + suffix: jnp.ones((2, 1))}
        missing = {name: value for name, value in parameters.items() if name != "paid_" + suffix}
        for values in (malformed, missing):
            with pytest.raises(ValueError, match=r"paid_|parameter|shape|missing"):
                prepared.apply(media_values, values)
    with pytest.raises(TypeError, match="mapping"):
        prepared.apply(media_values, jnp.ones(2))
    with pytest.raises(TypeError, match="real numeric"):
        prepared.apply(media_values, {**parameters, "paid_coefficient": jnp.ones(2, dtype=jnp.complex64)})


@pytest.mark.parametrize("adstock", _ADSTOCK_ROLES, ids=lambda function: function.__name__)
@pytest.mark.parametrize("saturation", _SATURATION_ROLES, ids=lambda function: function.__name__)
def test_selectors_declare_only_active_parameter_constraints(adstock, saturation, media_values):
    prepared = MediaEffect(
        max_lag=3, name="paid", adstock=adstock, saturation=saturation, group_specific_coefficients=True
    )._prepare(_data(media_values))
    parameters = _selected_parameters(adstock, saturation, grouped=True)
    assert set(prepared.parameters) == set(parameters)
    for name, declaration in prepared.parameters.items():
        role = name.removeprefix("paid_")
        assert declaration.shape == ((2, 2) if role == "coefficient" else (2,))
        if role in ("retention", "delay", "exponent"):
            assert isinstance(declaration, Interval)
            assert declaration.lower == 0.0
            assert declaration.upper == (3.0 if role == "delay" else 1.0)
        else:
            assert isinstance(declaration, Positive)


@pytest.mark.parametrize("adstock,saturation,adstock_first", _SELECTION_CASES)
def test_selected_transforms_compose_with_history_groups_and_finite_jit_gradients(
    adstock, saturation, adstock_first, media_values
):
    prepared = MediaEffect(
        max_lag=3,
        name="paid",
        adstock=adstock,
        saturation=saturation,
        adstock_first=adstock_first,
        normalize=adstock_first,
        group_specific_coefficients=adstock_first,
    )._prepare(_data(media_values))
    parameters = _selected_parameters(adstock, saturation, grouped=adstock_first)

    def objective(media, values):
        response = prepared.apply(media, values)
        return response.sum(), response

    (_, actual), (media_gradient, parameter_gradients) = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    )(jnp.asarray(media_values), parameters)
    options = {"adstock": adstock, "saturation": saturation, "normalize": adstock_first, "adstock_first": adstock_first}
    expected = _selected_reference(media_values, parameters, n_periods=3, **options)
    without_history = _selected_reference(media_values[-3:], parameters, n_periods=3, **options)
    assert actual.shape == (3, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    assert not np.allclose(actual[0], without_history[0])
    assert np.any(np.abs(media_gradient[:2]) > 1e-6)
    for gradient in jax.tree.leaves((media_gradient, parameter_gradients)):
        assert np.all(np.isfinite(gradient))
    inactive_values = {
        "paid_" + role: jnp.full((2,), jnp.nan) for role in _ALL_SUFFIXES if "paid_" + role not in parameters
    }
    np.testing.assert_allclose(prepared.apply(media_values, parameters | inactive_values), actual, rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize(
    "adstock,saturation", [(delayed_adstock, root_saturation), (weibull_cdf_adstock, log_saturation)]
)
def test_selected_parameter_gradients_match_finite_differences_and_vmap_draws(adstock, saturation, media_values):
    media = media_values[:, 0]
    with jax.enable_x64(True):
        prepared = MediaEffect(max_lag=3, name="paid", adstock=adstock, saturation=saturation)._prepare(_data(media))
        parameters = _selected_parameters(adstock, saturation)
        gradients = jax.jit(jax.grad(lambda values: prepared.apply(media, values).sum()))(parameters)
        for name, values in parameters.items():
            expected = []
            for delta in np.eye(2) * 1e-5:
                plus = _selected_reference(
                    media, parameters | {name: values + delta}, adstock=adstock, saturation=saturation, n_periods=3
                ).sum()
                minus = _selected_reference(
                    media, parameters | {name: values - delta}, adstock=adstock, saturation=saturation, n_periods=3
                ).sum()
                expected.append((plus - minus) / 2e-5)
            np.testing.assert_allclose(gradients[name], expected, rtol=2e-5, atol=2e-6)
        draws = jax.tree.map(lambda value: jnp.stack((value, value * 1.1)), parameters)
        exposures = jnp.stack((media, media * 1.7))
        responses = jax.jit(jax.vmap(prepared.apply))(exposures, draws)
        assert responses.shape == (2, 3, 2)
        for index in range(2):
            expected = _selected_reference(
                exposures[index],
                {name: value[index] for name, value in draws.items()},
                adstock=adstock,
                saturation=saturation,
                n_periods=3,
            )
            np.testing.assert_allclose(responses[index], expected, rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize("normalize", [True, False])
def test_delayed_zero_lag_uses_fixed_zero_delay_without_declaring_a_delay_parameter(normalize, media_values):
    media = media_values[:, 0]
    prepared = MediaEffect(
        max_lag=0, name="paid", adstock=delayed_adstock, saturation=log_saturation, normalize=normalize
    )._prepare(_data(media))
    parameters = _selected_parameters(delayed_adstock, log_saturation, max_lag=0)
    assert set(prepared.parameters) == {"paid_coefficient", "paid_retention"}
    np.testing.assert_allclose(
        prepared.apply(media, parameters), np.log1p(media[-3:]) * parameters["paid_coefficient"], rtol=3e-6
    )
    gradient = jax.jit(jax.grad(lambda values: prepared.apply(media, values).sum()))(parameters)
    np.testing.assert_array_equal(gradient["paid_retention"], np.zeros(2))


@pytest.mark.parametrize("adstock_first", [True, False])
def test_root_selection_has_finite_gradients_with_zero_exposures(adstock_first):
    media = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    prepared = MediaEffect(
        max_lag=2, name="paid", adstock=delayed_adstock, saturation=root_saturation, adstock_first=adstock_first
    )._prepare(_data(media))
    parameters = _selected_parameters(delayed_adstock, root_saturation)
    value, gradients = jax.jit(jax.value_and_grad(lambda values: prepared.apply(media, values).sum()))(parameters)
    assert np.isfinite(value)
    for gradient in gradients.values():
        assert np.all(np.isfinite(gradient))
        np.testing.assert_array_equal(gradient[1], 0.0)
    np.testing.assert_array_equal(prepared.apply(media, parameters)[:, 1], np.zeros(3))


@pytest.mark.parametrize("selector", ["adstock", "saturation"])
@pytest.mark.parametrize("invalid", [None, "geometric", 0.5, lambda value: value])
def test_selectors_reject_nonfunctions_and_unsupported_callables(selector, invalid):
    with pytest.raises((TypeError, ValueError), match=selector):
        MediaEffect(max_lag=3, **{selector: invalid})


def test_selectors_reject_functions_from_the_wrong_transform_family():
    with pytest.raises((TypeError, ValueError), match="adstock"):
        MediaEffect(max_lag=3, adstock=hill_saturation)
    with pytest.raises((TypeError, ValueError), match="saturation"):
        MediaEffect(max_lag=3, saturation=geometric_adstock)


def _rf_data(reach, frequency, *, n_periods=3, start=0, reverse=False, observed=True):
    grouped = reach.ndim == 3
    group_order = [1, 0] if grouped and reverse else ([0, 1] if grouped else [0])
    rows = []
    for time in range(len(reach)):
        for group in group_order:
            reach_row = reach[time, group] if grouped else reach[time]
            frequency_row = frequency[time, group] if grouped else frequency[time]
            row = {
                "time": start + time,
                "video_reach": reach_row[0],
                "search_reach": reach_row[1],
                "video_frequency": frequency_row[0],
                "search_frequency": frequency_row[1],
                "sales": 0.5 + time / 10,
            }
            if grouped:
                row["region"] = ("west", "east")[group]
            rows.append(row)
    history_rows = (len(reach) - n_periods) * len(group_order)
    channels = ["search", "video"] if reverse else ["video", "search"]
    return prepare_data(
        pl.DataFrame(rows[history_rows:]),
        time="time",
        groups=["region"] if grouped else [],
        reach=[channel + "_reach" for channel in channels],
        media_frequency=[channel + "_frequency" for channel in channels],
        rf_channels=channels,
        outcome="sales" if observed else None,
        media_history=pl.DataFrame(rows[:history_rows]) if history_rows else None,
    )


def _rf_reference(reach, frequency, parameters, *, n_periods=3, max_lag=3, normalize=True):
    coefficient, retention, half, slope = (
        np.asarray(parameters[f"paid_{suffix}"], dtype=np.float64) for suffix in _SUFFIXES
    )
    frequency = np.asarray(frequency, dtype=np.float64)
    source = np.asarray(reach, dtype=np.float64) * frequency**slope / (frequency**slope + half**slope)
    weights = retention ** np.arange(max_lag + 1)[:, None]
    if normalize:
        weights /= weights.sum(axis=0)
    carried = np.zeros_like(source)
    for time in range(len(source)):
        for lag in range(min(time, max_lag) + 1):
            carried[time] += weights[lag] * source[time - lag]
    return carried[-n_periods:] * coefficient


def test_rf_effect_defaults_frozen_metadata_and_explicit_priors(media_values, frequency_values):
    specification = ReachFrequencyEffect(max_lag=0)
    prepared = specification._prepare(_rf_data(media_values, frequency_values))
    assert mmmjax.ReachFrequencyEffect is ReachFrequencyEffect
    assert "ReachFrequencyEffect" in mmmjax.__all__
    assert specification.name == "paid_rf"
    assert specification.adstock is geometric_adstock and specification.saturation is hill_saturation
    assert specification.normalize is True and specification.group_specific_coefficients is False
    assert not hasattr(specification, "adstock_first")
    assert not hasattr(specification, "automatic_priors")
    assert all(not hasattr(specification, suffix + "_prior") for suffix in _ALL_SUFFIXES)
    assert not hasattr(prepared, "log_prior")
    assert set(prepared.parameters) == {"paid_rf_" + suffix for suffix in _SUFFIXES}
    assert prepared.channel_axis == "rf_channel"
    assert prepared.media_columns == ("video_reach", "search_reach")
    assert prepared.frequency_columns == ("video_frequency", "search_frequency")
    assert jax.tree_util.tree_leaves(prepared) == []
    ordinary = MediaEffect(max_lag=0)._prepare(_data(media_values))
    assert ordinary.channel_axis == "channel" and ordinary.frequency_columns == ()
    with pytest.raises(FrozenInstanceError):
        specification.max_lag = 2
    with pytest.raises(FrozenInstanceError):
        prepared.frequency_columns = ()


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("group_specific_coefficients", [False, True])
def test_rf_matches_independent_finite_lag_reference_and_group_coefficients(
    normalize, group_specific_coefficients, media_values, frequency_values
):
    prepared = ReachFrequencyEffect(
        max_lag=3, name="paid", normalize=normalize, group_specific_coefficients=group_specific_coefficients
    )._prepare(_rf_data(media_values, frequency_values))
    parameters = _parameters(grouped=group_specific_coefficients)
    actual = jax.jit(lambda component, reach, frequency, values: component.apply(reach, values, frequency=frequency))(
        prepared, jnp.asarray(media_values), jnp.asarray(frequency_values), parameters
    )
    expected = _rf_reference(media_values, frequency_values, parameters, normalize=normalize)
    without_history = _rf_reference(media_values[-3:], frequency_values[-3:], parameters, normalize=normalize)
    assert actual.shape == (3, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    assert not np.allclose(expected[0], without_history[0])


def test_rf_gradients_match_independent_differences_and_use_both_histories(media_values, frequency_values):
    prepared = ReachFrequencyEffect(max_lag=3, name="paid")._prepare(_rf_data(media_values, frequency_values))
    parameters = _parameters()

    def objective(reach, frequency, values):
        return prepared.apply(reach, values, frequency=frequency).sum()

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(media_values, frequency_values, parameters)
    values64 = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}
    for name, values in values64.items():
        expected = []
        for delta in np.eye(2) * 1e-4:
            plus = _rf_reference(media_values, frequency_values, values64 | {name: values + delta}).sum()
            minus = _rf_reference(media_values, frequency_values, values64 | {name: values - delta}).sum()
            expected.append((plus - minus) / 2e-4)
        np.testing.assert_allclose(gradients[2][name], expected, rtol=4e-5, atol=4e-6)
    shift = np.zeros_like(media_values)
    shift[0, 0, 0] = 1e-4
    for index in (0, 1):
        exposures = [media_values, frequency_values]
        plus, minus = exposures.copy(), exposures.copy()
        plus[index] = exposures[index] + shift
        minus[index] = exposures[index] - shift
        expected = (_rf_reference(*plus, values64).sum() - _rf_reference(*minus, values64).sum()) / 2e-4
        assert expected > 0
        np.testing.assert_allclose(gradients[index][0, 0, 0], expected, rtol=4e-5, atol=4e-6)


@pytest.mark.parametrize("adstock", _ADSTOCK_ROLES, ids=lambda function: function.__name__)
@pytest.mark.parametrize("saturation", _SATURATION_ROLES, ids=lambda function: function.__name__)
def test_rf_transformation_choices_declare_constraints_and_apply_fixed_order(
    adstock, saturation, media_values, frequency_values
):
    prepared = ReachFrequencyEffect(
        max_lag=3, name="paid", adstock=adstock, saturation=saturation, group_specific_coefficients=True
    )._prepare(_rf_data(media_values, frequency_values))
    parameters = _selected_parameters(adstock, saturation, grouped=True)
    assert set(prepared.parameters) == set(parameters)
    for name, declaration in prepared.parameters.items():
        role = name.removeprefix("paid_")
        assert declaration.shape == ((2, 2) if role == "coefficient" else (2,))
        if role in ("retention", "delay", "exponent"):
            assert isinstance(declaration, Interval)
            assert declaration.lower == 0 and declaration.upper == (3 if role == "delay" else 1)
        else:
            assert isinstance(declaration, Positive)
    arguments = [parameters["paid_" + role] for role in _SATURATION_ROLES[saturation]]
    weighted = media_values * saturation(frequency_values, *arguments)
    if adstock is geometric_adstock:
        carryover = {"alpha": parameters["paid_retention"]}
    elif adstock is delayed_adstock:
        carryover = {"alpha": parameters["paid_retention"], "theta": parameters["paid_delay"]}
    else:
        carryover = {"shape": parameters["paid_adstock_shape"], "scale": parameters["paid_adstock_scale"]}
    expected = adstock(weighted, **carryover, max_lag=3)[-3:] * parameters["paid_coefficient"]
    actual = prepared.apply(media_values, parameters, frequency=frequency_values)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)


def test_rf_vmap_preserves_dynamic_reach_frequency_and_parameter_draws(media_values, frequency_values):
    prepared = ReachFrequencyEffect(max_lag=3, name="paid")._prepare(
        _rf_data(media_values[:, 0], frequency_values[:, 0])
    )
    draws = jax.tree.map(lambda value: jnp.stack((value, value * 1.1)), _parameters())
    reaches = jnp.stack((media_values[:, 0], media_values[:, 0] * 1.7))
    frequencies = jnp.stack((frequency_values[:, 0], frequency_values[:, 0] * 0.4))
    apply = jax.jit(jax.vmap(lambda reach, frequency, values: prepared.apply(reach, values, frequency=frequency)))
    actual = apply(reaches, frequencies, draws)
    assert actual.shape == (2, 3, 2)
    for index in range(2):
        parameters = {name: value[index] for name, value in draws.items()}
        expected = _rf_reference(reaches[index], frequencies[index], parameters)
        np.testing.assert_allclose(actual[index], expected, rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize("saturation", [hill_saturation, root_saturation])
def test_rf_zero_frequency_has_finite_exposure_and_parameter_gradients(saturation):
    reach = np.array([[0.0, 2.0], [2.0, 1.0], [1.0, 0.0]])
    frequency = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    prepared = ReachFrequencyEffect(max_lag=2, name="paid", saturation=saturation)._prepare(_rf_data(reach, frequency))
    parameters = _selected_parameters(geometric_adstock, saturation)
    if saturation is hill_saturation:
        parameters["paid_slope"] = jnp.full(2, 0.5)

    def objective(reach, frequency, values):
        response = prepared.apply(reach, values, frequency=frequency)
        return response.sum(), response

    (_, response), gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True))(
        reach, frequency, parameters
    )
    for gradient in jax.tree.leaves(gradients):
        assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(response[:, 1], np.zeros(3))
    np.testing.assert_array_equal(gradients[1][frequency == 0], 0.0)
    for gradient in gradients[2].values():
        np.testing.assert_array_equal(gradient[1], 0.0)
    np.testing.assert_array_equal(prepared.apply(reach * 0, parameters, frequency=frequency), np.zeros((3, 2)))


@pytest.mark.parametrize("normalize", [True, False])
def test_rf_delayed_zero_lag_omits_delay_and_keeps_reach_linear(normalize, media_values, frequency_values):
    reach, frequency = media_values[:, 0], frequency_values[:, 0]
    prepared = ReachFrequencyEffect(
        max_lag=0, name="paid", adstock=delayed_adstock, saturation=log_saturation, normalize=normalize
    )._prepare(_rf_data(reach, frequency))
    parameters = _selected_parameters(delayed_adstock, log_saturation, max_lag=0)
    assert set(prepared.parameters) == {"paid_coefficient", "paid_retention"}
    expected = reach[-3:] * np.log1p(frequency[-3:]) * parameters["paid_coefficient"]
    np.testing.assert_allclose(prepared.apply(reach, parameters, frequency=frequency), expected, rtol=3e-6)
    gradient = jax.grad(lambda values: prepared.apply(reach, values, frequency=frequency).sum())(parameters)
    np.testing.assert_array_equal(gradient["paid_retention"], np.zeros(2))


def test_rf_forecast_alignment_retains_column_identities_and_training_precision(media_values, frequency_values):
    training = _rf_data(media_values, frequency_values)
    with jax.enable_x64(False):
        prepared = ReachFrequencyEffect(max_lag=3, name="paid", group_specific_coefficients=True)._prepare(training)
        parameters = _parameters(grouped=True)
    for periods in (1, 2):
        reach, frequency = media_values[:4] * 0.7, frequency_values[:4] * 1.2
        future = _rf_data(reach, frequency, n_periods=periods, start=6, reverse=True, observed=False)
        with pytest.raises(ValueError, match="ordering"):
            prepared.for_data(future)
        aligned = future._align_to(training)
        with jax.enable_x64(True):
            component = prepared.for_data(aligned)
        assert component.dtype == prepared.dtype == np.dtype("float32")
        for name, declaration in component.parameters.items():
            assert declaration.shape == prepared.parameters[name].shape
            assert declaration.dtype == prepared.parameters[name].dtype
        response = component.apply(aligned.arrays["reach"], parameters, frequency=aligned.arrays["media_frequency"])
        np.testing.assert_allclose(
            response, _rf_reference(reach, frequency, parameters, n_periods=periods), rtol=4e-6, atol=2e-6
        )
    for invalid in (
        replace(training, time_column="new_time"),
        replace(training, group_columns=("country",)),
        replace(training, group_values=(("west",), ("central",))),
        replace(training, rf_channels=("other", "search")),
        replace(training, columns=training.columns | {"reach": ("other", "search_reach")}),
        replace(training, columns=training.columns | {"media_frequency": ("search_frequency", "video_frequency")}),
    ):
        with pytest.raises(ValueError, match="ordering"):
            prepared.for_data(invalid)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_rf_response_and_declarations_follow_prepared_precision(enable_x64, media_values, frequency_values):
    with jax.enable_x64(enable_x64):
        prepared = ReachFrequencyEffect(max_lag=3, name="paid")._prepare(_rf_data(media_values, frequency_values))
        dtype = np.dtype("float64" if enable_x64 else "float32")
        response = jax.jit(prepared.apply)(media_values, _parameters(), frequency=frequency_values)
        assert response.dtype == prepared.dtype == dtype
        assert all(declaration.dtype == dtype for declaration in prepared.parameters.values())


def test_rf_configuration_validates_lag_names_flags_selectors_and_rejects_order_and_priors():
    with pytest.raises(TypeError, match="max_lag"):
        ReachFrequencyEffect()
    with pytest.raises(TypeError):
        ReachFrequencyEffect(3)
    for lag in (True, 1.5, np.int64(2)):
        with pytest.raises(TypeError, match="max_lag"):
            ReachFrequencyEffect(max_lag=lag)
    with pytest.raises(ValueError, match="max_lag"):
        ReachFrequencyEffect(max_lag=-1)
    with pytest.raises(TypeError, match="name"):
        ReachFrequencyEffect(max_lag=2, name=3)
    for name in ("", "not-valid", "class"):
        with pytest.raises(ValueError, match="name"):
            ReachFrequencyEffect(max_lag=2, name=name)
    for option in ("normalize", "group_specific_coefficients"):
        with pytest.raises(TypeError, match=option):
            ReachFrequencyEffect(max_lag=2, **{option: 1})
    for option in ("adstock_first", "automatic_priors", *(role + "_prior" for role in _ALL_SUFFIXES)):
        with pytest.raises(TypeError, match=option):
            ReachFrequencyEffect(max_lag=2, **{option: False})
    for selector in ("adstock", "saturation"):
        for invalid in (None, "geometric", 0.5, lambda value: value):
            with pytest.raises((TypeError, ValueError), match=selector):
                ReachFrequencyEffect(max_lag=2, **{selector: invalid})


def test_rf_preparation_requires_paired_inputs_and_matching_history_shapes(media_values, frequency_values):
    specification = ReachFrequencyEffect(max_lag=3, name="paid")
    training = _rf_data(media_values, frequency_values)
    with pytest.raises(TypeError, match="PreparedData"):
        specification._prepare({"reach": media_values})
    with pytest.raises(ValueError, match="reach and media_frequency"):
        specification._prepare(_data(media_values))
    with pytest.raises(ValueError, match="group_specific_coefficients"):
        ReachFrequencyEffect(max_lag=2, group_specific_coefficients=True)._prepare(
            _rf_data(media_values[:, 0], frequency_values[:, 0])
        )
    for role in ("reach", "media_frequency"):
        missing = replace(training, arrays={name: array for name, array in training.arrays.items() if name != role})
        with pytest.raises(ValueError, match="reach and media_frequency"):
            specification._prepare(missing)
        malformed = replace(training, arrays=training.arrays | {role: training.arrays[role][:-1]})
        with pytest.raises(ValueError, match="shape"):
            specification._prepare(malformed)
    for invalid in (
        replace(training, time_values=()),
        replace(training, media_time_values=()),
        replace(training, media_time_values=training.media_time_values[::-1]),
    ):
        with pytest.raises(ValueError, match="periods"):
            specification._prepare(invalid)


def test_rf_application_requires_frequency_and_matching_exposure_and_parameter_shapes(media_values, frequency_values):
    prepared = ReachFrequencyEffect(max_lag=3, name="paid")._prepare(_rf_data(media_values, frequency_values))
    parameters = _parameters()
    with pytest.raises(ValueError, match="frequency"):
        prepared.apply(media_values, parameters)
    for shape in ((), (5, 2), (5, 3, 2), (5, 2, 3), (2, 2, 2), (6, 2, 2)):
        with pytest.raises(ValueError, match="shape"):
            prepared.apply(jnp.ones(shape), parameters, frequency=frequency_values)
        with pytest.raises(ValueError, match="shape"):
            prepared.apply(media_values, parameters, frequency=jnp.ones(shape))
    for suffix in _SUFFIXES:
        malformed = parameters | {"paid_" + suffix: jnp.ones((2, 1))}
        missing = {name: value for name, value in parameters.items() if name != "paid_" + suffix}
        for values in (malformed, missing):
            with pytest.raises(ValueError, match="parameter"):
                prepared.apply(media_values, values, frequency=frequency_values)
