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
from mmmjax.media import MediaEffect
from mmmjax.saturation import hill_saturation, log_saturation, logistic_saturation, root_saturation

_MEDIA = np.array(
    [[[8, 1], [2, 4]], [[4, 2], [8, 1]], [[1, 4], [2, 2]], [[16, 8], [4, 1]], [[8, 2], [4, 4]]], dtype=float
)
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


def test_media_effect_export_defaults_and_frozen_preparation_metadata():
    spec = MediaEffect(max_lag=0)
    prepared = spec._prepare(_data(_MEDIA[:, 0]))

    assert mmmjax.MediaEffect is MediaEffect
    assert "MediaEffect" in mmmjax.__all__
    assert spec.name == "paid_media"
    assert spec.adstock is geometric_adstock and spec.saturation is hill_saturation
    assert spec.normalize is True and spec.adstock_first is True and spec.group_specific is False
    assert spec.automatic_priors is True
    assert all(getattr(spec, suffix + "_prior") is None for suffix in _SUFFIXES)
    assert set(prepared.parameters) == {"paid_media_" + suffix for suffix in _SUFFIXES}
    assert jax.tree_util.tree_leaves(prepared) == []
    with pytest.raises(FrozenInstanceError):
        spec.max_lag = 2
    attribute = fields(prepared)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(prepared, attribute, getattr(prepared, attribute))


@pytest.mark.parametrize("adstock_first,normalize", [(True, True), (False, True), (True, False), (False, False)])
def test_component_matches_finite_lag_hill_reference_with_full_history(adstock_first, normalize):
    prepared = MediaEffect(max_lag=3, name="paid", adstock_first=adstock_first, normalize=normalize)._prepare(
        _data(_MEDIA)
    )
    parameters = _parameters()
    actual = jax.jit(lambda component, media, values: component.apply(media, values))(
        prepared, jnp.asarray(_MEDIA), parameters
    )
    expected = _reference(_MEDIA, parameters, n_periods=3, adstock_first=adstock_first, normalize=normalize)
    without_history = _reference(_MEDIA[-3:], parameters, n_periods=3, adstock_first=adstock_first, normalize=normalize)

    assert actual.shape == (3, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    assert not np.allclose(expected[0], without_history[0])


@pytest.mark.parametrize("group_specific", [False, True])
def test_only_coefficients_become_group_specific(group_specific):
    prepared = MediaEffect(max_lag=3, name="paid", group_specific=group_specific)._prepare(_data(_MEDIA))
    parameters = _parameters(grouped=group_specific)
    for suffix in _SUFFIXES:
        declaration = prepared.parameters["paid_" + suffix]
        assert declaration.shape == ((2, 2) if suffix == "coefficient" and group_specific else (2,))
        assert isinstance(declaration, Interval if suffix == "retention" else Positive)
    retention = prepared.parameters["paid_retention"]
    assert retention.lower == 0 and retention.upper == 1
    np.testing.assert_allclose(
        prepared.apply(_MEDIA, parameters), _reference(_MEDIA, parameters, n_periods=3), rtol=4e-6, atol=2e-6
    )


def test_national_response_and_zero_lag_keep_channel_contributions_separate():
    media = _MEDIA[:, 0]
    prepared = MediaEffect(max_lag=0, name="paid")._prepare(_data(media))
    parameters = _parameters()
    response = prepared.apply(media, parameters)

    assert response.shape == (3, 2)
    np.testing.assert_allclose(response, _reference(media, parameters, n_periods=3, max_lag=0), rtol=4e-6, atol=2e-6)


def test_component_gradients_match_finite_differences_and_depend_on_historical_media():
    prepared = MediaEffect(max_lag=3, name="paid")._prepare(_data(_MEDIA))
    parameters = _parameters()

    def objective(media, values):
        return prepared.apply(media, values).sum()

    media_gradient, gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(jnp.asarray(_MEDIA), parameters)
    values64 = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}
    for name, values in values64.items():
        expected = []
        for delta in np.eye(len(values)) * 1e-4:
            plus = _reference(_MEDIA, {**values64, name: values + delta}, n_periods=3).sum()
            minus = _reference(_MEDIA, {**values64, name: values - delta}, n_periods=3).sum()
            expected.append((plus - minus) / 2e-4)
        assert np.all(np.abs(expected) > 1e-5)
        np.testing.assert_allclose(gradients[name], expected, rtol=3e-5, atol=3e-6)
    shift = np.zeros_like(_MEDIA)
    shift[0, 0, 0] = 1e-4
    expected_history = (
        _reference(_MEDIA + shift, values64, n_periods=3).sum()
        - _reference(_MEDIA - shift, values64, n_periods=3).sum()
    ) / 2e-4
    assert expected_history > 0
    np.testing.assert_allclose(media_gradient[0, 0, 0], expected_history, rtol=3e-5, atol=3e-6)


def test_vmap_uses_dynamic_media_and_parameter_draws():
    prepared = MediaEffect(max_lag=3, name="paid")._prepare(_data(_MEDIA))
    draws = jax.tree.map(lambda value: jnp.stack((value, value * 1.1)), _parameters())
    apply = jax.jit(jax.vmap(lambda component, media, values: component.apply(media, values), in_axes=(None, None, 0)))
    for multiplier in (1.0, 1.7):
        actual = apply(prepared, jnp.asarray(_MEDIA * multiplier), draws)
        assert actual.shape == (2, 3, 2, 2)
        for index in range(2):
            expected = _reference(
                _MEDIA * multiplier, {name: value[index] for name, value in draws.items()}, n_periods=3
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


def test_default_priors_have_correct_normalization_and_do_not_scale_with_observations():
    prepared = MediaEffect(max_lag=3, name="paid", group_specific=True)._prepare(_data(_MEDIA))
    parameters = _parameters(grouped=True)
    expected = 0.0
    for suffix in _SUFFIXES:
        value = np.asarray(parameters["paid_" + suffix], dtype=np.float64)
        expected += (
            (np.log(3) + 2 * np.log1p(-value)).sum()
            if suffix == "retention"
            else (np.log(2) - np.log(1.5) - 0.5 * np.log(2 * np.pi) - 0.5 * (value / 1.5) ** 2).sum()
        )
    value, gradient = jax.jit(jax.value_and_grad(lambda values, component: component.log_prior(values)))(
        parameters, prepared
    )
    assert value.shape == ()
    np.testing.assert_allclose(value, expected, rtol=3e-6, atol=2e-6)
    for suffix in _SUFFIXES:
        parameter = np.asarray(parameters["paid_" + suffix], dtype=np.float64)
        expected_gradient = -2 / (1 - parameter) if suffix == "retention" else -parameter / 1.5**2
        np.testing.assert_allclose(gradient["paid_" + suffix], expected_gradient, rtol=4e-6, atol=2e-6)
    future = prepared.for_data(_data(_MEDIA[:4], n_periods=1, start=6, observed=False))
    np.testing.assert_array_equal(future.log_prior(parameters), prepared.log_prior(parameters))


def test_custom_prior_overrides_accept_scalar_or_elementwise_results():
    def scalar_prior(value):
        return (-0.5 * value**2).sum()

    def elementwise_prior(value):
        return -0.5 * value**2

    spec = MediaEffect(
        max_lag=2,
        name="paid",
        coefficient_prior=scalar_prior,
        retention_prior=elementwise_prior,
        half_saturation_prior=scalar_prior,
        slope_prior=elementwise_prior,
    )
    prepared, parameters = spec._prepare(_data(_MEDIA)), _parameters()
    expected = sum(-0.5 * np.sum(np.asarray(value, dtype=np.float64) ** 2) for value in parameters.values())
    np.testing.assert_allclose(
        jax.jit(lambda component, values: component.log_prior(values))(prepared, parameters), expected, rtol=3e-6
    )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_disabled_automatic_priors_preserve_media_constraints_and_response(enable_x64):
    data = _data(_MEDIA)
    spec = MediaEffect(max_lag=3, name="paid", group_specific=True)
    with jax.enable_x64(enable_x64):
        default = spec._prepare(data)
        manual = replace(spec, automatic_priors=False)._prepare(data)
        parameters = _parameters(grouped=True)
        value, gradient = jax.jit(jax.value_and_grad(manual.log_prior))(parameters)

        assert value.shape == () and value.dtype == manual.dtype
        np.testing.assert_array_equal(value, 0.0)
        for name, declaration in manual.parameters.items():
            assert type(declaration) is type(default.parameters[name])
            assert declaration.shape == default.parameters[name].shape
            assert declaration.dtype == default.parameters[name].dtype
            np.testing.assert_array_equal(gradient[name], np.zeros_like(parameters[name]))
        np.testing.assert_array_equal(manual.apply(_MEDIA, parameters), default.apply(_MEDIA, parameters))
        future = manual.for_data(_data(_MEDIA[:2], n_periods=2, start=6, observed=False))
        np.testing.assert_array_equal(future.log_prior(parameters), value)


def test_disabled_automatic_priors_reject_all_media_prior_callbacks():
    for suffix in _ALL_SUFFIXES:
        with pytest.raises(ValueError, match=rf"{suffix}_prior.*automatic_priors=False"):
            MediaEffect(max_lag=3, automatic_priors=False, **{suffix + "_prior": lambda value: -value})


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", np.bool_(False)])
def test_media_automatic_priors_requires_a_python_boolean(invalid):
    with pytest.raises(TypeError, match="automatic_priors"):
        MediaEffect(max_lag=3, automatic_priors=invalid)


def test_forecasts_require_alignment_and_reuse_coefficients_with_changed_history_and_periods():
    training = _data(_MEDIA)
    with jax.enable_x64(False):
        prepared = MediaEffect(max_lag=3, name="paid", group_specific=True)._prepare(training)
        parameters = _parameters(grouped=True)
    for media, periods in ((_MEDIA[:4] * 0.7, 1), (_MEDIA[:2] * 1.2, 2)):
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


def test_configuration_validates_required_lag_names_booleans_and_priors():
    with pytest.raises(TypeError, match="max_lag"):
        MediaEffect()
    for lag in (True, 1.5, np.int64(2)):
        with pytest.raises(TypeError, match="max_lag"):
            MediaEffect(max_lag=lag)
    with pytest.raises(ValueError, match="max_lag"):
        MediaEffect(max_lag=-1)
    for option in ("normalize", "adstock_first", "group_specific"):
        with pytest.raises(TypeError, match=option):
            MediaEffect(max_lag=2, **{option: 1})
    for name in ("", "not-valid", "class"):
        with pytest.raises(ValueError, match="name"):
            MediaEffect(max_lag=2, name=name)
    for suffix in _SUFFIXES:
        with pytest.raises(TypeError, match="prior"):
            MediaEffect(max_lag=2, **{suffix + "_prior": 0.5})


def test_preparation_requires_media_and_consistent_forecast_identities():
    prepared = MediaEffect(max_lag=2, name="paid")._prepare(_data(_MEDIA))
    with pytest.raises(TypeError, match=r"data|PreparedData"):
        MediaEffect(max_lag=2)._prepare({"media": _MEDIA})
    missing = prepare_data(pl.DataFrame({"time": [0, 1], "sales": [0.3, 0.7]}), time="time", outcome="sales")
    with pytest.raises(ValueError, match="media"):
        MediaEffect(max_lag=2)._prepare(missing)
    with pytest.raises(ValueError, match=r"group_specific|group"):
        MediaEffect(max_lag=2, group_specific=True)._prepare(_data(_MEDIA[:, 0]))
    future = _data(_MEDIA, start=6, observed=False)
    for invalid in (
        replace(future, time_column="new_time"),
        replace(future, group_columns=("country",)),
        replace(future, group_values=(("west",), ("central",))),
        replace(future, channels=("other", "search")),
        replace(future, columns={"media": ("other", "search")}),
    ):
        with pytest.raises(ValueError):
            prepared.for_data(invalid)


@pytest.mark.parametrize("automatic_priors", [False, True])
def test_apply_and_prior_validate_media_and_parameter_shapes(automatic_priors):
    prepared = MediaEffect(max_lag=2, name="paid", automatic_priors=automatic_priors)._prepare(_data(_MEDIA))
    parameters = _parameters()
    for shape in ((), (5, 2), (5, 3, 2), (5, 2, 3), (2, 2, 2), (6, 2, 2)):
        with pytest.raises(ValueError, match=r"media|shape|period"):
            prepared.apply(jnp.ones(shape), parameters)
    for suffix in _SUFFIXES:
        malformed = {**parameters, "paid_" + suffix: jnp.ones((2, 1))}
        missing = {name: value for name, value in parameters.items() if name != "paid_" + suffix}
        for values in (malformed, missing):
            for operation in (prepared.log_prior, lambda values: prepared.apply(_MEDIA, values)):
                with pytest.raises(ValueError, match=r"paid_|parameter|shape|missing"):
                    operation(values)
    with pytest.raises(TypeError, match="mapping"):
        prepared.log_prior(jnp.ones(2))
    with pytest.raises(TypeError, match="real numeric"):
        prepared.log_prior({**parameters, "paid_coefficient": jnp.ones(2, dtype=jnp.complex64)})
    invalid_prior = MediaEffect(max_lag=2, name="paid", coefficient_prior=lambda value: jnp.zeros(3))._prepare(
        _data(_MEDIA)
    )
    with pytest.raises(ValueError, match=r"prior|shape"):
        invalid_prior.log_prior(parameters)


@pytest.mark.parametrize("adstock", _ADSTOCK_ROLES, ids=lambda function: function.__name__)
@pytest.mark.parametrize("saturation", _SATURATION_ROLES, ids=lambda function: function.__name__)
def test_selectors_declare_only_active_constraints_and_normalized_default_priors(adstock, saturation):
    prepared = MediaEffect(
        max_lag=3, name="paid", adstock=adstock, saturation=saturation, group_specific=True
    )._prepare(_data(_MEDIA))
    parameters = _selected_parameters(adstock, saturation, grouped=True)
    assert set(prepared.parameters) == set(parameters)
    expected_prior = 0.0
    for name, declaration in prepared.parameters.items():
        role = name.removeprefix("paid_")
        value = np.asarray(parameters[name], dtype=np.float64)
        assert declaration.shape == ((2, 2) if role == "coefficient" else (2,))
        if role in ("retention", "delay", "exponent"):
            assert isinstance(declaration, Interval)
            assert declaration.lower == 0.0
            assert declaration.upper == (3.0 if role == "delay" else 1.0)
        else:
            assert isinstance(declaration, Positive)
        if role == "retention":
            expected_prior += (np.log(3) + 2 * np.log1p(-value)).sum()
        elif role == "delay":
            expected_prior -= value.size * np.log(3)
        elif role != "exponent":
            expected_prior += (np.log(2) - np.log(1.5) - 0.5 * np.log(2 * np.pi) - 0.5 * (value / 1.5) ** 2).sum()
    np.testing.assert_allclose(prepared.log_prior(parameters), expected_prior, rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("adstock,saturation,adstock_first", _SELECTION_CASES)
def test_selected_transforms_compose_with_history_groups_and_finite_jit_gradients(adstock, saturation, adstock_first):
    prepared = MediaEffect(
        max_lag=3,
        name="paid",
        adstock=adstock,
        saturation=saturation,
        adstock_first=adstock_first,
        normalize=adstock_first,
        group_specific=adstock_first,
    )._prepare(_data(_MEDIA))
    parameters = _selected_parameters(adstock, saturation, grouped=adstock_first)

    def objective(media, values):
        response = prepared.apply(media, values)
        return response.sum(), response

    (_, actual), (media_gradient, parameter_gradients) = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    )(jnp.asarray(_MEDIA), parameters)
    options = {"adstock": adstock, "saturation": saturation, "normalize": adstock_first, "adstock_first": adstock_first}
    expected = _selected_reference(_MEDIA, parameters, n_periods=3, **options)
    without_history = _selected_reference(_MEDIA[-3:], parameters, n_periods=3, **options)
    assert actual.shape == (3, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    assert not np.allclose(actual[0], without_history[0])
    assert np.any(np.abs(media_gradient[:2]) > 1e-6)
    for gradient in jax.tree.leaves((media_gradient, parameter_gradients)):
        assert np.all(np.isfinite(gradient))
    inactive_values = {
        "paid_" + role: jnp.full((2,), jnp.nan) for role in _ALL_SUFFIXES if "paid_" + role not in parameters
    }
    np.testing.assert_allclose(prepared.apply(_MEDIA, parameters | inactive_values), actual, rtol=4e-6, atol=2e-6)
    np.testing.assert_array_equal(prepared.log_prior(parameters | inactive_values), prepared.log_prior(parameters))


@pytest.mark.parametrize(
    "adstock,saturation", [(delayed_adstock, root_saturation), (weibull_cdf_adstock, log_saturation)]
)
def test_selected_parameter_gradients_match_finite_differences_and_vmap_draws(adstock, saturation):
    media = _MEDIA[:, 0]
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
def test_delayed_zero_lag_uses_fixed_zero_delay_without_declaring_a_delay_parameter(normalize):
    media = _MEDIA[:, 0]
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
    with pytest.raises(ValueError, match="delay_prior"):
        MediaEffect(max_lag=0, adstock=delayed_adstock, delay_prior=lambda value: -value)


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


@pytest.mark.parametrize(
    "adstock,saturation", [(delayed_adstock, root_saturation), (weibull_pdf_adstock, logistic_saturation)]
)
def test_selected_custom_priors_override_every_active_role(adstock, saturation):
    parameters = _selected_parameters(adstock, saturation)
    priors = {
        name.removeprefix("paid_") + "_prior": (lambda value: (-0.5 * value**2).sum())
        if name.endswith("coefficient")
        else (lambda value: -0.5 * value**2)
        for name in parameters
    }
    prepared = MediaEffect(max_lag=3, name="paid", adstock=adstock, saturation=saturation, **priors)._prepare(
        _data(_MEDIA)
    )
    expected = sum(-0.5 * np.sum(np.asarray(value, dtype=np.float64) ** 2) for value in parameters.values())
    np.testing.assert_allclose(prepared.log_prior(parameters), expected, rtol=3e-6)
    for name in priors:
        with pytest.raises(TypeError, match=name):
            MediaEffect(max_lag=3, adstock=adstock, saturation=saturation, **{name: 0.5})


@pytest.mark.parametrize("adstock", _ADSTOCK_ROLES, ids=lambda function: function.__name__)
@pytest.mark.parametrize("saturation", _SATURATION_ROLES, ids=lambda function: function.__name__)
def test_unused_prior_overrides_are_rejected_at_construction(adstock, saturation):
    active = {"coefficient", *_ADSTOCK_ROLES[adstock], *_SATURATION_ROLES[saturation]}
    for role in set(_ALL_SUFFIXES) - active:
        prior_name = role + "_prior"
        with pytest.raises(ValueError, match=prior_name):
            MediaEffect(max_lag=3, adstock=adstock, saturation=saturation, **{prior_name: lambda value: -value})


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
