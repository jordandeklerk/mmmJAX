"""Tests for analyses of models whose inputs carry declared scalings."""

import jax
import numpy as np
import polars as pl
import pytest
import xarray as xr

from mmmjax import (
    Data,
    Model,
    Real,
    Scaling,
    contributions,
    frequency_curves,
    media_metrics,
    optimize_budget,
    prepare_data,
    response_curves,
)
from mmmjax.data._results import _collect_results


def _tolerances():
    # Marginal effects subtract totals near a thousand, and float32 rounds such totals by about 1e-4
    single = jax.dtypes.canonicalize_dtype(float) == np.float32
    tolerances = {"rtol": 2e-4, "atol": 2e-3} if single else {"rtol": 1e-9, "atol": 1e-9}
    return tolerances


def _frame(weeks, video, search):
    rows = []
    for week, video_exposure, search_exposure in zip(weeks, video, search, strict=True):
        for region, multiplier in (("east", 1.0), ("west", 2.5)):
            rows.append(
                {
                    "week": week,
                    "region": region,
                    "video": video_exposure * multiplier,
                    "search": search_exposure * multiplier,
                    "video_cost": video_exposure * multiplier / 2,
                    "search_cost": search_exposure * multiplier / 4,
                    "sales": 100.0 + 5 * week + 30 * (region == "west"),
                }
            )
    return pl.DataFrame(rows)


def _prepare(frame, first_week):
    return prepare_data(
        frame.filter(pl.col("week") > first_week),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Video", "Search"],
        media_history=frame.filter(pl.col("week") == first_week),
    )


def _data():
    # Video has no spend in week 2 and search none in week 3
    return _prepare(_frame(range(5), [6.0, 4.0, 0.0, 3.0, 5.0], [2.0, 3.0, 5.0, 0.0, 4.0]), 0)


def _new_data():
    return _prepare(_frame(range(4, 8), [3.0, 8.0, 2.0, 0.0], [4.0, 1.0, 0.0, 6.0]), 4)


def _response(media, coefficient):
    carried = media[1:] + 0.5 * media[:-1]
    return 1.0 + carried @ coefficient + 0.4 * carried[..., 0] * carried[..., 1]


def _models(data):
    maximum = data.arrays["media"].max(axis=0)
    offset = np.array([100.0, 130.0])
    scale = np.array([20.0, 50.0])

    def declared_block(media, coefficient):
        return {"mu": _response(media, coefficient)}

    def inline_block(media, reference, coefficient):
        scaled = media / reference.media.max(axis=0)
        return {"mu": _response(scaled, coefficient) * scale + offset}

    parameters = {"coefficient": Real(dims="channel")}
    declared = Model(
        parameters=parameters,
        log_density=lambda mu: 0.0,
        data=Data(
            data, scaling={"media": Scaling(offset=0.0, scale=maximum), "outcome": Scaling(offset=offset, scale=scale)}
        ),
        transformed_parameters=declared_block,
    )
    inline = Model(
        parameters=parameters,
        log_density=lambda mu: 0.0,
        data=Data(data),
        transformed_parameters=inline_block,
    )
    return declared, inline


def _results(data):
    draws = np.array([[[0.8, 1.5], [1.2, 0.6], [2.0, 1.1]], [[0.5, 0.9], [1.6, 1.3], [0.9, 2.2]]], dtype=np.float32)
    return _collect_results(
        {"coefficient": draws},
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [4, 8], "draw": [10, 20, 30]},
    )


@pytest.mark.parametrize("new_data", [False, True], ids=["training", "new_data"])
@pytest.mark.parametrize(
    "analysis",
    [contributions, response_curves, media_metrics, optimize_budget],
    ids=["contributions", "response_curves", "media_metrics", "optimize_budget"],
)
def test_analyses_of_a_declared_media_scaling_match_the_scaling_written_in_the_blocks(analysis, new_data):
    data = _data()
    declared, inline = _models(data)
    results = _results(data)
    options = {"quantity": "mu", "new_data": _new_data() if new_data else None}
    expected = analysis(inline, results, **options)

    result = analysis(declared, results, **options)

    assert declared.data.outcome_scaling.scale.shape == (2,)
    xr.testing.assert_allclose(result, expected, **_tolerances())


def test_frequency_curves_of_a_declared_reach_scaling_match_the_scaling_written_in_the_blocks():
    frame = pl.DataFrame(
        {
            "week": [0, 1, 2, 3, 4],
            "audience": [3.0, 4.0, 6.0, 0.0, 5.0],
            "frequency": [1.0, 2.0, 1.5, 1.0, 2.5],
            "audience_cost": [2.0, 3.0, 4.0, 0.0, 3.5],
            "sales": [10.0, 12.0, 15.0, 9.0, 14.0],
        }
    )
    data = prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        outcome="sales",
        reach=["audience"],
        media_frequency=["frequency"],
        rf_spend=["audience_cost"],
        rf_channels=["Video"],
        media_history=frame.filter(pl.col("week") == 0),
    )

    def declared_block(reach, media_frequency, coefficient):
        exposure = (reach * media_frequency / (1.0 + media_frequency))[:, 0]
        return {"mu": 2.0 + coefficient * (exposure[1:] + 0.5 * exposure[:-1])}

    def inline_block(reach, media_frequency, reference, coefficient):
        scaled = reach / reference.reach.max(axis=0)
        return declared_block(scaled, media_frequency, coefficient)

    declared = Model(
        parameters={"coefficient": Real()},
        log_density=lambda mu: 0.0,
        data=Data(data, scaling={"reach": Scaling(offset=0.0, scale=data.arrays["reach"].max(axis=0))}),
        transformed_parameters=declared_block,
    )
    inline = Model(
        parameters={"coefficient": Real()},
        log_density=lambda mu: 0.0,
        data=Data(data),
        transformed_parameters=inline_block,
    )
    results = _collect_results({"coefficient": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)})
    options = {"quantity": "mu", "frequencies": [0.5, 1.0, 3.0]}
    expected = frequency_curves(inline, results, **options)

    result = frequency_curves(declared, results, **options)

    xr.testing.assert_allclose(result, expected, **_tolerances())
