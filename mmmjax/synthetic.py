"""Synthetic marketing observations with a separate record of their generating process."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from mmmjax.adstock import geometric_adstock
from mmmjax.saturation import hill_saturation

__all__ = ["SyntheticData", "simulate_data"]


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticData:
    """Store synthetic observations, channel descriptions, and known effects.

    Attributes
    ----------
    frame : pandas.DataFrame
        Weekly observations with separate exposure and spend columns.
        Includes revenue, population, demand, price, promotions, and holidays.
    media_history : pandas.DataFrame
        Eight preceding weeks of exposures for carryover calculations.
        Pass this to ``prepare_data`` with the observations.
    channels : pandas.DataFrame
        Channel families, platforms, input column names, and illustrative
        generation settings. Email is organic and has no spend column.
    truth : xarray.Dataset
        Labeled parameters, true exposures, response curves, contributions,
        expected revenue, and observation noise. These are simulation truth,
        not posterior estimates or additional modeling inputs.
    """

    frame: pd.DataFrame
    media_history: pd.DataFrame
    channels: pd.DataFrame
    truth: xr.Dataset


def simulate_data(
    seed: int = 0,
    n_periods: int = 156,
    groups: Sequence[str] | None = ("north", "south", "west"),
    start: str | date = "2022-01-03",
    campaign_overlap: float = 0.7,
    noise_scale: float = 0.05,
    measurement_error: float = 0.0,
) -> SyntheticData:
    r"""Generate a fictional consumer brand's weekly marketing data.

    Ten paid channels and owned email combine always-on activity, campaign
    flights, and a later channel launch. Demand, holidays, and promotions
    influence both media execution and revenue. Costs and seasonal patterns
    change over time. Defaults are illustrative, not platform benchmarks.

    Revenue has conditional mean :math:`\mu` equal to baseline, seasonal,
    demand, price, promotion, holiday, and channel contributions. Each channel
    applies normalized geometric carryover and Hill saturation to exposures
    per person. Observation noise is mean-one lognormal multiplicative noise
    with standard deviation ``noise_scale`` relative to :math:`\mu`.

    The data is complete and nonnegative where required.
    Zero exposures mean inactivity, not missing reports.
    Measurement error changes reported exposures without changing true effects.

    Parameters
    ----------
    seed : int, default 0
        Nonnegative seed for reproducible observations and generating parameters.
    n_periods : int, default 156
        Number of weekly modeling periods, excluding the eight-week lead-in.
    groups : sequence of str or None, optional
        Unique region labels. Defaults to three regions. Use ``None`` for
        a single series without a region column or group dimension.
    start : str or datetime.date, default "2022-01-03"
        First modeling date, as an ISO date or date object. Subsequent
        observations are spaced seven days apart.
    campaign_overlap : float, default 0.7
        Probability that a channel in a region follows the shared campaign
        calendar. Zero uses independent calendars and one uses shared timing.
    noise_scale : float, default 0.05
        Nonnegative observation noise relative to expected revenue.
        Zero returns the conditional mean without observation noise.
    measurement_error : float, default 0.0
        Nonnegative relative noise in reported exposures. True zero activity
        remains zero. Spend and the underlying response remain unchanged.

    Returns
    -------
    SyntheticData
        Simulation containing the following fields.

        - **frame** — Observed weekly data for preparation and modeling
        - **media_history** — Earlier exposure observations for carryover
        - **channels** — Channel labels, column selections, and settings
        - **truth** — Parameters, effects, and full-window paid-channel ROI

        ROI removes one channel throughout the lead-in and modeling window,
        holds other inputs fixed, and divides the modeling-window revenue
        difference by that window's spend. Zero spend gives undefined ROI.
        It is not net-profit ROI. Some channels may be inactive in short windows.

    Examples
    --------
    Generate observations and select paid and organic exposure columns.
    Keep the true effects aside when preparing inputs for a model.

    .. ipython::

        In [1]: from mmmjax import prepare_data, simulate_data
           ...: example = simulate_data(seed=7, n_periods=104)
           ...: paid = example.channels.query("kind == 'paid'")
           ...: organic = example.channels.query("kind == 'organic'")
           ...: data = prepare_data(
           ...:     example.frame, time="week", groups=["region"],
           ...:     outcome="revenue", population="population",
           ...:     media=paid["exposure_column"].to_list(),
           ...:     spend=paid["spend_column"].to_list(),
           ...:     channels=paid["channel"].to_list(),
           ...:     organic_media=organic["exposure_column"].to_list(),
           ...:     organic_channels=organic["channel"].to_list(),
           ...:     controls=["demand", "holiday"],
           ...:     treatments=["price", "promotion"],
           ...:     media_history=example.media_history,
           ...: )
           ...: example.frame[["week", "region", "revenue"]].head()
    """
    first_date, group_names = _validate_inputs(
        seed, n_periods, groups, start, campaign_overlap, noise_scale, measurement_error
    )
    channels = _channel_catalog()
    channel_names = channels["channel"].to_list()
    paid_names = channels.loc[channels["kind"] == "paid", "channel"].to_list()
    n_groups, n_channels = len(group_names), len(channel_names)
    max_lag = 8
    periods = np.arange(-max_lag, n_periods)
    dates = [first_date + timedelta(weeks=int(period)) for period in periods]
    day = np.array([value.timetuple().tm_yday for value in dates])
    angle = 2 * np.pi * day / 365.25
    # Distinct streams keep measurement and revenue noise from changing the latent process.
    driver_rng, media_rng, response_rng, noise_rng, measurement_rng = (
        np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(5)
    )
    population = driver_rng.integers(100_000, 500_001, n_groups)
    demand = np.exp(0.2 * _persistent_noise(driver_rng, (len(dates), n_groups)) + 0.12 * np.sin(angle)[:, None])
    holiday = (np.exp(-0.5 * ((day - 330) / 10) ** 2) + np.exp(-0.5 * ((day - 355) / 6) ** 2))[:, None]
    holiday = np.broadcast_to(holiday, demand.shape).copy()
    campaign = _campaigns(driver_rng, len(dates), n_groups)
    promotion = ((campaign > 0) | (holiday > 0.5)).astype(float)
    price = 20 * (1 + 0.04 * np.arange(len(dates))[:, None] / 52) * (1 - 0.15 * promotion)
    price = np.broadcast_to(price, demand.shape).copy()

    independent = np.stack([_campaigns(media_rng, len(dates), n_groups) for _ in channel_names], axis=-1)
    shared = media_rng.uniform(size=(n_groups, n_channels)) < campaign_overlap
    channel_campaign = np.where(shared[None, ...], campaign[..., None], independent)
    always_on = (channels["activity"] == "always_on").to_numpy()
    activity = np.where(always_on, 0.5 + channel_campaign, channel_campaign)
    # A newly introduced platform has no execution before its regional launch.
    launch = 26 + np.arange(n_groups) * 2
    activity[..., channel_names.index("snapchat")] *= periods[:, None] >= launch
    budget = np.exp(0.16 * _persistent_noise(media_rng, (len(dates), n_groups, 1)))
    allocation = media_rng.lognormal(0, 0.15, size=(n_groups, n_channels))
    execution = activity * budget * allocation * demand[..., None] ** 0.6 * (1 + 0.4 * holiday[..., None])
    execution *= np.exp(0.08 * _persistent_noise(media_rng, execution.shape))
    # Search execution responds more strongly to demand even at the same campaign intensity.
    for channel in ("branded_search", "generic_search"):
        execution[..., channel_names.index(channel)] *= demand**0.7

    cpm = channels["cpm"].to_numpy()[:-1] * np.exp(
        0.08 * _persistent_noise(media_rng, (len(dates), n_groups, len(paid_names)))
        + 0.25 * holiday[..., None]
        + 0.03 * np.arange(len(dates))[:, None, None] / 52
    )
    spend = execution[..., :-1] * population[None, :, None] * channels["spend_per_person"].to_numpy()[:-1]
    exposure = np.empty_like(execution)
    exposure[..., :-1] = 1000 * spend / cpm
    exposure[..., -1] = execution[..., -1] * population[None, :] * 0.15
    observed_exposure = exposure * _multiplicative_noise(measurement_rng, exposure.shape, measurement_error)

    retention = channels["retention"].to_numpy()
    half_saturation = channels["half_saturation"].to_numpy()
    slope = channels["slope"].to_numpy()
    coefficient = np.asarray(population[:, None] * channels["coefficient_per_person"].to_numpy(), dtype=np.float64)
    coefficient *= response_rng.lognormal(0, 0.15, size=(n_groups, n_channels))
    carried = geometric_adstock(exposure / population[None, :, None], retention, max_lag=max_lag)
    response = np.asarray(hill_saturation(carried, half_saturation, slope))[max_lag:]
    contribution = response * coefficient

    regional_baseline = population * response_rng.uniform(0.8, 1.2, n_groups)
    baseline = np.broadcast_to(regional_baseline, demand.shape).copy()
    seasonal_wave = (0.08 + 0.02 * np.sin(2 * np.pi * periods / (3 * 52))) * np.sin(angle)
    seasonal_wave += 0.04 * np.cos(2 * angle) + 0.0005 * periods
    seasonality = seasonal_wave[:, None] * population
    demand_effect = 0.25 * population * (demand - 1)
    price_effect = -0.015 * population * (price - 20)
    promotion_effect = 0.06 * population * promotion
    holiday_effect = 0.12 * population * holiday
    baseline_terms = {
        "baseline": baseline[max_lag:],
        "seasonality": seasonality[max_lag:],
        "demand_effect": demand_effect[max_lag:],
        "price_effect": price_effect[max_lag:],
        "promotion_effect": promotion_effect[max_lag:],
        "holiday_effect": holiday_effect[max_lag:],
    }
    expected = sum(baseline_terms.values()) + contribution.sum(axis=-1)
    revenue = expected * _multiplicative_noise(noise_rng, expected.shape, noise_scale)
    observed_spend = spend[max_lag:].sum(axis=0)
    roi = np.divide(
        contribution[..., :-1].sum(axis=0),
        observed_spend,
        out=np.full_like(observed_spend, np.nan),
        where=observed_spend > 0,
    )

    columns = {name: observed_exposure[..., index] for index, name in enumerate(channels["exposure_column"])}
    columns.update({f"{name}_spend": spend[..., index] for index, name in enumerate(paid_names)})
    columns.update(
        {
            "population": np.broadcast_to(population, demand.shape),
            "demand": demand,
            "price": price,
            "promotion": promotion,
            "holiday": holiday,
        }
    )
    frame = _frame(dates[max_lag:], group_names, {name: values[max_lag:] for name, values in columns.items()}, groups)
    frame["revenue"] = revenue.ravel()
    history = _frame(
        dates[:max_lag],
        group_names,
        {name: observed_exposure[:max_lag, :, index] for index, name in enumerate(channels["exposure_column"])},
        groups,
    )

    truth = xr.Dataset(
        data_vars={
            "exposure": (("media_time", "group", "channel"), exposure),
            "observed_exposure": (("media_time", "group", "channel"), observed_exposure),
            "spend": (("media_time", "group", "paid_channel"), spend),
            "cpm": (("media_time", "group", "paid_channel"), cpm),
            "campaign": (("media_time", "group"), campaign),
            "response": (("time", "group", "channel"), response),
            "contribution": (("time", "group", "channel"), contribution),
            "coefficient": (("group", "channel"), coefficient),
            "retention": ("channel", retention),
            "half_saturation": ("channel", half_saturation),
            "slope": ("channel", slope),
            "population": ("group", population),
            "roi": (("group", "paid_channel"), roi),
            **{name: (("time", "group"), values) for name, values in baseline_terms.items()},
            "expected_revenue": (("time", "group"), expected),
            "noise": (("time", "group"), revenue - expected),
            "revenue": (("time", "group"), revenue),
        },
        coords={
            "time": np.array(dates[max_lag:], dtype="datetime64[D]"),
            "media_time": np.array(dates, dtype="datetime64[D]"),
            "group": list(group_names),
            "channel": channel_names,
            "paid_channel": paid_names,
            "family": ("channel", channels["family"].to_list()),
            "platform": ("channel", channels["platform"].to_list()),
            "tactic": ("channel", channels["tactic"].to_list()),
        },
        attrs={
            "seed": seed,
            "max_lag": max_lag,
            "frequency": "weekly",
            "start": first_date.isoformat(),
            "campaign_overlap": campaign_overlap,
            "noise_scale": noise_scale,
            "measurement_error": measurement_error,
            "scenario_version": "1",
            "adstock": "geometric",
            "saturation": "hill",
            "normalize": "true",
            "description": "Illustrative consumer brand, not calibrated platform performance",
            "roi_intervention": "Remove one channel over all media_time, hold other inputs fixed, measure over time",
        },
    )
    truth["half_saturation"].attrs["units"] = "exposures per person"
    truth["coefficient"].attrs["units"] = "revenue"
    truth["contribution"].attrs["units"] = "revenue"
    truth["roi"].attrs["units"] = "incremental revenue per unit spend"
    if groups is None:
        truth = truth.squeeze("group", drop=True)
    return SyntheticData(frame, history, channels, truth)


def _channel_catalog() -> pd.DataFrame:
    """Describe a fictional channel mix and its explicit per-person response settings."""
    # All costs and response settings describe this scenario, not measured platform effects.
    channels = pd.DataFrame(
        {
            "channel": [
                "meta",
                "tiktok",
                "snapchat",
                "youtube",
                "streaming",
                "linear_tv",
                "branded_search",
                "generic_search",
                "influencer",
                "display",
                "email",
            ],
            "family": [
                "social",
                "social",
                "social",
                "video",
                "video",
                "television",
                "search",
                "search",
                "influencer",
                "display",
                "email",
            ],
            "platform": [
                "meta",
                "tiktok",
                "snapchat",
                "youtube",
                "streaming",
                "linear_tv",
                "search",
                "search",
                "creators",
                "programmatic",
                "owned",
            ],
            "tactic": ["paid_social"] * 3
            + [
                "online_video",
                "streaming_video",
                "linear_tv",
                "brand",
                "nonbrand",
                "sponsorship",
                "display",
                "newsletter",
            ],
            "kind": ["paid"] * 10 + ["organic"],
            "activity": [
                "always_on",
                "flights",
                "flights",
                "flights",
                "flights",
                "flights",
                "always_on",
                "always_on",
                "flights",
                "always_on",
                "flights",
            ],
            "unit": ["impressions"] * 10 + ["sends"],
            "spend_per_person": [0.014, 0.009, 0.006, 0.012, 0.020, 0.030, 0.010, 0.014, 0.008, 0.008, 0.0],
            "cpm": [8.0, 7.0, 6.0, 12.0, 18.0, 20.0, 25.0, 18.0, 15.0, 5.0, None],
            "retention": [0.3, 0.25, 0.2, 0.5, 0.6, 0.7, 0.1, 0.15, 0.4, 0.25, 0.1],
            "half_saturation": [1.2, 0.8, 0.7, 0.8, 0.8, 1.0, 0.3, 0.6, 0.4, 1.1, 0.08],
            "slope": [1.2, 1.1, 1.3, 1.2, 1.1, 1.3, 0.9, 1.0, 1.2, 1.0, 1.0],
            "coefficient_per_person": [0.12, 0.08, 0.06, 0.12, 0.11, 0.15, 0.08, 0.10, 0.06, 0.06, 0.05],
        },
    )
    paid = channels["kind"] == "paid"
    channels["exposure_column"] = channels["channel"] + np.where(paid, "_impressions", "_sends")
    channels["spend_column"] = (channels["channel"] + "_spend").where(paid, None)
    return channels


def _persistent_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Draw a stationary unit-variance AR(1) process for smooth planning variation."""
    values = rng.normal(size=shape)
    for period in range(1, shape[0]):
        values[period] = 0.8 * values[period - 1] + np.sqrt(1 - 0.8**2) * values[period]
    return values


def _campaigns(rng: np.random.Generator, n_periods: int, n_groups: int) -> NDArray[np.float64]:
    """Build quarterly flights with small regional timing differences and inactive gaps."""
    values = np.zeros((n_periods, n_groups))
    for planned in range(-6, n_periods, 13):
        start = planned + int(rng.integers(-2, 3))
        length = int(rng.integers(3, 7))
        strength = rng.uniform(0.8, 1.4)
        for group in range(n_groups):
            regional_start = start + int(rng.integers(0, 3))
            stop = min(n_periods, regional_start + length)
            if stop > max(0, regional_start):
                values[max(0, regional_start) : stop, group] = strength
    return values


def _multiplicative_noise(
    rng: np.random.Generator, shape: tuple[int, ...], relative_scale: float
) -> NDArray[np.float64]:
    """Draw mean-one lognormal noise with the requested coefficient of variation."""
    log_variance = np.logaddexp(0.0, 2 * np.log(relative_scale)) if relative_scale > 0 else 0.0
    scale = np.sqrt(log_variance)
    return np.asarray(rng.lognormal(-0.5 * scale**2, scale, size=shape))


def _frame(
    dates: Sequence[date],
    group_names: tuple[str, ...],
    columns: dict[str, NDArray[np.float64]],
    groups: Sequence[str] | None,
) -> pd.DataFrame:
    """Create a time-major dataframe in the same order as the labeled tensors."""
    frame = pd.DataFrame({"week": [value for value in dates for _ in group_names]})
    if groups is not None:
        frame["region"] = list(group_names) * len(dates)
    return frame.assign(**{name: values.ravel() for name, values in columns.items()})


def _validate_inputs(
    seed: int,
    n_periods: int,
    groups: Sequence[str] | None,
    start: str | date,
    campaign_overlap: float,
    noise_scale: float,
    measurement_error: float,
) -> tuple[date, tuple[str, ...]]:
    """Check simulation sizes, dates, and settings before allocating arrays."""
    for name, value, minimum in (("seed", seed, 0), ("n_periods", n_periods, 1)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be a Python integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if isinstance(start, str):
        try:
            first_date = date.fromisoformat(start)
        except ValueError as error:
            raise ValueError("start must be an ISO date in YYYY-MM-DD format") from error
        if first_date.isoformat() != start:
            raise ValueError("start must be an ISO date in YYYY-MM-DD format")
    elif isinstance(start, date) and not isinstance(start, datetime):
        first_date = start
    else:
        raise TypeError("start must be an ISO date string or a date without a time component")
    try:
        first_date - timedelta(weeks=8)
        first_date + timedelta(weeks=n_periods - 1)
    except OverflowError as error:
        raise ValueError("start and n_periods must leave room for the exposure history and modeling dates") from error
    if groups is None:
        group_names: tuple[str, ...] = ("national",)
    else:
        if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
            raise TypeError("groups must be a sequence of unique region names or None")
        group_names = tuple(groups)
        if any(not isinstance(name, str) for name in group_names):
            raise TypeError("groups must contain only string names")
        if not group_names or any(not name.strip() for name in group_names):
            raise ValueError("groups must contain nonempty string names")
        if len(set(group_names)) != len(group_names):
            raise ValueError("groups must contain unique region names")
    for name, scale in (
        ("campaign_overlap", campaign_overlap),
        ("noise_scale", noise_scale),
        ("measurement_error", measurement_error),
    ):
        if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be a finite nonnegative number")
        if not np.isfinite(scale) or scale < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if campaign_overlap > 1:
        raise ValueError("campaign_overlap must be between zero and one")
    return first_date, group_names
