import mmmjax as mj

brand = mj.simulate_data(seed=7, groups=None)
channels = {
    "meta": "Meta",
    "tiktok": "TikTok",
    "snapchat": "Snapchat",
    "youtube": "YouTube",
    "streaming": "Streaming",
    "linear_tv": "Linear TV",
    "branded_search": "Branded search",
    "generic_search": "Generic search",
    "influencer": "Influencer",
    "display": "Display",
}
data = mj.prepare_data(
    brand.frame,
    time="week",
    outcome="revenue",
    media=[f"{name}_impressions" for name in channels],
    spend=[f"{name}_spend" for name in channels],
    channels=list(channels.values()),
    controls=["demand", "price", "promotion", "holiday", "email_sends"],
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)

parameters = {
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    "annual_coefficients": mj.Real(dims="annual_mode"),
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "control_coefficient": mj.Real(dims="control"),
    "sigma": mj.Positive(),
}

priors = {
    "intercept": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "growth": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "curvature": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "annual_coefficients": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "roi": mj.Prior(mj.lognormal, location=1.0, scale=0.6),
    "retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "sigma": mj.Prior(mj.half_normal, scale=1.0),
}


def transformed_data(day_of_year, time, reference):
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)
    trend = time / reference.time.max()
    return {"annual": annual, "trend": trend}


def hill_adstock(media, retention, half_saturation):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    return saturated


def transformed_parameters(
    media,
    controls,
    annual,
    trend,
    reference,
    outcome_scaling,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    control_coefficient,
):
    trained = hill_adstock(reference.media, retention, half_saturation)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    mu = baseline + media_effect + controls @ control_coefficient
    return {"mu": mu}


def log_density(
    outcome,
    mu,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
):
    target = priors["intercept"](intercept)
    target += priors["growth"](growth)
    target += priors["curvature"](curvature)
    target += priors["annual_coefficients"](annual_coefficients)
    target += priors["roi"](roi)
    target += priors["retention"](retention)
    target += priors["half_saturation"](half_saturation)
    target += priors["control_coefficient"](control_coefficient)
    target += priors["sigma"](sigma)
    target += mj.normal(outcome, mu, sigma)
    return target


def generated_quantities(key, outcome, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
    }


model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
    coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
)
