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
    organic_media=["email_sends"],
    organic_channels=["Email"],
    controls=["demand", "holiday"],
    treatments=["price", "promotion"],
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)

parameters = {
    # The baseline's level, the trend's growth and curvature, and the season.
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    # No data axis runs over the four Fourier weights, and they need no labels, so a plain shape does.
    "annual_coefficients": mj.Real(4),
    # Each paid channel's return, carryover, and half-saturation point.
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    # Email's share of revenue, carryover, and half-saturation point.
    "organic_share": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_retention": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_half_saturation": mj.Positive(dims="organic_channel"),
    # One coefficient for each control and each treatment, and the noise scale.
    "control_coefficient": mj.Real(dims="control"),
    "treatment_coefficient": mj.Real(dims="treatment"),
    "sigma": mj.Positive(),
}


def transformed_data(day_of_year, time):
    # Nothing here depends on a parameter, so this block runs once per dataset, not at every draw.
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)

    # Years keep trend and trend**2 on the scale the growth and curvature priors assume,
    # and time keeps counting past the training weeks, so a forecast extends the trend.
    trend = time / 365.25

    return {"annual": annual, "trend": trend}


# mmmJAX fills only a block's arguments, so media here is whatever the caller passes.
def hill_adstock(media, retention, half_saturation):
    # Eight weeks is the longest you believe exposure keeps working. It sets an array shape,
    # so it stays a fixed int rather than a parameter the sampler learns.
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)

    # The slope stays at one, so the curve rises fastest at the first impression.
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    return saturated


def transformed_parameters(
    media,
    organic_media,
    controls,
    treatments,
    reference,
    outcome_scaling,
    annual,
    trend,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
):
    # Paid media's coefficients come from the returns. reference holds the training data in every
    # scenario, so the coefficients keep their fitted values when an analysis changes media. The
    # prior sits on ROI because revenue per dollar is easier to judge than a scaled coefficient.
    trained = hill_adstock(reference.media, retention, half_saturation)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)

    # With no spend, email has no ROI, so its prior sits on its share of revenue. The share
    # is of dollars, since standardized revenue sums to zero over the training weeks.
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )

    # The baseline follows the trend and the season.
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients

    # Each effect reads the inputs the model is given, so a scenario that changes media,
    # sends, or prices changes it.
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient

    # The likelihood and every analysis read the expected revenue in each week.
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
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
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
    sigma,
):
    # The baseline's priors describe standardized revenue. The curvature's scale is smaller
    # because trend**2 outgrows trend after the first year.
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)

    # Paid media's returns, carryover, and saturation.
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)

    # You expect email to be small, and Beta(2, 98) puts its mean share of revenue at 2 percent.
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)

    # The treatments get a tighter prior than the controls. The price climbs with the trend,
    # and a wider prior would let the fit credit price with growth.
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)

    # The noise scale's prior and the likelihood, normal noise around the expected revenue.
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target


# The prior tools read this mapping and log_density never does, so keep the two in step.
priors = {
    "intercept": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "growth": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "curvature": mj.Prior(mj.normal, location=0.0, scale=0.25),
    "annual_coefficients": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "roi": mj.Prior(mj.lognormal, location=1.0, scale=0.6),
    "retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "organic_share": mj.Prior(mj.beta, alpha=2.0, beta=98.0),
    "organic_retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "organic_half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "treatment_coefficient": mj.Prior(mj.normal, location=0.0, scale=0.25),
    "sigma": mj.Prior(mj.half_normal, scale=1.0),
}


def generated_quantities(key, outcome, mu, sigma):
    # Simulated revenue for every week, and each observed week's log likelihood.
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)

    # "predictive" and "log_likelihood" are supplied names that tell mmmJAX where to store
    # the draws, and mmmJAX names both "outcome" so ArviZ pairs them with the observed data.
    return {"predictive": prediction, "log_likelihood": pointwise}


model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
