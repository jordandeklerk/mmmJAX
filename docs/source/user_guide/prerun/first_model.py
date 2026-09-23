import mmmjax as mj

example = mj.simulate_data(seed=7, groups=None, complexity="simple", noise_scale=0.02)
data = mj.prepare_data(
    example.frame,
    time="week",
    outcome="revenue",
    media=["linear_tv_impressions", "generic_search_impressions"],
    spend=["linear_tv_spend", "generic_search_spend"],
    channels=["TV", "Search"],
    controls=["price"],
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)

parameters = {
    "intercept": mj.Real(),
    "coefficient": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "control_coefficient": mj.Real(dims="control"),
    "sigma": mj.Positive(),
}


def transformed_parameters(
    media,
    controls,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    mu = intercept + saturated @ coefficient + controls @ control_coefficient
    return {"mu": mu}


def log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.half_normal(coefficient, 2.0)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.half_normal(sigma, 1.0)
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
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
