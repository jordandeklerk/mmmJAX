---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quickstart

This model runs end to end on simulated weekly data for a brand with ten paid
channels. If you want something quick to iterate on and play with, copy it, run
it, and change whatever you like. The [User Guide](../user_guide/index) goes
through this model in much more detail.

## Model

:::{admonition} Which names you can change
:class: important

Supplied names such as `media`, `outcome`, and `reference` come from mmmJAX and
keep their exact spelling, apart from the random key, which arrives first by
position. The rest, such as the keys of `parameters` and the `annual`, `trend`, and `mu`
the blocks return, are your names. A new name can't repeat a supplied one, and
it has to change everywhere it appears, in the blocks, in `priors`, and in
`quantity="mu"` below.
[What is mmmJAX](what_is_mmmjax.md#how-blocks-get-their-inputs) explains the
difference, and [Data and scaling](../user_guide/data.md#supplied-names) lists
the supplied names.
[From math to code](../user_guide/first_model.md#from-math-to-code) traces the
arguments in these blocks to the symbols in the model's math.
:::

```{code-cell} ipython3
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
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    # No data axis runs over the four Fourier weights, and they need no labels, so a plain shape does.
    "annual_coefficients": mj.Real(4),
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "organic_share": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_retention": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_half_saturation": mj.Positive(dims="organic_channel"),
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
    # reference holds the training data in every scenario, so the coefficients
    # keep their fitted values when an analysis changes media.
    trained = hill_adstock(reference.media, retention, half_saturation)
    # The prior sits on ROI because revenue per dollar is easier to judge than a scaled coefficient.
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    # With no spend, email has no ROI, so its prior sits on its share of revenue. The share
    # is of dollars, since standardized revenue sums to zero over the training weeks.
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
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
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    # You expect email to be small, and Beta(2, 98) puts its mean share of revenue at 2 percent.
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    # The price climbs with the trend. A wider prior would let the fit credit price with growth.
    target += mj.normal(treatment_coefficient, 0.0, 0.25)
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
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    # "predictive" and "log_likelihood" are supplied names that tell mmmJAX where to store
    # the draws, and "outcome" matches the observed data because ArviZ pairs them by name.
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
)
```

## Sampling

```{code-cell} ipython3
:tags: [skip-execution]

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
```

## Analysis

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import first_model_results

results = first_model_results(model)
```

```{code-cell} ipython3
:tags: [remove-output]

import arviz as az

az.summary(results)
mj.plot_fit(model, results)
```

```{code-cell} ipython3
:tags: [remove-output]

# The analyses compare the mean that transformed_parameters returns as mu.
curves = mj.response_curves(model, results, quantity="mu")
plan = mj.optimize_budget(model, results, quantity="mu")
mj.plot_response_curves(curves, plan=plan)
```
