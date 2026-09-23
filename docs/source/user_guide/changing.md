---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Changing the model

Because the model is a set of functions you wrote, changing it means editing
those functions, and nothing else in the workflow has to move. This page makes
three changes to the model from [A first model](first_model), to its response
curve, its likelihood, and its baseline, and then compares the four versions.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)
```

Each version below is fitted with the same settings as the first model.

```{code-cell} ipython3
settings = {"draws": 1000, "warmup": 1000, "chains": 4, "seed": 7}
```

## The response curve

The first model fixes the Hill curve's slope at one, which makes every
channel's response concave from the first impression. The general curve is

$$
h(x) = \frac{x^{s}}{x^{s} + \kappa^{s}},
$$

and letting the slope $s$ rise above one allows an S-shaped curve, where the
first impressions do little until exposure builds.

```{code-cell} ipython3
shaped_parameters = parameters | {"slope": mj.LowerBound(1.0, dims="channel")}


def shaped_transformed_parameters(
    media,
    controls,
    n_periods,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    slope,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=slope)
    recent = saturated[-n_periods:]
    mu = intercept + recent @ coefficient + controls @ control_coefficient
    return {"mu": mu}


def shaped_log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
    slope,
):
    target = log_density(
        outcome, mu, intercept, coefficient, retention, half_saturation, control_coefficient, sigma
    )
    target += mj.half_normal(slope - 1.0, 1.0)
    return target


shaped_model = mj.Model(
    parameters=shaped_parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_parameters=shaped_transformed_parameters,
    log_density=shaped_log_density,
    generated_quantities=generated_quantities,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

shaped = mj.sample(shaped_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import stored

shaped = stored("shaped", lambda: mj.sample(shaped_model, **settings), groups=["posterior", "log_likelihood"])
```

```{code-cell} ipython3
shaped["posterior"]["slope"].mean(("chain", "draw")).to_series().round(2)
```

The new density calls the first model's density and adds one prior term, since
blocks are ordinary Python functions. The posterior puts TV's slope at about
1.4, close to the 1.3 the simulation used. Search's stays close to its prior
at about 1.8, against the simulation's slope of one, because search never goes
dark and the data rarely shows the low end of its curve, where the slope
matters most. A slope below one would make the curve infinitely steep at zero
exposure, and with TV off the air in two thirds of the weeks, the gradient the
sampler follows would not be finite, which is why the declaration sets a lower
bound of one.

## The likelihood

A Student-t likelihood has heavier tails than the normal, so a few unusual
weeks pull less on the fit. Swapping it in changes the last line of the
density and the two lines of the generated quantities that simulate from it
and score it.

```{code-cell} ipython3
student_parameters = parameters | {"degrees_of_freedom": mj.Positive()}


def student_log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
    degrees_of_freedom,
):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.half_normal(coefficient, 2.0)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.half_normal(sigma, 1.0)
    target += mj.gamma(degrees_of_freedom, 2.0, 0.1)
    target += mj.student_t(outcome, degrees_of_freedom, mu, sigma)
    return target


def student_generated_quantities(key, outcome, mu, sigma, degrees_of_freedom):
    prediction = mj.student_t_rng(key, degrees_of_freedom, mu, sigma)
    pointwise = mj.student_t_logpdf(outcome, degrees_of_freedom, mu, sigma)
    return {
        "predictive": {"prediction": prediction},
        "log_likelihood": {"pointwise": pointwise},
    }


student_model = mj.Model(
    parameters=student_parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_parameters=transformed_parameters,
    log_density=student_log_density,
    generated_quantities=student_generated_quantities,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

student = mj.sample(student_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

student = stored("student", lambda: mj.sample(student_model, **settings), groups=["posterior", "log_likelihood"])
```

```{code-cell} ipython3
round(float(student["posterior"]["degrees_of_freedom"].mean()), 1)
```

With degrees of freedom around 29, the Student-t is close to a normal
distribution, so the data shows little sign of the outliers that heavier tails
would absorb.

## A seasonal baseline

Revenue often follows the calendar. Fourier terms $\sin(2\pi k d / 365.25)$
and $\cos(2\pi k d / 365.25)$ for $k = 1, 2$, computed from the day of the
year $d$, give the baseline a yearly pattern. Since they depend only on the
dates, they belong in `transformed_data`, which runs once for each dataset
instead of at every step of the sampler.

```{code-cell} ipython3
def transformed_data(day_of_year):
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)
    return {"annual": annual}


def seasonal_transformed_parameters(
    media,
    controls,
    n_periods,
    annual,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    annual_coefficients,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    recent = saturated[-n_periods:]
    baseline = intercept + annual @ annual_coefficients
    mu = baseline + recent @ coefficient + controls @ control_coefficient
    return {"mu": mu}


def seasonal_log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
    annual_coefficients,
):
    target = log_density(
        outcome, mu, intercept, coefficient, retention, half_saturation, control_coefficient, sigma
    )
    target += mj.normal(annual_coefficients, 0.0, 0.5)
    return target


seasonal_model = mj.Model(
    parameters=parameters | {"annual_coefficients": mj.Real(dims="annual_mode")},
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=seasonal_transformed_parameters,
    log_density=seasonal_log_density,
    generated_quantities=generated_quantities,
    coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
)
```

```{code-cell} ipython3
:tags: [skip-execution]

seasonal = mj.sample(seasonal_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

seasonal = stored(
    "seasonal", lambda: mj.sample(seasonal_model, **settings), groups=["posterior", "log_likelihood"]
)
```

```{code-cell} ipython3
seasonal["posterior"]["annual_coefficients"].mean(("chain", "draw")).to_series().round(2)
```

`annual` reaches `transformed_parameters` by name, the same way the data and
the parameters do. The new parameter's axis isn't one the data defines, so
`coords` labels it, following the order of the Fourier features, all sines and
then all cosines. All four coefficients sit close to zero, which is right,
since the simulation has no seasonal pattern. Because `day_of_year` comes from
the dates, a pattern like this follows the calendar in any scenario, including
one that starts in a different month from the training data.

## Comparing the versions

Leave-one-out cross-validation estimates how well each version predicts a
week it hasn't seen, using the pointwise log likelihood that every version's
generated quantities return.

```{code-cell} ipython3
import arviz as az

versions = {"first": results, "free slope": shaped, "student-t": student, "seasonal": seasonal}
az.compare(versions, round_to=1)[["elpd", "elpd_diff", "dse"]]
```

`elpd` is the expected log predictive density, where higher is better, and
`dse` is the standard error of each version's difference from the best one. No
version stands out. The free slope comes out ahead of the first model by about
a third of a standard error of the difference, and the Student-t and seasonal
versions fall behind by one standard error or less. That matches how the data
was made, with a TV slope close to one, noise without heavy tails, and no
season. None of this says which model is true, and a change the data can't
detect may still move the answers you care about, so it is worth running the
analyses from [Media effects and budgets](media_effects) on each version.
