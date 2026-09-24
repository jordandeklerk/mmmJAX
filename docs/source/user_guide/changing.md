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

```{code-cell} ipython3
:tags: [remove-cell]

import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = ".33"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
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
first impressions do little until exposure builds. The new model gives each
channel its own slope $s_c \geq 1$, so its model equation becomes

$$
\begin{aligned}
y_t &= \alpha + \sum_{c} \beta_c \operatorname{Hill}\Big(\operatorname{Adstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c\big);\, \kappa_c, s_c\Big) + \gamma\, p_t + \varepsilon_t, \\
s_c - 1 &\sim \operatorname{HalfNormal}(1),
\end{aligned}
$$

with the data transformations and the other priors of [A first
model](first_model). The prior puts the slope at one or above and lets the
data move it up.

```{code-cell} ipython3
shaped_parameters = parameters | {"slope": mj.LowerBound(1.0, dims="channel")}


def shaped_transformed_parameters(
    media,
    controls,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    slope,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=slope)
    mu = intercept + saturated @ coefficient + controls @ control_coefficient
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
1.3, close to the value the simulation used, although its 90 percent interval
runs from about 1.05 to 1.75, so the data narrows the slope without pinning it
down. Search's stays close to its prior at about
1.7, against the simulation's slope of one, because search never goes dark and
the data rarely shows the low end of its curve, where the slope matters most.
A slope below one would make the curve infinitely steep at zero exposure, and
with TV off the air in two thirds of the weeks, the gradient the sampler
follows would not be finite, which is why the declaration sets a lower bound
of one.

## The likelihood

A Student-t likelihood has heavier tails than the normal, so a few unusual
weeks pull less on the fit. Its degrees of freedom $\nu$ set how heavy the
tails are, and a gamma prior with shape two and rate 0.1 puts most of its
mass between a few and about fifty, which covers both heavy tails and
nearly normal ones. Only the noise changes,

$$
\varepsilon_t \sim \operatorname{StudentT}(\nu, 0, \sigma), \qquad \nu \sim \operatorname{Gamma}(2, 0.1),
$$

where the gamma distribution takes a shape and a rate, and the model equation
and every other prior stay as in [A first model](first_model). Swapping it in
adds a gamma prior line for $\nu$, replaces the likelihood on the last line of
the density, and changes the two lines of the generated quantities that
simulate from it and score it.

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
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
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

Revenue often follows the calendar. Fourier terms $\sin(2\pi k d_t / 365.25)$
and $\cos(2\pi k d_t / 365.25)$ for $k = 1, 2$, computed from the day of the
year $d_t$ of week $t$, give the baseline a yearly pattern with coefficients
$a_k$ and $b_k$, so the model equation becomes

$$
\begin{aligned}
y_t &= \alpha + \sum_{k=1}^{2} \Big( a_k \sin\frac{2\pi k d_t}{365.25} + b_k \cos\frac{2\pi k d_t}{365.25} \Big) \\
&\quad + \sum_{c} \beta_c \operatorname{HillAdstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c, \kappa_c\big) + \gamma\, p_t + \varepsilon_t, \\
a_k, b_k &\sim \operatorname{Normal}(0, 0.5),
\end{aligned}
$$

with everything else as in [A first model](first_model).
Since the Fourier terms depend only on the dates, they belong in
`transformed_data`, which runs once for each dataset instead of at every step
of the sampler.

```{code-cell} ipython3
def transformed_data(day_of_year):
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)
    return {"annual": annual}


def seasonal_transformed_parameters(
    media,
    controls,
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
    baseline = intercept + annual @ annual_coefficients
    mu = baseline + saturated @ coefficient + controls @ control_coefficient
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
then all cosines. All four coefficients stay within 0.1 of zero, which is right,
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
comparison = az.compare(versions, round_to=1)
print(comparison[["elpd", "elpd_diff", "dse"]])
```

`elpd` is the expected log predictive density, where higher is better, and
`dse` is the standard error of each version's difference from the best one. No
version stands out. The first model comes out ahead, the free slope and
seasonal versions trail it by less than a third of a standard error of the
difference, and the Student-t trails by 0.8, which is two standard errors but
still a small gap. That matches how the data
was made, with a TV slope of 1.3 that the data can barely tell from one, noise
without heavy tails, and no season. None of this says which model is true, and a change the data can't
detect may still move the answers you care about. A forest plot puts the four
posteriors side by side.

```{code-cell} ipython3
pc = az.plot_forest(
    versions,
    var_names=["coefficient", "half_saturation"],
    combined=True,
    ci_probs=(0.5, 0.9),
    figure_kwargs={"figsize": (12, 7)},
)
pc.add_legend("model")
plt.show()
```

Given a dictionary of results, ArviZ draws one line per version, with a circle
at the posterior mean and thick and thin bars for the 50 and 90 percent
intervals, and `combined=True` pools each version's four chains. The Student-t
and seasonal versions stay close to the first model. Freeing the slope moves
TV's coefficient from about 4.5 to 3.7, TV's half-saturation from 1.2 to 0.8,
and Search's coefficient from 2.1 to 1.5. A slope above one also changes what
the coefficient and half-saturation mean, so the answers to compare come from
[Media effects](media_effects) and [Budget optimization](budgets), run on each
version.

```{code-cell} ipython3
free = mj.contributions(shaped_model, shaped, quantity="mu")
free["incremental_response"].median(("chain", "draw")).to_series().round(-3)
```

Freeing the slope brings the model's structure closer to the simulation's, yet
both medians sit further from the truth than the first model's in [Recovering
the truth](recovery), about \$1.21
million for TV and \$1.19 million for search against true values of \$1.36
million and \$1.93 million. Search's slope settles near its prior instead of
the true one, so a structure closer to the truth did not give answers closer to
it here.
