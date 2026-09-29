# API Reference

Every function and class in the reference imports from `mmmjax`, and the
distribution functions also import from `mmmjax.distributions`. The first five
sections cover the work around a model, from building it through sampling,
analysis, and plots. The last four hold the functions your blocks call.

| Section | Description |
| --- | --- |
| [Model](model) | The model class and the declarations that give each parameter its shape and constraint |
| [Data](data.rst) | Preparing, checking, scaling, and simulating the data a model reads |
| [Sampling](sampling) | Prior draws, NUTS fits, generated quantities, and prior sensitivity |
| [Response and optimization](response) | Contributions, returns, response curves, and budget optimization from a fit |
| [Plotting](plotting) | Plots of the fit, its diagnostics, the media transformations, and each analysis |
| [Media effects](media) | Adstock and saturation functions and the media response they compose |
| [Seasonality and trends](baselines) | Fourier features for seasons and approximate Gaussian processes for smooth or time-varying effects |
| [Calibration](calibration) | Media coefficients solved from a return on spending or a contribution |
| [Distributions](distributions) | Log densities, draws, and cumulative functions for each family, and your own families through `custom_distribution` |

```{toctree}
:hidden:
:maxdepth: 1

model
data
sampling
response
plotting
media
baselines
calibration
distributions
```

<p class="mmmj-footer-logo">
  <img src="../_static/mmmjax-logo.svg" alt="mmmJAX logo">
</p>
