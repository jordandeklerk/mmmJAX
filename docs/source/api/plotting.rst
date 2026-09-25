Plotting
========

Plot a fitted model and the outputs of the analysis functions. The plots built
on plotnine return a ``ggplot`` that later layers, scales, and ``theme``
settings can change. The diagnostics built on ArviZ return its
``PlotCollection``, and extra keywords go to the ArviZ function. Intervals and
point estimates follow ArviZ's ``stats.ci_prob``, ``stats.ci_kind``, and
``stats.point_estimate`` settings unless ``ci_prob`` is given.

.. currentmodule:: mmmjax

Model fit and convergence
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_fit
   plot_residuals
   plot_ppc_dist
   plot_prior_posterior
   plot_rhat
   plot_rank
   plot_trace_dist

Media effects
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_roi
   plot_media_metrics
   plot_response_curves
   plot_frequency_curves
   plot_adstock

Budget allocation
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_budget_response
   plot_budget_spend

Style
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   theme_mmmjax
