Distributions
=============

Stan-style distributions built on TensorFlow Probability's JAX backend. Each
family provides a summed log density for model blocks, a pointwise log
density, random draws, and log cumulative and log survival functions where
they exist. Discrete families also offer log or logit parameterizations, and
distributions written as plain functions join through ``custom_distribution``.

.. currentmodule:: mmmjax

User-defined
------------

Write a pointwise log density and a draw function as plain JAX functions and
register them once. The returned summed density works in ``log_density``, and
``Prior`` accepts it like a built-in family.

.. autosummary::
   :toctree: generated
   :nosignatures:

   custom_distribution

Continuous
----------

Normal
^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   normal
   normal_logpdf
   normal_rng
   normal_logcdf
   normal_logsf

LogNormal
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   lognormal
   lognormal_logpdf
   lognormal_rng
   lognormal_logcdf
   lognormal_logsf

Truncated Normal
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   truncated_normal
   truncated_normal_logpdf
   truncated_normal_rng
   truncated_normal_logcdf
   truncated_normal_logsf

HalfNormal
^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   half_normal
   half_normal_logpdf
   half_normal_rng
   half_normal_logcdf
   half_normal_logsf

Student-t
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   student_t
   student_t_logpdf
   student_t_rng

Cauchy
^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   cauchy
   cauchy_logpdf
   cauchy_rng
   cauchy_logcdf
   cauchy_logsf

Laplace
^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   laplace
   laplace_logpdf
   laplace_rng
   laplace_logcdf
   laplace_logsf

Exponential
^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   exponential
   exponential_logpdf
   exponential_rng
   exponential_logcdf
   exponential_logsf

Gamma
^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   gamma
   gamma_logpdf
   gamma_rng
   gamma_logcdf
   gamma_logsf

Inverse Gamma
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   inverse_gamma
   inverse_gamma_logpdf
   inverse_gamma_rng
   inverse_gamma_logcdf
   inverse_gamma_logsf

Beta
^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   beta
   beta_logpdf
   beta_rng

Uniform
^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   uniform
   uniform_logpdf
   uniform_rng
   uniform_logcdf
   uniform_logsf

Discrete
--------

Bernoulli
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   bernoulli
   bernoulli_logpmf
   bernoulli_rng
   bernoulli_logcdf
   bernoulli_logsf
   bernoulli_logit
   bernoulli_logit_logpmf
   bernoulli_logit_rng
   bernoulli_logit_logcdf
   bernoulli_logit_logsf

Binomial
^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   binomial
   binomial_logpmf
   binomial_rng
   binomial_logcdf
   binomial_logsf
   binomial_logit
   binomial_logit_logpmf
   binomial_logit_rng
   binomial_logit_logcdf
   binomial_logit_logsf

Poisson
^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   poisson
   poisson_logpmf
   poisson_rng
   poisson_logcdf
   poisson_logsf
   poisson_log
   poisson_log_logpmf
   poisson_log_rng
   poisson_log_logcdf
   poisson_log_logsf

Negative Binomial
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   negative_binomial
   negative_binomial_logpmf
   negative_binomial_logcdf
   negative_binomial_logsf
   negative_binomial_rng
   negative_binomial_log
   negative_binomial_log_logpmf
   negative_binomial_log_logcdf
   negative_binomial_log_logsf
   negative_binomial_log_rng

Categorical
^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   categorical
   categorical_logpmf
   categorical_rng
   categorical_logit
   categorical_logit_logpmf
   categorical_logit_rng

Multivariate
------------

Dirichlet
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   dirichlet
   dirichlet_logpdf
   dirichlet_rng

Multinomial
^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   multinomial
   multinomial_logpmf
   multinomial_rng
   multinomial_logit
   multinomial_logit_logpmf
   multinomial_logit_rng

Multivariate Normal
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   multivariate_normal
   multivariate_normal_logpdf
   multivariate_normal_rng

LKJ correlation factors
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   lkj_cholesky
   lkj_cholesky_logpdf
   lkj_cholesky_rng
