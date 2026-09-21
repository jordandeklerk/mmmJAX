Response and optimization
=========================

Evaluate media scenarios, returns, and contributions from prior or posterior
draws, and allocate spending under explicit constraints.
Each function re-evaluates ``transformed_parameters`` under changed inputs,
restores original outcome units with the fitted scaling, and reads any
data-based normalization from ``reference`` when the blocks are written that
way.

.. currentmodule:: mmmjax

Response
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   response_curves
   frequency_curves
   media_metrics
   contributions

Budget optimization
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   optimize_budget
   SpendConstraint
