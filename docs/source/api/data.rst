Data preparation
================

Prepare dataframe inputs with time, group, and column labels intact.
Use ``prepare_data`` to select and validate observations, then
``PreparedData.to_jax`` to convert the numerical blocks for modeling.

.. currentmodule:: mmmjax

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_data
   PreparedData
