Data preparation
================

Prepare dataframe inputs with time, group, and column labels intact.
Use ``prepare_data`` to select and validate the outcome, media, spend,
and controls while retaining observation and channel labels.

Use ``media`` for exposure measurements such as impressions and ``spend``
for their associated costs. Columns are paired in the order supplied. Optional
``channels`` names label their shared channel axis and default to the
media column names. Select the same columns for media and spend when
the model uses spending as its media input.

Supply earlier media observations with ``media_history`` to retain exposure
before the first modeling period. This dataframe needs only the selected time,
group, and media columns. Outcomes, controls, and spend retain their original
periods. Preparation stores the history without applying adstock.

.. currentmodule:: mmmjax

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_data
