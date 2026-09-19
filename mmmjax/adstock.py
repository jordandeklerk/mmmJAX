"""Carryover transformations for marketing inputs."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["delayed_adstock", "geometric_adstock", "weibull_cdf_adstock", "weibull_pdf_adstock"]


def geometric_adstock(
    media: ArrayLike,
    alpha: ArrayLike,
    *,
    max_lag: int,
    axis: int = 0,
    normalize: bool = True,
) -> jax.Array:
    r"""Apply finite geometric carryover along a time axis.

    For a series :math:`x_t`, retention :math:`0 \leq \alpha \leq 1`,
    and maximum lag :math:`L`, the transformed series is

    .. math::

        a_t = \frac{\sum_{\ell=0}^{L} \alpha^\ell x_{t-\ell}}{Z},
        \qquad
        Z = \begin{cases}
            \sum_{\ell=0}^{L} \alpha^\ell & \text{if normalized}, \\
            1 & \text{otherwise}.
        \end{cases}

    The lag-zero weight is one, including when :math:`\alpha = 0`.

    Media before the supplied series is assumed to be zero. When
    normalization is enabled, weights are normalized over the full lag
    window, even for short series.

    Parameters
    ----------
    media : array_like
        Real-valued array with at least one dimension. Observations along
        ``axis`` must represent equally spaced time periods. Other axes
        identify independent series, such as channels and geographies.
    alpha : array_like
        Finite retention parameter between zero and one, inclusive. Its
        shape must broadcast to the shape of ``media`` with the time axis
        removed. For ``media.shape == (time, geo, channel)``, channel-level
        parameters have shape ``(channel,)`` and geo-channel parameters
        have shape ``(geo, channel)``. Use ``jax.vmap`` for additional draws.
    max_lag : int
        Nonnegative number of previous periods to include. The window has
        ``max_lag + 1`` weights, including the current period.
    axis : int, default 0
        Time axis in ``media``. Defaults to the first axis.
    normalize : bool, default True
        Whether to divide the weights by their sum. Defaults to ``True``.
        ``max_lag``, ``axis``, and ``normalize`` must be static under JIT.

    Returns
    -------
    jax.Array
        Transformed values with the same shape as ``media``. Inputs are
        promoted to a common floating-point dtype of at least float32.
        Invalid retention parameters produce ``nan`` for their series.

    Examples
    --------
    Start with weekly spending for two channels, one row per week in time
    order.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import geometric_adstock

        In [2]: spend = pl.DataFrame({
           ...:     "week": [1, 2, 3, 4],
           ...:     "search": [100.0, 120.0, 80.0, 0.0],
           ...:     "social": [60.0, 0.0, 40.0, 80.0],
           ...: })

    Give each channel its own retention rate, in the same order as the
    selected columns. Search keeps half of its effect from one week to the
    next and social keeps less.

    .. ipython::

        In [3]: media = spend.select("search", "social").to_numpy()
           ...: carried = geometric_adstock(
           ...:     media,
           ...:     alpha=np.array([0.5, 0.3]),
           ...:     max_lag=2,
           ...: )

    The transformed columns sit beside the originals, so the carryover into
    weeks with no spending is easy to see.

    .. ipython::

        In [4]: spend.with_columns(
           ...:     pl.Series("search_adstock", np.asarray(carried[:, 0])),
           ...:     pl.Series("social_adstock", np.asarray(carried[:, 1])),
           ...: )
    """
    media_array, parameters = _prepare_adstock(media, max_lag=max_lag, axis=axis, normalize=normalize, alpha=alpha)
    alpha_array = parameters["alpha"]
    valid_alpha = jnp.isfinite(alpha_array) & (alpha_array >= 0) & (alpha_array <= 1)
    safe_alpha = jnp.where(valid_alpha, alpha_array, 0.0)

    # Products keep derivatives at zero retention well-defined without differentiating 0**0
    factors = jnp.concatenate(
        (
            jnp.ones((*alpha_array.shape, 1), dtype=media_array.dtype),
            jnp.broadcast_to(safe_alpha[..., None], (*alpha_array.shape, max_lag)),
        ),
        axis=-1,
    )
    weights = jnp.cumprod(factors, axis=-1)
    if normalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(valid_alpha[..., None], weights, jnp.nan)
    return _convolve_adstock(media_array, weights, axis=axis)


def delayed_adstock(
    media: ArrayLike,
    alpha: ArrayLike,
    theta: ArrayLike,
    *,
    max_lag: int,
    axis: int = 0,
    normalize: bool = True,
) -> jax.Array:
    r"""Apply finite carryover with a delayed peak along a time axis.

    For a series :math:`x_t`, retention :math:`0 < \alpha \leq 1`,
    peak delay :math:`0 \leq \theta \leq L`, and maximum lag :math:`L`,
    the transformed series is

    .. math::

        w_\ell = \alpha^{(\ell-\theta)^2}, \qquad
        a_t = \frac{\sum_{\ell=0}^{L} w_\ell x_{t-\ell}}{Z},
        \qquad
        Z = \begin{cases}
            \sum_{\ell=0}^{L} w_\ell & \text{if normalized}, \\
            1 & \text{otherwise}.
        \end{cases}

    The delay can be fractional. For :math:`\alpha < 1`, the sampled
    weights are largest at the lag or lags nearest :math:`\theta`;
    :math:`\alpha = 1` gives equal weights throughout the window.
    Unlike geometric adstock, setting :math:`\theta = 0` gives weights
    :math:`\alpha^{\ell^2}`, not :math:`\alpha^\ell`.

    Media before the supplied series is assumed to be zero. When
    normalization is enabled, weights are normalized over the full lag
    window, even for short series.

    Parameters
    ----------
    media : array_like
        Real-valued array with at least one dimension. Observations along
        ``axis`` must represent equally spaced periods. Other axes identify
        independent series, such as channels and geographies.
    alpha : array_like
        Finite retention parameter greater than zero and at most one.
        Smaller values concentrate the effect around ``theta``. Zero is
        excluded because all weights vanish for a fractional delay.
        Its shape must broadcast to the non-time shape of ``media``.
    theta : array_like
        Finite peak delay between zero and ``max_lag``, inclusive, measured
        in periods. Its shape must broadcast to the non-time shape of
        ``media`` independently of ``alpha``. For inputs shaped
        ``(time, geo, channel)``, parameters can have shape ``(channel,)``,
        ``(geo, 1)``, or ``(geo, channel)``. Use ``jax.vmap`` for extra draws.
    max_lag : int
        Nonnegative number of previous periods to include. The window has
        ``max_lag + 1`` weights, including the current period.
    axis : int, default 0
        Time axis in ``media``. Defaults to the first axis.
    normalize : bool, default True
        Whether to divide the weights by their sum. Defaults to ``True``.
        ``max_lag``, ``axis``, and ``normalize`` must be static under JIT.

    Returns
    -------
    jax.Array
        Transformed values with the same shape as ``media``. Inputs are
        promoted to a common floating-point dtype of at least float32.
        Invalid retention or delay parameters produce ``nan`` for their series.

    Examples
    --------
    Suppose a video campaign runs in the first of four consecutive weeks and
    nothing is spent afterwards.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import delayed_adstock

        In [2]: spend = pl.DataFrame({
           ...:     "week": [1, 2, 3, 4],
           ...:     "video": [100.0, 0.0, 0.0, 0.0],
           ...: })

    Let the effect build for two weeks before it peaks, then fade with a
    retention of one half.

    .. ipython::

        In [3]: carried = delayed_adstock(
           ...:     spend["video"].to_numpy(),
           ...:     alpha=0.5,
           ...:     theta=2.0,
           ...:     max_lag=3,
           ...: )

    Placed next to the original spending, the transformed column shows the
    delayed peak in week three.

    .. ipython::

        In [4]: spend.with_columns(
           ...:     pl.Series("video_adstock", np.asarray(carried)),
           ...: )
    """
    media_array, parameters = _prepare_adstock(
        media, max_lag=max_lag, axis=axis, normalize=normalize, alpha=alpha, theta=theta
    )
    alpha_array, theta_array = parameters["alpha"], parameters["theta"]
    valid = (
        jnp.isfinite(alpha_array)
        & (alpha_array > 0)
        & (alpha_array <= 1)
        & jnp.isfinite(theta_array)
        & (theta_array >= 0)
        & (theta_array <= max_lag)
    )
    safe_alpha = jnp.where(valid, alpha_array, 1.0)
    safe_theta = jnp.where(valid, theta_array, 0.0)

    lags = jnp.arange(max_lag + 1, dtype=media_array.dtype)
    weights = safe_alpha[..., None] ** jnp.square(lags - safe_theta[..., None])
    if normalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(valid[..., None], weights, jnp.nan)
    return _convolve_adstock(media_array, weights, axis=axis)


def weibull_pdf_adstock(
    media: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
    *,
    max_lag: int,
    axis: int = 0,
    normalize: bool = True,
) -> jax.Array:
    r"""Apply finite carryover using rescaled Weibull density weights.

    For shape :math:`k > 0`, scale :math:`\lambda > 0`, and lags
    :math:`\ell = 0, \ldots, L`, sample the Weibull density at
    :math:`\ell + 1` and rescale it over the full lag window:

    .. math::

        q_\ell = \frac{k}{\lambda}
            \left(\frac{\ell + 1}{\lambda}\right)^{k-1}
            \exp\!\left[-\left(\frac{\ell + 1}{\lambda}\right)^k\right],
        \qquad
        w_\ell = \frac{q_\ell - \min_j q_j}{\max_j q_j - \min_j q_j}.

    The transformed series is

    .. math::

        a_t = \frac{\sum_{\ell=0}^{L} w_\ell x_{t-\ell}}{Z},
        \qquad
        Z = \begin{cases}
            \sum_{\ell=0}^{L} w_\ell & \text{if normalized}, \\
            1 & \text{otherwise}.
        \end{cases}

    Sampling starts at one to avoid a singular density at zero for shapes
    below one. The min/max rescaling is applied even with ``normalize=False``;
    the weights are not raw density values. It is piecewise differentiable,
    with possible kinks where the sampled minimum or maximum changes.
    A flat sampled kernel has undefined min/max rescaling and produces
    ``nan``. Setting ``max_lag=0`` retains only the current period.

    Media before the supplied series is assumed to be zero. When
    normalization is enabled, weights are normalized over the full lag
    window, even for short series.

    Parameters
    ----------
    media : array_like
        Real-valued array with at least one dimension. Observations along
        ``axis`` must represent equally spaced periods. Other axes identify
        independent series, such as channels and geographies.
    shape : array_like
        Finite positive Weibull shape. Shapes at most one give decreasing
        density weights; larger shapes can give a delayed peak. Its shape
        must broadcast to the non-time shape of ``media``.
    scale : array_like
        Finite positive Weibull scale, measured in periods. Its shape must
        broadcast to the non-time shape of ``media`` independently of
        ``shape``. For ``(time, geo, channel)`` inputs, parameters can have
        shape ``(channel,)``, ``(geo, 1)``, or ``(geo, channel)``.
    max_lag : int
        Nonnegative number of previous periods to include. The window has
        ``max_lag + 1`` weights, including the current period.
    axis : int, default 0
        Time axis in ``media``.
    normalize : bool, default True
        Whether to divide the rescaled weights by their sum.
        ``max_lag``, ``axis``, and ``normalize`` must be static under JIT.

    Returns
    -------
    jax.Array
        Transformed values with the same shape as ``media``. Inputs are
        promoted to a common floating-point dtype of at least float32.
        Invalid shape or scale parameters produce ``nan`` for their series.

    Examples
    --------
    Start with four consecutive weeks of video spending, including a week
    with none.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import weibull_pdf_adstock

        In [2]: spend = pl.DataFrame({
           ...:     "week": [1, 2, 3, 4],
           ...:     "video": [100.0, 0.0, 80.0, 40.0],
           ...: })

    Weight the carryover with a Weibull density, so the effect rises after
    exposure and then tails off.

    .. ipython::

        In [3]: carried = weibull_pdf_adstock(
           ...:     spend["video"].to_numpy(),
           ...:     shape=2.0,
           ...:     scale=3.0,
           ...:     max_lag=3,
           ...: )

    Keep the original spending beside the transformed values for comparison.

    .. ipython::

        In [4]: spend.with_columns(
           ...:     pl.Series("video_adstock", np.asarray(carried)),
           ...: )
    """
    media_array, parameters = _prepare_adstock(
        media, max_lag=max_lag, axis=axis, normalize=normalize, shape=shape, scale=scale
    )
    shape_array, scale_array = parameters["shape"], parameters["scale"]
    valid = jnp.isfinite(shape_array) & (shape_array > 0) & jnp.isfinite(scale_array) & (scale_array > 0)
    safe_shape = jnp.where(valid, shape_array, 1.0)[..., None]
    safe_scale = jnp.where(valid, scale_array, 1.0)[..., None]

    if max_lag == 0:
        weights = jnp.ones_like(safe_shape)
    else:
        periods = jnp.arange(max_lag + 1, dtype=media_array.dtype) + 1
        log_ratio = jnp.log(periods) - jnp.log(safe_scale)
        # The common density factor k/scale cancels in min/max rescaling
        log_weights = (safe_shape - 1) * log_ratio - jnp.exp(safe_shape * log_ratio)
        # Max-shifting preserves relative weights when the sampled densities underflow
        weights = jnp.exp(log_weights - jnp.max(log_weights, axis=-1, keepdims=True))
        minimum = jnp.min(weights, axis=-1, keepdims=True)
        width = 1 - minimum
        weights = (weights - minimum) / width

    if normalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(valid[..., None], weights, jnp.nan)
    return _convolve_adstock(media_array, weights, axis=axis)


def weibull_cdf_adstock(
    media: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
    *,
    max_lag: int,
    axis: int = 0,
    normalize: bool = True,
) -> jax.Array:
    r"""Apply finite carryover using cumulative Weibull survival weights.

    For the Weibull CDF :math:`F`, shape :math:`k > 0`, scale
    :math:`\lambda > 0`, and maximum lag :math:`L`, define

    .. math::

        w_0 = 1, \qquad
        w_\ell = \prod_{i=1}^{\ell} [1 - F(i; k, \lambda)]
            = \exp\!\left[-\sum_{i=1}^{\ell}
                \left(\frac{i}{\lambda}\right)^k\right],
        \qquad \ell = 1, \ldots, L.

    The transformed series is

    .. math::

        a_t = \frac{\sum_{\ell=0}^{L} w_\ell x_{t-\ell}}{Z},
        \qquad
        Z = \begin{cases}
            \sum_{\ell=0}^{L} w_\ell & \text{if normalized}, \\
            1 & \text{otherwise}.
        \end{cases}

    The largest weight is at lag zero. These weights are cumulative products
    of survival probabilities, not individual CDF or survival values.
    Setting ``max_lag=0`` retains only the current period.

    Media before the supplied series is assumed to be zero. When
    normalization is enabled, weights are normalized over the full lag
    window, even for short series.

    Parameters
    ----------
    media : array_like
        Real-valued array with at least one dimension. Observations along
        ``axis`` must represent equally spaced periods. Other axes identify
        independent series, such as channels and geographies.
    shape : array_like
        Finite positive Weibull shape, controlling how retention changes
        across lags. Its shape must broadcast to the non-time shape of ``media``.
    scale : array_like
        Finite positive Weibull scale, measured in periods. Larger values
        retain more of earlier inputs. Its shape must broadcast to the
        non-time shape of ``media`` independently of ``shape``. For
        ``(time, geo, channel)`` inputs, parameters can have shape
        ``(channel,)``, ``(geo, 1)``, or ``(geo, channel)``.
    max_lag : int
        Nonnegative number of previous periods to include. The window has
        ``max_lag + 1`` weights, including the current period.
    axis : int, default 0
        Time axis in ``media``.
    normalize : bool, default True
        Whether to divide the weights by their sum.
        ``max_lag``, ``axis``, and ``normalize`` must be static under JIT.

    Returns
    -------
    jax.Array
        Transformed values with the same shape as ``media``. Inputs are
        promoted to a common floating-point dtype of at least float32.
        Invalid shape or scale parameters produce ``nan`` for their series.

    Examples
    --------
    Start with four consecutive weeks of video spending, including a week
    with none.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import weibull_cdf_adstock

        In [2]: spend = pl.DataFrame({
           ...:     "week": [1, 2, 3, 4],
           ...:     "video": [100.0, 0.0, 80.0, 40.0],
           ...: })

    Weight the carryover with a Weibull survival curve, so the effect is
    strongest at exposure and decreases from there.

    .. ipython::

        In [3]: carried = weibull_cdf_adstock(
           ...:     spend["video"].to_numpy(),
           ...:     shape=2.0,
           ...:     scale=3.0,
           ...:     max_lag=3,
           ...: )

    Keep the original spending beside the transformed values for comparison.

    .. ipython::

        In [4]: spend.with_columns(
           ...:     pl.Series("video_adstock", np.asarray(carried)),
           ...: )
    """
    media_array, parameters = _prepare_adstock(
        media, max_lag=max_lag, axis=axis, normalize=normalize, shape=shape, scale=scale
    )
    shape_array, scale_array = parameters["shape"], parameters["scale"]
    valid = jnp.isfinite(shape_array) & (shape_array > 0) & jnp.isfinite(scale_array) & (scale_array > 0)
    safe_shape = jnp.where(valid, shape_array, 1.0)[..., None]
    safe_scale = jnp.where(valid, scale_array, 1.0)[..., None]

    periods = jnp.arange(max_lag, dtype=media_array.dtype) + 1
    # Work with log survival directly so a CDF rounded to one does not erase its tail
    log_survival = -((periods / safe_scale) ** safe_shape)
    # Keep lag zero separate so shape gradients never differentiate 0**shape
    weights = jnp.concatenate((jnp.ones_like(safe_shape), jnp.exp(jnp.cumsum(log_survival, axis=-1))), axis=-1)
    if normalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(valid[..., None], weights, jnp.nan)
    return _convolve_adstock(media_array, weights, axis=axis)


def _prepare_adstock(
    media: ArrayLike,
    *,
    max_lag: int,
    axis: int,
    normalize: bool,
    **parameters: ArrayLike,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError("max_lag must be a static nonnegative integer")
    if max_lag < 0:
        raise ValueError(f"max_lag must be nonnegative, got {max_lag}")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be a static integer identifying the time axis")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a static boolean")

    inputs = {"media": media, **parameters}
    leaves = []
    for name, value in inputs.items():
        value_leaves = jax.tree_util.tree_leaves(value)
        try:
            argument_dtype = jnp.result_type(*value_leaves)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be real numeric and array-like, got {type(value).__name__}") from exc
        if not (
            jnp.issubdtype(argument_dtype, jnp.floating)
            or jnp.issubdtype(argument_dtype, jnp.integer)
            or argument_dtype == jnp.bool_
        ):
            raise TypeError(f"{name} must have a real numeric dtype, got {argument_dtype}")
        leaves.extend(value_leaves)

    # Choose floating-point precision before converting integer exposure counts
    dtype = jnp.result_type(*leaves)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
    arrays = {}
    for name, value in inputs.items():
        try:
            arrays[name] = jnp.asarray(value, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be real numeric and array-like, got {type(value).__name__}") from exc

    media_array = arrays["media"]
    if media_array.ndim == 0:
        raise ValueError("media must have at least one dimension for time, got a scalar")
    if not -media_array.ndim <= axis < media_array.ndim:
        raise ValueError(f"axis {axis} is out of range for media with {media_array.ndim} dimensions")
    axis %= media_array.ndim

    batch_shape = media_array.shape[:axis] + media_array.shape[axis + 1 :]
    broadcast_parameters = {}
    for name in parameters:
        array = arrays[name]
        try:
            broadcast_parameters[name] = jnp.broadcast_to(array, batch_shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {array.shape} must broadcast to the non-time media shape {batch_shape}; "
                f"media has shape {media_array.shape} with time axis {axis}"
            ) from exc
    return media_array, broadcast_parameters


def _convolve_adstock(media: jax.Array, weights: jax.Array, *, axis: int) -> jax.Array:
    if media.size == 0:
        return media

    time_last = jnp.moveaxis(media, axis, -1)
    n_times = time_last.shape[-1]
    n_lags = min(weights.shape[-1], n_times)
    padding = [(0, 0)] * (time_last.ndim - 1) + [(n_lags - 1, 0)]
    padded = jnp.pad(time_last, padding)

    def add_lag(lag: int, accumulated: jax.Array) -> jax.Array:
        shifted: jax.Array = jax.lax.dynamic_slice_in_dim(padded, n_lags - 1 - lag, n_times, axis=-1)
        weight: jax.Array = jax.lax.dynamic_index_in_dim(weights, lag, axis=-1, keepdims=True)
        return accumulated + shifted * weight

    # Elementwise lag products keep singular derivatives within their own series.
    # Any normalization uses the full window, including lags beyond the data.
    transformed = jax.lax.fori_loop(0, n_lags, add_lag, jnp.zeros_like(time_last))
    return jnp.moveaxis(transformed, -1, axis)
