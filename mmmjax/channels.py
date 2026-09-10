"""Select channel values by their prepared labels."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["select_channels"]


def select_channels(
    values: ArrayLike,
    *,
    channels: Sequence[str],
    select: str | Sequence[str],
) -> jax.Array:
    """Select named channels without changing the preceding dimensions.

    Select parameters for channel-specific priors or retrieve contributions
    by channel name.

    Parameters
    ----------
    values : array_like
        Parameters or contributions with channels on the final axis.
    channels : sequence of str
        Unique channel labels in the array's current order, such as
        ``data.channels``. Keep these labels fixed when compiling.
    select : str or sequence of str
        One channel name or a nonempty sequence of unique names to retain,
        in the desired output order. Keep this selection fixed when compiling.

    Returns
    -------
    jax.Array
        Selected values with the input's preceding dimensions and dtype.
        Selecting one name retains a final channel axis of length one.

    Examples
    --------
    Use different prior families for two entries of the same coefficient
    array. For group-specific coefficients, the same calls select the
    channels across every group.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import half_normal, lognormal, select_channels
           ...: channels = ("video", "search")
           ...: coefficient = jnp.array([0.6, 0.4])
           ...: video = select_channels(
           ...:     coefficient, channels=channels, select="video",
           ...: )
           ...: search = select_channels(
           ...:     coefficient, channels=channels, select="search",
           ...: )
           ...: target = half_normal(video, scale=1.0)
           ...: target += lognormal(search, location=0.0, scale=0.5)
           ...: target
    """
    channel_names = _channel_names(channels, argument="channels")
    selected_names = _channel_names((select,) if isinstance(select, str) else select, argument="select")
    positions = {name: index for index, name in enumerate(channel_names)}
    unknown = [name for name in selected_names if name not in positions]
    if unknown:
        raise ValueError(f"select contains unknown channel names {unknown}. Use names from channels")

    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must have a final channel axis")
    if array.shape[-1] != len(channel_names):
        raise ValueError(
            f"The final axis of values has length {array.shape[-1]} but channels contains {len(channel_names)} names"
        )

    indices = tuple(positions[name] for name in selected_names)
    return jnp.take(array, jnp.asarray(indices, dtype=jnp.int32), axis=-1, unique_indices=True)


def _channel_names(value: object, *, argument: str) -> tuple[str, ...]:
    """Validate ordered labels before constructing array indices."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{argument} must be an ordered sequence of channel names")
    if not value:
        raise ValueError(f"{argument} must contain at least one channel name")

    names = []
    for name in value:
        if not isinstance(name, str):
            raise TypeError(f"{argument} must contain only string channel names")
        if not name:
            raise ValueError(f"{argument} must contain only nonempty channel names")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{argument} must contain unique channel names")
    return tuple(names)
