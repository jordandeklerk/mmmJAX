"""Tests for channel-label selection in model calculations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mmmjax
from mmmjax.data import select_channels
from mmmjax.distributions import half_normal, lognormal


def test_select_channels_is_exported_from_package():
    assert mmmjax.select_channels is select_channels
    assert "select_channels" in mmmjax.__all__


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 4)])
@pytest.mark.parametrize("selection", ["video", ("search",), ["audio", "video"], ("audio", "search", "video")])
def test_selection_preserves_batch_axes_and_requested_order(batch_shape, selection):
    channels = ("video", "search", "audio")
    values = np.arange(np.prod((*batch_shape, 3))).reshape(*batch_shape, 3)
    selected_names = (selection,) if isinstance(selection, str) else selection
    indices = [channels.index(name) for name in selected_names]

    result = select_channels(values, channels=channels, select=selection)

    assert result.shape == (*batch_shape, len(indices))
    np.testing.assert_array_equal(result, np.take(values, indices, axis=-1))


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.int32, jnp.float16, jnp.bfloat16, jnp.float32, jnp.complex64])
def test_selection_preserves_array_dtype(dtype):
    values = jnp.asarray([1, 0, 1], dtype=dtype)

    result = select_channels(values, channels=["video", "search", "audio"], select=["audio", "search"])

    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, values[jnp.array([2, 1])])


def test_selection_preserves_float64_when_enabled():
    if not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    values = jnp.asarray([1.1, 2.2], dtype=jnp.float64)

    result = select_channels(values, channels=("video", "search"), select="search")

    assert result.dtype == jnp.float64
    np.testing.assert_array_equal(result, values[1:])


def test_selection_accepts_array_like_values_and_empty_batch_dimensions():
    channels = ("video", "search")
    np.testing.assert_array_equal(select_channels([0.5, 1.5], channels=channels, select="search"), [1.5])
    assert select_channels(jnp.empty((0, 2)), channels=channels, select="video").shape == (0, 1)


def test_selection_handles_large_channel_groups_under_jit():
    channels = tuple(f"channel_{index}" for index in range(465))
    selected = channels[::7][::-1]
    values = jnp.arange(2 * len(channels)).reshape(2, len(channels))
    function = jax.jit(lambda array: select_channels(array, channels=channels, select=selected))

    np.testing.assert_array_equal(function(values), np.asarray(values)[:, ::7][:, ::-1])
    np.testing.assert_array_equal(function(values + 1), np.asarray(values + 1)[:, ::7][:, ::-1])


def test_selection_accepts_explicit_static_label_arguments():
    function = jax.jit(select_channels, static_argnames=("channels", "select"))
    values = jnp.asarray([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(function(values, channels=("video", "search", "audio"), select="audio"), [3.0])
    np.testing.assert_array_equal(function(values, channels=("audio", "search", "video"), select="audio"), [1.0])


def test_selection_supports_vmap_and_derivatives_on_original_channel_positions():
    channels = ("video", "search", "audio")
    values = jnp.asarray([[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]])

    def selected_square_sum(array):
        selected = select_channels(array, channels=channels, select=("audio", "video"))
        return jnp.square(selected).sum()

    result = jax.jit(jax.vmap(jax.value_and_grad(selected_square_sum)))(values)
    expected_gradient = 2 * np.asarray(values) * [1, 0, 1]
    expected_value = np.square(np.asarray(values)[:, [2, 0]]).sum(axis=-1)

    np.testing.assert_allclose(result[0], expected_value, rtol=2e-6)
    np.testing.assert_allclose(result[1], expected_gradient, rtol=2e-6)
    np.testing.assert_array_equal(jax.jit(jax.hessian(selected_square_sum))(values[0]), np.diag([2, 0, 2]))


def test_selected_channel_groups_can_receive_different_explicit_prior_families():
    channels = ("search", "video", "audio")
    values = jnp.asarray([[0.4, 0.8, 1.0], [0.7, 0.5, 1.2]])

    def density(array):
        video_audio = select_channels(array, channels=channels, select=("video", "audio"))
        search = select_channels(array, channels=channels, select="search")
        return half_normal(video_audio, scale=1.0) + lognormal(search, location=0.0, scale=0.5)

    def reference(array):
        return half_normal(array[..., 1:], scale=1.0) + lognormal(array[..., :1], location=0.0, scale=0.5)

    result = jax.jit(jax.value_and_grad(density))(values)
    expected = jax.value_and_grad(reference)(values)

    np.testing.assert_allclose(result[0], expected[0], rtol=2e-6)
    np.testing.assert_allclose(result[1], expected[1], rtol=2e-6)


def test_unselected_values_do_not_enter_the_density_or_its_gradient():
    def density(array):
        video = select_channels(array, channels=("video", "search"), select="video")
        return half_normal(video, scale=1.0)

    value, gradient = jax.jit(jax.value_and_grad(density))(jnp.array([0.5, jnp.nan]))

    assert jnp.isfinite(value)
    np.testing.assert_array_equal(gradient, [-0.5, 0.0])


@pytest.mark.parametrize("channels", ["video", b"video", {"video"}, {"video": 0}, None])
def test_selection_rejects_unordered_or_nonsequence_channel_labels(channels):
    with pytest.raises(TypeError, match="channels must be an ordered sequence"):
        select_channels([1.0], channels=channels, select="video")


@pytest.mark.parametrize("selection", [b"video", {"video"}, {"video": 0}, None])
def test_selection_rejects_unordered_or_nonsequence_selections(selection):
    with pytest.raises(TypeError, match="select must be an ordered sequence"):
        select_channels([1.0], channels=("video",), select=selection)


@pytest.mark.parametrize("argument", ["channels", "select"])
@pytest.mark.parametrize(
    "names,error,match",
    [
        ([], ValueError, "at least one"),
        ([""], ValueError, "nonempty"),
        (["video", "video"], ValueError, "unique"),
        (["video", 1], TypeError, "only string"),
    ],
)
def test_selection_rejects_empty_duplicate_or_nonstring_names(argument, names, error, match):
    arguments = {"channels": ("video", "search"), "select": "video", argument: names}
    with pytest.raises(error, match=match):
        select_channels([1.0, 2.0], **arguments)


@pytest.mark.parametrize("selection", ["social", ["video", "social"]])
def test_selection_rejects_unknown_names_instead_of_filling_or_clipping(selection):
    with pytest.raises(ValueError, match=r"unknown channel names.*social"):
        select_channels([1.0, 2.0], channels=("video", "search"), select=selection)


@pytest.mark.parametrize("values", [1.0, jnp.array(1.0)])
def test_selection_requires_channel_axis(values):
    with pytest.raises(ValueError, match="final channel axis"):
        select_channels(values, channels=("video",), select="video")


@pytest.mark.parametrize("values", [[], [1.0], [[1.0, 2.0, 3.0]]])
def test_selection_requires_one_label_per_channel(values):
    with pytest.raises(ValueError, match=r"final axis.*channels contains 2 names"):
        select_channels(values, channels=("video", "search"), select="video")
