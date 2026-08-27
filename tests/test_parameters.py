"""Tests for parameter declarations and their inference-space mappings."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from mmmjax import Parameterization, Positive, Real


@pytest.mark.parametrize("parameterization", [Real(), Positive()])
def test_parameterization_round_trip(parameterization: Real | Positive) -> None:
    position = jnp.array(0.75, dtype=jnp.float32)

    result = parameterization.unconstrain(parameterization.constrain(position))

    assert jnp.allclose(result, position)


def test_real_is_identity() -> None:
    parameterization = Real(shape=(3,))
    position = jnp.array([-1.0, 0.0, 2.0])

    assert jnp.array_equal(parameterization.constrain(position), position)
    assert jnp.array_equal(parameterization.unconstrain(position), position)


def test_positive_maps_to_positive_reals() -> None:
    parameterization = Positive(shape=(3,))
    position = jnp.array([-2.0, 0.0, 2.0])

    result = parameterization.constrain(position)

    assert jnp.all(result > 0)
    assert jnp.allclose(result, jnp.exp(position))


def test_positive_model_space_round_trip() -> None:
    parameterization = Positive(shape=(3,))
    parameter = jnp.array([0.25, 1.0, 4.0])

    result = parameterization.constrain(parameterization.unconstrain(parameter))

    assert jnp.allclose(result, parameter)


def test_real_log_density_adjustment_is_scalar_zero() -> None:
    adjustment = Real(shape=(3,)).log_density_adjustment(jnp.ones(3))

    assert adjustment.shape == ()
    assert adjustment == 0


def test_positive_log_density_adjustment_sums_event_dimensions() -> None:
    position = jnp.array([[-1.0, 0.5], [1.0, 2.0]])

    adjustment = Positive(shape=(2, 2)).log_density_adjustment(position)

    assert adjustment.shape == ()
    assert jnp.allclose(adjustment, jnp.sum(position))


@pytest.mark.parametrize("parameterization", [Real(shape=(2,)), Positive(shape=(2,))])
def test_parameterization_rejects_wrong_value_shape(parameterization: Real | Positive) -> None:
    with pytest.raises(ValueError, match=r"must have shape \(2,\), got \(3,\)"):
        parameterization.constrain(jnp.ones(3))


@pytest.mark.parametrize("parameterization_type", [Real, Positive])
@pytest.mark.parametrize("shape", [[2], (True,), (1.5,)])
def test_parameterization_rejects_invalid_shape_type(parameterization_type, shape) -> None:
    with pytest.raises(TypeError, match=r"shape(\[0\])? must be a (tuple of )?positive integer"):
        parameterization_type(shape=shape)


@pytest.mark.parametrize("parameterization_type", [Real, Positive])
@pytest.mark.parametrize("shape", [(0,), (-1,)])
def test_parameterization_rejects_nonpositive_shape(parameterization_type, shape) -> None:
    with pytest.raises(ValueError, match=rf"shape\[0\] must be positive, got {shape[0]}"):
        parameterization_type(shape=shape)


@pytest.mark.parametrize("parameterization_type", [Real, Positive])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, "not-a-dtype"])
def test_parameterization_requires_floating_dtype(parameterization_type, dtype) -> None:
    with pytest.raises(TypeError, match="dtype must be a floating-point dtype"):
        parameterization_type(dtype=dtype)


@pytest.mark.parametrize("parameterization", [Real(), Positive()])
def test_parameterization_requires_array_like_values(parameterization: Real | Positive) -> None:
    with pytest.raises(TypeError, match="position must be array-like and convertible"):
        parameterization.constrain(object())


@pytest.mark.parametrize("parameterization", [Real(shape=(2, 3)), Positive(shape=(2, 3))])
def test_initialization_is_shaped_deterministic_and_unconstrained(
    parameterization: Real | Positive,
) -> None:
    key, different_key = jax.random.split(jax.random.key(42))

    first = parameterization.initialize(key)
    repeated = parameterization.initialize(key)
    different = parameterization.initialize(different_key)
    compiled = jax.jit(parameterization.initialize)(key)

    assert first.shape == (2, 3)
    assert first.dtype == jnp.dtype(jnp.float32)
    assert jnp.array_equal(first, repeated)
    assert jnp.array_equal(first, compiled)
    assert not jnp.array_equal(first, different)
    assert jnp.all(first >= -2.0)
    assert jnp.all(first < 2.0)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
@pytest.mark.parametrize("parameterization", [Real(dtype=jnp.float64), Positive(dtype=jnp.float64)])
def test_initialization_supports_float64(parameterization: Real | Positive) -> None:
    assert parameterization.initialize(jax.random.key(0)).dtype == jnp.dtype(jnp.float64)


def test_dtype_is_canonicalized() -> None:
    from_string = Real(dtype="float32")
    from_type = Real(dtype=jnp.float32)

    assert from_string == from_type
    assert hash(from_string) == hash(from_type)


def test_dtype_follows_jax_precision_policy() -> None:
    parameterization = Real(dtype=jnp.float64)

    assert parameterization.dtype == jax.dtypes.canonicalize_dtype(jnp.float64)


@pytest.mark.parametrize("parameterization", [Real(shape=(3,)), Positive(shape=(3,))])
def test_position_shape_matches_model_shape_for_dimension_preserving_parameters(
    parameterization: Real | Positive,
) -> None:
    assert parameterization.position_shape == parameterization.shape


@pytest.mark.parametrize("parameterization", [Real(shape=(3,)), Positive(shape=(3,))])
def test_parameterization_target_can_be_jitted(parameterization: Real | Positive) -> None:
    position = jnp.array([-0.5, 0.0, 0.5])
    target = partial(_target, parameterization)

    eager = target(position)
    compiled = jax.jit(target)(position)

    assert jnp.allclose(compiled, eager)


@pytest.mark.parametrize("parameterization", [Real(shape=(3,)), Positive(shape=(3,))])
def test_parameterization_target_is_differentiable(parameterization: Real | Positive) -> None:
    position = jnp.array([-0.5, 0.0, 0.5])
    target = partial(_target, parameterization)

    gradient = jax.grad(target)(position)

    expected = -position if isinstance(parameterization, Real) else 1 - jnp.exp(2 * position)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, expected)


def test_positive_remains_finite_across_representative_float32_positions() -> None:
    parameterization = Positive(shape=(3,))
    position = jnp.array([-80.0, 0.0, 80.0], dtype=jnp.float32)

    constrained = parameterization.constrain(position)

    assert jnp.all(jnp.isfinite(constrained))
    assert jnp.all(constrained > 0)
    assert jnp.allclose(parameterization.unconstrain(constrained), position)


@pytest.mark.parametrize("parameterization", [Real(shape=(2,)), Positive(shape=(2,))])
def test_parameterization_target_can_be_vectorized(parameterization: Real | Positive) -> None:
    positions = jnp.array([[-1.0, 0.0], [0.5, 1.0]])
    target = partial(_target, parameterization)

    result = jax.vmap(target)(positions)

    expected = jnp.stack([target(position) for position in positions])
    assert jnp.allclose(result, expected)


def _target(parameterization: Parameterization, position: jax.Array) -> jax.Array:
    parameter = parameterization.constrain(position)
    return -0.5 * jnp.sum(parameter**2) + parameterization.log_density_adjustment(position)
