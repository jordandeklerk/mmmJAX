"""Tests for parameter declarations and their inference-space mappings."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from mmmjax import Interval, LowerBound, Parameterization, Positive, Real, Simplex, UpperBound


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(),
        Positive(),
        LowerBound(lower=-2.0),
        UpperBound(upper=2.0),
        Interval(lower=-2.0, upper=3.0),
    ],
)
def test_parameterization_round_trip(parameterization: Parameterization) -> None:
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


def test_lower_bound_maps_above_bound() -> None:
    parameterization = LowerBound(lower=-2.0, shape=(3,))
    position = jnp.log(jnp.array([0.25, 1.0, 4.0]))

    result = parameterization.constrain(position)

    assert jnp.all(result > parameterization.lower)
    assert jnp.allclose(result, jnp.array([-1.75, -1.0, 2.0]))


def test_upper_bound_maps_below_bound() -> None:
    parameterization = UpperBound(upper=3.0, shape=(3,))
    position = jnp.log(jnp.array([0.25, 1.0, 4.0]))

    result = parameterization.constrain(position)

    assert jnp.all(result < parameterization.upper)
    assert jnp.allclose(result, jnp.array([2.75, 2.0, -1.0]))


def test_interval_maps_between_bounds() -> None:
    parameterization = Interval(lower=-2.0, upper=3.0, shape=(3,))
    position = jnp.log(jnp.array([0.25, 1.0, 4.0]))

    result = parameterization.constrain(position)

    assert jnp.all(result > parameterization.lower)
    assert jnp.all(result < parameterization.upper)
    assert jnp.allclose(result, jnp.array([-1.0, 0.5, 2.0]))


@pytest.mark.parametrize(
    ("parameterization", "parameter", "expected"),
    [
        (LowerBound(lower=-2.0, shape=(2,)), jnp.array([-1.0, 0.0]), jnp.array([0.0, jnp.log(2.0)])),
        (UpperBound(upper=3.0, shape=(2,)), jnp.array([2.0, 1.0]), jnp.array([0.0, jnp.log(2.0)])),
        (
            Interval(lower=-2.0, upper=3.0, shape=(3,)),
            jnp.array([-0.75, 0.5, 1.75]),
            jnp.array([-jnp.log(3.0), 0.0, jnp.log(3.0)]),
        ),
    ],
)
def test_bounded_unconstrain_matches_analytical_values(
    parameterization: Parameterization,
    parameter: jax.Array,
    expected: jax.Array,
) -> None:
    result = parameterization.unconstrain(parameter)

    assert jnp.allclose(result, expected)


def test_real_log_density_adjustment_is_scalar_zero() -> None:
    adjustment = Real(shape=(3,)).log_density_adjustment(jnp.ones(3))

    assert adjustment.shape == ()
    assert adjustment == 0


def test_positive_log_density_adjustment_sums_event_dimensions() -> None:
    position = jnp.array([[-1.0, 0.5], [1.0, 2.0]])

    adjustment = Positive(shape=(2, 2)).log_density_adjustment(position)

    assert adjustment.shape == ()
    assert jnp.allclose(adjustment, jnp.sum(position))


@pytest.mark.parametrize(
    "parameterization",
    [LowerBound(lower=-2.0, shape=(2, 2)), UpperBound(upper=3.0, shape=(2, 2))],
)
def test_one_sided_log_density_adjustment_sums_event_dimensions(
    parameterization: LowerBound | UpperBound,
) -> None:
    position = jnp.array([[-1.0, 0.5], [1.0, 2.0]])

    adjustment = parameterization.log_density_adjustment(position)

    assert adjustment.shape == ()
    assert jnp.allclose(adjustment, 2.5)


def test_interval_log_density_adjustment_sums_event_dimensions() -> None:
    parameterization = Interval(lower=-2.0, upper=3.0, shape=(2, 2))
    position = jnp.zeros((2, 2))

    adjustment = parameterization.log_density_adjustment(position)

    assert adjustment.shape == ()
    assert jnp.allclose(adjustment, 4 * jnp.log(1.25))


@pytest.mark.parametrize(
    "parameterization",
    [
        LowerBound(lower=-2.0, shape=(2,)),
        UpperBound(upper=3.0, shape=(2,)),
        Interval(lower=-2.0, upper=3.0, shape=(2,)),
    ],
)
def test_bounded_adjustment_matches_transform_jacobian(parameterization: Parameterization) -> None:
    position = jnp.array([-0.75, 0.5])
    jacobian = jax.jacfwd(parameterization.constrain)(position)
    _, log_absolute_determinant = jnp.linalg.slogdet(jacobian)

    adjustment = parameterization.log_density_adjustment(position)

    assert jnp.allclose(adjustment, log_absolute_determinant)


def test_simplex_matches_known_isometric_log_ratio_values() -> None:
    parameterization = Simplex(shape=(3,))
    position = jnp.array([jnp.sqrt(2.0) * jnp.log(2.0), 0.0])

    parameter = parameterization.constrain(position)
    adjustment = parameterization.log_density_adjustment(position)

    assert jnp.allclose(parameter, jnp.array([4 / 7, 1 / 7, 2 / 7]))
    assert jnp.allclose(adjustment, jnp.log(8 / 343) + 0.5 * jnp.log(3.0))


def test_simplex_inference_space_round_trip() -> None:
    parameterization = Simplex(shape=(2, 4))
    position = jnp.array(
        [
            [-1.0, 0.25, 0.75],
            [0.5, -0.75, 1.25],
        ]
    )

    result = parameterization.unconstrain(parameterization.constrain(position))

    assert jnp.allclose(result, position, rtol=2e-6, atol=2e-6)


def test_simplex_model_space_round_trip() -> None:
    parameterization = Simplex(shape=(2, 4))
    parameter = jnp.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )

    result = parameterization.constrain(parameterization.unconstrain(parameter))

    assert jnp.allclose(result, parameter, rtol=2e-6, atol=2e-6)


def test_simplex_uses_final_axis_as_event_dimension() -> None:
    parameterization = Simplex(shape=(8, 465))
    position = parameterization.initialize(jax.random.key(0))

    parameter = jax.jit(parameterization.constrain)(position)

    assert parameterization.position_shape == (8, 464)
    assert position.shape == (8, 464)
    assert parameter.shape == (8, 465)
    assert jnp.all(parameter > 0)
    assert jnp.allclose(jnp.sum(parameter, axis=-1), jnp.ones(8), rtol=2e-6, atol=2e-6)


def test_single_weight_simplex_has_empty_inference_dimension() -> None:
    parameterization = Simplex(shape=(2, 1))
    position = parameterization.initialize(jax.random.key(0))

    parameter = parameterization.constrain(position)
    unconstrained = parameterization.unconstrain(parameter)
    adjustment = parameterization.log_density_adjustment(position)

    assert parameterization.position_shape == (2, 0)
    assert position.shape == (2, 0)
    assert jnp.array_equal(parameter, jnp.ones((2, 1)))
    assert unconstrained.shape == (2, 0)
    assert adjustment == 0


def test_simplex_adjustment_matches_transform_jacobian() -> None:
    parameterization = Simplex(shape=(4,))
    position = jnp.array([-0.75, 0.25, 1.0])
    jacobian = jax.jacfwd(lambda value: parameterization.constrain(value)[:-1])(position)
    _, log_absolute_determinant = jnp.linalg.slogdet(jacobian)

    adjustment = parameterization.log_density_adjustment(position)

    assert jnp.allclose(adjustment, log_absolute_determinant, rtol=2e-6, atol=2e-6)


def test_batched_simplex_adjustment_sums_each_event() -> None:
    batched = Simplex(shape=(2, 4))
    single = Simplex(shape=(4,))
    position = jnp.array(
        [
            [-0.75, 0.25, 1.0],
            [0.5, -1.0, 0.75],
        ]
    )

    result = batched.log_density_adjustment(position)
    expected = sum(single.log_density_adjustment(item) for item in position)

    assert result.shape == ()
    assert jnp.allclose(result, expected)


def test_simplex_can_be_jitted_vectorized_and_differentiated() -> None:
    parameterization = Simplex(shape=(4,))
    positions = jnp.array(
        [
            [-1.0, 0.0, 1.0],
            [0.5, -0.5, 0.25],
        ]
    )
    target = partial(_target, parameterization)

    constrained = jax.jit(jax.vmap(parameterization.constrain))(positions)
    gradients = jax.jit(jax.vmap(jax.grad(target)))(positions)

    assert constrained.shape == (2, 4)
    assert jnp.all(jnp.isfinite(constrained))
    assert jnp.all(jnp.isfinite(gradients))


def test_simplex_requires_an_event_dimension() -> None:
    with pytest.raises(ValueError, match=r"shape must include a final simplex dimension, got \(\)"):
        Simplex(shape=())


@pytest.mark.parametrize(
    ("method_name", "value", "message"),
    [
        ("constrain", jnp.ones(4), r"position must have shape \(3,\), got \(4,\)"),
        ("unconstrain", jnp.ones(3), r"parameter must have shape \(4,\), got \(3,\)"),
    ],
)
def test_simplex_rejects_wrong_value_shape(method_name: str, value: jax.Array, message: str) -> None:
    method = getattr(Simplex(shape=(4,)), method_name)

    with pytest.raises(ValueError, match=message):
        method(value)


def test_simplex_preserves_float32_dtype() -> None:
    parameterization = Simplex(shape=(3,), dtype=jnp.float32)
    position = jnp.array([-0.5, 0.5], dtype=jnp.float32)

    parameter = parameterization.constrain(position)

    assert parameterization.dtype == jnp.dtype(jnp.float32)
    assert parameterization.initialize(jax.random.key(0)).dtype == jnp.dtype(jnp.float32)
    assert parameter.dtype == jnp.dtype(jnp.float32)
    assert parameterization.unconstrain(parameter).dtype == jnp.dtype(jnp.float32)
    assert parameterization.log_density_adjustment(position).dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments", "shape"),
    [
        (Real, (), (2,)),
        (Positive, (), (2,)),
        (LowerBound, (0.0,), (2,)),
        (UpperBound, (1.0,), (2,)),
        (Interval, (0.0, 1.0), (2,)),
        (Simplex, (), (8, 465)),
    ],
)
def test_parameterizations_use_float32_for_low_precision_requests(
    parameterization_type,
    bound_arguments,
    shape,
    dtype,
) -> None:
    parameterization = parameterization_type(*bound_arguments, shape=shape, dtype=dtype)
    position = parameterization.initialize(jax.random.key(0))

    parameter = parameterization.constrain(position)
    unconstrained = parameterization.unconstrain(parameter)
    adjustment = parameterization.log_density_adjustment(position)

    assert parameterization.dtype == jnp.dtype(jnp.float32)
    assert position.dtype == jnp.dtype(jnp.float32)
    assert parameter.dtype == jnp.dtype(jnp.float32)
    assert unconstrained.dtype == jnp.dtype(jnp.float32)
    assert adjustment.dtype == jnp.dtype(jnp.float32)
    assert jnp.allclose(unconstrained, position, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(shape=(2,)),
        Positive(shape=(2,)),
        LowerBound(lower=0.0, shape=(2,)),
        UpperBound(upper=1.0, shape=(2,)),
        Interval(lower=0.0, upper=1.0, shape=(2,)),
    ],
)
def test_parameterization_rejects_wrong_value_shape(parameterization: Parameterization) -> None:
    with pytest.raises(ValueError, match=r"must have shape \(2,\), got \(3,\)"):
        parameterization.constrain(jnp.ones(3))


@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("shape", [[2], (True,), (1.5,)])
def test_parameterization_rejects_invalid_shape_type(parameterization_type, bound_arguments, shape) -> None:
    with pytest.raises(TypeError, match=r"shape(\[0\])? must be a (tuple of )?positive integer"):
        parameterization_type(*bound_arguments, shape=shape)


@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("shape", [(0,), (-1,)])
def test_parameterization_rejects_nonpositive_shape(parameterization_type, bound_arguments, shape) -> None:
    with pytest.raises(ValueError, match=rf"shape\[0\] must be positive, got {shape[0]}"):
        parameterization_type(*bound_arguments, shape=shape)


@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, "not-a-dtype"])
def test_parameterization_requires_floating_dtype(parameterization_type, bound_arguments, dtype) -> None:
    with pytest.raises(TypeError, match="dtype must be a floating-point dtype"):
        parameterization_type(*bound_arguments, dtype=dtype)


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(),
        Positive(),
        LowerBound(lower=0.0),
        UpperBound(upper=1.0),
        Interval(lower=0.0, upper=1.0),
    ],
)
def test_parameterization_requires_array_like_values(parameterization: Parameterization) -> None:
    with pytest.raises(TypeError, match="position must be array-like and convertible"):
        parameterization.constrain(object())


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(shape=(2, 3)),
        Positive(shape=(2, 3)),
        LowerBound(lower=0.0, shape=(2, 3)),
        UpperBound(upper=1.0, shape=(2, 3)),
        Interval(lower=0.0, upper=1.0, shape=(2, 3)),
    ],
)
def test_initialization_is_shaped_deterministic_and_unconstrained(
    parameterization: Parameterization,
) -> None:
    key, different_key = jax.random.split(jax.random.key(42))

    first = parameterization.initialize(key)
    repeated = parameterization.initialize(key)
    different = parameterization.initialize(different_key)
    compiled = jax.jit(parameterization.initialize)(key)

    assert first.shape == (2, 3)
    assert first.dtype == jnp.asarray(0.0).dtype
    assert jnp.array_equal(first, repeated)
    assert jnp.array_equal(first, compiled)
    assert not jnp.array_equal(first, different)
    assert jnp.all(first >= -2.0)
    assert jnp.all(first < 2.0)


@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
def test_default_dtype_follows_jax_precision_policy(parameterization_type, bound_arguments) -> None:
    parameterization = parameterization_type(*bound_arguments, shape=(2,))
    expected_dtype = jnp.asarray(0.0).dtype

    position = jnp.zeros(parameterization.position_shape, dtype=expected_dtype)
    parameter = parameterization.constrain(position)

    assert parameterization.dtype == expected_dtype
    assert parameterization.initialize(jax.random.key(0)).dtype == expected_dtype
    assert parameter.dtype == expected_dtype
    assert parameterization.unconstrain(parameter).dtype == expected_dtype
    assert parameterization.log_density_adjustment(position).dtype == expected_dtype


@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
def test_explicit_float32_is_preserved(parameterization_type, bound_arguments) -> None:
    parameterization = parameterization_type(*bound_arguments, shape=(2,), dtype=jnp.float32)
    position = jnp.array([-0.5, 0.5], dtype=jnp.float32)

    parameter = parameterization.constrain(position)
    unconstrained = parameterization.unconstrain(parameter)
    adjustment = parameterization.log_density_adjustment(position)

    assert parameterization.dtype == jnp.dtype(jnp.float32)
    assert parameterization.initialize(jax.random.key(0)).dtype == jnp.dtype(jnp.float32)
    assert parameter.dtype == jnp.dtype(jnp.float32)
    assert unconstrained.dtype == jnp.dtype(jnp.float32)
    assert adjustment.dtype == jnp.dtype(jnp.float32)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
def test_explicit_float64_is_preserved(parameterization_type, bound_arguments) -> None:
    parameterization = parameterization_type(*bound_arguments, shape=(2,), dtype=jnp.float64)
    position = jnp.array([-0.5, 0.5], dtype=jnp.float64)

    parameter = parameterization.constrain(position)

    assert parameterization.dtype == jnp.dtype(jnp.float64)
    assert parameterization.initialize(jax.random.key(0)).dtype == jnp.dtype(jnp.float64)
    assert parameter.dtype == jnp.dtype(jnp.float64)
    assert parameterization.unconstrain(parameter).dtype == jnp.dtype(jnp.float64)
    assert parameterization.log_density_adjustment(position).dtype == jnp.dtype(jnp.float64)


@pytest.mark.skipif(jax.config.x64_enabled, reason="JAX 64-bit mode is enabled")
@pytest.mark.parametrize(
    ("parameterization_type", "bound_arguments"),
    [
        (Real, ()),
        (Positive, ()),
        (LowerBound, (0.0,)),
        (UpperBound, (1.0,)),
        (Interval, (0.0, 1.0)),
    ],
)
def test_explicit_float64_requires_x64(parameterization_type, bound_arguments) -> None:
    with pytest.raises(ValueError, match="explicit dtype float64 requires JAX 64-bit mode"):
        parameterization_type(*bound_arguments, dtype=jnp.float64)


@pytest.mark.skipif(jax.config.x64_enabled, reason="JAX 64-bit mode is enabled")
@pytest.mark.parametrize("dtype", [jnp.float64, "float64", jnp.dtype("float64")])
def test_explicit_float64_aliases_require_x64(dtype) -> None:
    with pytest.raises(ValueError, match="JAX_ENABLE_X64=true"):
        Real(dtype=dtype)


def test_dtype_is_canonicalized() -> None:
    from_string = Real(dtype="float32")
    from_type = Real(dtype=jnp.float32)

    assert from_string == from_type
    assert hash(from_string) == hash(from_type)


def test_bound_metadata_is_canonicalized_for_equality_and_hashing() -> None:
    from_integers = Interval(lower=0, upper=1)
    from_floats = Interval(lower=0.0, upper=1.0)

    assert from_integers == from_floats
    assert hash(from_integers) == hash(from_floats)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        pytest.param(lambda: LowerBound(lower=True), "lower must be a finite real scalar", id="lower-bool"),
        pytest.param(lambda: LowerBound(lower=None), "lower must be a finite real scalar", id="lower-none"),
        pytest.param(
            lambda: LowerBound(lower=10**10000),
            "lower must be a finite real scalar",
            id="lower-overflow",
        ),
        pytest.param(lambda: UpperBound(upper="one"), "upper must be a finite real scalar", id="upper-string"),
        pytest.param(
            lambda: Interval(lower=0.0 + 1.0j, upper=1.0),
            "lower must be a finite real scalar",
            id="interval-complex",
        ),
    ],
)
def test_bounds_require_real_numeric_values(constructor, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        constructor()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: LowerBound(lower=jnp.array([0.0, 1.0])),
        lambda: UpperBound(upper=jnp.array([0.0, 1.0])),
        lambda: Interval(lower=jnp.array([0.0, 1.0]), upper=2.0),
    ],
)
def test_bounds_require_scalar_values(constructor) -> None:
    with pytest.raises(ValueError, match=r"must be scalar, got shape \(2,\)"):
        constructor()


@pytest.mark.parametrize(
    ("constructor", "name"),
    [
        (lambda: LowerBound(lower=-jnp.inf), "lower"),
        (lambda: UpperBound(upper=jnp.inf), "upper"),
        (lambda: Interval(lower=jnp.nan, upper=1.0), "lower"),
        (lambda: Interval(lower=0.0, upper=jnp.inf), "upper"),
    ],
)
def test_bounds_must_be_finite(constructor, name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be finite"):
        constructor()


@pytest.mark.parametrize(("lower", "upper"), [(1.0, 1.0), (2.0, 1.0)])
def test_interval_requires_ordered_bounds(lower: float, upper: float) -> None:
    with pytest.raises(ValueError, match="lower must be less than upper"):
        Interval(lower=lower, upper=upper)


def test_interval_rejects_bounds_that_collapse_in_declared_dtype() -> None:
    with pytest.raises(ValueError, match="after conversion to float32"):
        Interval(lower=1.0, upper=1.0 + 1e-8, dtype=jnp.float32)


def test_interval_requires_representable_width() -> None:
    with pytest.raises(ValueError, match="upper - lower must be finite in dtype float32"):
        Interval(lower=-3e38, upper=3e38, dtype=jnp.float32)


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(shape=(3,)),
        Positive(shape=(3,)),
        LowerBound(lower=0.0, shape=(3,)),
        UpperBound(upper=1.0, shape=(3,)),
        Interval(lower=0.0, upper=1.0, shape=(3,)),
    ],
)
def test_position_shape_matches_model_shape_for_dimension_preserving_parameters(
    parameterization: Parameterization,
) -> None:
    assert parameterization.position_shape == parameterization.shape


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(shape=(3,)),
        Positive(shape=(3,)),
        LowerBound(lower=-2.0, shape=(3,)),
        UpperBound(upper=2.0, shape=(3,)),
        Interval(lower=-2.0, upper=3.0, shape=(3,)),
    ],
)
def test_parameterization_target_can_be_jitted(parameterization: Parameterization) -> None:
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


@pytest.mark.parametrize(
    ("parameterization", "expected_gradient"),
    [
        (LowerBound(lower=-2.0), 2.0),
        (UpperBound(upper=2.0), 2.0),
        (Interval(lower=-2.0, upper=3.0), -0.625),
    ],
)
def test_bounded_target_gradient_matches_analytical_result(
    parameterization: Parameterization,
    expected_gradient: float,
) -> None:
    position = jnp.array(0.0)
    target = partial(_target, parameterization)

    gradient = jax.jit(jax.grad(target))(position)

    assert jnp.allclose(gradient, expected_gradient)


def test_positive_remains_finite_across_representative_float32_positions() -> None:
    parameterization = Positive(shape=(3,), dtype=jnp.float32)
    position = jnp.array([-80.0, 0.0, 80.0], dtype=jnp.float32)

    constrained = parameterization.constrain(position)

    assert jnp.all(jnp.isfinite(constrained))
    assert jnp.all(constrained > 0)
    assert jnp.allclose(parameterization.unconstrain(constrained), position)


def test_interval_adjustment_remains_finite_across_float32_tails() -> None:
    parameterization = Interval(lower=0.0, upper=1.0, shape=(3,), dtype=jnp.float32)
    position = jnp.array([-80.0, 0.0, 80.0], dtype=jnp.float32)

    constrained = parameterization.constrain(position)
    adjustment = parameterization.log_density_adjustment(position)

    assert jnp.all(jnp.isfinite(constrained))
    assert jnp.isfinite(adjustment)
    assert jnp.allclose(adjustment, -160.0 - 2 * jnp.log(2.0))


def test_interval_adjustment_gradient_matches_analytical_result() -> None:
    parameterization = Interval(lower=0.0, upper=1.0, shape=(3,))
    position = jnp.array([-jnp.log(3.0), 0.0, jnp.log(3.0)])

    gradient = jax.grad(parameterization.log_density_adjustment)(position)

    assert jnp.allclose(gradient, jnp.array([0.5, 0.0, -0.5]))


def test_interval_keeps_positive_tail_gradient() -> None:
    parameterization = Interval(lower=0.0, upper=1.0, dtype=jnp.float32)

    gradient = jax.grad(parameterization.constrain)(jnp.array(20.0, dtype=jnp.float32))

    assert jnp.allclose(gradient, 2.0611537e-9, rtol=1e-6)


@pytest.mark.parametrize(
    "parameterization",
    [
        LowerBound(lower=1e30, dtype=jnp.float32),
        UpperBound(upper=1e30, dtype=jnp.float32),
    ],
)
def test_one_sided_transform_can_round_to_bound(parameterization: LowerBound | UpperBound) -> None:
    constrained = parameterization.constrain(jnp.array(0.0, dtype=jnp.float32))

    assert constrained == 1e30
    assert jnp.isneginf(parameterization.unconstrain(constrained))


def test_bounded_unconstrain_has_mathematical_boundary_behavior() -> None:
    lower = LowerBound(lower=-2.0)
    upper = UpperBound(upper=3.0)
    interval = Interval(lower=-2.0, upper=3.0)

    assert jnp.isneginf(lower.unconstrain(-2.0))
    assert jnp.isnan(lower.unconstrain(-3.0))
    assert jnp.isneginf(upper.unconstrain(3.0))
    assert jnp.isnan(upper.unconstrain(4.0))
    assert jnp.isneginf(interval.unconstrain(-2.0))
    assert jnp.isposinf(interval.unconstrain(3.0))
    assert jnp.isnan(interval.unconstrain(-3.0))
    assert jnp.isnan(interval.unconstrain(4.0))


@pytest.mark.parametrize(
    "parameterization",
    [
        Real(shape=(2,)),
        Positive(shape=(2,)),
        LowerBound(lower=-2.0, shape=(2,)),
        UpperBound(upper=2.0, shape=(2,)),
        Interval(lower=-2.0, upper=3.0, shape=(2,)),
    ],
)
def test_parameterization_target_can_be_vectorized(parameterization: Parameterization) -> None:
    positions = jnp.array([[-1.0, 0.0], [0.5, 1.0]])
    target = partial(_target, parameterization)

    result = jax.vmap(target)(positions)

    expected = jnp.stack([target(position) for position in positions])
    assert jnp.allclose(result, expected)


def _target(parameterization: Parameterization, position: jax.Array) -> jax.Array:
    parameter = parameterization.constrain(position)
    return -0.5 * jnp.sum(parameter**2) + parameterization.log_density_adjustment(position)
