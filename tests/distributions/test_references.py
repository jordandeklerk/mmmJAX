"""Tests against established distribution implementations."""

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy import stats as jax_stats
from scipy import special, stats

import mmmjax
from benchmarks.references import JAX_REFERENCES


@dataclass(frozen=True)
class _SmoothReferenceCase:
    name: str
    arguments: tuple[jax.Array, ...]
    scipy_logpdf: np.ndarray


_SMOOTH_REFERENCE_CASES = (
    _SmoothReferenceCase(
        name="beta",
        arguments=(jnp.array([0.1, 0.8]), jnp.array([0.5, 5.0]), jnp.array([3.0, 0.8])),
        scipy_logpdf=stats.beta.logpdf([0.1, 0.8], [0.5, 5.0], [3.0, 0.8]),
    ),
    _SmoothReferenceCase(
        name="exponential",
        arguments=(jnp.array([0.1, 2.0]), jnp.array([0.5, 3.0])),
        scipy_logpdf=stats.expon.logpdf([0.1, 2.0], scale=1 / np.array([0.5, 3.0])),
    ),
    _SmoothReferenceCase(
        name="gamma",
        arguments=(jnp.array([0.1, 2.0]), jnp.array([0.5, 5.0]), jnp.array([0.8, 3.0])),
        scipy_logpdf=stats.gamma.logpdf([0.1, 2.0], [0.5, 5.0], scale=1 / np.array([0.8, 3.0])),
    ),
    _SmoothReferenceCase(
        name="half_normal",
        arguments=(jnp.array([0.25, 2.0]), jnp.array([0.5, 3.0])),
        scipy_logpdf=stats.halfnorm.logpdf([0.25, 2.0], scale=[0.5, 3.0]),
    ),
    _SmoothReferenceCase(
        name="inverse_gamma",
        arguments=(jnp.array([0.1, 2.0]), jnp.array([0.5, 5.0]), jnp.array([0.8, 3.0])),
        scipy_logpdf=stats.invgamma.logpdf([0.1, 2.0], [0.5, 5.0], scale=[0.8, 3.0]),
    ),
    _SmoothReferenceCase(
        name="laplace",
        arguments=(jnp.array([-2.0, 3.0]), jnp.array([-0.5, 2.0]), jnp.array([0.8, 3.0])),
        scipy_logpdf=stats.laplace.logpdf([-2.0, 3.0], loc=[-0.5, 2.0], scale=[0.8, 3.0]),
    ),
    _SmoothReferenceCase(
        name="lognormal",
        arguments=(jnp.array([0.1, 4.0]), jnp.array([-0.5, 1.0]), jnp.array([0.8, 0.5])),
        scipy_logpdf=stats.lognorm.logpdf(
            [0.1, 4.0],
            [0.8, 0.5],
            scale=np.exp([-0.5, 1.0]),
        ),
    ),
    _SmoothReferenceCase(
        name="normal",
        arguments=(jnp.array([-2.0, 3.0]), jnp.array([-0.5, 2.0]), jnp.array([0.8, 3.0])),
        scipy_logpdf=stats.norm.logpdf([-2.0, 3.0], loc=[-0.5, 2.0], scale=[0.8, 3.0]),
    ),
    _SmoothReferenceCase(
        name="student_t",
        arguments=(
            jnp.array([-2.0, 3.0]),
            jnp.array([3.0, 10.0]),
            jnp.array([-0.5, 2.0]),
            jnp.array([0.8, 3.0]),
        ),
        scipy_logpdf=stats.t.logpdf(
            [-2.0, 3.0],
            [3.0, 10.0],
            loc=[-0.5, 2.0],
            scale=[0.8, 3.0],
        ),
    ),
    _SmoothReferenceCase(
        name="uniform",
        arguments=(jnp.array([-0.5, 1.0]), jnp.array([-1.0, 0.0]), jnp.array([1.0, 3.0])),
        scipy_logpdf=stats.uniform.logpdf([-0.5, 1.0], loc=[-1.0, 0.0], scale=[2.0, 3.0]),
    ),
)


def test_jax_references_cover_every_distribution() -> None:
    assert set(JAX_REFERENCES) == {
        "beta",
        "exponential",
        "gamma",
        "half_normal",
        "inverse_gamma",
        "laplace",
        "lognormal",
        "normal",
        "student_t",
        "uniform",
    }


@pytest.mark.parametrize("case", _SMOOTH_REFERENCE_CASES, ids=lambda case: case.name)
def test_mmmjax_logpdf_matches_jax_and_scipy(case: _SmoothReferenceCase) -> None:
    reference = JAX_REFERENCES[case.name]
    implementation = getattr(mmmjax, f"{case.name}_logpdf")

    result = implementation(*case.arguments)
    jax_result = reference.logpdf(*case.arguments)
    compiled_jax_result = jax.jit(reference.logpdf)(*case.arguments)

    _assert_close(result, case.scipy_logpdf)
    _assert_close(result, jax_result)
    _assert_close(result, compiled_jax_result)


@pytest.mark.parametrize("case", _SMOOTH_REFERENCE_CASES, ids=lambda case: case.name)
def test_mmmjax_reverse_mode_matches_jax(case: _SmoothReferenceCase) -> None:
    reference = JAX_REFERENCES[case.name].logpdf
    implementation = getattr(mmmjax, f"{case.name}_logpdf")
    argnums = tuple(range(len(case.arguments)))

    result = jax.jit(jax.jacrev(implementation, argnums=argnums))(*case.arguments)
    jax_result = jax.jit(jax.jacrev(reference, argnums=argnums))(*case.arguments)

    for result_jacobian, jax_jacobian in zip(result, jax_result, strict=True):
        _assert_close(result_jacobian, jax_jacobian)


@pytest.mark.parametrize("case", _SMOOTH_REFERENCE_CASES, ids=lambda case: case.name)
def test_mmmjax_forward_mode_matches_jax(case: _SmoothReferenceCase) -> None:
    reference = JAX_REFERENCES[case.name].logpdf
    implementation = getattr(mmmjax, f"{case.name}_logpdf")
    argnums = tuple(range(len(case.arguments)))

    result = jax.jit(jax.jacfwd(implementation, argnums=argnums))(*case.arguments)
    jax_result = jax.jit(jax.jacfwd(reference, argnums=argnums))(*case.arguments)

    for result_jacobian, jax_jacobian in zip(result, jax_result, strict=True):
        _assert_close(result_jacobian, jax_jacobian)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_normal_log_probabilities_match_jax_and_scipy(operation: str) -> None:
    values = (
        jnp.array([-40.0, -10.0, -2.0, 0.0, 2.0]) if operation == "logcdf" else jnp.array([-2.0, 0.0, 2.0, 10.0, 40.0])
    )
    location = jnp.asarray(0.4)
    scale = jnp.asarray(1.7)
    implementation = getattr(mmmjax, f"normal_{operation}")
    jax_reference = getattr(JAX_REFERENCES["normal"], operation)
    scipy_reference = getattr(stats.norm, operation)
    assert jax_reference is not None

    result = implementation(values, location, scale)

    _assert_close(result, jax_reference(values, loc=location, scale=scale))
    _assert_scipy_tail_close(
        result,
        scipy_reference(np.asarray(values), loc=float(location), scale=float(scale)),
    )


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_lognormal_log_probabilities_match_jax_and_scipy(operation: str) -> None:
    values = (
        jnp.array([1e-20, 1e-8, 0.1, 1.0, 10.0]) if operation == "logcdf" else jnp.array([1.0, 10.0, 1e2, 1e8, 1e20])
    )
    location = jnp.asarray(0.4)
    scale = jnp.asarray(0.7)
    implementation = getattr(mmmjax, f"lognormal_{operation}")
    jax_reference = getattr(JAX_REFERENCES["lognormal"], operation)
    scipy_reference = getattr(stats.lognorm, operation)
    assert jax_reference is not None

    result = implementation(values, location, scale)
    jax_result = jax_reference(values, location, scale)
    scipy_result = scipy_reference(
        np.asarray(values),
        float(scale),
        scale=np.exp(float(location)),
    )

    _assert_close(result, jax_result)
    _assert_scipy_tail_close(result, scipy_result)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_exponential_log_probabilities_match_jax_and_scipy(operation: str) -> None:
    values = jnp.array([1e-10, 0.1, 0.5, 1.0])
    rate = jnp.asarray(1.7)
    implementation = getattr(mmmjax, f"exponential_{operation}")
    jax_reference = getattr(JAX_REFERENCES["exponential"], operation)
    scipy_reference = getattr(stats.expon, operation)
    assert jax_reference is not None

    result = implementation(values, rate)
    scipy_result = scipy_reference(np.asarray(values), scale=1 / float(rate))

    _assert_scipy_tail_close(result, scipy_result)
    _assert_close(result, jax_reference(values, rate))


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_half_normal_log_probabilities_match_jax_and_scipy(operation: str) -> None:
    values = (
        jnp.array([1e-10, 1e-5, 0.1, 0.5, 1.0, 2.0])
        if operation == "logcdf"
        else jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    )
    scale = jnp.asarray(1.7)
    implementation = getattr(mmmjax, f"half_normal_{operation}")
    jax_reference = getattr(JAX_REFERENCES["half_normal"], operation)
    scipy_reference = getattr(stats.halfnorm, operation)
    assert jax_reference is not None

    result = implementation(values, scale)
    jax_result = jax_reference(values, scale)
    scipy_result = scipy_reference(np.asarray(values), scale=float(scale))

    _assert_close(result, jax_result)
    _assert_scipy_tail_close(result, scipy_result)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_inverse_gamma_log_probabilities_match_jax_and_scipy(operation: str) -> None:
    values = (
        jnp.array([0.03, 0.05, 0.1, 0.2, 0.5, 1.0])
        if operation == "logcdf"
        else jnp.array([0.2, 0.5, 1.0, 2.0, 5.0, 20.0])
    )
    shape = jnp.asarray(3.5)
    scale = jnp.asarray(1.2)
    implementation = getattr(mmmjax, f"inverse_gamma_{operation}")
    jax_reference = getattr(JAX_REFERENCES["inverse_gamma"], operation)
    scipy_reference = getattr(stats.invgamma, operation)
    assert jax_reference is not None

    result = implementation(values, shape, scale)
    jax_result = jax_reference(values, shape, scale)
    scipy_result = scipy_reference(np.asarray(values), float(shape), scale=float(scale))

    _assert_close(result, jax_result)
    _assert_scipy_tail_close(result, scipy_result)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_uniform_log_probabilities_match_jax_and_scipy(operation: str) -> None:
    values = jnp.array([-1.75, -1.0, 0.5, 2.0, 2.75])
    lower = jnp.asarray(-2.0)
    upper = jnp.asarray(3.0)
    implementation = getattr(mmmjax, f"uniform_{operation}")
    jax_reference = getattr(JAX_REFERENCES["uniform"], operation)
    scipy_reference = getattr(stats.uniform, operation)
    assert jax_reference is not None

    result = implementation(values, lower, upper)
    jax_result = jax_reference(values, lower, upper)
    scipy_result = scipy_reference(
        np.asarray(values),
        loc=float(lower),
        scale=float(upper - lower),
    )

    _assert_close(result, jax_result)
    _assert_close(result, scipy_result)


@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_uniform_log_probability_gradients_match_jax(operation: str, differentiate) -> None:
    arguments = (jnp.asarray(-0.75), jnp.asarray(-2.0), jnp.asarray(3.0))
    implementation = getattr(mmmjax, f"uniform_{operation}")
    reference = getattr(JAX_REFERENCES["uniform"], operation)
    argnums = (0, 1, 2)
    assert reference is not None

    result = jax.jit(differentiate(implementation, argnums=argnums))(*arguments)
    jax_result = jax.jit(differentiate(reference, argnums=argnums))(*arguments)

    for result_jacobian, jax_jacobian in zip(result, jax_result, strict=True):
        _assert_close(result_jacobian, jax_jacobian)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_uniform_deep_tail_values_and_parameter_gradients_match_jax(operation: str) -> None:
    negative_log_probabilities = jnp.array([4.0, 8.0, 16.0, 32.0])
    tail_probabilities = jnp.exp(-negative_log_probabilities)
    implementation = getattr(mmmjax, f"uniform_{operation}")
    reference = getattr(JAX_REFERENCES["uniform"], operation)
    assert reference is not None

    if operation == "logcdf":
        values, lower, upper = tail_probabilities, jnp.asarray(0.0), jnp.asarray(1.0)
    else:
        values, lower, upper = -tail_probabilities, jnp.asarray(-1.0), jnp.asarray(0.0)

    result = implementation(values, lower, upper)
    jax_result = reference(values, lower, upper)

    def summed(function, current_lower, current_upper):
        return jnp.sum(function(values, current_lower, current_upper))

    implementation_value, implementation_gradient = jax.value_and_grad(
        partial(summed, implementation),
        argnums=(0, 1),
    )(lower, upper)
    jax_value, jax_gradient = jax.value_and_grad(
        partial(summed, reference),
        argnums=(0, 1),
    )(lower, upper)

    _assert_close(result, -negative_log_probabilities)
    _assert_close(result, jax_result)
    _assert_close(implementation_value, jax_value)
    for result_gradient, jax_gradient_value in zip(implementation_gradient, jax_gradient, strict=True):
        _assert_close(result_gradient, jax_gradient_value)


@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_half_normal_log_probability_gradients_match_jax(operation: str, differentiate) -> None:
    arguments = (jnp.asarray(1.25), jnp.asarray(1.7))
    implementation = getattr(mmmjax, f"half_normal_{operation}")
    reference = getattr(JAX_REFERENCES["half_normal"], operation)
    argnums = (0, 1)
    assert reference is not None

    result = jax.jit(differentiate(implementation, argnums=argnums))(*arguments)
    jax_result = jax.jit(differentiate(reference, argnums=argnums))(*arguments)

    for result_jacobian, jax_jacobian in zip(result, jax_result, strict=True):
        _assert_close(result_jacobian, jax_jacobian)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
@pytest.mark.parametrize("standardized_value", [1.0, 1.25, 4.0], ids=["boundary", "ordinary", "tail"])
def test_mmmjax_half_normal_log_probability_hessian_matches_distribution_identity(
    operation: str,
    standardized_value: float,
) -> None:
    scale = 1.7
    parameters = jnp.array([standardized_value * scale, scale])
    implementation = getattr(mmmjax, f"half_normal_{operation}")

    def implementation_from_array(current):
        return implementation(current[0], current[1])

    probability_function = stats.halfnorm.cdf if operation == "logcdf" else stats.halfnorm.sf
    direction = 1 if operation == "logcdf" else -1
    density_ratio = stats.halfnorm.pdf(standardized_value) / probability_function(standardized_value)
    standardized_gradient = direction * density_ratio
    standardized_curvature = -standardized_value * standardized_gradient - standardized_gradient**2
    expected = (
        np.array(
            [
                [
                    standardized_curvature,
                    -(standardized_gradient + standardized_value * standardized_curvature),
                ],
                [
                    -(standardized_gradient + standardized_value * standardized_curvature),
                    2 * standardized_value * standardized_gradient + standardized_value**2 * standardized_curvature,
                ],
            ]
        )
        / scale**2
    )

    result = jax.jit(jax.hessian(implementation_from_array))(parameters)

    tolerance = 5e-11 if result.dtype == jnp.dtype(jnp.float64) else 3e-5
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("dtype", "standardized_value"),
    [
        pytest.param(jnp.float32, 12.0, id="float32"),
        pytest.param(jnp.float64, 30.0, id="float64"),
    ],
)
def test_mmmjax_half_normal_deep_tail_gradients_match_scipy_mills_ratio(
    dtype,
    standardized_value: float,
) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")

    scale = 1.0
    parameters = jnp.array([standardized_value * scale, scale], dtype=dtype)
    inverse_mills = math.sqrt(2 / math.pi) / special.erfcx(standardized_value / math.sqrt(2))
    expected = jnp.array(
        [-inverse_mills / scale, standardized_value * inverse_mills / scale],
        dtype=dtype,
    )

    def logsf_from_array(current):
        return mmmjax.half_normal_logsf(current[0], current[1])

    forward = jax.jacfwd(logsf_from_array)(parameters)
    reverse = jax.jacrev(logsf_from_array)(parameters)

    tolerance = 5e-10 if dtype == jnp.float64 else 5e-6
    np.testing.assert_allclose(forward, expected, rtol=tolerance, atol=0)
    np.testing.assert_allclose(reverse, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_normal_log_probability_gradients_match_jax(operation: str, differentiate) -> None:
    arguments = (jnp.asarray(1.25), jnp.asarray(-0.3), jnp.asarray(1.7))
    implementation = getattr(mmmjax, f"normal_{operation}")
    reference = getattr(jax_stats.norm, operation)
    argnums = (0, 1, 2)

    result = jax.jit(differentiate(implementation, argnums=argnums))(*arguments)
    jax_result = jax.jit(differentiate(reference, argnums=argnums))(*arguments)

    for result_jacobian, jax_jacobian in zip(result, jax_result, strict=True):
        _assert_close(result_jacobian, jax_jacobian)


@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_lognormal_log_probability_gradients_match_jax_transform(operation: str, differentiate) -> None:
    arguments = (jnp.asarray(1.25), jnp.asarray(-0.3), jnp.asarray(1.7))
    implementation = getattr(mmmjax, f"lognormal_{operation}")
    normal_reference = getattr(jax_stats.norm, operation)
    argnums = (0, 1, 2)

    def reference(value, location, scale):
        return normal_reference(jnp.log(value), loc=location, scale=scale)

    result = jax.jit(differentiate(implementation, argnums=argnums))(*arguments)
    jax_result = jax.jit(differentiate(reference, argnums=argnums))(*arguments)

    for result_jacobian, jax_jacobian in zip(result, jax_result, strict=True):
        _assert_close(result_jacobian, jax_jacobian)


@pytest.mark.parametrize("distribution", ["normal", "lognormal"])
@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_log_probability_second_derivatives_match_jax(distribution: str, operation: str) -> None:
    parameters = jnp.array([1.25, -0.3, 1.7])
    implementation = getattr(mmmjax, f"{distribution}_{operation}")
    normal_reference = getattr(jax_stats.norm, operation)

    def implementation_from_array(current):
        return implementation(current[0], current[1], current[2])

    def reference_from_array(current):
        value = jnp.log(current[0]) if distribution == "lognormal" else current[0]
        return normal_reference(value, loc=current[1], scale=current[2])

    result = jax.jit(jax.hessian(implementation_from_array))(parameters)
    jax_result = jax.jit(jax.hessian(reference_from_array))(parameters)

    _assert_close(result, jax_result)


@pytest.mark.parametrize(
    ("dtype", "magnitude", "scale"),
    [
        pytest.param(jnp.float32, 20.0, 1.0, id="float32"),
        pytest.param(jnp.float64, 100.0, 1.0, id="float64"),
    ],
)
@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_normal_deep_tail_gradients_match_scipy_mills_ratio(
    dtype,
    magnitude: float,
    scale: float,
    operation: str,
) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")

    direction = 1 if operation == "logcdf" else -1
    arguments = jnp.array([direction * -magnitude, 0.0, scale], dtype=dtype)
    standardized_magnitude = magnitude / scale
    inverse_mills = math.sqrt(2 / math.pi) / special.erfcx(standardized_magnitude / math.sqrt(2))
    expected = jnp.array(
        [
            direction * inverse_mills / scale,
            -direction * inverse_mills / scale,
            inverse_mills * standardized_magnitude / scale,
        ],
        dtype=dtype,
    )
    implementation = getattr(mmmjax, f"normal_{operation}")

    forward = jax.jacfwd(lambda current: implementation(current[0], current[1], current[2]))(arguments)
    reverse = jax.jacrev(lambda current: implementation(current[0], current[1], current[2]))(arguments)

    _assert_scipy_tail_close(forward, expected)
    _assert_scipy_tail_close(reverse, expected)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_normal_curvature_matches_scipy_mills_ratio(operation: str) -> None:
    direction = 1 if operation == "logcdf" else -1
    value = direction * -2.0
    scale = 1.0
    standardized_magnitude = abs(value) / scale
    inverse_mills = math.sqrt(2 / math.pi) / special.erfcx(standardized_magnitude / math.sqrt(2))
    expected = -inverse_mills * (inverse_mills - standardized_magnitude) / scale**2
    implementation = getattr(mmmjax, f"normal_{operation}")

    result = jax.grad(jax.grad(lambda current: implementation(current, 0.0, scale)))(value)

    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=0)


@pytest.mark.parametrize("operation", ["logcdf", "logsf"])
def test_mmmjax_lognormal_deep_tail_gradients_match_scipy_mills_ratio(operation: str) -> None:
    direction = 1 if operation == "logcdf" else -1
    location = direction * 20.0
    scale = 1.0
    arguments = jnp.array([1.0, location, scale], dtype=jnp.float32)
    standardized_magnitude = abs(location) / scale
    inverse_mills = math.sqrt(2 / math.pi) / special.erfcx(standardized_magnitude / math.sqrt(2))
    expected = jnp.array(
        [
            direction * inverse_mills / scale,
            -direction * inverse_mills / scale,
            inverse_mills * standardized_magnitude / scale,
        ],
        dtype=jnp.float32,
    )
    implementation = getattr(mmmjax, f"lognormal_{operation}")

    forward = jax.jacfwd(lambda current: implementation(current[0], current[1], current[2]))(arguments)
    reverse = jax.jacrev(lambda current: implementation(current[0], current[1], current[2]))(arguments)

    _assert_scipy_tail_close(forward, expected)
    _assert_scipy_tail_close(reverse, expected)


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        pytest.param(
            "beta",
            (jnp.full((2, 1), 2.5), jnp.full((3,), 3.5)),
            id="beta",
        ),
        pytest.param("exponential", (jnp.full((2, 3), 1.3),), id="exponential"),
        pytest.param(
            "gamma",
            (jnp.full((2, 1), 2.5), jnp.full((3,), 1.3)),
            id="gamma",
        ),
        pytest.param("half_normal", (jnp.full((2, 3), 1.3),), id="half-normal"),
        pytest.param(
            "inverse_gamma",
            (jnp.full((2, 1), 3.5), jnp.full((3,), 1.2)),
            id="inverse-gamma",
        ),
        pytest.param(
            "laplace",
            (jnp.full((2, 1), 0.2), jnp.full((3,), 1.3)),
            id="laplace",
        ),
        pytest.param(
            "lognormal",
            (jnp.full((2, 1), 0.2), jnp.full((3,), 0.8)),
            id="lognormal",
        ),
        pytest.param(
            "normal",
            (jnp.full((2, 1), 0.2), jnp.full((3,), 1.3)),
            id="normal",
        ),
        pytest.param(
            "student_t",
            (
                jnp.full((2, 1), 5.0),
                jnp.full((3,), 0.2),
                jnp.asarray(1.3),
            ),
            id="student-t",
        ),
        pytest.param(
            "uniform",
            (jnp.full((2, 1), -1.0), jnp.full((3,), 1.0)),
            id="uniform",
        ),
    ],
)
def test_mmmjax_rng_matches_jax_benchmark_contract(
    name: str,
    parameters: tuple[jax.Array, ...],
) -> None:
    reference = JAX_REFERENCES[name]
    implementation = getattr(mmmjax, f"{name}_rng")
    compiled_reference = jax.jit(partial(reference.rng, sample_shape=(4,)))

    result = implementation(jax.random.key(0), *parameters, sample_shape=(4,))
    jax_result = reference.rng(jax.random.key(0), *parameters, sample_shape=(4,))
    compiled_jax_result = compiled_reference(jax.random.key(0), *parameters)

    expected_dtype = jnp.result_type(*parameters)
    assert result.shape == (4, 2, 3)
    assert result.dtype == expected_dtype
    assert jnp.all(jnp.isfinite(result))
    assert jax_result.shape == result.shape
    assert jax_result.dtype == result.dtype
    assert jnp.all(jnp.isfinite(jax_result))
    assert compiled_jax_result.shape == result.shape
    assert compiled_jax_result.dtype == result.dtype
    assert jnp.all(jnp.isfinite(compiled_jax_result))


def _assert_close(actual, expected) -> None:
    actual_array = jnp.asarray(actual)
    tolerance = 1e-12 if actual_array.dtype == jnp.dtype(jnp.float64) else 3e-6
    np.testing.assert_allclose(actual_array, expected, rtol=tolerance, atol=tolerance)


def _assert_scipy_tail_close(actual, expected) -> None:
    actual_array = jnp.asarray(actual)
    tolerance = 5e-11 if actual_array.dtype == jnp.dtype(jnp.float64) else 3e-6
    np.testing.assert_allclose(actual_array, expected, rtol=tolerance, atol=0)
