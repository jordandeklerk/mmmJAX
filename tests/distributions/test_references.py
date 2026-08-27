"""Tests against established distribution implementations."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import mmmjax
from benchmarks.jax_references import JAX_REFERENCES


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
