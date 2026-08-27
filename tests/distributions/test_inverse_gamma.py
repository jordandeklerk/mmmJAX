"""Tests for Inverse Gamma distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mmmjax import (
    gamma_logpdf,
    inverse_gamma,
    inverse_gamma_logpdf,
    inverse_gamma_rng,
)


def test_inverse_gamma_logpdf_matches_known_values() -> None:
    values = jnp.array([0.25, 1.0, 3.0], dtype=jnp.float32)
    expected = jnp.array(
        [-0.9060819788978761, -0.6581122428174931, -3.369921919822543],
        dtype=jnp.float32,
    )

    result = inverse_gamma_logpdf(values, 2.5, 1.7)

    assert jnp.allclose(result, expected)


def test_inverse_gamma_logpdf_matches_scipy_reference_grid() -> None:
    values = np.array([1e-30, 0.01, 0.1, 0.75, 1.0, 4.0, 20.0, 1e20], dtype=np.float32)
    shapes = np.array([0.1, 0.5, 1.0, 2.5, 7.999, 8.0, 50.0, 3.0], dtype=np.float32)
    scales = np.array([1e-10, 0.1, 1.0, 1.7, 8.0, 25.0, 0.001, 1e10], dtype=np.float32)
    expected = stats.invgamma.logpdf(
        values.astype(np.float64),
        a=shapes.astype(np.float64),
        scale=scales.astype(np.float64),
    )

    result = inverse_gamma_logpdf(values, shapes, scales)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_inverse_gamma_logpdf_gradients_match_analytic_reference_grid() -> None:
    values = np.array([1e-4, 0.2, 1.0, 3.0, 100.0], dtype=np.float32)
    shapes = np.array([0.2, 1.0, 2.5, 8.0, 50.0], dtype=np.float32)
    scales = np.array([1e-3, 0.7, 2.0, 8.0, 1.0], dtype=np.float32)
    expected = np.stack(
        [
            scales / values**2 - (shapes + 1) / values,
            np.log(scales) - np.log(values) - special.digamma(shapes),
            shapes / scales - 1 / values,
        ],
        axis=-1,
    )

    gradients = jax.vmap(jax.grad(inverse_gamma_logpdf, argnums=(0, 1, 2)))(values, shapes, scales)
    result = np.stack(gradients, axis=-1)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_inverse_gamma_logpdf_matches_stan_reference_values() -> None:
    result = inverse_gamma_logpdf(jnp.array([1.0, 0.5]), jnp.array([1.0, 2.9]), jnp.array([1.0, 3.1]))
    expected = jnp.array([-1.0, -0.8185294827413339])

    assert jnp.allclose(result, expected)


def test_inverse_gamma_returns_scalar_sum() -> None:
    result = inverse_gamma(jnp.array([0.25, 1.0, 3.0]), 2.5, 1.7)

    assert result.shape == ()
    assert jnp.allclose(result, -4.9341161415379124)


def test_inverse_gamma_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.25], [2.0]])
    shapes = jnp.array([0.5, 1.0, 3.0])
    scales = jnp.array([0.5, 1.0, 2.0])
    expected = jnp.array(
        [
            [-0.8394969869215134, -1.2274112777602189, -1.068528194400546],
            [-2.2086593040445908, -1.8862943611198906, -2.386294361119891],
        ]
    )

    result = inverse_gamma_logpdf(values, shapes, scales)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected)
    assert jnp.allclose(inverse_gamma(values, shapes, scales), jnp.sum(expected))


def test_inverse_gamma_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-jnp.inf, -1.0, -0.0, 0.0, jnp.inf, jnp.nan])

    result = inverse_gamma_logpdf(values, 2.5, 1.7)

    assert jnp.all(jnp.isneginf(result[:5]))
    assert jnp.isnan(result[5])


def test_inverse_gamma_logpdf_rejects_invalid_parameters_before_support_check() -> None:
    invalid_parameters = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_shapes = inverse_gamma_logpdf(-1.0, invalid_parameters, 1.0)
    invalid_scales = inverse_gamma_logpdf(-1.0, 1.0, invalid_parameters)

    assert jnp.all(jnp.isnan(invalid_shapes))
    assert jnp.all(jnp.isnan(invalid_scales))


def test_inverse_gamma_logpdf_obeys_scale_identity_at_extreme_values() -> None:
    value = jnp.float32(1.0)
    scale = jnp.float32(1.0)
    multiplier = jnp.asarray(jnp.finfo(jnp.float32).max)

    baseline = inverse_gamma_logpdf(value, 2.5, scale)
    scaled = inverse_gamma_logpdf(multiplier * value, 2.5, multiplier * scale)

    assert jnp.isfinite(scaled)
    assert jnp.allclose(scaled, baseline - jnp.log(multiplier))


def test_inverse_gamma_logpdf_remains_accurate_for_small_scale_ratio() -> None:
    value = jnp.float32(1.0)
    shape = jnp.float32(8.0)
    scale = jnp.float32(2.4e-7)

    result = inverse_gamma_logpdf(value, shape, scale)
    gradients = jax.grad(inverse_gamma, argnums=(0, 1, 2))(value, shape, scale)

    assert jnp.allclose(result, -130.46617126464844, rtol=0, atol=2e-5)
    assert jnp.allclose(
        jnp.asarray(gradients),
        jnp.array([-9.0, -17.258268356323242, 33333334.0]),
        rtol=2e-7,
        atol=0,
    )


def test_inverse_gamma_logpdf_remains_accurate_for_large_scale_ratio() -> None:
    value = jnp.float32(1.0)
    shape = jnp.float32(8.0)
    scale = jnp.float32(8e8)

    result = inverse_gamma_logpdf(value, shape, scale)
    gradients = jax.grad(inverse_gamma, argnums=(0, 1, 2))(value, shape, scale)

    assert jnp.allclose(result, -799999872.0, rtol=0, atol=64)
    assert jnp.allclose(
        jnp.asarray(gradients),
        jnp.array([800000000.0, 18.484479904174805, -1.0]),
        rtol=2e-7,
        atol=0,
    )


def test_inverse_gamma_logpdf_handles_maximum_finite_scale() -> None:
    value = jnp.float32(1.0)
    shape = jnp.float32(2.5)
    scale = jnp.asarray(jnp.finfo(jnp.float32).max)

    result = inverse_gamma_logpdf(value, shape, scale)
    gradients = jax.grad(inverse_gamma, argnums=(0, 1, 2))(value, shape, scale)

    assert jnp.isfinite(result)
    assert result == -scale
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))
    assert gradients[0] == scale
    assert jnp.allclose(gradients[1], 88.01968383789062, rtol=2e-7, atol=0)
    assert gradients[2] == -1


def test_inverse_gamma_logpdf_remains_accurate_near_large_shape_mode() -> None:
    shapes = jnp.array([1e5, 1e8], dtype=jnp.float32)
    expected = jnp.array([4.837523366028856, 8.29140183793818], dtype=jnp.float32)

    result = inverse_gamma_logpdf(1.0, shapes, shapes)
    gradients = jax.grad(inverse_gamma, argnums=(0, 1, 2))(
        jnp.float32(1.0),
        jnp.float32(1e8),
        jnp.float32(1e8),
    )

    assert jnp.allclose(result, expected, rtol=2e-6, atol=1e-6)
    assert jnp.allclose(gradients[0], -1.0)
    assert jnp.allclose(gradients[1], 5e-9, rtol=2e-6, atol=0)
    assert gradients[2] == 0


def test_inverse_gamma_logpdf_preserves_large_shape_off_mode_terms_and_gradients() -> None:
    value = jnp.float32(1.0)
    shape = jnp.float32(1e20)
    scale = jnp.float32(1.0001e20)

    result = inverse_gamma_logpdf(value, shape, scale)
    gradients = jax.grad(inverse_gamma, argnums=(0, 1, 2))(value, shape, scale)

    assert jnp.allclose(result, -500082442240.0, rtol=0, atol=32768)
    assert jnp.allclose(gradients[0], 1.0001157766250496e16, rtol=2e-6)
    assert jnp.allclose(gradients[1], 0.00010000657483397652, rtol=2e-6)
    assert jnp.allclose(gradients[2], -0.00010000157434316688, rtol=2e-6)


def test_inverse_gamma_logpdf_is_continuous_across_previous_series_cutoff() -> None:
    value = jnp.float32(1.0)
    shape = jnp.float32(1e8)
    scales = jnp.array([110517088.0, 110517096.0], dtype=jnp.float32)

    result = jax.vmap(inverse_gamma_logpdf, in_axes=(None, None, 0))(value, shape, scales)
    gradients = jax.vmap(
        jax.grad(inverse_gamma, argnums=(0, 1, 2)),
        in_axes=(None, None, 0),
    )(value, shape, scales)

    assert jnp.allclose(
        result,
        jnp.array([-517083.15625, -517083.90625]),
        rtol=0,
        atol=0.125,
    )
    assert result[1] < result[0]
    assert jnp.allclose(
        jnp.stack(gradients),
        jnp.array(
            [
                [10517087.0, 10517095.0],
                [0.09999997168779373, 0.1000000461935997],
                [-0.09516254812479019, -0.09516261518001556],
            ]
        ),
        rtol=2e-7,
        atol=0,
    )


def test_inverse_gamma_logpdf_is_monotonic_across_deviance_regions() -> None:
    lower_tail_scales = jnp.array([36787940.0, 36787944.0, 36787948.0], dtype=jnp.float32)
    upper_tail_scales = jnp.array([271828160.0, 271828192.0, 271828224.0], dtype=jnp.float32)

    lower_tail = inverse_gamma_logpdf(1.0, jnp.float32(1e8), lower_tail_scales)
    upper_tail = inverse_gamma_logpdf(1.0, jnp.float32(1e8), upper_tail_scales)

    assert jnp.all(jnp.diff(lower_tail) > 0)
    assert jnp.all(jnp.diff(upper_tail) < 0)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_inverse_gamma_logpdf_remains_accurate_at_extreme_float64_shape() -> None:
    shape = jnp.float64(1e20)

    result = inverse_gamma_logpdf(jnp.float64(1.0), shape, shape)

    assert jnp.allclose(result, 22.106912396735784, rtol=1e-14, atol=0)


def test_inverse_gamma_matches_reciprocal_gamma_change_of_variables() -> None:
    values = jnp.array([0.25, 1.0, 3.0])
    shape = 2.5
    scale = 1.7
    expected = gamma_logpdf(1 / values, shape, scale) - 2 * jnp.log(values)

    result = inverse_gamma_logpdf(values, shape, scale)

    assert jnp.allclose(result, expected)


def test_inverse_gamma_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([0.25, 1.0, 3.0])
    expected = jnp.array([13.2, -1.8, -0.9777777777777777])

    result = jax.grad(lambda current_values: inverse_gamma(current_values, 2.5, 1.7))(values)

    assert jnp.allclose(result, expected)


def test_inverse_gamma_is_differentiable_with_respect_to_shape() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = jax.grad(lambda current_shape: inverse_gamma(values, current_shape, 1.7))(2.5)

    assert jnp.allclose(result, -0.22990309629743777)


def test_inverse_gamma_is_differentiable_with_respect_to_scale() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = jax.grad(lambda current_scale: inverse_gamma(values, 2.5, current_scale))(1.7)

    assert jnp.allclose(result, -0.9215686274509807)


def test_inverse_gamma_gradients_ignore_values_outside_support() -> None:
    values = jnp.array([-1.0, 1.0, jnp.inf])

    gradients = jax.grad(inverse_gamma, argnums=(0, 1, 2))(values, 2.5, 1.7)

    assert jnp.allclose(gradients[0], jnp.array([0.0, -1.8, 0.0]))
    assert jnp.allclose(gradients[1], -0.17252856)
    assert jnp.allclose(gradients[2], 0.4705882)


def test_inverse_gamma_logpdf_supports_forward_mode_differentiation() -> None:
    primals = (jnp.float32(1.0), jnp.float32(2.5), jnp.float32(1.7))
    tangents = (jnp.float32(0.2), jnp.float32(-0.3), jnp.float32(0.4))

    _, result = jax.jvp(inverse_gamma_logpdf, primals, tangents)

    assert jnp.allclose(result, -0.12000614178)


def test_inverse_gamma_logpdf_supports_higher_order_differentiation() -> None:
    parameters = jnp.array([1.0, 2.5, 1.7], dtype=jnp.float32)
    expected = jnp.array(
        [
            [0.1, -1.0, 1.0],
            [-1.0, -0.49035776, 0.5882353],
            [1.0, 0.5882353, -0.8650519],
        ]
    )

    result = jax.hessian(lambda arguments: inverse_gamma_logpdf(*arguments))(parameters)

    assert jnp.allclose(result, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(
    "arguments",
    [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (jnp.nan, 1.0, 1.0),
    ],
)
def test_inverse_gamma_logpdf_gradients_propagate_invalid_inputs(arguments) -> None:
    gradients = jax.grad(inverse_gamma_logpdf, argnums=(0, 1, 2))(*arguments)

    assert jnp.all(jnp.isnan(jnp.asarray(gradients)))


def test_inverse_gamma_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.25, 1.0], [0.5, 2.0]])
    shapes = jnp.array([1.5, 3.0])
    scales = jnp.array([0.5, 2.0])

    result = jax.vmap(inverse_gamma)(values, shapes, scales)
    expected = jnp.stack(
        [inverse_gamma(value, shape, scale) for value, shape, scale in zip(values, shapes, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_inverse_gamma_rng_inverts_log_space_unit_rate_draws() -> None:
    key = jax.random.key(42)
    shapes = jnp.array([0.5, 2.5], dtype=jnp.float32)
    scales = jnp.array([1.7, 0.8], dtype=jnp.float32)
    expected = jnp.exp(jnp.log(scales) - jax.random.loggamma(key, shapes, shape=(3, 2), dtype=jnp.float32))

    result = inverse_gamma_rng(key, shapes, scales, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result > 0)


def test_inverse_gamma_rng_uses_broadcast_parameter_shape() -> None:
    shapes = jnp.ones((2, 1))
    scales = jnp.ones(3)

    result = inverse_gamma_rng(jax.random.key(0), shapes, scales, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_inverse_gamma_rng_preserves_scaled_draw_for_tiny_shape() -> None:
    key = jax.random.key(7)
    shape = jnp.float32(0.03)
    scale = jnp.float32(1e-30)
    log_unit_gamma = jax.random.loggamma(key, shape, dtype=jnp.float32)

    assert jnp.array_equal(inverse_gamma_rng(key, shape, scale), jnp.exp(jnp.log(scale) - log_unit_gamma))
    assert jnp.isfinite(inverse_gamma_rng(key, shape, scale))


def test_inverse_gamma_rng_can_be_jitted_with_dynamic_parameters() -> None:
    key = jax.random.key(3)
    shapes = jnp.array([2.0, 4.0])
    scales = jnp.array([1.5, 3.0])
    compiled = jax.jit(
        lambda current_key, shape, scale: inverse_gamma_rng(current_key, shape, scale, sample_shape=(3,))
    )

    result = compiled(key, shapes, scales)

    assert jnp.array_equal(result, inverse_gamma_rng(key, shapes, scales, sample_shape=(3,)))


def test_inverse_gamma_rng_matches_distribution_moments() -> None:
    samples = inverse_gamma_rng(jax.random.key(7), 6.0, 5.0, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 1.0, rtol=0, atol=0.02)
    assert jnp.allclose(jnp.var(samples), 0.25, rtol=0, atol=0.025)


def test_inverse_gamma_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        inverse_gamma_rng(jax.random.key(0), jnp.ones(2), jnp.ones(3))
