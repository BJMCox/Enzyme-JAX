import math

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call


class HloGamma(absltest.TestCase):
    def test_lgamma_second_and_third_derivatives(self):
        source = """
        module {
          func.func @main(%x: tensor<f32>) -> tensor<f32> {
            %y = chlo.lgamma %x : tensor<f32> -> tensor<f32>
            return %y : tensor<f32>
          }
        }
        """
        imported = lambda x: hlo_call(x, source=source)[0]
        second = lambda x: jax.jvp(jax.grad(imported), (x,), (jnp.ones_like(x),))[1]
        x = jnp.float32(1)
        expected_second = math.pi**2 / 6
        expected_third = -2 * 1.2020569031595942854
        for derivative, expected in (
            (jax.grad(jax.grad(imported)), expected_second),
            (second, expected_second),
            (jax.grad(second), expected_third),
        ):
            np.testing.assert_allclose(
                jax.jit(derivative)(x), expected, rtol=3e-5, atol=3e-5
            )

    def test_polygamma_derivative_with_constant_order(self):
        source = """
        module {
          func.func @main(%x: tensor<f32>) -> tensor<f32> {
            %n = stablehlo.constant dense<2.0> : tensor<f32>
            %y = "chlo.polygamma"(%n, %x)
                : (tensor<f32>, tensor<f32>) -> tensor<f32>
            return %y : tensor<f32>
          }
        }
        """
        imported = lambda x: hlo_call(x, source=source)[0]
        forward = lambda x: jax.jvp(imported, (x,), (jnp.ones_like(x),))[1]
        for derivative in (jax.grad(imported), forward):
            np.testing.assert_allclose(
                jax.jit(derivative)(jnp.float32(1)),
                math.pi**4 / 15,
                rtol=3e-5,
                atol=3e-5,
            )

    def test_float32_lgamma_and_gradient_near_roots(self):
        # References are mpmath values rounded to float32. The CHLO
        # legalization misses them by 594 and 2.8e8 ulp for digamma at 1.45
        # and 1.4616321.
        source = """
        module {
          func.func @main(%x: tensor<5xf32>) -> tensor<5xf32> {
            %y = chlo.lgamma %x : tensor<5xf32> -> tensor<5xf32>
            return %y : tensor<5xf32>
          }
        }
        """
        x = jnp.array([0.999, 1.4616321, 1.9995, 1.45, 3.7], jnp.float32)
        lgamma_ref = np.array(
            [5.7803106e-04, -0.12148629, -2.1129624e-04, -0.12142059, 1.4280723],
            np.float32,
        )
        digamma_ref = np.array(
            [-0.5788618, -1.2036052e-08, 0.42246184, -0.011316376, 1.1671536],
            np.float32,
        )
        imported = lambda x: hlo_call(x, source=source)[0]
        gradient = jax.grad(lambda x: jnp.sum(imported(x)))
        for got, ref in (
            (jax.jit(imported)(x), lgamma_ref),
            (jax.jit(gradient)(x), digamma_ref),
        ):
            ulp = np.abs(np.asarray(got, np.float64) - ref) / np.spacing(np.abs(ref))
            self.assertLessEqual(float(np.max(ulp)), 2.0, (got, ref, ulp))

    def test_float32_lgamma_value_matches_under_ad(self):
        # A density and its gradient must evaluate the same lgamma, so the
        # value-only call agrees bit for bit with every differentiated path.
        source = """
        module {
          func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
            %y = chlo.lgamma %x : tensor<4xf32> -> tensor<4xf32>
            return %y : tensor<4xf32>
          }
        }
        """
        x = jnp.array([0.3, 1.4616321, 2.4, -2.457], jnp.float32)
        xs = jnp.stack([x, x + 0.25, x + 7.0])
        for passes in (None, ""):
            kwargs = {} if passes is None else {"passes": passes}
            imported = lambda x: hlo_call(x, source=source, **kwargs)[0]
            value = np.asarray(jax.jit(imported)(x))
            jvp_value = jax.jit(
                lambda x: jax.jvp(imported, (x,), (jnp.ones_like(x),))[0]
            )(x)
            vjp_value = jax.jit(lambda x: jax.vjp(imported, x)[0])(x)
            np.testing.assert_array_equal(np.asarray(jvp_value), value)
            np.testing.assert_array_equal(np.asarray(vjp_value), value)
            batched = np.asarray(jax.jit(jax.vmap(imported))(xs))
            for row, item in zip(batched, xs):
                np.testing.assert_array_equal(row, np.asarray(jax.jit(imported)(item)))


if __name__ == "__main__":
    absltest.main()
