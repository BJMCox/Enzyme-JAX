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


if __name__ == "__main__":
    absltest.main()
