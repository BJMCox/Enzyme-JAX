import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import hlo_call


class HloLogNdtr(absltest.TestCase):
    def test_value_slope_and_nested_ad(self):
        source = """
        module {
          func.func private @log_ndtr(%x: tensor<3xf32>)
              -> (tensor<3xf32>, tensor<3xf32>) {
            %scale = stablehlo.constant dense<-0.7071067812> : tensor<3xf32>
            %half = stablehlo.constant dense<0.5> : tensor<3xf32>
            %norm = stablehlo.constant dense<0.3989422804> : tensor<3xf32>
            %z = stablehlo.multiply %x, %scale : tensor<3xf32>
            %erfc = chlo.erfc %z : tensor<3xf32> -> tensor<3xf32>
            %cdf = stablehlo.multiply %erfc, %half : tensor<3xf32>
            %y = stablehlo.log %cdf : tensor<3xf32>
            %neg_square = stablehlo.multiply %z, %z : tensor<3xf32>
            %exponent = stablehlo.negate %neg_square : tensor<3xf32>
            %weight = stablehlo.exponential %exponent : tensor<3xf32>
            %density = stablehlo.multiply %weight, %norm : tensor<3xf32>
            %slope = stablehlo.divide %density, %cdf : tensor<3xf32>
            return %y, %slope : tensor<3xf32>, tensor<3xf32>
          }
          func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
            %result:2 = stablehlo.composite "enzyme.math.log_ndtr" %x
                {decomposition = @log_ndtr, version = 1 : i32}
                : (tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
            return %result#0 : tensor<3xf32>
          }
        }
        """

        def score(x):
            return hlo_call(x, source=source)[0]

        def total(x):
            return score(x).sum()

        x = jnp.array([-2.0, 0.0, 1.0], dtype=jnp.float32)
        expected = jax.scipy.special.log_ndtr(x)
        slope = jnp.exp(-0.5 * x**2 - 0.5 * jnp.log(2 * jnp.pi) - expected)
        np.testing.assert_allclose(jax.jit(score)(x), expected, rtol=3e-6)
        np.testing.assert_allclose(jax.jit(jax.grad(total))(x), slope, rtol=3e-6)
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(score))(x), np.diag(slope), rtol=3e-6
        )
        curvature = -slope * (x + slope)
        np.testing.assert_allclose(
            jax.jit(jax.jacrev(jax.grad(total)))(x), np.diag(curvature), rtol=4e-6
        )
        np.testing.assert_allclose(
            jax.jit(jax.vmap(score))(jnp.stack((x, x))),
            jnp.stack((expected, expected)),
            rtol=3e-6,
        )


if __name__ == "__main__":
    absltest.main()
