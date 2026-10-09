from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call


class HloSlice(absltest.TestCase):
    def test_shared_strided_selection(self):
        source = """
        module {
          func.func @main(%x: tensor<8xf32>) -> (tensor<4xf32>, tensor<3xf32>) {
            %y = stablehlo.exponential %x : tensor<8xf32>
            %a = stablehlo.slice %y [0:7:2] : (tensor<8xf32>) -> tensor<4xf32>
            %b = stablehlo.slice %y [2:7:2] : (tensor<8xf32>) -> tensor<3xf32>
            return %a, %b : tensor<4xf32>, tensor<3xf32>
          }
        }
        """
        x = jnp.arange(8, dtype=jnp.float32) / 8
        selected = lambda x: hlo_call(
            x,
            source=source,
            passes="enzyme-hlo-generate-td{patterns=slice_elementwise<1>},"
            "transform-interpreter,enzyme-hlo-remove-transform",
        )
        first, second = jax.jit(selected)(x)
        expected = np.exp(np.asarray(x))
        np.testing.assert_allclose(first, expected[0:7:2], rtol=2e-6)
        np.testing.assert_allclose(second, expected[2:7:2], rtol=2e-6)
        density = lambda x: sum(jnp.sum(value) for value in selected(x))
        weights = np.array([1, 0, 2, 0, 2, 0, 2, 0])
        np.testing.assert_allclose(
            jax.jit(jax.grad(density))(x), weights * expected, rtol=2e-6
        )


if __name__ == "__main__":
    absltest.main()
