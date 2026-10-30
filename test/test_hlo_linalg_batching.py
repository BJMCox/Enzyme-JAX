import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import hlo_call

SOURCE = """
module {
  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %l = stablehlo.cholesky %a, lower = true : tensor<2x2xf32>
    %y = "stablehlo.triangular_solve"(%l, %b) <{
      left_side = true, lower = true, unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    }> : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    return %y : tensor<2x1xf32>
  }
}
"""


class HloLinalgBatching(absltest.TestCase):
    def test_native_matrices_keep_tensor_batches(self):
        factors = np.array(
            [[[2, 0], [1, 3]], [[1, 0], [-0.5, 2]], [[3, 0], [0.2, 1]]],
            dtype=np.float32,
        )
        matrices = factors @ factors.swapaxes(-1, -2)
        rhs = np.arange(6, dtype=np.float32).reshape(3, 2, 1) / 3 + 1
        imported = lambda a, b: hlo_call(a, b, source=SOURCE)[0]
        mapped = jax.vmap(imported)
        expected = np.linalg.solve(factors, rhs)
        np.testing.assert_allclose(jax.jit(mapped)(matrices, rhs), expected, rtol=2e-6)
        lowered = jax.jit(mapped).lower(matrices, rhs)
        self.assertNotIn("stablehlo.while", str(lowered.compiler_ir()))

        seeds = jnp.arange(6, dtype=jnp.float32).reshape(3, 2, 1) / 5 + 0.3
        gradient = jax.jit(jax.grad(lambda b: jnp.sum(mapped(matrices, b) * seeds)))
        expected_gradient = np.linalg.solve(factors.swapaxes(-1, -2), seeds)
        np.testing.assert_allclose(gradient(rhs), expected_gradient, rtol=2e-6)

        shared = jax.vmap(imported, in_axes=(None, 0))
        np.testing.assert_allclose(
            jax.jit(shared)(matrices[0], rhs),
            np.linalg.solve(factors[0], rhs),
            rtol=2e-6,
        )
        nested = jax.vmap(mapped)
        np.testing.assert_allclose(
            jax.jit(nested)(np.stack([matrices, matrices]), np.stack([rhs, 2 * rhs])),
            np.stack([expected, 2 * expected]),
            rtol=2e-6,
        )


if __name__ == "__main__":
    absltest.main()
