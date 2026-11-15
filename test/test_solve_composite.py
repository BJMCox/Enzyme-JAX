from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)
SOURCE = Path(__file__).with_name("solve_composite.mlir").read_text()
MATRIX_SOURCE = Path(__file__).with_name("solve_composite_2x2.mlir").read_text()


class SolveCompositeTest(absltest.TestCase):
    def test_solve_composite(self):
        @jax.jit
        def solve(a, b):
            x, _ = hlo_call(a.reshape(1, 1), b.reshape(1, 1), source=SOURCE)
            return x.reshape(())

        a, b = jnp.float64(2), jnp.float64(3)
        np.testing.assert_allclose(solve(a, b), 1.5)
        _, tangent = jax.jvp(solve, (a, b), (jnp.float64(1), jnp.float64(2)))
        np.testing.assert_allclose(tangent, 0.25)
        da, db = jax.grad(solve, argnums=(0, 1))(a, b)
        np.testing.assert_allclose((da, db), (-0.75, 0.5))
        _, hessian_vector = jax.jvp(
            jax.grad(solve, argnums=(0, 1)),
            (a, b),
            (jnp.float64(1), jnp.float64(2)),
        )
        np.testing.assert_allclose(hessian_vector, (0.25, -0.25))
        x, valid = jax.jit(lambda a, b: hlo_call(a, b, source=SOURCE))(
            jnp.zeros((1, 1), dtype=jnp.float64),
            jnp.ones((1, 1), dtype=jnp.float64),
        )
        np.testing.assert_equal((np.asarray(x).item(), bool(valid)), (0.0, False))

    def test_solve_composite_mapped_shared_and_nested(self):
        def solve(a, b):
            x, _ = hlo_call(a.reshape(1, 1), b.reshape(1, 1), source=SOURCE)
            return x.reshape(())

        a = jnp.array([2.0, 3.0, 4.0])
        b = jnp.array([1.0, 2.0, 5.0])
        mapped = jax.jit(jax.vmap(solve))
        lowered = mapped.lower(a, b)
        np.testing.assert_allclose(lowered.compile()(a, b), b / a, rtol=2e-12)
        shared = jax.vmap(solve, in_axes=(None, 0))
        np.testing.assert_allclose(jax.jit(shared)(a[0], b), b / a[0], rtol=2e-12)
        grad = jax.jit(jax.grad(lambda x: shared(x, b).sum()))(a[0])
        np.testing.assert_allclose(grad, -np.sum(np.asarray(b)) / 4, rtol=2e-12)
        nested = jax.jit(jax.vmap(jax.vmap(solve)))
        aa = jnp.stack((a, a + 1))
        bb = jnp.stack((b, b + 2))
        np.testing.assert_allclose(nested(aa, bb), bb / aa, rtol=2e-12)

    def test_shared_batch_keeps_transitive_symbols_without_name_collisions(self):
        source = SOURCE.replace("@identity(", "@enzyme_batch_cell(").replace(
            "%av = stablehlo.reshape %a",
            "%copy = func.call @enzyme_batch_cell(%a) : "
            "(tensor<1x1xf64>) -> tensor<1x1xf64>\n"
            "    %av = stablehlo.reshape %copy",
        )

        def solve(a, b):
            return hlo_call(a.reshape(1, 1), b.reshape(1, 1), source=source)[0].reshape(
                ()
            )

        mapped = jax.vmap(solve, in_axes=(None, 0))
        a, b = jnp.float64(2), jnp.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(jax.jit(mapped)(a, b), b / a, rtol=2e-12)
        grad = jax.jit(jax.grad(lambda aa: mapped(aa, b).sum()))(a)
        np.testing.assert_allclose(grad, -np.sum(np.asarray(b)) / 4, rtol=2e-12)

    def test_nonsymmetric_matrix_rhs_derivatives_and_mixed_invalid(self):
        a = np.array([[0.0, 2.0], [1.0, 3.0]])
        b = np.array([[1.0, 2.0], [-2.0, 3.0]])
        da = np.array([[0.3, -0.4], [0.7, 0.2]])
        db = np.array([[0.1, 0.8], [-0.5, 0.6]])
        weight = np.array([[0.2, -0.3], [0.9, 0.5]])
        x = np.linalg.solve(a, b)
        dx = np.linalg.solve(a, db - da @ x)
        adjoint = np.linalg.solve(a.T, weight)
        dadjoint = -np.linalg.solve(a.T, da.T @ adjoint)
        expected_gradient = (-adjoint @ x.T, adjoint)
        expected_hvp = (-dadjoint @ x.T - adjoint @ dx.T, dadjoint)

        def solve(a, b):
            return hlo_call(a, b, source=MATRIX_SOURCE)[0]

        loss = lambda a, b: jnp.sum(solve(a, b) * weight)
        value, tangent = jax.jit(lambda a, b: jax.jvp(solve, (a, b), (da, db)))(a, b)
        gradient = jax.grad(loss, argnums=(0, 1))
        grad = jax.jit(gradient)(a, b)
        hvp = jax.jit(lambda a, b: jax.jvp(gradient, (a, b), (da, db))[1])(a, b)
        for actual, expected in zip(
            (value, tangent, *grad, *hvp), (x, dx, *expected_gradient, *expected_hvp)
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
        jacobian = jax.jit(jax.jacfwd(solve, argnums=1))(a, b)
        expected = np.einsum("ik,jl->ijkl", np.linalg.inv(a), np.eye(2))
        np.testing.assert_allclose(jacobian, expected, rtol=2e-11, atol=2e-11)

        matrices = np.stack((a, np.zeros_like(a)))
        mapped = jax.vmap(solve, in_axes=(0, None))
        value = jax.jit(mapped)(matrices, b)
        np.testing.assert_allclose(value, np.stack((x, np.zeros_like(x))), atol=2e-11)
        ga, gb = jax.jit(
            jax.grad(lambda aa, bb: jnp.sum(mapped(aa, bb) * weight), argnums=(0, 1))
        )(matrices, b)
        np.testing.assert_allclose(ga[0], expected_gradient[0], rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(ga[1], 0, atol=2e-11)
        np.testing.assert_allclose(gb, expected_gradient[1], rtol=2e-11, atol=2e-11)

    def test_unrelated_composite_uses_its_decomposition(self):
        source = """
        module {
          func.func @main(%x: tensor<f64>) -> tensor<f64> {
            %y = stablehlo.composite "example.square" %x {decomposition = @square}
              : (tensor<f64>) -> tensor<f64>
            return %y : tensor<f64>
          }
          func.func private @square(%x: tensor<f64>) -> tensor<f64> {
            %y = stablehlo.multiply %x, %x : tensor<f64>
            return %y : tensor<f64>
          }
        }
        """
        function = lambda x: hlo_call(x, source=source)[0]
        x = jnp.array([-2.0, 0.5, 3.0])
        value, gradient = jax.jit(jax.vmap(jax.value_and_grad(function)))(x)
        np.testing.assert_allclose(value, x * x, rtol=2e-12)
        np.testing.assert_allclose(gradient, 2 * x, rtol=2e-12)


if __name__ == "__main__":
    absltest.main()
