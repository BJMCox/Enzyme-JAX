from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import re
from jaxlib.mlir import ir

from enzyme_ad.jax import hlo_call

HAS_EFFECT_QUERIES = (
    hasattr(getattr(ir, "MemoryEffectsOpInterface", None), "get_effects")
    and hasattr(ir.Operation, "has_trait")
    and hasattr(ir, "RecursiveMemoryEffectsTrait")
)


def import_hlo(function, *arguments):
    source = str(jax.jit(function).lower(*arguments).compiler_ir("stablehlo"))
    return lambda *args: hlo_call(*args, source=source)


def recurrence(parameter):
    def step(state, observation):
        state = jnp.tanh(state * parameter + observation)
        return state, state

    _, states = jax.lax.scan(step, jnp.float32(0.2), jnp.linspace(-0.3, 0.7, 32))
    return states.sum()


class HLOLinearize(absltest.TestCase):
    @absltest.skipUnless(
        HAS_EFFECT_QUERIES,
        "requires MLIR memory effect queries",
    )
    def test_scan_value_and_grad_shares_forward(self):
        point = jnp.float32(0.4)
        imported = import_hlo(recurrence, point)
        compiled = (
            jax.jit(jax.value_and_grad(lambda x: imported(x)[0])).lower(point).compile()
        )
        expected = jax.jit(jax.value_and_grad(recurrence))(point)
        np.testing.assert_allclose(compiled(point), expected, rtol=3e-6, atol=3e-6)
        # One forward traversal saves values for one reverse traversal.
        self.assertEqual(len(re.findall(r"\bwhile\(", compiled.as_text())), 2)

    def test_reusable_pullback_and_primal_dependent_seed(self):
        point = jnp.float32(0.4)
        imported = import_hlo(recurrence, point)

        def reuse(x):
            value, pullback = jax.vjp(lambda p: imported(p)[0], x)
            return value, pullback(jnp.float32(1))[0], pullback(value)[0]

        value, gradient = jax.jit(jax.value_and_grad(recurrence))(point)
        np.testing.assert_allclose(
            jax.jit(reuse)(point), (value, gradient, value * gradient), rtol=3e-6
        )

    def test_constant_input_and_region_capture(self):
        def function(x, data):
            return jax.lax.cond(
                x > 0, lambda: jnp.sin(x * data).sum(), lambda: jnp.cos(x + data).sum()
            )

        point, data = jnp.float32(0.4), jnp.arange(1, 4, dtype=jnp.float32)
        imported = import_hlo(function, point, data)
        actual = jax.jit(jax.value_and_grad(lambda x, y: imported(x, y)[0]))
        eager = jax.value_and_grad(jax.jit(lambda x, y: imported(x, y)[0]))
        expected = jax.jit(jax.value_and_grad(function))
        for x in (point, -point):
            np.testing.assert_allclose(actual(x, data), expected(x, data), rtol=3e-6)
            np.testing.assert_allclose(eager(x, data), expected(x, data), rtol=3e-6)

    def test_inactive_nonfinite_output(self):
        point = jnp.float32(-2)
        imported = import_hlo(lambda x: (x * x, jnp.sqrt(x)), point)
        np.testing.assert_allclose(
            jax.jit(jax.value_and_grad(lambda x: imported(x)[0]))(point), (4, -4)
        )

    def test_multiple_active_outputs(self):
        x, y = jnp.float32(0.7), jnp.float32(-1.3)
        function = lambda x, y: (x * y, jnp.sin(x) + y * y)
        imported = import_hlo(function, x, y)

        def objective(call, x, y):
            first, second = call(x, y)
            return first * first + jnp.cos(second)

        actual = jax.jit(
            jax.value_and_grad(lambda x, y: objective(imported, x, y), argnums=(0, 1))
        )
        expected = jax.jit(
            jax.value_and_grad(lambda x, y: objective(function, x, y), argnums=(0, 1))
        )
        np.testing.assert_allclose(
            jax.tree.leaves(actual(x, y)), jax.tree.leaves(expected(x, y)), rtol=3e-6
        )

    def test_inactive_unsupported_derivative(self):
        source = """
        module {
          func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %y = stablehlo.cholesky %x, lower = true : tensor<2x2xf32>
            return %y : tensor<2x2xf32>
          }
        }
        """

        def objective(x):
            matrix = jnp.diag(jnp.array([x, 4 * x]))
            value = hlo_call(matrix, source=source)[0].sum()
            return x * x + jax.lax.stop_gradient(value)

        np.testing.assert_allclose(
            jax.jit(jax.value_and_grad(objective))(jnp.float32(4)), (22, 8)
        )

    @absltest.skipUnless(
        HAS_EFFECT_QUERIES,
        "requires MLIR memory effect queries",
    )
    def test_residuals_remain_differentiable(self):
        point = jnp.float32(0.4)
        imported = import_hlo(lambda x: jnp.sin(x) * x * x, point)

        def hessian_vector(x):
            return jax.jvp(jax.grad(lambda x: imported(x)[0]), (x,), (jnp.float32(1),))[
                1
            ]

        expected = (
            2 * jnp.sin(point)
            + 4 * point * jnp.cos(point)
            - point * point * jnp.sin(point)
        )
        np.testing.assert_allclose(jax.jit(hessian_vector)(point), expected, rtol=3e-6)

    def test_module_symbols_retain_the_original_path(self):
        source = """
        module {
          sdy.mesh @mesh = <["axis"=1]>
          func.func @main(%x: tensor<f32>) -> tensor<f32> {
            %y = stablehlo.multiply %x, %x : tensor<f32>
            return %y : tensor<f32>
          }
        }
        """
        function = lambda x: hlo_call(x, source=source)[0]
        np.testing.assert_allclose(
            jax.jit(jax.value_and_grad(function))(jnp.float32(2)), (4, 4)
        )


if __name__ == "__main__":
    absltest.main()
