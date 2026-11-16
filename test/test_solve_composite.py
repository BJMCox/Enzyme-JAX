import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.interpreters import mlir
from jaxlib.mlir import ir

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)
SOURCE = Path(__file__).with_name("solve_composite.mlir").read_text()
MATRIX_SOURCE = Path(__file__).with_name("solve_composite_2x2.mlir").read_text()
SCAN_SOURCE = Path(__file__).with_name("solve_scan.mlir").read_text()


class SolveCompositeTest(parameterized.TestCase):
    @parameterized.named_parameters(("jax", False), ("stablehlo", True))
    def test_scan_reuses_factors_and_preserves_higher_derivatives(self, internal_scan):
        for precision in ("f32", "f64"):
            dtype = np.float32 if precision == "f32" else np.float64
            tolerance = 2e-5 if precision == "f32" else 2e-11
            source = (
                MATRIX_SOURCE.replace("0x7FF0000000000000", "0x7F800000")
                if precision == "f32"
                else MATRIX_SOURCE
            )
            source = source.replace("f64", precision)
            a = np.array([[0.3, 2.0], [1.0, 3.0]], dtype=dtype)
            b = np.array([[1.0, 2.0], [-2.0, 3.0]], dtype=dtype)
            da = np.array([[0.3, -0.4], [0.7, 0.2]], dtype=dtype)
            db = np.array([[0.1, 0.8], [-0.5, 0.6]], dtype=dtype)
            weight = np.array([[0.2, -0.3], [0.9, 0.5]], dtype=dtype)
            for transpose in (False, True):
                program = source
                if transpose:
                    program = source.replace(
                        "    %x, %valid = func.call",
                        f"    %at = stablehlo.transpose %a, dims = [1, 0] : (tensor<2x2x{precision}>) -> tensor<2x2x{precision}>\n    %x, %valid = func.call",
                        1,
                    ).replace(
                        "@__enzyme_linalg_solve_v1_matrix(%a, %b)",
                        "@__enzyme_linalg_solve_v1_matrix(%at, %b)",
                        1,
                    )

                def loss(a, b):
                    def step(total, i):
                        x, _ = hlo_call(
                            a + i * 0.001 * jnp.eye(2, dtype=a.dtype), b, source=program
                        )
                        return total + jnp.sum(x * weight), None

                    return jax.lax.scan(
                        step,
                        jnp.zeros((), dtype=a.dtype),
                        jnp.arange(32, dtype=a.dtype),
                    )[0]

                if internal_scan:
                    with mlir.make_ir_context(), ir.Location.unknown():
                        module = ir.Module.parse(program)
                        single = module.body.operations[0]
                        single.attributes["sym_name"] = ir.StringAttr.get("single")
                        single.attributes["sym_visibility"] = ir.StringAttr.get(
                            "private"
                        )
                        wrapper = ir.Module.parse(SCAN_SOURCE.replace("f64", precision))
                        module.body.append(wrapper.body.operations[1].operation.clone())
                        module.operation.verify()
                        wrapped = str(module)
                    loss = lambda a, b: hlo_call(
                        a, b, jnp.asarray(weight), source=wrapped
                    )[0]

                value = 0.0
                grad = [np.zeros((2, 2)), np.zeros((2, 2))]
                hvp = [np.zeros((2, 2)), np.zeros((2, 2))]
                for i in range(32):
                    matrix = a.astype(float) + i * 0.001 * np.eye(2)
                    delta = da.astype(float)
                    if transpose:
                        matrix, delta = matrix.T, delta.T
                    x = np.linalg.solve(matrix, b.astype(float))
                    dx = np.linalg.solve(matrix, db.astype(float) - delta @ x)
                    adjoint = np.linalg.solve(matrix.T, weight.astype(float))
                    dadjoint = -np.linalg.solve(matrix.T, delta.T @ adjoint)
                    ga = -adjoint @ x.T
                    ha = -dadjoint @ x.T - adjoint @ dx.T
                    value += np.sum(x * weight)
                    grad[0] += ga.T if transpose else ga
                    grad[1] += adjoint
                    hvp[0] += ha.T if transpose else ha
                    hvp[1] += dadjoint
                compiled = (
                    jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
                    .lower(a, b)
                    .compile()
                )
                actual = compiled(a, b)
                gradient = jax.grad(loss, argnums=(0, 1))
                actual_hvp = jax.jit(
                    lambda a, b: jax.jvp(gradient, (a, b), (da, db))[1]
                )(a, b)
                reverse_hvp = jax.jit(
                    jax.grad(
                        lambda a, b: sum(
                            jnp.sum(g * direction)
                            for g, direction in zip(gradient(a, b), (da, db))
                        ),
                        argnums=(0, 1),
                    )
                )(a, b)
                for actual, expected in zip(
                    jax.tree.leaves((actual, actual_hvp, reverse_hvp)),
                    [value, *grad, *hvp, *hvp],
                ):
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                if (
                    jax.default_backend() in ("cpu", "gpu")
                    and hasattr(
                        getattr(ir, "MemoryEffectsOpInterface", None), "get_effects"
                    )
                    and hasattr(ir.Operation, "has_trait")
                    and hasattr(ir, "RecursiveMemoryEffectsTrait")
                ):
                    self.assertLen(
                        re.findall(
                            r'custom_call_target="[^"\n]*getrf[^"\n]*"',
                            compiled.as_text(),
                        ),
                        1,
                    )

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

        nonfinite = a.copy()
        nonfinite[0, 1] = np.nan
        matrices = np.stack((a, np.zeros_like(a), nonfinite))
        mapped = jax.vmap(solve, in_axes=(0, None))
        value = jax.jit(mapped)(matrices, b)
        np.testing.assert_allclose(
            value, np.stack((x, np.zeros_like(x), np.zeros_like(x))), atol=2e-11
        )
        ga, gb = jax.jit(
            jax.grad(lambda aa, bb: jnp.sum(mapped(aa, bb) * weight), argnums=(0, 1))
        )(matrices, b)
        np.testing.assert_allclose(ga[0], expected_gradient[0], rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(ga[1], 0, atol=2e-11)
        np.testing.assert_allclose(ga[2], 0, atol=2e-11)
        np.testing.assert_allclose(gb, expected_gradient[1], rtol=2e-11, atol=2e-11)

    def test_native_solve_accepts_small_nonzero_pivots(self):
        source = MATRIX_SOURCE.replace("0x7FF0000000000000", "0x7F800000").replace(
            "f64", "f32"
        )
        function = jax.jit(lambda a, b: hlo_call(a, b, source=source))
        tiny = np.float32(2**-30)
        matrix = np.diag(np.array([1, tiny], dtype=np.float32))
        rhs = np.array([[1, 2], [tiny, -tiny]], dtype=np.float32)
        value, valid = function(matrix, rhs)
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(value, [[1, 2], [1, -1]], rtol=1e-6, atol=1e-6)

        mapped = jax.jit(jax.vmap(lambda a, b: hlo_call(a, b, source=source)))
        values, valid = mapped(
            np.stack((np.zeros_like(matrix), matrix)),
            np.stack((np.eye(2, dtype=np.float32), rhs)),
        )
        np.testing.assert_array_equal(valid, [False, True])
        np.testing.assert_array_equal(values[0], np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_allclose(values[1], [[1, 2], [1, -1]], rtol=1e-6, atol=1e-6)

    def test_native_adjoint_keeps_the_primal_validity(self):
        source = MATRIX_SOURCE.replace("0x7FF0000000000000", "0x7F800000").replace(
            "f64", "f32"
        )
        # A rounded LU pivot can vanish only after transposing the matrix.
        source = source.replace(
            "%nonzero = stablehlo.compare NE, %det, %zero, FLOAT",
            """%abs00 = stablehlo.abs %a00 : tensor<f32>
    %abs10 = stablehlo.abs %a10 : tensor<f32>
    %swap = stablehlo.compare GT, %abs10, %abs00, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %pivot = stablehlo.select %swap, %a10, %a00 : tensor<i1>, tensor<f32>
    %upper = stablehlo.select %swap, %a11, %a01 : tensor<i1>, tensor<f32>
    %lower = stablehlo.select %swap, %a00, %a10 : tensor<i1>, tensor<f32>
    %last = stablehlo.select %swap, %a01, %a11 : tensor<i1>, tensor<f32>
    %multiplier = stablehlo.divide %lower, %pivot : tensor<f32>
    %product = stablehlo.multiply %multiplier, %upper : tensor<f32>
    %rounded = stablehlo.optimization_barrier %product : tensor<f32>
    %trailing = stablehlo.subtract %last, %rounded : tensor<f32>
    %nonzero = stablehlo.compare NE, %trailing, %zero, FLOAT""",
        )
        matrix = np.array(
            [[1, 7], [np.nextafter(np.float32(3), np.float32(np.inf)), 21]],
            dtype=np.float32,
        )
        rhs = np.eye(2, dtype=np.float32)
        function = jax.jit(lambda a, b: hlo_call(a, b, source=source))
        _, valid = function(matrix, rhs)
        self.assertTrue(bool(valid))
        gradient = jax.jit(
            jax.grad(lambda a, b: jnp.trace(function(a, b)[0]), argnums=1)
        )(matrix, rhs)
        a = matrix.astype(np.float64)
        g = np.asarray(gradient, dtype=np.float64)
        residual = np.linalg.norm(a.T @ g - rhs, ord=np.inf)
        scale = np.linalg.norm(a.T, ord=np.inf) * np.linalg.norm(g, ord=np.inf) + 1
        self.assertLess(residual / scale, 8 * np.finfo(np.float32).eps)

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
