import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from enzyme_ad.jax import enzyme_call, hlo_call

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64


def native_cholesky(shape, dtype, lower):
    tensor_type = "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
    source = f"""
    module {{
      func.func @main(%a: {tensor_type}) -> {tensor_type} {{
        %l = stablehlo.cholesky %a, lower = {str(lower).lower()} : {tensor_type}
        return %l : {tensor_type}
      }}
    }}
    """
    return lambda a: hlo_call(a, source=source)[0]


class HloCholesky(parameterized.TestCase):
    @parameterized.product(lower=(True, False), forward=(True, False))
    def test_native_width_two(self, lower, forward):
        # Explicit Enzyme width, separate from JAX's choice of Jacobian batching.
        source = f"""
        func.func private @factor(%a: tensor<3x2x2xf64>) -> tensor<3x2x2xf64> {{
          %l = stablehlo.cholesky %a, lower = {str(lower).lower()}
            : tensor<3x2x2xf64>
          return %l : tensor<3x2x2xf64>
        }}
        func.func @main(%a: tensor<3x2x2xf64>, %seed: tensor<2x3x2x2xf64>)
            -> tensor<2x3x2x2xf64> {{
          %d = enzyme.{"fwddiff" if forward else "autodiff"} @factor(%a, %seed) <{{
            activity = [#enzyme.activity<enzyme_{"dup" if forward else "active"}>],
            ret_activity = [#enzyme.activity<enzyme_{"dup" if forward else "active"}noneed>],
            width = 2 : i64
          }}> : (tensor<3x2x2xf64>, tensor<2x3x2x2xf64>) -> tensor<2x3x2x2xf64>
          return %d : tensor<2x3x2x2xf64>
        }}
        """
        _, lowered = enzyme_call.run_pass_pipeline(
            [],
            source,
            "enzyme,inline,canonicalize,remove-unnecessary-enzyme-ops,arith-raise,symbol-dce",
        )
        from jax.interpreters import mlir
        from jaxlib.mlir import ir

        with mlir.make_ir_context():
            module = ir.Module.parse(lowered)
            for function in module.body:
                if function.sym_name.value == "main":
                    function.attributes["sym_visibility"] = ir.StringAttr.get("public")
            lowered = str(module)

        # Three distinct factorizations, L = [[p, 0], [q, r]].
        p = np.array([2.0, 3.0, 1.0])
        q = np.array([1.0, -1.0, 0.5])
        r = np.array([3.0, 2.0, 1.5])
        a = np.array([[[4, 37], [2, 10]], [[9, 41], [-3, 5]], [[1, 43], [0.5, 2.5]]])
        seed = np.array(
            [
                [[[3, 7], [-2, 5]], [[-1, 8], [0.25, -0.5]], [[0.5, 9], [0.75, -1]]],
                [
                    [[-0.4, 10], [0.6, -0.8]],
                    [[0.2, 11], [-0.3, 0.7]],
                    [[-1.2, 12], [0.4, 0.9]],
                ],
            ]
        )
        expected = np.zeros_like(seed)
        if forward:
            # Differentiate p=sqrt(a00), q=a10/p, r=sqrt(a11-q*q).
            dp = seed[..., 0, 0] / (2 * p)
            dq = (seed[..., 1, 0] - q * dp) / p
            dr = (seed[..., 1, 1] - 2 * q * dq) / (2 * r)
            expected[..., 0, 0] = dp
            expected[..., 1, 0] = dq
            expected[..., 1, 1] = dr
        else:
            g00, g10, g11 = seed[..., 0, 0], seed[..., 1, 0], seed[..., 1, 1]
            expected[..., 0, 0] = (
                g00 / (2 * p) - g10 * q / (2 * p * p) + g11 * q * q / (2 * r * p * p)
            )
            expected[..., 1, 0] = g10 / p - g11 * q / (r * p)
            expected[..., 1, 1] = g11 / (2 * r)
        if not lower:
            a, seed, expected = (np.swapaxes(x, -1, -2) for x in (a, seed, expected))
        with enable_x64():
            actual = jax.jit(
                lambda a, seed: hlo_call(a, seed, source=lowered, passes="symbol-dce")[
                    0
                ]
            )(a, seed)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    @parameterized.product(dtype=(np.float32, np.float64), lower=(True, False))
    def test_selected_triangle_jvp_and_vjp(self, dtype, lower):
        # Independent derivatives of sqrt(a), b/sqrt(a), sqrt(c-b*b/a).
        with enable_x64():
            orient = lambda a: a if lower else a.T
            a = orient(np.array([[4, 73], [2, 10]], dtype=dtype))
            h = orient(np.array([[3, -17], [-2, 5]], dtype=dtype))
            g = orient(np.array([[2, 19], [-3, 4]], dtype=dtype))
            expected_l = orient(np.array([[2, 0], [1, 3]]))
            expected_jvp = orient(np.array([[3 / 4, 0], [-11 / 8, 31 / 24]]))
            expected_vjp = orient(np.array([[25 / 24, 0], [-13 / 6, 2 / 3]]))
            scalar = "f32" if dtype == np.float32 else "f64"
            imported = native_cholesky(a.shape, scalar, lower)
            forward = jax.jit(lambda a, h: jax.jvp(imported, (a,), (h,)))
            primal, tangent = forward(a, h)
            triangle = np.tril if lower else np.triu
            tolerance = 2e-6 if dtype == np.float32 else 2e-13
            np.testing.assert_allclose(triangle(primal), expected_l, rtol=tolerance)
            np.testing.assert_allclose(
                tangent, expected_jvp, rtol=tolerance, atol=tolerance
            )
            gradient = jax.jit(lambda a, g: jax.vjp(imported, a)[1](g)[0])(a, g)
            np.testing.assert_allclose(
                gradient, expected_vjp, rtol=tolerance, atol=tolerance
            )
            ignored = orient(np.array([[0, 1], [0, 0]], dtype=dtype))
            np.testing.assert_array_equal(forward(a, ignored)[1], np.zeros_like(a))
            expected_jacobian = np.zeros((2, 2, 2, 2))
            expected_jacobian[:, :, 0, 0] = [[1 / 4, 0], [-1 / 8, 1 / 24]]
            expected_jacobian[:, :, 1, 0] = [[0, 0], [1 / 2, -1 / 6]]
            expected_jacobian[:, :, 1, 1] = [[0, 0], [0, 1 / 6]]
            if not lower:
                expected_jacobian = expected_jacobian.transpose(1, 0, 3, 2)
            for jacobian in (jax.jacfwd(imported), jax.jacrev(imported)):
                np.testing.assert_allclose(
                    jax.jit(jacobian)(a),
                    expected_jacobian,
                    rtol=tolerance,
                    atol=tolerance,
                )

    @parameterized.parameters(True, False)
    def test_batched_active_covariance_and_hessian(self, lower):
        # logdet(S) has gradient S^-1 and Hessian direction -S^-1 H S^-1.
        with enable_x64():
            factor = np.array(
                [
                    [[[2.0, 0, 0], [0.7, 1.5, 0], [-0.4, 0.3, 1.2]]],
                    [[[1.1, 0, 0], [-0.2, 2.3, 0], [0.5, -0.6, 1.7]]],
                ]
            )
            transpose = lambda x: np.swapaxes(x, -1, -2)
            orient = lambda x: x if lower else transpose(x)
            covariance = factor @ transpose(factor)
            h = np.arange(18, dtype=np.float64).reshape(2, 1, 3, 3) / 19 - 0.4
            symmetric_h = np.tril(h) + transpose(np.tril(h, -1))
            inverse = np.linalg.inv(covariance)

            def fold(x):
                return orient(2 * np.tril(x, -1) + np.eye(3) * x)

            imported = native_cholesky(covariance.shape, "f64", lower)

            def loss(a):
                result = imported(a)
                return 2 * jnp.log(jnp.diagonal(result, axis1=-2, axis2=-1)).sum()

            a = covariance + orient(np.triu(np.full_like(covariance, 37), 1))
            direction = orient(h)
            gradient = jax.grad(loss)
            np.testing.assert_allclose(
                jax.jit(gradient)(a), fold(inverse), rtol=2e-12, atol=2e-12
            )
            expected_hvp = fold(-inverse @ symmetric_h @ inverse)
            for derivative in (
                lambda a: jax.jvp(gradient, (a,), (direction,))[1],
                jax.grad(lambda a: jax.jvp(loss, (a,), (direction,))[1]),
            ):
                np.testing.assert_allclose(
                    jax.jit(derivative)(a), expected_hvp, rtol=2e-11, atol=2e-11
                )
            triangle = np.tril if lower else np.triu
            np.testing.assert_allclose(
                triangle(jax.jit(imported)(a)),
                orient(factor),
                rtol=2e-13,
                atol=2e-13,
            )

    @parameterized.parameters(True, False)
    def test_second_directional_factor_derivative(self, lower):
        # Differentiate the defining factorization twice, independently of AD.
        factor = np.array([[2.0, 0, 0], [0.7, 1.5, 0], [-0.4, 0.3, 1.2]])
        covariance = factor @ factor.T
        h = np.array([[0.3, 7, 8], [-0.2, 0.7, 9], [0.5, 0.1, -0.4]])
        k = np.array([[-0.2, 1, 2], [0.6, -0.1, 3], [0.3, -0.7, 0.2]])

        def phi(x):
            return np.tril(x, -1) + np.eye(3) * x / 2

        def solve_both(x):
            return np.linalg.solve(factor, np.linalg.solve(factor, x).T).T

        def first(x):
            symmetric = np.tril(x) + np.tril(x, -1).T
            return factor @ phi(solve_both(symmetric))

        dh, dk = first(h), first(k)
        expected = factor @ phi(solve_both(-(dh @ dk.T + dk @ dh.T)))
        with enable_x64():
            orient = lambda x: x if lower else x.T
            imported = native_cholesky(covariance.shape, "f64", lower)
            first_direction = lambda a: jax.jvp(imported, (a,), (orient(h),))[1]
            mixed = lambda a: jax.jvp(first_direction, (a,), (orient(k),))[1]
            np.testing.assert_allclose(
                jax.jit(mixed)(orient(covariance)),
                orient(expected),
                rtol=2e-11,
                atol=2e-11,
            )


if __name__ == "__main__":
    absltest.main()
