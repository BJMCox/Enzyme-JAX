import itertools

import jax
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import hlo_call


def imported_cholesky(dtype, lower, passes=None):
    source = f"""
    module {{
      func.func @main(%a: tensor<2x2x{dtype}>) -> tensor<2x2x{dtype}> {{
        %l = stablehlo.cholesky %a, lower = {str(lower).lower()} : tensor<2x2x{dtype}>
        return %l : tensor<2x2x{dtype}>
      }}
    }}
    """
    options = {} if passes is None else {"passes": passes}
    return lambda a: hlo_call(a, source=source, **options)[0]


class HloLinalgLowering(absltest.TestCase):
    def test_cholesky_target_lowering_and_raw_import(self):
        for scalar, dtype, off_diagonal in (
            ("f32", np.float32, 0.7),
            ("complex<f32>", np.complex64, 0.7 + 0.4j),
        ):
            factor = np.array([[2, 0], [off_diagonal, 3]], dtype=dtype)
            matrix = factor @ factor.T.conj()
            for lower in (True, False):
                expected = factor if lower else factor.T.conj()
                function = imported_cholesky(scalar, lower)
                np.testing.assert_allclose(
                    jax.jit(function)(matrix), expected, atol=2e-6
                )
                raw = jax.jit(imported_cholesky(scalar, lower, passes="")).lower(matrix)
                self.assertIn("stablehlo.cholesky", str(raw.compiler_ir()))
                if jax.default_backend() in ("cpu", "gpu"):
                    lowered = str(jax.jit(function).lower(matrix).compiler_ir())
                    self.assertNotIn("stablehlo.cholesky", lowered)
                    self.assertIn("stablehlo.custom_call", lowered)

    def test_triangular_solve_attributes(self):
        for left, lower, unit, transpose in itertools.product(
            (True, False),
            (True, False),
            (True, False),
            ("NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"),
        ):
            matrix = np.array(
                [[2 + 1j, 0.7 + 0.5j], [-0.4 + 0.3j, 3 - 0.2j]], np.complex64
            )
            triangular = np.tril(matrix) if lower else np.triu(matrix)
            if unit:
                np.fill_diagonal(triangular, 1)
            effective = triangular
            if transpose == "TRANSPOSE":
                effective = triangular.T
            elif transpose == "ADJOINT":
                effective = triangular.T.conj()
            rhs = np.array([[1 - 0.4j], [2 + 0.7j]], np.complex64)
            if not left:
                rhs = rhs.T
            expected = (
                np.linalg.solve(effective, rhs)
                if left
                else np.linalg.solve(effective.T, rhs.T).T
            )
            rhs_type = "2x1" if left else "1x2"
            source = f"""
            module {{
              func.func @main(%a: tensor<2x2xcomplex<f32>>, %b: tensor<{rhs_type}xcomplex<f32>>)
                -> tensor<{rhs_type}xcomplex<f32>> {{
                %y = "stablehlo.triangular_solve"(%a, %b) <{{
                  left_side = {str(left).lower()}, lower = {str(lower).lower()},
                  unit_diagonal = {str(unit).lower()}, transpose_a = #stablehlo<transpose {transpose}>
                }}> : (tensor<2x2xcomplex<f32>>, tensor<{rhs_type}xcomplex<f32>>)
                  -> tensor<{rhs_type}xcomplex<f32>>
                return %y : tensor<{rhs_type}xcomplex<f32>>
              }}
            }}
            """
            function = lambda a, b, source=source: hlo_call(a, b, source=source)[0]
            np.testing.assert_allclose(
                jax.jit(function)(matrix, rhs), expected, rtol=3e-6, atol=3e-6
            )

    def test_nested_call_and_loop(self):
        source = """
        module {
          func.func private @factor(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %l = stablehlo.cholesky %a, lower = true : tensor<2x2xf32>
            return %l : tensor<2x2xf32>
          }
          func.func @main(%a: tensor<2x2xf32>, %n: tensor<i32>) -> tensor<2x2xf32> {
            %zero = stablehlo.constant dense<0> : tensor<i32>
            %one = stablehlo.constant dense<1> : tensor<i32>
            %r:2 = stablehlo.while(%i = %zero, %x = %a) : tensor<i32>, tensor<2x2xf32>
            cond {
              %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
              stablehlo.return %c : tensor<i1>
            } do {
              %l = func.call @factor(%x) : (tensor<2x2xf32>) -> tensor<2x2xf32>
              %j = stablehlo.add %i, %one : tensor<i32>
              stablehlo.return %j, %l : tensor<i32>, tensor<2x2xf32>
            }
            return %r#1 : tensor<2x2xf32>
          }
        }
        """
        function = jax.jit(lambda a, n: hlo_call(a, n, source=source)[0])
        matrix = np.diag(np.array([16, 81], np.float32))
        for count in (1, 2):
            expected = np.diag(np.diag(matrix) ** (0.5**count))
            np.testing.assert_allclose(
                function(matrix, np.int32(count)), expected, rtol=2e-6
            )


if __name__ == "__main__":
    absltest.main()
