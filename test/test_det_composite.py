from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)


def source_for(precision):
    source = Path(__file__).with_name("logabsdet_composite.mlir").read_text()
    source = source.replace("__enzyme_linalg_logabsdet_v1_", "__enzyme_linalg_det_v1_")
    source = source.replace("%value = stablehlo.log %magnitude : tensor<f64>", "")
    source = source.replace("return %value,", "return %det,")
    source = source.replace(
        "%nan = stablehlo.constant dense<0x7FF8000000000000>",
        "%nan = stablehlo.constant dense<0.0>",
    )
    # Preserve the polynomial derivative when the inverse formula is invalid.
    helper = source[source.index("  func.func private @__enzyme_linalg_det_v1_") :]
    helper = helper.rsplit("}", 1)[0].replace(
        "__enzyme_linalg_det_v1_matrix", "portable_det"
    )
    source = source.rsplit("}", 1)[0] + helper + "}"
    source = source.replace(
        "return %det, %inverse, %valid : tensor<f64>, tensor<2x2xf64>, tensor<i1>",
        """%result = "stablehlo.if"(%valid) ({
          stablehlo.return %value : tensor<f64>
        }, {
          %fallback:3 = func.call @portable_det(%a)
            : (tensor<2x2xf64>) -> (tensor<f64>, tensor<2x2xf64>, tensor<i1>)
          stablehlo.return %fallback#0 : tensor<f64>
        }) : (tensor<i1>) -> tensor<f64>
        return %result, %inverse, %valid : tensor<f64>, tensor<2x2xf64>, tensor<i1>""",
        1,
    )
    return (
        source.replace("0x7FF0000000000000", "0x7F800000").replace("f64", "f32")
        if precision == "f32"
        else source
    )


class DetCompositeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("f32", "f32", np.float32, 3e-5), ("f64", "f64", np.float64, 3e-11)
    )
    def test_guarded_determinant_cofactors_and_higher_derivatives(
        self, precision, dtype, tolerance
    ):
        source = source_for(precision)
        function = lambda a: hlo_call(a, source=source)[0]
        matrices = np.array(
            [[[0.3, 2], [1, 3]], [[0, 0], [0, 3]], [[0, 0], [0, 0]]], dtype=dtype
        )
        directions = np.array([[[0.2, -0.1], [0.4, 0.3]]] * 3, dtype=dtype)
        cofactor = lambda a: np.array([[a[1, 1], -a[1, 0]], [-a[0, 1], a[0, 0]]])
        expected_values = (
            matrices[:, 0, 0] * matrices[:, 1, 1]
            - matrices[:, 0, 1] * matrices[:, 1, 0]
        )
        expected_gradient = np.stack([cofactor(a) for a in matrices])
        expected_hvp = np.stack([cofactor(a) for a in directions])
        for f in (function, jax.checkpoint(function)):
            value, gradient = jax.jit(jax.vmap(jax.value_and_grad(f)))(matrices)
            hvp = jax.jit(jax.vmap(lambda a, d: jax.jvp(jax.grad(f), (a,), (d,))[1]))(
                matrices, directions
            )
            for actual, expected in zip(
                (value, gradient, hvp),
                (expected_values, expected_gradient, expected_hvp),
            ):
                np.testing.assert_allclose(
                    actual, expected, rtol=tolerance, atol=tolerance
                )
        # Separate auxiliary activity must retain the inverse's own derivative.
        a = matrices[0]
        inverse = np.linalg.inv(a)
        joint = lambda a: (
            hlo_call(a, source=source)[0] + hlo_call(a, source=source)[1].sum()
        )
        expected = cofactor(a) - inverse.T @ np.ones_like(a) @ inverse.T
        np.testing.assert_allclose(
            jax.jit(jax.grad(joint))(a), expected, rtol=tolerance, atol=tolerance
        )


if __name__ == "__main__":
    absltest.main()
