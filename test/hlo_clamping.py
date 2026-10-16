from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call
from enzyme_ad.jax.primitives import optimization_passes


class HLOClamping(parameterized.TestCase):
    @parameterized.product(
        operation=("dynamic_slice", "dynamic_update_slice"),
        case=(
            ("ui16", "65535", "0", 1, 0),
            ("ui32", "2147483648", "0", 1, 0),
            ("ui64", "18446744073709551615", "0", 1, 0),
            ("i32", "-1", "2147483647", 0, 2),
        ),
    )
    def test_constant_indices(self, operation, case):
        index_type, first, second, row, column = case
        update = operation == "dynamic_update_slice"
        signature = "%x: tensor<2x3xf32>, %u: tensor<1x1xf32>"
        result_type = "2x3xf32" if update else "1x1xf32"
        if update:
            body = f"""stablehlo.dynamic_update_slice %x, %u, %i, %j :
              (tensor<2x3xf32>, tensor<1x1xf32>, tensor<{index_type}>,
               tensor<{index_type}>) -> tensor<2x3xf32>"""
        else:
            signature = "%x: tensor<2x3xf32>"
            body = f"""stablehlo.dynamic_slice %x, %i, %j, sizes = [1, 1] :
              (tensor<2x3xf32>, tensor<{index_type}>, tensor<{index_type}>)
              -> tensor<1x1xf32>"""
        source = f"""func.func @main({signature}) -> tensor<{result_type}> {{
          %i = stablehlo.constant dense<{first}> : tensor<{index_type}>
          %j = stablehlo.constant dense<{second}> : tensor<{index_type}>
          %out = {body}
          return %out : tensor<{result_type}>
        }}"""
        function = lambda *args: hlo_call(
            *args, source=source, passes=optimization_passes()
        )[0]
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        if update:
            arguments = (jnp.asarray(x), jnp.array([[9]], dtype=jnp.float32))
            expected = x.copy()
            expected[row, column] = 9
            weights = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
            dx = weights.copy()
            dx[row, column] = 0
            gradients = (dx, np.array([[weights[row, column]]], dtype=np.float32))
        else:
            arguments = (jnp.asarray(x),)
            expected = x[row : row + 1, column : column + 1]
            weights = np.ones((1, 1), dtype=np.float32)
            dx = np.zeros_like(x)
            dx[row, column] = 1
            gradients = (dx,)
        np.testing.assert_array_equal(jax.jit(function)(*arguments), expected)
        score = lambda *args: jnp.sum(function(*args) * weights)
        actual = jax.jit(jax.grad(score, argnums=tuple(range(len(arguments)))))(
            *arguments
        )
        for value, reference in zip(actual, gradients, strict=True):
            np.testing.assert_array_equal(value, reference)


if __name__ == "__main__":
    absltest.main()
