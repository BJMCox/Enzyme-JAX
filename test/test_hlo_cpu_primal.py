from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax.primitives import LANG_MHLO, XLAPipeline, ffi_call


class HloCpuPrimal(absltest.TestCase):
    def test_multiple_outputs_and_input_alias(self):
        if jax.default_backend() != "cpu":
            self.skipTest("CPU primal kernel")
        source = """
        module {
          func.func @main(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
              -> (tensor<32x32xf32>, tensor<32x32xf32>) {
            %product = stablehlo.dot_general %a, %b,
              contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST]
              : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
            %shifted = stablehlo.add %product, %a : tensor<32x32xf32>
            return %a, %shifted : tensor<32x32xf32>, tensor<32x32xf32>
          }
        }
        """
        shape = jax.core.ShapedArray((32, 32), np.dtype("float32"))
        abi = (None, ((0, 0), (1, 1)), ((0, -1), (1, -1)), source, ())
        function = jax.jit(
            lambda a, b: ffi_call(
                a,
                b,
                source=abi,
                fn="main",
                out_shapes=(shape, shape),
                lang=LANG_MHLO,
                pipeline_options=XLAPipeline(),
            )
        )
        a = jnp.arange(1024, dtype=jnp.float32).reshape(32, 32) / 32
        scales = jnp.arange(1, 33, dtype=jnp.float32)
        b = jnp.roll(jnp.diag(scales), 1, axis=0)

        def check(value):
            original, product = function(value, b)
            np.testing.assert_array_equal(original, value)
            np.testing.assert_array_equal(
                product, jnp.roll(value, -1, axis=1) * scales + value
            )

        check(a)  # Compile before sharing the cached executable across threads.
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(check, [a + i for i in range(4)]))
        np.testing.assert_array_equal(
            a, np.arange(1024, dtype=np.float32).reshape(32, 32) / 32
        )


if __name__ == "__main__":
    absltest.main()
