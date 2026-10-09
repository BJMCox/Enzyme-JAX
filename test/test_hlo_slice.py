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

    def test_strided_adjoint_padding(self):
        dtype = "f64" if jax.config.x64_enabled else "f32"
        cases = [
            ((8,), (0,), (8,), (2,)),
            ((9,), (2,), (9,), (3,)),
            ((5, 8), (1, 0), (5, 8), (3, 2)),
            ((8,), (3,), (3,), (2,)),
            ((0, 3), (0, 1), (0, 3), (2, 2)),
        ]
        for shape, starts, limits, strides in cases:
            with self.subTest(shape=shape, starts=starts, strides=strides):
                axes = tuple(zip(starts, limits, strides))
                selection = tuple(slice(*axis) for axis in axes)
                sizes = tuple(len(range(*axis)) for axis in axes)
                input_type = "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
                output_type = "tensor<" + "x".join(map(str, sizes)) + "x" + dtype + ">"
                indices = ", ".join(":".join(map(str, axis)) for axis in axes)
                source = f"""
                module {{
                  func.func @main(%x: {input_type}) -> {output_type} {{
                    %y = stablehlo.slice %x [{indices}] : ({input_type}) -> {output_type}
                    return %y : {output_type}
                  }}
                }}
                """

                def loss(x):
                    y = hlo_call(x, source=source)[0]
                    return jnp.sum(y * y)

                x = jnp.arange(np.prod(shape), dtype=float).reshape(shape) / 8
                direction = jnp.ones_like(x)
                expected = np.zeros(shape)
                expected[selection] = 2 * np.asarray(x)[selection]
                gradient = jax.grad(loss)
                np.testing.assert_array_equal(jax.jit(gradient)(x), expected)
                expected_hvp = np.zeros(shape)
                expected_hvp[selection] = 2
                hvp = jax.jit(lambda x: jax.jvp(gradient, (x,), (direction,))[1])(x)
                np.testing.assert_array_equal(hvp, expected_hvp)
                mapped = jax.jit(jax.vmap(gradient))(jnp.stack([x, 2 * x]))
                np.testing.assert_array_equal(
                    mapped, np.stack([expected, 2 * expected])
                )


if __name__ == "__main__":
    absltest.main()
