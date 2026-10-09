from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import enzyme_call, hlo_call


def polynomial(x, axes):
    retained = tuple(i for i in range(x.ndim) if i not in axes)
    shape = tuple(x.shape[i] for i in retained)
    count = int(np.prod([x.shape[i] for i in axes]))
    terms = jnp.transpose(x, retained + tuple(sorted(axes))).reshape(*shape, count)
    result = jnp.ones(shape, dtype=x.dtype)
    for i in range(count):
        result = result * terms[..., i]
    return result


class ProductDerivatives(absltest.TestCase):
    def test_ordered_slice_product_preserves_finite_cotangents(self):
        large = 1e200 if jax.config.x64_enabled else 1e20
        x = jnp.array([[0.0, large, 1 / large, large]])
        seed = jnp.array([1e-10])

        def product(x):
            return ((x[:, 0] * x[:, 1]) * x[:, 2]) * x[:, 3]

        source = str(jax.jit(product).lower(x).compiler_ir())

        def imported(x):
            return hlo_call(x, source=source)[0]

        def pullback(x, seed):
            value, reverse = jax.vjp(imported, x)
            return value, reverse(seed)[0]

        value, gradient = jax.jit(pullback)(x, seed)
        np.testing.assert_array_equal(value, [0.0])
        np.testing.assert_allclose(
            gradient, [[large * 1e-10, 0.0, 0.0, 0.0]], rtol=3e-6, atol=0
        )

    def test_slice_product_zeros_and_mixed_hessians(self):
        x = jnp.array([[0.0, 2.0, 3.0], [0.0, 0.0, 3.0]])
        seed = jnp.array([7.0, -2.0])
        direction = jnp.array([[2.0, -1.0, 0.5], [1.0, 2.0, -0.5]])
        source = str(
            jax.jit(lambda x: x[:, 0] * x[:, 1] * x[:, 2]).lower(x).compiler_ir()
        )

        def imported(x):
            return hlo_call(x, source=source)[0]

        def gradient(x, seed):
            return jax.vjp(imported, x)[1](seed)[0]

        np.testing.assert_array_equal(
            jax.jit(gradient)(x, seed), [[42.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        expected = [[-14.0, 42.0, 28.0], [-12.0, -6.0, 0.0]]
        np.testing.assert_array_equal(
            jax.jit(
                lambda x, seed, dx: jax.jvp(lambda x: gradient(x, seed), (x,), (dx,))[1]
            )(x, seed, direction),
            expected,
        )
        np.testing.assert_array_equal(
            jax.jit(jax.grad(lambda x, seed, dx: jnp.vdot(gradient(x, seed), dx)))(
                x, seed, direction
            ),
            expected,
        )

    def test_default_passes_preserve_wide_product_hessians(self):
        for size in (5, 9):
            x = jnp.linspace(0.7, 1.3, size).at[:2].set(0)
            direction = jnp.linspace(-0.3, 0.8, size)
            source = str(jax.jit(jnp.prod).lower(x).compiler_ir())

            def imported(x):
                return hlo_call(x, source=source)[0]

            def hessian_vector(function):
                gradient = jax.grad(function)
                return (
                    jax.jit(lambda x: jax.jvp(gradient, (x,), (direction,))[1])(x),
                    jax.jit(jax.grad(lambda x: jnp.vdot(gradient(x), direction)))(x),
                )

            for actual, expected in zip(
                hessian_vector(imported),
                hessian_vector(lambda x: polynomial(x, (0,))),
            ):
                np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)

    def test_native_width_two(self):
        dtype = "f64" if jax.config.x64_enabled else "f32"
        x = jnp.array([[0.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 0.0, 0.0]])
        direction = jnp.arange(20.0).reshape(2, 2, 5) / 10
        seeds = jnp.array([[2.0, 7.0], [-3.0, 1.0]])
        for mode in ("fwddiff", "autodiff"):
            forward = mode == "fwddiff"
            activity = "dup" if forward else "active"
            incoming = f"tensor<2x2x5x{dtype}>" if forward else f"tensor<2x2x{dtype}>"
            outgoing = f"tensor<2x2x{dtype}>" if forward else f"tensor<2x2x5x{dtype}>"
            source = f"""
            func.func private @product(%x: tensor<2x5x{dtype}>) -> tensor<2x{dtype}> {{
              %one = stablehlo.constant dense<1.0> : tensor<{dtype}>
              %p = stablehlo.reduce(%x init: %one) applies stablehlo.multiply
                across dimensions = [1] : (tensor<2x5x{dtype}>, tensor<{dtype}>) -> tensor<2x{dtype}>
              return %p : tensor<2x{dtype}>
            }}
            func.func @main(%x: tensor<2x5x{dtype}>, %seed: {incoming}) -> {outgoing} {{
              %d = enzyme.{mode} @product(%x, %seed) <{{
                activity = [#enzyme.activity<enzyme_{activity}>],
                ret_activity = [#enzyme.activity<enzyme_{activity}noneed>],
                width = 2 : i64
              }}> : (tensor<2x5x{dtype}>, {incoming}) -> {outgoing}
              return %d : {outgoing}
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
                        function.attributes["sym_visibility"] = ir.StringAttr.get(
                            "public"
                        )
                lowered = str(module)
            actual = jax.jit(
                lambda x, seed: hlo_call(x, seed, source=lowered, passes="symbol-dce")[
                    0
                ]
            )(x, direction if forward else seeds)
            function = lambda x: polynomial(x, (1,))
            expected = (
                jax.vmap(lambda dx: jax.jvp(function, (x,), (dx,))[1])(direction)
                if forward
                else jax.vmap(lambda seed: jax.vjp(function, x)[1](seed)[0])(seeds)
            )
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)

    def test_no_axes_ignores_active_init(self):
        dtype = "f64" if jax.config.x64_enabled else "f32"
        source = f"""
        func.func @main(%x: tensor<3x{dtype}>, %init: tensor<{dtype}>) -> tensor<3x{dtype}> {{
          %p = stablehlo.reduce(%x init: %init) applies stablehlo.multiply
            across dimensions = [] : (tensor<3x{dtype}>, tensor<{dtype}>) -> tensor<3x{dtype}>
          return %p : tensor<3x{dtype}>
        }}
        """

        def imported(x, init):
            return hlo_call(x, init, source=source, passes="symbol-dce")[0]

        x, init = jnp.array([2.0, 0.0, 3.0]), jnp.array(0.0)
        direction = jnp.array([1.0, 2.0, 3.0])
        actual = jax.jit(
            lambda x, init: jax.jvp(imported, (x, init), (direction, jnp.array(7.0)))
        )(x, init)
        np.testing.assert_array_equal(actual[0], x)
        np.testing.assert_array_equal(actual[1], direction)
        gradients = jax.jit(jax.grad(lambda x, init: imported(x, init).sum(), (0, 1)))(
            x, init
        )
        np.testing.assert_array_equal(gradients[0], jnp.ones(3))
        np.testing.assert_array_equal(gradients[1], 0)
        np.testing.assert_array_equal(
            jax.jit(
                lambda init: jax.jvp(
                    lambda init: imported(x, init), (init,), (jnp.array(7.0),)
                )[1]
            )(init),
            jnp.zeros(3),
        )

    def test_zeros_shapes_and_higher_derivatives(self):
        cases = [
            ((5,), (0,)),
            ((2, 3, 5), (2, 0)),
            ((2, 1, 3), (1,)),
            ((2, 0, 3), (1,)),
            ((0, 3), (1,)),
        ]
        for shape, axes in cases:
            x = jnp.linspace(0.7, 1.3, int(np.prod(shape))).reshape(shape)
            source = str(
                jax.jit(lambda x: jnp.prod(x, axis=axes)).lower(x).compiler_ir()
            )
            source = source.replace("dimensions = [0, 2]", "dimensions = [2, 0]")

            def imported(x):
                return hlo_call(x, source=source, passes="symbol-dce")[0]

            expected = lambda x: polynomial(x, axes)
            seed = (
                jnp.arange(np.prod(expected(x).shape), dtype=x.dtype).reshape(
                    expected(x).shape
                )
                + 2
            )
            direction = jnp.linspace(-0.3, 0.8, x.size).reshape(shape)

            def operations(function):
                gradient = jax.grad(lambda x: jnp.sum(seed * function(x)))
                return (
                    function,
                    lambda x: jax.jvp(function, (x,), (direction,))[1],
                    gradient,
                    lambda x: jax.jvp(gradient, (x,), (direction,))[1],
                    jax.grad(lambda x: jnp.vdot(gradient(x), direction)),
                )

            for count in (0, 1, 2):
                point = x.reshape(-1).at[:count].set(0).reshape(shape)
                with self.subTest(shape=shape, zeros=count):
                    for actual, oracle in zip(
                        operations(imported), operations(expected)
                    ):
                        np.testing.assert_allclose(
                            jax.jit(actual)(point),
                            jax.jit(oracle)(point),
                            rtol=3e-5,
                            atol=3e-5,
                        )

    def test_arbitrary_cotangent_and_mixed_hessian(self):
        x = jnp.array([0.0, 2.0, 3.0])
        source = str(jax.jit(jnp.prod).lower(x).compiler_ir())

        def imported(x):
            return hlo_call(x, source=source)[0]

        def pullback(x):
            return jax.vjp(imported, x)[1](jnp.array(7.0))[0]

        np.testing.assert_array_equal(jax.jit(pullback)(x), [42.0, 0.0, 0.0])
        point = jnp.array([0.0, 0.0, 3.0])
        direction = jnp.array([2.0, -1.0, 0.0])
        np.testing.assert_array_equal(
            jax.jit(lambda x: jax.jvp(jax.grad(imported), (x,), (direction,))[1])(
                point
            ),
            [-3.0, 6.0, 0.0],
        )
        points = jnp.stack((x, point, jnp.array([1.0, 2.0, 3.0])))
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(pullback))(points),
            [[42.0, 0.0, 0.0], [0.0, 0.0, 0.0], [42.0, 21.0, 14.0]],
        )


if __name__ == "__main__":
    absltest.main()
