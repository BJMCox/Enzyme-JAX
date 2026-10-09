from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import cpp_call, hlo_call


class InactiveOutput(absltest.TestCase):
    def test_cpp_batching_keeps_its_pipeline_contract(self):
        def imported(x):
            return cpp_call(
                x,
                out_shapes=[jax.core.ShapedArray(x.shape, x.dtype)],
                source="template<typename T> void f(T& out, const T& in) { out = in; }",
            )[0]

        shape = jax.eval_shape(jax.vmap(imported), jnp.ones((2, 3), jnp.float32))
        self.assertEqual(shape.shape, (2, 3))

    def test_jacfwd_preserves_unmapped_primals(self):
        for size in (0, 3):
            x, scalar = jnp.arange(float(size)), jnp.array(2.0)

            def reference(x, scalar):
                return x**3 + scalar, jnp.array([4.0, 7.0])

            source = str(jax.jit(reference).lower(x, scalar).compiler_ir())

            def imported(x, scalar):
                return tuple(hlo_call(x, scalar, source=source, passes="symbol-dce"))

            for actual, expected in zip(
                jax.tree.leaves(jax.jit(jax.jacfwd(imported, (0, 1)))(x, scalar)),
                jax.tree.leaves(jax.jacfwd(reference, (0, 1))(x, scalar)),
            ):
                np.testing.assert_array_equal(actual, expected)

            for actual, expected in zip(
                jax.tree.leaves(jax.jit(jax.jacfwd(imported))(x, scalar)),
                jax.tree.leaves(jax.jacfwd(reference)(x, scalar)),
            ):
                np.testing.assert_array_equal(actual, expected)

            for count in (0, 2):
                directions = jnp.arange(float(count * size)).reshape(count, size)

                def directional(function):
                    return jax.jit(
                        jax.vmap(
                            lambda dx: jax.jvp(
                                function, (x, scalar), (dx, jnp.array(0.0))
                            ),
                            out_axes=(None, 0),
                        )
                    )(directions)

                for actual, expected in zip(
                    jax.tree.leaves(directional(imported)),
                    jax.tree.leaves(directional(reference)),
                ):
                    np.testing.assert_array_equal(actual, expected)

            points = jnp.stack((x, x + 1), axis=1)
            for actual, expected in zip(
                jax.tree.leaves(
                    jax.jit(
                        jax.vmap(
                            jax.jacfwd(imported),
                            in_axes=(1, None),
                            out_axes=-1,
                        )
                    )(points, scalar)
                ),
                jax.tree.leaves(
                    jax.vmap(
                        jax.jacfwd(reference),
                        in_axes=(1, None),
                        out_axes=-1,
                    )(points, scalar)
                ),
            ):
                np.testing.assert_array_equal(actual, expected)

    def test_jacfwd_of_gradient(self):
        x = jnp.array([1.0, 2.0, 3.0])
        source = str(jax.jit(lambda x: (x**3).sum()).lower(x).compiler_ir())

        def imported(x):
            return hlo_call(x, source=source, passes="symbol-dce")[0]

        actual = jax.jit(jax.jacfwd(jax.grad(imported)))(x)
        np.testing.assert_array_equal(actual, jnp.diag(6 * x))

    def test_higher_derivatives_and_pullback_seeds(self):
        x, direction = jnp.array(2.0), jnp.array(3.0)
        source = str(jax.jit(lambda x: (x**3, x)).lower(x).compiler_ir())

        def loss(x):
            return hlo_call(x, source=source, passes="symbol-dce")[0]

        gradient = jax.grad(loss)
        forward = jax.jit(lambda x: jax.jvp(gradient, (x,), (direction,))[1])
        reverse = jax.jit(jax.grad(lambda x: gradient(x) * direction))
        for derivative in (forward, reverse):
            np.testing.assert_allclose(derivative(x), 6 * x * direction)
        np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(gradient)))(x), 6)

        def pullback(seed):
            return jax.vjp(loss, x)[1](seed)[0]

        np.testing.assert_allclose(jax.jit(jax.grad(pullback))(direction), 3 * x**2)
        tangent = jax.jit(lambda seed: jax.jvp(pullback, (seed,), (direction,))[1])
        np.testing.assert_allclose(tangent(direction), 3 * x**2 * direction)


if __name__ == "__main__":
    absltest.main()
