import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import hlo_call

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64


def imported(function, *args, **options):
    source = str(jax.jit(function, keep_unused=True).lower(*args).compiler_ir())
    return lambda *xs: hlo_call(*xs, source=source, **options)[0]


class ScatterWinner(absltest.TestCase):
    def test_runtime_swap_preserves_identity_and_second_derivatives(self):
        with enable_x64():
            for dtype in (jnp.float32, jnp.float64):
                for shape in ((3,), (3, 2)):
                    with self.subTest(dtype=dtype, shape=shape):
                        x = jnp.arange(1, 1 + np.prod(shape), dtype=dtype).reshape(
                            shape
                        )
                        seed = jnp.arange(7, 7 + x.size, dtype=dtype).reshape(shape)
                        direction = jnp.ones_like(x) / 4

                        def swap(x, pivot):
                            zero = jnp.int32(0)
                            return x.at[jnp.stack((zero, pivot))].set(
                                x[jnp.stack((pivot, zero))]
                            )

                        function = imported(swap, x, jnp.int32(0))
                        loss = lambda x, p, function=function, seed=seed: jnp.sum(
                            function(x, p) ** 3 * seed
                        )
                        gradient = jax.grad(loss)
                        value_gradient = jax.jit(jax.value_and_grad(loss))
                        hvp = jax.jit(
                            lambda x, p, gradient=gradient, direction=direction: (
                                jax.jvp(lambda z: gradient(z, p), (x,), (direction,))[1]
                            )
                        )
                        reverse_hvp = jax.jit(
                            jax.grad(
                                lambda x, p, gradient=gradient, direction=direction: (
                                    jnp.vdot(gradient(x, p), direction)
                                )
                            )
                        )
                        for pivot in (0, 1):
                            # Runtime pivots share the same compiled executable.
                            permutation = np.arange(3)
                            permutation[[0, pivot]] = permutation[[pivot, 0]]
                            weight = np.asarray(seed)[permutation]
                            expected = np.asarray(x)[permutation]
                            value, grad = value_gradient(x, jnp.int32(pivot))
                            np.testing.assert_array_equal(
                                value, np.sum(expected**3 * np.asarray(seed))
                            )
                            np.testing.assert_array_equal(
                                grad, 3 * np.asarray(x) ** 2 * weight
                            )
                            second = 6 * np.asarray(x) * np.asarray(direction) * weight
                            np.testing.assert_array_equal(
                                hvp(x, jnp.int32(pivot)), second
                            )
                            np.testing.assert_array_equal(
                                reverse_hvp(x, jnp.int32(pivot)), second
                            )

    def test_selected_writer_and_dropped_updates_match_primal(self):
        base = jnp.array([1.0, -0.0, 2.0])
        updates = jnp.array([5.0, 3.0, jnp.nan])
        indices = jnp.array([0, 0, 9], dtype=jnp.int32)
        function = imported(
            lambda b, u, i: b.at[i].set(u, mode="drop"), base, updates, indices
        )

        def loss(b, u, i):
            value = function(b, u, i)
            return jnp.sum(value**2), value

        (value, output), (db, du) = jax.jit(
            jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        )(base, updates, indices)
        # Last scalar update ID wins; it is not the largest update value.
        np.testing.assert_array_equal(output, [3.0, -0.0, 2.0])
        self.assertTrue(np.signbit(np.asarray(output)[1]))
        np.testing.assert_array_equal(value, 13.0)
        np.testing.assert_array_equal(db, [0.0, -0.0, 4.0])
        np.testing.assert_array_equal(du, [0.0, 6.0, 0.0])

    def test_overlapping_windows_and_empty_updates(self):
        base = jnp.array([11.0, 13.0, 17.0, 19.0])
        updates = jnp.array([[2.0, 3.0], [5.0, 7.0]])
        indices = jnp.array([[0], [1]], dtype=jnp.int32)
        dims = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,),
        )
        function = imported(
            lambda b, u, i: jax.lax.scatter(b, i, u, dims), base, updates, indices
        )

        def loss(b, u, i):
            value = function(b, u, i)
            return jnp.sum(value**2), value

        (_, output), (db, du) = jax.jit(
            jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        )(base, updates, indices)
        np.testing.assert_array_equal(output, [2.0, 5.0, 7.0, 19.0])
        np.testing.assert_array_equal(db, [0.0, 0.0, 0.0, 38.0])
        np.testing.assert_array_equal(du, [[4.0, 0.0], [10.0, 14.0]])

        empty = jnp.zeros((0,), dtype=base.dtype)
        indices = jnp.zeros((0,), dtype=jnp.int32)
        function = imported(
            lambda b, u, i: b.at[i].set(u),
            base,
            empty,
            indices,
            passes="symbol-dce",
        )
        db, du = jax.jit(
            jax.grad(lambda b, u, i: jnp.sum(function(b, u, i) ** 2), argnums=(0, 1))
        )(base, empty, indices)
        np.testing.assert_array_equal(db, 2 * base)
        np.testing.assert_array_equal(du, empty)

    def test_partial_windows_match_actual_primal_writer(self):
        dims = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,),
        )
        with enable_x64():
            for dtype in (jnp.float32, jnp.float64):
                base = jnp.array([11, 13, 17], dtype=dtype)
                updates = jnp.array([[2, 3], [5, 7]], dtype=dtype)
                seed = jnp.array([5, 7, 11], dtype=dtype)
                direction = (jnp.ones_like(base) / 4, jnp.ones_like(updates) / 2)
                # Two external windows prevent singleton-index promotion to
                # unique=true and exercise the cached-winner reverse branch.
                indices = jnp.array([[-1], [-1]], dtype=jnp.int32)
                function = imported(
                    lambda b, u, i: jax.lax.scatter(b, i, u, dims),
                    base,
                    updates,
                    indices,
                    passes="symbol-dce",
                )

                def loss(b, u, i, function=function, seed=seed):
                    value = function(b, u, i)
                    return jnp.sum(value**3 * seed), value

                gradient = jax.grad(
                    lambda b, u, i, loss=loss: loss(b, u, i)[0], argnums=(0, 1)
                )
                value_gradient = jax.jit(
                    jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
                )
                hvp = jax.jit(
                    lambda b, u, i, gradient=gradient, direction=direction: jax.jvp(
                        lambda b, u: gradient(b, u, i), (b, u), direction
                    )[1]
                )
                for start in (-1, 0, 1, 2, -2, 3):
                    with self.subTest(dtype=dtype, start=start):
                        db = 3 * np.array(base) ** 2 * np.array(seed)
                        du = np.zeros_like(updates)
                        hb = (
                            6 * np.array(base) * np.array(direction[0]) * np.array(seed)
                        )
                        hu = np.zeros_like(updates)
                        indices = jnp.array([[start], [start]], dtype=jnp.int32)
                        (_, output), actual = value_gradient(base, updates, indices)
                        output = np.asarray(output)
                        # XLA currently drops entire partial windows, whereas
                        # StableHLO specifies scalar writes. Infer the actual
                        # selection from distinct values; AD must match it.
                        for destination in range(3):
                            allowed = [float(base[destination])]
                            offset = destination - start
                            if 0 <= offset < 2:
                                allowed.extend(float(u[offset]) for u in updates)
                            self.assertIn(float(output[destination]), allowed)
                        for window in range(2):
                            for offset in range(2):
                                destination = start + offset
                                if (
                                    0 <= destination < 3
                                    and output[destination] == updates[window, offset]
                                ):
                                    db[destination] = hb[destination] = 0
                                    du[window, offset] = (
                                        3
                                        * updates[window, offset] ** 2
                                        * seed[destination]
                                    )
                                    hu[window, offset] = (
                                        6
                                        * updates[window, offset]
                                        * direction[1][window, offset]
                                        * seed[destination]
                                    )
                        for observed, oracle in zip(actual, (db, du)):
                            np.testing.assert_array_equal(observed, oracle)
                        for observed, oracle in zip(
                            hvp(base, updates, indices), (hb, hu)
                        ):
                            np.testing.assert_array_equal(observed, oracle)


if __name__ == "__main__":
    absltest.main()
