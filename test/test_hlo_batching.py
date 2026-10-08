from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call


def polynomial(x, scale):
    return tuple(
        hlo_call(
            x,
            scale,
            source="""
            module {
              func.func @main(%x: tensor<f32>, %scale: tensor<f32>)
                  -> (tensor<f32>, tensor<2xf32>) {
                %square = stablehlo.multiply %x, %x : tensor<f32>
                %scaled = stablehlo.multiply %square, %scale : tensor<f32>
                %cube = stablehlo.multiply %square, %x : tensor<f32>
                %first = stablehlo.reshape %x : (tensor<f32>) -> tensor<1xf32>
                %last = stablehlo.reshape %cube : (tensor<f32>) -> tensor<1xf32>
                %vector = stablehlo.concatenate %first, %last, dim = 0
                  : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
                return %scaled, %vector : tensor<f32>, tensor<2xf32>
              }
            }
            """,
        )
    )


def weighted_polynomial(x, scale):
    scalar, vector = polynomial(x, scale)
    return scalar + 2 * vector[0] + 3 * vector[1]


class HloBatching(absltest.TestCase):
    def test_distinct_rng_states_stay_in_their_lanes(self):
        def random_bits(key):
            return jax.lax.rng_bit_generator(key, (4,))

        key = jnp.array([1, 2, 3, 4], dtype=jnp.uint32)
        source = str(jax.jit(random_bits).lower(key).compiler_ir())

        def imported(key):
            return tuple(hlo_call(key, source=source))

        keys = jnp.stack((key, key + 17, key + 101))
        actual = jax.jit(jax.vmap(imported))(keys)
        expected = tuple(
            jnp.stack(parts) for parts in zip(*(random_bits(k) for k in keys))
        )
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value, reference)

    def test_varying_loop_counts_stay_in_their_lanes(self):
        def recurrence(count, initial):
            return jax.lax.while_loop(
                lambda state: state[0] < count,
                lambda state: (state[0] + 1, 2 * state[1] + 1),
                (jnp.int32(0), initial),
            )[1]

        source = str(
            jax.jit(recurrence).lower(jnp.int32(1), jnp.float32(0)).compiler_ir()
        )
        imported = lambda n, x: hlo_call(n, x, source=source)[0]
        n = jnp.array([0, 1, 4], dtype=jnp.int32)
        x = jnp.array([3, -2, 0.5], dtype=jnp.float32)
        actual = jax.jit(jax.vmap(imported))(n, x)
        np.testing.assert_allclose(actual, (x + 1) * 2.0**n - 1)

    def test_nested_map_with_nonleading_axis_and_shared_input(self):
        x = jnp.array([[-2, 0.5, 3], [1, -1, 0]], dtype=jnp.float32)
        scale = jnp.float32(2)
        mapped = jax.jit(
            jax.vmap(
                jax.vmap(polynomial, in_axes=(0, None)),
                in_axes=(1, None),
                out_axes=(1, 1),
            )
        )
        scalar, vector = mapped(x, scale)
        np.testing.assert_allclose(scalar, scale * x**2)
        np.testing.assert_allclose(vector, jnp.stack((x, x**3), axis=-1))

    def test_both_gradient_orders_and_shared_input_accumulation(self):
        x = jnp.array([-2, 0.5, 3], dtype=jnp.float32)
        scale = jnp.float32(2)
        mapped_gradient = jax.jit(
            jax.vmap(jax.grad(weighted_polynomial, argnums=(0, 1)), in_axes=(0, None))
        )
        gx, gs = mapped_gradient(x, scale)
        np.testing.assert_allclose(gx, 2 * scale * x + 2 + 9 * x**2)
        np.testing.assert_allclose(gs, x**2)

        def total(x, scale):
            return jnp.sum(jax.vmap(weighted_polynomial, in_axes=(0, None))(x, scale))

        gx, gs = jax.jit(jax.grad(total, argnums=(0, 1)))(x, scale)
        np.testing.assert_allclose(gx, 2 * scale * x + 2 + 9 * x**2)
        np.testing.assert_allclose(gs, jnp.sum(x**2))

    def test_batched_jvp_and_weighted_vjp(self):
        x = jnp.array([-2, 0.5, 3], dtype=jnp.float32)
        dx = jnp.array([1, -2, 0.5], dtype=jnp.float32)
        scale, ds = jnp.float32(2), jnp.float32(-3)
        mapped = jax.vmap(polynomial, in_axes=(0, None))
        tangent = jax.jit(lambda x, s: jax.jvp(mapped, (x, s), (dx, ds))[1])
        da, db = tangent(x, scale)
        np.testing.assert_allclose(da, 2 * scale * x * dx + ds * x**2)
        np.testing.assert_allclose(db, jnp.stack((dx, 3 * x**2 * dx), axis=-1))

        wa = jnp.array([1, -2, 3], dtype=jnp.float32)
        wb = jnp.array([[2, 1], [-3, 2], [0.5, -1]], dtype=jnp.float32)

        @jax.jit
        def pullback(x, scale):
            _, transpose = jax.vjp(mapped, x, scale)
            return transpose((wa, wb))

        gx, gs = pullback(x, scale)
        np.testing.assert_allclose(
            gx, wa * 2 * scale * x + wb[:, 0] + 3 * x**2 * wb[:, 1]
        )
        np.testing.assert_allclose(gs, jnp.sum(wa * x**2))

    def test_empty_and_singleton_maps(self):
        value = jax.jit(jax.vmap(weighted_polynomial, in_axes=(0, None)))
        gradient = jax.jit(jax.vmap(jax.grad(weighted_polynomial), in_axes=(0, None)))
        for values in ([], [2]):
            with self.subTest(values=values):
                x = jnp.array(values, dtype=jnp.float32)
                scale = jnp.float32(2)
                np.testing.assert_allclose(
                    value(x, scale), scale * x**2 + 2 * x + 3 * x**3
                )
                np.testing.assert_allclose(
                    gradient(x, scale), 2 * scale * x + 2 + 9 * x**2
                )


if __name__ == "__main__":
    absltest.main()
