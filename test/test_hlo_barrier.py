from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call


class OptimizationBarrier(absltest.TestCase):
    def test_derivatives_preserve_slots_and_accumulate_repeated_operands(self):
        x, scalar = jnp.zeros(3), jnp.array(0.0)
        tag, inactive = jnp.int32(7), jnp.array([2.0, -3.0])

        def barrier(x, tag, inactive, scalar):
            return jax.lax.optimization_barrier((x, tag, inactive, scalar, x))

        source = str(jax.jit(barrier).lower(x, tag, inactive, scalar).compiler_ir())

        def imported(x, scalar):
            return hlo_call(
                x, tag, inactive, scalar, source=source, passes="symbol-dce"
            )

        for actual, expected in zip(
            jax.jit(imported)(x, scalar), barrier(x, tag, inactive, scalar)
        ):
            np.testing.assert_array_equal(actual, expected)

        def active(x, scalar):
            values = imported(x, scalar)
            return values[0], values[3], values[4]

        direction = (jnp.arange(1.0, 4.0), jnp.array(2.0))
        _, tangent = jax.jit(lambda x, s: jax.jvp(active, (x, s), direction))(x, scalar)
        for actual, expected in zip(tangent, (*direction, direction[0])):
            np.testing.assert_array_equal(actual, expected)

        def pullback(x, scalar):
            _, backward = jax.vjp(active, x, scalar)
            return backward((direction[0], direction[1], 3 * direction[0]))

        actual = jax.jit(pullback)(x, scalar)
        np.testing.assert_array_equal(actual[0], 4 * direction[0])
        np.testing.assert_array_equal(actual[1], direction[1])

        def loss(x, scalar):
            first, s, repeated = active(x, scalar)
            return jnp.sum(first**2) + 2 * jnp.sum(repeated**2) + 3 * s**2

        gradient = jax.grad(loss, argnums=(0, 1))
        forward_over_reverse = jax.jit(
            lambda x, s: jax.jvp(gradient, (x, s), direction)[1]
        )
        reverse_over_reverse = jax.jit(
            jax.grad(
                lambda x, s: sum(
                    jnp.vdot(g, d) for g, d in zip(gradient(x, s), direction)
                ),
                argnums=(0, 1),
            )
        )
        for function in (forward_over_reverse, reverse_over_reverse):
            for actual, expected in zip(function(x, scalar), direction):
                np.testing.assert_array_equal(actual, 6 * expected)
        values = jnp.arange(6.0).reshape(2, 3)
        actual = jax.jit(jax.vmap(gradient, in_axes=(0, None)))(values, scalar)
        np.testing.assert_array_equal(actual[0], 6 * values)
        np.testing.assert_array_equal(actual[1], jnp.zeros(2))


if __name__ == "__main__":
    absltest.main()
