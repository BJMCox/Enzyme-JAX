from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call
from enzyme_ad.jax.primitives import optimization_passes


class ScatterPadding(absltest.TestCase):
    def test_discarded_rows_preserve_values_and_derivatives(self):
        indices = jnp.array([[2, 0, 5], [1, 3, 5]])
        values = jnp.arange(36.0).reshape(2, 3, 2, 3)

        def reference(x):
            return jnp.zeros((6, 2, 3), dtype=x.dtype).at[indices].add(x)[:4]

        source = str(jax.jit(reference).lower(values).compiler_ir())
        imported = lambda x: hlo_call(
            x,
            source=source,
            passes=optimization_passes(enable_loop_raising_passes=False),
        )[0]
        compiled = jax.jit(imported).lower(values).compile()
        self.assertNotIn(" scatter(", compiled.as_text())
        special = values.at[:, 2].set(jnp.nan).at[0, 0, 0, 0].set(-0.0)
        for x in (values, special):
            actual = np.asarray(compiled(x))
            np.testing.assert_array_equal(actual, reference(x))
            np.testing.assert_array_equal(np.signbit(actual), False)
            batch = jnp.stack([x, x])
            np.testing.assert_array_equal(
                jax.jit(jax.vmap(imported))(batch), jax.vmap(reference)(batch)
            )
        seed = jnp.arange(24.0).reshape(4, 2, 3)
        gradient = jax.jit(jax.grad(lambda x: jnp.vdot(imported(x), seed)))
        expected = jnp.where(
            (indices < 4)[..., None, None], seed[jnp.minimum(indices, 3)], 0
        )
        np.testing.assert_array_equal(gradient(values), expected)
        direction = values + 1
        loss = lambda x: jnp.sum(imported(x) ** 2)
        forward = jax.jit(lambda x: jax.jvp(jax.grad(loss), (x,), (direction,))[1])
        reverse = jax.jit(jax.grad(lambda x: jax.jvp(loss, (x,), (direction,))[1]))
        expected = jnp.where((indices < 4)[..., None, None], 2 * direction, 0)
        np.testing.assert_array_equal(forward(jnp.zeros_like(values)), expected)
        np.testing.assert_array_equal(reverse(jnp.zeros_like(values)), expected)

    def test_retained_duplicates_holes_and_observed_padding(self):
        values = jnp.arange(18.0).reshape(2, 3, 3)
        for indices, observe_padding in (
            (jnp.array([[2, 0, 5], [1, 3, 5]]), True),
            (jnp.array([[2, 0, 5], [1, 2, 5]]), False),
        ):
            with self.subTest(indices=indices, observe_padding=observe_padding):

                def reference(x):
                    scattered = jnp.zeros((6, 3), dtype=x.dtype).at[indices].add(x)
                    result = scattered[:4]
                    return result + scattered[5] if observe_padding else result

                source = str(jax.jit(reference).lower(values).compiler_ir())
                imported = lambda x: hlo_call(
                    x,
                    source=source,
                    passes=optimization_passes(enable_loop_raising_passes=False),
                )[0]
                np.testing.assert_array_equal(
                    jax.jit(imported)(values), reference(values)
                )


if __name__ == "__main__":
    absltest.main()
