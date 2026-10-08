from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call
from enzyme_ad.jax.primitives import optimization_passes


class StaticScatter(absltest.TestCase):
    def test_duplicate_and_out_of_bounds_updates_keep_scatter_semantics(self):
        base = jnp.ones(5)
        large = 2.0 ** (np.finfo(base.dtype).nmant + 3)
        for indices, updates in (
            (jnp.array([3, 3]), jnp.array([large, -large])),
            (jnp.array([9, 1]), jnp.array([large, 2.0])),
        ):
            with self.subTest(indices=indices):
                reference = lambda b, u: b.at[indices].add(u, mode="drop")
                source = str(jax.jit(reference).lower(base, updates).compiler_ir())
                imported = lambda b, u: hlo_call(
                    b,
                    u,
                    source=source,
                    passes=optimization_passes(enable_loop_raising_passes=False),
                )[0]
                np.testing.assert_array_equal(
                    jax.jit(imported)(base, updates), jax.jit(reference)(base, updates)
                )

    def test_addition_preserves_zero_sign_after_inlining(self):
        indices = jnp.array([3, 1])
        base, updates = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0]), jnp.array([-0.0, -0.0])
        reference = lambda base, updates: base.at[indices].add(updates)
        source = str(jax.jit(reference).lower(base, updates).compiler_ir())

        def imported(updates):
            return hlo_call(
                base,
                updates,
                source=source,
                passes=optimization_passes(enable_loop_raising_passes=False),
            )[0]

        compiled = jax.jit(imported).lower(updates).compile()
        actual = compiled(updates)
        np.testing.assert_array_equal(actual, reference(base, updates))
        np.testing.assert_array_equal(np.signbit(actual), np.zeros(5, dtype=bool))
        self.assertNotIn(" scatter(", compiled.as_text())

    def test_shared_and_mapped_index_tables(self):
        base = jnp.arange(18.0).reshape(2, 9)
        updates = jnp.array([[1.0, 2.0, 3.0], [-2.0, 1.5, 4.0]])
        same = jnp.array([[7, 1, 4], [7, 1, 4]])
        different = jnp.array([[7, 1, 4], [2, 2, 8]])
        for tables in (same, different):
            with self.subTest(tables=tables):
                reference = lambda base, updates: jax.vmap(
                    lambda b, u, i: b.at[i].add(u)
                )(base, updates, tables)
                source = str(jax.jit(reference).lower(base, updates).compiler_ir())
                imported = lambda b, u: hlo_call(
                    b,
                    u,
                    source=source,
                    passes=optimization_passes(enable_loop_raising_passes=False),
                )[0]
                np.testing.assert_array_equal(
                    jax.jit(imported)(base, updates), reference(base, updates)
                )
        reference = lambda b, u: jax.vmap(lambda b, u: b.at[same[0]].add(u))(b, u)
        source = str(jax.jit(reference).lower(base, updates).compiler_ir())
        imported = lambda b, u: hlo_call(
            b,
            u,
            source=source,
            passes=optimization_passes(enable_loop_raising_passes=False),
        )[0]
        compiled = jax.jit(imported).lower(base, updates).compile()
        np.testing.assert_array_equal(compiled(base, updates), reference(base, updates))
        self.assertNotIn(" scatter(", compiled.as_text())

    def test_compact_updates_preserve_holes_and_derivatives(self):
        indices = jnp.array([8, 2, 5])
        base = jnp.array([-0.0, np.nan, 2, 3, -0.0, 5, np.inf, 7, 8, 9, 10])
        updates = jnp.array([1.5, -2.0, -0.0])
        untouched = [0, 1, 3, 4, 6, 7, 9, 10]
        for replace in (False, True):
            with self.subTest(replace=replace):

                def reference(base, updates):
                    target = base.at[indices]
                    return target.set(updates) if replace else target.add(updates)

                source = str(jax.jit(reference).lower(base, updates).compiler_ir())

                def imported(base, updates):
                    return hlo_call(
                        base,
                        updates,
                        source=source,
                        passes=optimization_passes(enable_loop_raising_passes=False),
                    )[0]

                compiled = jax.jit(imported).lower(base, updates).compile()
                actual = np.asarray(compiled(base, updates))
                np.testing.assert_array_equal(actual, reference(base, updates))
                np.testing.assert_array_equal(
                    np.signbit(actual[untouched]),
                    np.signbit(np.asarray(base)[untouched]),
                )
                self.assertNotIn(" scatter(", compiled.as_text())

                def gradient(function):
                    return jax.grad(
                        lambda b, u: jnp.sum(function(b, u) ** 2), argnums=(0, 1)
                    )

                finite = (jnp.zeros_like(base), jnp.zeros_like(updates))
                direction = (
                    jnp.arange(base.size, dtype=base.dtype),
                    jnp.ones_like(updates),
                )
                for transform in (
                    gradient,
                    lambda f: lambda b, u: jax.jvp(gradient(f), (b, u), direction)[1],
                    lambda f: jax.grad(
                        lambda b, u: sum(
                            jnp.vdot(g, d) for g, d in zip(gradient(f)(b, u), direction)
                        ),
                        argnums=(0, 1),
                    ),
                ):
                    for actual, expected in zip(
                        jax.jit(transform(imported))(*finite),
                        jax.jit(transform(reference))(*finite),
                    ):
                        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    absltest.main()
