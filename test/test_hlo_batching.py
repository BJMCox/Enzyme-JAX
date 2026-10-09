from absl.testing import absltest
import re
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call
from enzyme_ad.jax._hlo_batch import batch_hlo
from enzyme_ad.jax.primitives import optimization_passes


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
    def test_optimization_barrier_batches_mixed_outputs(self):
        function = lambda x, tag: jax.lax.optimization_barrier((x, tag))
        x = jnp.array([[0.0, -0.0], [np.nan, np.inf], [-2.0, 3.0]])
        tags = jnp.array([3, 1, 7], dtype=jnp.int32)
        source = str(jax.jit(function).lower(x[0], tags[0]).compiler_ir())
        imported = lambda x, tag: tuple(
            hlo_call(x, tag, source=source, passes="symbol-dce")
        )
        for axis, tag in ((None, tags[0]), (0, tags)):
            axes = (0, axis)
            compiled = (
                jax.jit(jax.vmap(imported, in_axes=axes)).lower(x, tag).compile()
            )
            actual = compiled(x, tag)
            expected = jax.vmap(function, in_axes=axes)(x, tag)
            for value, reference in zip(actual, expected):
                np.testing.assert_array_equal(value, reference)
            np.testing.assert_array_equal(np.signbit(actual[0][0]), [False, True])
            self.assertEmpty(re.findall(r"\bwhile\(", compiled.as_text()))

    def test_postoptimized_batch_remains_reusable(self):
        source = str(jax.jit(lambda x: x * x).lower(jnp.float32(1)).compiler_ir())
        passes = optimization_passes(enable_loop_raising_passes=False)
        batch_hlo.cache_clear()
        plain = batch_hlo(source, "", 3)
        optimized = batch_hlo(source, "", 3, post_pipeline=passes)
        self.assertIsNotNone(plain)
        self.assertIsNotNone(optimized)
        self.assertEqual(batch_hlo.cache_info().misses, 2)
        self.assertEqual(batch_hlo(source, "", 3, post_pipeline=passes), optimized)
        self.assertEqual(batch_hlo.cache_info().hits, 1)

        def total(x):
            return hlo_call(x, source=optimized, passes="symbol-dce")[0].sum()

        x = jnp.array([-0.5, 1.0, 2.0], dtype=jnp.float32)
        value, gradient = jax.jit(jax.value_and_grad(total))(x)
        np.testing.assert_allclose(value, (x * x).sum())
        np.testing.assert_allclose(gradient, 2 * x)

    def test_scatter_update_cannot_capture_outer_values(self):
        # Native batching clones the update region unchanged. A capture must
        # never refer to an outer value whose type gained a batch dimension.
        source = """
        module {
          func.func @main(%base: tensor<5xf32>, %indices: tensor<4x1xi32>,
                          %updates: tensor<4xf32>, %scale: tensor<f32>) -> tensor<5xf32> {
            %result = "stablehlo.scatter"(%base, %indices, %updates) ({
              ^bb0(%old: tensor<f32>, %update: tensor<f32>):
                %scaled = stablehlo.multiply %update, %scale : tensor<f32>
                %next = stablehlo.add %old, %scaled : tensor<f32>
                stablehlo.return %next : tensor<f32>
            }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0],
                scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,
                indices_are_sorted = false, unique_indices = false}
                : (tensor<5xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<5xf32>
            return %result : tensor<5xf32>
          }
        }
        """
        self.assertIsNone(batch_hlo(source, "", 3))

    def test_padding_batches_shared_and_mapped_fill_values(self):
        function = lambda x, fill: jax.lax.pad(x, fill, [(1, 2, 1)])
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        fills = jnp.array([-1.0, 0.5, 2.0], dtype=x.dtype)
        source = str(jax.jit(function).lower(x[0], fills[0]).compiler_ir())
        imported = lambda x, fill: hlo_call(x, fill, source=source)[0]
        for axis, fill in ((None, fills[0]), (0, fills)):
            axes = (0, axis)
            compiled = (
                jax.jit(jax.vmap(imported, in_axes=axes)).lower(x, fill).compile()
            )
            np.testing.assert_array_equal(
                compiled(x, fill), jax.vmap(function, in_axes=axes)(x, fill)
            )

        # A constant fill uses a tensor pad, rather than a loop over lanes.
        constant = lambda x: function(x, jnp.float32(0))
        source = str(jax.jit(constant).lower(x[0]).compiler_ir())
        imported = lambda x: hlo_call(x, source=source)[0]
        compiled = jax.jit(jax.vmap(imported)).lower(x).compile()
        np.testing.assert_array_equal(compiled(x), jax.vmap(constant)(x))
        self.assertEmpty(re.findall(r"\bwhile\(", compiled.as_text()))

    def test_gather_pullbacks_batch_repeated_indices(self):
        function = lambda x, indices: jnp.sum(x[indices] ** 2)
        x = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 7
        indices = jnp.array([[0, 2, 2], [4, 1, 3], [1, 1, 1]])
        source = str(jax.jit(function).lower(x[0], indices[0]).compiler_ir())
        imported = lambda x, i: hlo_call(x, i, source=source)[0]
        for axis, selected in ((None, indices[0]), (0, indices)):
            axes = (0, axis)
            transforms = (
                lambda f: jax.vmap(jax.value_and_grad(f), in_axes=axes),
                lambda f: jax.value_and_grad(
                    lambda x, i: jax.vmap(f, in_axes=axes)(x, i).sum()
                ),
            )
            for transform in transforms:
                compiled = jax.jit(transform(imported)).lower(x, selected).compile()
                for actual, expected in zip(
                    jax.tree.leaves(compiled(x, selected)),
                    jax.tree.leaves(transform(function)(x, selected)),
                ):
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                self.assertEmpty(re.findall(r"\bwhile\(", compiled.as_text()))

    def test_fallback_keeps_imported_symbols_distinct_from_jax_helpers(self):
        def function(xs):
            def step(state, x):
                return state + x, state + x

            return jnp.sort(jax.lax.scan(step, jnp.float32(0.2), xs)[1])

        x = jnp.array([2, -3, 1, -0.5], dtype=jnp.float32)
        source = str(jax.jit(function).lower(x).compiler_ir())
        imported = lambda x: hlo_call(x, source=source)[0]
        points = jnp.stack((x, 2 * x))
        np.testing.assert_allclose(
            jax.jit(jax.vmap(lambda x: imported(x) + imported(-x)))(points),
            jax.vmap(lambda x: function(x) + function(-x))(points),
            atol=1e-6,
        )

    def test_pointwise_derivatives_have_no_lane_loop(self):
        x = jnp.linspace(-2, 3, 17, dtype=jnp.float32)
        scale = jnp.float32(2)
        mapped = jax.vmap(weighted_polynomial, in_axes=(0, None))
        calls = (
            jax.vmap(
                jax.value_and_grad(weighted_polynomial, argnums=(0, 1)),
                in_axes=(0, None),
            ),
            jax.value_and_grad(lambda x, s: mapped(x, s).sum(), argnums=(0, 1)),
        )
        for call in calls:
            compiled = jax.jit(call).lower(x, scale).compile()
            self.assertEmpty(re.findall(r"\bwhile\(", compiled.as_text()))

    def test_fixed_scan_batches_time_steps_and_shared_derivatives(self):
        def scan(scale, xs):
            def step(state, x):
                value = jnp.tanh(scale * state + x)
                return value, value

            return jax.lax.scan(step, jnp.float32(0.2), xs)[1].sum()

        scale = jnp.float32(0.4)
        xs = jnp.linspace(-0.3, 0.7, 40, dtype=scale.dtype).reshape(5, 8)
        source = str(jax.jit(scan).lower(scale, xs[0]).compiler_ir())
        imported = lambda a, x: hlo_call(a, x, source=source)[0]

        def summed(f):
            return lambda a, x: jax.vmap(f, in_axes=(None, 0))(a, x).sum()

        for transform, loops in (
            (lambda f: jax.vmap(f, in_axes=(None, 0)), 1),
            (
                lambda f: jax.vmap(
                    jax.value_and_grad(f, argnums=(0, 1)), in_axes=(None, 0)
                ),
                2,
            ),
            (lambda f: jax.value_and_grad(summed(f), argnums=(0, 1)), 2),
        ):
            compiled = jax.jit(transform(imported)).lower(scale, xs).compile()
            actual, expected = compiled(scale, xs), transform(scan)(scale, xs)
            for value, reference in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected)
            ):
                np.testing.assert_allclose(value, reference, rtol=3e-5, atol=3e-5)
            self.assertLen(re.findall(r"\bwhile\(", compiled.as_text()), loops)

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
        lowered = jax.jit(jax.vmap(imported)).lower(n, x)
        self.assertNotIn("arith.", str(lowered.compiler_ir()))
        actual = lowered.compile()(n, x)
        np.testing.assert_allclose(actual, (x + 1) * 2.0**n - 1)

    def test_loop_counter_comparison_and_narrowing_are_preserved(self):
        for start, compare, update, expected in (
            (4294967295, "UNSIGNED", "%next = stablehlo.add %i, %one : tensor<i32>", 0),
            (
                -130,
                "SIGNED",
                """
                %sum = stablehlo.add %i, %one : tensor<i32>
                %small = stablehlo.convert %sum : (tensor<i32>) -> tensor<i8>
                %next = stablehlo.convert %small : (tensor<i8>) -> tensor<i32>
            """,
                1,
            ),
        ):
            with self.subTest(start=start):
                source = f"""
                module {{
                  func.func @main(%x: tensor<f32>) -> tensor<f32> {{
                    %start = stablehlo.constant dense<{start}> : tensor<i32>
                    %one = stablehlo.constant dense<1> : tensor<i32>
                    %inc = stablehlo.constant dense<1.0> : tensor<f32>
                    %loop:2 = stablehlo.while(%i = %start, %y = %x) : tensor<i32>, tensor<f32>
                    cond {{
                      %more = stablehlo.compare LT, %i, %one, {compare} : (tensor<i32>, tensor<i32>) -> tensor<i1>
                      stablehlo.return %more : tensor<i1>
                    }} do {{
                      {update}
                      %value = stablehlo.add %y, %inc : tensor<f32>
                      stablehlo.return %next, %value : tensor<i32>, tensor<f32>
                    }}
                    return %loop#1 : tensor<f32>
                  }}
                }}
                """
                if compare == "UNSIGNED":
                    source = source.replace("tensor<i32>", "tensor<ui32>")
                function = lambda x: hlo_call(x, source=source)[0]
                x = jnp.array([-2, 0.5, 3], dtype=jnp.float32)
                np.testing.assert_allclose(jax.jit(jax.vmap(function))(x), x + expected)

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
