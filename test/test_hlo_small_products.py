import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)


def source_for(m, k, n, precision, left=1, right=0):
    a = (m, k) if left == 1 else (k, m)
    b = (k, n) if right == 0 else (n, k)
    return f"""module {{
      func.func @main(%a: tensor<{a[0]}x{a[1]}x{precision}>,
                      %b: tensor<{b[0]}x{b[1]}x{precision}>)
          -> tensor<{m}x{n}x{precision}> {{
        %x = stablehlo.dot_general %a, %b,
            contracting_dims = [{left}] x [{right}], precision = [HIGHEST, HIGHEST]
            : (tensor<{a[0]}x{a[1]}x{precision}>, tensor<{b[0]}x{b[1]}x{precision}>)
            -> tensor<{m}x{n}x{precision}>
        return %x : tensor<{m}x{n}x{precision}>
      }}
    }}"""


class SmallProducts(parameterized.TestCase):
    @parameterized.named_parameters(
        ("paired", (2, 3, 4), (2, 4, 5), (((2,), (1,)), ((0,), (0,))), "bmk,bkn->bmn"),
        ("shared_left", (3, 4), (2, 4, 5), (((1,), (1,)), ((), ())), "mk,bkn->mbn"),
        ("shared_right", (2, 3, 4), (4, 5), (((2,), (0,)), ((), ())), "bmk,kn->bmn"),
        (
            "multiple_free",
            (2, 3, 4),
            (4, 2, 3),
            (((2,), (0,)), ((), ())),
            "abk,kcd->abcd",
        ),
        (
            "multiple_contract",
            (2, 2, 3),
            (3, 2, 4),
            (((1, 2), (1, 0)), ((), ())),
            "mij,jin->mn",
        ),
        (
            "batch_free_contract",
            (2, 2, 3, 2, 2),
            (2, 2, 2, 2, 3),
            (((3, 4), (2, 1)), ((0,), (0,))),
            "bxyij,bjiuv->bxyuv",
        ),
        (
            "permuted_batches",
            (2, 3, 2, 4),
            (3, 4, 2, 5),
            (((3,), (1,)), ((1, 0), (0, 2))),
            "abmk,bkan->bamn",
        ),
        (
            "scalar",
            (2, 2, 3),
            (3, 2, 2),
            (((0, 1, 2), (2, 1, 0)), ((), ())),
            "ijk,kji->",
        ),
    )
    def test_general_contractions_and_higher_derivatives(
        self, a_shape, b_shape, dims, equation
    ):
        rng = np.random.default_rng(317)
        a, b, da, db = [
            rng.normal(size=s).astype(np.float32)
            for s in (a_shape, b_shape, a_shape, b_shape)
        ]
        lhs, output = equation.split("->")
        left, right = lhs.split(",")
        dot = lambda a, b: np.einsum(equation, a, b)
        a64, b64, da64, db64 = (x.astype(np.float64) for x in (a, b, da, db))
        expected = dot(a64, b64)
        weight = rng.normal(size=expected.shape).astype(np.float32)
        w64 = weight.astype(np.float64)
        value = np.sum(expected * w64)
        tangent = np.sum((dot(da64, b64) + dot(a64, db64)) * w64)
        ga = lambda b: np.einsum(f"{output},{right}->{left}", w64, b)
        gb = lambda a: np.einsum(f"{left},{output}->{right}", a, w64)
        gradient = (2 * value * ga(b64), 2 * value * gb(a64))
        hvp = (
            2 * tangent * ga(b64) + 2 * value * ga(db64),
            2 * tangent * gb(a64) + 2 * value * gb(da64),
        )
        inputs = tuple(jnp.asarray(x) for x in (a, b))
        directions = tuple(jnp.asarray(x) for x in (da, db))
        reference = lambda a, b: jax.lax.dot_general(
            a, b, dims, precision=jax.lax.Precision.HIGHEST
        )
        source = str(jax.jit(reference).lower(*inputs).compiler_ir())
        product = lambda a, b: hlo_call(a, b, source=source)[0]
        loss = lambda a, b: jnp.sum(product(a, b) * jnp.asarray(weight)) ** 2
        grad = jax.grad(loss, argnums=(0, 1))
        actual = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(*inputs)
        forward_hvp = jax.jit(lambda a, b: jax.jvp(grad, (a, b), directions)[1])(
            *inputs
        )
        reverse_hvp = jax.jit(
            jax.grad(
                lambda a, b: sum(
                    jnp.sum(g * d) for g, d in zip(grad(a, b), directions)
                ),
                argnums=(0, 1),
            )
        )(*inputs)
        # Small bilinear products and two AD levels accumulate f32 roundoff.
        tolerance = 512 * np.finfo(np.float32).eps
        for result, oracle in zip(
            jax.tree.leaves(
                (jax.jit(product)(*inputs), actual, forward_hvp, reverse_hvp)
            ),
            jax.tree.leaves((expected, (value**2, gradient), hvp, hvp)),
        ):
            np.testing.assert_allclose(result, oracle, rtol=tolerance, atol=tolerance)

    @parameterized.named_parameters(
        ("output_limit", 16, 16, 8, np.float32, False),
        ("output_fallback", 17, 16, 8, np.float32, True),
        ("contraction_fallback", 16, 17, 8, np.float32, True),
        ("f64_fallback", 16, 16, 8, np.float64, True),
        ("unit", 3, 1, 4, np.float32, None),
        ("empty", 3, 0, 4, np.float32, None),
        ("left_vector", 1, 8, 4, np.float32, True),
        ("right_vector", 4, 8, 1, np.float32, True),
    )
    def test_general_size_and_precision_bounds(self, m, k, n, dtype, native):
        a = jnp.arange(2 * m * k, dtype=dtype).reshape(2, m, k) / 11
        b = jnp.arange(2 * k * n, dtype=dtype).reshape(2, k, n) / 13
        dims = (((2,), (1,)), ((0,), (0,)))
        source = str(
            jax.jit(
                lambda a, b: jax.lax.dot_general(
                    a, b, dims, precision=jax.lax.Precision.HIGHEST
                )
            )
            .lower(a, b)
            .compiler_ir()
        )
        lowered = jax.jit(lambda a, b: hlo_call(a, b, source=source)[0]).lower(a, b)
        tolerance = 32 * max(1, k) * np.finfo(dtype).eps
        np.testing.assert_allclose(
            lowered.compile()(a, b), np.asarray(a) @ np.asarray(b), rtol=tolerance
        )
        if native is not None:
            self.assertEqual(
                "stablehlo.dot_general" in str(lowered.compiler_ir()),
                native or jax.default_backend() != "cpu",
            )

    @parameterized.named_parameters(("matrix", False), ("batched", True))
    def test_explicit_algorithm_and_multiplatform_export(self, batched):
        from jax import export

        a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7
        b = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 11
        dims = (((1,), (0,)), ((), ()))
        if batched:
            a, b = jnp.stack((a, a + 1)), jnp.stack((b, b + 1))
            dims = (((2,), (1,)), ((0,), (0,)))

        def specified(a, b):
            return jax.lax.dot_general(
                a,
                b,
                dims,
                precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
            )

        source = str(jax.jit(specified).lower(a, b).compiler_ir())
        function = lambda a, b: hlo_call(a, b, source=source)[0]
        lowered = jax.jit(function).lower(a, b)
        self.assertIn("algorithm =", str(lowered.compiler_ir()))
        np.testing.assert_allclose(
            lowered.compile()(a, b), np.asarray(a) @ np.asarray(b), rtol=3e-6
        )

        plain_source = str(
            jax.jit(
                lambda a, b: jax.lax.dot_general(
                    a, b, dims, precision=jax.lax.Precision.HIGHEST
                )
            )
            .lower(a, b)
            .compiler_ir()
        )
        portable = lambda a, b: hlo_call(a, b, source=plain_source)[0]
        for platforms in (("cpu", "cuda"), ("cuda",)):
            exported = export.export(jax.jit(portable), platforms=platforms)(a, b)
            self.assertIn("stablehlo.dot_general", exported.mlir_module())
            if jax.default_backend() in platforms:
                np.testing.assert_allclose(
                    exported.call(a, b), np.asarray(a) @ np.asarray(b), rtol=3e-6
                )

    @parameterized.named_parameters(
        ("f32", "f32", np.float32, 4e-5), ("f64", "f64", np.float64, 3e-11)
    )
    def test_rectangular_contractions_preserve_gradients_and_hessians(
        self, precision, dtype, tolerance
    ):
        rng = np.random.default_rng(317)
        a, b, da, db, weight = [
            rng.normal(size=s) for s in ((3, 5), (5, 4), (3, 5), (5, 4), (3, 4))
        ]
        value = np.sum((a @ b) * weight)
        tangent = np.sum((da @ b + a @ db) * weight)
        ga, gb = weight @ b.T, a.T @ weight
        gradient = (2 * value * ga, 2 * value * gb)
        hvp = (
            2 * tangent * ga + 2 * value * (weight @ db.T),
            2 * tangent * gb + 2 * value * (da.T @ weight),
        )
        for left, right in ((1, 0), (0, 0), (1, 1), (0, 1)):
            orient = lambda pair: (
                pair[0] if left == 1 else pair[0].T,
                pair[1] if right == 0 else pair[1].T,
            )
            inputs = tuple(jnp.asarray(x, dtype=dtype) for x in orient((a, b)))
            directions = tuple(jnp.asarray(x, dtype=dtype) for x in orient((da, db)))
            source = source_for(3, 5, 4, precision, left, right)
            loss = lambda a, b: (
                jnp.sum(
                    hlo_call(a, b, source=source)[0] * jnp.asarray(weight, dtype=dtype)
                )
                ** 2
            )
            actual = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(*inputs)
            grad = jax.grad(loss, argnums=(0, 1))
            forward_hvp = jax.jit(lambda a, b: jax.jvp(grad, (a, b), directions)[1])(
                *inputs
            )
            reverse_hvp = jax.jit(
                jax.grad(
                    lambda a, b: sum(
                        jnp.sum(g * d) for g, d in zip(grad(a, b), directions)
                    ),
                    argnums=(0, 1),
                )
            )(*inputs)
            for result, expected in zip(
                jax.tree.leaves((actual, forward_hvp, reverse_hvp)),
                jax.tree.leaves(
                    ((value**2, orient(gradient)), orient(hvp), orient(hvp))
                ),
            ):
                np.testing.assert_allclose(
                    result, expected, rtol=tolerance, atol=tolerance
                )

    def test_size_boundary_empty_contraction_and_shared_batch(self):
        for m, k, n in ((3, 0, 4), (3, 1, 4), (3, 16, 4), (3, 17, 4), (17, 4, 3)):
            a = np.arange(m * k, dtype=np.float32).reshape(m, k) / 11
            b = np.arange(k * n, dtype=np.float32).reshape(k, n) / 13
            source = source_for(m, k, n, "f32")
            f = lambda a, b: hlo_call(a, b, source=source)[0]
            lowered = jax.jit(f).lower(a, b)
            np.testing.assert_allclose(
                lowered.compile()(a, b), a @ b, rtol=3e-6, atol=3e-6
            )
            if jax.default_backend() == "cpu" and max(m, k, n) >= 16:
                self.assertEqual(
                    "stablehlo.dot_general" in str(lowered.compiler_ir()),
                    max(m, k, n) > 16,
                )
            mapped = jax.jit(jax.vmap(f, in_axes=(None, 0)))
            batch = np.stack([b, b + 1])
            np.testing.assert_allclose(
                mapped(a, batch), a @ batch, rtol=3e-6, atol=3e-6
            )


if __name__ == "__main__":
    absltest.main()
