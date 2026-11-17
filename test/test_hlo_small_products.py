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
    def test_explicit_algorithm_and_multiplatform_export(self):
        from jax import export

        a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7
        b = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 11

        def specified(a, b):
            return jax.lax.dot_general(
                a,
                b,
                (((1,), (0,)), ((), ())),
                precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
            )

        source = str(jax.jit(specified).lower(a, b).compiler_ir())
        function = lambda a, b: hlo_call(a, b, source=source)[0]
        lowered = jax.jit(function).lower(a, b)
        self.assertIn("algorithm =", str(lowered.compiler_ir()))
        np.testing.assert_allclose(
            lowered.compile()(a, b), np.asarray(a) @ np.asarray(b), rtol=3e-6
        )

        portable = lambda a, b: hlo_call(a, b, source=source_for(3, 4, 5, "f32"))[0]
        exported = export.export(jax.jit(portable), platforms=("cpu", "cuda"))(a, b)
        self.assertIn("stablehlo.dot_general", exported.mlir_module())
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
