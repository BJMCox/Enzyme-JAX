from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)
SOURCE = Path(__file__).with_name("logabsdet_composite.mlir").read_text()


class LogabsdetCompositeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("f32", "f32", np.float32, 3e-5), ("f64", "f64", np.float64, 3e-11)
    )
    def test_values_derivatives_and_output_activity(self, precision, dtype, tolerance):
        source = (
            SOURCE.replace("0x7FF0000000000000", "0x7F800000").replace(
                "0x7FF8000000000000", "0x7FC00000"
            )
            if precision == "f32"
            else SOURCE
        )
        source = source.replace("f64", precision)
        a = np.array([[0.3, 2.0], [1.0, 3.0]], dtype=dtype)
        direction = np.array([[0.2, -0.1], [0.4, 0.3]], dtype=dtype)
        weight = np.array([[0.2, -0.3], [0.9, 0.5]], dtype=dtype)
        inverse = np.linalg.inv(a.astype(float))
        dinverse = -inverse @ direction @ inverse
        dtranspose = dinverse.T
        logvalue = np.linalg.slogdet(a.astype(float))[1]
        for log_weight, inverse_weight in ((1, 0), (0, 1), (1, 1)):

            def loss(a):
                value, result, _ = hlo_call(a, source=source)
                return (value if log_weight else 0) + (
                    jnp.sum(result * weight) if inverse_weight else 0
                )

            expected_value = log_weight * logvalue + inverse_weight * np.sum(
                inverse * weight
            )
            expected_gradient = (
                log_weight * inverse.T - inverse_weight * inverse.T @ weight @ inverse.T
            )
            expected_hvp = log_weight * dtranspose - inverse_weight * (
                dtranspose @ weight @ inverse.T + inverse.T @ weight @ dtranspose
            )
            for function in (loss, jax.checkpoint(loss)):
                actual = jax.jit(jax.value_and_grad(function))(a)
                for value, oracle in zip(actual, (expected_value, expected_gradient)):
                    np.testing.assert_allclose(
                        value, oracle, rtol=tolerance, atol=tolerance
                    )
                gradient = jax.grad(function)
                _, pushforward = jax.linearize(jax.jit(gradient), a)
                np.testing.assert_allclose(
                    jax.jit(pushforward)(direction),
                    expected_hvp,
                    rtol=tolerance,
                    atol=tolerance,
                )
                reverse_hvp = jax.jit(
                    jax.grad(lambda a: jnp.sum(gradient(a) * direction))
                )(a)
                np.testing.assert_allclose(
                    reverse_hvp, expected_hvp, rtol=tolerance, atol=tolerance
                )

        singular = np.zeros_like(a)
        values, inverses, valid = jax.jit(
            jax.vmap(lambda a: hlo_call(a, source=source))
        )(np.stack([a, singular, a]))
        np.testing.assert_allclose(np.asarray(values)[[0, 2]], logvalue, rtol=tolerance)
        self.assertTrue(np.isneginf(values[1]))
        np.testing.assert_allclose(
            np.asarray(inverses)[[0, 2]],
            np.stack([inverse, inverse]),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_array_equal(valid, [True, False, True])
        gradient = jax.jit(jax.vmap(jax.grad(lambda a: hlo_call(a, source=source)[0])))(
            np.stack([a, singular, a])
        )
        np.testing.assert_allclose(
            np.asarray(gradient)[[0, 2]],
            np.stack([inverse.T, inverse.T]),
            rtol=tolerance,
            atol=tolerance,
        )
        compiled = jax.jit(lambda a: hlo_call(a, source=source)[0]).lower(a).compile()
        # Primal-only evaluation needs one factorization and no inverse solve.
        self.assertLen(
            re.findall(r'custom_call_target="[^"\n]*getrf[^"\n]*"', compiled.as_text()),
            1,
        )
        self.assertNotRegex(
            compiled.as_text(), r'custom_call_target="[^"\n]*(?:trsm|trtrs)[^"\n]*"'
        )


if __name__ == "__main__":
    absltest.main()
