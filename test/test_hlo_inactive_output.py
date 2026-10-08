from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call


class InactiveOutput(absltest.TestCase):
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
