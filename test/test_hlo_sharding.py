from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from enzyme_ad.jax import hlo_call


def native(x, scale):
    return scale * x * x + x**3


SOURCE = str(jax.jit(native).lower(jnp.float32(0), jnp.float32(1)).compiler_ir())


def imported(x, scale):
    return hlo_call(x, scale, source=SOURCE)[0]


class HloSharding(absltest.TestCase):
    def setUp(self):
        super().setUp()
        if jax.device_count() < 2:
            self.skipTest("requires two devices")
        self.mesh = Mesh(np.array(jax.devices()), ("devices",))
        self.x = jnp.arange(jax.device_count(), dtype=jnp.float32) - 0.5
        self.scale = jnp.float32(2)

    def mapped(self, function):
        return jax.shard_map(
            lambda x, scale: function(x[0], scale)[None],
            mesh=self.mesh,
            in_specs=(P("devices"), P()),
            out_specs=P("devices"),
        )

    def test_shared_parameter_gradient_sums_across_devices(self):
        mapped = self.mapped(imported)
        value, (gx, gs) = jax.jit(
            jax.value_and_grad(lambda x, s: jnp.sum(mapped(x, s)), argnums=(0, 1))
        )(self.x, self.scale)
        np.testing.assert_allclose(value, jnp.sum(native(self.x, self.scale)))
        np.testing.assert_allclose(gx, 2 * self.scale * self.x + 3 * self.x**2)
        np.testing.assert_allclose(gs, jnp.sum(self.x**2))

    def test_value_and_gradient_inside_shard_map(self):
        def mapped(function):
            return jax.shard_map(
                lambda x, s: jax.tree.map(
                    lambda v: v[None],
                    jax.value_and_grad(function, argnums=(0, 1))(x[0], s),
                ),
                mesh=self.mesh,
                in_specs=(P("devices"), P()),
                out_specs=(P("devices"), (P("devices"), P("devices"))),
            )(self.x, self.scale)

        for value, expected in zip(
            jax.tree.leaves(mapped(imported)), jax.tree.leaves(mapped(native))
        ):
            np.testing.assert_allclose(value, expected)

    def test_two_manual_axes_normalize_variance(self):
        if jax.device_count() < 4:
            self.skipTest("requires four devices")
        mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("a", "b"))
        x = jnp.array([[-1.0], [2.0]])
        scale = jnp.array([[0.5, 3.0]])
        mapped = jax.shard_map(
            lambda x, s: imported(x[0, 0], s[0, 0])[None, None],
            mesh=mesh,
            in_specs=(P("a", None), P(None, "b")),
            out_specs=P("a", "b"),
        )
        np.testing.assert_allclose(mapped(x, scale), native(x, scale))
        gx, gs = jax.jit(jax.grad(lambda x, s: jnp.sum(mapped(x, s)), argnums=(0, 1)))(
            x, scale
        )
        np.testing.assert_allclose(
            gx, jnp.sum(2 * scale * x + 3 * x**2, axis=1, keepdims=True)
        )
        np.testing.assert_allclose(gs, jnp.broadcast_to(jnp.sum(x**2), scale.shape))

    def test_auto_partitioning_and_explicit_mesh_boundary(self):
        def native_total(x):
            return jnp.sum(x) ** 2

        source = str(jax.jit(native_total).lower(self.x).compiler_ir())
        total = lambda x: hlo_call(x, source=source)[0]
        sharded = NamedSharding(self.mesh, P("devices"))
        replicated = NamedSharding(self.mesh, P())
        value, grad = jax.jit(
            jax.value_and_grad(total),
            in_shardings=sharded,
            out_shardings=(replicated, sharded),
        )(self.x)
        np.testing.assert_allclose(value, native_total(self.x))
        np.testing.assert_allclose(grad, jnp.full_like(self.x, 2 * jnp.sum(self.x)))

        mesh = Mesh(
            np.array(jax.devices()), ("devices",), axis_types=(AxisType.Explicit,)
        )
        with jax.set_mesh(mesh):
            total = jax.sharding.auto_axes(
                jax.value_and_grad(total), out_sharding=(P(), P("devices"))
            )
            x = jax.device_put(self.x, NamedSharding(mesh, P("devices")))
            value, grad = jax.jit(total)(x)
            np.testing.assert_allclose(value, native_total(self.x))
            np.testing.assert_allclose(grad, jnp.full_like(self.x, 2 * jnp.sum(self.x)))

    def test_pmap_remains_compatible(self):
        value, grad = jax.pmap(jax.value_and_grad(imported), in_axes=(0, None))(
            self.x, self.scale
        )
        np.testing.assert_allclose(value, native(self.x, self.scale))
        np.testing.assert_allclose(grad, 2 * self.scale * self.x + 3 * self.x**2)

    def test_device_ids_cannot_be_declared_invariant(self):
        source = """
        module {
          func.func @main() -> tensor<ui32> {
            %id = stablehlo.partition_id : tensor<ui32>
            return %id : tensor<ui32>
          }
        }
        """
        # The imported ID previously received empty variance, silently allowing
        # one partition's ID to stand for every device. Keep collectives in JAX.
        mapped = jax.shard_map(
            lambda: hlo_call(source=source)[0],
            mesh=self.mesh,
            in_specs=(),
            out_specs=P(),
        )
        with self.assertRaises(NotImplementedError):
            mapped()

    def test_scan_residuals_retain_manual_axes(self):
        def recurrence(x, scale):
            def step(carry, item):
                carry = jnp.sin(carry * scale + item)
                return carry, carry

            return jax.lax.scan(step, x, jnp.array([0.5, -1.0, 0.25]))[0]

        source = str(
            jax.jit(recurrence).lower(jnp.float32(0), jnp.float32(1)).compiler_ir()
        )
        imported = lambda x, s: hlo_call(x, s, source=source)[0]

        def evaluated(function):
            mapped = self.mapped(function)
            return jax.jit(
                jax.value_and_grad(lambda x, s: jnp.sum(mapped(x, s)), argnums=(0, 1))
            )(self.x, self.scale)

        for value, expected in zip(
            jax.tree.leaves(evaluated(imported)), jax.tree.leaves(evaluated(recurrence))
        ):
            np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-6)

    def test_unused_output_does_not_poison_cotangents(self):
        source = str(
            jax.jit(lambda x: (x * x, jnp.log(-x))).lower(jnp.float32(1)).compiler_ir()
        )
        imported = lambda x, _: hlo_call(x, source=source)[0]
        mapped = self.mapped(imported)
        x = self.x + 2
        grad = jax.jit(jax.grad(lambda x: jnp.sum(mapped(x, self.scale))))(x)
        np.testing.assert_allclose(grad, 2 * x)


if __name__ == "__main__":
    absltest.main()
