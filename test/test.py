from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import cpp_call, enzyme_jax_ir, hlo_call
import test_utils

jax.config.update("jax_platforms", "cpu")

argv = test_utils.argv


class EnzymePipeline(absltest.TestCase):
    def test_pipeline(self):
        def fn(x):
            return x

        x = jnp.ones(3)
        if hasattr(fn, "trace"):
            module = jax.jit(fn).trace(x).lower().compiler_ir(dialect="stablehlo")
            # Only applies if jax-mlir and enzyme-mlir are built on the same version
            # optimize_module(module)
            print(str(module))


class EnzymeJax(absltest.TestCase):
    def test_rsqrt_scaled_derivatives(self):
        source = """
        module {
          func.func @main(%x: tensor<6xf32>) -> tensor<6xf32> {
            %y = stablehlo.rsqrt %x : tensor<6xf32>
            return %y : tensor<6xf32>
          }
        }
        """
        function = lambda x: hlo_call(x, source=source)[0]
        x = np.array([4, 9, 1e-30, 1e30, 0.5, 0.5], dtype=np.float32)
        seed = np.array(
            [1, 1, 1e-30, 1e30, 2e38, np.finfo(np.float32).tiny], dtype=np.float32
        )
        expected = -0.5 * seed.astype(np.float64) / x.astype(np.float64) ** 1.5
        forward = jax.jit(lambda x, d: jax.jvp(function, (x,), (d,))[1])
        reverse = jax.jit(lambda x, d: jax.vjp(function, x)[1](d)[0])
        np.testing.assert_allclose(forward(x, seed), expected, rtol=3e-6, atol=0)
        np.testing.assert_allclose(reverse(x, seed), expected, rtol=3e-6, atol=0)

    def test_rsqrt_complex_jvp(self):
        @enzyme_jax_ir(argv=argv)
        def function(x):
            return jax.lax.rsqrt(x)

        x = np.array([-2, 0.5j, -0.5j], dtype=np.complex64)
        seed = np.full_like(x, 1 + 0.5j)
        expected = (
            -0.5
            * seed.astype(np.complex128)
            / (x.astype(np.complex128) * np.sqrt(x.astype(np.complex128)))
        )
        forward = jax.jit(lambda x, d: jax.jvp(function, (x,), (d,))[1])
        np.testing.assert_allclose(forward(x, seed), expected, rtol=3e-6, atol=0)

    def test_custom_cpp_kernel(self):
        @jax.jit
        def do_something(ones):
            shape = jax.core.ShapedArray(ones.shape, ones.dtype)
            a, b = cpp_call(
                ones,
                out_shapes=[shape, shape],
                source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0,
                  enzyme::tensor<float, N, M>& out1,
                  const enzyme::tensor<float, N, M>& in0) {
          for (int j=0; j<N; j++) {
            for (int k=0; k<M; k++) {
                out0[j][k] = in0[j][k] + 42;
            }
          }
          for (int j=0; j<2; j++) {
            for (int k=0; k<3; k++) {
                out1[j][k] = in0[j][k] + 2 * 42;
            }
          }
        }
        """,
                fn="myfn",
                argv=argv,
            )
            c = cpp_call(
                a,
                out_shapes=[jax.core.ShapedArray([4, 4], jnp.float32)],
                source="""
        template<typename T1, typename T2>
        void f(T1& out0, const T2& in1) {
          out0 = 56.0f;
        }
        """,
                argv=argv,
            )
            return a, b, c

        ones = jnp.ones((2, 3), jnp.float32)
        x, y, z = do_something(ones)

        self.assertTrue((x == 43).all())
        self.assertTrue((y == 85).all())
        self.assertTrue((z[0] == 56).all())

        # JVP
        primals, tangents = jax.jvp(do_something, (ones,), (ones,))
        self.assertTrue((primals[0] == 43).all())
        self.assertTrue((primals[1] == 85).all())
        self.assertTrue((primals[2][0] == 56).all())
        self.assertTrue((tangents[0] == 1).all())
        self.assertTrue((tangents[1] == 1).all())
        self.assertTrue((tangents[2][0] == 0).all())

        # VJP
        primals, f_vjp = jax.vjp(do_something, ones)
        (grads,) = f_vjp((x, y, z))
        self.assertTrue((primals[0] == 43).all())
        self.assertTrue((primals[1] == 85).all())
        self.assertTrue((primals[2][0] == 56).all())

        self.assertTrue(
            (
                grads[1]
                == jnp.array(
                    [
                        [128.0, 128.0, 128.0],
                    ]
                )
            ).all()
        )

    def test_enzyme_mlir_jit(self):
        @jax.jit
        @enzyme_jax_ir(argv=argv)
        def add_one(x: jax.Array, y) -> jax.Array:
            return x + 1 + y

        add_one(jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0]))

        primals, tangents = jax.jvp(
            add_one,
            (jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0])),
            (jnp.array([0.1, 0.2, 0.3]), jnp.array([50.0, 70.0, 110.0])),
        )
        self.assertTrue(
            (
                primals
                == jnp.array(
                    [
                        [12.0, 23.0, 34.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                tangents
                == jnp.array(
                    [
                        [50.1, 70.2, 110.3],
                    ]
                )
            ).all()
        )

        primals, f_vjp = jax.vjp(
            add_one, jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0])
        )
        grads = f_vjp(jnp.array([500.0, 700.0, 110.0]))
        self.assertTrue(
            (
                primals
                == jnp.array(
                    [
                        [12.0, 23.0, 34.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                grads[0]
                == jnp.array(
                    [
                        [500.0, 700.0, 110.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                grads[1]
                == jnp.array(
                    [
                        [500.0, 700.0, 110.0],
                    ]
                )
            ).all()
        )


if __name__ == "__main__":
    absltest.main()
