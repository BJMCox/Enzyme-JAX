from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.interpreters import mlir
from jaxlib.mlir import ir
import numpy as np

from enzyme_ad.jax import enzyme_call, hlo_call


def power_product(form, exponents, dtype=None):
    dtype = dtype or ("f64" if jax.config.x64_enabled else "f32")
    tensor = f"tensor<4x{dtype}>"
    definitions = []
    operands = []
    for i, value in enumerate(exponents):
        operand = f"%arg{i + 1}"
        if value is not None:
            operand = f"%c{i}"
            definitions.append(
                f"{operand} = stablehlo.constant dense<{value:.17e}> : {tensor}"
            )
        operands.append(operand)
    left, right = (
        ("%p", "%q"),
        ("%p", "%arg0"),
        ("%arg0", "%p"),
        ("%p", "%square"),
        ("%square", "%p"),
    )[form]
    constants = "\n".join(definitions)
    source = f"""
    func.func @main(%arg0: {tensor}, %arg1: {tensor}, %arg2: {tensor}) -> {tensor} {{
      {constants}
      %p = stablehlo.power %arg0, {operands[0]} : {tensor}
      %q = stablehlo.power %arg0, {operands[1]} : {tensor}
      %square = stablehlo.multiply %arg0, %arg0 : {tensor}
      %result = stablehlo.multiply {left}, {right} : {tensor}
      return %result : {tensor}
    }}
    """
    _, optimized = enzyme_call.run_pass_pipeline(
        [],
        source,
        "canonicalize,cse,enzyme-hlo-generate-td{patterns=power_multiply_to_power},"
        "transform-interpreter,enzyme-hlo-remove-transform,canonicalize,cse",
    )
    with mlir.make_ir_context():
        module = ir.Module.parse(optimized)
        for function in module.body:
            if function.sym_name.value == "main":
                function.attributes["sym_visibility"] = ir.StringAttr.get("public")
        return str(module)


class PowerProducts(absltest.TestCase):
    def test_low_precision_coefficient_overflow_declines(self):
        source = power_product(0, (200.0, 200.0), "f16")
        self.assertIn("stablehlo.multiply", source)

    def test_domains_and_signed_zero(self):
        x = jnp.array([-1.0, -0.0, 0.0, 2.0])
        for exponents in ((None, None), (0.5, 0.5), (-1.0, 1.0)):
            a, b = (0.5 if value is None else value for value in exponents)
            for form in range(5):
                with self.subTest(exponents=exponents, form=form):
                    source = power_product(form, exponents)
                    actual = jax.jit(
                        lambda x: hlo_call(
                            x,
                            jnp.full_like(x, a),
                            jnp.full_like(x, b),
                            source=source,
                            passes="symbol-dce",
                        )[0]
                    )(x)
                    host = np.asarray(x)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        p = np.power(host, np.full_like(host, a))
                        q = np.power(host, np.full_like(host, b))
                        expected = (
                            p * q,
                            p * host,
                            host * p,
                            p * (host * host),
                            (host * host) * p,
                        )[form]
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
                    zeros = expected == 0
                    np.testing.assert_array_equal(
                        np.signbit(np.asarray(actual)[zeros]),
                        np.signbit(expected[zeros]),
                    )

    def test_integer_powers_and_higher_derivatives(self):
        x = jnp.array([-2.0, -1.0, 0.0, 2.0])
        for form, degree in enumerate((5, 3, 3, 4, 4)):
            with self.subTest(form=form):
                source = power_product(form, (2.0, 3.0))
                self.assertEqual(source.count("stablehlo.power"), 1)

                def function(x):
                    return hlo_call(x, x, x, source=source, passes="symbol-dce")[
                        0
                    ].sum()

                gradient = jax.grad(function)
                np.testing.assert_allclose(
                    jax.jit(gradient)(x), degree * x ** (degree - 1)
                )
                expected = degree * (degree - 1) * x ** (degree - 2)
                for derivative in (
                    lambda x: jax.jvp(gradient, (x,), (jnp.ones_like(x),))[1],
                    jax.grad(lambda x: gradient(x).sum()),
                ):
                    np.testing.assert_allclose(jax.jit(derivative)(x), expected)

    def test_large_integer_powers_preserve_gradient_sign(self):
        exponent = float(2 ** (53 if jax.config.x64_enabled else 24))
        source = power_product(0, (exponent, exponent))
        x = jnp.full(4, -1.0)

        def function(x):
            return hlo_call(x, x, x, source=source, passes="symbol-dce")[0].sum()

        np.testing.assert_array_equal(jax.jit(jax.grad(function))(x), -2 * exponent)


if __name__ == "__main__":
    absltest.main()
