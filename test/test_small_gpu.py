import ctypes
import os
import struct
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src.interpreters import mlir
from jax.experimental import checkify
from jaxlib.mlir import ir

from enzyme_ad.jax import SmallGPUPipeline, enzyme_call, enzyme_jax_ir


class SmallGpu(absltest.TestCase):
    def setUp(self):
        super().setUp()
        if jax.default_backend() not in ("gpu", "cuda"):
            self.skipTest("CUDA small-program profile")
        self.profile = SmallGPUPipeline(os.environ.get("ENZYME_CUDA_CHIP", "sm_80"))

    def compile(self, function):
        return enzyme_jax_ir(pipeline_options=self.profile)(function)

    def test_value_carries_and_dropped_scatter_updates(self):
        def function(n, row):
            def body(state):
                i, a, b = state
                return i + 1, b, a + b

            _, a, b = jax.lax.while_loop(
                lambda state: state[0] < n,
                body,
                (jnp.int32(0), jnp.int32(2), jnp.int32(3)),
            )
            values = jnp.arange(6, dtype=jnp.int32).reshape(2, 3)
            values = values.at[row].add(jnp.array([a, b, a + b]), mode="drop")
            return {"values": values, "ordered": a < b}

        compiled = self.compile(function)
        for n, row in ((0, 0), (1, 1), (5, 0), (5, 2), (5, -3)):
            a, b = 2, 3
            for _ in range(n):
                a, b = b, a + b
            expected = np.arange(6, dtype=np.int32).reshape(2, 3)
            if row in (0, 1):
                expected[row] += [a, b, a + b]
            result = compiled(jnp.int32(n), jnp.int32(row))
            np.testing.assert_array_equal(result["values"], expected)
            self.assertEqual(bool(result["ordered"]), a < b)

    def test_typed_rng_key_and_batched_layout(self):
        def draw(key, value):
            first, second = jax.random.split(key)
            return jax.random.key_data(second), value + jax.random.uniform(
                first, value.shape, dtype=value.dtype
            )

        key = jax.random.key(41)
        values = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        expected = jax.jit(draw)(key, values)
        actual = self.compile(draw)(key, values)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

        def matrix(value):
            padded = jnp.pad(value, ((1, 0), (0, 1)), constant_values=0.125)
            return padded.T + 2 * padded.T

        batch = jnp.stack([values, values + 7])
        actual = jax.jit(jax.vmap(self.compile(matrix)))(batch)
        padded = np.pad(
            np.asarray(batch), ((0, 0), (1, 0), (0, 1)), constant_values=0.125
        )
        np.testing.assert_array_equal(actual, 3 * padded.transpose(0, 2, 1))

    def test_checked_error_survives_fusion(self):
        def function(value):
            checkify.check(value > 0, "value must be positive")
            return jnp.log(value)

        compiled = self.compile(checkify.checkify(function))
        error, result = compiled(jnp.float32(2))
        error.throw()
        np.testing.assert_allclose(result, np.log(np.float32(2)), rtol=2e-7)
        error, _ = compiled(jnp.float32(-1))
        with self.assertRaisesRegex(checkify.JaxRuntimeError, "value must be positive"):
            error.throw()

    def test_tensor_limit_is_explicit(self):
        compiled = self.compile(lambda value: value + 1)
        with self.assertRaises(ValueError):
            compiled(jnp.zeros(129, dtype=jnp.float32))

    def test_foreign_buffer_alignment_preserves_all_bits(self):
        for dtype in (np.float32, np.float64):
            width = np.dtype(dtype).itemsize
            element = f"f{8 * width}"
            source = f"""
            module {{
              func.func @main(%x: tensor<{element}>) -> (tensor<{element}>, tensor<i1>) {{
                %c = stablehlo.constant dense<0.125> : tensor<{element}>
                %zero = stablehlo.constant dense<0.0> : tensor<{element}>
                %sum = stablehlo.add %x, %c : tensor<{element}>
                %positive = stablehlo.compare GT, %x, %zero : (tensor<{element}>, tensor<{element}>) -> tensor<i1>
                return %sum, %positive : tensor<{element}>, tensor<i1>
              }}
            }}
            """
            _, lowered = enzyme_call.run_pass_pipeline(
                [], source, self.profile.pass_pipeline()
            )
            with mlir.make_ir_context():
                module = ir.Module.parse(lowered)
                function = next(iter(module.body.operations))
                call = next(
                    op
                    for op in function.regions[0].blocks[0].operations
                    if op.operation.name == "stablehlo.custom_call"
                )
                config = ir.DictAttr(call.attributes["backend_config"])
                packed = ir.StringAttr(config["attr"]).value_bytes
            run_type = ctypes.CFUNCTYPE(
                None, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
            )
            run = run_type(struct.unpack("P", packed)[0])
            offsets = [0, 0]

            @run_type
            def callback(buffers, stream, offsets=offsets, run=run):
                shifted = (ctypes.c_void_p * 3)(
                    buffers[0] + offsets[0],
                    buffers[1] + offsets[1],
                    buffers[2] + offsets[1],
                )
                run(shifted, stream)

            address = ctypes.cast(callback, ctypes.c_void_p).value
            shape = jax.ShapeDtypeStruct((2 * width,), np.uint8)
            invoke = jax.ffi.ffi_call(
                "enzymexla_compile_gpu",
                (shape, shape),
                input_output_aliases={1: 0, 2: 1},
                has_side_effect=True,
            )
            call = jax.jit(partial(invoke, attr=struct.pack("P", address)))
            value = dtype(0.6931471805599453)
            expected = np.frombuffer((value + dtype(0.125)).tobytes(), np.uint8)
            first = jnp.full(2 * width, 0xA5, dtype=jnp.uint8)
            second = jnp.full(2 * width, 0x5A, dtype=jnp.uint8)
            for input_offset in range(width):
                raw = np.zeros(2 * width, np.uint8)
                raw[input_offset : input_offset + width] = np.frombuffer(
                    value.tobytes(), np.uint8
                )
                data = jnp.asarray(raw)
                for output_offset in range(width):
                    offsets[:] = input_offset, output_offset
                    result, predicate = jax.block_until_ready(call(data, first, second))
                    wanted = np.full(2 * width, 0xA5, np.uint8)
                    wanted[output_offset : output_offset + width] = expected
                    np.testing.assert_array_equal(result, wanted)
                    wanted = np.full(2 * width, 0x5A, np.uint8)
                    wanted[output_offset] = 1
                    np.testing.assert_array_equal(predicate, wanted)
                np.testing.assert_array_equal(data, raw)


class SmallGpuPlacement(absltest.TestCase):
    def test_cpu_requires_its_own_pipeline(self):
        if jax.default_backend() != "cpu":
            self.skipTest("CPU placement")
        function = enzyme_jax_ir(pipeline_options=SmallGPUPipeline("sm_90"))(
            lambda value: value + 1
        )
        with self.assertRaises(NotImplementedError):
            function(jnp.float32(1))


if __name__ == "__main__":
    absltest.main()
