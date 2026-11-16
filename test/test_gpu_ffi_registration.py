import ctypes
import struct

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import enzyme_call


class GpuFfiRegistration(absltest.TestCase):
    def test_handler_capsules(self):
        for with_error in (False, True):
            self.assertIn(
                'capsule object "xla._CUSTOM_CALL_TARGET"',
                repr(enzyme_call.enzymexla_gpu_handler(with_error)),
            )

    def require_gpu(self):
        if jax.default_backend() not in ("gpu", "cuda", "rocm"):
            self.skipTest("GPU FFI plugin")

    def call(self, name, callback):
        values = jnp.arange(4, dtype=jnp.float32)
        address = ctypes.cast(callback, ctypes.c_void_p).value
        invoke = jax.ffi.ffi_call(
            name,
            jax.ShapeDtypeStruct(values.shape, values.dtype),
            input_output_aliases={0: 0},
            has_side_effect=True,
        )
        return jax.jit(lambda x: invoke(x, attr=struct.pack("P", address)))(values)

    def test_normal_handler_reaches_the_gpu_plugin(self):
        self.require_gpu()
        calls = []

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
        def callback(buffers, stream):
            calls.append((buffers[0], stream))

        result = self.call("enzymexla_compile_gpu", callback)
        np.testing.assert_array_equal(result, np.arange(4, dtype=np.float32))
        self.assertLen(calls, 1)
        self.assertIsNotNone(calls[0][0])
        self.assertIsNotNone(calls[0][1])

    def test_error_handler_reaches_the_gpu_plugin(self):
        self.require_gpu()
        message = ctypes.create_string_buffer(b"synthetic GPU launch failure")

        @ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        )
        def callback(buffers, stream):
            return ctypes.addressof(message)

        with self.assertRaisesRegex(
            jax.errors.JaxRuntimeError, "synthetic GPU launch failure"
        ):
            jax.block_until_ready(
                self.call("enzymexla_compile_gpu_with_error", callback)
            )


if __name__ == "__main__":
    absltest.main()
