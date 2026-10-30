import os
import platform
import tempfile
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from enzyme_ad.jax import cpp_call, primitives


def libcxx_include(flags):
    return flags[flags.index("-isystem") + 1]


class MacOSFlags(absltest.TestCase):
    def test_sdkroot_selects_the_sdk(self):
        sdk = "/sdks/MacOSX.sdk"
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch.dict(os.environ, {"SDKROOT": sdk}),
        ):
            flags = primitives.cflags()
        self.assertEqual(flags[flags.index("-isysroot") + 1], sdk)
        self.assertEqual(
            libcxx_include(flags), os.path.join(sdk, "usr", "include", "c++", "v1")
        )

    @absltest.skipUnless(platform.system() == "Darwin", "macOS host only")
    def test_libcxx_headers_exist_on_host(self):
        include = libcxx_include(primitives.cflags())
        self.assertTrue(os.path.isfile(os.path.join(include, "cstdint")), include)


class ClangResources(absltest.TestCase):
    def test_cpp_call_compiles_builtin_header(self):
        source = """
        #include <stdarg.h>
        template<typename T>
        void f(T& out, const T& input) { out[0] = input[0] + 7.0f; }
        """
        shape = jax.core.ShapedArray((1,), np.float32)
        function = jax.jit(lambda x: cpp_call(x, source=source, out_shapes=(shape,))[0])
        np.testing.assert_array_equal(function(jnp.array([2], dtype=jnp.float32)), [9])

    def test_wheel_resources_precede_bazel_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            module = root / "wheel/enzyme_ad/jax"
            wheel = root / "wheel/clang/staging"
            bazel = root / "external/llvm-project/clang/staging"
            module.mkdir(parents=True)
            (wheel / "include").mkdir(parents=True)
            (bazel / "include").mkdir(parents=True)
            with (
                mock.patch.object(
                    primitives.enzyme_call, "__file__", str(module / "enzyme_call.so")
                ),
                mock.patch.dict(os.environ),
            ):
                os.environ.pop("ENZYME_BAZEL_NOWHEEL", None)
                self.assertEqual(Path(primitives.resource_dir()).resolve(), wheel)

    def test_bazel_resources_without_wheel_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            module = root / "wheel/enzyme_ad/jax"
            bazel = root / "external/llvm-project/clang/staging"
            module.mkdir(parents=True)
            (bazel / "include").mkdir(parents=True)
            with (
                mock.patch.object(
                    primitives.enzyme_call, "__file__", str(module / "enzyme_call.so")
                ),
                mock.patch.dict(os.environ),
            ):
                os.environ.pop("ENZYME_BAZEL_NOWHEEL", None)
                self.assertEqual(Path(primitives.resource_dir()).resolve(), bazel)

    def test_explicit_resource_layout_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary) / "enzyme_ad/jax/enzyme_call.so"
            with (
                mock.patch.object(primitives.enzyme_call, "__file__", str(module)),
                mock.patch.dict(os.environ, {"ENZYME_BAZEL_NOWHEEL": ""}),
            ):
                self.assertEqual(
                    Path(primitives.resource_dir()).resolve(),
                    (Path(temporary) / "clang/staging").resolve(),
                )


if __name__ == "__main__":
    absltest.main()
