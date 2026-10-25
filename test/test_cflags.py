import os
import platform
from unittest import mock

from absl.testing import absltest

from enzyme_ad.jax import primitives


def libcxx_include(flags):
    return flags[flags.index("-isystem") + 1]


class MacOSFlags(absltest.TestCase):
    def test_sdkroot_selects_the_sdk(self):
        sdk = "/sdks/MacOSX.sdk"
        with mock.patch("platform.system", return_value="Darwin"), mock.patch.dict(
            os.environ, {"SDKROOT": sdk}
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


if __name__ == "__main__":
    absltest.main()
