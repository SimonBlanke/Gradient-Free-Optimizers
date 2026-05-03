"""Build configuration for optional C extension.

The C extension provides accelerated array operations for the pure Python
backend. If compilation fails (no C compiler), the package installs
normally and falls back to pure Python.
"""

import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_c_source = os.path.join(
    "src",
    "gradient_free_optimizers",
    "_array_backend",
    "_fast_ops.c",
)


class OptionalBuildExt(build_ext):
    """Build C extensions, silently skip on failure.

    Set GFO_REQUIRE_C_EXTENSION=1 to make compilation failures fatal
    (used in CI to prevent silent fallback to pure Python).
    """

    def build_extension(self, ext):  # noqa: D102
        try:
            super().build_extension(ext)
        except Exception as e:
            if os.environ.get("GFO_REQUIRE_C_EXTENSION"):
                raise RuntimeError(
                    f"C extension build failed (GFO_REQUIRE_C_EXTENSION is set): {e}"
                ) from e
            print(
                f"WARNING: Could not build C extension '{ext.name}': {e}\n"
                f"Falling back to pure Python array backend."
            )


_ext_modules = []
if not os.environ.get("GFO_DISABLE_C_EXTENSION"):
    _ext_modules.append(
        Extension(
            "gradient_free_optimizers._array_backend._fast_ops",
            sources=[_c_source],
            extra_compile_args=["-O2"],
        ),
    )

setup(
    cmdclass={"build_ext": OptionalBuildExt},
    ext_modules=_ext_modules,
)
