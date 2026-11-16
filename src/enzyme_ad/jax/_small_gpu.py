"""Compiler setup for small, explicit CUDA program lowering."""

import ctypes
import json
import operator
import os
import re
from functools import cache
from pathlib import Path


def pipeline(chip, toolkit_path, max_tensor_elements, features):
    if not re.fullmatch(r"sm_[0-9]+[a-z]?", chip):
        raise ValueError("chip must name a CUDA architecture, such as sm_90")
    limit = operator.index(max_tensor_elements)
    if limit < 1:
        raise ValueError("max_tensor_elements must be positive")
    if not re.fullmatch(r"[+a-zA-Z0-9_,.-]*", features):
        raise ValueError("features must contain LLVM target features")
    toolkit = (
        Path(
            toolkit_path
            or os.environ.get("CUDA_ROOT")
            or os.environ.get("CUDA_PATH")
            or "/usr/local/cuda"
        )
        .expanduser()
        .resolve()
    )
    quoted = json.dumps(str(toolkit), ensure_ascii=False)
    passes = (
        "inline,symbol-dce,lower-chlo-gamma-f32,"
        "func.func(chlo-legalize-to-stablehlo),"
        "lower-small-program-control,stablehlo-legalize-to-linalg,canonicalize,"
        f"lower-small-tensor-values{{max-elements={limit}}},canonicalize,"
        "outline-small-gpu-program,lower-kernel,convert-vector-to-llvm,"
        "convert-polygeist-to-llvm{backend=cuda preserve-xla-funcs=true},"
        f'nvvm-attach-target{{chip={chip} features="{features}"}},'
        "enzymexla-gpu-module-to-binary{format=fatbin "
        f'toolkit={quoted} llvm-opts="-nvptx-fma-level=0"}},'
        "lower-jit,symbol-privatize{exclude=main},symbol-dce"
    )
    return passes, str(toolkit)


@cache
def load_runtime(toolkit):
    root = Path(toolkit)
    candidates = [root / "lib64/libcudart.so"]
    candidates.extend(sorted(root.glob("lib64/libcudart.so.*")))
    candidates.extend(sorted(root.glob("targets/*/lib/libcudart.so*")))
    for library in candidates:
        if library.is_file():
            # The host launch JIT resolves CUDA symbols from the process. Keep
            # this handle alive for the same lifetime as its cached kernels.
            return ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    raise RuntimeError(f"CUDA runtime library not found under {root}")
