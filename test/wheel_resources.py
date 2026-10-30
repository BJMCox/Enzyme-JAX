"""Check a wheel against the compiler's builtin-header outputs."""

import argparse
import zipfile
from pathlib import Path


def check_resources(wheel, headers):
    with zipfile.ZipFile(wheel) as archive:
        for header in headers:
            relative = str(header).split("/clang/staging/", 1)[1]
            packaged = f"clang/staging/{relative}"
            assert archive.read(packaged) == header.read_bytes(), packaged
    print(f"Validated {len(headers)} builtin headers in {wheel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--headers", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    check_resources(args.wheel, args.headers)
