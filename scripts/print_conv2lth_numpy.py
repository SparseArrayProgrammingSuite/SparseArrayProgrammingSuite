#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from onnxpy import compile_model

from saps.benchmarks.lth_conv2_onnxpy import _model_path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="saps_conv2lth_numpy_") as tmpdir:
        generated_path = Path(tmpdir) / "conv2_generated.py"
        compile_model(_model_path(), generated_path)
        sys.stdout.write(generated_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
