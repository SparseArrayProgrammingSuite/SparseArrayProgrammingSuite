"""ONNXPY compiler: direct ONNX → Python+NumPy source generation."""

from saps.util.onnxpy.compiler.emit import compile_to_source
from saps.util.onnxpy.compiler.ops import supported_ops

__all__ = ["compile_to_source", "supported_ops"]
