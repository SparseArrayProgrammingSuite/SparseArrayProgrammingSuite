"""ONNXPY: compile ONNX models directly to Python + NumPy source."""

from .compiler import compile_to_source, supported_ops

__all__ = ["compile_to_source", "supported_ops"]
