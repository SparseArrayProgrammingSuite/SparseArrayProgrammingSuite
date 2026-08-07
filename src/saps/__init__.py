import ssgetpy.matrix as ssm

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Ref,
    ShellBenchmark,
    ccs_xml_to_tags,
)
from saps.framework import load_framework
from saps.ssgetpy_patch import _patched_ssget_download
from saps.storage import build_storage_backend

ssm.Matrix.download = _patched_ssget_download

__all__ = [
    "Author",
    "Benchmark",
    "Contributor",
    "DataInstance",
    "Ref",
    "ShellBenchmark",
    "build_storage_backend",
    "ccs_xml_to_tags",
    "load_framework",
]
