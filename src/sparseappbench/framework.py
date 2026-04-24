import os

from .frameworks import numpy_framework, scipy_framework, sparse_framework

framework_name = os.environ.get("SAPS_FRAMEWORK", "np")
xp = {
    "np": numpy_framework.NumpyFramework(),
    "scipy": scipy_framework.SciPyFramework(),
    "sparse": sparse_framework.PyDataSparseFramework(),
}[framework_name]
