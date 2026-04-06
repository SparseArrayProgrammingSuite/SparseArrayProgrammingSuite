import os

from .frameworks import numpy_framework, sparse_framework

xp = os.environ.get("SAPS_FRAMEWORK", "np")
xp = {
    "np": numpy_framework.NumpyFramework(),
    "sparse": sparse_framework.PyDataSparseFramework(),
}[xp]
