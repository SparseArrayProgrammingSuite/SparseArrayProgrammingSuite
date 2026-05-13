import pytest

import numpy as np

import saps.benchmarks.preconditioned_cg as pcg
from frameworks.saps_numpy import NumpyFramework


def as_dense(array):
    if hasattr(array, "todense"):
        return np.asarray(array.todense()).ravel()
    return np.asarray(array).ravel()


def run_preconditioned_cg(xp, data, meta):
    benchmark = pcg.PreconditionedCGBenchmark()
    prev_xp = getattr(pcg, "xp", None)
    pcg.xp = xp
    try:
        (x_sol,) = benchmark.benchmark(data, meta)
    finally:
        pcg.xp = prev_xp
    return x_sol


A0 = np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]])
A1 = np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]])
A2 = np.array(
    [
        [8.0, -1.0, 0.0, 0.0],
        [-1.0, 8.0, -1.0, 0.0],
        [0.0, -1.0, 8.0, -1.0],
        [0.0, 0.0, -1.0, 8.0],
    ]
)
A3 = np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]])
A4 = np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]])
A5 = np.array(
    [
        [15.0, -2.0, 0.0, 0.0, -1.0],
        [-2.0, 14.0, -3.0, 0.0, 0.0],
        [0.0, -3.0, 16.0, -2.0, 0.0],
        [0.0, 0.0, -2.0, 15.0, -3.0],
        [-1.0, 0.0, 0.0, -3.0, 17.0],
    ]
)


@pytest.mark.parametrize(
    "generator",
    [
        pcg.BlockJacobiCGGenerator(),
        pcg.JacobiCGGenerator(),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pcg.PreconditionedCGDataset("A0", "", A=A0),
        pcg.PreconditionedCGDataset("A1", "", A=A1),
        pcg.PreconditionedCGDataset("A2", "", A=A2),
        pcg.PreconditionedCGDataset("A3", "", A=A3),
        pcg.PreconditionedCGDataset("A4", "", A=A4),
        pcg.PreconditionedCGDataset("A5", "", A=A5),
    ],
)
def test_preconditioned_cg(generator, dataset):
    xp = NumpyFramework()
    data_bin, meta = generator.generate(dataset)
    data = [xp.from_binsparse(array) for array in data_bin]

    A, b, x, M = data
    x_sol = run_preconditioned_cg(xp, [A, b, x, M], meta)

    residual = as_dense(b - A @ x_sol)
    assert np.linalg.norm(residual) < 1e-6 * np.linalg.norm(as_dense(b)) + 1e-6
