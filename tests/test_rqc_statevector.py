from fractions import Fraction
from types import SimpleNamespace

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmark import Param
from saps.benchmarks.rqc_statevector import (
    QGates,
    RQCStatevectorBenchmark,
    product_state_reference,
    statevector_error_bound,
)

GATES = ["H", "X", "Y", "Z", "S", "T"]


def _exact_state(nqubits, gate_sequence):
    """The circuit's state in exact rational arithmetic on the float64 gates."""

    def exact(z):
        return Fraction(z.real), Fraction(z.imag)

    def multiply(a, b):
        return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]

    zero, one = (Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))
    qubits = [[one, zero] for _ in range(nqubits)]
    for name, qubit in gate_sequence:
        gate, (a, b) = getattr(QGates, name), qubits[qubit]
        qubits[qubit] = [
            tuple(
                x + y
                for x, y in zip(
                    multiply(exact(gate[row, 0]), a),
                    multiply(exact(gate[row, 1]), b),
                    strict=True,
                )
            )
            for row in range(2)
        ]
    state = [one]
    for vector in qubits:
        state = [multiply(a, b) for a in state for b in vector]
    return state


def _within(state, exact, bound):
    squared = sum(
        (Fraction(z.real) - re) ** 2 + (Fraction(z.imag) - im) ** 2
        for z, (re, im) in zip(state, exact, strict=True)
    )
    return squared <= Fraction(bound) ** 2


def test_gates_are_unitary_up_to_rounding():
    eps = np.finfo(np.float64).eps
    for name in GATES:
        gate = getattr(QGates, name)
        assert np.linalg.norm(gate, 2) <= 1 + 2 * eps, name
        assert np.linalg.norm(gate) <= np.sqrt(2) * (1 + 2 * eps), name


@pytest.mark.parametrize("seed", range(5))
def test_bounds_cover_exact_arithmetic_on_deep_circuits(seed):
    rng = np.random.default_rng(seed)
    nqubits = 4
    gate_sequence = [
        (str(rng.choice(GATES)), int(rng.integers(nqubits))) for _ in range(60)
    ]
    exact = _exact_state(nqubits, gate_sequence)
    xp = NumpyFramework()
    inputs = [
        product_state_reference(nqubits, []),
        *(getattr(QGates, name) for name in GATES),
    ]
    output = RQCStatevectorBenchmark().benchmark(
        xp, {"nqubits": nqubits, "gate_sequence": gate_sequence}, *inputs
    )
    steps = len(gate_sequence)
    assert _within(np.asarray(output), exact, statevector_error_bound(steps))
    reference = product_state_reference(nqubits, gate_sequence)
    assert _within(reference, exact, statevector_error_bound(steps + nqubits - 1))


@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
def test_every_dataset_passes_its_bound_check(framework):
    benchmark = RQCStatevectorBenchmark()
    for generator in benchmark.generators:
        for dataset in generator.datasets:
            param = Param(generator, dataset)
            benchmark.setup(param, use_cache=False, xp=framework())
            benchmark.run(param)
            benchmark.teardown(param)


def test_check_rejects_a_sign_flip_that_keeps_the_norm():
    benchmark = RQCStatevectorBenchmark()
    generator = benchmark.generators[0]
    dataset = next(d for d in generator.datasets if d.name == "basic")
    problem = generator.generate(dataset)
    benchmark._meta = problem.meta
    benchmark._ref_outputs = problem.ref_outputs
    benchmark._ref_meta = problem.ref_meta
    param = SimpleNamespace(dataset=dataset)

    state = to_numpy(problem.ref_outputs[0])
    benchmark._output = [from_numpy(state)]
    benchmark.check(param)

    flipped = state.copy()
    flipped[np.flatnonzero(flipped)[0]] *= -1
    benchmark._output = [from_numpy(flipped)]
    with pytest.raises(AssertionError, match="beyond the bound"):
        benchmark.check(param)
