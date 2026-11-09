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
    circuit_instance,
    named_gates,
    simulate_reference,
    statevector_error_bound,
)
from saps.downloaders.qasmbench import parse_transpiled_qasm

ONE_QUBIT_GATES = ["H", "X", "Y", "Z", "S", "T", "SX"]


def _exact_state(nqubits, gates):
    """The circuit's state in exact rational arithmetic on the float64 gates."""

    def exact(z):
        return Fraction(z.real), Fraction(z.imag)

    def multiply(a, b):
        return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]

    def add(a, b):
        return a[0] + b[0], a[1] + b[1]

    zero = (Fraction(0), Fraction(0))
    state = [zero] * (1 << nqubits)
    state[0] = (Fraction(1), Fraction(0))
    for name, qubits, angle in gates:
        matrix = QGates.RZ(angle) if name == "RZ" else getattr(QGates, name)
        # Qubit 0 is the most significant bit of an amplitude's index.
        bits = [nqubits - 1 - qubit for qubit in qubits]
        matrix = matrix.reshape(1 << len(qubits), 1 << len(qubits))
        new = [zero] * len(state)
        for index, amplitude in enumerate(state):
            old = 0
            for bit in bits:
                old = 2 * old + ((index >> bit) & 1)
            for row in range(1 << len(qubits)):
                target = index
                for position, bit in enumerate(bits):
                    value = (row >> (len(bits) - 1 - position)) & 1
                    target = (target & ~(1 << bit)) | (value << bit)
                product = multiply(exact(matrix[row, old]), amplitude)
                new[target] = add(new[target], product)
        state = new
    return state


def _within(state, exact, bound):
    squared = sum(
        (Fraction(z.real) - re) ** 2 + (Fraction(z.imag) - im) ** 2
        for z, (re, im) in zip(state, exact, strict=True)
    )
    return squared <= Fraction(bound) ** 2


def test_gates_are_unitary_up_to_rounding():
    eps = np.finfo(np.float64).eps
    gates = {name: getattr(QGates, name) for name in ONE_QUBIT_GATES}
    gates.update({f"RZ({angle})": QGates.RZ(angle) for angle in (0.3, -2.5, np.pi)})
    for name, gate in gates.items():
        assert np.linalg.norm(gate, 2) <= 1 + 2 * eps, name
        assert np.linalg.norm(gate) <= np.sqrt(2) * (1 + 2 * eps), name
    cx = QGates.CX.reshape(4, 4)
    assert np.array_equal(cx @ cx.conj().T, np.eye(4))


@pytest.mark.parametrize("seed", range(5))
def test_bounds_cover_exact_arithmetic_on_deep_circuits(seed):
    rng = np.random.default_rng(seed)
    nqubits = 4
    sequence = []
    for _ in range(40):
        kind = rng.integers(3)
        if kind == 0:
            control, target = (int(q) for q in rng.choice(nqubits, 2, replace=False))
            sequence.append(("CX", control, target))
        elif kind == 1:
            sequence.append(("RZ", int(rng.integers(nqubits)), float(rng.normal())))
        else:
            name = str(rng.choice(ONE_QUBIT_GATES))
            sequence.append((name, int(rng.integers(nqubits))))
    gates = named_gates(sequence)
    exact = _exact_state(nqubits, gates)
    problem = circuit_instance(nqubits, gates)
    state, gates1, gates2 = (to_numpy(tensor) for tensor in problem.inputs)
    output = RQCStatevectorBenchmark().benchmark(
        NumpyFramework(), problem.meta, state, gates1, gates2
    )
    two_qubit = sum(len(qubits) == 2 for _, qubits, _ in gates)
    bound = statevector_error_bound(len(gates) - two_qubit, two_qubit)
    assert _within(np.asarray(output), exact, bound)
    reference = simulate_reference(
        nqubits, state, gates1, gates2, problem.meta["gate_sequence"]
    )
    assert _within(reference, exact, bound)


def test_sparse_storage_round_trips_through_the_check():
    nqubits, gates = parse_transpiled_qasm(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[3];
        creg c[5];
        rz(pi/2) a[0];
        sx a[0];
        rz(pi/2) a[0];
        cx a[0],b[2];
        x b[0];
        cx b[2],a[1];
        measure a[0] -> c[0];
        """
    )
    assert nqubits == 5
    assert gates[3] == ("cx", (0, 4), None)
    assert gates[5] == ("cx", (4, 1), None)
    problem = circuit_instance(nqubits, gates, sparse_storage=True)
    with pytest.raises(TypeError):
        to_numpy(problem.inputs[0])
    for framework in (NumpyFramework, SmartSparseFramework):
        benchmark = RQCStatevectorBenchmark()
        benchmark._input = problem.inputs
        benchmark._meta = problem.meta
        benchmark._ref_outputs = problem.ref_outputs
        benchmark._ref_meta = problem.ref_meta
        xp = framework()
        output = benchmark.benchmark(
            xp, problem.meta, *(xp.from_binsparse(t) for t in problem.inputs)
        )
        benchmark._output = [xp.to_binsparse(output)]
        benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="inline")))


@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
def test_every_dataset_passes_its_bound_check(framework):
    benchmark = RQCStatevectorBenchmark()
    # The QASMBench circuits need a download and up to 23 qubits.
    for generator in benchmark.generators[:2]:
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
