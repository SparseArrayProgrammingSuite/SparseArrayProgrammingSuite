import pytest

import numpy as np

from saps.benchmarks.quantum import (
    QGates,
    QuantumStatevectorBenchmark,
    apply_single_qubit_gate,
)
import saps.benchmarks.quantum as quantum
from saps_framework import BinsparseFormat
from frameworks.saps_numpy import NumpyFramework


def run_quantum_benchmark(xp, state, nqubits):
    benchmark = QuantumStatevectorBenchmark()
    prev_xp = getattr(quantum, "xp", None)
    quantum.xp = xp
    try:
        (final_state,) = benchmark.benchmark(
            [state], {"nqubits": nqubits, "num_layers": 1}
        )
    finally:
        quantum.xp = prev_xp
    return final_state


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_quantum_statevector_basic(xp):
    """
    Test that RQC statevector simulation runs without errors
    and produces correct output shape and dtype.
    """
    nqubits = 10
    dim = 1 << nqubits
    state_np = np.zeros(dim, dtype=np.complex128)
    state_np[0] = 1.0 + 0j
    state = xp.from_binsparse(BinsparseFormat.from_numpy(state_np))

    final_state = run_quantum_benchmark(xp, state, nqubits)
    final_state_bin = xp.to_binsparse(final_state)

    # Expected shape: 2**nqubits complex entries
    assert final_state_bin.data["shape"] == (dim,)
    assert final_state_bin.data["values"].dtype == np.complex128

    # Very basic sanity: norm should be close to 1 (unitary evolution)
    vals = final_state_bin.data["values"]
    norm = np.sqrt(np.sum(np.abs(vals) ** 2))
    assert abs(norm - 1.0) < 1e-4, f"Final state norm not preserved: {norm:.6f}"

    print(f"RQC statevector basic test passed with {xp.__class__.__name__}")


@pytest.mark.parametrize(
    "gate_np, gate_name",
    [
        (QGates.H, "H"),
        (QGates.X, "X"),
        (QGates.Y, "Y"),
        (QGates.Z, "Z"),
        (QGates.S, "S"),
        (QGates.T, "T"),
    ],
)
@pytest.mark.parametrize("qubit", [0, 1, 2, 3])
def test_every_gate_on_zero_state(gate_np, gate_name, qubit):
    nqubits = 4
    xp = NumpyFramework()

    # Prepare |000...0⟩
    dim = 1 << nqubits
    state_np = np.zeros(dim, dtype=np.complex128)
    state_np[0] = 1.0
    state = xp.from_binsparse(BinsparseFormat.from_numpy(state_np))

    # Prepare gate
    gate_xp = xp.from_binsparse(BinsparseFormat.from_numpy(gate_np))

    # Apply gate
    state_after = apply_single_qubit_gate(xp, state, gate_xp, qubit, nqubits)

    computed = state_after
    bench = xp.to_binsparse(computed)
    result = np.array(bench.data["values"], dtype=np.complex128).reshape(
        bench.data["shape"]
    )

    expected = np.zeros(dim, dtype=np.complex128)
    flipped_idx = 1 << (nqubits - 1 - qubit)
    expected[0] = gate_np[0, 0]  # new=0, old=0
    expected[flipped_idx] = gate_np[1, 0]  # new=1, old=0

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-13,
        rtol=1e-13,
        err_msg=f"Gate {gate_name} on qubit {qubit} failed (n={nqubits})",
    )


def test_H_twice_returns_to_original():
    nqubits = 5
    xp = NumpyFramework()

    dim = 1 << nqubits
    state_np = np.zeros(dim, dtype=np.complex128)
    state_np[0] = 1.0
    state = xp.from_binsparse(BinsparseFormat.from_numpy(state_np))

    H_xp = xp.from_binsparse(BinsparseFormat.from_numpy(QGates.H))

    mid = apply_single_qubit_gate(xp, state, H_xp, 2, nqubits)
    back = apply_single_qubit_gate(xp, mid, H_xp, 2, nqubits)

    computed = back
    bench = xp.to_binsparse(computed)
    result = np.array(bench.data["values"], dtype=np.complex128).reshape(
        bench.data["shape"]
    )

    np.testing.assert_allclose(result, state_np, atol=1e-13)
