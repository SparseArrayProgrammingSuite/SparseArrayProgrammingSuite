import pytest
import numpy as np 
from sparseappbench.benchmarks.quantum import (
    benchmark_rqc_statevector,
    dg_single_layer_small,
    dg_single_layer_tiny,
    apply_single_qubit_gate,
    QGates
)

from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.checker_framework import CheckerFramework

@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_quantum_statevector_basic(xp):
    """
    Test that RQC statevector simulation runs without errors
    and produces correct output shape and dtype.
    """
    state_bin, nqubits = dg_single_layer_small()
    final_state_bin = benchmark_rqc_statevector(xp, state_bin, nqubits, num_layers=1)

    # Expected shape: 2**nqubits complex entries
    expected_dim = 1 << nqubits
    assert final_state_bin.data["shape"] == (expected_dim,)
    assert final_state_bin.data["values"].dtype == np.complex128

    # Very basic sanity: norm should be close to 1 (unitary evolution)
    vals = final_state_bin.data["values"]
    norm = np.sqrt(np.sum(np.abs(vals)**2))
    assert abs(norm - 1.0) < 1e-4, f"Final state norm not preserved: {norm:.6f}"

    print(f"RQC statevector basic test passed with {xp.__class__.__name__}")


@pytest.mark.parametrize("gate_np, gate_name", [
    (QGates.H, "H"),
    (QGates.X, "X"),
    (QGates.Y, "Y"),
    (QGates.Z, "Z"),
    (QGates.S, "S"),
    (QGates.T, "T"),
])
@pytest.mark.parametrize("qubit", [0, 1, 2, 3])
def test_every_gate_on_zero_state(gate_np, gate_name, qubit):
    nqubits = 4
    xp = NumpyFramework()

    # Prepare |000...0⟩
    dim = 1 << nqubits
    state_np = np.zeros(dim, dtype=np.complex128)
    state_np[0] = 1.0
    state = xp.from_benchmark(BinsparseFormat.from_numpy(state_np))

    # Prepare gate
    gate_xp = xp.from_benchmark(BinsparseFormat.from_numpy(gate_np))

    # Apply gate
    state_after = apply_single_qubit_gate(xp, state, gate_xp, qubit, nqubits)

    computed = xp.compute(state_after)
    bench = xp.to_benchmark(computed)
    result = np.array(bench.data["values"], dtype=np.complex128).reshape(bench.data["shape"])

    expected = np.zeros(dim, dtype=np.complex128)
    flipped_idx = 1 << (nqubits - 1 - qubit)
    expected[0]          = gate_np[0, 0]   # new=0, old=0
    expected[flipped_idx] = gate_np[1, 0]   # new=1, old=0

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-13,
        rtol=1e-13,
        err_msg=f"Gate {gate_name} on qubit {qubit} failed (n={nqubits})"
    )
    
def test_H_twice_returns_to_original():
    nqubits = 5
    xp = NumpyFramework()

    dim = 1 << nqubits
    state_np = np.zeros(dim, dtype=np.complex128)
    state_np[0] = 1.0
    state = xp.from_benchmark(BinsparseFormat.from_numpy(state_np))

    H_xp = xp.from_benchmark(BinsparseFormat.from_numpy(QGates.H))

    mid = apply_single_qubit_gate(xp, state, H_xp, 2, nqubits)
    back = apply_single_qubit_gate(xp, mid, H_xp, 2, nqubits)

    computed = xp.compute(back)
    bench = xp.to_benchmark(computed)
    result = np.array(bench.data["values"], dtype=np.complex128).reshape(bench.data["shape"])

    np.testing.assert_allclose(result, state_np, atol=1e-13)