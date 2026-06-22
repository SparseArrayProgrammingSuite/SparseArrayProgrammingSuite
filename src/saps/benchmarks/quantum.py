from typing import Any

import numpy as np

import saps
from saps.benchmark import Benchmark, Contributor, DataInstance, Dataset, Generator, Ref
from saps_framework import BinsparseFormat

xp = saps.xp


class QuantumDataset(Dataset):
    def __init__(self, source_name: str, nqubits: int, description: str):
        self._suites: list[str] = []
        self.source_name = source_name
        self.nqubits = nqubits
        self.dataset_description = description

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"Quantum {self.source_name}"

    @property
    def description(self) -> str:
        return self.dataset_description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nqubits"] = self.nqubits
        return data


# BEGIN COPIED TEST FILE: tests/test_quantum.py
# import pytest
#
# import numpy as np
#
# import saps.benchmarks.quantum as quantum
# from frameworks.saps_numpy import NumpyFramework
# from saps.benchmarks.quantum import (
#     QGates,
#     QuantumStatevectorBenchmark,
#     apply_single_qubit_gate,
# )
# from saps_framework import BinsparseFormat
#
#
# def run_quantum_benchmark(xp, state, nqubits):
#     benchmark = QuantumStatevectorBenchmark()
#     prev_xp = getattr(quantum, "xp", None)
#     quantum.xp = xp
#     try:
#         (final_state,) = benchmark.benchmark(
#             [state], {"nqubits": nqubits, "num_layers": 1}
#         )
#     finally:
#         quantum.xp = prev_xp
#     return final_state
#
#
# @pytest.mark.parametrize("xp", [NumpyFramework()])
# def test_quantum_statevector_basic(xp):
#     """
#     Test that RQC statevector simulation runs without errors
#     and produces correct output shape and dtype.
#     """
#     nqubits = 10
#     dim = 1 << nqubits
#     state_np = np.zeros(dim, dtype=np.complex128)
#     state_np[0] = 1.0 + 0j
#     state = xp.from_binsparse(BinsparseFormat.from_numpy(state_np))
#
#     final_state = run_quantum_benchmark(xp, state, nqubits)
#     final_state_bin = xp.to_binsparse(final_state)
#
#     # Expected shape: 2**nqubits complex entries
#     assert final_state_bin.data["shape"] == (dim,)
#     assert final_state_bin.data["values"].dtype == np.complex128
#
#     # Very basic sanity: norm should be close to 1 (unitary evolution)
#     vals = final_state_bin.data["values"]
#     norm = np.sqrt(np.sum(np.abs(vals) ** 2))
#     assert abs(norm - 1.0) < 1e-4, f"Final state norm not preserved: {norm:.6f}"
#
#     print(f"RQC statevector basic test passed with {xp.__class__.__name__}")
#
#
# @pytest.mark.parametrize(
#     "gate_np, gate_name",
#     [
#         (QGates.H, "H"),
#         (QGates.X, "X"),
#         (QGates.Y, "Y"),
#         (QGates.Z, "Z"),
#         (QGates.S, "S"),
#         (QGates.T, "T"),
#     ],
# )
# @pytest.mark.parametrize("qubit", [0, 1, 2, 3])
# def test_every_gate_on_zero_state(gate_np, gate_name, qubit):
#     nqubits = 4
#     xp = NumpyFramework()
#
#     # Prepare |000...0⟩
#     dim = 1 << nqubits
#     state_np = np.zeros(dim, dtype=np.complex128)
#     state_np[0] = 1.0
#     state = xp.from_binsparse(BinsparseFormat.from_numpy(state_np))
#
#     # Prepare gate
#     gate_xp = xp.from_binsparse(BinsparseFormat.from_numpy(gate_np))
#
#     # Apply gate
#     state_after = apply_single_qubit_gate(xp, state, gate_xp, qubit, nqubits)
#
#     computed = state_after
#     bench = xp.to_binsparse(computed)
#     result = np.array(bench.data["values"], dtype=np.complex128).reshape(
#         bench.data["shape"]
#     )
#
#     expected = np.zeros(dim, dtype=np.complex128)
#     flipped_idx = 1 << (nqubits - 1 - qubit)
#     expected[0] = gate_np[0, 0]  # new=0, old=0
#     expected[flipped_idx] = gate_np[1, 0]  # new=1, old=0
#
#     np.testing.assert_allclose(
#         result,
#         expected,
#         atol=1e-13,
#         rtol=1e-13,
#         err_msg=f"Gate {gate_name} on qubit {qubit} failed (n={nqubits})",
#     )
#
#
# def test_H_twice_returns_to_original():
#     nqubits = 5
#     xp = NumpyFramework()
#
#     dim = 1 << nqubits
#     state_np = np.zeros(dim, dtype=np.complex128)
#     state_np[0] = 1.0
#     state = xp.from_binsparse(BinsparseFormat.from_numpy(state_np))
#
#     H_xp = xp.from_binsparse(BinsparseFormat.from_numpy(QGates.H))
#
#     mid = apply_single_qubit_gate(xp, state, H_xp, 2, nqubits)
#     back = apply_single_qubit_gate(xp, mid, H_xp, 2, nqubits)
#
#     computed = back
#     bench = xp.to_binsparse(computed)
#     result = np.array(bench.data["values"], dtype=np.complex128).reshape(
#         bench.data["shape"]
#     )
#
#     np.testing.assert_allclose(result, state_np, atol=1e-13)
# END COPIED TEST FILE: tests/test_quantum.py

class QuantumStateGenerator(Generator[QuantumDataset]):
    @property
    def name(self) -> str:
        return "quantum_state_inputs"

    @property
    def pretty_name(self) -> str:
        return "Quantum Statevector Data Generator"

    @property
    def description(self) -> str:
        return "Generates zero-initialized n-qubit state vectors."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return QuantumStatevectorBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return QuantumStatevectorBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return QuantumStatevectorBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return QuantumStatevectorBenchmark().motivation

    @property
    def datasets(self) -> list[QuantumDataset]:
        return [
            QuantumDataset(
                "single_layer_large", 40, "Small instance of 10 qubits, 1 layer"
            ),
            QuantumDataset(
                "single_layer_small", 10, "Small instance of 10 qubits, 1 layer"
            ),
            QuantumDataset(
                "single_layer_tiny", 5, "Tiny instance of 5 qubits, 1 layer"
            ),
        ]

    def generate(self, dataset: QuantumDataset) -> DataInstance:
        nqubits = dataset.nqubits
        dim = 1 << nqubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0j  # |000...0
        state_bin = BinsparseFormat.from_numpy(state)
        return DataInstance(inputs=[state_bin], meta={"nqubits": nqubits})


def apply_single_qubit_gate(xp, state, gate, qubit, nqubits):
    left = 1 << qubit
    right = 1 << (nqubits - qubit - 1)
    start_resh = xp.reshape(state, (left, 2, right))
    # gate[new, old] convention => einsum "ijk,lj->ilk"
    new_resh = xp.einsum(
        "new_resh[i, j, k] += start_resh[i, l, k] * gate[j, l]",
        start_resh=start_resh,
        gate=gate,
    )
    return xp.reshape(new_resh, state.shape)


class QGates:
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    all_gates = [H, X, Y, Z, S, T]


class QuantumStatevectorBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "rqc_statevector"

    @property
    def pretty_name(self) -> str:
        return "Random Quantum Circuit Statevector"

    @property
    def description(self) -> str:
        return (
            "Simulates a random quantum circuit on an n-qubit state vector, "
            "using the standard reshape + einsum gate application pattern."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return ""

    @property
    def motivation(self) -> str:
        return self.description

    @property
    def generators(self):
        return [QuantumStateGenerator()]

    def benchmark(self, data: list[Any], meta: dict[str, Any]):
        nqubits = meta["nqubits"]
        rng = np.random.default_rng(seed=42)
        single_qubit_gates = QGates.all_gates

        # Pre-build all the gates we will need for the circuit
        gates = []
        for _ in range(nqubits):
            g_np = rng.choice(single_qubit_gates)
            g_bench = BinsparseFormat.from_numpy(g_np)
            g_xp = xp.from_binsparse(g_bench)
            gates.append(g_xp)

        # Load the initial state
        state = data[0]

        # Apply each gate to each qubit sequentially.
        for q in range(nqubits):
            state = apply_single_qubit_gate(xp, state, gates[q], q, nqubits)

        return [state]
