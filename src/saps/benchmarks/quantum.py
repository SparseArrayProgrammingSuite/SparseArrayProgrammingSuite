from typing import Any

import numpy as np

import saps
from saps.benchmark import Benchmark, Contributor, Dataset, Generator, Ref
from saps_framework import BinsparseFormat

xp = saps.xp


class QuantumDataset(Dataset):
    def __init__(self, source_name: str, nqubits: int, description: str):
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
    def tags(self) -> list[str]:
        return ["quantum", "statevector"]

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nqubits"] = self.nqubits
        return data


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
    def tags(self) -> list[str]:
        return ["quantum", "statevector"]

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

    def generate(
        self, dataset: QuantumDataset
    ) -> tuple[list[BinsparseFormat], dict[str, Any]]:
        nqubits = dataset.nqubits
        dim = 1 << nqubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0j  # |000...0
        state_bin = BinsparseFormat.from_numpy(state)
        return [state_bin], {"nqubits": nqubits}


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
    def tags(self) -> list[str]:
        return ["quantum", "statevector", "einsum"]

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
