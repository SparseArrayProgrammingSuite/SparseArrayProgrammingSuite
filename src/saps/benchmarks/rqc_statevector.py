from typing import Any

import numpy as np

import sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_sparse, to_numpy, to_sparse

from saps.benchmark import Benchmark, Contributor, DataInstance, Dataset, Generator, Ref
from saps.benchmarks.qasmbench import QASMBENCH_REFERENCES, fetch_qasmbench_circuit
from saps.util.error_bounds import summation_error_bound


class RQCStatevectorDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        nqubits: int,
        description: str,
        gate_sequence: list[tuple[Any, ...]],
        suites: list[str] | None = None,
        pretty_name: str | None = None,
    ):
        self._suites = suites or []
        self.source_name = source_name
        self._pretty_name = pretty_name or source_name
        self.nqubits = nqubits
        self.dataset_description = description
        self.gate_sequence = gate_sequence

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

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
        data["gate_sequence"] = self.gate_sequence
        return data


class RQCStatevectorSyntheticGenerator(Generator[RQCStatevectorDataset]):
    @property
    def name(self) -> str:
        return "rqc_statevector_synthetic"

    @property
    def pretty_name(self) -> str:
        return "Random Quantum Circuit (RQC) Statevector Synthetic"

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
        return RQCStatevectorBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return RQCStatevectorBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return RQCStatevectorBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return RQCStatevectorBenchmark().motivation

    @property
    def datasets(self) -> list[RQCStatevectorDataset]:
        return [
            # RQCStatevectorDataset(
            #    "single_layer_large",
            #    40,
            #    "Large instance of 40 qubits, 1 layer",
            #    [
            #        ("H", 0),
            #        ("S", 1),
            #        ("Z", 2),
            #        ("Y", 3),
            #        ("Y", 4),
            #        ("T", 5),
            #        ("H", 6),
            #        ("S", 7),
            #        ("X", 8),
            #        ("H", 9),
            #        ("Z", 10),
            #        ("T", 11),
            #        ("S", 12),
            #        ("S", 13),
            #        ("S", 14),
            #        ("S", 15),
            #        ("Z", 16),
            #        ("H", 17),
            #        ("T", 18),
            #        ("Y", 19),
            #        ("Z", 20),
            #        ("Y", 21),
            #        ("X", 22),
            #        ("T", 23),
            #        ("S", 24),
            #        ("Z", 25),
            #        ("Y", 26),
            #        ("S", 27),
            #        ("Z", 28),
            #        ("Y", 29),
            #        ("Y", 30),
            #        ("X", 31),
            #        ("H", 32),
            #        ("Z", 33),
            #        ("T", 34),
            #        ("H", 35),
            #        ("T", 36),
            #        ("S", 37),
            #        ("X", 38),
            #        ("Z", 39),
            #    ],
            # ),
            RQCStatevectorDataset(
                "single_layer_small",
                10,
                "Small instance of 10 qubits, 1 layer",
                [
                    ("H", 0),
                    ("S", 1),
                    ("Z", 2),
                    ("Y", 3),
                    ("Y", 4),
                    ("T", 5),
                    ("H", 6),
                    ("S", 7),
                    ("X", 8),
                    ("H", 9),
                ],
                pretty_name="Single Layer Small",
                suites=["standard", "trace", "train"],
            ),
            RQCStatevectorDataset(
                "single_layer_tiny",
                5,
                "Tiny instance of 5 qubits, 1 layer",
                [
                    ("H", 0),
                    ("S", 1),
                    ("Z", 2),
                    ("Y", 3),
                    ("Y", 4),
                ],
                pretty_name="Single Layer Tiny",
                suites=["test"],
            ),
        ]

    def generate(self, dataset: RQCStatevectorDataset) -> DataInstance:
        return circuit_instance(dataset.nqubits, named_gates(dataset.gate_sequence))


class QGates:
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
    # Two-qubit gates are indexed [new control, new target, old control, old target].
    CX = np.einsum("ac,bd->abcd", np.eye(2), np.eye(2)).astype(np.complex128)
    CX[1, :, 1, :] = X

    @staticmethod
    def RZ(angle: float) -> np.ndarray:
        return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def named_gates(gate_sequence):
    """Normalize hand-written gates to (name, qubits, angle).

    Entries are (name, qubit) for a one-qubit gate, (name, control, target) for
    a two-qubit gate, or ("RZ", qubit, angle).
    """
    gates = []
    for name, *rest in gate_sequence:
        if name == "RZ":
            qubit, angle = rest
            gates.append(("RZ", (qubit,), float(angle)))
        else:
            gates.append((name, tuple(rest), None))
    return gates


def _gate_matrix(name: str, angle: float | None) -> np.ndarray:
    name = name.upper()
    if name == "RZ":
        assert angle is not None, "RZ needs an angle"
        return QGates.RZ(angle)
    return getattr(QGates, name)


def circuit_instance(nqubits, gates, *, sparse_storage=False) -> DataInstance:
    """Build a statevector instance for (name, qubits, angle) gates on |0...0>.

    The kernel receives each distinct one-qubit gate once, stacked in `gates1`,
    and each distinct two-qubit gate in `gates2`; the gate sequence in `meta`
    indexes those stacks. With `sparse_storage`, the initial and final states
    are stored as sparse tensors.
    """
    one_qubit: dict[tuple, int] = {}
    two_qubit: dict[tuple, int] = {}
    stack1: list[np.ndarray] = []
    stack2: list[np.ndarray] = []
    sequence: list[list[int]] = []
    for name, qubits, angle in gates:
        key = (name.upper(), angle)
        if len(qubits) == 1:
            if key not in one_qubit:
                one_qubit[key] = len(stack1)
                stack1.append(_gate_matrix(name, angle))
            sequence.append([one_qubit[key], qubits[0]])
        else:
            if key not in two_qubit:
                two_qubit[key] = len(stack2)
                stack2.append(_gate_matrix(name, angle))
            sequence.append([two_qubit[key], qubits[0], qubits[1]])
    # Every circuit passes both stacks, so empty ones hold an unused gate.
    gates1 = np.stack(stack1 or [QGates.X])
    gates2 = np.stack(stack2 or [QGates.CX])
    state = np.zeros(1 << nqubits, dtype=np.complex128)
    state[0] = 1.0
    reference = simulate_reference(nqubits, state, gates1, gates2, sequence)

    def store(array):
        return (
            from_sparse(sparse.COO.from_numpy(array))
            if sparse_storage
            else (from_numpy(array))
        )

    two_qubit_steps = sum(len(gate) == 3 for gate in sequence)
    bound = statevector_error_bound(len(sequence) - two_qubit_steps, two_qubit_steps)
    return DataInstance(
        inputs=[store(state), from_numpy(gates1), from_numpy(gates2)],
        meta={"nqubits": nqubits, "gate_sequence": sequence},
        ref_outputs=[store(reference)],
        # The framework and the reference each stay within the bound of exact.
        ref_meta={"error_bound": 2 * bound},
    )


def simulate_reference(nqubits, state, gates1, gates2, sequence):
    """Apply the gate sequence with NumPy, contracting as the kernel does."""
    for gate in sequence:
        if len(gate) == 2:
            k, qubit = gate
            view = state.reshape(1 << qubit, 2, 1 << (nqubits - qubit - 1))
            state = np.einsum("jl,ilk->ijk", gates1[k], view)
        else:
            k, first, second = gate
            low, high = sorted((first, second))
            view = state.reshape(
                1 << low, 2, 1 << (high - low - 1), 2, 1 << (nqubits - high - 1)
            )
            spec = "jlpr,ipkrm->ijklm" if first < second else "ljrp,ipkrm->ijklm"
            state = np.einsum(spec, gates2[k], view)
        state = state.reshape(-1)
    return state


def statevector_error_bound(steps, two_qubit_steps=0):
    """Bound the 2-norm distance between exact arithmetic and any float64
    evaluation of `steps` one-qubit and `two_qubit_steps` two-qubit gates.

    Each one-qubit step computes every amplitude from at most two complex
    products and a two-qubit step from at most four, so a step rounds by at
    most gamma * ||G||_F * ||psi||, where ||G||_F is at most sqrt(2) or 2 times
    (1 + 2 eps). Later steps are unitary up to the rounding of their entries,
    so they grow an earlier error by at most 1 + 2 eps, never compounding.
    """
    eps = np.finfo(np.float64).eps
    # Dot products of two and four terms, counted as in the matrix benchmarks.
    gamma2 = summation_error_bound(np, 2 + 1, 1.0, np.complex128)
    gamma4 = summation_error_bound(np, 4 + 1, 1.0, np.complex128)
    total = steps + two_qubit_steps
    per_step = steps * gamma2 * np.sqrt(2) + two_qubit_steps * gamma4 * 2
    return float(per_step * (1 + 2 * eps) ** (total + 1))


def _dense(tensor: BinsparseTensor) -> np.ndarray:
    try:
        return to_numpy(tensor)
    except TypeError:
        return np.asarray(to_sparse(tensor).todense())


class RQCStatevectorTestGenerator(Generator[RQCStatevectorDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rqc_statevector_test"

    @property
    def pretty_name(self) -> str:
        return "Random Quantum Circuit (RQC) Statevector Test"

    @property
    def description(self) -> str:
        return "Small statevector and single-qubit circuit checks."

    @property
    def suites(self) -> list[str]:
        return ["test"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return RQCStatevectorBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return RQCStatevectorBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return RQCStatevectorBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return RQCStatevectorBenchmark().motivation

    @property
    def datasets(self) -> list[RQCStatevectorDataset]:
        return [
            RQCStatevectorDataset(
                "basic",
                10,
                "RQC statevector sanity check.",
                [
                    ("H", 0),
                    ("S", 1),
                    ("Z", 2),
                    ("Y", 3),
                    ("Y", 4),
                    ("T", 5),
                    ("H", 6),
                    ("S", 7),
                    ("X", 8),
                    ("H", 9),
                ],
                pretty_name="Basic",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_h_q0",
                4,
                "H on qubit 0.",
                [("H", 0)],
                pretty_name="H Gate (Qubit 0)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_h_q1",
                4,
                "H on qubit 1.",
                [("H", 1)],
                pretty_name="H Gate (Qubit 1)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_h_q2",
                4,
                "H on qubit 2.",
                [("H", 2)],
                pretty_name="H Gate (Qubit 2)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_h_q3",
                4,
                "H on qubit 3.",
                [("H", 3)],
                pretty_name="H Gate (Qubit 3)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_x_q0",
                4,
                "X on qubit 0.",
                [("X", 0)],
                pretty_name="X Gate (Qubit 0)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_x_q1",
                4,
                "X on qubit 1.",
                [("X", 1)],
                pretty_name="X Gate (Qubit 1)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_x_q2",
                4,
                "X on qubit 2.",
                [("X", 2)],
                pretty_name="X Gate (Qubit 2)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_x_q3",
                4,
                "X on qubit 3.",
                [("X", 3)],
                pretty_name="X Gate (Qubit 3)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_y_q0",
                4,
                "Y on qubit 0.",
                [("Y", 0)],
                pretty_name="Y Gate (Qubit 0)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_y_q1",
                4,
                "Y on qubit 1.",
                [("Y", 1)],
                pretty_name="Y Gate (Qubit 1)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_y_q2",
                4,
                "Y on qubit 2.",
                [("Y", 2)],
                pretty_name="Y Gate (Qubit 2)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_y_q3",
                4,
                "Y on qubit 3.",
                [("Y", 3)],
                pretty_name="Y Gate (Qubit 3)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_z_q0",
                4,
                "Z on qubit 0.",
                [("Z", 0)],
                pretty_name="Z Gate (Qubit 0)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_z_q1",
                4,
                "Z on qubit 1.",
                [("Z", 1)],
                pretty_name="Z Gate (Qubit 1)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_z_q2",
                4,
                "Z on qubit 2.",
                [("Z", 2)],
                pretty_name="Z Gate (Qubit 2)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_z_q3",
                4,
                "Z on qubit 3.",
                [("Z", 3)],
                pretty_name="Z Gate (Qubit 3)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_s_q0",
                4,
                "S on qubit 0.",
                [("S", 0)],
                pretty_name="S Gate (Qubit 0)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_s_q1",
                4,
                "S on qubit 1.",
                [("S", 1)],
                pretty_name="S Gate (Qubit 1)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_s_q2",
                4,
                "S on qubit 2.",
                [("S", 2)],
                pretty_name="S Gate (Qubit 2)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_s_q3",
                4,
                "S on qubit 3.",
                [("S", 3)],
                pretty_name="S Gate (Qubit 3)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_t_q0",
                4,
                "T on qubit 0.",
                [("T", 0)],
                pretty_name="T Gate (Qubit 0)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_t_q1",
                4,
                "T on qubit 1.",
                [("T", 1)],
                pretty_name="T Gate (Qubit 1)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_t_q2",
                4,
                "T on qubit 2.",
                [("T", 2)],
                pretty_name="T Gate (Qubit 2)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "gate_t_q3",
                4,
                "T on qubit 3.",
                [("T", 3)],
                pretty_name="T Gate (Qubit 3)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "h_twice",
                5,
                "Applying H twice returns to the original state.",
                [("H", 2), ("H", 2)],
                pretty_name="H Twice",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "bell",
                3,
                "H then CX prepares a Bell pair on qubits 0 and 1.",
                [("H", 0), ("CX", 0, 1)],
                pretty_name="Bell Pair",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "cx_control_after_target",
                4,
                "CX whose control qubit follows its target.",
                [("SX", 3), ("CX", 3, 1), ("CX", 1, 0)],
                pretty_name="CX (Control After Target)",
                suites=["test"],
            ),
            RQCStatevectorDataset(
                "rz_phases",
                3,
                "RZ rotations, the parameterized gates of transpiled circuits.",
                [("H", 0), ("RZ", 0, 0.7), ("SX", 1), ("RZ", 1, -2.5), ("CX", 0, 2)],
                pretty_name="RZ Phases",
                suites=["test"],
            ),
        ]

    def generate(self, dataset: RQCStatevectorDataset) -> DataInstance:
        return circuit_instance(dataset.nqubits, named_gates(dataset.gate_sequence))


class RQCStatevectorQASMBenchDataset(Dataset):
    """A QASMBench circuit, simulated from |0...0>."""

    def __init__(self, name: str, *, suites: list[str] | None = None):
        self._name = name
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"QASMBench circuit {self._name}, simulated from |0...0>."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class RQCStatevectorQASMBenchGenerator(Generator[RQCStatevectorQASMBenchDataset]):
    @property
    def name(self) -> str:
        return "rqc_statevector_qasmbench"

    @property
    def pretty_name(self) -> str:
        return "Random Quantum Circuit (RQC) Statevector QASMBench"

    @property
    def description(self) -> str:
        return (
            "Transpiled QASMBench circuits whose statevectors stay sparse from"
            " |0...0> to the end, at most 1.6% nonzero at any gate in float64."
            " Initial and final states are stored as sparse tensors."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return RQCStatevectorBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return QASMBENCH_REFERENCES

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "Many quantum algorithms, such as state preparation, reversible"
            " arithmetic, and error correction, keep their statevectors sparse,"
            " which sparse array frameworks can exploit."
        )

    @property
    def datasets(self) -> list[RQCStatevectorQASMBenchDataset]:
        return [
            RQCStatevectorQASMBenchDataset(name, suites=["standard", "trace"])
            for name in (
                "cat_state_n22",
                "ghz_state_n23",
                "multiplier_n15",
                "multiply_n13",
                "qec9xz_n17",
                "qram_n20",
            )
        ]

    def generate(self, dataset: RQCStatevectorQASMBenchDataset) -> DataInstance:
        nqubits, gates = fetch_qasmbench_circuit(dataset.name)
        return circuit_instance(nqubits, gates, sparse_storage=True)


class RQCStatevectorBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "rqc_statevector"

    @property
    def pretty_name(self) -> str:
        return "Random Quantum Circuit (RQC) Statevector"

    @property
    def description(self) -> str:
        return (
            "Simulates a specified quantum circuit on an n-qubit state vector, "
            "using the standard reshape + einsum gate application pattern."
        )

    @property
    def suites(self) -> list[str]:
        return ["quantum", "numeric-tensor-networks"]

    @property
    def concepts(self) -> str:
        return """
        <ccs2012>
        <concept>
        <concept_id>10010583.10010786.10010813.10011726</concept_id>
        <concept_desc>Hardware~Quantum computation</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10010520.10010521.10010542.10010550</concept_id>
        <concept_desc>Computer systems organization~Quantum computing</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        </ccs2012>
        """

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
        return [
            RQCStatevectorTestGenerator(),
            RQCStatevectorSyntheticGenerator(),
            RQCStatevectorQASMBenchGenerator(),
        ]

    def benchmark(self, xp, meta: dict[str, Any], state, gates1, gates2):
        nqubits = meta["nqubits"]

        for gate in meta["gate_sequence"]:
            if len(gate) == 2:
                k, qubit = gate
                left = 1 << qubit
                right = 1 << (nqubits - qubit - 1)
                start_resh = xp.reshape(state, (left, 2, right))
                # gate[new, old] convention => einsum "ijk,lj->ilk"
                new_resh = xp.einsum(
                    "new_resh[i, j, k] += start_resh[i, l, k] * gate[j, l]",
                    start_resh=start_resh,
                    gate=gates1[k],
                )
            else:
                # gate[new control, new target, old control, old target]
                k, control, target = gate
                low, high = sorted((control, target))
                start_resh = xp.reshape(
                    state,
                    (1 << low, 2, 1 << (high - low - 1), 2, 1 << (nqubits - high - 1)),
                )
                if control < target:
                    new_resh = xp.einsum(
                        "new_resh[i, j, k, l, m] += start_resh[i, p, k, r, m]"
                        " * gate[j, l, p, r]",
                        start_resh=start_resh,
                        gate=gates2[k],
                    )
                else:
                    new_resh = xp.einsum(
                        "new_resh[i, j, k, l, m] += start_resh[i, p, k, r, m]"
                        " * gate[l, j, r, p]",
                        start_resh=start_resh,
                        gate=gates2[k],
                    )
            state = xp.reshape(new_resh, (1 << nqubits,))

        return state

    def check(self, param):
        super().check(param)
        output_bin = self._output[0]
        assert output_bin.shape == (1 << self._meta["nqubits"],)
        # A sparse framework may return a sparse state.
        result = _dense(output_bin)
        assert result.dtype == np.complex128
        assert self._ref_outputs is not None, "No reference state; regenerate it"

        distance = np.linalg.norm(result - _dense(self._ref_outputs[0]))
        # Computing the distance itself rounds by at most this factor.
        slack = 1 + summation_error_bound(np, 2 * result.size + 2, 1.0, np.float64)
        assert distance <= self._ref_meta["error_bound"] * slack, (
            f"Statevector for {param.dataset.name} is {distance} from the reference,"
            f" beyond the bound {self._ref_meta['error_bound']}"
        )
