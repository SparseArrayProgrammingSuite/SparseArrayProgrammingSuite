from functools import reduce
from typing import Any

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import Benchmark, Contributor, DataInstance, Dataset, Generator, Ref
from saps.util.error_bounds import summation_error_bound


class RQCStatevectorDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        nqubits: int,
        description: str,
        gate_sequence: list[tuple[str, int]],
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
        nqubits = dataset.nqubits
        dim = 1 << nqubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0j  # |000...0
        state_bin = from_numpy(state)
        return DataInstance(
            inputs=[
                state_bin,
                from_numpy(QGates.H),
                from_numpy(QGates.X),
                from_numpy(QGates.Y),
                from_numpy(QGates.Z),
                from_numpy(QGates.S),
                from_numpy(QGates.T),
            ],
            meta={"nqubits": nqubits, "gate_sequence": dataset.gate_sequence},
            ref_outputs=[
                from_numpy(product_state_reference(nqubits, dataset.gate_sequence))
            ],
            # The framework and the reference each stay within a bound of exact.
            ref_meta={
                "error_bound": statevector_error_bound(len(dataset.gate_sequence))
                + statevector_error_bound(len(dataset.gate_sequence) + nqubits - 1)
            },
        )


class QGates:
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def _zero_state(nqubits):
    state = np.zeros(1 << nqubits, dtype=np.complex128)
    state[0] = 1.0 + 0j
    return state


def product_state_reference(nqubits, gate_sequence):
    """The circuit's final state, built one qubit at a time.

    Every gate acts on a single qubit, so the state stays a Kronecker product of
    per-qubit states, with qubit 0 the most significant.
    """
    qubits = [np.array([1, 0], dtype=np.complex128) for _ in range(nqubits)]
    for gate_name, qubit in gate_sequence:
        qubits[qubit] = getattr(QGates, gate_name) @ qubits[qubit]
    return reduce(np.kron, qubits)


def statevector_error_bound(steps):
    """Bound the 2-norm distance between exact arithmetic and any float64
    evaluation of `steps` single-qubit gates or Kronecker products on |0...0>.

    Each step computes every amplitude from at most two complex products, so it
    rounds by at most gamma * ||G||_F * ||psi||, where ||G||_F is at most
    sqrt(2) (1 + 2 eps). Later steps are unitary up to the rounding of their
    entries, so they grow an earlier error by at most 1 + 2 eps, never compounding.
    """
    eps = np.finfo(np.float64).eps
    # A dot product of two terms, counted as in the matrix benchmarks.
    gamma = summation_error_bound(np, 2 + 1, 1.0, np.complex128)
    return float(steps * gamma * np.sqrt(2) * (1 + 2 * eps) ** (steps + 1))


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
        ]

    def generate(self, dataset: RQCStatevectorDataset) -> DataInstance:
        nqubits = dataset.nqubits
        return DataInstance(
            inputs=[
                from_numpy(_zero_state(dataset.nqubits)),
                from_numpy(QGates.H),
                from_numpy(QGates.X),
                from_numpy(QGates.Y),
                from_numpy(QGates.Z),
                from_numpy(QGates.S),
                from_numpy(QGates.T),
            ],
            meta={
                "nqubits": nqubits,
                "gate_sequence": dataset.gate_sequence,
            },
            ref_outputs=[
                from_numpy(product_state_reference(nqubits, dataset.gate_sequence))
            ],
            # The framework and the reference each stay within a bound of exact.
            ref_meta={
                "error_bound": statevector_error_bound(len(dataset.gate_sequence))
                + statevector_error_bound(len(dataset.gate_sequence) + nqubits - 1)
            },
        )


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
        return ["standard-quantum"]

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
        return [RQCStatevectorTestGenerator(), RQCStatevectorSyntheticGenerator()]

    def benchmark(self, xp, meta: dict[str, Any], state, H, X, Y, Z, S, T):
        nqubits = meta["nqubits"]

        for gate_name, qubit in meta["gate_sequence"]:
            if gate_name == "H":
                gate = H
            elif gate_name == "X":
                gate = X
            elif gate_name == "Y":
                gate = Y
            elif gate_name == "Z":
                gate = Z
            elif gate_name == "S":
                gate = S
            elif gate_name == "T":
                gate = T
            else:
                raise ValueError(f"Unknown quantum gate: {gate_name}")
            left = 1 << qubit
            right = 1 << (nqubits - qubit - 1)
            start_resh = xp.reshape(state, (left, 2, right))
            # gate[new, old] convention => einsum "ijk,lj->ilk"
            new_resh = xp.einsum(
                "new_resh[i, j, k] += start_resh[i, l, k] * gate[j, l]",
                start_resh=start_resh,
                gate=gate,
            )
            state = xp.reshape(new_resh, state.shape)

        return state

    def check(self, param):
        super().check(param)
        output_bin = self._output[0]
        assert output_bin.shape == (1 << self._meta["nqubits"],)
        result = to_numpy(output_bin)
        assert result.dtype == np.complex128
        assert self._ref_outputs is not None, "No reference state; regenerate it"

        distance = np.linalg.norm(result - to_numpy(self._ref_outputs[0]))
        # Computing the distance itself rounds by at most this factor.
        slack = 1 + summation_error_bound(np, 2 * result.size + 2, 1.0, np.float64)
        assert distance <= self._ref_meta["error_bound"] * slack, (
            f"Statevector for {param.dataset.name} is {distance} from the reference,"
            f" beyond the bound {self._ref_meta['error_bound']}"
        )
