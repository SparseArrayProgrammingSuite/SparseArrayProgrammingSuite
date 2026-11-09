"""Shared QASMBench quantum circuits for circuit simulation benchmarks."""

from typing import Any

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.qasmbench import (
    QASMBENCH_CIRCUITS,
    QASMBENCH_COMMIT,
    QASMBENCH_REPOSITORY_URL,
    download_qasmbench_circuit,
    parse_transpiled_qasm,
    qasmbench_source_url,
)

QASMBENCH_REFERENCES = [
    Ref(
        title=(
            "QASMBench: A Low-Level Quantum Benchmark Suite for NISQ Evaluation"
            " and Simulation"
        ),
        authors=[
            Author("Ang Li"),
            Author("Samuel Stein"),
            Author("Sriram Krishnamoorthy"),
            Author("James Ang"),
        ],
        journal="ACM Transactions on Quantum Computing",
        volume=4,
        number=2,
        year=2023,
        url="https://doi.org/10.1145/3550488",
        doi="10.1145/3550488",
    ),
    Ref(
        title="QASMBench",
        authors=[Author("QASMBench Contributors")],
        url=f"{QASMBENCH_REPOSITORY_URL}/tree/{QASMBENCH_COMMIT}",
        year=2025,
    ),
]


class QASMBenchDataset(Dataset):
    """A transpiled QASMBench circuit, named by its QASMBench ID."""

    def __init__(self, name: str, *, suites: list[str] | None = None):
        if name not in QASMBENCH_CIRCUITS:
            raise ValueError(f"Unknown QASMBench circuit: {name!r}")
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
        return f"QASMBench {QASMBENCH_CIRCUITS[self._name]}-scale circuit."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_url"] = qasmbench_source_url(self._name)
        return data


class QASMBenchCircuitGenerator(Generator[QASMBenchDataset]):
    @property
    def name(self) -> str:
        return "qasmbench_circuit"

    @property
    def pretty_name(self) -> str:
        return "QASMBench Circuits"

    @property
    def description(self) -> str:
        return (
            "Downloads transpiled OpenQASM circuits from QASMBench at a pinned"
            " commit and parses them into gate lists over IBM's native basis"
            " (rz, sx, x, cx)."
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
        return QASMBENCH_REFERENCES

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "QASMBench collects quantum algorithm circuits, such as state"
            " preparation, arithmetic, and error correction, as standard inputs"
            " for evaluating quantum simulators."
        )

    @property
    def datasets(self) -> list[QASMBenchDataset]:
        return [QASMBenchDataset(name) for name in QASMBENCH_CIRCUITS]

    def generate(self, dataset: QASMBenchDataset) -> DataInstance:
        path = download_qasmbench_circuit(dataset.name)
        nqubits, gates = parse_transpiled_qasm(path.read_text(encoding="utf-8"))
        # No local path: it would point into whichever machine cached this.
        return DataInstance(
            inputs=[],
            meta={
                "nqubits": nqubits,
                "gates": [[name, list(qubits), angle] for name, qubits, angle in gates],
                "source_url": qasmbench_source_url(dataset.name),
            },
        )


class QASMBenchCircuitShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return QASMBenchCircuitGenerator()


def qasmbench_circuit(name: str) -> QASMBenchDataset:
    """Look up a declared QASMBench circuit."""
    for dataset in QASMBenchCircuitGenerator().datasets:
        if dataset.name == name:
            return dataset
    raise ValueError(f"QASMBench circuit {name!r} is not declared")


def fetch_qasmbench_circuit(name: str) -> tuple[int, list[tuple]]:
    """Read a declared circuit from prepared storage as (nqubits, gates)."""
    meta = QASMBenchCircuitGenerator().cached_generate(qasmbench_circuit(name)).meta
    return meta["nqubits"], [
        (gate, tuple(qubits), angle) for gate, qubits, angle in meta["gates"]
    ]
