"""Shared prepared SNAP graphs for the graph benchmarks."""

from saps.benchmark import (
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.snap import download_snap_dataset


class SNAPDataset(Dataset):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self.name

    @property
    def description(self) -> str:
        return f"SNAP graph {self.name.removeprefix('snap-')} with remapped node IDs."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


# Complete source inventory; add a graph here before using it in a consumer.
_GRAPHS = [
    SNAPDataset("snap-ca-GrQc"),
    SNAPDataset("snap-email-Eu-core"),
    SNAPDataset("snap-email-Eu-core-temporal-Dept1"),
    SNAPDataset("snap-email-Eu-core-temporal-Dept2"),
    SNAPDataset("snap-email-Eu-core-temporal-Dept3"),
    SNAPDataset("snap-email-Eu-core-temporal-Dept4"),
    SNAPDataset("snap-facebook_combined"),
    SNAPDataset("snap-p2p-Gnutella04"),
]


class SNAPGraphGenerator(Generator[SNAPDataset]):
    @property
    def name(self) -> str:
        return "snap_graph"

    @property
    def pretty_name(self) -> str:
        return "Stanford Network Analysis Project Graphs"

    @property
    def description(self) -> str:
        return "Shared sparse SNAP adjacency matrices and original node IDs."

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
        return "This shell generator was written with assistance from OpenAI Codex."

    @property
    def motivation(self) -> str:
        return "Prepare each SNAP graph once for reuse across graph benchmarks."

    @property
    def datasets(self) -> list[SNAPDataset]:
        return _GRAPHS

    def generate(self, dataset: SNAPDataset) -> DataInstance:
        inputs, meta = download_snap_dataset(
            dataset.name, data_dir=self.backend.cache_dir / "snap"
        )
        return DataInstance(inputs=inputs, meta=meta)


class SNAPGraphBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return SNAPGraphGenerator()


def fetch_snap_graph(name: str) -> DataInstance:
    """Read a declared graph's adjacency and original node IDs from prepared storage."""
    generator = SNAPGraphGenerator()
    dataset = next((d for d in generator.datasets if d.name == name), None)
    if dataset is None:
        raise ValueError(
            f"Dataset {name!r} is not listed in SNAPGraphGenerator.datasets. "
            "Add it to the shell dataset list before using it."
        )
    return generator.cached_generate(dataset)
