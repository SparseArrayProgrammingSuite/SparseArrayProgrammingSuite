from pathlib import Path
from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.mccomp import (
    MCCOMP_REPOSITORY_URL,
    MCCOMP_TRACKS,
    download_mccomp_instance,
    list_mccomp_instances,
    mccomp_source_url,
    normalize_mccomp_source_path,
    parse_dimacs,
)


def clauses_to_einsum(clauses):
    if len(clauses) == 0:
        return None
    if any(len(clause) == 0 for clause in clauses):
        return "s[] += False"

    clause_strings = []
    for clause in clauses:
        literal_strings = []
        for val in clause:
            var_idx = abs(val)
            var_name = f"B[v{var_idx}]"

            if val < 0:
                literal_strings.append(f"not {var_name}")
            else:
                literal_strings.append(var_name)

        clause_str = "(" + " or ".join(literal_strings) + ")"
        clause_strings.append(clause_str)

    full_str = " and ".join(clause_strings)

    return f"s[] += {full_str}"


class MCDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str,
        description: str,
        suites: list[str],
        cnf_text: str,
        expected: int,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.cnf_text = cnf_text
        self.expected = expected

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class MCGenerator(Generator[MCDataset]):
    @property
    def name(self) -> str:
        return "mc_generator"

    @property
    def pretty_name(self) -> str:
        return "Model Counting Generator"

    @property
    def description(self) -> str:
        return (
            "Parses standard DIMACS CNF test strings into sparse arrays for model"
            " counting."
        )

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Richard Wan", "rwan41@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return """No generative AI was used to write the benchmark function itself.
        Generative AI was used to debug code. This statement was written by hand."""

    @property
    def motivation(self) -> str:
        return "Uses a predefined set of formulas to verify correctness."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MCDataset]:
        return [
            MCDataset(
                name="test_1",
                pretty_name="Test 1: Standard SAT",
                description="3 variables, 2 clauses",
                suites=["test", "trace"],
                cnf_text="""
                    p cnf 3 2
                    1 -3 0
                    2 3 -1 0
                """,
                expected=5,
            ),
            MCDataset(
                name="test_2",
                pretty_name="Test 2: Contradiction",
                description="V1 and not V1",
                suites=["test", "trace"],
                cnf_text="""
                    c contradiction
                    p cnf 1 2
                    1 0
                    -1 0
                """,
                expected=0,
            ),
            MCDataset(
                name="test_3",
                pretty_name="Test 3: Single Solution",
                description="Forces all 3 variables to be true",
                suites=["test", "trace"],
                cnf_text="""
                    c single_solution
                    p cnf 3 3
                    1 0
                    2 0
                    3 0
                """,
                expected=1,
            ),
            MCDataset(
                name="test_4",
                pretty_name="Test 4: Empty Formula",
                description="No clauses, 2 variables",
                suites=["test", "trace"],
                cnf_text="""
                    c empty_formula
                    p cnf 2 0
                """,
                expected=4,
            ),
        ]

    def generate(self, dataset: MCDataset):
        num_vars, clauses = parse_dimacs(dataset.cnf_text)
        expr = clauses_to_einsum(clauses)

        data_list: list[BinsparseTensor] = [from_numpy(np.array([0, 1]))]

        default_total = 2**num_vars

        meta = {
            "expr": expr,
            "num_vars": num_vars,
            "expected_result": dataset.expected,
            "default_total": default_total,
        }

        return DataInstance(
            inputs=data_list,
            meta=meta,
            ref_outputs=[from_numpy(np.array(dataset.expected))],
        )


class MCCompDataset(Dataset):
    def __init__(self, source_path: str, *, suites: list[str] | None = None):
        self.source_path = source_path
        self.track = source_path.split("/", 1)[0]
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self.source_path.removesuffix(".cnf").replace("/", "_").lower()

    @property
    def pretty_name(self) -> str:
        return f"MCComp {self.source_path.removesuffix('.cnf')}"

    @property
    def description(self) -> str:
        track_description = MCCOMP_TRACKS.get(self.track, ("", "", "model counting"))[2]
        return f"Model Counting Competition {track_description} instance."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data.update(
            {
                "source_path": self.source_path,
                "track": self.track,
                "source_url": mccomp_source_url(self.source_path),
            }
        )
        return data


class MCCompGenerator(Generator[MCCompDataset]):
    @property
    def name(self) -> str:
        return "mccomp"

    @property
    def pretty_name(self) -> str:
        return "Model Counting Competition Test Instances"

    @property
    def description(self) -> str:
        return (
            "Downloads and parses DIMACS CNF instances from the Model Counting "
            "Competition test-instance repository."
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
        return [
            Ref(
                title="Model Counting Competition test instances",
                authors=[],
                url=MCCOMP_REPOSITORY_URL,
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "Model counting competition instances provide standard CNF inputs "
            "for evaluating exact, projected, weighted, and algebraic counting "
            "workloads."
        )

    @property
    def datasets(self) -> list[MCCompDataset]:
        return [MCCompDataset(source_path) for source_path in list_mccomp_instances()]

    def generate(self, dataset: MCCompDataset):
        source_path = normalize_mccomp_source_path(dataset.source_path)
        local_path = download_mccomp_instance(source_path)
        return DataInstance(
            inputs=[],
            meta={
                "source_repository": MCCOMP_REPOSITORY_URL,
                "source_path": source_path,
                "source_url": mccomp_source_url(source_path),
                "local_path": str(local_path),
                "track": dataset.track,
            },
        )


class MCCompBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return MCCompGenerator()


class MCCompMCGenerator(Generator[MCCompDataset]):
    @property
    def name(self) -> str:
        return "mccomp_mc"

    @property
    def pretty_name(self) -> str:
        return "Model Counting Competition Track1 Generator"

    @property
    def description(self) -> str:
        return "Loads exact model counting CNF instances from MCComp Track1."

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
        return MCCompGenerator().references

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "Track1 instances have exact integer model counts, so they can be "
            "run by the existing unweighted model-counting einsum benchmark."
        )

    @property
    def datasets(self) -> list[MCCompDataset]:
        return [
            MCCompDataset(source_path, suites=["standard"])
            for source_path in list_mccomp_instances("Track1_MC")
        ]

    def generate(self, dataset: MCCompDataset):
        source = fetch_mccomp_instance(dataset.source_path)
        cnf_text = Path(source.meta["local_path"]).read_text(encoding="utf-8")
        num_vars, clauses = parse_dimacs(cnf_text)
        expr = clauses_to_einsum(clauses)
        exact_type, exact_value = parse_mccomp_exact(cnf_text)
        if exact_type != "int" or exact_value is None:
            raise ValueError(
                f"MCComp Track1 instance lacks an integer answer: {dataset.source_path}"
            )

        return DataInstance(
            inputs=[from_numpy(np.asarray([0, 1], dtype=np.int64))],
            meta={
                "expr": expr,
                "num_vars": num_vars,
                "expected_result": int(exact_value),
                "default_total": 2**num_vars,
                "source_generator": MCCompGenerator().name,
                "source_path": dataset.source_path,
                "source_problem_type": parse_mccomp_problem_type(cnf_text),
                "source_num_clauses": len(clauses),
            },
            ref_outputs=[from_numpy(np.asarray(int(exact_value), dtype=np.int64))],
        )


def fetch_mccomp_instance(source_name: str) -> DataInstance:
    """Fetch and cache a parsed MC competition source instance."""
    source_key = source_name.removesuffix(".cnf")
    matches = [
        path
        for path in list_mccomp_instances()
        if path.removesuffix(".cnf") == source_key
        or path.rsplit("/", 1)[-1].removesuffix(".cnf") == source_key
    ]
    if len(matches) != 1:
        message = f"Unknown or ambiguous MC competition instance: {source_name!r}"
        raise ValueError(message)

    source_path = matches[0]
    raw_generator = MCCompGenerator()
    raw_dataset = next(
        dataset
        for dataset in raw_generator.datasets
        if dataset.source_path == source_path
    )
    return raw_generator.cached_generate(raw_dataset)


def parse_mccomp_problem_type(cnf_text: str) -> str | None:
    for raw_line in cnf_text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 3 and parts[:2] == ["c", "t"]:
            return parts[2]
    return None


def parse_mccomp_exact(cnf_text: str) -> tuple[str | None, str | None]:
    for raw_line in cnf_text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 6 and parts[0:4] == ["c", "c", "s", "exact"]:
            return parts[5], " ".join(parts[6:]) if len(parts) > 6 else None
    return None, None


class ModelCounting(Benchmark):
    @property
    def tag(self):
        return "model_counting"

    @property
    def name(self):
        return "Model Counting using einsum"

    @property
    def pretty_name(self):
        return "Model Counting using einsum"

    @property
    def description(self):
        return "Benchmarks Model Counting Algorithm using einsum operations."

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10010583.10010717.10010721.10010727</concept_id>
<concept_desc>Hardware~Theorem proving and SAT solving</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003648.10003662</concept_id>
<concept_desc>Mathematics of computing~Probabilistic inference problems</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010583.10010717.10010721.10003791</concept_id>
<concept_desc>Hardware~Model checking</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self):
        return [Contributor("Richard Wan", "rwan41@gatech.edu")]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self) -> str:
        return """No generative AI was used to write the benchmark function itself.
            Generative AI was used to debug code. This statement was written by hand."""

    @property
    def motivation(self):
        return (
            "Model Counting is used to determine the total number of satisfying"
            " assignments for a SAT problem."
        )

    @property
    def generators(self) -> list[Generator[Any]]:
        return [MCGenerator(), MCCompMCGenerator()]

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]) -> list[Any]:
        expr = meta["expr"]

        if expr is None:
            return [xp.array(meta["default_total"], dtype=np.int64)]

        result = xp.einsum(expr, B=data[0])

        return [result]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        result = int(to_numpy(self._output[0]))
        expected = int(to_numpy(self._ref_outputs[0]))
        assert result == expected, (
            f"Test '{param.dataset.name}' failed: expected {expected}, got {result}"
        )
