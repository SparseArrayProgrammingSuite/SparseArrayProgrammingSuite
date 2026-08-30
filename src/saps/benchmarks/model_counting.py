from typing import Any

import numpy as np

from saps.benchmark import (
    Benchmark,
    BinsparseTensor,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)


def parse_dimacs(text):
    lines = [line.strip() for line in text.strip().split("\n")]
    cleaned = [line for line in lines if not line.startswith("c") and line]

    clauses = []
    num_vars = 0
    num_clauses = 0
    rest = []

    for i, line in enumerate(cleaned):
        if line.startswith("p cnf"):
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])

            rest = " ".join(cleaned[i + 1 :]).split()
            break

    clauses = []
    current_clause = []
    idx = 0

    while len(clauses) < num_clauses and idx < len(rest):
        val = int(rest[idx])

        if val == 0:
            clauses.append(current_clause)
            current_clause = []
        else:
            current_clause.append(val)

        idx += 1

    return num_vars, clauses


def clauses_to_einsum(clauses):
    if len(clauses) == 0:
        return None

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

        data_list: list[BinsparseTensor] = [
            BinsparseTensor.from_numpy(np.array([0, 1]))
        ]

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
            ref_outputs=[BinsparseTensor.from_numpy(np.array(dataset.expected))],
        )


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
        return [MCGenerator()]

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
        result = int(
            self._output[0].data["values"].reshape(self._output[0].data["shape"])
        )
        expected = int(
            self._ref_outputs[0]
            .data["values"]
            .reshape(self._ref_outputs[0].data["shape"])
        )
        assert result == expected, (
            f"Test '{param.dataset.name}' failed: expected {expected}, got {result}"
        )
