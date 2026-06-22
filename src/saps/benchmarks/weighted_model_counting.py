import textwrap

import numpy as np

from pyparsing import Any

import saps
from saps.benchmark import (
    Benchmark,
    BinsparseFormat,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)

xp = saps.xp

# TODO add generator for https://github.com/arijitsh/mccomp-test-instances/tree/main/Track4_PWMC


def parse_weight(s):
    s = s.strip()
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def parse_format(text):
    lines = [line.strip() for line in text.strip().split("\n")]

    weights = {}
    weight_lines = [line for line in lines if line.startswith("c p weight")]
    for line in weight_lines:
        parts = line.split()
        literal = int(parts[3])
        weights[literal] = parse_weight(parts[4])

    cleaned = [line for line in lines if not line.startswith("c") and line]
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

    for lit in list(weights.keys()):
        if -lit not in weights:
            w = weights[lit]
            if 0 < w < 1:
                weights[-lit] = 1 - w

    for lit in range(1, num_vars + 1):
        if lit not in weights:
            weights[lit] = 0.5
            weights[-lit] = 0.5

    return num_vars, clauses, weights


def clauses_to_einsum(clauses, num_vars):
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

    mask_str = "(" + " and ".join(clause_strings) + ")"
    weight_strings = [f"W{i}[v{i}]" for i in range(1, num_vars + 1)]
    weights_str = " * ".join(weight_strings)

    return f"s[] += {mask_str} * {weights_str}"


class WMCDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str,
        description: str,
        suites: list[str],
        cnf_text: str,
        expected: float,
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


# BEGIN COPIED TEST FILE: tests/test_weighted_model_counting.py
# import numpy as np
#
# import saps.benchmarks.weighted_model_counting as wmc
# from frameworks.saps_numpy import NumpyFramework
#
#
# def test_weighted_model_counting_datasets():
#     xp = NumpyFramework()
#     wmc.xp = xp
#
#     generator = wmc.WMCGenerator()
#     benchmark = wmc.WeightedModelCounting()
#
#     for dataset in generator.datasets:
#         problem = generator.generate(dataset)
#         raw_matrices = problem.inputs
#         meta = problem.meta
#
#         input_arrays = [xp.from_binsparse(m) for m in raw_matrices]
#
#         results = benchmark.benchmark(input_arrays, meta)
#
#         res = float(results[0])
#         expected = meta["expected_result"]
#
#         msg = f"Test '{dataset.name}' failed: Expected {expected}, got {res}"
#         assert np.isclose(res, expected, rtol=10e-8), msg
# END COPIED TEST FILE: tests/test_weighted_model_counting.py

class WMCGenerator(Generator[WMCDataset]):
    @property
    def name(self) -> str:
        return "wmc_generator"

    @property
    def pretty_name(self) -> str:
        return "Weighted Model Counting Generator"

    @property
    def description(self) -> str:
        return "Parses DIMACS CNF test strings into sparse arrays."

    @property
    def suites(self) -> list[str]:
        return []

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
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Uses a predefined set of formulas to verify correctness."

    @property
    def datasets(self) -> list[WMCDataset]:
        return [
            WMCDataset(
                name="test_1",
                pretty_name="Test 1: Satisfiable",
                description="(V1 or V2) and (not V1 or V2)",
                suites=[],
                cnf_text=textwrap.dedent(
                    """\
                    c (V1 or V2) and (not V1 or V2)
                    p cnf 2 2
                    c p weight 1 0.6
                    c p weight -1 0.4
                    c p weight 2 0.8
                    c p weight -2 0.2
                    1 2 0
                    -1 2 0
                """
                ),
                expected=0.8,
            ),
            WMCDataset(
                name="test_2",
                pretty_name="Test 2: Unsatisfiable",
                description="V1 and not V1",
                suites=[],
                cnf_text=textwrap.dedent(
                    """\
                    c V1 and not V1 (unsatisfiable)
                    p cnf 1 2
                    c p weight 1 0.7
                    c p weight -1 0.3
                    1 0
                    -1 0
                """
                ),
                expected=0.0,
            ),
            WMCDataset(
                name="test_3",
                pretty_name="Test 3: No Clauses",
                description="p cnf 2 0",
                suites=[],
                cnf_text=textwrap.dedent(
                    """\
                    c no clauses
                    p cnf 2 0
                    c p weight 1 0.7
                    c p weight -1 0.3
                    c p weight 2 0.9
                    c p weight -2 0.1
                """
                ),
                expected=1.0,
            ),
            WMCDataset(
                name="test_4",
                pretty_name="Test 4: Default Weights",
                description="V1 or V2 (default weights)",
                suites=[],
                cnf_text=textwrap.dedent(
                    """\
                    c V1 or V2 (default weights)
                    p cnf 2 1
                    1 2 0
                """
                ),
                expected=0.75,
            ),
            WMCDataset(
                name="test_5",
                pretty_name="Test 5: 3-Var Formula",
                description="(V1 or V2) and (not V2 or V3)",
                suites=[],
                cnf_text=textwrap.dedent(
                    """\
                    c (V1 or V2) and (not V2 or V3)
                    p cnf 3 2
                    c p weight 1 0.2
                    c p weight -1 0.8
                    c p weight 2 0.6
                    c p weight -2 0.4
                    c p weight 3 0.9
                    c p weight -3 0.1
                    1 2 0
                    -2 3 0
                """
                ),
                expected=0.62,
            ),
            WMCDataset(
                name="test_6",
                pretty_name="Test 6: 20-Var Formula",
                description="""
                    (V1 or not V2 or V3) and (not V1 or V4 or V5) and
                    (V2 or not V5 or V6) and (not V3 or V7 or not V8)
                    and (V4 or not V9 or V10) and (not V6 or V11 or V12)
                    and (V8 or not V13 or V14) and (V10 or V15 or not V16)
                    and (not V12 or not V17 or V18) and (V14 or V19 or not V20)
                    and (not V15 or V16 or V20) and (V17 or not V18 or V19)
                """,
                suites=[],
                cnf_text=textwrap.dedent(
                    """\
                    c 20 var WMC problem
                    p cnf 20 12
                    c p weight 1 0.6
                    c p weight 2 0.3
                    c p weight 3 0.8
                    c p weight 4 0.5
                    c p weight 5 0.7
                    c p weight 6 0.2
                    c p weight 7 0.9
                    c p weight 8 0.4
                    c p weight 9 0.6
                    c p weight 10 0.1
                    c p weight 11 0.8
                    c p weight 12 0.5
                    c p weight 13 0.3
                    c p weight 14 0.7
                    c p weight 15 0.4
                    c p weight 16 0.9
                    c p weight 17 0.2
                    c p weight 18 0.6
                    c p weight 19 0.5
                    c p weight 20 0.8
                    1 -2 3 0
                    -1 4 5 0
                    2 -5 6 0
                    -3 7 -8 0
                    4 -9 10 0
                    -6 11 12 0
                    8 -13 14 0
                    10 15 -16 0
                    -12 -17 18 0
                    14 19 -20 0
                    -15 16 20 0
                    17 -18 19 0
                """
                ),
                expected=0.1180836534707074,
            ),
        ]

    def generate(self, dataset: WMCDataset):
        num_vars, clauses, weights = parse_format(dataset.cnf_text)
        expr = clauses_to_einsum(clauses, num_vars)

        data_list: list[BinsparseFormat] = [xp.to_binsparse(xp.array([0, 1]))]

        data_list.extend(
            xp.to_binsparse(xp.array([weights[-i], weights[i]]))
            for i in range(1, num_vars + 1)
        )

        default_total = 1.0
        if expr is None:
            for i in range(1, num_vars + 1):
                default_total *= weights[i] + weights[-i]

        meta = {
            "expr": expr,
            "num_vars": num_vars,
            "expected_result": dataset.expected,
            "default_total": default_total,
        }

        return DataInstance(inputs=data_list, meta=meta)


class WeightedModelCounting(Benchmark):
    @property
    def tag(self):
        return "weighted_model_counting"

    @property
    def name(self):
        return "Weighted Model Counting using einsum"

    @property
    def pretty_name(self):
        return "Weighted Model Counting using einsum"

    @property
    def description(self):
        return "Benchmarks Weighted Model Counting Algorithm using einsum operations."

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return [Contributor("Richard Wan", "rwan41@gatech.edu")]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self):
        return """Weighted Model Counting extends model counting
                by assigning specific weights to the variables."""

    @property
    def generators(self) -> list[Generator[Any]]:
        return [WMCGenerator()]

    def benchmark(self, data: list[Any], meta: dict[str, Any]) -> list[Any]:
        expr = meta["expr"]

        if expr is None:
            return [xp.array(meta["default_total"], dtype=np.float64)]

        num_vars = meta["num_vars"]

        args = {"B": data[0]}
        for i in range(1, num_vars + 1):
            args[f"W{i}"] = data[i]

        result = xp.einsum(expr, **args)

        return [result]
