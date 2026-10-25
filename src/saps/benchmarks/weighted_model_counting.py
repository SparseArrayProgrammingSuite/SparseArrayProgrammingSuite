import textwrap
from abc import ABC
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
)
from saps.codegen import (
    constant_function_source,
    define_function,
    einsums_function_source,
)
from saps.downloaders.mccomp import (
    MCCOMP_REPOSITORY_URL,
    MCCOMP_TRACKS,
    download_mccomp_instance,
    mccomp_source_url,
    normalize_mccomp_source_path,
    parse_dimacs,
)
from saps.util.error_bounds import summation_error_bound


def parse_weight(s):
    s = s.strip()
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def parse_format(text):
    lines = [line.strip() for line in text.strip().split("\n")]
    num_vars, clauses = parse_dimacs(text)

    weights = {}
    weight_lines = [line for line in lines if line.startswith("c p weight")]
    for line in weight_lines:
        parts = line.split()
        literal = int(parts[3])
        weights[literal] = parse_weight(parts[4])

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


def parse_shown(text, num_vars):
    """The variables a projected count sums over: those on "c p show" lines, or
    every variable when the instance has none."""
    shown = None
    for line in text.splitlines():
        parts = line.split()
        if parts[:3] == ["c", "p", "show"]:
            shown = (shown or []) + [int(part) for part in parts[3:] if part != "0"]
    if shown is None:
        return list(range(1, num_vars + 1))
    return list(dict.fromkeys(shown))


def clauses_to_einsums(clauses, num_vars, shown):
    """Einsums for the weighted count of clauses' models projected onto shown.

    An or= over the other variables finds which shown assignments extend to a
    model; a += then sums those assignments' weights.
    """
    if len(clauses) == 0:
        return None
    if any(len(clause) == 0 for clause in clauses):
        return ["s[] += False"]

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
    # S is indexed by every shown variable, so each must appear in the formula.
    mentioned = {abs(literal) for clause in clauses for literal in clause}
    clause_strings.extend(
        f"(B[v{var}] or not B[v{var}])" for var in shown if var not in mentioned
    )

    mask_str = "(" + " and ".join(clause_strings) + ")"
    indices = ",".join(f"v{var}" for var in shown)
    terms = [f"S[{indices}]", *(f"W{var}[v{var}]" for var in shown)]
    if not shown:
        terms.append("1.0")
    return [f"S[{indices}] or= {mask_str}", f"s[] += {' * '.join(terms)}"]


# Brute force enumerates 2**num_vars assignments.
_MAX_REFERENCE_VARS = 24


def _projected_terms(num_vars, clauses, weights, shown):
    """The weight of each projected model: for each assignment of the shown
    variables that extends to a model, the product of their weights."""
    if num_vars > _MAX_REFERENCE_VARS:
        raise ValueError(
            f"Can't brute-force a reference count over {num_vars} variables"
        )
    num_rows = 1 << num_vars
    bit_index = np.arange(num_rows, dtype=np.int64)[:, None]
    positions = np.arange(num_vars, dtype=np.int64)[None, :]
    assignments = ((bit_index >> positions) & 1).astype(bool)

    satisfied = np.ones(num_rows, dtype=bool)
    for clause in clauses:
        if not clause:
            satisfied[:] = False
            break
        clause_mask = np.zeros(num_rows, dtype=bool)
        for lit in clause:
            column = assignments[:, abs(lit) - 1]
            clause_mask |= column if lit > 0 else ~column
        satisfied &= clause_mask

    columns = [var - 1 for var in shown]
    models = np.unique(assignments[satisfied][:, columns], axis=0)
    products = np.ones(len(models), dtype=np.float64)
    for j, var in enumerate(shown):
        products *= np.where(
            models[:, j], np.float64(weights[var]), np.float64(weights[-var])
        )
    return products


def referenceWeightedModelCount(num_vars, clauses, weights, shown=None):
    """The weighted model count of clauses, projected onto shown (by default
    every variable), by brute force."""
    shown = list(range(1, num_vars + 1)) if shown is None else shown
    return np.float64(np.sum(_projected_terms(num_vars, clauses, weights, shown)))


def _weighted_problem(cnf_text, expected=None):
    """The inputs, metadata and reference of a projected weighted count.

    The count sums one product of the k shown variables' weights per projected
    model. Assignments that aren't models contribute exact zeros, which don't
    round, so whatever the order each term rounds at most
    (models - 1) + (k - 1) times.
    """
    num_vars, clauses, weights = parse_format(cnf_text)
    shown = parse_shown(cnf_text, num_vars)
    terms = _projected_terms(num_vars, clauses, weights, shown)
    reference = float(np.sum(terms))
    steps = max(len(terms) - 1, 0) + max(len(shown) - 1, 0)
    bound = summation_error_bound(
        np, steps + 1, float(np.sum(np.abs(terms))), np.float64
    )
    inputs: list[BinsparseTensor] = [from_numpy(np.asarray([0, 1], dtype=np.int64))]
    inputs.extend(
        from_numpy(np.asarray([weights[-var], weights[var]], dtype=np.float64))
        for var in shown
    )
    meta = {
        "exprs": clauses_to_einsums(clauses, num_vars, shown),
        "num_vars": num_vars,
        "num_clauses": len(clauses),
        "shown": shown,
        "default_total": reference,
    }
    return DataInstance(
        inputs=inputs,
        meta=meta,
        ref_outputs=[
            from_numpy(np.asarray(reference if expected is None else expected))
        ],
        ref_meta={"error_bound": float(bound), "reference_count": reference},
    )


class WeightedModelCountingTestDataset(Dataset):
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


class WeightedModelCountingMCCompDataset(Dataset):
    def __init__(self, name: str, *, suites: list[str] | None = None):
        self._name = name
        self.source_path = f"{name}.cnf"
        self.track = name.split("/", 1)[0]
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        track_description = MCCOMP_TRACKS.get(
            self.track, ("", "", "weighted model counting")
        )[2]
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


class _WMCFunctionGenerator(Generator[Any], ABC):
    """Builds each formula's benchmark function, with parameters ``B, W1, ..., Wn``."""

    def generate_benchmark_function(self, dataset, problem, benchmark):
        meta = problem.meta
        params = ["B", *(f"W{var}" for var in meta["shown"])]
        if meta["exprs"] is None:
            source = constant_function_source(meta["default_total"], "float64", params)
        else:
            source = einsums_function_source(meta["exprs"], params)
        return define_function(
            source, f"<saps-generated {self.name}.{dataset.name}>", {"np": np}
        )


class WeightedModelCountingTestGenerator(
    _WMCFunctionGenerator, Generator[WeightedModelCountingTestDataset]
):
    @property
    def name(self) -> str:
        return "weighted_model_counting_test"

    @property
    def pretty_name(self) -> str:
        return "Weighted Model Counting Test"

    @property
    def description(self) -> str:
        return "Parses DIMACS CNF test strings into sparse arrays."

    @property
    def suites(self) -> list[str]:
        return ["test"]

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
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[WeightedModelCountingTestDataset]:
        return [
            WeightedModelCountingTestDataset(
                name="satisfiable",
                pretty_name="Satisfiable",
                description="(V1 or V2) and (not V1 or V2)",
                suites=["test"],
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
            WeightedModelCountingTestDataset(
                name="unsatisfiable",
                pretty_name="Unsatisfiable",
                description="V1 and not V1",
                suites=["test"],
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
            WeightedModelCountingTestDataset(
                name="no_clauses",
                pretty_name="No Clauses",
                description="p cnf 2 0",
                suites=["test"],
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
            WeightedModelCountingTestDataset(
                name="default_weights",
                pretty_name="Default Weights",
                description="V1 or V2 (default weights)",
                suites=["test"],
                cnf_text=textwrap.dedent(
                    """\
                    c V1 or V2 (default weights)
                    p cnf 2 1
                    1 2 0
                """
                ),
                expected=0.75,
            ),
            WeightedModelCountingTestDataset(
                name="three_var_formula",
                pretty_name="3-Var Formula",
                description="(V1 or V2) and (not V2 or V3)",
                suites=["test"],
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
            WeightedModelCountingTestDataset(
                name="twenty_var_formula",
                pretty_name="20-Var Formula",
                description="""
                    (V1 or not V2 or V3) and (not V1 or V4 or V5) and
                    (V2 or not V5 or V6) and (not V3 or V7 or not V8)
                    and (V4 or not V9 or V10) and (not V6 or V11 or V12)
                    and (V8 or not V13 or V14) and (V10 or V15 or not V16)
                    and (not V12 or not V17 or V18) and (V14 or V19 or not V20)
                    and (not V15 or V16 or V20) and (V17 or not V18 or V19)
                """,
                suites=["test"],
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

    def generate(self, dataset: WeightedModelCountingTestDataset):
        problem = _weighted_problem(dataset.cnf_text, dataset.expected)
        problem.meta["expected_result"] = dataset.expected
        return problem


class WeightedModelCountingMCCompGenerator(
    _WMCFunctionGenerator, Generator[WeightedModelCountingMCCompDataset]
):
    @property
    def name(self) -> str:
        return "weighted_model_counting_mccomp"

    @property
    def pretty_name(self) -> str:
        return "Weighted Model Counting MCComp"

    @property
    def description(self) -> str:
        return (
            "Loads projected weighted model counting CNF instances from MCComp Track4."
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
            "Track4 instances provide weighted CNF formulas from a standard "
            "model-counting corpus."
        )

    @property
    def datasets(self) -> list[WeightedModelCountingMCCompDataset]:
        return [
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_1", suites=["standard", "trace", "train"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_2", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_3", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_4", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_5", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_6", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_7", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_8", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_9", suites=["standard", "trace"]
            ),
            WeightedModelCountingMCCompDataset(
                "Track4_PWMC/random_pwmc_10", suites=["standard", "trace"]
            ),
        ]

    def generate(self, dataset: WeightedModelCountingMCCompDataset):
        source_path = normalize_mccomp_source_path(dataset.source_path)
        local_path = download_mccomp_instance(source_path)
        cnf_text = Path(local_path).read_text(encoding="utf-8")
        exact_type, exact_value = parse_mccomp_exact(cnf_text)
        # The competition's answer is the projected count the benchmark
        # computes, so check checks it against the brute-force reference too.
        problem = _weighted_problem(
            cnf_text, None if exact_value is None else float(exact_value)
        )
        problem.meta.update(
            {
                "source_repository": MCCOMP_REPOSITORY_URL,
                "source_path": source_path,
                "source_url": mccomp_source_url(source_path),
                "local_path": str(local_path),
                "source_problem_type": parse_mccomp_problem_type(cnf_text),
                "source_num_clauses": problem.meta["num_clauses"],
                "source_exact_type": exact_type,
                "source_exact_value": exact_value,
                "source_exact_is_projected": True,
            }
        )
        return problem


class WeightedModelCountingBenchmark(Benchmark):
    @property
    def name(self):
        return "weighted_model_counting"

    @property
    def pretty_name(self):
        return "Weighted Model Counting"

    @property
    def description(self):
        return "Benchmarks Weighted Model Counting Algorithm using einsum operations."

    @property
    def suites(self):
        return ["standard-logic"]

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
        return [
            WeightedModelCountingTestGenerator(),
            WeightedModelCountingMCCompGenerator(),
        ]

    def benchmark(self, xp, meta):
        raise NotImplementedError(
            "Weighted model counting functions are generated per formula by the "
            "generator's generate_benchmark_function."
        )

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        result = float(to_numpy(self._output[0]))
        expected = float(to_numpy(self._ref_outputs[0]))
        assert self._ref_meta is not None, "No error bound; regenerate the dataset"
        bound = self._ref_meta["error_bound"]
        # The stored answer is typed in or parsed from the instance file, so check
        # it against an independent brute-force count before trusting it.
        assert "reference_count" in self._ref_meta, (
            "No NumPy reference count; regenerate the cached dataset"
        )
        reference = self._ref_meta["reference_count"]
        assert np.isclose(expected, reference, rtol=0, atol=bound), (
            f"Test '{param.dataset.name}' has stored count {expected}, but the "
            f"NumPy brute-force reference counts {reference}"
        )
        assert np.isclose(result, expected, rtol=0, atol=bound), (
            f"Test '{param.dataset.name}' failed: expected {expected}, got {result}"
        )


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
