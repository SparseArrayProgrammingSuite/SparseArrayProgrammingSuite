import textwrap
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps import licenses
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    License,
    Ref,
    ReferenceDataset,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.benchmarks.mccomp_pool_instance import (
    MCCOMP_POOL_WEIGHTED_INSTANCES,
    MCCompPoolDataset,
    MCCompPoolInstanceGenerator,
    fetch_mccomp_pool_instance,
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
    """The variable count, clauses and literal weights of a DIMACS CNF in the
    Model Counting Competition format (version 1.1, June 2024).

    Weights come from "c p weight <literal> <weight> 0" lines. As the format
    specifies, a weight 0 < w < 1 given for one literal alone gives its negation
    1 - w, and a variable with no weight for either literal weighs 1 both ways.
    Any other weight given for one literal alone is a format error.
    """
    lines = [line.strip() for line in text.strip().split("\n")]
    num_vars, clauses = parse_dimacs(text)

    weights = {}
    weight_lines = [line for line in lines if line.startswith("c p weight")]
    for line in weight_lines:
        parts = line.split()
        literal = int(parts[3])
        weights[literal] = parse_weight(parts[4])

    for var in range(1, num_vars + 1):
        if var not in weights and -var not in weights:
            weights[var] = 1.0
            weights[-var] = 1.0
        elif var not in weights or -var not in weights:
            literal = var if var in weights else -var
            if not 0 < weights[literal] < 1:
                raise ValueError(
                    f"Literal {literal} has weight {weights[literal]}, which "
                    f"needs a weight for literal {-literal} too"
                )
            weights[-literal] = 1 - weights[literal]

    return num_vars, clauses, weights


def parse_shown(text, num_vars):
    """The variables a projected count sums over: those on "c p show" lines.

    As the competition format specifies, a projected instance ("c t pmc" or
    "c t pwmc") with no such lines projects onto no variables, which asks only
    whether the formula is satisfiable. Any other instance without them counts
    every variable.
    """
    shown = None
    projected = False
    for line in text.splitlines():
        parts = line.split()
        if parts[:3] == ["c", "p", "show"]:
            shown = (shown or []) + [int(part) for part in parts[3:] if part != "0"]
        if parts[:2] == ["c", "t"] and parts[2:3] in (["pmc"], ["pwmc"]):
            projected = True
    if shown is None:
        return [] if projected else list(range(1, num_vars + 1))
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


def _projected_counts(num_vars, clauses, weights, shown):
    """PySDD's count of the projected models of clauses, with their total weight
    and total absolute weight, in doubles.

    A projected model is an assignment of the shown variables that extends to a
    model, and it weighs the product of its literals' weights.
    """
    from pysdd.sdd import SddManager

    # A manager needs at least one variable, so declare one that no clause
    # uses; like the variables not shown, it is quantified away.
    manager = SddManager(num_vars + 1, False)
    formula = manager.true()
    for clause in clauses:
        disjunction = manager.false()
        for literal in clause:
            disjunction = disjunction | manager.literal(literal)
        formula = formula & disjunction
    hidden = [var for var in range(1, num_vars + 2) if var not in set(shown)]
    for var in hidden:
        formula = manager.exists(var, formula)
    models = formula.global_model_count() // 2 ** len(hidden)

    absolute_weights = {literal: abs(weight) for literal, weight in weights.items()}
    totals = []
    for literal_weights in (weights, absolute_weights):
        counter = formula.wmc(log_mode=False)
        for var in range(1, num_vars + 2):
            if var in hidden:
                # The projected formula no longer mentions var, and weights 1
                # and 0 count each of its models once.
                positive, negative = 1.0, 0.0
            else:
                positive, negative = literal_weights[var], literal_weights[-var]
            counter.set_literal_weight(manager.literal(var), positive)
            counter.set_literal_weight(manager.literal(-var), negative)
        totals.append(counter.propagate())
    return models, totals[0], totals[1]


def referenceWeightedModelCount(num_vars, clauses, weights, shown=None):
    """The weighted model count of clauses, projected onto shown (by default
    every variable), from PySDD in doubles."""
    shown = list(range(1, num_vars + 1)) if shown is None else shown
    return np.float64(_projected_counts(num_vars, clauses, weights, shown)[1])


def _weighted_inputs(cnf_text):
    """The inputs and metadata of a projected weighted count, without its
    reference."""
    num_vars, clauses, weights = parse_format(cnf_text)
    shown = parse_shown(cnf_text, num_vars)
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
    }
    return DataInstance(inputs=inputs, meta=meta)


def _weighted_reference(cnf_text):
    """The reference projected weighted count and the error bound to check it
    with.

    The count sums one product of the k shown variables' weights per projected
    model. Assignments that aren't models contribute exact zeros, which don't
    round, so whatever the order each term rounds at most
    (models - 1) + (k - 1) times.
    """
    num_vars, clauses, weights = parse_format(cnf_text)
    shown = parse_shown(cnf_text, num_vars)
    models, reference, absolute = _projected_counts(num_vars, clauses, weights, shown)
    steps = max(models - 1, 0) + max(len(shown) - 1, 0)
    bound = summation_error_bound(np, steps + 1, absolute, np.float64)
    return reference, float(bound)


def _weighted_problem(cnf_text, expected=None):
    """The inputs, metadata and reference of a projected weighted count."""
    problem = _weighted_inputs(cnf_text)
    reference, bound = _weighted_reference(cnf_text)
    problem.meta["default_total"] = reference
    problem.ref_outputs = [
        from_numpy(np.asarray(reference if expected is None else expected))
    ]
    problem.ref_meta = {"error_bound": bound, "reference_count": reference}
    return problem


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
    def notes(self) -> str:
        return "Uses a predefined set of formulas to verify correctness."

    @property
    def license(self) -> License | None:
        return licenses.SAPS

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
                description="V1 or V2, whose literals weigh 1 by default",
                suites=["test"],
                cnf_text=textwrap.dedent(
                    """\
                    c V1 or V2, whose literals weigh 1 by default
                    p cnf 2 1
                    1 2 0
                """
                ),
                expected=3.0,
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
                authors=[Author("Arijit Shaw")],
                year=2025,
                url=MCCOMP_REPOSITORY_URL,
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def notes(self) -> str:
        return (
            "Track4 instances provide weighted CNF formulas from a standard "
            "model-counting corpus."
        )

    @property
    def license(self) -> License | None:
        return licenses.MCCOMP_TEST_INSTANCES

    @property
    def datasets(self) -> list[WeightedModelCountingMCCompDataset]:
        # De-listed: the MC 2025 test instances have no license (see
        # licenses.MCCOMP_TEST_INSTANCES), so these are in no cached suite, nor
        # trace or train.
        return [
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_1", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_2", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_3", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_4", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_5", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_6", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_7", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_8", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_9", suites=[]),
            WeightedModelCountingMCCompDataset("Track4_PWMC/random_pwmc_10", suites=[]),
        ]

    def generate(self, dataset: WeightedModelCountingMCCompDataset):
        source_path = normalize_mccomp_source_path(dataset.source_path)
        local_path = download_mccomp_instance(source_path)
        cnf_text = Path(local_path).read_text(encoding="utf-8")
        exact_type, exact_value = parse_mccomp_exact(cnf_text)
        # The competition's answer is the projected count the benchmark
        # computes, so check checks it against the PySDD reference too.
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


class WeightedModelCountingMCCompPoolGenerator(
    _WMCFunctionGenerator, Generator[MCCompPoolDataset]
):
    @property
    def name(self) -> str:
        return "weighted_model_counting_mccomp_pool"

    @property
    def pretty_name(self) -> str:
        return "Weighted Model Counting MCComp Pool"

    @property
    def description(self) -> str:
        return (
            "Counts the weighted models of the small Track 2 and Track 4 instances "
            "in the Model Counting Competition 2020-2024 archives."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return MCCompPoolInstanceGenerator().references

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def notes(self) -> str:
        return (
            "Weighted and projected weighted formulas from the Model Counting "
            "Competitions, small enough that a dense evaluation of the counting "
            "einsums finishes."
        )

    @property
    def license(self) -> License | None:
        return licenses.MCCOMP_POOL

    @property
    def cacheable(self) -> bool:
        # The instances come from the cached shell generator and the counts from
        # the cached reference generator.
        return False

    @property
    def datasets(self) -> list[MCCompPoolDataset]:
        # The largest instance, mc2024_track4_182, traces and trains the
        # benchmark: 13 variables projected onto 9, so it exercises projection.
        return [
            MCCompPoolDataset(
                dataset.source_path,
                suites=(
                    ["trace", "train"]
                    if dataset.source_path
                    == "mc2024-track4-pwmc_competition/mc2024_track4_182.cnf.xz"
                    else []
                ),
            )
            for dataset in MCCOMP_POOL_WEIGHTED_INSTANCES
        ]

    def generate(self, dataset: MCCompPoolDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = WeightedModelCountingMCCompPoolReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        assert reference.ref_meta is not None
        problem.meta["default_total"] = reference.ref_meta["reference_count"]
        problem.meta["expected_result"] = reference.ref_meta["reference_count"]
        problem.ref_outputs = reference.ref_outputs
        problem.ref_meta = reference.ref_meta
        return problem

    def generate_inputs(self, dataset: MCCompPoolDataset) -> DataInstance:
        raw = fetch_mccomp_pool_instance(dataset.source_path)
        problem = _weighted_inputs(to_numpy(raw.inputs[0]).tobytes().decode())
        problem.meta.update(
            {
                "source_generator": MCCompPoolInstanceGenerator().name,
                "source_path": dataset.source_path,
            }
        )
        return problem


class WeightedModelCountingMCCompPoolReferenceGenerator(
    ReferenceGenerator[MCCompPoolDataset]
):
    @property
    def name(self) -> str:
        return "weighted_model_counting_mccomp_pool_reference"

    @property
    def pretty_name(self) -> str:
        return "Weighted Model Counting MCComp Pool Reference"

    @property
    def description(self) -> str:
        return (
            "PySDD weighted model counts and their error bounds for Weighted Model "
            "Counting MCComp Pool, cached apart from its inputs."
        )

    @property
    def license(self) -> License | None:
        return WeightedModelCountingMCCompPoolGenerator().license

    @property
    def datasets(self) -> list[ReferenceDataset[MCCompPoolDataset]]:
        return [
            ReferenceDataset(dataset)
            for dataset in WeightedModelCountingMCCompPoolGenerator().datasets
        ]

    def generate(self, dataset: ReferenceDataset[MCCompPoolDataset]) -> DataInstance:
        raw = fetch_mccomp_pool_instance(dataset.dataset.source_path)
        reference, bound = _weighted_reference(
            to_numpy(raw.inputs[0]).tobytes().decode()
        )
        return DataInstance(
            inputs=[],
            meta={},
            ref_outputs=[from_numpy(np.asarray(reference))],
            ref_meta={"error_bound": bound, "reference_count": reference},
        )


class WeightedModelCountingMCCompPoolReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return WeightedModelCountingMCCompPoolReferenceGenerator()


class WeightedModelCountingBenchmark(Benchmark):
    @property
    def name(self):
        return "weighted_model_counting"

    @property
    def pretty_name(self):
        return "Weighted Model Counting"

    @property
    def description(self) -> str:
        return (
            "Counts the total weight of the satisfying assignments of a SAT problem, "
            "given variable weights, typically probabilities."
        )

    @property
    def suites(self):
        return ["counting", "boolean-tensor-networks"]

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
    def notes(self) -> str:
        return (
            "Benchmarks Weighted Model Counting Algorithm using einsum operations. "
            "Weighted Model Counting extends model counting by assigning specific "
            "weights to the variables."
        )

    @property
    def correctness(self) -> str:
        return (
            "Each weighted count sums one product of weights per model, so we check it "
            "against an independent brute-force count up to the rounding error of that "
            "sum under any floating-point "
            r"reassociation~\cite[(2.6)]{higham_accuracy_1993}."
        )

    @property
    def generators(self) -> list[Generator[Any]]:
        return [
            WeightedModelCountingTestGenerator(),
            WeightedModelCountingMCCompGenerator(),
            WeightedModelCountingMCCompPoolGenerator(),
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
        # The stored answer may be typed in or parsed from the instance file, so
        # check it against an independent PySDD count before trusting it.
        assert "reference_count" in self._ref_meta, (
            "No PySDD reference count; regenerate the cached dataset"
        )
        reference = self._ref_meta["reference_count"]
        assert np.isclose(expected, reference, rtol=0, atol=bound), (
            f"Test '{param.dataset.name}' has stored count {expected}, but the "
            f"PySDD reference counts {reference}"
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
