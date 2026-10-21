#!/usr/bin/env python3
"""Classify SAPS competition failures from combined results.json.

Uses result["status"] first, then each dataset's stderr regexes and errcode.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STDERR_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "timeout",
        re.compile(r"asv:\s*benchmark timed out|TimeoutError|\btimed?\s*out\b", re.I),
    ),
    (
        "oom",
        re.compile(
            r"MemoryError|_ArrayMemoryError|Cannot allocate memory|bad_alloc|"
            r"out of memory|\boom\b",
            re.I,
        ),
    ),
    ("not_implemented", re.compile(r"\bNotImplementedError\b")),
    ("type_error", re.compile(r"\bTypeError\b")),
    ("attribute_error", re.compile(r"\bAttributeError\b")),
    ("value_error", re.compile(r"\bValueError\b")),
    ("runtime_error", re.compile(r"\bRuntimeError\b")),
    ("assertion", re.compile(r"\bAssertionError\b")),
    ("import_error", re.compile(r"\b(?:ImportError|ModuleNotFoundError)\b")),
]

# When stderr has no exception line, fall back to process exit / signal.
ERRCODE_FALLBACK = {
    "-256": "timeout",  # ASV timeout kill
    "-9": "killed_sigkill",  # SIGKILL
    "-15": "killed_sigterm",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_results() -> Path | None:
    runs = [
        path
        for path in (REPO_ROOT / "competition").glob("run_*")
        if path.is_dir() and path.name.removeprefix("run_").isdigit()
    ]
    if not runs:
        return None
    return max(runs, key=lambda path: int(path.name[4:])) / "results.json"


def failure_output(run: dict | None, parameter: str) -> tuple[str, int | None]:
    if not run:
        return "", None
    stderr = run.get("stderr") or ""
    if len(run.get("parameters") or []) <= 1:
        return stderr, run.get("errcode")
    sections: dict[str, list[str]] = {}
    lines: list[str] = []
    for line in stderr.splitlines():
        if line.startswith("For parameters: "):
            lines = sections.setdefault(line.removeprefix("For parameters: "), [])
        else:
            lines.append(line)
    return "\n".join(sections.get(parameter, [])), None


def classify_failure(stderr: str | None, errcode) -> str:
    text = stderr or ""
    for name, pattern in STDERR_RULES:
        if pattern.search(text):
            return name
    if not text.strip():
        return ERRCODE_FALLBACK.get(str(errcode), "empty_stderr")
    if re.search(r"^INFO |^For parameters:", text, re.M) and not re.search(
        r"Error|Exception", text
    ):
        return ERRCODE_FALLBACK.get(str(errcode), "log_only_then_fail")
    return ERRCODE_FALLBACK.get(str(errcode), "other")


def iter_rows(doc: dict):
    diagnostics = doc["diagnostics"]
    frameworks = {key: value["name"] for key, value in doc["frameworks"].items()}
    for benchmark in doc["benchmarks"]:
        for generator in benchmark["generators"]:
            for dataset in generator["datasets"]:
                for result in dataset["results"]:
                    status = result["status"]
                    if status.endswith("_timeout"):
                        outcome = "timeout"
                    elif status != "failed":
                        outcome = status
                    elif result.get("reason") == "nonfinite":
                        outcome = "nonfinite"
                    else:
                        stderr, errcode = failure_output(
                            diagnostics.get(result.get("diagnostics")),
                            f"{generator['name']}.{dataset['name']}",
                        )
                        outcome = classify_failure(stderr, errcode)
                    yield {
                        "framework": frameworks[result["framework"]],
                        "status": status,
                        "class": outcome,
                    }


# Stack order
OUTCOME_ORDER = (
    "ok",
    "not_implemented",
    "attribute_error",
    "type_error",
    "value_error",
    "runtime_error",
    "assertion",
    "import_error",
    "killed_sigkill",
    "oom",
    "timeout",
    "skipped",
    "nonfinite",
    "empty_stderr",
    "log_only_then_fail",
    "other",
)
OUTCOME_COLORS = {
    "ok": "#08519C",
    "not_implemented": "#E31A1C",
    "attribute_error": "#6A3D9A",
    "type_error": "#FF7F00",
    "value_error": "#8C510A",
    "runtime_error": "#33A02C",
    "killed_sigkill": "#F781BF",
    "oom": "#FFD92F",
    "timeout": "#1F9E89",
    "assertion": "#000000",
    "import_error": "#A6CEE3",
    "skipped": "#BDBDBD",
    "nonfinite": "#636363",
    "empty_stderr": "#FB9A99",
    "log_only_then_fail": "#80B1D3",
    "other": "#B3DE69",
}
FRAMEWORK_ORDER = ("numpy", "scipy", "sparse", "smart", "pytorch")
NOT_FAILURES = ("ok", "skipped")


def _pie(counts: Counter[str], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    items = counts.most_common()
    labels = [name.replace("_", " ") for name, _ in items]
    sizes = [count for _, count in items]
    total = sum(sizes)
    figure, axes = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = axes.pie(
        sizes,
        startangle=90,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        pctdistance=0.72,
    )
    for text in autotexts:
        text.set_fontsize(9)
    axes.legend(
        wedges,
        [
            f"{label} ({count / total:.1%})"
            for label, count in zip(labels, sizes, strict=True)
        ],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    axes.set_title(f"{title} (n={total})")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_pies(
    overall: Counter[str],
    by_framework: dict[str, Counter[str]],
    output_dir: Path,
) -> list[Path]:
    failure_types = Counter(
        {name: count for name, count in overall.items() if name not in NOT_FAILURES}
    )
    by_framework_totals = Counter(
        {
            name: sum(
                count for label, count in counts.items() if label not in NOT_FAILURES
            )
            for name, counts in by_framework.items()
        }
    )
    paths = [
        output_dir / "failures_by_framework.pdf",
        output_dir / "failures_by_type.pdf",
    ]
    _pie(by_framework_totals, "Failures by framework", paths[0])
    _pie(failure_types, "Failures by type", paths[1])
    return paths


def _outcome_label(name: str) -> str:
    return "success" if name == "ok" else name.replace("_", " ")


def plot_outcomes(
    by_framework: dict[str, Counter[str]],
    path: Path,
    *,
    include_success: bool = True,
) -> None:

    import matplotlib.pyplot as plt

    frameworks = [name for name in FRAMEWORK_ORDER if name in by_framework]
    frameworks += sorted(set(by_framework) - set(frameworks))
    present = {name for counts in by_framework.values() for name in counts}
    if not include_success:
        present -= set(NOT_FAILURES)
    outcomes = [name for name in OUTCOME_ORDER if name in present]
    outcomes += sorted(present - set(outcomes))

    figure, axes = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * len(frameworks)
    for outcome in outcomes:
        heights = []
        for framework in frameworks:
            counts = by_framework[framework]
            total = sum(counts.values())
            heights.append(counts[outcome] / total if total else 0.0)
        axes.bar(
            frameworks,
            heights,
            bottom=bottoms,
            width=0.72,
            color=OUTCOME_COLORS.get(outcome, "#B0B0B0"),
            edgecolor="white",
            linewidth=0.6,
            label=_outcome_label(outcome),
        )
        bottoms = [
            bottom + height for bottom, height in zip(bottoms, heights, strict=True)
        ]
    axes.set_ylim(0, 1)
    axes.set_ylabel("Fraction of results")
    axes.set_xlabel("Framework")
    axes.set_title(
        "Outcomes by framework" if include_success else "Failures by framework"
    )
    axes.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _pos: f"{value:.0%}")
    )
    axes.legend(
        title="Outcome",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        help="competition/run_*/results.json (default: highest numbered run)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        metavar="DIR",
        help=(
            "Write pie charts and both stacked bar charts "
            "(default directory: <run>/plots)"
        ),
    )
    args = parser.parse_args()
    path = args.results or latest_results()
    if path is None or not path.is_file():
        parser.error("No competition/run_*/results.json found.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    overall: Counter[str] = Counter()
    by_framework: dict[str, Counter[str]] = defaultdict(Counter)
    for row in iter_rows(doc):
        overall[row["class"]] += 1
        by_framework[row["framework"]][row["class"]] += 1
    print(f"results: {path}")
    print("counts:", dict(overall))
    if args.plot is not None:
        output_dir = Path(args.plot) if args.plot else path.parent / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = [
            *plot_pies(overall, by_framework, output_dir),
            output_dir / "outcomes_by_framework.pdf",
            output_dir / "failures_by_framework_stacked.pdf",
        ]
        plot_outcomes(by_framework, plot_paths[-2])
        plot_outcomes(by_framework, plot_paths[-1], include_success=False)
        for plot_path in plot_paths:
            print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
