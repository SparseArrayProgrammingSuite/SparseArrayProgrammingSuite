#!/usr/bin/env python3
"""Classify SAPS competition failures from combined results.json.

Uses result["status"] first, then diagnostics errcode + stderr regexes.
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


def _pick_field(run: dict, dataset_name: str, generator_name: str, field: str):
    params = run.get("parameters") or []
    value = run.get(field)
    want = f"{generator_name}.{dataset_name}"
    idx = None
    for i, param in enumerate(params):
        flat = param[0] if isinstance(param, (list, tuple)) and param else param
        if flat == want or (isinstance(flat, str) and flat.endswith(dataset_name)):
            idx = i
            break
    if idx is None and len(params) == 1:
        idx = 0
    if isinstance(value, list):
        if idx is not None and idx < len(value):
            return value[idx]
        if field == "stderr":
            return "\n".join(item or "" for item in value)
        return next(
            (item for item in value if item not in (None, 0)),
            value[0] if value else None,
        )
    return value


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
                    row = {
                        "framework": frameworks[result["framework"]],
                        "status": status,
                        "class": status,
                    }
                    if status != "failed":
                        yield row
                        continue
                    run = diagnostics[result["diagnostics"]]
                    stderr = (
                        _pick_field(run, dataset["name"], generator["name"], "stderr")
                        or ""
                    )
                    errcode = _pick_field(
                        run, dataset["name"], generator["name"], "errcode"
                    )
                    row["class"] = classify_failure(stderr, errcode)
                    yield row


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
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_types = Counter(
        {name: count for name, count in overall.items() if name != "ok"}
    )
    by_framework_totals = Counter(
        {name: sum(counts.values()) for name, counts in by_framework.items()}
    )
    paths = [
        output_dir / "failures_by_framework.pdf",
        output_dir / "failures_by_type.pdf",
    ]
    _pie(by_framework_totals, "Failures by framework", paths[0])
    _pie(failure_types, "Failures by type", paths[1])
    return paths


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
        help="Write pie charts (default directory: <run>/plots)",
    )
    args = parser.parse_args()
    path = args.results or latest_results()
    if path is None or not path.is_file():
        parser.error("No competition/run_*/results.json found.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    overall: Counter[str] = Counter()
    by_framework: dict[str, Counter[str]] = defaultdict(Counter)
    for row in iter_rows(doc):
        if row["status"] != "failed":
            overall[row["status"]] += 1
            continue
        overall[row["class"]] += 1
        by_framework[row["framework"]][row["class"]] += 1
    print(f"results: {path}")
    print("counts:", dict(overall))
    if args.plot is not None:
        output_dir = Path(args.plot) if args.plot else path.parent / "plots"
        for plot_path in plot_pies(overall, by_framework, output_dir):
            print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
