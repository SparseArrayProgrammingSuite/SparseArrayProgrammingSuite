"""Descriptions and correctness notes are LaTeX, and cite references by key."""

from __future__ import annotations

import re

from saps.metadata import _benchmark_instances

CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def _cited(text: str) -> set[str]:
    return {key.strip() for keys in CITE.findall(text) for key in keys.split(",")}


def test_citations_name_a_reference_of_their_own():
    missing = []
    titles: dict[str, set[str]] = {}
    for benchmark in _benchmark_instances():
        records = [
            (benchmark, benchmark.references, benchmark.description),
            (benchmark, benchmark.references, benchmark.correctness),
        ]
        for generator in benchmark.generators:
            records.append((generator, generator.references, generator.description))
            # Datasets have no references of their own; they cite their generator's.
            records.extend(
                (dataset, generator.references, dataset.description)
                for dataset in generator.datasets
            )
        for record, references, text in records:
            for ref in references:
                if ref.key:
                    titles.setdefault(ref.key, set()).add(ref.title.casefold())
            keys = {ref.key for ref in references if ref.key}
            missing.extend(
                f"{record.name}: \\cite{{{key}}}" for key in sorted(_cited(text) - keys)
            )
    assert not missing, "cited keys missing from references: " + ", ".join(missing)
    clashes = {key: names for key, names in titles.items() if len(names) > 1}
    assert not clashes, f"one key names several works: {clashes}"
