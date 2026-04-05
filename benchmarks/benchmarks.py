from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import saps
xp = saps.xp

#poetry run asv run --python=same -v --set-commit-hash $(git rev-parse HEAD) --record-samples

class TimeSuite(saps.Benchmark):
    @property
    def dataset_names(self):
        return [0, 2, 3]

    @property
    def pretty_name(self):
        return "Random Numerical Linear Algebra: JL Approximate NN"

    @property
    def authors(self):
        return [saps.Author("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def description(self):
        return (
            "Benchmarks Johnson-Lindenstrauss random projection followed by "
            "k-nearest-neighbor distance ranking."
        )

    @property
    def motivation(self):
        return (
            "Evaluates a foundational RNLA workload used in graph algorithms, "
            "PDE methods, and scientific machine learning."
        )

    @property
    def references(self):
        return [
            saps.Ref(
                title="Random projection implementation reference",
                authors=[saps.Author("scikit-learn contributors")],
                url="https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615",
            ),
            saps.Ref(
                title="Randomized numerical linear algebra: A perspective on the field with an eye to software",
                authors=[
                    saps.Author("Murray, R."),
                    saps.Author("Demmel, J."),
                    saps.Author("Mahoney, M. W."),
                    saps.Author("Erichson, N. B."),
                    saps.Author("Melnichenko, M."),
                    saps.Author("Malik, O. A."),
                    saps.Author("Dongarra, J."),
                ],
                year=2023,
                url="https://arxiv.org/abs/2302.11474",
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    def setup(self, dataset):
        self.d = {}
        for x in range(dataset):
            self.d[x] = None

    def run(self, dataset):
        d = self.d
        for key in range(dataset):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
