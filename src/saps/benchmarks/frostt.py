from saps.benchmark import (
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.frostt import load_frostt_tensor
from saps_framework import BinsparseFormat


class FrosttDataset(Dataset):
    """Base Dataset for benchmarks backed by a FROSTT sparse tensor."""

    def __init__(
        self,
        name: str,
        *,
        path: str,
        order: int,
        shape: tuple[int, ...],
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self.path = path
        self.order = order
        self.shape = shape
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self._name

    @property
    def description(self) -> str:
        return self._description or f"FROSTT tensor {self._name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


# Small, deterministic 3rd-order tensors (matrix-multiplication structure: M*K * K*N).
# Real, genuinely sparse data, but tiny enough to be cheap to download every run.
_MATMUL_SIZES = [
    (2, 2, 2),
    (3, 3, 3),
    (4, 3, 2),
    (4, 4, 3),
    (4, 4, 4),
    (5, 5, 5),
    (6, 3, 3),
]

# Real-world tensors from the FROSTT catalog (frostt.io). Some of these (e.g.
# amazon_reviews, patents, reddit_2015) are tens of GB and will take a long time
# to download and generate; select datasets explicitly by name rather than
# running this whole catalog unfiltered. 
# 
# fb-m, darpa, and lanl2 are excluded:
# fb-m/darpa live in a separate bucket that currently denies listing (their
# exact object keys couldn't be confirmed), and lanl2's listed bucket prefix is
# empty (no files were ever uploaded there).
_TENSORS: list[FrosttDataset] = [
    FrosttDataset(
        f"matmul_{m}_{k}_{n}",
        path=f"matrix-multiplication/matmul_{m}-{k}-{n}.tns.gz",
        order=3,
        # Tensor order via real downloads: (k*n, m*n, m*k).
        shape=(k * n, m * n, m * k),
        description=(
            f"FROSTT matrix-multiplication tensor for a ({m}x{k}) * ({k}x{n}) product."
        ),
    )
    for m, k, n in _MATMUL_SIZES
] + [
    FrosttDataset(
        "nell_2",
        path="nell/nell-2.tns.gz",
        order=3,
        shape=(12092, 9184, 28818),
        description=(
            "NELL-2 knowledge base snapshot: entity x relation x entity, from the"
            " Never-Ending Language Learning project."
        ),
    ),
    FrosttDataset(
        "chicago_crime_comm",
        path="chicago-crime/comm/chicago-crime-comm.tns.gz",
        order=4,
        shape=(6186, 24, 77, 32),
        description=(
            "Chicago crime reports (2001-2017): day x hour x community-area x"
            " crime-type, non-zeros are counts."
        ),
    ),
    FrosttDataset(
        "lbnl_network",
        path="lbnl-network/lbnl-network.tns.gz",
        order=5,
        shape=(1605, 4198, 1631, 4209, 868131),
        description=(
            "Ten days of LBNL/ICSI internal network traffic: sender-IP x"
            " sender-port x dest-IP x dest-port x time, values are packet"
            " lengths."
        ),
    ),
    FrosttDataset(
        "toy",
        path="toy/toy.tns.gz",
        order=4,
        shape=(3, 3, 2, 2),
        description="Tiny toy tensor for smoke-testing the FROSTT pipeline.",
    ),
    FrosttDataset(
        "nips",
        path="nips/nips.tns.gz",
        order=4,
        shape=(2482, 2862, 14036, 17),
        description=(
            "NIPS conference proceedings: paper x author x word x year,"
            " from 17 years of NIPS papers."
        ),
    ),
    FrosttDataset(
        "uber_pickups",
        path="uber-pickups/uber.tns.gz",
        order=4,
        shape=(183, 24, 1140, 1717),
        description=(
            "Uber pickups in New York City: date x hour x latitude x"
            " longitude, non-zeros are pickup counts."
        ),
    ),
    FrosttDataset(
        "chicago_crime_geo",
        path="chicago-crime/geo/chicago-crime-geo.tns.gz",
        order=5,
        shape=(6185, 24, 380, 395, 32),
        description=(
            "Chicago crime reports (2001-2017): day x hour x latitude x"
            " longitude x crime-type, non-zeros are counts. Companion to"
            " chicago_crime_comm with geographic coordinates instead of"
            " community areas."
        ),
    ),
    FrosttDataset(
        "vast_2015_mc1_3d",
        path="vast-2015-mc1/vast-2015-mc1-3d.tns.gz",
        order=3,
        shape=(165427, 11374, 2),
        description=(
            "VAST Challenge 2015 mini-challenge 1 (3-way): time x person x"
            " action, from a simulated theme-park sensor/movement log."
        ),
    ),
    FrosttDataset(
        "nell_1",
        path="nell/nell-1.tns.gz",
        order=3,
        shape=(2902330, 2143368, 25495389),
        description=(
            "NELL-1 knowledge base snapshot: entity x relation x entity,"
            " from the Never-Ending Language Learning project (larger than"
            " nell_2)."
        ),
    ),
    FrosttDataset(
        "vast_2015_mc1_5d",
        path="vast-2015-mc1/vast-2015-mc1-5d.tns.gz",
        order=5,
        shape=(165427, 11374, 2, 100, 89),
        description=(
            "VAST Challenge 2015 mini-challenge 1 (5-way): time x person x"
            " action x x-location x y-location."
        ),
    ),
    FrosttDataset(
        "enron",
        path="enron/enron.tns.gz",
        order=4,
        shape=(6066, 5699, 244268, 1176),
        description=(
            "Enron email corpus: sender x receiver x word x date, non-zeros"
            " are word counts."
        ),
    ),
    FrosttDataset(
        "flickr_3d",
        path="flickr/flickr-3d.tns.gz",
        order=3,
        shape=(319686, 28153045, 1607191),
        description=(
            "Flickr user-image-tag associations (3-way, dates removed):"
            " user x image x tag."
        ),
    ),
    FrosttDataset(
        "flickr_4d",
        path="flickr/flickr-4d.tns.gz",
        order=4,
        shape=(319686, 28153045, 1607191, 731),
        description=(
            "Flickr user-image-tag-date associations (4-way): user x image"
            " x tag x date."
        ),
    ),
    FrosttDataset(
        "delicious_3d",
        path="delicious/delicious-3d.tns.gz",
        order=3,
        shape=(532924, 17262471, 2480308),
        description=(
            "Delicious bookmarking service (3-way, dates removed): user x"
            " webpage x tag."
        ),
    ),
    FrosttDataset(
        "delicious_4d",
        path="delicious/delicious-4d.tns.gz",
        order=4,
        shape=(532924, 17262471, 2480308, 1443),
        description=(
            "Delicious bookmarking service (4-way): user x webpage x tag x"
            " date, binary values indicating whether a user tagged a"
            " webpage on a given day."
        ),
    ),
    # The remaining 3 are the largest tensors in the FROSTT catalog.
    # Their shape/nnz below are from FROSTT's own documentation.
    FrosttDataset(
        "amazon_reviews",
        path="amazon/amazon-reviews.tns.gz",
        order=3,
        shape=(4821207, 1774269, 1805187),
        description=(
            "Amazon product reviews (from SNAP): user x product x word,"
            " after stopword removal and stemming."
        ),
    ),
    FrosttDataset(
        "patents",
        path="patents/patents.tns.gz",
        order=3,
        shape=(46, 239172, 239172),
        description=(
            "US utility patents: year x term x term, pairwise co-occurrence"
            " of terms within a 7-word window; each yearly slice is"
            " symmetric."
        ),
    ),
    FrosttDataset(
        "reddit_2015",
        path="reddit-2015/reddit-2015.tns.gz",
        order=3,
        shape=(8211298, 176962, 8116559),
        description=(
            "Reddit comments from 2015: user x subreddit x word, counting"
            " how often a user posted a word in a subreddit; users,"
            " subreddits, and words appearing fewer than 5 times are"
            " excluded."
        ),
    ),
]


class FrosttTensorGenerator(Generator[FrosttDataset]):
    """Downloads and caches raw FROSTT tensors, shared across every benchmark."""

    @property
    def name(self) -> str:
        return "frostt_tensor"

    @property
    def pretty_name(self) -> str:
        return "FROSTT Sparse Tensor Collection"

    @property
    def description(self) -> str:
        return (
            "Downloads and caches raw tensors from FROSTT (the Formidable Repository"
            " of Open Sparse Tensors and Tools, frostt.io). Benchmark-specific"
            " generators compose this generator instead of downloading tensors"
            " themselves, so a tensor used by multiple benchmarks is only"
            " downloaded, cached, and uploaded once."
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
                title=(
                    "FROSTT: The Formidable Repository of Open Sparse Tensors and Tools"
                ),
                authors=[],
                url="http://frostt.io/",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "This generator and its downloader were written by a generative AI"
            " assistant (Claude) at the user's direction."
        )

    @property
    def motivation(self) -> str:
        return (
            "Many tensor-decomposition benchmarks reuse the same FROSTT tensors."
            " Sharing a single cacheable generator for the raw download avoids"
            " redundant downloads and redundant cached copies of the same tensor."
        )

    @property
    def datasets(self) -> list[FrosttDataset]:
        return _TENSORS

    def generate(self, dataset: FrosttDataset) -> DataInstance:
        indices, values, meta = load_frostt_tensor(dataset.path)
        tensor_bin = BinsparseFormat.from_coo(indices, values, meta["shape"])
        return DataInstance(inputs=[tensor_bin], meta=meta)


class FrosttTensorBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return FrosttTensorGenerator()


def fetch_frostt_tensor(name: str) -> DataInstance:
    """Fetch (and cache) the raw tensor via the shared `FrosttTensorGenerator`."""
    raw_generator = FrosttTensorGenerator()
    raw_dataset = next(d for d in raw_generator.datasets if d.name == name)
    return raw_generator.cached_generate(raw_dataset)


def frostt_tensor_shape(name: str) -> tuple[int, ...]:
    """Statically declared shape of a raw FROSTT tensor."""
    return next(d for d in _TENSORS if d.name == name).shape
