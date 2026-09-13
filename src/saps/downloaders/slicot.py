"""Downloader for SLICOT model-reduction benchmark problems."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLICOT_BENCHMARK_PAGE_URL = (
    "https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction"
)
SLICOT_BENCH_DATA_BASE_URL = "https://www.slicot.org/objects/software/shared/bench-data"
SLICOT_ALL_DATA_ARCHIVE = "All-Data.zip"


@dataclass(frozen=True)
class SlicotProblem:
    title: str
    mat_filename: str
    description: str
    order: int
    inputs: int
    outputs: int

    @property
    def name(self) -> str:
        return self.mat_filename.removesuffix(".mat")

    @property
    def archive_filename(self) -> str:
        return f"{self.name}.zip"


SLICOT_PROBLEMS: tuple[SlicotProblem, ...] = (
    SlicotProblem(
        "Eady example",
        "eady.mat",
        "model of an atmospheric storm track",
        598,
        1,
        1,
    ),
    SlicotProblem(
        "Transmission line model",
        "tline.mat",
        "example of a transmission line model",
        256,
        2,
        2,
    ),
    SlicotProblem("CD player", "CDplayer.mat", "classical CD player model", 120, 2, 2),
    SlicotProblem(
        "PEEC model",
        "peec.mat",
        "partial element equivalent circuit model",
        480,
        1,
        1,
    ),
    SlicotProblem("FOM model", "fom.mat", "", 1006, 1, 1),
    SlicotProblem("Random example", "random.mat", "", 200, 1, 1),
    SlicotProblem("PDE example", "pde.mat", "partial differential equation", 84, 1, 1),
    SlicotProblem(
        "Heat equation (continuous case)",
        "heat-cont.mat",
        "heat equation in a thin rod",
        200,
        1,
        1,
    ),
    SlicotProblem(
        "Heat equation (discrete case)",
        "heat-disc.mat",
        "discretization of the previous equation",
        200,
        1,
        1,
    ),
    SlicotProblem(
        "Orr-Somerfeld example",
        "Orr-Som.mat",
        "Orr-Sommerfeld operator for Couette flow",
        200,
        1,
        1,
    ),
    SlicotProblem(
        "MNA example - 1",
        "MNA_1.mat",
        "Modified Nodal Analysis model",
        578,
        9,
        9,
    ),
    SlicotProblem(
        "MNA example - 2",
        "MNA_2.mat",
        "Modified Nodal Analysis model",
        9223,
        18,
        18,
    ),
    SlicotProblem(
        "MNA example - 3",
        "MNA_3.mat",
        "Modified Nodal Analysis model",
        4863,
        22,
        22,
    ),
    SlicotProblem(
        "MNA example - 4",
        "MNA_4.mat",
        "Modified Nodal Analysis model",
        980,
        4,
        4,
    ),
    SlicotProblem(
        "MNA example - 5",
        "MNA_5.mat",
        "Modified Nodal Analysis model",
        10913,
        9,
        9,
    ),
    SlicotProblem(
        "International space station",
        "iss.mat",
        "component 1r of the International Space Station",
        270,
        3,
        3,
    ),
    SlicotProblem(
        "Building model",
        "build.mat",
        "motion problem in a building",
        48,
        1,
        1,
    ),
    SlicotProblem(
        "Clamped beam model",
        "beam.mat",
        "Clamped beam model",
        348,
        1,
        1,
    ),
)


def list_slicot_problems() -> list[str]:
    """Return known SLICOT model-reduction MAT filenames."""
    return [problem.mat_filename for problem in SLICOT_PROBLEMS]


def normalize_slicot_source_name(source_name: str) -> str:
    """Return the canonical MAT filename for a SLICOT model-reduction problem."""
    name = source_name.strip().removeprefix("slicot://").removeprefix("slicot:")
    if not name or Path(name).name != name or "\\" in name:
        raise ValueError(f"Invalid SLICOT problem name: {source_name!r}")

    lower_name = name.lower()
    if lower_name.endswith(".zip"):
        lower_name = lower_name.removesuffix(".zip") + ".mat"
    elif not lower_name.endswith(".mat"):
        lower_name = f"{lower_name}.mat"

    matches = [
        problem.mat_filename
        for problem in SLICOT_PROBLEMS
        if problem.mat_filename.lower() == lower_name
        or problem.name.lower() == lower_name.removesuffix(".mat")
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown SLICOT problem name: {source_name!r}")


def slicot_problem_metadata(source_name: str) -> SlicotProblem:
    """Return registry metadata for a SLICOT model-reduction problem."""
    mat_filename = normalize_slicot_source_name(source_name)
    return next(
        problem for problem in SLICOT_PROBLEMS if problem.mat_filename == mat_filename
    )


def slicot_source_url(source_name: str) -> str:
    """Return the SLICOT-hosted zip URL for a model-reduction problem."""
    problem = slicot_problem_metadata(source_name)
    return f"{SLICOT_BENCH_DATA_BASE_URL}/{problem.archive_filename}"


def slicot_collection_url() -> str:
    """Return the SLICOT-hosted zip URL for the full model-reduction collection."""
    return f"{SLICOT_BENCH_DATA_BASE_URL}/{SLICOT_ALL_DATA_ARCHIVE}"


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "slicot"


def download_slicot_collection(*, data_dir: str | Path | None = None) -> Path:
    """Download the full SLICOT model-reduction archive and return its path."""
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    archive_path = root / SLICOT_ALL_DATA_ARCHIVE
    if archive_path.exists():
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(slicot_collection_url(), tmp_path)  # noqa: S310
        tmp_path.replace(archive_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return archive_path


def download_slicot_problem(
    source_name: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Download and extract one SLICOT model-reduction MAT file.

    The SLICOT page publishes individual examples as zip archives containing
    MATLAB MAT-files. Files are cached under ``data/slicot`` unless *data_dir*
    is provided.
    """
    problem = slicot_problem_metadata(source_name)
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    mat_path = root / problem.mat_filename
    if mat_path.exists():
        return mat_path

    archive_path = root / problem.archive_filename
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(  # noqa: S310
                slicot_source_url(source_name), tmp_path
            )
            tmp_path.replace(archive_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    _extract_slicot_mat_archive(archive_path, root, problem.mat_filename)
    return mat_path


def load_slicot_problem(
    source_name: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download and read a SLICOT problem MAT-file.

    Returns ``(variables, meta)``. MATLAB private keys from ``scipy.io.loadmat``
    are omitted from *variables*.
    """
    from scipy.io import loadmat

    problem = slicot_problem_metadata(source_name)
    local_path = download_slicot_problem(source_name, data_dir=data_dir)
    raw_variables = loadmat(local_path)
    variables = {
        key: value for key, value in raw_variables.items() if not key.startswith("__")
    }
    meta = {
        "dataset_name": problem.mat_filename,
        "title": problem.title,
        "description": problem.description,
        "order": problem.order,
        "inputs": problem.inputs,
        "outputs": problem.outputs,
        "source_url": slicot_source_url(problem.mat_filename),
        "source_page_url": SLICOT_BENCHMARK_PAGE_URL,
        "local_path": str(local_path),
    }
    return variables, meta


def _extract_slicot_mat_archive(
    archive_path: Path,
    dest_dir: Path,
    mat_filename: str,
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        mat_members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() == ".mat"
        ]
        named_members = [
            member
            for member in mat_members
            if Path(member.filename).name.lower() == mat_filename.lower()
        ]
        if len(named_members) == 1:
            member = named_members[0]
        elif len(mat_members) == 1:
            member = mat_members[0]
        else:
            names = [member.filename for member in mat_members]
            raise ValueError(
                f"Expected one MAT file named {mat_filename!r} in {archive_path}, "
                f"found {names}"
            )

        dest_path = dest_dir / mat_filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        try:
            with archive.open(member) as source, tmp_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            tmp_path.replace(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
