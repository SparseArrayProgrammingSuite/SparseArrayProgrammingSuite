from __future__ import annotations

import gzip
import hashlib
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy

from saps.downloaders.cache import download_lock, source_cache_dir

NEMO_ARCHIVE_BASE_URL = "https://carma.astro.umd.edu/nemo/archive"

# Particle tables that scripts/generate_nemo_particles.py made with NEMO, and
# their MD5s, so that changing a table changes this module and with it the
# freshness of the datasets built from it.
SAPS_NEMO_DIR = Path(__file__).resolve().parents[1] / "datafiles" / "nemo"
SAPS_NEMO_TABLES = {
    "plummer_128.tab.gz": "a1d49b7a4e4a9e8193305155739a5813",
    "plummer_1024.tab.gz": "0ee4ef2d5bdee7a4501258b1bc1dd265",
    "mw_m31.tab.gz": "1a15a6636661fc83adcc7b21ad3304fc",
}


def download_nemo_dataset(
    archive_path: str,
    *,
    columns: tuple[str, ...],
    wrap: bool = False,
    expected_particles: int | None = None,
    data_dir: str | Path | None = None,
    num_steps: int = 50,
    box_size: float | None = None,
    include_mass: bool = False,
) -> tuple[list[BinsparseTensor], dict[str, Any]]:
    """Download and parse an ASCII NEMO archive particle snapshot.

    The NEMO archive documents these files as tables convertible with
    ``tabtos``.  This lightweight parser handles the same whitespace-delimited
    table data directly, including Sellwood's wrapped six-coordinate stream.
    Positions are translated so the minimum x/y/z coordinate is at the
    simulation origin. No rescaling is applied: distances and
    velocities remain in the archive's units.
    """
    return _nemo_dataset(
        _ensure_downloaded(archive_path, data_dir),
        {
            "source_archive": "NEMO",
            "source_url": f"{NEMO_ARCHIVE_BASE_URL}/{archive_path}",
            "source_path": archive_path,
        },
        columns=columns,
        wrap=wrap,
        expected_particles=expected_particles,
        num_steps=num_steps,
        box_size=box_size,
        include_mass=include_mass,
    )


def load_saps_nemo_dataset(
    table: str,
    *,
    columns: tuple[str, ...],
    wrap: bool = False,
    expected_particles: int | None = None,
    num_steps: int = 50,
    box_size: float | None = None,
    include_mass: bool = False,
) -> tuple[list[BinsparseTensor], dict[str, Any]]:
    """Parse one of SAPS's own NEMO particle tables, like
    ``download_nemo_dataset``."""
    if table not in SAPS_NEMO_TABLES:
        raise ValueError(f"Unknown SAPS NEMO table: {table!r}")
    path = SAPS_NEMO_DIR / table
    md5 = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    if md5 != SAPS_NEMO_TABLES[table]:
        raise ValueError(f"{path} does not match its MD5 {SAPS_NEMO_TABLES[table]}")
    return _nemo_dataset(
        path,
        {"source_archive": "SAPS", "source_path": f"src/saps/datafiles/nemo/{table}"},
        columns=columns,
        wrap=wrap,
        expected_particles=expected_particles,
        num_steps=num_steps,
        box_size=box_size,
        include_mass=include_mass,
    )


def _nemo_dataset(
    local_path: Path,
    source: dict[str, Any],
    *,
    columns: tuple[str, ...],
    wrap: bool,
    expected_particles: int | None,
    num_steps: int,
    box_size: float | None,
    include_mass: bool,
) -> tuple[list[BinsparseTensor], dict[str, Any]]:
    rows, column_index = _parse_nemo_rows(
        local_path,
        columns=columns,
        expected_particles=expected_particles,
    )
    x_raw = rows[:, column_index["x"]].copy()
    y_raw = rows[:, column_index["y"]].copy()
    z_raw = rows[:, column_index["z"]].copy()
    vx_raw = rows[:, column_index["vx"]].copy()
    vy_raw = rows[:, column_index["vy"]].copy()
    vz_raw = rows[:, column_index["vz"]].copy()
    n_particles = len(x_raw)
    x, y, z, size, offsets = _translate_to_box(x_raw, y_raw, z_raw, box_size)
    inputs = [
        from_numpy(x),
        from_numpy(y),
        from_numpy(z),
        from_numpy(vx_raw),
        from_numpy(vy_raw),
        from_numpy(vz_raw),
    ]
    meta = {
        "size": size,
        "steps": num_steps,
        "n_particles": n_particles,
        **source,
        "source_columns": list(columns),
        "source_wrap": wrap,
        "source_x_min": float(x_raw.min()),
        "source_x_max": float(x_raw.max()),
        "source_y_min": float(y_raw.min()),
        "source_y_max": float(y_raw.max()),
        "source_z_min": float(z_raw.min()),
        "source_z_max": float(z_raw.max()),
        "position_offset_x": float(offsets[0]),
        "position_offset_y": float(offsets[1]),
        "position_offset_z": float(offsets[2]),
    }
    if include_mass:
        if "mass" not in column_index:
            raise ValueError(f"NEMO columns for {local_path} do not include mass")
        mass = rows[:, column_index["mass"]].copy()
        inputs.append(from_numpy(mass))
        meta["source_mass_min"] = float(mass.min())
        meta["source_mass_max"] = float(mass.max())
        meta["source_mass_sum"] = float(mass.sum())

    return (
        inputs,
        meta,
    )


def parse_nemo_snapshot(
    path: str | Path,
    *,
    columns: tuple[str, ...],
    wrap: bool = False,
    expected_particles: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse x/y/z/vx/vy/vz arrays from a NEMO ASCII particle table."""
    if not {"x", "y", "z", "vx", "vy", "vz"}.issubset(columns):
        raise ValueError("NEMO columns must include x, y, z, vx, vy, and vz")

    rows, column_index = _parse_nemo_rows(
        path,
        columns=columns,
        expected_particles=expected_particles,
    )
    return (
        rows[:, column_index["x"]].copy(),
        rows[:, column_index["y"]].copy(),
        rows[:, column_index["z"]].copy(),
        rows[:, column_index["vx"]].copy(),
        rows[:, column_index["vy"]].copy(),
        rows[:, column_index["vz"]].copy(),
    )


def _parse_nemo_rows(
    path: str | Path,
    *,
    columns: tuple[str, ...],
    expected_particles: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    values = np.fromiter(_iter_floats(path), dtype=np.float64)
    width = len(columns)
    if values.size % width != 0:
        raise ValueError(
            f"Malformed NEMO snapshot {path}: {values.size} values is not divisible "
            f"by {width} columns"
        )

    rows = values.reshape((-1, width))
    if expected_particles is not None and rows.shape[0] != expected_particles:
        raise ValueError(
            f"Expected {expected_particles} particles in {path}, found {rows.shape[0]}"
        )

    column_index = {name: idx for idx, name in enumerate(columns)}
    return rows, column_index


def _ensure_downloaded(archive_path: str, data_dir: str | Path | None) -> Path:
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dest_path = root / archive_path
    with download_lock(dest_path):
        if dest_path.exists():
            return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        download_url = f"{NEMO_ARCHIVE_BASE_URL}/{archive_path}"
        tmp_path = dest_path.with_name(dest_path.name + ".tmp")
        try:
            urllib.request.urlretrieve(download_url, tmp_path)  # noqa: S310
            tmp_path.replace(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest_path


def _default_data_dir() -> Path:
    return source_cache_dir("nemo")


def _iter_floats(path: str | Path) -> Iterator[float]:
    with _open_text(Path(path)) as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            for value in stripped.split():
                yield float(value)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _translate_to_box(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    box_size: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float]]:
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-12)
    size = float(box_size) if box_size is not None else float(span)
    return (
        (x - x_min).astype(np.float64),
        (y - y_min).astype(np.float64),
        (z - z_min).astype(np.float64),
        size,
        (float(x_min), float(y_min), float(z_min)),
    )
