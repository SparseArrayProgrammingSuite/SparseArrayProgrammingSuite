from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy

_OPENTRAJ_BASE = (
    "https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH"
)
KNOWN_SCENES = ("seq_eth", "seq_hotel")


def download_ewap_dataset(
    scene: str,
    *,
    data_dir: str | Path | None = None,
    num_steps: int = 50,
) -> tuple[list[BinsparseTensor], dict]:
    """Download the ETH EWAP pedestrian dataset and return particle initial conditions.

    Each unique pedestrian's first observed position and velocity becomes one
    particle. Coordinates and velocities are preserved in the dataset's units.

    ``scene`` must be one of ``"seq_eth"`` or ``"seq_hotel"``.
    """
    if scene not in KNOWN_SCENES:
        raise ValueError(f"Unknown EWAP scene {scene!r}. Choose from {KNOWN_SCENES}.")

    root = _default_data_dir() if data_dir is None else Path(data_dir)
    dataset_dir = root / "ewap"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    obsmat_path = _ensure_downloaded(dataset_dir, scene)
    x, y, z, vx, vy, vz = _parse_obsmat(obsmat_path)

    n = len(x)
    size = _coordinate_extent(x, y, z)
    mass = np.asarray(0.01, dtype=np.float64)

    return (
        [
            from_numpy(x),
            from_numpy(y),
            from_numpy(z),
            from_numpy(vx),
            from_numpy(vy),
            from_numpy(vz),
            from_numpy(mass),
        ],
        {
            "size": size,
            "steps": num_steps,
            "source_scene": scene,
            "source_url": f"{_OPENTRAJ_BASE}/{scene}/obsmat.txt",
            "n_particles": n,
            "source_dimensions": 2,
            "simulation_dimensions": 3,
            "source_x_min": float(x.min()),
            "source_x_max": float(x.max()),
            "source_y_min": float(y.min()),
            "source_y_max": float(y.max()),
            "source_z_min": float(z.min()),
            "source_z_max": float(z.max()),
        },
    )


def load_toy_ewap_dataset(
    num_steps: int = 10,
) -> tuple[list[BinsparseTensor], dict]:
    """Return a minimal EWAP-shaped dataset from hard-coded data (no network access).

    Useful for unit tests.  The toy data contains 4 pedestrians with known
    first-frame positions and velocities.
    """
    # Minimal obsmat.txt content: frame  ped  pos_x  pos_z  pos_y  v_x  v_z  v_y
    _TOY_OBSMAT = """\
% Frame_number  Pedestrian_ID  pos_x  pos_z  pos_y  v_x  v_z  v_y
1 1  0.0  0.0  0.0   0.1  0.0  0.0
1 2  3.0  0.0  0.0  -0.1  0.0  0.0
1 3  0.0  0.0  4.0   0.0  0.0  0.2
1 4  3.0  0.0  4.0   0.0  0.0 -0.2
2 1  0.1  0.0  0.0   0.1  0.0  0.0
"""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(_TOY_OBSMAT)
        tmp_path = Path(f.name)

    try:
        x, y, z, vx, vy, vz = _parse_obsmat(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    n = len(x)
    size = _coordinate_extent(x, y, z)
    mass = np.asarray(0.01, dtype=np.float64)

    return (
        [
            from_numpy(x),
            from_numpy(y),
            from_numpy(z),
            from_numpy(vx),
            from_numpy(vy),
            from_numpy(vz),
            from_numpy(mass),
        ],
        {
            "size": size,
            "steps": num_steps,
            "source_scene": "toy",
            "source_url": "N/A",
            "n_particles": n,
            "source_dimensions": 2,
            "simulation_dimensions": 3,
            "source_x_min": float(x.min()),
            "source_x_max": float(x.max()),
            "source_y_min": float(y.min()),
            "source_y_max": float(y.max()),
            "source_z_min": float(z.min()),
            "source_z_max": float(z.max()),
        },
    )


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "ewap"


def _ensure_downloaded(dataset_dir: Path, scene: str) -> Path:
    obsmat_path = dataset_dir / scene / "obsmat.txt"
    if obsmat_path.exists():
        return obsmat_path

    url = f"{_OPENTRAJ_BASE}/{scene}/obsmat.txt"
    obsmat_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = obsmat_path.with_suffix(".txt.tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(obsmat_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return obsmat_path


def _parse_obsmat(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse obsmat.txt and return first-frame x/y/z and vx/vy/vz arrays.

    File columns: frame_id  ped_id  pos_x  pos_z  pos_y  v_x  v_z  v_y
    """
    first: dict[int, tuple[float, float, float, float, float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(("%", "#")):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            ped_id = int(float(parts[1]))
            if ped_id in first:
                continue
            first[ped_id] = (
                float(parts[2]),  # pos_x
                float(parts[4]),  # pos_y
                float(parts[3]),  # pos_z
                float(parts[5]),  # v_x
                float(parts[7]),  # v_y
                float(parts[6]),  # v_z
            )

    entries = list(first.values())
    x = np.array([e[0] for e in entries], dtype=np.float64)
    y = np.array([e[1] for e in entries], dtype=np.float64)
    z = np.array([e[2] for e in entries], dtype=np.float64)
    vx = np.array([e[3] for e in entries], dtype=np.float64)
    vy = np.array([e[4] for e in entries], dtype=np.float64)
    vz = np.array([e[5] for e in entries], dtype=np.float64)
    return x, y, z, vx, vy, vz


def _coordinate_extent(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Return a benchmark size parameter without changing dataset coordinates."""
    return float(
        max(
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min(),
            1e-9,
        )
    )
