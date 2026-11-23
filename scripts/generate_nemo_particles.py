#!/usr/bin/env python
"""Generate the NEMO particle tables in src/saps/datafiles/nemo.

Each table has one row per particle: mass, x, y, z, vx, vy, vz, in N-body
units (G = 1).

- plummer_128.tab.gz, plummer_1024.tab.gz: Plummer spheres from mkplummer with
  seed 123. They match the NEMO archive's plummer/tab128.gz and tab1024.gz
  byte for byte once uncompressed.
- mw_m31.tab.gz: a Milky Way/M31 collision of 81920 particles, set up like
  model B of Dubinski, Mihos & Hernquist (1996), whose initial conditions are
  the NEMO archive's dubinski/dubinski.tab.gz. Both galaxies are Kuijken &
  Dubinski (1995) model B disk-bulge-halo galaxies from mkkd95, with 16384
  disk, 8192 bulge and 16384 halo particles. As in that table, M31 is the Milky
  Way scaled by 2.09505 in mass and 1.5 in length, and the disk spins, centers
  of mass and their velocities below were measured from it. The rows are in
  its order: Milky Way disk, M31 disk, Milky Way bulge, M31 bulge, Milky Way
  halo, M31 halo.

Requires NEMO (https://github.com/teuben/nemo) on PATH with mkplummer,
snapprint and mkkd95, and the GalactICS programs mkkd95 runs (built from
$NEMO/usr/kuijken/GalactICS-exp) in $NEMOBIN. The tables were generated with
NEMO commit cba48135a925aec0679f8a4d86a029ccb4dacb22 (mkplummer 3.0d, mkkd95
1.5c).

Usage: python scripts/generate_nemo_particles.py [output_dir]
"""

import gzip
import hashlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "src" / "saps" / "datafiles" / "nemo"

COLUMNS = "m,x,y,z,vx,vy,vz"

# Kuijken & Dubinski (1995) model B, from GalactICS's Milky_Way/B inputs.
KD95_MODEL_B = "psi0=-5.202 v0=1.36 psicut=-2.89 nr=3200"
N_DISK, N_BULGE, N_HALO = 16384, 8192, 16384

M31_MASS_SCALE = 2.09505
M31_LENGTH_SCALE = 1.5
# Polar and azimuthal angles (degrees) of each disk's spin axis, and each
# galaxy's center of mass and its velocity.
MW_SPIN = (180.0, 0.0)
M31_SPIN = (117.65, -120.15)
MW_POSITION = (1.7456, -19.0384, 6.7963)
MW_VELOCITY = (-0.4310, 0.8564, -0.3779)
M31_POSITION = (-0.8332, 9.0873, -3.2440)
M31_VELOCITY = (0.2057, -0.4088, 0.1804)


def run(command: str, cwd: Path) -> str:
    return subprocess.run(
        command, shell=True, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def write_gzip(path: Path, text: str) -> None:
    """Write text gzipped with no timestamp, so the bytes are reproducible."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as file:
        file.write(text.encode())
    path.write_bytes(buffer.getvalue())
    print(path.name, hashlib.md5(path.read_bytes()).hexdigest())


def spin_rotation(theta: float, phi: float) -> np.ndarray:
    """The rotation taking the z axis to the direction (theta, phi)."""
    t, p = np.radians(theta), np.radians(phi)
    ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    rz = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
    return rz @ ry


def galaxy(seed: int, work: Path) -> np.ndarray:
    """A centered KD95 model B galaxy whose disk spins about +z."""
    snap = f"galaxy_{seed}.snap"
    run(
        f"mkkd95 out={snap} ndisk={N_DISK} nbulge={N_BULGE} nhalo={N_HALO} "
        f"{KD95_MODEL_B} seed={seed} bin={os.environ['NEMOBIN']}",
        work,
    )
    return np.loadtxt(io.StringIO(run(f"snapprint {snap} {COLUMNS}", work)))


def place(rows, mass_scale, length_scale, spin, position, velocity):
    rotation = spin_rotation(*spin)
    velocity_scale = np.sqrt(mass_scale / length_scale)
    return np.column_stack(
        [
            rows[:, 0] * mass_scale,
            rows[:, 1:4] * length_scale @ rotation.T + position,
            rows[:, 4:7] * velocity_scale @ rotation.T + velocity,
        ]
    )


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for n in (128, 1024):
            table = run(f"mkplummer - {n} seed=123 | snapprint - {COLUMNS}", work)
            write_gzip(out / f"plummer_{n}.tab.gz", table)

        mw = place(galaxy(1, work), 1.0, 1.0, MW_SPIN, MW_POSITION, MW_VELOCITY)
        m31 = place(
            galaxy(2, work),
            M31_MASS_SCALE,
            M31_LENGTH_SCALE,
            M31_SPIN,
            M31_POSITION,
            M31_VELOCITY,
        )
    disk, bulge = slice(0, N_DISK), slice(N_DISK, N_DISK + N_BULGE)
    halo = slice(N_DISK + N_BULGE, None)
    rows = np.vstack([g[part] for part in (disk, bulge, halo) for g in (mw, m31)])
    table = "".join(" ".join(f"{v:.7g}" for v in row) + "\n" for row in rows)
    write_gzip(out / "mw_m31.tab.gz", table)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
