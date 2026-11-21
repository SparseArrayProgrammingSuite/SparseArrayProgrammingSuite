"""Downloader utilities for Model Counting Competition instances.

Two sources: the MC 2025 test instances on GitHub, fetched one file at a time,
and the MC 2020-2024 Track 1 instance archives on Zenodo, fetched and unpacked
whole.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
from pathlib import Path

from saps.downloaders.cache import download_lock, source_cache_dir

MCCOMP_REPOSITORY_URL = "https://github.com/arijitsh/mccomp-test-instances"
MCCOMP_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/arijitsh/mccomp-test-instances/main"
)

MCCOMP_TRACKS = {
    "Track1_MC": ("random_mc", "mc", "exact model counting"),
    "Track3_PMC": ("random_pmc", "pmc", "projected model counting"),
    "Track4_PWMC": ("random_pwmc", "pwmc", "projected weighted model counting"),
    "Track5B_AMC": ("random_amc", "amc", "algebraic model counting"),
}

MCCOMP_SOURCE_PATHS = tuple(
    f"{track}/{prefix}_{index}.cnf"
    for track, (prefix, _problem_type, _description) in MCCOMP_TRACKS.items()
    for index in range(1, 11)
)

# The archives the MC 2020-2024 organizers published on Zenodo for Track 1
# (model counting), Track 2 (weighted model counting) and Track 4 (projected
# weighted model counting): each year's competition instances and the full pool
# of instances they collected, where they published one. MC 2021's Track 4
# archive holds plain model counting instances ("c t mc"), so it is left out, as
# are MC 2024's Track 2b ("bonus") archives. Each maps to its Zenodo record and
# the MD5 that Zenodo lists for the file. An archive unpacks into a directory of
# its own name.
MCCOMP_POOL_ARCHIVES = {
    "mc2020-track1-mc_competition": ("10031810", "604e5d0a1da757fc763810782ffe0092"),
    "mc2021-track1-mc_competition": ("10012857", "b12ca80b995e51bb90a151ff629a1a91"),
    "mc2022-track1-mc_competition": ("10012860", "f47ae7ec395fc2a89120799c3a520d55"),
    "mc2023-track1-mc_competition": ("10012864", "1d1e8097c997aa47fc7f4a2a2a1de66c"),
    "mc2024-track1-mc_competition": ("14249068", "2818d425fab0d89e666237241ca610bc"),
    "mc2020-track1-mc_collected": ("10004947", "c69a07dc1f1fed5cdbc434d005ea5752"),
    "mc2021-track1-mc_collected": ("10006441", "74463335843cdbec64a5d0993fd66518"),
    "mc2022-track1-mc_collected": ("10014715", "afe4d1d517c12f018f17e08b85aa27d1"),
    "mc2023-track1-mc_collected": ("10012822", "0519f1ae0003c68ec06ff217259a8be2"),
    "mc2024-track1-mc_collected": ("14969231", "d2976e44c6a678404c01da779cb09007"),
    "mc2020-track2-wmc_competition": ("10031810", "fbef4c39410bb98ecdb439d7029f664a"),
    "mc2021-track2-wmc_competition": ("10012857", "d65be0c5af8015ddd6aeca9c72d2a783"),
    "mc2022-track2-wmc_competition": ("10012860", "02ae27769ac795f3dfc64186f6a141fd"),
    "mc2023-track2-wmc_competition": ("10012864", "6b605e3ccf958dea7905f17e49eed91a"),
    "mc2024-track2-wmc_competition": ("14249068", "1c7e6279eaf29bf3731cebe07eedf107"),
    "mc2022-track4-pwmc_competition": ("10012860", "019d2c1ea41f685d986b5dfde9954dec"),
    "mc2023-track4-pwmc_competition": ("10012864", "12a5f3ca1f9cb83f3db1d0262d83bbfd"),
    "mc2024-track4-pwmc_competition": ("14249068", "aa8edaeeb04132dee5693f1cd7eb9d8a"),
    "mc2020-track2-wmc_collected": ("10004947", "418f772ab6a56f9cc591989574fe41f4"),
    "mc2024-track2-wmc_collected": ("14969231", "5c13508b51c6d4beaa7c611d4bb2cfa0"),
}


def parse_dimacs(text: str) -> tuple[int, list[list[int]]]:
    lines = [line.strip() for line in text.splitlines()]
    cleaned = [line for line in lines if line and not line.startswith("c")]

    header_index = None
    num_vars = 0
    num_clauses = 0
    for i, line in enumerate(cleaned):
        parts = line.split()
        if len(parts) >= 4 and parts[:2] == ["p", "cnf"]:
            if len(parts) != 4:
                raise ValueError(f"Malformed DIMACS problem line: {line!r}")
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
            if num_vars < 0 or num_clauses < 0:
                raise ValueError(
                    "DIMACS variable and clause counts must be nonnegative"
                )
            header_index = i
            break

    if header_index is None:
        raise ValueError("DIMACS input is missing a 'p cnf' problem line")

    clauses: list[list[int]] = []
    current_clause: list[int] = []
    tokens = " ".join(cleaned[header_index + 1 :]).split()
    for token in tokens:
        if token == "%":
            break

        literal = int(token)
        if literal == 0:
            clauses.append(current_clause)
            current_clause = []
        else:
            if abs(literal) > num_vars:
                raise ValueError(
                    f"DIMACS literal {literal} exceeds declared variable count "
                    f"{num_vars}"
                )
            current_clause.append(literal)

    if current_clause:
        raise ValueError("DIMACS clause is missing a terminating 0")

    if len(clauses) != num_clauses:
        raise ValueError(
            f"DIMACS declared {num_clauses} clauses but parsed {len(clauses)}"
        )

    return num_vars, clauses


def list_mccomp_instances(track: str | None = None) -> list[str]:
    """Return known source paths from the MC competition test-instance repo."""
    if track is None:
        return list(MCCOMP_SOURCE_PATHS)

    track_dir = normalize_mccomp_track(track)
    return [path for path in MCCOMP_SOURCE_PATHS if path.startswith(f"{track_dir}/")]


def download_mccomp_instance(
    source_name: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Download one MC competition CNF instance and return its local path."""
    source_path = normalize_mccomp_source_path(source_name)
    return _ensure_downloaded(source_path, data_dir)


def mccomp_source_url(source_path: str) -> str:
    return f"{MCCOMP_REPOSITORY_URL}/blob/main/{source_path}"


def mccomp_raw_url(source_path: str) -> str:
    return f"{MCCOMP_RAW_BASE_URL}/{source_path}"


def normalize_mccomp_track(track: str) -> str:
    lowered = track.lower()
    for track_dir, (_prefix, problem_type, _description) in MCCOMP_TRACKS.items():
        if lowered in {track_dir.lower(), problem_type.lower()}:
            return track_dir
    raise ValueError(f"Unknown MC competition track: {track!r}")


def normalize_mccomp_source_path(source_name: str) -> str:
    name = source_name.strip().removeprefix("mccomp://").removeprefix("mccomp:")
    name = name.removesuffix(".cnf") + ".cnf"
    if name in MCCOMP_SOURCE_PATHS:
        return name
    matches = [path for path in MCCOMP_SOURCE_PATHS if Path(path).name == name]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous MC competition instance name: {source_name!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.cnf", name):
        raise ValueError(f"Invalid MC competition source path: {source_name!r}")
    return name


def download_mccomp_pool_instance(
    source_path: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Unpack the Zenodo archive holding a pool instance and return its file.

    ``source_path`` starts with the archive's name, as in
    ``mc2021-track1-mc_collected/MC2021_Track1_Set3_satlib_Hoos/cnf/uf/uf20-0661.cnf.xz``.
    """
    archive = source_path.split("/", 1)[0]
    directory = download_mccomp_pool_archive(archive, data_dir=data_dir)
    path = directory.parent / source_path
    if not path.is_file():
        raise FileNotFoundError(f"{archive} has no instance {source_path!r}")
    return path


def download_mccomp_pool_archive(
    archive: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Download and unpack one Zenodo archive, returning the directory it fills."""
    if archive not in MCCOMP_POOL_ARCHIVES:
        raise ValueError(f"Unknown MC competition archive: {archive!r}")
    _record, md5 = MCCOMP_POOL_ARCHIVES[archive]
    root = (Path(data_dir) if data_dir is not None else _default_data_dir()) / "zenodo"
    tar_path = root / f"{archive}.tar"
    marker = root / f".extracted-{archive}-{md5}"
    with download_lock(tar_path):
        if marker.exists():
            return root / archive
        if not tar_path.exists() or _md5_file(tar_path) != md5:
            tmp_path = tar_path.with_name(tar_path.name + ".tmp")
            try:
                urllib.request.urlretrieve(mccomp_pool_archive_url(archive), tmp_path)  # noqa: S310
                if _md5_file(tmp_path) != md5:
                    raise ValueError(f"{archive}.tar does not match Zenodo's MD5 {md5}")
                tmp_path.replace(tar_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        with tarfile.open(tar_path) as tar:
            tar.extractall(root, filter="data")
        # The 2020-2023 pools pack each submitted benchmark set as a tar of its
        # own, whose top directory is the set's name. Some sit beside macOS
        # AppleDouble files (._<name>.tar), which are not tars.
        for inner in sorted((root / archive).glob("*.tar")):
            if inner.name.startswith("._"):
                continue
            with tarfile.open(inner) as tar:
                tar.extractall(inner.parent, filter="data")
            inner.unlink()
        marker.touch()
    return root / archive


def mccomp_pool_record_url(source_path: str) -> str:
    record, _md5 = MCCOMP_POOL_ARCHIVES[source_path.split("/", 1)[0]]
    return f"https://zenodo.org/records/{record}"


def mccomp_pool_archive_url(archive: str) -> str:
    record, _md5 = MCCOMP_POOL_ARCHIVES[archive]
    return f"https://zenodo.org/api/records/{record}/files/{archive}.tar/content"


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_data_dir() -> Path:
    return source_cache_dir("mccomp")


def _ensure_downloaded(source_path: str, data_dir: str | Path | None) -> Path:
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dest_path = root / source_path
    with download_lock(dest_path):
        if dest_path.exists():
            return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_name(dest_path.name + ".tmp")
        try:
            urllib.request.urlretrieve(mccomp_raw_url(source_path), tmp_path)  # noqa: S310
            tmp_path.replace(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest_path
