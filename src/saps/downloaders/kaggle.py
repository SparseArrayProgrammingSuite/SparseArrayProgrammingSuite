"""Cache Kaggle source datasets in the shared SAPS cache."""

import tempfile
from pathlib import Path

from saps.downloaders.cache import download_lock, source_cache_dir


def download_kaggle_dataset(handle: str, *, data_dir: str | Path | None = None) -> Path:
    import kagglehub

    root = Path(data_dir) if data_dir is not None else source_cache_dir("kaggle")
    destination = root / handle
    with download_lock(destination):
        if destination.is_dir():
            return destination
        with tempfile.TemporaryDirectory(
            prefix=".saps-", dir=destination.parent
        ) as staging:
            output = Path(staging) / "dataset"
            kagglehub.dataset_download(handle, output_dir=str(output))
            output.replace(destination)
    return destination
