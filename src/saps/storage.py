from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
from binsparse import BinsparseTensor, HDF5BinsparseContainer

if TYPE_CHECKING:
    from saps.benchmark import DataInstance, Dataset, Generator

DEFAULT_REMOTE_STORAGE_BACKEND = "s3"
DEFAULT_REMOTE_STORAGE_BUCKET = "sparse-array-programming-suite"
DEFAULT_CACHE_DIR = ".saps/outputs/cache"
DEFAULT_MANIFEST_PATH = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_storage_bucket(bucket: str) -> str:
    if bucket.startswith("s3://"):
        bucket = bucket[len("s3://") :]
    return bucket.split("/", 1)[0]


class StorageBackend(ABC):
    def __init__(self, manifest_path: Path, cache_dir: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        """Upload a file to remote storage."""

    @abstractmethod
    def file_exists(self, remote_prefix: str) -> bool:
        """Check if a file exists in remote storage."""

    @abstractmethod
    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        """Download a file from remote storage."""

    def prefix(self, generator: Generator, dataset: Dataset, digest: str) -> str:
        return f"{generator.name}/{dataset.name}/{digest}.bsp.h5"

    def manifest_record_prefix(self, manifest_key: str, record: dict) -> str:
        generator_name, dataset_name = manifest_key.split(".", 1)
        return f"{generator_name}/{dataset_name}/{record['digest']}.bsp.h5"

    def manifest_record_exists(self, manifest_key: str, record: dict) -> bool:
        return self.file_exists(self.manifest_record_prefix(manifest_key, record))

    def uri_for_prefix(self, remote_prefix: str) -> str:
        return remote_prefix

    def serialize_data_to_file(self, data: DataInstance, local_path: Path) -> str:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(local_path, "w", track_order=True) as file:
            file.attrs["saps_version"] = 1
            file.attrs["meta"] = json.dumps(data.meta, sort_keys=True)
            if data.ref_meta is not None:
                file.attrs["ref_meta"] = json.dumps(data.ref_meta, sort_keys=True)
            for name, tensors in (
                ("inputs", data.inputs),
                ("ref_outputs", data.ref_outputs),
            ):
                if tensors is None:
                    continue
                parent = file.create_group(name, track_order=True)
                for index, tensor in enumerate(tensors):
                    tensor.serialize(
                        HDF5BinsparseContainer(parent.create_group(str(index)))
                    )
        return sha256_file(local_path)

    def deserialize_data_from_file(self, local_path: Path) -> DataInstance:
        from saps.benchmark import DataInstance

        with h5py.File(local_path, "r") as file:
            tensors = {
                name: (
                    [
                        BinsparseTensor.parse(HDF5BinsparseContainer(file[name][key]))
                        for key in sorted(file[name], key=int)
                    ]
                    if name in file
                    else None
                )
                for name in ("inputs", "ref_outputs")
            }
            return DataInstance(
                inputs=tensors["inputs"] or [],
                meta=json.loads(file.attrs["meta"]),
                ref_outputs=tensors["ref_outputs"],
                ref_meta=(
                    json.loads(file.attrs["ref_meta"])
                    if "ref_meta" in file.attrs
                    else None
                ),
            )

    def _generate_and_cache(
        self, generator: Generator, dataset: Dataset
    ) -> tuple[DataInstance, str, Path]:
        data = generator.generate(dataset)
        parent = self.cache_dir / generator.name / dataset.name
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".saps-", dir=parent) as staging:
            staging_path = Path(staging) / "data.bsp.h5"
            digest = self.serialize_data_to_file(data, staging_path)
            cache_path = self.cache_dir / self.prefix(generator, dataset, digest)
            staging_path.replace(cache_path)
        return data, digest, cache_path

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def _dataset_manifest_metadata(self, dataset: Dataset) -> dict:
        return {
            "file": dataset.file,
            "freshness": dataset.freshness,
        }

    def update_manifest(
        self, generator: Generator, dataset: Dataset, digest: str
    ) -> None:
        manifest = self._read_manifest()
        manifest[f"{generator.name}.{dataset.name}"] = {
            "digest": digest,
            **self._dataset_manifest_metadata(dataset),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def check_manifest(self, generator: Generator, dataset: Dataset) -> str | None:
        manifest = self._read_manifest()
        dataset_key = f"{generator.name}.{dataset.name}"
        if dataset_key not in manifest:
            return None
        record = manifest[dataset_key]
        if {
            key: record.get(key) for key in ("file", "freshness")
        } != self._dataset_manifest_metadata(dataset):
            logging.info(
                f"Dataset {generator.name}.{dataset.name} manifest metadata is stale."
            )
            return None
        return record["digest"]

    def upload_dataset(self, generator: Generator, dataset: Dataset) -> bool:
        work_log = logging.getLogger("saps.work")
        dataset_key = f"{generator.name}.{dataset.name}"
        work_log.info("caching %s", dataset_key)
        try:
            _, digest, local_path = self._generate_and_cache(generator, dataset)
            prefix = self.prefix(generator, dataset, digest)
            if self.file_exists(prefix):
                self.update_manifest(generator, dataset, digest)
                work_log.info("cached %s (already uploaded)", dataset_key)
                return True
            successful = self.upload_file(local_path, prefix)
            if successful:
                self.update_manifest(generator, dataset, digest)
            work_log.info("cached %s: %s", dataset_key, successful)
            return successful
        except Exception:
            work_log.exception("failed to cache %s", dataset_key)
            raise

    def retrieve_dataset(self, generator: Generator, dataset: Dataset) -> DataInstance:
        """Retrieve the dataset by, in order:
        1. The cache
        2. Remote Storage
        3. Generating it
        """
        digest = self.check_manifest(generator, dataset)
        if digest:
            prefix = self.prefix(generator, dataset, digest)
            cache_path = self.cache_dir / prefix
            cached = cache_path.exists()
            if cached:
                logging.info(f"Dataset {generator.name}.{dataset.name} found in cache.")
                return self.deserialize_data_from_file(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"Dataset {generator.name}.{dataset.name} not found in cache at "
                f"{cache_path}"
            )
            if self.download_file(prefix, cache_path):
                assert digest == sha256_file(cache_path), (
                    "Data integrity check failed: hash mismatch"
                )
                return self.deserialize_data_from_file(cache_path)
            logging.error(
                "Failed to download dataset "
                f"{generator.name}.{dataset.name} from remote storage."
            )

        data, digest, _ = self._generate_and_cache(generator, dataset)
        self.update_manifest(generator, dataset, digest)
        logging.info(f"Dataset {generator.name}.{dataset.name} regenerated.")
        return data


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: Path | str, manifest_path: Path, cache_dir: Path):
        super().__init__(manifest_path, cache_dir)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        dest_path = self.base_path / remote_prefix
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(local_path, dest_path)
            return True
        except OSError as e:
            logging.error(f"Error uploading file to local storage: {e}")
            return False

    def file_exists(self, remote_prefix: str) -> bool:
        return (self.base_path / remote_prefix).exists()

    def uri_for_prefix(self, remote_prefix: str) -> str:
        return str(self.base_path / remote_prefix)

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        source_path = self.base_path / remote_prefix
        if not source_path.exists():
            logging.error(f"File not found in local storage: {source_path}")
            return False
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, local_path)
            return True
        except OSError as e:
            logging.error(f"Error downloading file from local storage: {e}")
            return False


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket_name: str, manifest_path: Path, cache_dir: Path):
        import boto3
        from botocore import UNSIGNED

        Config = __import__("botocore.config", fromlist=["Config"]).Config

        super().__init__(manifest_path, cache_dir)
        self.bucket_name = normalize_storage_bucket(bucket_name)
        self.s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self._upload_s3 = None

    @property
    def upload_s3(self):
        if self._upload_s3 is None:
            import boto3

            self._upload_s3 = boto3.client("s3")
        return self._upload_s3

    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        import botocore

        try:
            self.upload_s3.upload_file(str(local_path), self.bucket_name, remote_prefix)
            return True
        except (
            botocore.exceptions.BotoCoreError,
            botocore.exceptions.ClientError,
        ) as e:
            logging.error(f"Error uploading file to S3: {e}")
            return False

    def file_exists(self, remote_prefix: str) -> bool:
        import botocore

        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=remote_prefix)
            return True
        except botocore.exceptions.ClientError:
            return False

    def uri_for_prefix(self, remote_prefix: str) -> str:
        return f"s3://{self.bucket_name}/{remote_prefix}"

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        import botocore

        try:
            self.s3.download_file(self.bucket_name, remote_prefix, str(local_path))
            return True
        except (
            botocore.exceptions.BotoCoreError,
            botocore.exceptions.ClientError,
        ) as e:
            logging.error(f"Error downloading file from S3: {e}")
            return False


def build_storage_backend(
    type: str | None = None,
    bucket: str | None = None,
    *,
    manifest_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> StorageBackend:
    backend_type = type or os.getenv("REMOTE_STORAGE_BACKEND")
    backend_type = backend_type or DEFAULT_REMOTE_STORAGE_BACKEND
    backend_bucket = bucket or os.getenv("REMOTE_STORAGE_BUCKET")
    backend_bucket = backend_bucket or DEFAULT_REMOTE_STORAGE_BUCKET
    manifest_path_resolved = Path(
        manifest_path or os.getenv("SAPS_MANIFEST_PATH") or DEFAULT_MANIFEST_PATH
    )
    cache_dir_resolved = Path(
        cache_dir or os.getenv("SAPS_CACHE_DIR") or DEFAULT_CACHE_DIR
    )
    if backend_type == "local":
        return LocalStorageBackend(
            backend_bucket, manifest_path_resolved, cache_dir_resolved
        )
    if backend_type == "s3":
        return S3StorageBackend(
            backend_bucket, manifest_path_resolved, cache_dir_resolved
        )
    raise ValueError(f"Unsupported storage backend type: {backend_type}")
