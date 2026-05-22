from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
import pickle
from typing import TYPE_CHECKING
import boto3
import json
import hashlib
from pathlib import Path

import botocore

if TYPE_CHECKING:
    from .benchmark import DataInstance, Generator, Dataset


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
        return f"{generator.name}/{dataset.name}/{digest}.pkl"

    def serialize_data(self, data: DataInstance, local_path: Path) -> None:
        # TODO: don't use pickle, instead write a serializer for BinSparseFormat
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            pickle.dump(data, f)  

    def deserialize_data(self, local_path: Path) -> DataInstance:
        # TODO: don't use pickle, instead write a deserializer for BinSparseFormat
        with open(local_path, "rb") as f:
            return pickle.load(f)  # TODO: don't use pickle

    def code_and_data_hash(self, generator: Generator, dataset: Dataset, data: DataInstance) -> str:
        m = hashlib.sha256()
        m.update(pickle.dumps(data))
        m.update(pickle.dumps(dataset.__dict__))
        m.update(generator.generate.__code__.co_code)
        return m.hexdigest()

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def update_manifest(self, generator: Generator, dataset: Dataset, digest: str) -> None:
        manifest = self._read_manifest()
        dataset_key = f"{generator.name}.{dataset.name}"
        manifest[dataset_key] = {"digest": digest}
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def check_manifest(self, generator: Generator, dataset: Dataset) -> str | None:
        manifest = self._read_manifest()
        dataset_key = f"{generator.name}.{dataset.name}"
        if dataset_key not in manifest:
            return None
        return manifest[dataset_key]["digest"]

    def upload_dataset(self, generator: Generator, dataset: Dataset) -> bool:
        data = generator.generate(dataset)
        digest = self.code_and_data_hash(generator, dataset, data)
        prefix = self.prefix(generator, dataset, digest)
        if self.file_exists(prefix):
            return True
        local_path = self.cache_dir / prefix
        self.serialize_data(data, local_path)
        successful = self.upload_file(local_path, prefix)
        if successful:
            self.update_manifest(generator, dataset, digest)
        return successful

    def retrieve_dataset(self, generator: Generator, dataset: Dataset) -> DataInstance | None:
        """Retrieve the dataset by, in order:
            1. The cache
            2. Remote Storage
            3. Generating it
        """
        cacheable = generator.cacheable
        if not cacheable:
            return generator.generate(dataset)
        
        digest = self.check_manifest(generator, dataset)
        if not digest:
            data = generator.generate(dataset)
            digest = self.code_and_data_hash(generator, dataset, data)
            prefix = self.prefix(generator, dataset, digest)
            self.serialize_data(data, self.cache_dir / prefix)
            self.update_manifest(generator, dataset, digest)
            logging.info(f"Dataset {generator.name}.{dataset.name} regenerated.")
            return data

        prefix = self.prefix(generator, dataset, digest)
        cache_path = self.cache_dir / prefix
        if cache_path.exists():
            logging.info(f"Dataset {generator.name}.{dataset.name} found in cache.")
            logging.info(f"Manifest path: {self.manifest_path}")
        if not cache_path.exists():
            logging.info(f"Dataset {generator.name}.{dataset.name} not found in cache.")
            logging.info(f"Manifest path: {self.manifest_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.download_file(prefix, cache_path):
                logging.error(f"Failed to download dataset {generator.name}.{dataset.name} from remote storage.")
                return None
        data = self.deserialize_data(cache_path)
        assert digest == self.code_and_data_hash(generator, dataset, data), \
            "Data integrity check failed: hash mismatch"
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
            dest_path.write_bytes(local_path.read_bytes())
            return True
        except Exception as e:
            logging.error(f"Error uploading file to local storage: {e}")
            return False

    def file_exists(self, remote_prefix: str) -> bool:
        return os.path.exists(self.base_path / remote_prefix)

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        source_path = self.base_path / remote_prefix
        if not source_path.exists():
            logging.error(f"File not found in local storage: {source_path}")
            return False
        try:
            local_path.write_bytes(source_path.read_bytes())
            return True
        except Exception as e:
            logging.error(f"Error downloading file from local storage: {e}")
            return False


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket_name: str, manifest_path: Path, cache_dir: Path):
        super().__init__(manifest_path, cache_dir)
        # Accept either "s3://bucket" or plain "bucket"
        if bucket_name.startswith("s3://"):
            bucket_name = bucket_name[len("s3://"):]
        bucket_name = bucket_name.split("/", 1)[0]
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")

    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        try:
            self.s3.upload_file(str(local_path), self.bucket_name, remote_prefix)
            return True
        except Exception as e:
            logging.error(f"Error uploading file to S3: {e}")
            return False

    def file_exists(self, remote_prefix: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=remote_prefix)
            return True
        except botocore.exceptions.ClientError as e:
            return False

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        try:
            self.s3.download_file(self.bucket_name, remote_prefix, str(local_path))
            return True
        except Exception as e:
            logging.error(f"Error downloading file from S3: {e}")
            return False


def _repo_root() -> Path:
    """Find the repo root by walking up from cwd looking for pyproject.toml.

    Walks from the *cwd* (not __file__) so that the installed copy of storage.py
    inside an ASV-managed virtualenv still finds the host repo's manifest. The
    ASV runner sets cwd to a per-child tmpdir, so we also accept SAPS_REPO_ROOT
    as an explicit override that run_benchmark.py sets in env_nobuild.
    """
    env = os.environ.get("SAPS_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here

def build_storage_backend(
    type: str,
    bucket: str,
) -> StorageBackend:
    manifest_path = os.environ.get("SAPS_MANIFEST_PATH")
    cache_dir = os.environ.get("SAPS_CACHE_DIR")
    if type == "local":
        return LocalStorageBackend(bucket, manifest_path, cache_dir)
    elif type == "s3":
        return S3StorageBackend(bucket, manifest_path, cache_dir)
    else:
        raise ValueError(f"Unsupported storage backend type: {type}")