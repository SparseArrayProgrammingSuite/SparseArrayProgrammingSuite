from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from binsparse import BinsparseTensor

from saps.dependencies import dependency_versions
from saps_framework.binsparse_utils import deserialize, serialize

if TYPE_CHECKING:
    from saps.benchmark import DataInstance, Dataset, Generator

DEFAULT_REMOTE_STORAGE_BACKEND = "s3"
DEFAULT_REMOTE_STORAGE_BUCKET = "sparse-array-programming-suite"
DEFAULT_CACHE_DIR = ".saps/outputs/cache"
DEFAULT_MANIFEST_PATH = "manifest.json"


def normalize_storage_bucket(bucket: str) -> str:
    if bucket.startswith("s3://"):
        bucket = bucket[len("s3://") :]
    return bucket.split("/", 1)[0]


def manifest_record_prefix(manifest_key: str, digest: str) -> str:
    generator_name, dataset_name = manifest_key.split(".", 1)
    return f"{generator_name}/{dataset_name}/{digest}.json"


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
        return f"{generator.name}/{dataset.name}/{digest}.json"

    def manifest_record_prefix(self, manifest_key: str, record: dict) -> str:
        return manifest_record_prefix(manifest_key, record["digest"])

    def manifest_record_exists(self, manifest_key: str, record: dict) -> bool:
        return self.file_exists(self.manifest_record_prefix(manifest_key, record))

    def uri_for_prefix(self, remote_prefix: str) -> str:
        return remote_prefix

    def _serialize_binsparse_list(self, values: list[BinsparseTensor] | None):
        if values is None:
            return None
        return [serialize(tensor) for tensor in values]

    def serialize_data(self, data: DataInstance) -> str:
        return json.dumps(
            {
                "inputs": self._serialize_binsparse_list(data.inputs),
                "meta": data.meta,
                "ref_outputs": self._serialize_binsparse_list(data.ref_outputs),
                "ref_meta": data.ref_meta,
            },
            sort_keys=True,
            indent=2,
        )

    def serialize_data_to_file(self, data: DataInstance, local_path: Path) -> None:
        os.makedirs(local_path.parent, exist_ok=True)
        with open(local_path, "w") as f:
            f.write(self.serialize_data(data))

    def _deserialize_binsparse_list(self, values: list[str] | None):
        if values is None:
            return None
        return [deserialize(string) for string in values]

    def deserialize_data(self, json_str: str) -> DataInstance:
        from saps.benchmark import DataInstance

        data = json.loads(json_str)
        inputs = data.get("inputs", data.get("binsparse"))
        return DataInstance(
            inputs=self._deserialize_binsparse_list(inputs) or [],
            meta=data["meta"],
            ref_outputs=self._deserialize_binsparse_list(data.get("ref_outputs")),
            ref_meta=data.get("ref_meta"),
        )

    def deserialize_data_from_file(self, local_path: Path) -> DataInstance:
        with open(local_path) as f:
            return self.deserialize_data(f.read())

    def data_hash(self, data: DataInstance) -> str:
        m = hashlib.sha256()
        m.update(self.serialize_data(data).encode("utf-8"))
        return m.hexdigest()

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def _dataset_key(self, generator: Generator, dataset: Dataset) -> str:
        return f"{generator.name}.{dataset.name}"

    def _dataset_manifest_metadata(self, dataset: Dataset) -> dict:
        return {
            "file": dataset.file,
            "freshness": dataset.freshness,
            "dependencies": dataset.dependencies,
            "dependency_versions": dependency_versions(dataset.dependencies),
        }

    def update_manifest(
        self, generator: Generator, dataset: Dataset, digest: str
    ) -> None:
        manifest = self._read_manifest()
        manifest[self._dataset_key(generator, dataset)] = {
            "digest": digest,
            **self._dataset_manifest_metadata(dataset),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def check_manifest(self, generator: Generator, dataset: Dataset) -> str | None:
        manifest = self._read_manifest()
        dataset_key = self._dataset_key(generator, dataset)
        if dataset_key not in manifest:
            return None
        record = manifest[dataset_key]
        if {
            key: record.get(key)
            for key in ("file", "freshness", "dependencies", "dependency_versions")
        } != self._dataset_manifest_metadata(dataset):
            logging.info(
                f"Dataset {generator.name}.{dataset.name} manifest metadata is stale."
            )
            return None
        return record["digest"]

    def upload_dataset(self, generator: Generator, dataset: Dataset) -> bool:
        data = generator.generate(dataset)
        digest = self.data_hash(data)
        prefix = self.prefix(generator, dataset, digest)
        if self.file_exists(prefix):
            self.update_manifest(generator, dataset, digest)
            return True
        local_path = self.cache_dir / prefix
        self.serialize_data_to_file(data, local_path)
        successful = self.upload_file(local_path, prefix)
        if successful:
            self.update_manifest(generator, dataset, digest)
        return successful

    def retrieve_dataset(
        self, generator: Generator, dataset: Dataset
    ) -> DataInstance | None:
        """Retrieve the dataset by, in order:
        1. The cache
        2. Remote Storage
        3. Generating it
        """
        digest = self.check_manifest(generator, dataset)
        if not digest:
            data = generator.generate(dataset)
            digest = self.data_hash(data)
            prefix = self.prefix(generator, dataset, digest)
            self.serialize_data_to_file(data, self.cache_dir / prefix)
            self.update_manifest(generator, dataset, digest)
            logging.info(f"Dataset {generator.name}.{dataset.name} regenerated.")
            return data

        prefix = self.prefix(generator, dataset, digest)
        cache_path = self.cache_dir / prefix
        if cache_path.exists():
            logging.info(f"Dataset {generator.name}.{dataset.name} found in cache.")
            logging.info(f"Manifest path: {self.manifest_path}")
        if not cache_path.exists():
            logging.info(
                f"Dataset {generator.name}.{dataset.name} not found in cache at "
                f"{cache_path}"
            )
            logging.info(f"Manifest path: {self.manifest_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.download_file(prefix, cache_path):
                logging.error(
                    "Failed to download dataset "
                    f"{generator.name}.{dataset.name} from remote storage."
                )
                data = generator.generate(dataset)
                digest = self.data_hash(data)
                prefix = self.prefix(generator, dataset, digest)
                cache_path = self.cache_dir / prefix
                self.serialize_data_to_file(data, cache_path)
                self.update_manifest(generator, dataset, digest)
                logging.info(f"Dataset {generator.name}.{dataset.name} regenerated.")
                return data
        data = self.deserialize_data_from_file(cache_path)
        assert digest == self.data_hash(data), (
            "Data integrity check failed: hash mismatch"
        )
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
        except OSError as e:
            logging.error(f"Error uploading file to local storage: {e}")
            return False

    def file_exists(self, remote_prefix: str) -> bool:
        return os.path.exists(self.base_path / remote_prefix)

    def uri_for_prefix(self, remote_prefix: str) -> str:
        return str(self.base_path / remote_prefix)

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        source_path = self.base_path / remote_prefix
        if not source_path.exists():
            logging.error(f"File not found in local storage: {source_path}")
            return False
        try:
            local_path.write_bytes(source_path.read_bytes())
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
    backend_type = (
        type
        or os.environ.get("REMOTE_STORAGE_BACKEND")
        or DEFAULT_REMOTE_STORAGE_BACKEND
    )
    backend_bucket = (
        bucket
        or os.environ.get("REMOTE_STORAGE_BUCKET")
        or DEFAULT_REMOTE_STORAGE_BUCKET
    )
    manifest_path_value = manifest_path or os.environ.get("SAPS_MANIFEST_PATH")
    if manifest_path_value is None:
        manifest_path_value = DEFAULT_MANIFEST_PATH
    cache_dir_value = cache_dir or os.environ.get("SAPS_CACHE_DIR")
    if cache_dir_value is None:
        cache_dir_value = DEFAULT_CACHE_DIR

    manifest_path_resolved = Path(manifest_path_value)
    cache_dir_resolved = Path(cache_dir_value)
    print(f"manifest_path: {manifest_path_resolved}")
    print(f"cache_dir: {cache_dir_resolved}")
    if backend_type == "local":
        return LocalStorageBackend(
            backend_bucket, manifest_path_resolved, cache_dir_resolved
        )
    if backend_type == "s3":
        return S3StorageBackend(
            backend_bucket, manifest_path_resolved, cache_dir_resolved
        )
    raise ValueError(f"Unsupported storage backend type: {backend_type}")
