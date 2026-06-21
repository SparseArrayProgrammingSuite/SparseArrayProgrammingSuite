from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from saps_framework.binsparse_format import BinsparseFormat

if TYPE_CHECKING:
    from .benchmark import DataInstance, Dataset, Generator


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

    def serialize_data(self, data: DataInstance) -> str:
        binsparse_list, meta = data
        binsparse_strings = [binsparse.serialize() for binsparse in binsparse_list]
        return json.dumps(
            {"binsparse": binsparse_strings, "meta": meta}, sort_keys=True, indent=2
        )

    def serialize_data_to_file(self, data: DataInstance, local_path: Path) -> None:
        os.makedirs(local_path.parent, exist_ok=True)
        with open(local_path, "w") as f:
            f.write(self.serialize_data(data))

    def deserialize_data(self, json_str: str) -> DataInstance:
        data = json.loads(json_str)
        binsparse_strings = data["binsparse"]
        meta = data["meta"]
        binsparse_list = [
            BinsparseFormat.deserialize(string) for string in binsparse_strings
        ]
        return (binsparse_list, meta)

    def deserialize_data_from_file(self, local_path: Path) -> DataInstance:
        with open(local_path) as f:
            return self.deserialize_data(f.read())

    def code_and_data_hash(
        self, generator: Generator, dataset: Dataset, data: DataInstance
    ) -> str:
        m = hashlib.sha256()
        m.update(self.serialize_data(data).encode("utf-8"))
        m.update(json.dumps(dataset.metadata, sort_keys=True).encode("utf-8"))
        #        m.update(generator.generate.__code__.co_code)
        print(f"Generator Name: {generator.name} Code and data hash: {m.hexdigest()}")
        return m.hexdigest()

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def _find_dataset_record(
        self, manifest: dict, generator: Generator, dataset: Dataset
    ) -> dict | None:
        for benchmark in manifest.get("benchmarks", []):
            for gen_record in benchmark.get("generators", []):
                if gen_record.get("name") != generator.name:
                    continue
                for dataset_record in gen_record.get("datasets", []):
                    if dataset_record.get("name") == dataset.name:
                        return dataset_record
        return None

    def _dataset_key(self, generator: Generator, dataset: Dataset) -> str:
        return f"{generator.name}.{dataset.name}"

    def update_manifest(
        self, generator: Generator, dataset: Dataset, digest: str
    ) -> None:
        manifest = self._read_manifest()
        manifest.setdefault("digests", {})[self._dataset_key(generator, dataset)] = (
            digest
        )
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def check_manifest(self, generator: Generator, dataset: Dataset) -> str | None:
        manifest = self._read_manifest()
        dataset_key = self._dataset_key(generator, dataset)
        if dataset_key in manifest.get("digests", {}):
            return manifest["digests"][dataset_key]

        dataset_record = self._find_dataset_record(manifest, generator, dataset)
        if dataset_record is not None:
            return dataset_record.get("digest")

        return None

    def upload_dataset(self, generator: Generator, dataset: Dataset) -> bool:
        data = generator.generate(dataset)
        digest = self.code_and_data_hash(generator, dataset, data)
        prefix = self.prefix(generator, dataset, digest)
        if self.file_exists(prefix):
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
            digest = self.code_and_data_hash(generator, dataset, data)
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
                return None
        data = self.deserialize_data_from_file(cache_path)
        assert digest == self.code_and_data_hash(generator, dataset, data), (
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
        import boto3

        super().__init__(manifest_path, cache_dir)
        # Accept either "s3://bucket" or plain "bucket"
        if bucket_name.startswith("s3://"):
            bucket_name = bucket_name[len("s3://") :]
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
        import botocore

        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=remote_prefix)
            return True
        except botocore.exceptions.ClientError:
            return False

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        try:
            self.s3.download_file(self.bucket_name, remote_prefix, str(local_path))
            return True
        except Exception as e:
            logging.error(f"Error downloading file from S3: {e}")
            return False


def build_storage_backend(
    type: str,
    bucket: str,
) -> StorageBackend:
    manifest_path = Path(os.environ["SAPS_MANIFEST_PATH"])
    cache_dir = Path(os.environ["SAPS_CACHE_DIR"])
    print(f"manifest_path: {manifest_path}")
    print(f"cache_dir: {cache_dir}")
    if type == "local":
        return LocalStorageBackend(bucket, manifest_path, cache_dir)
    if type == "s3":
        return S3StorageBackend(bucket, manifest_path, cache_dir)
    raise ValueError(f"Unsupported storage backend type: {type}")
