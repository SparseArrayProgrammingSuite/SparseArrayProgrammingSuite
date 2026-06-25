from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from saps.dependencies import dependency_versions
from saps_framework.binsparse_format import BinsparseFormat

if TYPE_CHECKING:
    from saps.benchmark import DataInstance, Dataset, Generator


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

    def _serialize_binsparse_list(self, values: list[BinsparseFormat] | None):
        if values is None:
            return None
        return [binsparse.serialize() for binsparse in values]

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
        return [BinsparseFormat.deserialize(string) for string in values]

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
                f"Dataset {generator.name}.{dataset.name} manifest metadata "
                "is stale."
            )
            return None
        return record["digest"]

    def upload_dataset(self, generator: Generator, dataset: Dataset) -> bool:
        data = generator.generate(dataset)
        digest = self.code_and_data_hash(generator, dataset, data)
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
                data = generator.generate(dataset)
                digest = self.code_and_data_hash(generator, dataset, data)
                prefix = self.prefix(generator, dataset, digest)
                cache_path = self.cache_dir / prefix
                self.serialize_data_to_file(data, cache_path)
                self.update_manifest(generator, dataset, digest)
                logging.info(f"Dataset {generator.name}.{dataset.name} regenerated.")
                return data
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
        except OSError as e:
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
        except OSError as e:
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
        import botocore

        try:
            self.s3.upload_file(str(local_path), self.bucket_name, remote_prefix)
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
