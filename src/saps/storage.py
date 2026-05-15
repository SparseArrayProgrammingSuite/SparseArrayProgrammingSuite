from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
from typing import TYPE_CHECKING, Any
import boto3
import json
import hashlib
from pathlib import Path

if TYPE_CHECKING:
    from .benchmark import DataInstance, Generator, Dataset


class StorageBackend(ABC):

    @abstractmethod
    def __init__(self, manifest_path: Path, cache_dir: Path, *args) -> None:
        pass

    @abstractmethod
    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        """Upload a file to remote storage."""
        pass

    @abstractmethod
    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        """Download a file from remote storage."""
        pass

    def prefix(self, generator: Generator, dataset: Dataset, digest: str) -> str:
        """Generate a unique prefix for the dataset based on the generator, dataset, and digest."""
        return f"{generator.name}/{dataset.name}/{digest}.pkl"
    
    def serialize_data(self, data: DataInstance, local_path: Path) -> None:
        with open(local_path, "wb") as f:
            pickle.dump(data, f) # TODO: don't use pickle

    def deserialize_data(self, local_path: Path) -> DataInstance:
        with open(local_path, "rb") as f:
            return pickle.load(f) # TODO: don't use pickle

    def code_and_data_hash(self, generator: Generator, dataset: Dataset, data: DataInstance) -> str:
        m = hashlib.sha256()
        m.update(pickle.dumps(data))
        m.update(pickle.dumps(dataset))
        m.update(pickle.dumps(generator.generate.__code__))
        return m.hexdigest()

    def update_manifest(self, generator: Generator, dataset: Dataset, digest: str) -> None:
        """Update the manifest file with the new dataset information."""
        manifest = json.loads(self.manifest_path)
        dataset_key = f"{generator.name}.{dataset.name}"
        manifest[dataset_key] = {
            "digest": digest,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f)

    def check_manifest(self, generator: Generator, dataset: Dataset) -> bool:
        """Check if the dataset is already uploaded according to the manifest."""
        manifest = json.loads(self.manifest_path)
        dataset_key = f"{generator.name}.{dataset.name}"
        if dataset_key not in manifest:
            return None
        else:
            return manifest[dataset_key]["digest"]

    def upload_dataset(self, generator: Generator, dataset: Dataset) -> bool:
        """Serialize generated data to a temp directory, then call upload_file."""
        digest = self.code_and_data_hash(generator, dataset, generator.generate(dataset))
        prefix = self.prefix(generator, dataset, digest)
        data = generator.generate(dataset)
        local_path = self.cache_dir / prefix
        self.serialize_data(data, local_path)
        sucessful = self.upload_file(local_path, prefix)
        if sucessful:
            self.update_manifest(generator, dataset, digest)
        return sucessful

    def retrieve_dataset(self, generator: Generator, dataset: Dataset) -> DataInstance:
        """Retrieve the dataset by, in order:
            1. The cache
            2. Remote Storage
            3. Generating it
        """
        digest = self.check_manifest(generator, dataset)
        if not digest:
            data = generator.generate(dataset)
            digest = self.code_and_data_hash(generator, dataset, data)
            self.serialize_data(data, self.cache_dir / prefix)
            self.update_manifest(generator, dataset, digest)
            return data
        
        prefix = self.prefix(generator, dataset, digest)
        cache_path = Path(self.cache_dir) / prefix
        data = None
        if not cache_path.exists():
            self.download_file(prefix, cache_path)
        data = self.deserialize_data(str(cache_path))
        assert digest == self.code_and_data_hash(generator, dataset, data), \
            "Data integrity check failed: hash mismatch"
        return data
    

class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: Path | str):
        self.base_path = Path(base_path)

    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        dest_path = self.base_path / remote_prefix
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest_path.write_bytes(local_path.read_bytes())
            return True
        except Exception as e:
            print(f"Error uploading file to local storage: {e}")
            return False

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        source_path = self.base_path / remote_prefix
        if not source_path.exists():
            print(f"File not found in local storage: {source_path}")
            return False
        try:
            local_path.write_bytes(source_path.read_bytes())
            return True
        except Exception as e:
            print(f"Error downloading file from local storage: {e}")
            return False

class S3StorageBackend(StorageBackend):
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")

    def upload_file(self, local_path: Path, remote_prefix: str) -> bool:
        try:
            self.s3.upload_file(str(local_path), self.bucket_name, remote_prefix)
            return True
        except Exception as e:
            print(f"Error uploading file to S3: {e}")
            return False

    def download_file(self, remote_prefix: str, local_path: Path) -> bool:
        try:
            self.s3.download_file(self.bucket_name, remote_prefix, str(local_path))
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {e}")
            return False
        
def build_storage_backend(type: str, bucket: str) -> StorageBackend:
    if type == "local":
        return LocalStorageBackend(bucket)
    elif type == "s3":
        return S3StorageBackend(bucket)
    else:
        raise ValueError(f"Unsupported storage backend type: {type}")