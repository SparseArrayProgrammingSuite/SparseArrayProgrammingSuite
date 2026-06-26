from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from saps.storage import S3StorageBackend


def _expired_token_error(operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "ExpiredToken",
                "Message": "The provided token has expired",
            }
        },
        operation,
    )


class _FakeS3Client:
    def __init__(
        self,
        name: str,
        calls: list[tuple[str, str]],
        *,
        fail_head: bool = False,
        fail_download: bool = False,
    ) -> None:
        self.name = name
        self.calls = calls
        self.fail_head = fail_head
        self.fail_download = fail_download

    def head_object(self, *, Bucket: str, Key: str) -> None:
        self.calls.append((self.name, f"head:{Bucket}/{Key}"))
        if self.fail_head:
            raise _expired_token_error("HeadObject")

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.calls.append((self.name, f"download:{bucket}/{key}"))
        if self.fail_download:
            raise _expired_token_error("GetObject")
        Path(filename).write_text("{}", encoding="utf-8")


def _backend_with_fake_clients(monkeypatch, tmp_path, signed, unsigned):
    def fake_client(service_name, *, config=None):
        assert service_name == "s3"
        return unsigned if config is not None else signed

    monkeypatch.setattr(boto3, "client", fake_client)
    return S3StorageBackend("s3://example-bucket", tmp_path / "manifest.json", tmp_path)


def test_s3_file_exists_checks_public_object_without_signed_credentials(
    monkeypatch, tmp_path
):
    calls: list[tuple[str, str]] = []
    signed = _FakeS3Client("signed", calls, fail_head=True)
    unsigned = _FakeS3Client("unsigned", calls)
    backend = _backend_with_fake_clients(monkeypatch, tmp_path, signed, unsigned)

    assert backend.file_exists("datasets/example.json")
    assert calls == [("unsigned", "head:example-bucket/datasets/example.json")]


def test_s3_download_reads_public_object_without_signed_credentials(
    monkeypatch, tmp_path
):
    calls: list[tuple[str, str]] = []
    signed = _FakeS3Client("signed", calls, fail_download=True)
    unsigned = _FakeS3Client("unsigned", calls)
    backend = _backend_with_fake_clients(monkeypatch, tmp_path, signed, unsigned)
    local_path = tmp_path / "downloaded.json"

    assert backend.download_file("datasets/example.json", local_path)
    assert local_path.read_text(encoding="utf-8") == "{}"
    assert calls == [("unsigned", "download:example-bucket/datasets/example.json")]
