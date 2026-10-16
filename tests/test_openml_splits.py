import json
from types import SimpleNamespace

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import DataInstance
from saps.benchmarks.openml import (
    OpenMLDataset,
    OpenMLDatasetGenerator,
    _fetch_openml_task_split,
    _parse_split,
    fetch_openml_train_test_features,
)
from saps.storage import LocalStorageBackend


def split_arff(rows, *, sample=False):
    header = (
        "@relation splits\n@attribute type {TRAIN,TEST}\n@attribute rowid numeric\n"
        "@attribute repeat numeric\n@attribute fold numeric\n"
    )
    if sample:
        header += "@attribute sample numeric\n"
    return header + "@data\n" + rows


def dataset(**kwargs):
    return OpenMLDataset(
        "mnist",
        data_id=554,
        openml_name="mnist_784",
        version=1,
        task_id=3573,
        scale=255.0,
        **kwargs,
    )


def task_description(data_id=554):
    return json.dumps(
        {
            "task": {
                "task_id": "3573",
                "input": [
                    {"name": "source_data", "data_set": {"data_set_id": str(data_id)}},
                    {
                        "name": "estimation_procedure",
                        "estimation_procedure": {
                            "data_splits_url": "https://openml.org/splits.arff",
                        },
                    },
                ],
            }
        }
    )


def test_openml_downloader_caches_features_and_task_indices(monkeypatch, tmp_path):
    features = np.arange(8, dtype=np.uint8).reshape(4, 2)
    monkeypatch.setattr(
        "saps.benchmarks.openml._fetch_openml",
        lambda _: SimpleNamespace(
            data=features,
            details={"id": "554", "version": "1"},
        ),
    )
    downloads = {
        "https://www.openml.org/api/v1/json/task/3573": task_description(),
        "https://openml.org/splits.arff": split_arff(
            "TRAIN,2,0,0\nTEST,1,0,0\nTRAIN,0,0,0\nTRAIN,3,0,0\n"
            "TEST,0,0,1\nTRAIN,1,0,1\nTRAIN,2,0,1\nTRAIN,3,0,1\n"
        ),
    }
    monkeypatch.setattr("saps.benchmarks.openml._download_text", downloads.__getitem__)
    generator = OpenMLDatasetGenerator()
    source = dataset()
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    generator._backend = backend
    assert backend.upload_dataset(generator, source)
    monkeypatch.setattr(
        generator, "generate", lambda _: pytest.fail("Cache not reused")
    )
    prepared = generator.cached_generate(source)

    np.testing.assert_array_equal(
        to_numpy(prepared.inputs[0]), features.astype(np.float32) / 255
    )
    np.testing.assert_array_equal(to_numpy(prepared.inputs[1]), [2, 0, 3])
    np.testing.assert_array_equal(to_numpy(prepared.inputs[2]), [1])
    assert prepared.meta["task_id"] == 3573
    assert (
        prepared.meta["repeat"] == prepared.meta["fold"] == prepared.meta["sample"] == 0
    )
    assert prepared.meta["num_train"] == 3
    assert prepared.meta["num_test"] == 1
    assert prepared.meta["num_rows"] == 4
    assert "num_rows" not in source.metadata


def test_openml_task_rejects_wrong_source_before_downloading_splits(monkeypatch):
    urls = []

    def download(url):
        urls.append(url)
        return task_description(data_id=999)

    monkeypatch.setattr("saps.benchmarks.openml._download_text", download)
    with pytest.raises(ValueError, match="unexpected source"):
        _fetch_openml_task_split(dataset(), num_rows=4)
    assert len(urls) == 1


def test_openml_split_selects_repeat_fold_and_sample():
    text = split_arff(
        "TEST,0,0,0,0\nTRAIN,0,1,2,0\n"
        "TRAIN,3,1,2,1\nTEST,2,1,2,1\nTRAIN,0,1,2,1\nTRAIN,1,1,2,1\n",
        sample=True,
    )
    train, test = _parse_split(text, dataset(repeat=1, fold=2, sample=1), num_rows=4)
    np.testing.assert_array_equal(train, [3, 0, 1])
    np.testing.assert_array_equal(test, [2])
    assert train.dtype == test.dtype == np.int64


@pytest.mark.parametrize(
    "rows",
    [
        "TRAIN,0,0,0\nTRAIN,1,0,0\nTEST,1,0,0\nTEST,3,0,0\n",  # overlap
        "TRAIN,0,0,0\nTRAIN,0,0,0\nTEST,2,0,0\nTEST,3,0,0\n",  # duplicate
        "TRAIN,0,0,0\nTRAIN,1,0,0\nTEST,2,0,0\n",  # incomplete
        "TRAIN,-1,0,0\nTRAIN,1,0,0\nTEST,2,0,0\nTEST,3,0,0\n",  # negative
        "TRAIN,4,0,0\nTRAIN,1,0,0\nTEST,2,0,0\nTEST,3,0,0\n",  # out of range
        "TRAIN,0.5,0,0\nTRAIN,1,0,0\nTEST,2,0,0\nTEST,3,0,0\n",  # fractional
        "TRAIN,0,0,1\nTRAIN,1,0,1\nTEST,2,0,1\nTEST,3,0,1\n",  # missing fold
        "TRAIN,0,0,0\nTRAIN,1,0,0\nTRAIN,2,0,0\nTRAIN,3,0,0\n",  # no queries
    ],
)
def test_openml_split_rejects_invalid_partition(rows):
    with pytest.raises(ValueError, match="must partition"):
        _parse_split(split_arff(rows), dataset(), num_rows=4)


def test_openml_split_rejects_missing_sample():
    with pytest.raises(ValueError, match="has no sample 1"):
        _parse_split(split_arff("TRAIN,0,0,0\n"), dataset(sample=1), num_rows=4)


def test_openml_query_fetch_rejects_legacy_cache(monkeypatch):
    monkeypatch.setattr(
        "saps.benchmarks.openml.fetch_openml_dataset",
        lambda _: DataInstance(
            inputs=[from_numpy(np.zeros((4, 2)))],
            meta={},
        ),
    )
    with pytest.raises(RuntimeError, match="--cache-datasets"):
        fetch_openml_train_test_features("mnist")
