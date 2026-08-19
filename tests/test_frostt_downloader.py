import gzip

import numpy as np

from saps.benchmarks.frostt import _TENSORS
from saps.downloaders.frostt import _RHS_DTYPES, _parse_tns


def _write_tns(path, contents: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(contents)


def test_rhs_dtype_registry_covers_every_frostt_dataset():
    assert set(_RHS_DTYPES) == {dataset.path for dataset in _TENSORS}


def test_rhs_dtype_registry_represents_frostt_value_kinds():
    assert _RHS_DTYPES["flickr/flickr-4d.tns.gz"] == np.bool_
    assert _RHS_DTYPES["reddit-2015/reddit-2015.tns.gz"] == np.int64
    assert _RHS_DTYPES["patents/patents.tns.gz"] == np.float64


def test_parse_tns_uses_requested_rhs_dtype_and_narrows_indices(tmp_path):
    path = tmp_path / "small.tns.gz"
    _write_tns(path, "1  2\t3.5\n2\t1   -4.25\n")

    indices, values, shape = _parse_tns(path, np.float32)

    assert shape == (2, 2)
    assert all(index.dtype == np.int32 for index in indices)
    assert values.dtype == np.float32


def test_parse_tns_keeps_int64_indices_for_large_dimensions(tmp_path):
    path = tmp_path / "large.tns.gz"
    _write_tns(path, "2147483648 1 7\n")

    indices, values, shape = _parse_tns(path, np.int16)

    assert shape == (2147483648, 1)
    assert all(index.dtype == np.int64 for index in indices)
    assert values.dtype == np.int16
