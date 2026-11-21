"""Brain MRI scans from the IXI dataset."""

from __future__ import annotations

import gzip
import hashlib
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

from saps.benchmark import Author, Ref
from saps.downloaders.cache import download_lock, source_cache_dir

IXI_REF = Ref(
    title="IXI Dataset",
    authors=[Author("IXI - Information eXtraction from Images")],
    url="https://brain-development.org/ixi-dataset/",
)

_IXI_TAR_URL = (
    "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-{contrast}.tar"
)

# Byte offset, size and MD5 of each scan's .nii.gz file inside IXI's tar of
# its contrast, so a Range request fetches just that file. The tars are
# unchanged since 2014-10-07.
IXI_SCANS = {
    "IXI002-Guys-0828-T1": (512, 6943524, "b7873629cbb6eebfa8264179271da09c"),
    "IXI012-HH-1211-T2": (5938176, 6757589, "67061b12d36031d85d514c3cd18c6b18"),
    "IXI035-IOP-0873-PD": (135271424, 8291888, "a4e31c5f738c876423725ba7ca6f56a0"),
    "IXI035-IOP-0873-T1": (174050304, 11622897, "87acedea76837a0c81e85c324b8e86ce"),
}


def load_ixi_scan(scan: str, *, data_dir: str | Path | None = None) -> np.ndarray:
    """Return a scan's voxels as stored, int16 with slices first.

    The header's scale factor is not applied.
    """
    path = _ensure_downloaded(scan, data_dir)
    return read_nifti_int16(gzip.decompress(path.read_bytes()))


def read_nifti_int16(raw: bytes) -> np.ndarray:
    """Read the voxels of an uncompressed little-endian NIfTI-1 file of int16."""
    (sizeof_hdr,) = struct.unpack_from("<i", raw, 0)
    if sizeof_hdr != 348:
        raise ValueError("Not a little-endian NIfTI-1 file")
    dim = struct.unpack_from("<8h", raw, 40)
    (datatype,) = struct.unpack_from("<h", raw, 70)
    if datatype != 4:
        raise ValueError(f"Expected int16 voxels (NIfTI datatype 4), not {datatype}")
    (vox_offset,) = struct.unpack_from("<f", raw, 108)
    # NIfTI stores its first axis fastest, so in C order the slices come first.
    shape = tuple(reversed(dim[1 : dim[0] + 1]))
    voxels = np.frombuffer(
        raw, dtype="<i2", count=math.prod(shape), offset=int(vox_offset)
    )
    return voxels.reshape(shape).astype(np.int16)


def _ensure_downloaded(scan: str, data_dir: str | Path | None) -> Path:
    if scan not in IXI_SCANS:
        raise ValueError(f"Unknown IXI scan: {scan!r}")
    offset, size, md5 = IXI_SCANS[scan]
    root = Path(data_dir) if data_dir is not None else source_cache_dir("ixi")
    dest_path = root / f"{scan}.nii.gz"
    with download_lock(dest_path):
        if dest_path.exists():
            return dest_path
        contrast = scan.rsplit("-", 1)[1]
        request = urllib.request.Request(
            _IXI_TAR_URL.format(contrast=contrast),
            headers={"Range": f"bytes={offset}-{offset + size - 1}"},
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310
            # Anything but a partial response would be the whole tar.
            if response.status != 206:
                raise ValueError(f"IXI ignored the Range request for {scan}")
            data = response.read()
        if hashlib.md5(data, usedforsecurity=False).hexdigest() != md5:
            raise ValueError(f"{scan} does not match its MD5 {md5}")
        tmp_path = dest_path.with_name(dest_path.name + ".tmp")
        try:
            tmp_path.write_bytes(data)
            tmp_path.replace(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest_path
