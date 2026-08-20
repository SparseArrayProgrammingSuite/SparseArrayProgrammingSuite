import os

import requests
from ssgetpy import bundle
from tqdm.auto import tqdm

_CHUNK_SIZE = 16 * 1024


def _patched_ssget_download(self, format="MM", destpath=None, extract=False):
    """
    Downloads this `Matrix` instance to the local machine,
    optionally unpacking any TAR.GZ files.
    """
    # destpath is the directory containing the matrix
    # It is of the form ~/.PyUFGet/MM/HB
    destpath = destpath or self._defaultdestpath(format)

    # localdest is matrix file (.MAT or .TAR.GZ)
    # if extract = True, localdestpath is the directory
    # containing the unzipped matrix
    localdestpath, localdest = self.localpath(format, destpath, extract)

    if not os.access(localdestpath, os.F_OK):
        # Create the destination path if necessary
        os.makedirs(destpath, exist_ok=True)

        response = requests.get(self.url(format), stream=True)
        content_length = int(response.headers["content-length"])

        with (
            open(localdest, "wb") as outfile,
            tqdm(total=content_length, desc=self.name, unit="B") as pbar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                outfile.write(chunk)
                pbar.update(_CHUNK_SIZE)

        if extract and (format == "MM" or format == "RB"):
            bundle.extract(localdest)

    return localdestpath, localdest
