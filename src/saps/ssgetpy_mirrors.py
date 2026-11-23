"""Fall back to the University of Florida mirror when sparse.tamu.edu is down.

The SuiteSparse Matrix Collection moved from the University of Florida to Texas
A&M, but UF still serves a frozen copy under the same paths (``MM/``, ``mat/``,
``RB/`` and ``files/ssstats.csv``). That copy dates from 2015, so it lacks
matrices added since; it is only consulted when TAMU fails.

This module deliberately does not import ssgetpy, since importing ssgetpy
fetches the collection index and that fetch must already go through
:func:`index_fallback`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

TAMU_ROOT_URL = "https://sparse.tamu.edu"
UF_ROOT_URL = "https://www.cise.ufl.edu/research/sparse"
# (connect, read) seconds. ssgetpy sets no timeout, so a hung server would
# never count as down and the mirror would never be tried.
TIMEOUT = (10, 60)

# Captured at import so the fallback keeps working while index_fallback has
# replaced requests.get.
_requests_get = requests.get


def mirror_urls(url: str) -> list[str]:
    """Return *url* followed by its UF mirror when it points into TAMU."""
    prefix = TAMU_ROOT_URL + "/"
    if url.startswith(prefix):
        return [url, UF_ROOT_URL + "/" + url[len(prefix) :]]
    return [url]


def get(url: str, **kwargs: Any) -> requests.Response:
    """``requests.get`` that retries a failed TAMU request on the UF mirror.

    A request fails on a connection error, a timeout, or an HTTP error status.
    Raises :class:`requests.RequestException` naming every URL tried when all
    of them fail.
    """
    return _get(url, **kwargs)[0]


def _get(url: str, **kwargs: Any) -> tuple[requests.Response, bool]:
    """Like :func:`get`, also reporting whether the UF mirror served it."""
    kwargs.setdefault("timeout", TIMEOUT)
    candidates = mirror_urls(url)
    errors: list[str] = []
    for candidate in candidates:
        if errors:
            logger.warning(
                "SuiteSparse request to %s failed (%s); trying UF mirror %s",
                url,
                errors[-1],
                candidate,
            )
        try:
            response = _requests_get(candidate, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            errors.append(str(exc))
            continue
        return response, candidate != url
    tried = "\n".join(
        f"  {candidate}: {error}"
        for candidate, error in zip(candidates, errors, strict=True)
    )
    raise requests.RequestException(
        f"SuiteSparse request failed on every mirror:\n{tried}"
    )


@dataclass
class IndexFallback:
    """Records whether the index fetched during :func:`index_fallback` came from UF."""

    used_mirror: bool = False


@contextmanager
def index_fallback() -> Iterator[IndexFallback]:
    """Route TAMU requests made while importing ssgetpy through :func:`get`.

    ssgetpy downloads ``files/ssstats.csv`` from TAMU while it is being
    imported, whenever its cached index is empty or more than 90 days old, so
    there is no ssgetpy function to patch beforehand. Instead ``requests.get``
    is swapped for the duration of the import; requests for other hosts pass
    through unchanged.
    """
    state = IndexFallback()
    original = requests.get

    def get_with_fallback(
        url: str, params: Any = None, **kwargs: Any
    ) -> requests.Response:
        if len(mirror_urls(url)) == 1:
            return original(url, params=params, **kwargs)
        response, used_mirror = _get(url, params=params, **kwargs)
        state.used_mirror = state.used_mirror or used_mirror
        return response

    requests.get = get_with_fallback  # type: ignore[assignment]
    try:
        yield state
    finally:
        requests.get = original  # type: ignore[assignment]


def expire_index(instance: Any) -> None:
    """Mark ssgetpy's index stale so the next import retries TAMU.

    An index taken from the UF mirror is the 2015 snapshot. Left with a fresh
    timestamp, ssgetpy would keep it for 90 days and hide newer matrices even
    after TAMU recovers. Clearing the update log keeps the snapshot usable now
    while forcing a refresh attempt on the next import.
    """
    instance.conn.execute(f"DELETE FROM {instance.update_table}")
    instance.conn.commit()
