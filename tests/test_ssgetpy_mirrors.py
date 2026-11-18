import logging
import sqlite3
from types import SimpleNamespace

import pytest

import requests

from saps import ssgetpy_mirrors

TAMU_INDEX = "https://sparse.tamu.edu/files/ssstats.csv"
UF_INDEX = "https://www.cise.ufl.edu/research/sparse/files/ssstats.csv"
TAMU_MATRIX = "https://sparse.tamu.edu/MM/HB/1138_bus.tar.gz"
UF_MATRIX = "https://www.cise.ufl.edu/research/sparse/MM/HB/1138_bus.tar.gz"


class _Response(SimpleNamespace):
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for url: {self.url}")


class _FakeServer:
    """Stands in for requests.get, answering each URL from a table."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        answer = self.answers.get(url, 404)
        if isinstance(answer, Exception):
            raise answer
        return _Response(url=url, status_code=answer)


@pytest.fixture
def server(monkeypatch):
    def install(answers):
        fake = _FakeServer(answers)
        monkeypatch.setattr(ssgetpy_mirrors, "_requests_get", fake)
        return fake

    return install


def test_mirror_urls_map_tamu_paths_onto_uf():
    assert ssgetpy_mirrors.mirror_urls(TAMU_MATRIX) == [TAMU_MATRIX, UF_MATRIX]
    assert ssgetpy_mirrors.mirror_urls(TAMU_INDEX) == [TAMU_INDEX, UF_INDEX]
    other = "https://example.com/MM/HB/1138_bus.tar.gz"
    assert ssgetpy_mirrors.mirror_urls(other) == [other]
    # A host that merely shares the prefix is not TAMU.
    lookalike = "https://sparse.tamu.edu.example.com/x"
    assert ssgetpy_mirrors.mirror_urls(lookalike) == [lookalike]


def test_get_uses_tamu_when_it_is_up(server):
    fake = server({TAMU_MATRIX: 200})

    response = ssgetpy_mirrors.get(TAMU_MATRIX, stream=True)

    assert response.url == TAMU_MATRIX
    assert fake.calls == [
        (TAMU_MATRIX, {"stream": True, "timeout": ssgetpy_mirrors.TIMEOUT})
    ]


@pytest.mark.parametrize(
    "failure",
    [403, 503, requests.ConnectionError("refused"), requests.Timeout("timed out")],
)
def test_get_falls_back_to_uf_when_tamu_is_down(server, caplog, failure):
    fake = server({TAMU_MATRIX: failure, UF_MATRIX: 200})

    with caplog.at_level(logging.WARNING, logger=ssgetpy_mirrors.__name__):
        response = ssgetpy_mirrors.get(TAMU_MATRIX, stream=True)

    assert response.url == UF_MATRIX
    assert [url for url, _ in fake.calls] == [TAMU_MATRIX, UF_MATRIX]
    assert fake.calls[1][1] == {"stream": True, "timeout": ssgetpy_mirrors.TIMEOUT}
    assert "trying UF mirror" in caplog.text


def test_get_names_every_mirror_when_all_fail(server):
    # E.g. a matrix added after the UF copy was frozen in 2015, while TAMU is down.
    server({TAMU_MATRIX: 403, UF_MATRIX: 404})

    with pytest.raises(requests.RequestException) as excinfo:
        ssgetpy_mirrors.get(TAMU_MATRIX)

    message = str(excinfo.value)
    assert f"{TAMU_MATRIX}: 403" in message
    assert f"{UF_MATRIX}: 404" in message


def test_get_does_not_retry_other_hosts(server):
    url = "https://example.com/data.tar.gz"
    fake = server({url: 500})

    with pytest.raises(requests.RequestException):
        ssgetpy_mirrors.get(url)

    assert [called for called, _ in fake.calls] == [url]


def test_index_fallback_reroutes_tamu_only_while_active(monkeypatch, server):
    fake = server({TAMU_INDEX: 403, UF_INDEX: 200})
    passthrough = []
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kwargs: passthrough.append((url, kwargs)) or "passthrough",
    )
    original = requests.get

    with ssgetpy_mirrors.index_fallback() as fallback:
        assert requests.get("https://example.com/other") == "passthrough"
        assert not fallback.used_mirror
        assert requests.get(TAMU_INDEX).url == UF_INDEX

    assert fallback.used_mirror
    assert requests.get is original
    assert passthrough == [("https://example.com/other", {"params": None})]
    assert [url for url, _ in fake.calls] == [TAMU_INDEX, UF_INDEX]


def test_index_fallback_restores_requests_get_on_error(server):
    server({TAMU_INDEX: 403, UF_INDEX: 403})
    original = requests.get

    with (
        pytest.raises(requests.RequestException),
        ssgetpy_mirrors.index_fallback(),
    ):
        requests.get(TAMU_INDEX)

    assert requests.get is original


def test_expire_index_clears_the_update_log():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE update_table (update_date TIMESTAMP)")
    conn.execute("INSERT INTO update_table VALUES (datetime('now'))")
    instance = SimpleNamespace(conn=conn, update_table="update_table")

    ssgetpy_mirrors.expire_index(instance)

    # ssgetpy treats an empty update log as an index last updated in 1970.
    assert conn.execute("SELECT MAX(update_date) FROM update_table").fetchone() == (
        None,
    )
