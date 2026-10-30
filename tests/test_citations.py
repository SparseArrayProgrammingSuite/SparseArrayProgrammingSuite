import importlib
import inspect
import pkgutil
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import cast

import requests

import saps
import saps.benchmarks
from saps import Author, Ref

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

# The `arxiv` package's Client queries export.arxiv.org/api/query, whose
# id_list lookups (unlike common cached search_query lookups) currently hang
# or 500 rather than responding, and the package's session.get() call omits
# a timeout so a stall blocks forever (see arxiv/__init__.py:567 in
# arxiv==4.0.0, unchanged as of the project's current master). arXiv's own
# docs recommend the OAI-PMH interface for metadata harvesting instead; it
# hits a different backend (oaipmh.arxiv.org) and has proven reliable where
# the legacy query API has not, so we use it directly rather than going
# through the `arxiv` package.
_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}

# New-style arXiv IDs (post-2007) encode their original submission year and
# month as YYMM, e.g. "1508.03619" -> 2015-08. This is the v1 submission
# year, matching what benchmarks cite. The OAI record's created/updated
# dates are NOT a substitute: they track the latest revision (e.g. for
# 1508.03619 they reflect a 2017 revision, four years after the 2015 v1).
_NEW_STYLE_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")


def _arxiv_submission_year(arxiv_id: str) -> int | None:
    match = _NEW_STYLE_ARXIV_ID_RE.match(arxiv_id)
    return 2000 + int(match.group(1)) if match else None


def _oai_arxiv_ref(session: requests.Session, arxiv_id: str) -> Ref | None:
    bare_id = re.sub(r"v\d+$", "", arxiv_id)
    response = session.get(
        "https://export.arxiv.org/oai2",
        params={
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{bare_id}",
            "metadataPrefix": "arXiv",
        },
        timeout=10,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    if root.find("oai:error", _OAI_NS) is not None:
        return None
    record = root.find(".//arxiv:arXiv", _OAI_NS)
    if record is None:
        return None

    authors = []
    for author in record.findall("arxiv:authors/arxiv:author", _OAI_NS):
        name = " ".join(
            part
            for part in [
                author.findtext("arxiv:forenames", namespaces=_OAI_NS),
                author.findtext("arxiv:keyname", namespaces=_OAI_NS),
            ]
            if part
        )
        if name:
            authors.append(Author(name))

    year = _arxiv_submission_year(bare_id)
    if year is None:
        created = record.findtext("arxiv:created", default="", namespaces=_OAI_NS)
        year = int(created[:4]) if created[:4].isdigit() else None

    return Ref(
        title=(
            record.findtext("arxiv:title", default="", namespaces=_OAI_NS) or ""
        ).strip(),
        authors=authors,
        journal="Arxiv",
        volume=f"arXiv:{arxiv_id}",
        year=year,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _clean_reference_token(value: str) -> str:
    return value.rstrip(".,);]")


def _reference_doi(reference: str) -> str | None:
    match = _DOI_RE.search(reference)
    if match is None:
        return None
    return _clean_reference_token(match.group(0))


def _arxiv_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(
        r"arxiv\.org/(?:abs|pdf|html)/(?P<id>[^?#\s/]+)",
        url,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return re.sub(r"\.pdf$", "", match.group("id"), flags=re.IGNORECASE)


def _normalize_citation_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    text = re.sub(r"\s*:\s*", ":", text)
    return text.rstrip(".").casefold()


def _author_key(author: Author) -> str:
    words = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", author.name.casefold())
    return words[-1] if words else ""


def _first(values) -> str | None:
    if isinstance(values, list) and values:
        return values[0]
    if isinstance(values, str):
        return values
    return None


def _crossref_year(message: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = message.get(key, {}).get("date-parts")
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


def _crossref_authors(message: dict) -> list[Author]:
    authors = []
    for author in message.get("author", []):
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part
            for part in [author.get("given"), author.get("family")]
            if isinstance(part, str) and part
        )
        if not name and isinstance(author.get("name"), str):
            name = author["name"]
        if name:
            authors.append(Author(name))
    return authors


def _crossref_ref(message: dict) -> Ref:
    ref_type = message.get("type")
    container_title = _first(message.get("container-title"))
    kwargs = {}
    if ref_type == "journal-article" and container_title:
        kwargs["journal"] = container_title
    elif ref_type == "proceedings-article" and container_title:
        kwargs["booktitle"] = container_title
    elif container_title:
        kwargs["publisher"] = container_title

    if message.get("publisher") and "publisher" not in kwargs:
        kwargs["publisher"] = message["publisher"]

    return Ref(
        title=_first(message.get("title")) or "",
        authors=_crossref_authors(message),
        volume=message.get("volume"),
        number=message.get("issue"),
        pages=message.get("page"),
        year=_crossref_year(message),
        url=message.get("URL"),
        doi=message.get("DOI"),
        **kwargs,
    )


def _ref_constructor(ref: Ref) -> str:
    fields = [
        ("title", ref.title),
        ("authors", ref.authors),
        ("journal", ref.journal),
        ("conference", ref.conference),
        ("booktitle", ref.booktitle),
        ("publisher", ref.publisher),
        ("institution", ref.institution),
        ("volume", ref.volume),
        ("number", ref.number),
        ("pages", ref.pages),
        ("city", ref.city),
        ("year", ref.year),
        ("url", ref.url),
        ("doi", ref.doi),
    ]
    lines = ["Ref("]
    for name, value in fields:
        if value is None or value == []:
            continue
        if name == "authors":
            authors_value = cast(list[Author], value)
            authors = ", ".join(f"Author({author.name!r})" for author in authors_value)
            lines.append(f"    authors=[{authors}],")
        else:
            lines.append(f"    {name}={value!r},")
    lines.append(")")
    return "\n".join(lines)


def _effective_ref_doi(ref: Ref) -> str | None:
    if ref.doi:
        return ref.doi.lower()
    if ref.url:
        return _reference_doi(ref.url)
    return None


def _ref_mismatches(actual: Ref, expected: Ref) -> list[str]:
    mismatches = []
    if _normalize_citation_text(actual.title) != _normalize_citation_text(
        expected.title
    ):
        mismatches.append("title")
    if expected.year is not None and actual.year != expected.year:
        mismatches.append("year")

    actual_doi = _effective_ref_doi(actual)
    expected_doi = _effective_ref_doi(expected)
    if expected_doi and actual_doi and actual_doi.lower() != expected_doi.lower():
        mismatches.append("doi")

    actual_authors = [_author_key(author) for author in actual.authors]
    expected_authors = [_author_key(author) for author in expected.authors]
    if expected_authors and actual_authors != expected_authors:
        mismatches.append("authors")
    return mismatches


def _benchmark_instances():
    seen = set()
    for module_info in pkgutil.iter_modules(saps.benchmarks.__path__):
        module = importlib.import_module(f"saps.benchmarks.{module_info.name}")
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if cls in seen or inspect.isabstract(cls):
                continue
            if not issubclass(cls, saps.Benchmark) or cls is saps.Benchmark:
                continue
            seen.add(cls)
            try:
                yield cls()
            except (TypeError, ValueError):
                continue


def _references_by_owner() -> dict[str, tuple[Ref, list[str]]]:
    refs: dict[str, Ref] = {}
    owners = defaultdict(list)
    for benchmark in _benchmark_instances():
        for ref in benchmark.references:
            refs.setdefault(str(ref), ref)
            owners[str(ref)].append(benchmark.name)
        for generator in benchmark.generators:
            for ref in generator.references:
                refs.setdefault(str(ref), ref)
                owners[str(ref)].append(f"{benchmark.name} / {generator.name}")
    return {
        ref_string: (refs[ref_string], ref_owners)
        for ref_string, ref_owners in owners.items()
    }


def test_citations_match_crossref_or_arxiv():
    from habanero import Crossref
    from habanero.exceptions import RequestError
    from httpx2 import HTTPStatusError

    crossref_client = Crossref(mailto="ahrens@gatech.edu", timeout=10)
    arxiv_session = requests.Session()
    failures = []

    for ref, owners in _references_by_owner().values():
        doi = ref.doi or _reference_doi(ref.url or "")
        arxiv_id = _arxiv_id_from_url(ref.url)

        try:
            if doi:
                expected = _crossref_ref(crossref_client.works(ids=doi)["message"])
            elif arxiv_id:
                expected = _oai_arxiv_ref(arxiv_session, arxiv_id)
                if expected is None:
                    continue
            else:
                continue
        except (
            HTTPStatusError,
            RequestError,
            requests.exceptions.RequestException,
            ET.ParseError,
        ) as exc:
            failures.append(f"Could not fetch {ref}\nowners={owners}\n{exc}")
            continue

        mismatches = _ref_mismatches(ref, expected)
        if mismatches:
            failures.append(
                "\n".join(
                    [
                        f"Mismatch ({', '.join(mismatches)}): {ref}",
                        f"owners={owners}",
                        _ref_constructor(expected),
                    ]
                )
            )

    assert not failures, "\n\n".join(failures)
