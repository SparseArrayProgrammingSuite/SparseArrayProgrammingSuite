import importlib
import inspect
import pkgutil
import re
from collections import defaultdict
from typing import cast

import pytest

import saps
import saps.benchmarks
from saps import Author, Ref

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


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


def _arxiv_ref(result, arxiv_id: str) -> Ref:
    return Ref(
        title=result.title,
        authors=[Author(author.name) for author in result.authors],
        journal="Arxiv",
        volume=f"arXiv:{arxiv_id}",
        year=result.published.year if result.published else None,
        url=f"https://arxiv.org/abs/{arxiv_id}",
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
    arxiv = pytest.importorskip("arxiv")
    from habanero import Crossref
    from habanero.exceptions import RequestError
    from httpx2 import HTTPStatusError

    crossref_client = Crossref(mailto="ahrens@gatech.edu", timeout=10)
    arxiv_client = arxiv.Client(page_size=1, delay_seconds=0, num_retries=2)
    failures = []

    for ref, owners in _references_by_owner().values():
        doi = ref.doi or _reference_doi(ref.url or "")
        arxiv_id = _arxiv_id_from_url(ref.url)

        try:
            if doi:
                expected = _crossref_ref(crossref_client.works(ids=doi)["message"])
            elif arxiv_id:
                result = next(
                    arxiv_client.results(
                        arxiv.Search(id_list=[arxiv_id], max_results=1)
                    ),
                    None,
                )
                if result is None:
                    continue
                expected = _arxiv_ref(result, arxiv_id)
            else:
                continue
        except (
            HTTPStatusError,
            RequestError,
            arxiv.ArxivError,
            arxiv.HTTPError,
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
