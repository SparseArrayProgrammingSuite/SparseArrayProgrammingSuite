from typing import Any

from saps.benchmark import Dataset


class SuiteSparseDataset(Dataset):
    """Base Dataset for benchmarks backed by a SuiteSparse Matrix Collection matrix."""

    def __init__(
        self,
        name: str,
        *,
        source_name: str | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        nnz: int | None = None,
        has_b_file: bool = False,
    ):
        self._name = name
        self.source_name = source_name if source_name is not None else name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []
        self.nnz = nnz
        self.has_b_file = has_b_file

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self._name

    @property
    def description(self) -> str:
        return self._description or f"SuiteSparse matrix {self.source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nnz"] = self.nnz
        data["has_b_file"] = self.has_b_file
        return data
