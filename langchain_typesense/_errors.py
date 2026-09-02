"""Exceptions raised by the Typesense vector-store integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class TypesenseVectorStoreError(RuntimeError):
    """Base class for malformed or unusable Typesense responses."""


class TypesenseCollectionError(TypesenseVectorStoreError):
    """Raised when a collection or stored document has an incompatible schema."""


class TypesenseImportError(TypesenseVectorStoreError):
    """Raised when Typesense reports one or more failed document imports.

    Typesense's bulk import endpoint can partially succeed. Inspect ``failures`` to
    identify rejected records; successful records are not rolled back.
    """

    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        """Build an error containing copies of the per-document failures."""
        self.failures = tuple(dict(failure) for failure in failures)
        details_parts: list[str] = []
        for failure in self.failures:
            raw_document = failure.get("document")
            document_id = (
                raw_document.get("id", "<unknown>")
                if isinstance(raw_document, Mapping)
                else "<unknown>"
            )
            details_parts.append(
                f"id={document_id!r}: {failure.get('error', 'unknown import error')}"
            )
        details = "; ".join(details_parts)
        super().__init__(
            f"Typesense rejected {len(self.failures)} document(s). "
            f"Other documents in the batch may have succeeded. {details}"
        )


__all__ = [
    "TypesenseCollectionError",
    "TypesenseImportError",
    "TypesenseVectorStoreError",
]
