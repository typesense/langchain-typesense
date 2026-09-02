"""Pure validation and request-encoding helpers for the vector-store adapter."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import quote, urlparse

from langchain_core.documents import Document
from typesense.configuration import NodeConfigDict

from langchain_typesense._errors import (
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStoreError,
)
from langchain_typesense._types import Filter, VectorDistance

_FILTER_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_UNQUOTED_FILTER_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.~-]+$")


def validate_k(k: int) -> None:
    """Validate a requested result count."""
    if k < 0:
        raise ValueError("`k` must be greater than or equal to 0.")


def validate_id(document_id: str) -> None:
    """Validate an ID before it is placed in a Typesense URL or filter."""
    if not document_id:
        raise ValueError("Typesense document IDs must not be empty.")
    if quote(document_id, safe="") != document_id:
        raise ValueError(
            "Typesense document IDs must contain only URL-safe, unreserved "
            f"characters; got {document_id!r}."
        )


def resolve_ids(
    documents: Sequence[Document],
    ids: Sequence[str] | None,
) -> list[str]:
    """Return caller-supplied IDs or generate UUIDs for missing IDs."""
    if ids is not None and len(ids) != len(documents):
        raise ValueError(
            "The number of IDs must match the number of documents. "
            f"Got {len(ids)} IDs and {len(documents)} documents."
        )

    resolved_ids = (
        list(ids)
        if ids is not None
        else [document.id or str(uuid.uuid4()) for document in documents]
    )
    for document_id in resolved_ids:
        validate_id(document_id)
    return resolved_ids


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
) -> list[list[float]]:
    """Validate vector count, dimensions, and finite numeric values."""
    if len(vectors) != expected_count:
        raise ValueError(
            "Embedding model returned an unexpected number of vectors. "
            f"Got {len(vectors)} vectors for {expected_count} documents."
        )
    if not vectors:
        return []

    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("Embedding vectors must not be empty.")

    validated: list[list[float]] = []
    for index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise ValueError(
                "Embedding vectors must have equal dimensions. "
                f"Vector 0 has {dimension}; vector {index} has {len(vector)}."
            )
        converted = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError("Embedding vectors must contain only finite numbers.")
        validated.append(converted)
    return validated


def raise_for_import_failures(response: object) -> None:
    """Raise an integration error for malformed or failed bulk imports."""
    if not isinstance(response, list):
        raise TypesenseVectorStoreError("Typesense returned an unexpected bulk import response.")
    if any(
        not isinstance(item, Mapping) or not isinstance(item.get("success"), bool)
        for item in response
    ):
        raise TypesenseVectorStoreError(
            "Typesense returned a malformed record in its bulk import response."
        )
    failures = [
        cast(Mapping[str, Any], item)
        for item in response
        if isinstance(item, Mapping) and item.get("success") is False
    ]
    if failures:
        raise TypesenseImportError(failures)


def encode_filter_value(value: object) -> str:
    """Encode one scalar using Typesense filter literal syntax."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Filter numbers must be finite.")
        return repr(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported Typesense filter value: {value!r}")

    if _UNQUOTED_FILTER_VALUE_PATTERN.fullmatch(value):
        return value
    escaped = value.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def to_filter_by(filter_value: Filter, metadata_key: str) -> str:
    """Translate a metadata mapping or raw expression to Typesense syntax."""
    if isinstance(filter_value, str):
        return filter_value

    clauses: list[str] = []
    for key, value in filter_value.items():
        if not _FILTER_FIELD_PATTERN.fullmatch(key):
            raise ValueError(
                "Dictionary filter keys may contain only letters, numbers, `_`, "
                f"`-`, and `.`; got {key!r}. Use a raw Typesense filter string "
                "for other field names."
            )
        field_name = f"{metadata_key}.{key}"
        if isinstance(value, Sequence) and not isinstance(value, str):
            encoded = ",".join(encode_filter_value(item) for item in value)
            clauses.append(f"{field_name}:=[{encoded}]")
        else:
            clauses.append(f"{field_name}:={encode_filter_value(value)}")
    return " && ".join(clauses)


def validate_vector_query_options(
    distance_threshold: float | None,
    ef: int | None,
    flat_search_cutoff: int | None,
    vec_dist: VectorDistance,
) -> None:
    """Validate optional Typesense vector-search tuning parameters."""
    if distance_threshold is not None:
        if not math.isfinite(distance_threshold):
            raise ValueError("`distance_threshold` must be finite.")
        if vec_dist == "cosine" and distance_threshold < 0:
            raise ValueError("`distance_threshold` must be non-negative for cosine distance.")
    if ef is not None and ef <= 0:
        raise ValueError("`ef` must be greater than 0.")
    if flat_search_cutoff is not None and flat_search_cutoff < 0:
        raise ValueError("`flat_search_cutoff` must be greater than or equal to 0.")


def build_vector_query(
    embedding: Sequence[float],
    k: int,
    *,
    vector_key: str,
    vec_dist: VectorDistance,
    distance_threshold: float | None,
    ef: int | None,
    flat_search_cutoff: int | None,
    alpha: float | None = None,
) -> str:
    """Serialize an embedding and its Typesense vector-query options."""
    validate_vector_query_options(distance_threshold, ef, flat_search_cutoff, vec_dist)
    validated = validate_vectors([embedding], 1)[0]

    vector_options = [f"k:{k}"]
    if distance_threshold is not None:
        vector_options.append(f"distance_threshold:{distance_threshold!r}")
    if ef is not None:
        vector_options.append(f"ef:{ef}")
    if flat_search_cutoff is not None:
        vector_options.append(f"flat_search_cutoff:{flat_search_cutoff}")
    if alpha is not None:
        vector_options.append(f"alpha:{alpha!r}")

    vector = ",".join(repr(value) for value in validated)
    return f"{vector_key}:([{vector}], {', '.join(vector_options)})"


def normalize_hybrid_query_by(
    query_by: str | Sequence[str] | None,
    *,
    text_key: str,
    vector_key: str,
) -> str:
    """Return a comma-separated keyword-field list for hybrid search."""
    if query_by is None:
        fields = [text_key]
    elif isinstance(query_by, str):
        fields = [field.strip() for field in query_by.split(",")]
    else:
        fields = [field.strip() for field in query_by]
    if not fields or any(not field for field in fields):
        raise ValueError("`query_by` must contain at least one non-empty field name.")
    if vector_key in fields:
        raise ValueError(
            f"`query_by` must not contain vector field `{vector_key}` when "
            "the adapter supplies a manual `vector_query`."
        )
    return ",".join(fields)


def validate_hybrid_query(query: str, alpha: float) -> None:
    """Validate the keyword query and vector weight used for rank fusion."""
    if not query.strip():
        raise ValueError("Hybrid search `query` must not be empty.")
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("`alpha` must be a finite number between 0 and 1.")


def document_from_typesense(
    raw: Mapping[str, Any],
    *,
    text_key: str,
    vector_key: str,
    metadata_key: str,
) -> Document:
    """Deserialize one Typesense document into a LangChain document."""
    values = dict(raw)
    try:
        document_id = values.pop("id")
        page_content = values.pop(text_key)
    except KeyError as error:
        raise TypesenseCollectionError(
            "Stored document is missing the configured ID or text field."
        ) from error
    if not isinstance(document_id, str) or not document_id:
        raise TypesenseCollectionError("Stored document has an invalid ID field.")
    if not isinstance(page_content, str):
        raise TypesenseCollectionError(f"Stored document field `{text_key}` must be a string.")
    values.pop(vector_key, None)
    raw_metadata = values.pop(metadata_key, {})
    if not isinstance(raw_metadata, Mapping):
        raise TypesenseCollectionError(f"Field `{metadata_key}` must contain an object.")
    metadata = dict(values)
    metadata.update(raw_metadata)
    return Document(id=document_id, page_content=page_content, metadata=metadata)


def parse_search_response(
    response: object,
    *,
    text_key: str,
    vector_key: str,
    metadata_key: str,
    include_vectors: bool,
) -> list[tuple[Document, float, list[float] | None]]:
    """Convert Typesense hits to documents, distances, and optional vectors."""
    if not isinstance(response, Mapping):
        raise TypesenseVectorStoreError("Typesense search response must be an object.")
    if "hits" not in response:
        raise TypesenseVectorStoreError(
            "Typesense search response is missing `hits`; grouped or malformed "
            "responses are not supported."
        )
    hits = response["hits"]
    if not isinstance(hits, list):
        raise TypesenseVectorStoreError("Typesense search response has invalid `hits`.")

    results: list[tuple[Document, float, list[float] | None]] = []
    for raw_hit in hits:
        if not isinstance(raw_hit, Mapping) or not isinstance(raw_hit.get("document"), Mapping):
            raise TypesenseVectorStoreError("Typesense returned a malformed search hit.")

        raw_document = dict(cast(Mapping[str, Any], raw_hit["document"]))
        try:
            document_id = raw_document.pop("id")
            page_content = raw_document.pop(text_key)
            distance = float(raw_hit["vector_distance"])
        except (KeyError, TypeError, ValueError) as error:
            raise TypesenseVectorStoreError(
                "Typesense search hit is missing a valid ID, text, or vector distance."
            ) from error
        if not isinstance(document_id, str) or not document_id:
            raise TypesenseVectorStoreError(
                "Typesense search hit is missing a valid ID, text, or vector distance."
            )
        if not isinstance(page_content, str) or not math.isfinite(distance):
            raise TypesenseVectorStoreError(
                "Typesense search hit is missing a valid ID, text, or vector distance."
            )

        raw_metadata = raw_document.pop(metadata_key, {})
        if not isinstance(raw_metadata, Mapping):
            raise TypesenseCollectionError(f"Field `{metadata_key}` must contain an object.")

        raw_vector = raw_document.pop(vector_key, None)
        metadata = dict(raw_document)
        metadata.update(raw_metadata)

        vector: list[float] | None = None
        if include_vectors:
            if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str):
                raise TypesenseVectorStoreError(
                    f"Typesense search hit is missing vector field `{vector_key}`."
                )
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as error:
                raise TypesenseVectorStoreError(
                    f"Typesense search hit has an invalid vector field `{vector_key}`."
                ) from error
            if not vector or not all(math.isfinite(value) for value in vector):
                raise TypesenseVectorStoreError(
                    f"Typesense search hit has an invalid vector field `{vector_key}`."
                )

        results.append(
            (
                Document(id=document_id, page_content=page_content, metadata=metadata),
                distance,
                vector,
            )
        )
    return results


def parse_hybrid_search_response(
    response: object,
    *,
    text_key: str,
    vector_key: str,
    metadata_key: str,
) -> list[tuple[Document, float]]:
    """Convert Typesense hybrid hits to documents and rank-fusion scores."""
    if not isinstance(response, Mapping):
        raise TypesenseVectorStoreError("Typesense hybrid search response must be an object.")
    hits = response.get("hits")
    if not isinstance(hits, list):
        raise TypesenseVectorStoreError("Typesense hybrid search response is missing valid `hits`.")

    results: list[tuple[Document, float]] = []
    for raw_hit in hits:
        if not isinstance(raw_hit, Mapping) or not isinstance(raw_hit.get("document"), Mapping):
            raise TypesenseVectorStoreError("Typesense returned a malformed hybrid search hit.")
        raw_info = raw_hit.get("hybrid_search_info")
        if not isinstance(raw_info, Mapping):
            raise TypesenseVectorStoreError(
                "Typesense hybrid search hit is missing `hybrid_search_info`."
            )
        raw_score = raw_info.get("rank_fusion_score")
        if raw_score is None or isinstance(raw_score, bool):
            raise TypesenseVectorStoreError(
                "Typesense hybrid search hit has an invalid rank-fusion score."
            )
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise TypesenseVectorStoreError(
                "Typesense hybrid search hit has an invalid rank-fusion score."
            ) from error
        if not math.isfinite(score):
            raise TypesenseVectorStoreError(
                "Typesense hybrid search hit has an invalid rank-fusion score."
            )
        results.append(
            (
                document_from_typesense(
                    cast(Mapping[str, Any], raw_hit["document"]),
                    text_key=text_key,
                    vector_key=vector_key,
                    metadata_key=metadata_key,
                ),
                score,
            )
        )
    return results


def parse_export_response(response: object) -> list[Mapping[str, Any]]:
    """Parse the JSONL returned by Typesense's document export endpoint."""
    if not isinstance(response, str):
        raise TypesenseVectorStoreError(
            "Typesense returned an unexpected document export response."
        )
    documents: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(response.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise TypesenseVectorStoreError(
                f"Typesense returned invalid JSON on export line {line_number}."
            ) from error
        if not isinstance(raw, Mapping):
            raise TypesenseVectorStoreError(
                f"Typesense returned a non-object on export line {line_number}."
            )
        documents.append(raw)
    return documents


def ids_filter(ids: Sequence[str]) -> str:
    """Build a Typesense ID filter after validating every ID."""
    for document_id in ids:
        validate_id(document_id)
    return f"id:=[{','.join(ids)}]"


def node_from_url(typesense_url: str) -> NodeConfigDict:
    """Parse an absolute HTTP(S) URL into a Typesense node mapping."""
    if not typesense_url or not typesense_url.strip():
        raise ValueError("`typesense_url` is required and must not be empty.")
    parsed = urlparse(typesense_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("`typesense_url` must be an absolute http:// or https:// URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("`typesense_url` must not contain credentials, a query, or a fragment.")
    if parsed.path not in ("", "/"):
        raise ValueError("`typesense_url` must not contain a path.")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("`typesense_url` contains an invalid port.") from error
    if port is not None and port <= 0:
        raise ValueError("`typesense_url` contains an invalid port.")
    return {
        "host": parsed.hostname,
        "port": port if port is not None else (443 if parsed.scheme == "https" else 80),
        "protocol": parsed.scheme,
    }


__all__ = [
    "build_vector_query",
    "document_from_typesense",
    "encode_filter_value",
    "ids_filter",
    "node_from_url",
    "normalize_hybrid_query_by",
    "parse_export_response",
    "parse_hybrid_search_response",
    "parse_search_response",
    "raise_for_import_failures",
    "resolve_ids",
    "to_filter_by",
    "validate_hybrid_query",
    "validate_id",
    "validate_k",
    "validate_vector_query_options",
    "validate_vectors",
]
