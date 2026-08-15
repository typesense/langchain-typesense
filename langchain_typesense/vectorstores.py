"""Typesense-backed vector storage for LangChain.

The integration stores each :class:`~langchain_core.documents.Document` as a
Typesense document with three configurable fields: text, vector, and nested
metadata. Collections are created lazily from the first batch of embeddings and
are validated on subsequent writes. Search uses Typesense cosine distance; the
``*_with_score`` methods therefore return a raw distance where lower is better,
while LangChain's relevance-score adapter maps that distance to ``[0, 1]``.

Use :meth:`TypesenseVectorStore.from_client_params` for a convenience
constructor, or pass already configured synchronous and asynchronous clients to
:class:`TypesenseVectorStore` when connection lifecycle is managed by the
application.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeAlias, cast
from urllib.parse import quote

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from typesense import AsyncClient, Client
from typesense.configuration import ConfigDict
from typesense.exceptions import ObjectAlreadyExists, ObjectNotFound
from typesense.types.collection import CollectionCreateSchema, CollectionSchema
from typesense.types.document import (
    DeleteQueryParameters,
    DocumentSchema,
    DocumentWriteParameters,
    SearchParameters,
)

DEFAULT_COLLECTION_NAME = "langchain-typesense"
DEFAULT_TEXT_KEY = "text"
DEFAULT_VECTOR_KEY = "vec"
DEFAULT_METADATA_KEY = "metadata"

FilterScalar: TypeAlias = str | int | float | bool
FilterValue: TypeAlias = FilterScalar | Sequence[FilterScalar]
Filter: TypeAlias = str | Mapping[str, FilterValue]

_FILTER_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_RESERVED_SEARCH_PARAMETERS = frozenset(
    {"q", "vector_query", "per_page", "page", "filter_by", "exclude_fields"}
)


class TypesenseVectorStoreError(RuntimeError):
    """Base class for runtime failures defined by this integration.

    Direct instances report malformed or otherwise unusable Typesense responses;
    subclasses describe collection and bulk-import failures. Typesense client and
    embedding errors propagate unchanged so callers retain the originating type.
    """


class TypesenseCollectionError(TypesenseVectorStoreError):
    """Raised when a collection has an incompatible schema.

    A collection must contain the configured text and vector fields, with the
    vector dimension and cosine distance matching the store. The metadata field
    is required and must be indexed when ``index_metadata=True``; when present,
    it must be a nested object field.
    """


class TypesenseImportError(TypesenseVectorStoreError):
    """Raised when Typesense reports one or more failed document imports.

    Typesense's bulk import endpoint can partially succeed. Inspect ``failures`` to
    identify records that were rejected. Successfully imported records are not rolled
    back by Typesense.
    """

    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        """Build an error containing the per-document Typesense failures.

        Args:
            failures: Failed records returned by Typesense's bulk import API.
                The mappings are copied so the exception remains inspectable even
                if the caller reuses its response object.
        """
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


class TypesenseVectorStore(VectorStore):
    """LangChain vector store backed by Typesense.

    LangChain metadata is kept under one nested object field. This prevents metadata
    keys such as ``id`` or ``text`` from overwriting Typesense's internal fields.
    Metadata is indexed by default so dictionary filters can target nested values.
    Set ``index_metadata=False`` if metadata only needs to be stored and returned.

    Args:
        client: Configured synchronous Typesense client.
        embedding: Embedding model used for documents and queries.
        collection_name: Typesense collection name.
        async_client: Optional configured asynchronous Typesense client. When omitted,
            LangChain async methods use a worker thread around the sync client.
        text_key: Field used to store document text.
        vector_key: Field used to store embeddings.
        metadata_key: Nested object field used to store document metadata.
        index_metadata: Whether a newly created collection indexes metadata fields.
            Dictionary filters require indexed metadata. Raw Typesense filters remain
            available for user-managed collection fields.
    """

    def __init__(
        self,
        client: Client,
        embedding: Embeddings,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        async_client: AsyncClient | None = None,
        text_key: str = DEFAULT_TEXT_KEY,
        vector_key: str = DEFAULT_VECTOR_KEY,
        metadata_key: str = DEFAULT_METADATA_KEY,
        index_metadata: bool = True,
    ) -> None:
        """Initialize a store without making network requests.

        Collection creation and schema validation are deferred until a
        non-empty write, when the embedding dimension is known. Searches use an
        existing collection as-is and return no results when it is absent.

        Args:
            client: Configured synchronous Typesense client.
            embedding: Embedding model used for document and query embeddings.
            collection_name: Typesense collection name.
            async_client: Optional asynchronous client. If omitted, asynchronous
                store methods execute their synchronous counterparts in a worker
                thread.
            text_key: Field used to store document text.
            vector_key: Field used to store document embeddings.
            metadata_key: Nested object field used to store document metadata.
            index_metadata: Whether metadata fields in a newly created collection
                should be indexed, enabling dictionary filters.

        Raises:
            ValueError: If the collection or field names are invalid or collide
                with one another or Typesense's reserved ``id`` field.
        """
        field_names = (text_key, vector_key, metadata_key)
        if not collection_name:
            raise ValueError("`collection_name` must not be empty.")
        if any(not field_name for field_name in field_names):
            raise ValueError("Typesense field names must not be empty.")
        if len(set(field_names)) != len(field_names) or "id" in field_names:
            raise ValueError(
                "`text_key`, `vector_key`, and `metadata_key` must be distinct and "
                "must not equal Typesense's reserved `id` field."
            )

        self._client = client
        self._async_client = async_client
        self._embedding = embedding
        self._collection_name = collection_name
        self._text_key = text_key
        self._vector_key = vector_key
        self._metadata_key = metadata_key
        self._index_metadata = index_metadata

    @property
    def client(self) -> Client:
        """Return the configured synchronous Typesense client.

        This is the original client instance, not a copy. Calling :meth:`close`
        closes its underlying HTTP resources.
        """
        return self._client

    @property
    def async_client(self) -> AsyncClient | None:
        """Return the configured asynchronous client, if one was supplied."""
        return self._async_client

    @property
    def collection_name(self) -> str:
        """Return the Typesense collection used by this store."""
        return self._collection_name

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding model used for writes and text searches."""
        return self._embedding

    # ------------------------------------------------------------------
    # Validation and serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_k(k: int) -> None:
        """Validate a requested result count."""
        if k < 0:
            raise ValueError("`k` must be greater than or equal to 0.")

    @staticmethod
    def _validate_id(document_id: str) -> None:
        """Validate an ID before it is placed in a Typesense URL or filter."""
        if not document_id:
            raise ValueError("Typesense document IDs must not be empty.")
        if quote(document_id, safe="") != document_id:
            raise ValueError(
                "Typesense document IDs must contain only URL-safe, unreserved "
                f"characters; got {document_id!r}."
            )

    @classmethod
    def _resolve_ids(
        cls,
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
            cls._validate_id(document_id)
        return resolved_ids

    @staticmethod
    def _validate_vectors(
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

    def _prepare_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str] | None,
        vectors: Sequence[Sequence[float]],
    ) -> tuple[list[str], list[DocumentSchema], int | None]:
        """Serialize LangChain documents into Typesense bulk-import records."""
        resolved_ids = self._resolve_ids(documents, ids)
        validated_vectors = self._validate_vectors(vectors, len(documents))
        if not documents:
            return resolved_ids, [], None

        typesense_documents = [
            cast(
                DocumentSchema,
                {
                    "id": document_id,
                    self._text_key: document.page_content,
                    self._vector_key: vector,
                    self._metadata_key: dict(document.metadata),
                },
            )
            for document_id, document, vector in zip(
                resolved_ids, documents, validated_vectors, strict=True
            )
        ]
        return resolved_ids, typesense_documents, len(validated_vectors[0])

    @staticmethod
    def _raise_for_import_failures(response: object) -> None:
        """Raise an integration error for malformed or failed bulk imports."""
        if not isinstance(response, list):
            raise TypesenseVectorStoreError(
                "Typesense returned an unexpected bulk import response."
            )
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

    @staticmethod
    def _encode_filter_value(value: FilterScalar) -> str:
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

        if re.fullmatch(r"[A-Za-z0-9_.~-]+", value):
            return value
        escaped = value.replace("\\", "\\\\").replace("`", "\\`")
        return f"`{escaped}`"

    def _to_filter_by(self, filter_value: Filter) -> str:
        """Translate a filter mapping or raw expression to Typesense syntax.

        Mapping keys are treated as metadata field names and combined with
        ``&&``. A string is passed through unchanged, allowing callers to use the
        full Typesense filter language for advanced queries.
        """
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
            field_name = f"{self._metadata_key}.{key}"
            if isinstance(value, Sequence) and not isinstance(value, str):
                encoded = ",".join(self._encode_filter_value(item) for item in value)
                clauses.append(f"{field_name}:=[{encoded}]")
            else:
                clauses.append(f"{field_name}:={self._encode_filter_value(value)}")
        return " && ".join(clauses)

    @staticmethod
    def _validate_vector_query_options(
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
    ) -> None:
        """Validate optional Typesense vector-search tuning parameters."""
        if distance_threshold is not None and (
            not math.isfinite(distance_threshold) or distance_threshold < 0
        ):
            raise ValueError("`distance_threshold` must be a finite non-negative number.")
        if ef is not None and ef <= 0:
            raise ValueError("`ef` must be greater than 0.")
        if flat_search_cutoff is not None and flat_search_cutoff < 0:
            raise ValueError("`flat_search_cutoff` must be greater than or equal to 0.")

    def _build_search_parameters(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        filter: Filter | None,
        search_parameters: Mapping[str, Any] | None,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
        include_vectors: bool,
    ) -> SearchParameters:
        """Build a vector search request while protecting managed parameters."""
        self._validate_k(k)
        self._validate_vector_query_options(distance_threshold, ef, flat_search_cutoff)
        validated = self._validate_vectors([embedding], 1)[0]

        vector_options = [f"k:{k}"]
        if distance_threshold is not None:
            vector_options.append(f"distance_threshold:{distance_threshold!r}")
        if ef is not None:
            vector_options.append(f"ef:{ef}")
        if flat_search_cutoff is not None:
            vector_options.append(f"flat_search_cutoff:{flat_search_cutoff}")

        vector = ",".join(repr(value) for value in validated)
        parameters: dict[str, Any] = {
            "q": "*",
            "vector_query": (f"{self._vector_key}:([{vector}], {', '.join(vector_options)})"),
            "per_page": k,
        }
        if not include_vectors:
            parameters["exclude_fields"] = self._vector_key

        if search_parameters:
            conflicts = _RESERVED_SEARCH_PARAMETERS.intersection(search_parameters)
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValueError(
                    "`search_parameters` must not override integration-managed "
                    f"parameters: {names}."
                )
            parameters.update(search_parameters)

        if filter is not None:
            filter_by = self._to_filter_by(filter)
            if filter_by:
                parameters["filter_by"] = filter_by
        return cast(SearchParameters, parameters)

    def _parse_search_response(
        self,
        response: Mapping[str, Any],
        *,
        include_vectors: bool,
    ) -> list[tuple[Document, float, list[float] | None]]:
        """Convert Typesense hits to documents, distances, and optional vectors."""
        hits = response.get("hits", [])
        if not isinstance(hits, list):
            raise TypesenseVectorStoreError("Typesense search response has invalid `hits`.")

        results: list[tuple[Document, float, list[float] | None]] = []
        for raw_hit in hits:
            if not isinstance(raw_hit, Mapping) or not isinstance(raw_hit.get("document"), Mapping):
                raise TypesenseVectorStoreError("Typesense returned a malformed search hit.")

            raw_document = dict(cast(Mapping[str, Any], raw_hit["document"]))
            try:
                document_id = str(raw_document.pop("id"))
                page_content = str(raw_document.pop(self._text_key))
                distance = float(raw_hit["vector_distance"])
            except (KeyError, TypeError, ValueError) as error:
                raise TypesenseVectorStoreError(
                    "Typesense search hit is missing a valid ID, text, or vector distance."
                ) from error

            raw_metadata = raw_document.pop(self._metadata_key, {})
            if not isinstance(raw_metadata, Mapping):
                raise TypesenseCollectionError(
                    f"Field `{self._metadata_key}` must contain an object."
                )

            raw_vector = raw_document.pop(self._vector_key, None)
            metadata = dict(raw_document)
            metadata.update(raw_metadata)

            vector: list[float] | None = None
            if include_vectors:
                if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str):
                    raise TypesenseVectorStoreError(
                        f"Typesense search hit is missing vector field `{self._vector_key}`."
                    )
                vector = [float(value) for value in raw_vector]

            results.append(
                (
                    Document(
                        id=document_id,
                        page_content=page_content,
                        metadata=metadata,
                    ),
                    distance,
                    vector,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def _collection_schema(self, num_dim: int) -> CollectionCreateSchema:
        """Build the schema used when creating the backing collection."""
        return {
            "name": self._collection_name,
            "enable_nested_fields": True,
            "fields": [
                {"name": self._text_key, "type": "string"},
                {
                    "name": self._vector_key,
                    "type": "float[]",
                    "num_dim": num_dim,
                    "vec_dist": "cosine",
                },
                {
                    "name": self._metadata_key,
                    "type": "object",
                    "optional": True,
                    "index": self._index_metadata,
                },
            ],
        }

    def _validate_collection_schema(
        self,
        schema: CollectionSchema,
        num_dim: int,
    ) -> None:
        """Verify that an existing collection can serve this store."""
        fields = {
            str(field.get("name")): field
            for field in schema.get("fields", [])
            if isinstance(field, Mapping)
        }
        text_field = fields.get(self._text_key)
        vector_field = fields.get(self._vector_key)
        metadata_field = fields.get(self._metadata_key)

        errors: list[str] = []
        if text_field is None or text_field.get("type") != "string":
            errors.append(f"`{self._text_key}` must have type `string`")
        if vector_field is None or vector_field.get("type") != "float[]":
            errors.append(f"`{self._vector_key}` must have type `float[]`")
        else:
            raw_num_dim = vector_field.get("num_dim")
            if not isinstance(raw_num_dim, (int, float)) or int(raw_num_dim) != num_dim:
                errors.append(f"`{self._vector_key}` must have `num_dim={num_dim}`")
            if vector_field.get("vec_dist", "cosine") != "cosine":
                errors.append(f"`{self._vector_key}` must use cosine distance")
        if metadata_field is None:
            if self._index_metadata:
                errors.append(f"`{self._metadata_key}` must have type `object`")
        elif metadata_field.get("type") != "object":
            errors.append(f"`{self._metadata_key}` must have type `object`")
        elif self._index_metadata and metadata_field.get("index", True) is False:
            errors.append(f"`{self._metadata_key}` must be indexed")
        if metadata_field is not None and schema.get("enable_nested_fields") is not True:
            errors.append("`enable_nested_fields` must be true")

        if errors:
            joined = "; ".join(errors)
            raise TypesenseCollectionError(
                f"Collection `{self._collection_name}` is incompatible: {joined}."
            )

    def create_collection(self, num_dim: int) -> None:
        """Create or validate the backing collection synchronously.

        Creation is idempotent. An existing collection is not modified; instead,
        its required fields, vector dimension, and distance metric are validated.

        Args:
            num_dim: Number of dimensions in the embeddings that will be stored.

        Raises:
            ValueError: If ``num_dim`` is not positive.
            TypesenseCollectionError: If an existing collection is incompatible.
        """
        if num_dim <= 0:
            raise ValueError("`num_dim` must be greater than 0.")
        try:
            schema = self._client.collections[self._collection_name].retrieve()
        except ObjectNotFound:
            try:
                self._client.collections.create(self._collection_schema(num_dim))
                return
            except ObjectAlreadyExists:
                schema = self._client.collections[self._collection_name].retrieve()
        self._validate_collection_schema(schema, num_dim)

    async def acreate_collection(self, num_dim: int) -> None:
        """Asynchronously create or validate the backing collection.

        Uses the configured asynchronous client when available; otherwise it
        delegates to :meth:`create_collection` in a worker thread.

        Args:
            num_dim: Number of dimensions in the embeddings that will be stored.

        Raises:
            ValueError: If ``num_dim`` is not positive.
            TypesenseCollectionError: If an existing collection is incompatible.
        """
        if self._async_client is None:
            await run_in_executor(None, self.create_collection, num_dim)
            return
        if num_dim <= 0:
            raise ValueError("`num_dim` must be greater than 0.")
        try:
            schema = await self._async_client.collections[self._collection_name].retrieve()
        except ObjectNotFound:
            try:
                await self._async_client.collections.create(self._collection_schema(num_dim))
                return
            except ObjectAlreadyExists:
                schema = await self._async_client.collections[self._collection_name].retrieve()
        self._validate_collection_schema(schema, num_dim)

    def delete_collection(self) -> bool:
        """Delete the backing collection.

        Returns:
            ``True`` when the collection was deleted; ``False`` when it was
            already absent.
        """
        try:
            self._client.collections[self._collection_name].delete()
        except ObjectNotFound:
            return False
        return True

    async def adelete_collection(self) -> bool:
        """Asynchronously delete the backing collection.

        If no asynchronous client is configured, the synchronous operation runs
        in a worker thread. The method is idempotent for a missing collection.

        Returns:
            ``True`` when the collection was deleted; ``False`` when it was
            already absent.
        """
        if self._async_client is None:
            return await run_in_executor(None, self.delete_collection)
        try:
            await self._async_client.collections[self._collection_name].delete()
        except ObjectNotFound:
            return False
        return True

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert LangChain documents into Typesense.

        The operation creates the collection lazily and uses Typesense's bulk
        ``upsert`` action, so repeated IDs replace existing documents. Bulk
        imports may partially succeed; in that case
        :class:`TypesenseImportError` exposes the failed records.

        Args:
            documents: Documents whose page content is embedded and stored.
            ids: Optional stable Typesense IDs. Missing document IDs are replaced
                with generated UUIDs.
            batch_size: Optional number of records per Typesense import request.
            **kwargs: Reserved for the LangChain interface; unsupported options
                raise ``TypeError``.

        Returns:
            IDs assigned to the documents, in input order.

        Raises:
            ValueError: If IDs, batch size, or embeddings are invalid.
            TypesenseImportError: If one or more records fail to import.
        """
        if kwargs:
            raise TypeError(f"Unsupported add options: {', '.join(sorted(kwargs))}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("`batch_size` must be greater than 0.")
        if not documents:
            if ids:
                raise ValueError("IDs were supplied for an empty document list.")
            return []

        vectors = self._embedding.embed_documents([document.page_content for document in documents])
        resolved_ids, payload, num_dim = self._prepare_documents(documents, ids, vectors)
        if num_dim is None:  # pragma: no cover - guarded by the empty check above
            return []
        self.create_collection(num_dim)
        import_parameters: DocumentWriteParameters = {"action": "upsert"}
        response = self._client.collections[self._collection_name].documents.import_(
            payload, import_parameters, batch_size=batch_size
        )
        self._raise_for_import_failures(response)
        return resolved_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert text strings into Typesense.

        Args:
            texts: Text content to embed and store.
            metadatas: Optional metadata dictionaries aligned with ``texts``.
            ids: Optional stable Typesense IDs aligned with ``texts``.
            **kwargs: Options forwarded to :meth:`add_documents`.

        Returns:
            IDs assigned to the texts, in input order.
        """
        text_list = list(texts)
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError(
                "The number of metadatas must match the number of texts. "
                f"Got {len(metadatas)} metadatas and {len(text_list)} texts."
            )
        metadata_list = metadatas if metadatas is not None else [{} for _ in text_list]
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(text_list, metadata_list, strict=True)
        ]
        return self.add_documents(documents, ids=ids, **kwargs)

    async def aadd_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously embed and upsert LangChain documents into Typesense.

        Uses native asynchronous embedding and Typesense clients when configured;
        otherwise falls back to the synchronous implementation in a worker thread.
        See :meth:`add_documents` for ID, batching, and partial-import behavior.
        """
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.add_documents,
                documents,
                ids=ids,
                batch_size=batch_size,
                **kwargs,
            )
        if kwargs:
            raise TypeError(f"Unsupported add options: {', '.join(sorted(kwargs))}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("`batch_size` must be greater than 0.")
        if not documents:
            if ids:
                raise ValueError("IDs were supplied for an empty document list.")
            return []

        vectors = await self._embedding.aembed_documents(
            [document.page_content for document in documents]
        )
        resolved_ids, payload, num_dim = self._prepare_documents(documents, ids, vectors)
        if num_dim is None:  # pragma: no cover - guarded by the empty check above
            return []
        await self.acreate_collection(num_dim)
        import_parameters: DocumentWriteParameters = {"action": "upsert"}
        response = await self._async_client.collections[self._collection_name].documents.import_(
            payload, import_parameters, batch_size=batch_size
        )
        self._raise_for_import_failures(response)
        return resolved_ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously embed and upsert text strings into Typesense.

        See :meth:`add_texts` for metadata and ID alignment rules.
        """
        text_list = list(texts)
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError(
                "The number of metadatas must match the number of texts. "
                f"Got {len(metadatas)} metadatas and {len(text_list)} texts."
            )
        metadata_list = metadatas if metadatas is not None else [{} for _ in text_list]
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(text_list, metadata_list, strict=True)
        ]
        return await self.aadd_documents(documents, ids=ids, **kwargs)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _search_by_vector(
        self,
        embedding: list[float],
        k: int,
        *,
        filter: Filter | None,
        search_parameters: Mapping[str, Any] | None,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
        include_vectors: bool,
    ) -> list[tuple[Document, float, list[float] | None]]:
        """Run a Typesense vector query and parse its hits synchronously."""
        self._validate_k(k)
        if k == 0:
            return []
        parameters = self._build_search_parameters(
            embedding,
            k,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
            include_vectors=include_vectors,
        )
        try:
            response = self._client.collections[self._collection_name].documents.search(parameters)
        except ObjectNotFound:
            return []
        return self._parse_search_response(response, include_vectors=include_vectors)

    async def _asearch_by_vector(
        self,
        embedding: list[float],
        k: int,
        *,
        filter: Filter | None,
        search_parameters: Mapping[str, Any] | None,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
        include_vectors: bool,
    ) -> list[tuple[Document, float, list[float] | None]]:
        """Run a Typesense vector query using native async I/O when available."""
        if self._async_client is None:
            return await run_in_executor(
                None,
                self._search_by_vector,
                embedding,
                k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                include_vectors=include_vectors,
            )
        self._validate_k(k)
        if k == 0:
            return []
        parameters = self._build_search_parameters(
            embedding,
            k,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
            include_vectors=include_vectors,
        )
        try:
            response = await self._async_client.collections[self._collection_name].documents.search(
                parameters
            )
        except ObjectNotFound:
            return []
        return self._parse_search_response(response, include_vectors=include_vectors)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: Mapping[str, Any] | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents nearest to a text query.

        Args:
            query: Text to embed and search for.
            k: Maximum number of documents to return.
            filter: Metadata equality mapping or raw Typesense ``filter_by``
                expression.
            search_parameters: Additional non-managed Typesense search parameters.
            distance_threshold: Optional maximum cosine distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional threshold for exact flat search.
            **kwargs: Reserved for the LangChain interface; unsupported options
                raise ``TypeError``.

        Returns:
            Documents ordered by increasing Typesense cosine distance.
        """
        return [
            document
            for document, _ in self.similarity_search_with_score(
                query,
                k=k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                **kwargs,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: Mapping[str, Any] | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents with raw Typesense cosine distances.

        The second tuple item is a distance, not a LangChain relevance score:
        lower values indicate more similar documents. Use the inherited
        :meth:`similarity_search_with_relevance_scores` method when a bounded
        relevance score is required.

        Args:
            query: Text to embed and search for.
            k: Maximum number of results.
            filter: Metadata equality mapping or raw Typesense filter expression.
            search_parameters: Additional non-managed Typesense parameters.
            distance_threshold: Optional maximum cosine distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.
            **kwargs: Reserved interface options; unsupported values raise
                ``TypeError``.

        Returns:
            ``(Document, distance)`` pairs ordered from nearest to farthest.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        self._validate_k(k)
        if k == 0:
            return []
        embedding = self._embedding.embed_query(query)
        return [
            (document, distance)
            for document, distance, _ in self._search_by_vector(
                embedding,
                k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                include_vectors=False,
            )
        ]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: Mapping[str, Any] | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents nearest to a caller-supplied embedding.

        Args:
            embedding: Query vector. Its dimension must match the collection.
            k: Maximum number of documents to return.
            filter: Metadata equality mapping or raw Typesense filter expression.
            search_parameters: Additional non-managed Typesense parameters.
            distance_threshold: Optional maximum cosine distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.
            **kwargs: Reserved interface options; unsupported values raise
                ``TypeError``.

        Returns:
            Documents ordered by increasing cosine distance.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        return [
            document
            for document, _, _ in self._search_by_vector(
                embedding,
                k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                include_vectors=False,
            )
        ]

    @staticmethod
    def _cosine_distance_to_relevance(distance: float) -> float:
        """Map a cosine distance to LangChain's bounded relevance score."""
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Return the distance-to-relevance conversion used by LangChain."""
        return self._cosine_distance_to_relevance

    def close(self) -> None:
        """Close the synchronous client's underlying HTTP resources.

        This also closes a client supplied by the caller. Call it only when that
        client is no longer needed; the client must not be reused afterward.
        """
        self._client.api_call.close()

    async def aclose(self) -> None:
        """Close the configured clients' underlying HTTP resources.

        The asynchronous client's resources are closed first, followed by the
        synchronous client. Supplied clients must not be reused afterward. This
        method is safe when no asynchronous client was supplied.
        """
        if self._async_client is not None:
            await self._async_client.api_call.aclose()
        self.close()

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: Mapping[str, Any] | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents nearest to a text query.

        Parameters match :meth:`similarity_search`; native asynchronous embedding
        and Typesense I/O are used when an async client was configured. Otherwise,
        the synchronous implementation runs in a worker thread.
        """
        return [
            document
            for document, _ in await self.asimilarity_search_with_score(
                query,
                k=k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                **kwargs,
            )
        ]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: Mapping[str, Any] | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Asynchronously return documents and raw cosine distances.

        The returned distances are Typesense cosine distances, where lower is
        better. Parameters match :meth:`similarity_search_with_score`. Without an
        async client, the synchronous implementation runs in a worker thread.
        """
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.similarity_search_with_score,
                query,
                k=k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                **kwargs,
            )
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        self._validate_k(k)
        if k == 0:
            return []
        embedding = await self._embedding.aembed_query(query)
        return [
            (document, distance)
            for document, distance, _ in await self._asearch_by_vector(
                embedding,
                k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                include_vectors=False,
            )
        ]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: Mapping[str, Any] | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents nearest to an embedding vector.

        Parameters match :meth:`similarity_search_by_vector`. Without an async
        client, the synchronous implementation runs in a worker thread.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        return [
            document
            for document, _, _ in await self._asearch_by_vector(
                embedding,
                k,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                include_vectors=False,
            )
        ]

    # ------------------------------------------------------------------
    # Maximal marginal relevance
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_mmr(k: int, fetch_k: int, lambda_mult: float) -> None:
        """Validate maximal-marginal-relevance search parameters."""
        TypesenseVectorStore._validate_k(k)
        if fetch_k < 0:
            raise ValueError("`fetch_k` must be greater than or equal to 0.")
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("`lambda_mult` must be between 0 and 1.")

    @staticmethod
    def _select_mmr_documents(
        embedding: list[float],
        results: Sequence[tuple[Document, float, list[float] | None]],
        *,
        k: int,
        lambda_mult: float,
    ) -> list[Document]:
        """Select diverse documents from retrieved hits using LangChain MMR."""
        candidate_vectors = [vector for _, _, vector in results if vector is not None]
        if len(candidate_vectors) != len(results):
            raise TypesenseVectorStoreError("MMR search requires vectors in every hit.")
        selected = maximal_marginal_relevance(
            np.asarray(embedding, dtype=float),
            candidate_vectors,
            lambda_mult=lambda_mult,
            k=k,
        )
        return [results[index][0] for index in selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return diverse documents selected with maximal marginal relevance.

        ``fetch_k`` candidates are retrieved from Typesense, then reranked locally
        using LangChain's MMR implementation. ``lambda_mult=1`` favors query
        similarity, while ``lambda_mult=0`` favors diversity.

        Args:
            query: Text to embed and search for.
            k: Number of diverse documents to return.
            fetch_k: Number of candidates to fetch before local reranking.
            lambda_mult: Similarity/diversity trade-off in the inclusive range
                ``[0, 1]``.
            **kwargs: Search filters and Typesense tuning options accepted by
                :meth:`similarity_search_by_vector`.

        Returns:
            Up to ``k`` documents selected by MMR.
        """
        self._validate_mmr(k, fetch_k, lambda_mult)
        if k == 0 or fetch_k == 0:
            return []
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return diverse documents for a caller-supplied embedding.

        Args:
            embedding: Query vector whose dimension matches the collection.
            k: Number of diverse documents to return.
            fetch_k: Number of candidates fetched before local MMR reranking.
            lambda_mult: Similarity/diversity trade-off in ``[0, 1]``.
            **kwargs: Optional ``filter``, ``search_parameters``,
                ``distance_threshold``, ``ef``, and ``flat_search_cutoff`` values.

        Returns:
            Up to ``k`` documents selected by local MMR reranking.
        """
        self._validate_mmr(k, fetch_k, lambda_mult)
        if k == 0 or fetch_k == 0:
            return []
        results = self._search_by_vector(
            embedding,
            max(k, fetch_k),
            filter=kwargs.pop("filter", None),
            search_parameters=kwargs.pop("search_parameters", None),
            distance_threshold=kwargs.pop("distance_threshold", None),
            ef=kwargs.pop("ef", None),
            flat_search_cutoff=kwargs.pop("flat_search_cutoff", None),
            include_vectors=True,
        )
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        return self._select_mmr_documents(
            embedding,
            results,
            k=k,
            lambda_mult=lambda_mult,
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents selected with MMR.

        Parameters and reranking semantics match
        :meth:`max_marginal_relevance_search`. Without an async client, the
        synchronous implementation runs in a worker thread.
        """
        self._validate_mmr(k, fetch_k, lambda_mult)
        if k == 0 or fetch_k == 0:
            return []
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.max_marginal_relevance_search,
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                **kwargs,
            )
        embedding = await self._embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return diverse documents for an embedding vector.

        Parameters and reranking semantics match
        :meth:`max_marginal_relevance_search_by_vector`. Without an async client,
        the synchronous implementation runs in a worker thread.
        """
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.max_marginal_relevance_search_by_vector,
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                **kwargs,
            )
        self._validate_mmr(k, fetch_k, lambda_mult)
        if k == 0 or fetch_k == 0:
            return []
        results = await self._asearch_by_vector(
            embedding,
            max(k, fetch_k),
            filter=kwargs.pop("filter", None),
            search_parameters=kwargs.pop("search_parameters", None),
            distance_threshold=kwargs.pop("distance_threshold", None),
            ef=kwargs.pop("ef", None),
            flat_search_cutoff=kwargs.pop("flat_search_cutoff", None),
            include_vectors=True,
        )
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        return self._select_mmr_documents(
            embedding,
            results,
            k=k,
            lambda_mult=lambda_mult,
        )

    # ------------------------------------------------------------------
    # Reads and deletes by ID
    # ------------------------------------------------------------------
    def _document_from_typesense(self, raw: Mapping[str, Any]) -> Document:
        """Deserialize one Typesense document into a LangChain document."""
        values = dict(raw)
        try:
            document_id = str(values.pop("id"))
            page_content = str(values.pop(self._text_key))
        except KeyError as error:
            raise TypesenseCollectionError(
                "Stored document is missing the configured ID or text field."
            ) from error
        values.pop(self._vector_key, None)
        raw_metadata = values.pop(self._metadata_key, {})
        if not isinstance(raw_metadata, Mapping):
            raise TypesenseCollectionError(f"Field `{self._metadata_key}` must contain an object.")
        metadata = dict(values)
        metadata.update(raw_metadata)
        return Document(id=document_id, page_content=page_content, metadata=metadata)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Return found documents for the supplied IDs.

        IDs are de-duplicated and missing documents are ignored, matching
        LangChain's ``get_by_ids`` contract. This implementation currently
        preserves first-seen order, but callers should not depend on result order.

        Args:
            ids: Typesense document IDs to retrieve.

        Returns:
            Found documents. The result may be shorter than ``ids`` and its order
            is not part of the public contract.
        """
        documents: list[Document] = []
        collection = self._client.collections[self._collection_name]
        for document_id in dict.fromkeys(ids):
            self._validate_id(document_id)
            try:
                raw = collection.documents[document_id].retrieve()
            except ObjectNotFound:
                continue
            documents.append(self._document_from_typesense(raw))
        return documents

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Asynchronously return found documents for the supplied IDs.

        Missing documents are ignored and duplicate IDs are de-duplicated. Uses
        native asynchronous retrieval when configured, otherwise a worker thread.
        """
        if self._async_client is None:
            return await run_in_executor(None, self.get_by_ids, ids)
        documents: list[Document] = []
        collection = self._async_client.collections[self._collection_name]
        for document_id in dict.fromkeys(ids):
            self._validate_id(document_id)
            try:
                raw = await collection.documents[document_id].retrieve()
            except ObjectNotFound:
                continue
            documents.append(self._document_from_typesense(raw))
        return documents

    @classmethod
    def _ids_filter(cls, ids: Sequence[str]) -> str:
        """Build a Typesense ID filter after validating every ID."""
        for document_id in ids:
            cls._validate_id(document_id)
        return f"id:=[{','.join(ids)}]"

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete selected documents or truncate the collection.

        Args:
            ids: IDs to delete. ``None`` truncates all documents; an empty list is
                a no-op.
            **kwargs: Reserved for the LangChain interface; unsupported options
                raise ``TypeError``.

        Returns:
            Always ``True`` after a successful request or when the collection is
            already absent.
        """
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(sorted(kwargs))}")
        parameters: DeleteQueryParameters
        if ids is None:
            parameters = {"truncate": True}
        elif not ids:
            return True
        else:
            parameters = {"filter_by": self._ids_filter(ids)}
        try:
            self._client.collections[self._collection_name].documents.delete(parameters)
        except ObjectNotFound:
            return True
        return True

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Asynchronously delete selected documents or truncate the collection.

        Parameters and idempotent missing-collection behavior match :meth:`delete`.
        Native asynchronous I/O is used when configured; otherwise a worker thread
        runs the synchronous operation.
        """
        if self._async_client is None:
            return await run_in_executor(None, self.delete, ids, **kwargs)
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(sorted(kwargs))}")
        parameters: DeleteQueryParameters
        if ids is None:
            parameters = {"truncate": True}
        elif not ids:
            return True
        else:
            parameters = {"filter_by": self._ids_filter(ids)}
        try:
            await self._async_client.collections[self._collection_name].documents.delete(parameters)
        except ObjectNotFound:
            return True
        return True

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_client_params(
        cls,
        embedding: Embeddings,
        *,
        host: str = "localhost",
        port: int = 8108,
        protocol: str = "http",
        api_key: str | None = None,
        typesense_api_key: str | None = None,
        connection_timeout_seconds: float = 2.0,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Build a store and clients from Typesense connection parameters.

        Both synchronous and asynchronous clients are configured with the same
        node and credentials. ``api_key`` takes precedence over the legacy
        ``typesense_api_key`` name, and either falls back to the
        ``TYPESENSE_API_KEY`` environment variable.

        Args:
            embedding: Embedding model used by the returned store.
            host: Typesense node hostname or address.
            port: Typesense node port.
            protocol: Node protocol, usually ``"http"`` or ``"https"``.
            api_key: Typesense API key.
            typesense_api_key: Backward-compatible API-key argument.
            connection_timeout_seconds: Client request timeout.
            collection_name: Backing collection name.
            **kwargs: Additional :class:`TypesenseVectorStore` options, such as
                field names or metadata indexing.

        Returns:
            A store with configured synchronous and asynchronous clients.

        Raises:
            ValueError: If both API-key arguments are supplied with different
                values, or no API key can be resolved.
        """
        if api_key and typesense_api_key and api_key != typesense_api_key:
            raise ValueError("`api_key` and `typesense_api_key` must not conflict.")
        resolved_api_key = (
            api_key or typesense_api_key or get_from_env("api_key", "TYPESENSE_API_KEY")
        )
        config: ConfigDict = {
            "nodes": [{"host": host, "port": port, "protocol": protocol}],
            "api_key": resolved_api_key,
            "connection_timeout_seconds": connection_timeout_seconds,
        }
        return cls(
            client=Client(config),
            async_client=AsyncClient(config),
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        client: Client | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a store and immediately add texts using an existing client.

        This helper is useful for applications that already manage a configured
        Typesense client. The collection is created lazily by the initial write.

        Args:
            texts: Text strings to embed and store.
            embedding: Embedding model used for the texts.
            metadatas: Optional metadata dictionaries aligned with ``texts``.
            ids: Optional stable IDs aligned with ``texts``.
            client: Configured synchronous Typesense client.
            collection_name: Backing collection name.
            **kwargs: Additional constructor options such as ``async_client`` or
                custom field names.

        Returns:
            A populated :class:`TypesenseVectorStore`.

        Raises:
            ValueError: If ``client`` is omitted or input lengths are inconsistent.
            TypesenseImportError: If Typesense partially rejects the import.
        """
        if client is None:
            raise ValueError("`client` (a `typesense.Client`) is required.")
        store = cls(
            client=client,
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


# Compatibility alias for applications migrating from langchain-community.
Typesense = TypesenseVectorStore

__all__ = [
    "Typesense",
    "TypesenseCollectionError",
    "TypesenseImportError",
    "TypesenseVectorStore",
    "TypesenseVectorStoreError",
]
