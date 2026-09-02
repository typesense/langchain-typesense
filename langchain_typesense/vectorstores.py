"""Typesense-backed vector storage for LangChain.

The integration stores each :class:`~langchain_core.documents.Document` as a
Typesense document with three configurable fields: text, vector, and nested
metadata. Collections are created lazily from the first batch of embeddings and
are validated on writes. Search uses the configured Typesense vector distance;
the ``*_with_score`` methods return raw distance where lower is better. The
LangChain relevance-score adapter is available for cosine distance.

Use :meth:`TypesenseVectorStore.from_client_params` for a convenience
constructor, or pass an already configured synchronous client, asynchronous
client, or both to
:class:`TypesenseVectorStore` when connection lifecycle is managed by the
application.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

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

from langchain_typesense._codec import (
    build_vector_query,
    document_from_typesense,
    encode_filter_value,
    ids_filter,
    node_from_url,
    normalize_hybrid_query_by,
    parse_export_response,
    parse_hybrid_search_response,
    parse_search_response,
    raise_for_import_failures,
    resolve_ids,
    to_filter_by,
    validate_hybrid_query,
    validate_id,
    validate_k,
    validate_vector_query_options,
    validate_vectors,
)
from langchain_typesense._errors import (
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStoreError,
)
from langchain_typesense._types import (
    MANAGED_SEARCH_PARAMETERS,
    SAFE_HYBRID_SEARCH_PARAMETERS,
    SAFE_SEARCH_PARAMETERS,
    ClientMode,
    Filter,
    TypesenseHybridSearchParameters,
    TypesenseSearchParameters,
    VectorDistance,
)

DEFAULT_COLLECTION_NAME = "langchain-typesense"
DEFAULT_TEXT_KEY = "text"
DEFAULT_VECTOR_KEY = "vec"
DEFAULT_METADATA_KEY = "metadata"


class TypesenseVectorStore(VectorStore):
    """LangChain vector store backed by Typesense.

    LangChain metadata is kept under one nested object field. This prevents metadata
    keys such as ``id`` or ``text`` from overwriting Typesense's internal fields.
    Metadata is indexed by default so dictionary filters can target nested values.
    Set ``index_metadata=False`` if metadata only needs to be stored and returned.

    Args:
        client: Configured synchronous Typesense client. May be omitted when an
            asynchronous client is supplied and only async methods will be used.
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
        vec_dist: Typesense vector distance metric. ``"cosine"`` supports every
            feature. ``"ip"`` supports raw-distance similarity search, but not
            LangChain's cosine-based MMR or bounded relevance-score adapter.
    """

    def __init__(
        self,
        client: Client | None,
        embedding: Embeddings,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        async_client: AsyncClient | None = None,
        text_key: str = DEFAULT_TEXT_KEY,
        vector_key: str = DEFAULT_VECTOR_KEY,
        metadata_key: str = DEFAULT_METADATA_KEY,
        index_metadata: bool = True,
        vec_dist: VectorDistance = "cosine",
    ) -> None:
        """Initialize a store without making network requests.

        Collection creation and schema validation are deferred until a
        non-empty write, when the embedding dimension is known. Searches use an
        existing collection as-is; Typesense errors propagate when it is absent.

        Args:
            client: Configured synchronous Typesense client, or ``None`` for an
                async-only store.
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
            vec_dist: ``"cosine"`` or ``"ip"`` (inner product).

        Raises:
            ValueError: If no client is supplied, or configuration is invalid.
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
        if client is None and async_client is None:
            raise ValueError("At least one of `client` or `async_client` is required.")
        if vec_dist not in ("cosine", "ip"):
            raise ValueError("`vec_dist` must be either 'cosine' or 'ip'.")

        self._client = client
        self._async_client = async_client
        self._embedding = embedding
        self._collection_name = collection_name
        self._text_key = text_key
        self._vector_key = vector_key
        self._metadata_key = metadata_key
        self._index_metadata = index_metadata
        self._vec_dist = vec_dist
        self._sync_validated_num_dim: int | None = None
        self._async_validated_num_dim: int | None = None

    @property
    def client(self) -> Client | None:
        """Return the configured synchronous Typesense client, if supplied.

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

    @property
    def vec_dist(self) -> VectorDistance:
        """Return the Typesense vector distance metric used by the store."""
        return self._vec_dist

    def _require_sync_client(self) -> Client:
        """Return the sync client or fail clearly for an async-only store."""
        if self._client is None:
            raise RuntimeError(
                "This store was configured with only an async Typesense client; "
                "use the corresponding async method."
            )
        return self._client

    # ------------------------------------------------------------------
    # Validation and serialization
    # ------------------------------------------------------------------
    _validate_k = staticmethod(validate_k)
    _validate_id = staticmethod(validate_id)
    _resolve_ids = staticmethod(resolve_ids)
    _validate_vectors = staticmethod(validate_vectors)

    def _prepare_documents(
        self,
        documents: Sequence[Document],
        resolved_ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> tuple[list[str], list[DocumentSchema], int | None]:
        """Serialize LangChain documents into Typesense bulk-import records."""
        validated_vectors = self._validate_vectors(vectors, len(documents))
        if not documents:
            return list(resolved_ids), [], None

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
        return list(resolved_ids), typesense_documents, len(validated_vectors[0])

    _raise_for_import_failures = staticmethod(raise_for_import_failures)
    _encode_filter_value = staticmethod(encode_filter_value)

    def _to_filter_by(self, filter_value: Filter) -> str:
        """Translate a filter mapping or raw expression to Typesense syntax.

        Mapping keys are treated as metadata field names and combined with
        ``&&``. A string is passed through unchanged, allowing callers to use the
        full Typesense filter language for advanced queries.
        """
        return to_filter_by(filter_value, self._metadata_key)

    def _validate_vector_query_options(
        self,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
    ) -> None:
        """Validate optional Typesense vector-search tuning parameters."""
        validate_vector_query_options(distance_threshold, ef, flat_search_cutoff, self._vec_dist)

    def _build_search_parameters(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        filter: Filter | None,
        search_parameters: TypesenseSearchParameters | None,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
        include_vectors: bool,
    ) -> SearchParameters:
        """Build a vector search request while protecting managed parameters."""
        self._validate_k(k)
        vector_query = self._build_vector_query(
            embedding,
            k,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
        )

        parameters: dict[str, Any] = {
            "q": "*",
            "vector_query": vector_query,
            "per_page": k,
        }
        if not include_vectors:
            parameters["exclude_fields"] = self._vector_key

        if search_parameters:
            conflicts = MANAGED_SEARCH_PARAMETERS.intersection(search_parameters)
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValueError(
                    "`search_parameters` must not override integration-managed "
                    f"parameters: {names}."
                )

            unsupported = set(search_parameters).difference(SAFE_SEARCH_PARAMETERS)
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(
                    "Unsupported Typesense `search_parameters`: "
                    f"{names}. This integration only forwards options that preserve "
                    "the vector-search response contract."
                )
            parameters.update(search_parameters)

        if filter is not None:
            filter_by = self._to_filter_by(filter)
            if filter_by:
                parameters["filter_by"] = filter_by
        return cast(SearchParameters, parameters)

    def _build_vector_query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
        alpha: float | None = None,
    ) -> str:
        """Serialize an embedding and its Typesense vector-query options."""
        return build_vector_query(
            embedding,
            k,
            vector_key=self._vector_key,
            vec_dist=self._vec_dist,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
            alpha=alpha,
        )

    def _normalize_hybrid_query_by(self, query_by: str | Sequence[str] | None) -> str:
        """Return a comma-separated keyword-field list for hybrid search."""
        return normalize_hybrid_query_by(
            query_by, text_key=self._text_key, vector_key=self._vector_key
        )

    _validate_hybrid_query = staticmethod(validate_hybrid_query)

    def _validate_hybrid_search_options(
        self,
        query: str,
        k: int,
        alpha: float,
        query_by: str | Sequence[str] | None,
        search_parameters: TypesenseHybridSearchParameters | None,
    ) -> str:
        """Validate hybrid options before doing embedding or network work."""
        self._validate_k(k)
        self._validate_hybrid_query(query, alpha)
        normalized_query_by = self._normalize_hybrid_query_by(query_by)
        if search_parameters:
            conflicts = MANAGED_SEARCH_PARAMETERS.intersection(search_parameters)
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValueError(
                    "`search_parameters` must not override hybrid-search-managed "
                    f"parameters: {names}."
                )
            unsupported = set(search_parameters).difference(SAFE_HYBRID_SEARCH_PARAMETERS)
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"Unsupported Typesense hybrid `search_parameters`: {names}.")
        return normalized_query_by

    def _build_hybrid_search_parameters(
        self,
        query: str,
        embedding: Sequence[float],
        k: int,
        *,
        alpha: float,
        query_by: str,
        filter: Filter | None,
        search_parameters: TypesenseHybridSearchParameters | None,
        distance_threshold: float | None,
        ef: int | None,
        flat_search_cutoff: int | None,
    ) -> SearchParameters:
        """Build a Typesense request that fuses keyword and vector ranking."""
        parameters: dict[str, Any] = {
            "q": query,
            "query_by": query_by,
            "vector_query": self._build_vector_query(
                embedding,
                k,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
                alpha=alpha,
            ),
            "per_page": k,
            "exclude_fields": self._vector_key,
        }

        if search_parameters:
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
        return parse_search_response(
            response,
            text_key=self._text_key,
            vector_key=self._vector_key,
            metadata_key=self._metadata_key,
            include_vectors=include_vectors,
        )

    def _parse_hybrid_search_response(
        self,
        response: Mapping[str, Any],
    ) -> list[tuple[Document, float]]:
        """Convert Typesense hybrid hits to documents and rank-fusion scores."""
        return parse_hybrid_search_response(
            response,
            text_key=self._text_key,
            vector_key=self._vector_key,
            metadata_key=self._metadata_key,
        )

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
                    "vec_dist": self._vec_dist,
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
        raw_schema = cast(Mapping[str, Any], schema)
        raw_fields = raw_schema.get("fields", [])
        fields = {
            str(field.get("name")): field for field in raw_fields if isinstance(field, Mapping)
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
            if (
                not isinstance(raw_num_dim, int)
                or isinstance(raw_num_dim, bool)
                or raw_num_dim != num_dim
            ):
                errors.append(f"`{self._vector_key}` must have `num_dim={num_dim}`")
            if vector_field.get("vec_dist", "cosine") != self._vec_dist:
                errors.append(f"`{self._vector_key}` must use {self._vec_dist} distance")
        if metadata_field is None:
            if self._index_metadata:
                errors.append(f"`{self._metadata_key}` must have type `object`")
        elif metadata_field.get("type") != "object":
            errors.append(f"`{self._metadata_key}` must have type `object`")
        elif self._index_metadata and metadata_field.get("index", True) is False:
            errors.append(f"`{self._metadata_key}` must be indexed")
        if metadata_field is not None and raw_schema.get("enable_nested_fields") is not True:
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
        if self._sync_validated_num_dim == num_dim:
            return
        client = self._require_sync_client()
        try:
            schema = client.collections[self._collection_name].retrieve()
        except ObjectNotFound:
            self._create_missing_collection(num_dim)
            return
        self._validate_collection_schema(schema, num_dim)
        self._sync_validated_num_dim = num_dim

    def _create_missing_collection(self, num_dim: int) -> None:
        """Create a collection already known to be missing, handling a creator race."""
        client = self._require_sync_client()
        try:
            client.collections.create(self._collection_schema(num_dim))
        except ObjectAlreadyExists:
            schema = client.collections[self._collection_name].retrieve()
            self._validate_collection_schema(schema, num_dim)
        self._sync_validated_num_dim = num_dim

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
        if self._async_validated_num_dim == num_dim:
            return
        try:
            schema = await self._async_client.collections[self._collection_name].retrieve()
        except ObjectNotFound:
            await self._acreate_missing_collection(num_dim)
            return
        self._validate_collection_schema(schema, num_dim)
        self._async_validated_num_dim = num_dim

    async def _acreate_missing_collection(self, num_dim: int) -> None:
        """Asynchronously create a known-missing collection, handling a creator race."""
        if self._async_client is None:  # pragma: no cover - internal native-async helper
            raise RuntimeError("An asynchronous Typesense client is required.")
        try:
            await self._async_client.collections.create(self._collection_schema(num_dim))
        except ObjectAlreadyExists:
            schema = await self._async_client.collections[self._collection_name].retrieve()
            self._validate_collection_schema(schema, num_dim)
        self._async_validated_num_dim = num_dim

    def delete_collection(self) -> bool:
        """Delete the backing collection.

        Returns:
            ``True`` when the collection was deleted; ``False`` when it was
            already absent.
        """
        try:
            self._require_sync_client().collections[self._collection_name].delete()
        except ObjectNotFound:
            return False
        self._sync_validated_num_dim = None
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
        self._async_validated_num_dim = None
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

        resolved_ids = self._resolve_ids(documents, ids)
        vectors = self._embedding.embed_documents([document.page_content for document in documents])
        resolved_ids, payload, num_dim = self._prepare_documents(documents, resolved_ids, vectors)
        if num_dim is None:  # pragma: no cover - guarded by the empty check above
            return []
        self.create_collection(num_dim)
        import_parameters: DocumentWriteParameters = {"action": "upsert"}
        documents_api = self._require_sync_client().collections[self._collection_name].documents
        try:
            response = documents_api.import_(payload, import_parameters, batch_size=batch_size)
        except ObjectNotFound:
            # Recover if the collection was removed after this store cached its
            # schema validation.
            self._sync_validated_num_dim = None
            self._create_missing_collection(num_dim)
            response = documents_api.import_(payload, import_parameters, batch_size=batch_size)
        self._raise_for_import_failures(response)
        return resolved_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert text strings into Typesense.

        Args:
            texts: Text content to embed and store.
            metadatas: Optional metadata dictionaries aligned with ``texts``.
            ids: Optional stable Typesense IDs aligned with ``texts``.
            batch_size: Optional records per Typesense import request.

        Returns:
            IDs assigned to the texts, in input order.
        """
        if kwargs:
            raise TypeError(f"Unsupported add options: {', '.join(sorted(kwargs))}")
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
        return self.add_documents(documents, ids=ids, batch_size=batch_size)

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

        resolved_ids = self._resolve_ids(documents, ids)
        vectors = await self._embedding.aembed_documents(
            [document.page_content for document in documents]
        )
        resolved_ids, payload, num_dim = self._prepare_documents(documents, resolved_ids, vectors)
        if num_dim is None:  # pragma: no cover - guarded by the empty check above
            return []
        await self.acreate_collection(num_dim)
        import_parameters: DocumentWriteParameters = {"action": "upsert"}
        documents_api = self._async_client.collections[self._collection_name].documents
        try:
            response = await documents_api.import_(
                payload, import_parameters, batch_size=batch_size
            )
        except ObjectNotFound:
            self._async_validated_num_dim = None
            await self._acreate_missing_collection(num_dim)
            response = await documents_api.import_(
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
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously embed and upsert text strings into Typesense.

        See :meth:`add_texts` for metadata and ID alignment rules.
        """
        if kwargs:
            raise TypeError(f"Unsupported add options: {', '.join(sorted(kwargs))}")
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
        return await self.aadd_documents(documents, ids=ids, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _search_by_vector(
        self,
        embedding: list[float],
        k: int,
        *,
        filter: Filter | None,
        search_parameters: TypesenseSearchParameters | None,
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
        response = (
            self._require_sync_client()
            .collections[self._collection_name]
            .documents.search(parameters)
        )
        return self._parse_search_response(response, include_vectors=include_vectors)

    async def _asearch_by_vector(
        self,
        embedding: list[float],
        k: int,
        *,
        filter: Filter | None,
        search_parameters: TypesenseSearchParameters | None,
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
        response = await self._async_client.collections[self._collection_name].documents.search(
            parameters
        )
        return self._parse_search_response(response, include_vectors=include_vectors)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
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
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional threshold for exact flat search.
        Returns:
            Documents in Typesense result order. This is increasing vector
            distance unless explicit curation options reorder the hits.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
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
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents with raw Typesense vector distances.

        The second tuple item is a distance, not a LangChain relevance score:
        lower values indicate more similar documents. Use the inherited
        :meth:`similarity_search_with_relevance_scores` method when a bounded
        relevance score is required.

        Args:
            query: Text to embed and search for.
            k: Maximum number of results.
            filter: Metadata equality mapping or raw Typesense filter expression.
            search_parameters: Additional non-managed Typesense parameters.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.
        Returns:
            ``(Document, distance)`` pairs in Typesense result order. This is
            nearest to farthest unless explicit curation options reorder the hits.
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
        search_parameters: TypesenseSearchParameters | None = None,
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
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.
        Returns:
            Documents in Typesense result order. This is increasing vector
            distance unless explicit curation options reorder the hits.
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
        if self._vec_dist == "ip":
            raise ValueError(
                "Inner-product distances cannot be normalized to LangChain's "
                "[0, 1] relevance range without constraints on the embeddings. "
                "Use `similarity_search_with_score` to consume raw distances."
            )
        return self._cosine_distance_to_relevance

    def close(self) -> None:
        """Close the synchronous client's underlying HTTP resources.

        This also closes a client supplied by the caller. Call it only when that
        client is no longer needed; the client must not be reused afterward.
        """
        if self._client is not None:
            self._client.api_call.close()

    async def aclose(self) -> None:
        """Close the configured clients' underlying HTTP resources.

        The asynchronous client's resources are closed first, followed by the
        synchronous client. Supplied clients must not be reused afterward. This
        method is safe when no asynchronous client was supplied.
        """
        try:
            if self._async_client is not None:
                await self._async_client.api_call.aclose()
        finally:
            self.close()

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
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
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
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
            )
        ]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Asynchronously return documents and raw vector distances.

        The returned Typesense distances use the configured metric; lower is
        better. Parameters match :meth:`similarity_search_with_score`. Without an
        async client, the synchronous implementation runs in a worker thread.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
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
            )
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
        search_parameters: TypesenseSearchParameters | None = None,
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
    # Hybrid keyword and vector search
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        *,
        alpha: float = 0.3,
        query_by: str | Sequence[str] | None = None,
        filter: Filter | None = None,
        search_parameters: TypesenseHybridSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents ranked by Typesense keyword/vector rank fusion.

        ``alpha`` is the vector-ranking weight: ``0`` favors keyword ranking and
        ``1`` favors vector ranking. Keyword search targets the configured text
        field unless ``query_by`` supplies other indexed string fields.
        """
        if kwargs:
            raise TypeError(f"Unsupported hybrid search options: {', '.join(sorted(kwargs))}")
        return [
            document
            for document, _ in self.hybrid_search_with_score(
                query,
                k=k,
                alpha=alpha,
                query_by=query_by,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
            )
        ]

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        alpha: float = 0.3,
        query_by: str | Sequence[str] | None = None,
        filter: Filter | None = None,
        search_parameters: TypesenseHybridSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents with Typesense rank-fusion scores.

        Higher fusion scores rank first. They are rank-dependent hybrid scores,
        not raw vector distances or normalized LangChain relevance scores, and
        should not be compared across separate searches.
        """
        if kwargs:
            raise TypeError(f"Unsupported hybrid search options: {', '.join(sorted(kwargs))}")
        self._validate_k(k)
        if k == 0:
            return []
        normalized_query_by = self._validate_hybrid_search_options(
            query, k, alpha, query_by, search_parameters
        )
        embedding = self._embedding.embed_query(query)
        parameters = self._build_hybrid_search_parameters(
            query,
            embedding,
            k,
            alpha=alpha,
            query_by=normalized_query_by,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
        )
        response = (
            self._require_sync_client()
            .collections[self._collection_name]
            .documents.search(parameters)
        )
        return self._parse_hybrid_search_response(response)

    async def ahybrid_search(
        self,
        query: str,
        k: int = 4,
        *,
        alpha: float = 0.3,
        query_by: str | Sequence[str] | None = None,
        filter: Filter | None = None,
        search_parameters: TypesenseHybridSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents ranked by keyword/vector fusion."""
        if kwargs:
            raise TypeError(f"Unsupported hybrid search options: {', '.join(sorted(kwargs))}")
        return [
            document
            for document, _ in await self.ahybrid_search_with_score(
                query,
                k=k,
                alpha=alpha,
                query_by=query_by,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
            )
        ]

    async def ahybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        alpha: float = 0.3,
        query_by: str | Sequence[str] | None = None,
        filter: Filter | None = None,
        search_parameters: TypesenseHybridSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Asynchronously return documents with rank-fusion scores."""
        if kwargs:
            raise TypeError(f"Unsupported hybrid search options: {', '.join(sorted(kwargs))}")
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.hybrid_search_with_score,
                query,
                k=k,
                alpha=alpha,
                query_by=query_by,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
            )
        self._validate_k(k)
        if k == 0:
            return []
        normalized_query_by = self._validate_hybrid_search_options(
            query, k, alpha, query_by, search_parameters
        )
        embedding = await self._embedding.aembed_query(query)
        parameters = self._build_hybrid_search_parameters(
            query,
            embedding,
            k,
            alpha=alpha,
            query_by=normalized_query_by,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
        )
        response = await self._async_client.collections[self._collection_name].documents.search(
            parameters
        )
        return self._parse_hybrid_search_response(response)

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

    def _require_cosine_mmr(self) -> None:
        """Ensure local LangChain MMR uses the collection's ranking metric."""
        if self._vec_dist != "cosine":
            raise ValueError(
                "LangChain MMR uses cosine similarity and is only available when "
                "`vec_dist='cosine'`."
            )

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
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
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
            filter: Metadata equality mapping or raw Typesense filter expression.
            search_parameters: Additional non-managed Typesense parameters.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            Up to ``k`` documents selected by MMR.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        self._validate_mmr(k, fetch_k, lambda_mult)
        self._require_cosine_mmr()
        if k == 0 or fetch_k == 0:
            return []
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return diverse documents for a caller-supplied embedding.

        Args:
            embedding: Query vector whose dimension matches the collection.
            k: Number of diverse documents to return.
            fetch_k: Number of candidates fetched before local MMR reranking.
            lambda_mult: Similarity/diversity trade-off in ``[0, 1]``.
            filter: Metadata equality mapping or raw Typesense filter expression.
            search_parameters: Additional non-managed Typesense parameters.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            Up to ``k`` documents selected by local MMR reranking.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        self._validate_mmr(k, fetch_k, lambda_mult)
        self._require_cosine_mmr()
        if k == 0 or fetch_k == 0:
            return []
        results = self._search_by_vector(
            embedding,
            fetch_k,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
            include_vectors=True,
        )
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
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents selected with MMR.

        Parameters and reranking semantics match
        :meth:`max_marginal_relevance_search`. Without an async client, the
        synchronous implementation runs in a worker thread.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        self._validate_mmr(k, fetch_k, lambda_mult)
        self._require_cosine_mmr()
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
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
            )
        embedding = await self._embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Filter | None = None,
        search_parameters: TypesenseSearchParameters | None = None,
        distance_threshold: float | None = None,
        ef: int | None = None,
        flat_search_cutoff: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return diverse documents for an embedding vector.

        Parameters and reranking semantics match
        :meth:`max_marginal_relevance_search_by_vector`. Without an async client,
        the synchronous implementation runs in a worker thread.
        """
        if kwargs:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(kwargs))}")
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.max_marginal_relevance_search_by_vector,
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                search_parameters=search_parameters,
                distance_threshold=distance_threshold,
                ef=ef,
                flat_search_cutoff=flat_search_cutoff,
            )
        self._validate_mmr(k, fetch_k, lambda_mult)
        self._require_cosine_mmr()
        if k == 0 or fetch_k == 0:
            return []
        results = await self._asearch_by_vector(
            embedding,
            fetch_k,
            filter=filter,
            search_parameters=search_parameters,
            distance_threshold=distance_threshold,
            ef=ef,
            flat_search_cutoff=flat_search_cutoff,
            include_vectors=True,
        )
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
        return document_from_typesense(
            raw,
            text_key=self._text_key,
            vector_key=self._vector_key,
            metadata_key=self._metadata_key,
        )

    _parse_export_response = staticmethod(parse_export_response)

    def _documents_in_requested_order(
        self,
        raw_documents: Sequence[Mapping[str, Any]],
        ids: Sequence[str],
    ) -> list[Document]:
        """Deserialize exported documents and retain the caller's first-seen order."""
        requested = dict.fromkeys(ids)
        by_id: dict[str, Document] = {}
        for raw in raw_documents:
            document = self._document_from_typesense(raw)
            if document.id in requested and document.id not in by_id:
                by_id[document.id] = document
        return [by_id[document_id] for document_id in requested if document_id in by_id]

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
        if not ids:
            return []
        unique_ids = list(dict.fromkeys(ids))
        response = (
            self._require_sync_client()
            .collections[self._collection_name]
            .documents.export({"filter_by": self._ids_filter(unique_ids)})
        )
        return self._documents_in_requested_order(self._parse_export_response(response), unique_ids)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Asynchronously return found documents for the supplied IDs.

        Missing documents are ignored and duplicate IDs are de-duplicated. Uses
        native asynchronous retrieval when configured, otherwise a worker thread.
        """
        if self._async_client is None:
            return await run_in_executor(None, self.get_by_ids, ids)
        if not ids:
            return []
        unique_ids = list(dict.fromkeys(ids))
        response = await self._async_client.collections[self._collection_name].documents.export(
            {"filter_by": self._ids_filter(unique_ids)}
        )
        return self._documents_in_requested_order(self._parse_export_response(response), unique_ids)

    _ids_filter = staticmethod(ids_filter)

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        delete_all_documents: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Delete selected documents, with an explicit guard for truncation.

        Args:
            ids: IDs to delete. An empty list is a no-op. ``None`` requires
                ``delete_all_documents=True`` and truncates every document.
            delete_all_documents: Explicit confirmation for collection truncation.

        Returns:
            ``True`` after a successful request.
        """
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(sorted(kwargs))}")
        parameters: DeleteQueryParameters
        if ids is None:
            if not delete_all_documents:
                raise ValueError(
                    "Refusing to delete every document without `delete_all_documents=True`."
                )
            parameters = {"truncate": True}
        elif delete_all_documents:
            raise ValueError("`delete_all_documents` cannot be combined with explicit IDs.")
        elif not ids:
            return True
        else:
            parameters = {"filter_by": self._ids_filter(ids)}
        self._require_sync_client().collections[self._collection_name].documents.delete(parameters)
        return True

    async def adelete(
        self,
        ids: list[str] | None = None,
        *,
        delete_all_documents: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Asynchronously delete documents, with guarded truncation.

        Parameters and error behavior match :meth:`delete`. Native asynchronous
        I/O is used when configured; otherwise a worker thread runs the
        synchronous operation.
        """
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(sorted(kwargs))}")
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.delete,
                ids,
                delete_all_documents=delete_all_documents,
            )
        parameters: DeleteQueryParameters
        if ids is None:
            if not delete_all_documents:
                raise ValueError(
                    "Refusing to delete every document without `delete_all_documents=True`."
                )
            parameters = {"truncate": True}
        elif delete_all_documents:
            raise ValueError("`delete_all_documents` cannot be combined with explicit IDs.")
        elif not ids:
            return True
        else:
            parameters = {"filter_by": self._ids_filter(ids)}
        await self._async_client.collections[self._collection_name].documents.delete(parameters)
        return True

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    _node_from_url = staticmethod(node_from_url)

    @classmethod
    def from_client_params(
        cls,
        embedding: Embeddings,
        *,
        typesense_url: str,
        api_key: str | None = None,
        client_mode: ClientMode = "sync",
        connection_timeout_seconds: float = 2.0,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        text_key: str = DEFAULT_TEXT_KEY,
        vector_key: str = DEFAULT_VECTOR_KEY,
        metadata_key: str = DEFAULT_METADATA_KEY,
        index_metadata: bool = True,
        vec_dist: VectorDistance = "cosine",
    ) -> TypesenseVectorStore:
        """Build a store and clients from Typesense connection parameters.

        ``client_mode`` controls which connection pools are created. ``api_key``
        falls back to ``TYPESENSE_API_KEY`` only when omitted.

        Args:
            embedding: Embedding model used by the returned store.
            typesense_url: Absolute Typesense HTTP(S) URL. Port 80 is inferred for
                HTTP and port 443 for HTTPS when the URL omits a port. An explicit
                port is preserved.
            api_key: Typesense API key.
            client_mode: Create a sync client (the default), async client, or both.
            connection_timeout_seconds: Client request timeout.
            collection_name: Backing collection name.
            text_key: Typesense field used for LangChain ``page_content``.
            vector_key: Typesense vector field.
            metadata_key: Typesense nested metadata field.
            index_metadata: Whether newly created metadata is indexed.
            vec_dist: Typesense vector metric, ``"cosine"`` or ``"ip"``.

        Returns:
            A store with the requested client connection pool(s).

        Raises:
            ValueError: If the URL, API key, mode, or timeout is invalid.
        """
        if client_mode not in ("sync", "async", "both"):
            raise ValueError("`client_mode` must be 'sync', 'async', or 'both'.")
        if not math.isfinite(connection_timeout_seconds) or connection_timeout_seconds <= 0:
            raise ValueError("`connection_timeout_seconds` must be finite and greater than 0.")
        resolved_api_key = (
            get_from_env("api_key", "TYPESENSE_API_KEY") if api_key is None else api_key
        )
        if not resolved_api_key.strip():
            raise ValueError("`api_key` is required and must not be empty.")
        config: ConfigDict = {
            "nodes": [cls._node_from_url(typesense_url)],
            "api_key": resolved_api_key,
            "connection_timeout_seconds": connection_timeout_seconds,
        }
        return cls(
            client=Client(config) if client_mode in ("sync", "both") else None,
            async_client=(AsyncClient(config) if client_mode in ("async", "both") else None),
            embedding=embedding,
            collection_name=collection_name,
            text_key=text_key,
            vector_key=vector_key,
            metadata_key=metadata_key,
            index_metadata=index_metadata,
            vec_dist=vec_dist,
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a populated store from documents and their metadata.

        LangChain's base implementation passes every document ID when at least
        one document has an ID.  That includes ``None`` for mixed-ID batches,
        which Typesense cannot accept.  Generate IDs only for the missing entries
        while preserving every caller-provided ID.
        """
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        if "ids" not in kwargs:
            document_ids = [document.id for document in documents]
            if any(document_ids):
                kwargs["ids"] = [document_id or str(uuid.uuid4()) for document_id in document_ids]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Asynchronously create a populated store from documents.

        Mixed batches use generated IDs for documents without IDs, matching
        :meth:`from_documents` while retaining IDs supplied by the caller.
        """
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        if "ids" not in kwargs:
            document_ids = [document.id for document in documents]
            if any(document_ids):
                kwargs["ids"] = [document_id or str(uuid.uuid4()) for document_id in document_ids]
        return await cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        client: Client | None = None,
        async_client: AsyncClient | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        text_key: str = DEFAULT_TEXT_KEY,
        vector_key: str = DEFAULT_VECTOR_KEY,
        metadata_key: str = DEFAULT_METADATA_KEY,
        index_metadata: bool = True,
        vec_dist: VectorDistance = "cosine",
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a store and add texts using native async I/O when available."""
        if kwargs:
            raise TypeError(f"Unsupported constructor options: {', '.join(sorted(kwargs))}")
        store = cls(
            client=client,
            async_client=async_client,
            embedding=embedding,
            collection_name=collection_name,
            text_key=text_key,
            vector_key=vector_key,
            metadata_key=metadata_key,
            index_metadata=index_metadata,
            vec_dist=vec_dist,
        )
        await store.aadd_texts(texts, metadatas=metadatas, ids=ids, batch_size=batch_size)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        client: Client | None = None,
        async_client: AsyncClient | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        text_key: str = DEFAULT_TEXT_KEY,
        vector_key: str = DEFAULT_VECTOR_KEY,
        metadata_key: str = DEFAULT_METADATA_KEY,
        index_metadata: bool = True,
        vec_dist: VectorDistance = "cosine",
        batch_size: int | None = None,
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
            async_client: Optional configured asynchronous Typesense client.
            collection_name: Backing collection name.
            text_key: Typesense field used for LangChain ``page_content``.
            vector_key: Typesense vector field.
            metadata_key: Typesense nested metadata field.
            index_metadata: Whether newly created metadata is indexed.
            vec_dist: Typesense vector metric.
            batch_size: Optional records per Typesense import request.

        Returns:
            A populated :class:`TypesenseVectorStore`.

        Raises:
            ValueError: If input lengths or configuration are invalid.
            TypesenseImportError: If Typesense partially rejects the import.
        """
        if kwargs:
            raise TypeError(f"Unsupported constructor options: {', '.join(sorted(kwargs))}")
        if client is None:
            raise ValueError("`client` (a `typesense.Client`) is required.")
        store = cls(
            client=client,
            async_client=async_client,
            embedding=embedding,
            collection_name=collection_name,
            text_key=text_key,
            vector_key=vector_key,
            metadata_key=metadata_key,
            index_metadata=index_metadata,
            vec_dist=vec_dist,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids, batch_size=batch_size)
        return store


# Compatibility alias for applications migrating from langchain-community.
Typesense = TypesenseVectorStore

__all__ = [
    "ClientMode",
    "Typesense",
    "TypesenseCollectionError",
    "TypesenseHybridSearchParameters",
    "TypesenseImportError",
    "TypesenseSearchParameters",
    "TypesenseVectorStore",
    "TypesenseVectorStoreError",
    "VectorDistance",
]
