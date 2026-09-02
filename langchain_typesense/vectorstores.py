"""Typesense vector storage for LangChain.

Documents use configurable text, vector, and nested-metadata fields. Collections
are managed lazily by default, and scored searches return raw Typesense distance.
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

    Metadata is stored in one nested field, preventing keys such as ``id`` or
    ``text`` from overwriting managed fields. It is indexed by default so mapping
    filters can target nested values. ``"cosine"`` supports all features;
    ``"ip"`` does not support MMR or bounded relevance scores.
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
        """Initialize the store without network I/O.

        Collection management is deferred until the first non-empty write.

        Args:
            client: Configured sync client, or ``None`` for an async-only store.
            embedding: Model used to embed documents and queries.
            collection_name: Typesense collection name.
            async_client: Optional async client. Without one, async methods run
                their sync counterparts in a worker thread.
            text_key: Field that stores document text.
            vector_key: Field that stores embeddings.
            metadata_key: Nested field that stores document metadata.
            index_metadata: Whether newly created collections index metadata.
            vec_dist: Vector distance metric: ``"cosine"`` or ``"ip"``.

        Raises:
            ValueError: If clients or configuration are invalid.
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
        """Return the configured synchronous client, if any."""
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
        """Create the collection or validate its managed schema.

        Successful validation is cached by dimension for this sync store instance.

        Args:
            num_dim: Embedding dimension expected by the vector field.

        Raises:
            ValueError: If ``num_dim`` is not positive.
            TypesenseCollectionError: If the existing schema is incompatible.
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
        """Create or validate the collection asynchronously.

        Native async validation has its own dimension cache; otherwise the sync
        method runs in a worker thread.

        Args:
            num_dim: Embedding dimension expected by the vector field.

        Raises:
            ValueError: If ``num_dim`` is not positive.
            TypesenseCollectionError: If the existing schema is incompatible.
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
        """Delete the collection.

        Returns:
            Whether the collection existed.
        """
        try:
            self._require_sync_client().collections[self._collection_name].delete()
        except ObjectNotFound:
            return False
        self._sync_validated_num_dim = None
        return True

    async def adelete_collection(self) -> bool:
        """Delete the collection asynchronously.

        Returns:
            Whether the collection existed.
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
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert documents, returning their IDs.

        By default, the first write creates or validates the collection. Set
        ``create_collection_if_not_exists=False`` to skip that request for a
        caller-managed collection. Bulk imports can partially succeed; failures
        are reported by :class:`TypesenseImportError`.

        Args:
            documents: Documents to embed and store.
            ids: Optional stable IDs. Missing IDs are generated.
            batch_size: Optional records per import request.
            create_collection_if_not_exists: Whether to create or validate the
                collection before writing. Disable only for a caller-managed schema.

        Returns:
            Document IDs in input order.

        Raises:
            TypeError: If unsupported add options are supplied.
            ValueError: If IDs, batch size, or embeddings are invalid.
            TypesenseCollectionError: If a managed collection is incompatible.
            TypesenseImportError: If any records fail to import.
            ObjectNotFound: If collection management is disabled and it is absent.
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
        if create_collection_if_not_exists:
            self.create_collection(num_dim)
        import_parameters: DocumentWriteParameters = {"action": "upsert"}
        documents_api = self._require_sync_client().collections[self._collection_name].documents
        try:
            response = documents_api.import_(payload, import_parameters, batch_size=batch_size)
        except ObjectNotFound:
            if not create_collection_if_not_exists:
                raise
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
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert texts.

        Args:
            texts: Texts to embed and store.
            metadatas: Optional metadata aligned with ``texts``.
            ids: Optional stable IDs aligned with ``texts``.
            batch_size: Optional records per import request.
            create_collection_if_not_exists: Whether to create or validate the
                collection before writing.

        Returns:
            Document IDs in input order.

        Raises:
            TypeError: If unsupported add options are supplied.
            ValueError: If inputs or embeddings are invalid.
            TypesenseCollectionError: If a managed collection is incompatible.
            TypesenseImportError: If any records fail to import.
            ObjectNotFound: If collection management is disabled and it is absent.
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
        return self.add_documents(
            documents,
            ids=ids,
            batch_size=batch_size,
            create_collection_if_not_exists=create_collection_if_not_exists,
        )

    async def aadd_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        batch_size: int | None = None,
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert documents asynchronously.

        Parameters and errors match :meth:`add_documents`. Native async I/O is
        used when configured; otherwise the sync method runs in a worker thread.
        """
        if self._async_client is None:
            return await run_in_executor(
                None,
                self.add_documents,
                documents,
                ids=ids,
                batch_size=batch_size,
                create_collection_if_not_exists=create_collection_if_not_exists,
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
        if create_collection_if_not_exists:
            await self.acreate_collection(num_dim)
        import_parameters: DocumentWriteParameters = {"action": "upsert"}
        documents_api = self._async_client.collections[self._collection_name].documents
        try:
            response = await documents_api.import_(
                payload, import_parameters, batch_size=batch_size
            )
        except ObjectNotFound:
            if not create_collection_if_not_exists:
                raise
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
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and upsert texts asynchronously.

        Parameters and errors match :meth:`add_texts`.
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
        return await self.aadd_documents(
            documents,
            ids=ids,
            batch_size=batch_size,
            create_collection_if_not_exists=create_collection_if_not_exists,
        )

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
        """Return documents nearest to an embedded text query.

        Args:
            query: Text to embed and search for.
            k: Maximum results to return.
            filter: Metadata mapping or raw Typesense filter expression.
            search_parameters: Additional safe Typesense search options.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            Documents in Typesense result order.

        Raises:
            TypeError: If unsupported search options are supplied.
            ValueError: If search options are invalid.
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

        Lower distance is better. Use
        :meth:`similarity_search_with_relevance_scores` for bounded cosine scores.

        Args:
            query: Text to embed and search for.
            k: Maximum results to return.
            filter: Metadata mapping or raw Typesense filter expression.
            search_parameters: Additional safe Typesense search options.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            ``(document, distance)`` pairs in Typesense result order.

        Raises:
            TypeError: If unsupported search options are supplied.
            ValueError: If search options are invalid.
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
        """Return documents nearest to an embedding vector.

        Args:
            embedding: Query vector matching the collection dimension.
            k: Maximum results to return.
            filter: Metadata mapping or raw Typesense filter expression.
            search_parameters: Additional safe Typesense search options.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            Documents in Typesense result order.

        Raises:
            TypeError: If unsupported search options are supplied.
            ValueError: If search options are invalid.
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
        """Return documents nearest to an embedded text query asynchronously.

        Parameters match :meth:`similarity_search`. Native async I/O is used when
        configured; otherwise the sync method runs in a worker thread.
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
        """Return documents and raw distances asynchronously.

        Parameters match :meth:`similarity_search_with_score`; lower distance is
        better. Without an async client, the sync method runs in a worker thread.
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
        """Return documents nearest to an embedding vector asynchronously.

        Parameters match :meth:`similarity_search_by_vector`.
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

        Parameters match :meth:`hybrid_search_with_score`; scores are omitted.
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

        Higher is better. Scores are query-specific, not normalized relevance.

        Args:
            query: Text used for keyword and vector search.
            k: Maximum results to return.
            alpha: Vector weight from ``0`` (keyword) to ``1`` (vector).
            query_by: Keyword fields. Defaults to the configured text field.
            filter: Metadata mapping or raw Typesense filter expression.
            search_parameters: Additional safe hybrid-search options.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            ``(document, fusion_score)`` pairs in rank-fusion order.

        Raises:
            TypeError: If unsupported hybrid-search options are supplied.
            ValueError: If hybrid-search options are invalid.
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
        """Return documents ranked by keyword/vector fusion asynchronously.

        Parameters match :meth:`hybrid_search`.
        """
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
        """Return documents with rank-fusion scores asynchronously.

        Parameters match :meth:`hybrid_search_with_score`.
        """
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
        """Return documents selected with maximal marginal relevance.

        Typesense returns ``fetch_k`` candidates for local reranking.
        ``lambda_mult=1`` favors similarity; ``0`` favors diversity.

        Args:
            query: Text to embed and search for.
            k: Maximum documents to return.
            fetch_k: Candidates to fetch before local reranking.
            lambda_mult: Similarity/diversity weight in ``[0, 1]``.
            filter: Metadata mapping or raw Typesense filter expression.
            search_parameters: Additional safe Typesense search options.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            Up to ``k`` MMR-selected documents.

        Raises:
            TypeError: If unsupported search options are supplied.
            ValueError: If options are invalid or cosine distance is not configured.
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
        """Return MMR-selected documents for an embedding vector.

        Args:
            embedding: Query vector matching the collection dimension.
            k: Maximum documents to return.
            fetch_k: Candidates to fetch before local reranking.
            lambda_mult: Similarity/diversity weight in ``[0, 1]``.
            filter: Metadata mapping or raw Typesense filter expression.
            search_parameters: Additional safe Typesense search options.
            distance_threshold: Optional maximum vector distance.
            ef: Optional HNSW search expansion factor.
            flat_search_cutoff: Optional exact-search cutoff.

        Returns:
            Up to ``k`` MMR-selected documents.

        Raises:
            TypeError: If unsupported search options are supplied.
            ValueError: If options are invalid or cosine distance is not configured.
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
        """Return MMR-selected documents for a text query asynchronously.

        Parameters match :meth:`max_marginal_relevance_search`.
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
        """Return MMR-selected documents for an embedding asynchronously.

        Parameters match :meth:`max_marginal_relevance_search_by_vector`.
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

        Duplicate IDs are collapsed and missing documents are ignored. Result order
        is not part of the contract.

        Args:
            ids: Typesense document IDs to retrieve.

        Returns:
            Found documents, which may be fewer than requested.

        Raises:
            ValueError: If an ID is invalid.
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
        """Return found documents for the supplied IDs asynchronously.

        Parameters and ordering semantics match :meth:`get_by_ids`.
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
        """Delete documents.

        ``ids=None`` requires ``delete_all_documents=True`` to prevent accidental
        truncation. An empty ID list is a no-op.

        Args:
            ids: IDs to delete, an empty list for a no-op, or ``None`` to truncate.
            delete_all_documents: Required confirmation when ``ids`` is ``None``.

        Returns:
            ``True`` after a successful request or no-op.

        Raises:
            TypeError: If unsupported delete options are supplied.
            ValueError: If arguments or IDs are invalid.
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
        """Delete documents asynchronously.

        Parameters and truncation safeguards match :meth:`delete`.
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

        ``client_mode`` selects the connection pools. A missing ``api_key`` uses
        ``TYPESENSE_API_KEY``; URL ports default to 80 for HTTP and 443 for HTTPS.

        Args:
            embedding: Model used to embed documents and queries.
            typesense_url: Absolute Typesense HTTP(S) URL.
            api_key: Typesense API key.
            client_mode: Whether to create sync, async, or both clients.
            connection_timeout_seconds: Client request timeout.
            collection_name: Typesense collection name.
            text_key: Field that stores document text.
            vector_key: Field that stores embeddings.
            metadata_key: Nested field that stores document metadata.
            index_metadata: Whether newly created collections index metadata.
            vec_dist: Vector distance metric: ``"cosine"`` or ``"ip"``.

        Returns:
            A store with the requested client connection pools.

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
        *,
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a populated store, generating only missing document IDs.

        Other parameters and errors match :meth:`from_texts`.
        """
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        if "ids" not in kwargs:
            document_ids = [document.id for document in documents]
            if any(document_ids):
                kwargs["ids"] = [document_id or str(uuid.uuid4()) for document_id in document_ids]
        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            create_collection_if_not_exists=create_collection_if_not_exists,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        *,
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a populated store asynchronously.

        Missing document IDs are generated. Other parameters and errors match
        :meth:`afrom_texts`.
        """
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        if "ids" not in kwargs:
            document_ids = [document.id for document in documents]
            if any(document_ids):
                kwargs["ids"] = [document_id or str(uuid.uuid4()) for document_id in document_ids]
        return await cls.afrom_texts(
            texts,
            embedding,
            metadatas=metadatas,
            create_collection_if_not_exists=create_collection_if_not_exists,
            **kwargs,
        )

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
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a store and add texts asynchronously.

        Parameters and errors match :meth:`from_texts`. Native async I/O is used
        when an async client is configured.
        """
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
        await store.aadd_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            create_collection_if_not_exists=create_collection_if_not_exists,
        )
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
        create_collection_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> TypesenseVectorStore:
        """Create a store and add texts with an existing sync client.

        Args:
            texts: Texts to embed and store.
            embedding: Model used to embed the texts.
            metadatas: Optional metadata aligned with ``texts``.
            ids: Optional stable IDs aligned with ``texts``.
            client: Configured synchronous Typesense client.
            async_client: Optional async client for later async methods.
            collection_name: Typesense collection name.
            text_key: Field that stores document text.
            vector_key: Field that stores embeddings.
            metadata_key: Nested field that stores document metadata.
            index_metadata: Whether newly created collections index metadata.
            vec_dist: Vector distance metric.
            batch_size: Optional records per import request.
            create_collection_if_not_exists: Whether to create or validate the
                collection before writing.
            **kwargs: Reserved for LangChain compatibility; unsupported options fail.

        Returns:
            A populated store.

        Raises:
            TypeError: If unsupported constructor options are supplied.
            ValueError: If inputs or configuration are invalid.
            TypesenseCollectionError: If a managed collection is incompatible.
            TypesenseImportError: If any records fail to import.
            ObjectNotFound: If collection management is disabled and it is absent.
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
        store.add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            create_collection_if_not_exists=create_collection_if_not_exists,
        )
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
