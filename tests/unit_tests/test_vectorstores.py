"""Unit tests for the Typesense vector store; no network is used."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typesense import AsyncClient, Client
from typesense.exceptions import ObjectNotFound

from langchain_typesense import (
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStore,
    TypesenseVectorStoreError,
)


class FakeEmbeddings(Embeddings):
    """Small deterministic embedding implementation for unit tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("a")), 1.0]


def make_sync_store() -> tuple[TypesenseVectorStore, MagicMock, MagicMock]:
    client = MagicMock()
    collections = MagicMock()
    collection = MagicMock()
    client.collections = collections
    collections.__getitem__.return_value = collection
    store = TypesenseVectorStore(
        client=cast(Client, client),
        embedding=FakeEmbeddings(),
        collection_name="test-collection",
    )
    return store, collections, collection


def make_async_store() -> tuple[TypesenseVectorStore, MagicMock, MagicMock]:
    store, _, _ = make_sync_store()
    async_client = MagicMock()
    collections = MagicMock()
    collection = MagicMock()
    async_client.collections = collections
    collections.__getitem__.return_value = collection
    store._async_client = cast(AsyncClient, async_client)
    return store, collections, collection


def collection_schema(num_dim: int = 3) -> dict[str, Any]:
    return {
        "name": "test-collection",
        "created_at": 0,
        "num_documents": 0,
        "num_memory_shards": 1,
        "enable_nested_fields": True,
        "fields": [
            {"name": "text", "type": "string"},
            {"name": "vec", "type": "float[]", "num_dim": num_dim},
            {"name": "metadata", "type": "object", "optional": True},
        ],
    }


def search_response(*, include_vectors: bool = False) -> dict[str, Any]:
    first = {
        "id": "doc-1",
        "text": "bar",
        "metadata": {"source": "tweet", "id": 7},
    }
    second = {
        "id": "doc-2",
        "text": "foo",
        "metadata": {"source": "news"},
    }
    if include_vectors:
        first["vec"] = [3.0, 1.0, 1.0]
        second["vec"] = [3.0, 0.0, 1.0]
    return {
        "hits": [
            {"document": first, "vector_distance": 0.1},
            {"document": second, "vector_distance": 0.4},
        ]
    }


def test_typesense_vectorstore_is_a_vectorstore() -> None:
    store, _, _ = make_sync_store()

    assert isinstance(store, VectorStore)
    assert store.collection_name == "test-collection"
    assert store.embeddings is not None


def test_constructor_rejects_reserved_or_duplicate_fields() -> None:
    _, collections, _ = make_sync_store()
    client = cast(Client, MagicMock(collections=collections))

    with pytest.raises(ValueError, match="must be distinct"):
        TypesenseVectorStore(client, FakeEmbeddings(), text_key="id")
    with pytest.raises(ValueError, match="must be distinct"):
        TypesenseVectorStore(client, FakeEmbeddings(), text_key="same", vector_key="same")


def test_dict_filter_targets_nested_metadata_and_escapes_values() -> None:
    store, _, _ = make_sync_store()

    assert store._to_filter_by({"source": "tweets", "year": 2024}) == (
        "metadata.source:=tweets && metadata.year:=2024"
    )
    assert store._to_filter_by({"category": ["Running Shoes", "Boots"]}) == (
        "metadata.category:=[`Running Shoes`,Boots]"
    )
    assert store._to_filter_by({"published": True}) == "metadata.published:=true"


def test_string_filter_passes_through_unchanged() -> None:
    store, _, _ = make_sync_store()
    raw = "year:>=2020 && category:=[electronics,books]"

    assert store._to_filter_by(raw) == raw


def test_add_documents_creates_schema_and_preserves_reserved_metadata_keys() -> None:
    store, collections, collection = make_sync_store()
    collection.retrieve.side_effect = ObjectNotFound("missing")
    collection.documents.import_.return_value = [{"success": True}]
    documents = [Document(page_content="alpha", metadata={"id": 99, "text": "metadata text"})]

    ids = store.add_documents(documents, ids=["doc-1"])

    assert ids == ["doc-1"]
    schema = collections.create.call_args.args[0]
    assert schema["enable_nested_fields"] is True
    assert schema["fields"][1] == {
        "name": "vec",
        "type": "float[]",
        "num_dim": 3,
        "vec_dist": "cosine",
    }
    payload = collection.documents.import_.call_args.args[0]
    assert payload == [
        {
            "id": "doc-1",
            "text": "alpha",
            "vec": [5.0, 2.0, 1.0],
            "metadata": {"id": 99, "text": "metadata text"},
        }
    ]
    assert documents[0].id is None


def test_add_documents_validates_lengths_and_empty_input() -> None:
    store, _, collection = make_sync_store()

    assert store.add_documents([]) == []
    collection.documents.import_.assert_not_called()
    with pytest.raises(ValueError, match="number of IDs"):
        store.add_documents([Document(page_content="one")], ids=["1", "2"])
    with pytest.raises(ValueError, match="URL-safe"):
        store.add_documents([Document(page_content="one")], ids=["not safe"])
    with pytest.raises(ValueError, match="number of metadatas"):
        store.add_texts(["one"], metadatas=[])


def test_add_documents_validates_existing_collection_schema() -> None:
    store, _, collection = make_sync_store()
    collection.retrieve.return_value = collection_schema(num_dim=5)

    with pytest.raises(TypesenseCollectionError, match="num_dim=3"):
        store.add_documents([Document(page_content="alpha")], ids=["doc-1"])


def test_bulk_import_failures_are_not_silently_ignored() -> None:
    store, _, collection = make_sync_store()
    collection.retrieve.return_value = collection_schema()
    collection.documents.import_.return_value = [
        {"success": False, "code": 400, "error": "bad vector", "document": {"id": "x"}}
    ]

    with pytest.raises(TypesenseImportError, match="bad vector") as error:
        store.add_documents([Document(page_content="alpha")], ids=["x"])

    assert error.value.failures[0]["code"] == 400


def test_similarity_search_builds_bounded_query_and_restores_metadata() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response()

    results = store.similarity_search(
        "bar",
        k=2,
        filter={"source": "tweet"},
        distance_threshold=0.5,
        ef=40,
        search_parameters={"enable_lazy_filter": True},
    )

    assert results == [
        Document(id="doc-1", page_content="bar", metadata={"source": "tweet", "id": 7}),
        Document(id="doc-2", page_content="foo", metadata={"source": "news"}),
    ]
    parameters = collection.documents.search.call_args.args[0]
    assert parameters["q"] == "*"
    assert parameters["per_page"] == 2
    assert parameters["exclude_fields"] == "vec"
    assert "k:2" in parameters["vector_query"]
    assert "distance_threshold:0.5" in parameters["vector_query"]
    assert "ef:40" in parameters["vector_query"]
    assert parameters["filter_by"] == "metadata.source:=tweet"
    assert parameters["enable_lazy_filter"] is True


def test_similarity_search_returns_empty_for_missing_collection_or_zero_k() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.side_effect = ObjectNotFound("missing")

    assert store.similarity_search("query", k=1) == []
    assert store.similarity_search("query", k=0) == []


def test_similarity_search_rejects_internal_parameter_override() -> None:
    store, _, _ = make_sync_store()

    with pytest.raises(ValueError, match="integration-managed"):
        store.similarity_search("query", search_parameters={"per_page": 100})
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        store.similarity_search("query", k=-1)


def test_relevance_scores_normalize_typesense_cosine_distance() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response()

    results = store.similarity_search_with_relevance_scores("bar", k=2)

    assert results[0][1] == pytest.approx(0.95)
    assert results[1][1] == pytest.approx(0.8)


def test_mmr_search_requests_vectors_and_returns_documents() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response(include_vectors=True)

    results = store.max_marginal_relevance_search("bar", k=1, fetch_k=2)

    assert len(results) == 1
    assert results[0].id == "doc-1"
    parameters = collection.documents.search.call_args.args[0]
    assert "exclude_fields" not in parameters
    assert parameters["per_page"] == 2


def test_get_by_ids_deduplicates_and_ignores_missing_ids() -> None:
    store, _, collection = make_sync_store()
    first = MagicMock()
    missing = MagicMock()
    first.retrieve.return_value = {
        "id": "doc-1",
        "text": "bar",
        "vec": [1.0, 0.0, 0.0],
        "metadata": {"source": "tweet"},
    }
    missing.retrieve.side_effect = ObjectNotFound("missing")
    collection.documents.__getitem__.side_effect = lambda document_id: {
        "doc-1": first,
        "missing": missing,
    }[document_id]

    results = store.get_by_ids(["doc-1", "missing", "doc-1"])

    assert results == [Document(id="doc-1", page_content="bar", metadata={"source": "tweet"})]
    first.retrieve.assert_called_once()


def test_delete_uses_safe_bulk_filter_and_supports_truncate() -> None:
    store, _, collection = make_sync_store()

    assert store.delete(["doc-1", "doc-2"]) is True
    collection.documents.delete.assert_called_with({"filter_by": "id:=[doc-1,doc-2]"})
    assert store.delete() is True
    collection.documents.delete.assert_called_with({"truncate": True})


def test_unexpected_bulk_response_raises_integration_error() -> None:
    with pytest.raises(TypesenseVectorStoreError, match="unexpected bulk import"):
        TypesenseVectorStore._raise_for_import_failures("not-json-records")
    with pytest.raises(TypesenseVectorStoreError, match="malformed record"):
        TypesenseVectorStore._raise_for_import_failures([{"not_success": True}])


def test_from_client_params_rejects_conflicting_api_key_aliases() -> None:
    with pytest.raises(ValueError, match="must not conflict"):
        TypesenseVectorStore.from_client_params(
            FakeEmbeddings(),
            api_key="new-key",
            typesense_api_key="old-key",
        )


@pytest.mark.asyncio
async def test_native_async_add_and_search() -> None:
    store, collections, collection = make_async_store()
    collection.retrieve = AsyncMock(side_effect=ObjectNotFound("missing"))
    collections.create = AsyncMock(return_value=collection_schema())
    collection.documents.import_ = AsyncMock(return_value=[{"success": True}])
    collection.documents.search = AsyncMock(return_value=search_response())

    ids = await store.aadd_documents([Document(page_content="bar")], ids=["doc-1"])
    results = await store.asimilarity_search("bar", k=2)

    assert ids == ["doc-1"]
    assert [document.id for document in results] == ["doc-1", "doc-2"]
    collection.documents.import_.assert_awaited_once()
    collection.documents.search.assert_awaited_once()
