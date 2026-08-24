"""Unit tests for the Typesense vector store; no network is used."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typesense import AsyncClient, Client
from typesense.exceptions import ObjectAlreadyExists, ObjectNotFound

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


def collection_schema(num_dim: int = 3, vec_dist: str = "cosine") -> dict[str, Any]:
    return {
        "name": "test-collection",
        "created_at": 0,
        "num_documents": 0,
        "num_memory_shards": 1,
        "enable_nested_fields": True,
        "fields": [
            {"name": "text", "type": "string"},
            {
                "name": "vec",
                "type": "float[]",
                "num_dim": num_dim,
                "vec_dist": vec_dist,
            },
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
    with pytest.raises(ValueError, match="At least one"):
        TypesenseVectorStore(None, FakeEmbeddings())
    with pytest.raises(ValueError, match="vec_dist"):
        TypesenseVectorStore(client, FakeEmbeddings(), vec_dist=cast(Any, "euclidean"))


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


def test_custom_field_keys_are_used_for_schema_write_search_filter_and_read() -> None:
    client = MagicMock()
    collections = MagicMock()
    collection = MagicMock()
    client.collections = collections
    collections.__getitem__.return_value = collection
    store = TypesenseVectorStore(
        client=cast(Client, client),
        embedding=FakeEmbeddings(),
        collection_name="custom-fields",
        text_key="body",
        vector_key="embedding",
        metadata_key="attributes",
    )
    collection.retrieve.side_effect = ObjectNotFound("missing")
    collection.documents.import_.return_value = [{"success": True}]

    store.add_documents(
        [Document(page_content="alpha", metadata={"source": "docs"})], ids=["doc-1"]
    )

    schema = collections.create.call_args.args[0]
    assert [field["name"] for field in schema["fields"]] == [
        "body",
        "embedding",
        "attributes",
    ]
    assert collection.documents.import_.call_args.args[0] == [
        {
            "id": "doc-1",
            "body": "alpha",
            "embedding": [5.0, 2.0, 1.0],
            "attributes": {"source": "docs"},
        }
    ]

    collection.documents.search.return_value = {
        "hits": [
            {
                "document": {
                    "id": "doc-1",
                    "body": "alpha",
                    "attributes": {"source": "docs"},
                },
                "vector_distance": 0.0,
            }
        ]
    }
    assert store.similarity_search("alpha", filter={"source": "docs"}) == [
        Document(id="doc-1", page_content="alpha", metadata={"source": "docs"})
    ]
    parameters = collection.documents.search.call_args.args[0]
    assert parameters["vector_query"].startswith("embedding:")
    assert parameters["exclude_fields"] == "embedding"
    assert parameters["filter_by"] == "attributes.source:=docs"

    collection.documents.export.return_value = json.dumps(
        {
            "id": "doc-1",
            "body": "alpha",
            "embedding": [5.0, 2.0, 1.0],
            "attributes": {"source": "docs"},
        }
    )
    assert store.get_by_ids(["doc-1"]) == [
        Document(id="doc-1", page_content="alpha", metadata={"source": "docs"})
    ]


def test_add_documents_validates_lengths_and_empty_input() -> None:
    store, _, collection = make_sync_store()

    assert store.add_documents([]) == []
    collection.documents.import_.assert_not_called()
    with patch.object(store.embeddings, "embed_documents") as embed_documents:
        with pytest.raises(ValueError, match="number of IDs"):
            store.add_documents([Document(page_content="one")], ids=["1", "2"])
        embed_documents.assert_not_called()
    with pytest.raises(ValueError, match="URL-safe"):
        store.add_documents([Document(page_content="one")], ids=["not safe"])
    with pytest.raises(ValueError, match="number of metadatas"):
        store.add_texts(["one"], metadatas=[])


def test_add_documents_validates_existing_collection_schema() -> None:
    store, _, collection = make_sync_store()
    collection.retrieve.return_value = collection_schema(num_dim=5)

    with pytest.raises(TypesenseCollectionError, match="num_dim=3"):
        store.add_documents([Document(page_content="alpha")], ids=["doc-1"])

    fractional_schema = collection_schema()
    fractional_schema["fields"][1]["num_dim"] = 3.5
    collection.retrieve.return_value = fractional_schema
    with pytest.raises(TypesenseCollectionError, match="num_dim=3"):
        store.create_collection(3)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda schema: schema["fields"].pop(0), "text"),
        (lambda schema: schema["fields"][0].update(type="int32"), "string"),
        (lambda schema: schema["fields"].pop(1), "vec"),
        (lambda schema: schema["fields"][1].update(type="string[]"), "float"),
        (lambda schema: schema["fields"][1].update(vec_dist="ip"), "cosine"),
        (lambda schema: schema["fields"].pop(2), "metadata"),
        (lambda schema: schema["fields"][2].update(type="string"), "object"),
        (lambda schema: schema["fields"][2].update(index=False), "indexed"),
        (lambda schema: schema.update(enable_nested_fields=False), "nested"),
    ],
)
def test_collection_schema_rejects_every_incompatible_managed_setting(
    mutate: Any, message: str
) -> None:
    store, _, collection = make_sync_store()
    schema = collection_schema()
    mutate(schema)
    collection.retrieve.return_value = schema

    with pytest.raises(TypesenseCollectionError, match=message):
        store.create_collection(3)


def test_collection_creation_handles_another_creator_winning_the_race() -> None:
    store, collections, collection = make_sync_store()
    collection.retrieve.side_effect = [ObjectNotFound("missing"), collection_schema()]
    collections.create.side_effect = ObjectAlreadyExists("created concurrently")

    store.create_collection(3)
    store.create_collection(3)

    assert collection.retrieve.call_count == 2
    collections.create.assert_called_once()


def test_bulk_import_failures_are_not_silently_ignored() -> None:
    store, _, collection = make_sync_store()
    collection.retrieve.return_value = collection_schema()
    collection.documents.import_.return_value = [
        {"success": False, "code": 400, "error": "bad vector", "document": {"id": "x"}}
    ]

    with pytest.raises(TypesenseImportError, match="bad vector") as error:
        store.add_documents([Document(page_content="alpha")], ids=["x"])

    assert error.value.failures[0]["code"] == 400


def test_schema_validation_is_cached_and_recovers_after_collection_removal() -> None:
    store, collections, collection = make_sync_store()
    collection.retrieve.side_effect = ObjectNotFound("missing")
    collection.documents.import_.side_effect = [
        [{"success": True}],
        ObjectNotFound("collection removed"),
        [{"success": True}],
    ]

    store.add_documents([Document(page_content="alpha")], ids=["first"])
    store.add_documents([Document(page_content="alpha")], ids=["second"])

    assert collections.create.call_count == 2
    assert collection.retrieve.call_count == 1
    assert collection.documents.import_.call_count == 3


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


def test_similarity_search_raises_for_missing_collection_and_skips_zero_k() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.side_effect = ObjectNotFound("missing")

    with pytest.raises(ObjectNotFound):
        store.similarity_search("query", k=1)
    assert store.similarity_search("query", k=0) == []


def test_similarity_search_rejects_internal_parameter_override() -> None:
    store, _, _ = make_sync_store()

    with pytest.raises(ValueError, match="integration-managed"):
        store.similarity_search("query", search_parameters=cast(Any, {"per_page": 100}))
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        store.similarity_search("query", k=-1)


@pytest.mark.parametrize(
    "parameter",
    [
        "sort_by",
        "preset",
        "offset",
        "limit",
        "group_by",
        "include_fields",
        "unknown_option",
    ],
)
def test_similarity_search_rejects_parameters_that_break_vector_contract(
    parameter: str,
) -> None:
    store, _, _ = make_sync_store()

    with pytest.raises(ValueError, match="search_parameters|integration-managed"):
        store.similarity_search("query", search_parameters={parameter: "value"})


def test_v30_curation_search_parameters_are_forwarded() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response()

    store.similarity_search(
        "query",
        search_parameters={"curation_tags": "diverse", "diversity_lambda": 0.75},
    )

    parameters = collection.documents.search.call_args.args[0]
    assert parameters["curation_tags"] == "diverse"
    assert parameters["diversity_lambda"] == 0.75


@pytest.mark.parametrize(
    "response",
    [
        {"grouped_hits": []},
        {"hits": {}},
        {"hits": [{"document": {}}]},
    ],
)
def test_similarity_search_rejects_malformed_typesense_responses(
    response: dict[str, Any],
) -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = response

    with pytest.raises(TypesenseVectorStoreError):
        store.similarity_search("query")


def test_relevance_scores_normalize_typesense_cosine_distance() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response()

    results = store.similarity_search_with_relevance_scores("bar", k=2)

    assert results[0][1] == pytest.approx(0.95)
    assert results[1][1] == pytest.approx(0.8)


def test_inherited_search_and_retriever_use_similarity_contract() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response()

    assert [document.id for document in store.search("bar", "similarity", k=2)] == [
        "doc-1",
        "doc-2",
    ]
    retriever = store.as_retriever(search_kwargs={"k": 2})
    assert [document.id for document in retriever.invoke("bar")] == ["doc-1", "doc-2"]


@pytest.mark.asyncio
async def test_inherited_async_search_and_relevance_scores() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response()

    assert [document.id for document in await store.asearch("bar", "similarity", k=2)] == [
        "doc-1",
        "doc-2",
    ]
    relevant = await store.asimilarity_search_with_relevance_scores("bar", k=2)
    assert [score for _, score in relevant] == pytest.approx([0.95, 0.8])


def test_mmr_search_requests_vectors_and_returns_documents() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response(include_vectors=True)

    results = store.max_marginal_relevance_search("bar", k=1, fetch_k=2)

    assert len(results) == 1
    assert results[0].id == "doc-1"
    parameters = collection.documents.search.call_args.args[0]
    assert "exclude_fields" not in parameters
    assert parameters["per_page"] == 2


def test_mmr_fetches_exactly_fetch_k_candidates() -> None:
    store, _, collection = make_sync_store()
    collection.documents.search.return_value = search_response(include_vectors=True)

    results = store.max_marginal_relevance_search("bar", k=3, fetch_k=2)

    assert len(results) == 2
    assert collection.documents.search.call_args.args[0]["per_page"] == 2


def test_get_by_ids_deduplicates_and_ignores_missing_ids() -> None:
    store, _, collection = make_sync_store()
    collection.documents.export.return_value = "\n".join(
        [
            json.dumps(
                {
                    "id": "doc-2",
                    "text": "foo",
                    "vec": [1.0, 0.0, 0.0],
                    "metadata": {"source": "news"},
                }
            ),
            json.dumps(
                {
                    "id": "doc-1",
                    "text": "bar",
                    "vec": [1.0, 0.0, 0.0],
                    "metadata": {"source": "tweet"},
                }
            ),
        ]
    )

    results = store.get_by_ids(["doc-1", "missing", "doc-2", "doc-1"])

    assert results == [
        Document(id="doc-1", page_content="bar", metadata={"source": "tweet"}),
        Document(id="doc-2", page_content="foo", metadata={"source": "news"}),
    ]
    collection.documents.export.assert_called_once_with({"filter_by": "id:=[doc-1,missing,doc-2]"})


def test_get_by_ids_raises_when_collection_is_missing() -> None:
    store, _, collection = make_sync_store()
    collection.documents.export.side_effect = ObjectNotFound("missing collection")

    with pytest.raises(ObjectNotFound):
        store.get_by_ids(["doc-1"])


@pytest.mark.parametrize("response", [None, "not-json", "[]"])
def test_get_by_ids_rejects_malformed_export_responses(response: object) -> None:
    store, _, collection = make_sync_store()
    collection.documents.export.return_value = response

    with pytest.raises(TypesenseVectorStoreError, match="export|JSON|non-object"):
        store.get_by_ids(["doc-1"])


def test_delete_uses_safe_bulk_filter_and_requires_explicit_truncate() -> None:
    store, _, collection = make_sync_store()

    assert store.delete(["doc-1", "doc-2"]) is True
    collection.documents.delete.assert_called_with({"filter_by": "id:=[doc-1,doc-2]"})
    with pytest.raises(ValueError, match="delete_all_documents"):
        store.delete()
    assert store.delete(delete_all_documents=True) is True
    collection.documents.delete.assert_called_with({"truncate": True})


def test_unexpected_bulk_response_raises_integration_error() -> None:
    with pytest.raises(TypesenseVectorStoreError, match="unexpected bulk import"):
        TypesenseVectorStore._raise_for_import_failures("not-json-records")
    with pytest.raises(TypesenseVectorStoreError, match="malformed record"):
        TypesenseVectorStore._raise_for_import_failures([{"not_success": True}])


def test_from_client_params_requires_url_and_key_and_supports_async_only() -> None:
    with pytest.raises(ValueError, match="typesense_url"):
        TypesenseVectorStore.from_client_params(FakeEmbeddings(), typesense_url="", api_key="key")
    with pytest.raises(ValueError, match="api_key"):
        TypesenseVectorStore.from_client_params(
            FakeEmbeddings(), typesense_url="http://localhost:8108", api_key=""
        )
    with pytest.raises(ValueError, match="timeout"):
        TypesenseVectorStore.from_client_params(
            FakeEmbeddings(),
            typesense_url="http://localhost:8108",
            api_key="key",
            connection_timeout_seconds=float("nan"),
        )
    with pytest.raises(ValueError, match="invalid port"):
        TypesenseVectorStore.from_client_params(
            FakeEmbeddings(), typesense_url="http://localhost:0", api_key="key"
        )

    with (
        patch("langchain_typesense.vectorstores.Client") as sync_client,
        patch("langchain_typesense.vectorstores.AsyncClient") as async_client,
    ):
        store = TypesenseVectorStore.from_client_params(
            FakeEmbeddings(),
            typesense_url="https://example.typesense.net",
            api_key="key",
            client_mode="async",
        )

    sync_client.assert_not_called()
    config = async_client.call_args.args[0]
    assert config["nodes"] == [{"host": "example.typesense.net", "port": 443, "protocol": "https"}]
    assert store.client is None
    assert store.async_client is async_client.return_value


def test_from_client_params_defaults_to_sync_and_can_create_both_clients() -> None:
    with (
        patch("langchain_typesense.vectorstores.Client") as sync_client,
        patch("langchain_typesense.vectorstores.AsyncClient") as async_client,
    ):
        default_store = TypesenseVectorStore.from_client_params(
            FakeEmbeddings(),
            typesense_url="http://localhost:8108",
            api_key="key",
        )
        both_store = TypesenseVectorStore.from_client_params(
            FakeEmbeddings(),
            typesense_url="http://localhost:8108",
            api_key="key",
            client_mode="both",
        )

    assert default_store.client is sync_client.return_value
    assert default_store.async_client is None
    assert sync_client.call_count == 2
    assert async_client.call_count == 1
    assert both_store.client is sync_client.return_value
    assert both_store.async_client is async_client.return_value


def test_from_client_params_preserves_an_explicit_standard_port() -> None:
    with patch("langchain_typesense.vectorstores.Client") as sync_client:
        TypesenseVectorStore.from_client_params(
            FakeEmbeddings(),
            typesense_url="https://example.typesense.net:443",
            api_key="key",
        )

    config = sync_client.call_args.args[0]
    assert config["nodes"] == [{"host": "example.typesense.net", "port": 443, "protocol": "https"}]


@pytest.mark.asyncio
async def test_close_closes_sync_client_and_aclose_closes_both_clients() -> None:
    store, _, _ = make_sync_store()
    client = cast(Any, store.client)
    client.api_call.close = MagicMock()
    store.close()
    client.api_call.close.assert_called_once()

    async_client = MagicMock()
    async_client.api_call.aclose = AsyncMock()
    store._async_client = cast(AsyncClient, async_client)

    await store.aclose()
    async_client.api_call.aclose.assert_awaited_once()
    assert client.api_call.close.call_count == 2

    async_client.api_call.aclose.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        await store.aclose()
    assert client.api_call.close.call_count == 3


def test_from_documents_generates_only_missing_mixed_ids() -> None:
    client = MagicMock()
    collection = MagicMock()
    client.collections.__getitem__.return_value = collection
    collection.retrieve.return_value = collection_schema()
    collection.documents.import_.return_value = [
        {"success": True},
        {"success": True},
    ]
    documents = [
        Document(id="provided", page_content="one"),
        Document(page_content="two"),
    ]

    store = TypesenseVectorStore.from_documents(
        documents,
        FakeEmbeddings(),
        client=cast(Client, client),
    )

    assert store.client is client
    payload = collection.documents.import_.call_args.args[0]
    assert payload[0]["id"] == "provided"
    UUID(payload[1]["id"])


@pytest.mark.asyncio
async def test_afrom_documents_generates_only_missing_mixed_ids() -> None:
    client = MagicMock()
    collection = MagicMock()
    client.collections.__getitem__.return_value = collection
    collection.retrieve.return_value = collection_schema()
    collection.documents.import_.return_value = [
        {"success": True},
        {"success": True},
    ]
    documents = [
        Document(id="provided", page_content="one"),
        Document(page_content="two"),
    ]

    store = await TypesenseVectorStore.afrom_documents(
        documents,
        FakeEmbeddings(),
        client=cast(Client, client),
    )

    payload = collection.documents.import_.call_args.args[0]
    assert payload[0]["id"] == "provided"
    UUID(payload[1]["id"])
    assert store.collection_name == "langchain-typesense"


@pytest.mark.asyncio
async def test_afrom_documents_supports_an_async_only_client() -> None:
    async_client = MagicMock()
    collections = MagicMock()
    collection = MagicMock()
    async_client.collections = collections
    collections.__getitem__.return_value = collection
    collection.retrieve = AsyncMock(side_effect=ObjectNotFound("missing"))
    collections.create = AsyncMock(return_value=collection_schema())
    collection.documents.import_ = AsyncMock(return_value=[{"success": True}])

    store = await TypesenseVectorStore.afrom_documents(
        [Document(id="provided", page_content="one")],
        FakeEmbeddings(),
        client=None,
        async_client=cast(AsyncClient, async_client),
    )

    assert store.client is None
    assert store.async_client is async_client
    collection.documents.import_.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_methods_fall_back_to_sync_client() -> None:
    store, collections, collection = make_sync_store()
    collection.documents.search.return_value = search_response()
    collection.retrieve.return_value = collection_schema()
    collection.documents.delete.return_value = {"num_deleted": 1}
    collection.documents.import_.return_value = [{"success": True}]
    collection.documents.export.return_value = json.dumps(
        {
            "id": "doc-1",
            "text": "bar",
            "vec": [3.0, 1.0, 1.0],
            "metadata": {},
        }
    )

    assert [doc.id for doc in await store.asimilarity_search("bar", k=1)] == [
        "doc-1",
        "doc-2",
    ]
    assert await store.asimilarity_search_by_vector([3.0, 1.0, 1.0], k=1)
    assert await store.aadd_documents([Document(page_content="bar")], ids=["doc-2"])
    assert await store.aget_by_ids(["doc-1"])
    assert await store.adelete(["doc-1"])
    assert await store.adelete_collection() is True
    assert await store.acreate_collection(3) is None
    assert collections.__getitem__.called


def test_inner_product_schema_and_cosine_only_features() -> None:
    client = cast(Client, MagicMock())
    store = TypesenseVectorStore(client, FakeEmbeddings(), vec_dist="ip")

    assert store._collection_schema(3)["fields"][1]["vec_dist"] == "ip"
    with pytest.raises(ValueError, match="cannot be normalized"):
        store._select_relevance_score_fn()
    with pytest.raises(ValueError, match="only available"):
        store.max_marginal_relevance_search("query")


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
