"""Integration tests against LangChain's standard VectorStore test suite.

These run against a real Typesense server, so they need one running first, e.g.:
    docker compose up -d
    OR:
    docker run -p 8108:8108 -v /tmp/typesense-data:/data \
        typesense/typesense:30.2 \
        --data-dir /data --api-key=xyz --enable-cors

Run with:

    uv run pytest tests/integration_tests
"""

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
import typesense
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests
from typesense.configuration import ConfigDict
from typesense.exceptions import ObjectNotFound

from langchain_typesense import TypesenseVectorStore


class HardcodedEmbeddings(Embeddings):
    """Exact vectors for deterministic distance and threshold assertions."""

    _vectors = {
        "alpha": [1.0, 0.0, 0.0],
        "near-alpha": [0.8, 0.6, 0.0],
        "opposite": [-1.0, 0.0, 0.0],
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vectors[text])


def client_config() -> ConfigDict:
    """Return the local integration-test Typesense configuration."""
    return {
        "nodes": [{"host": "localhost", "port": 8108, "protocol": "http"}],
        "api_key": "xyz",
        "connection_timeout_seconds": 2,
    }


class TestTypesenseVectorStore(VectorStoreIntegrationTests):
    @pytest_asyncio.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:
        config = client_config()
        client = typesense.Client(config)
        async_client = typesense.AsyncClient(config)
        collection_name = f"langchain-typesense-test-{uuid4()}"
        embedding = self.get_embeddings()
        store = TypesenseVectorStore(
            client=client,
            async_client=async_client,
            embedding=embedding,
            collection_name=collection_name,
        )
        try:
            await store.acreate_collection(len(embedding.embed_query("dimension probe")))
            yield store
        finally:
            try:
                await async_client.collections[collection_name].delete()
            except ObjectNotFound:
                pass
            await store.aclose()

    def test_metadata_filter(self, vectorstore: VectorStore) -> None:
        vectorstore.add_documents(
            [
                Document(page_content="alpha", metadata={"source": "tweet"}),
                Document(page_content="beta", metadata={"source": "news"}),
            ],
            ids=["alpha", "beta"],
        )

        results = vectorstore.similarity_search(
            "alpha",
            k=2,
            filter={"source": "tweet"},
        )

        assert results == [Document(id="alpha", page_content="alpha", metadata={"source": "tweet"})]

    def test_relevance_scores_are_normalized(self, vectorstore: VectorStore) -> None:
        vectorstore.add_documents(
            [Document(page_content="alpha")],
            ids=["alpha"],
        )

        results = vectorstore.similarity_search_with_relevance_scores("alpha", k=1)

        assert len(results) == 1
        assert 0.0 <= results[0][1] <= 1.0


class TestHardcodedTypesenseMethods:
    """Exercise Typesense-specific methods with known cosine geometry."""

    @pytest_asyncio.fixture()
    async def store(self) -> AsyncGenerator[TypesenseVectorStore, None]:
        config = client_config()
        collection_name = f"langchain-typesense-hardcoded-{uuid4()}"
        store = TypesenseVectorStore(
            client=typesense.Client(config),
            async_client=typesense.AsyncClient(config),
            embedding=HardcodedEmbeddings(),
            collection_name=collection_name,
        )
        try:
            yield store
        finally:
            await store.adelete_collection()
            await store.aclose()

    def test_sync_add_search_score_vector_mmr_and_delete(self, store: TypesenseVectorStore) -> None:
        ids = store.add_texts(
            ["alpha", "near-alpha", "opposite"],
            metadatas=[{"rank": 1}, {"rank": 2}, {"rank": 3}],
            ids=["alpha", "near", "opposite"],
            batch_size=2,
        )
        assert ids == ["alpha", "near", "opposite"]
        assert [document.id for document in store.similarity_search("alpha", k=3)] == ids

        scored = store.similarity_search_with_score("alpha", k=3)
        assert [score for _, score in scored] == pytest.approx([0.0, 0.2, 2.0])
        assert [
            document.id for document in store.similarity_search_by_vector([1.0, 0.0, 0.0], k=2)
        ] == ["alpha", "near"]

        relevant = store.similarity_search_with_relevance_scores("alpha", k=3, score_threshold=0.95)
        assert [(document.id, score) for document, score in relevant] == [
            ("alpha", pytest.approx(1.0))
        ]

        assert [
            document.id
            for document in store.max_marginal_relevance_search(
                "alpha", k=2, fetch_k=3, lambda_mult=0.0
            )
        ] == ["alpha", "opposite"]
        assert [
            document.id
            for document in store.max_marginal_relevance_search_by_vector(
                [1.0, 0.0, 0.0], k=2, fetch_k=3, lambda_mult=0.0
            )
        ] == ["alpha", "opposite"]

        assert [document.id for document in store.get_by_ids(["near", "missing"])] == ["near"]
        assert store.delete(["near"]) is True
        with pytest.raises(ValueError, match="allow_delete_all"):
            store.delete()
        assert store.delete(allow_delete_all=True) is True
        assert store.similarity_search("alpha") == []

    async def test_native_async_add_search_mmr_get_and_delete(
        self, store: TypesenseVectorStore
    ) -> None:
        ids = await store.aadd_documents(
            [Document(page_content="alpha"), Document(page_content="opposite")],
            ids=["alpha", "opposite"],
        )
        assert ids == ["alpha", "opposite"]
        assert [document.id for document in await store.asimilarity_search("alpha", k=2)] == ids
        assert [
            document.id for document, _ in await store.asimilarity_search_with_score("alpha", k=2)
        ] == ids
        assert [
            document.id
            for document in await store.asimilarity_search_by_vector([1.0, 0.0, 0.0], k=2)
        ] == ids
        assert [
            document.id
            for document in await store.amax_marginal_relevance_search(
                "alpha", k=2, fetch_k=2, lambda_mult=0.0
            )
        ] == ids
        assert [
            document.id
            for document in await store.amax_marginal_relevance_search_by_vector(
                [1.0, 0.0, 0.0], k=2, fetch_k=2, lambda_mult=0.0
            )
        ] == ids
        assert [document.id for document in await store.aget_by_ids(ids)] == ids
        assert await store.adelete(["opposite"]) is True
        assert await store.adelete(allow_delete_all=True) is True

    async def test_async_only_client_and_missing_collection_error(self) -> None:
        config = client_config()
        collection_name = f"langchain-typesense-async-only-{uuid4()}"
        store = TypesenseVectorStore(
            client=None,
            async_client=typesense.AsyncClient(config),
            embedding=HardcodedEmbeddings(),
            collection_name=collection_name,
        )
        try:
            await store.aadd_texts(["alpha"], ids=["alpha"])
            assert (await store.asimilarity_search("alpha"))[0].id == "alpha"
            with pytest.raises(RuntimeError, match="only an async"):
                store.similarity_search("alpha")
            await store.adelete_collection()
            with pytest.raises(ObjectNotFound):
                await store.asimilarity_search("alpha")
        finally:
            await store.aclose()

    def test_from_texts_one_step_contract(self) -> None:
        config = client_config()
        collection_name = f"langchain-typesense-from-texts-{uuid4()}"
        store = TypesenseVectorStore.from_texts(
            ["alpha"],
            HardcodedEmbeddings(),
            ids=["alpha"],
            client=typesense.Client(config),
            collection_name=collection_name,
        )
        try:
            assert store.get_by_ids(["alpha"])[0].page_content == "alpha"
        finally:
            store.delete_collection()
            store.close()

    def test_inner_product_metric(self) -> None:
        config = client_config()
        collection_name = f"langchain-typesense-ip-{uuid4()}"
        store = TypesenseVectorStore(
            client=typesense.Client(config),
            embedding=HardcodedEmbeddings(),
            collection_name=collection_name,
            vec_dist="ip",
        )
        try:
            store.add_texts(
                ["alpha", "near-alpha", "opposite"],
                ids=["alpha", "near", "opposite"],
            )
            assert [
                document.id for document, _ in store.similarity_search_with_score("alpha", k=3)
            ] == ["alpha", "near", "opposite"]
            with pytest.raises(ValueError, match="cannot be normalized"):
                store.similarity_search_with_relevance_scores("alpha")
            with pytest.raises(ValueError, match="only available"):
                store.max_marginal_relevance_search("alpha")
        finally:
            store.delete_collection()
            store.close()
