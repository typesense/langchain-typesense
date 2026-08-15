"""Integration tests against LangChain's standard VectorStore test suite.

These run against a real Typesense server, so they need one running first, e.g.:

    docker run -p 8108:8108 -v /tmp/typesense-data:/data \
        typesense/typesense:29.0 \
        --data-dir /data --api-key=xyz --enable-cors

Run with:

    uv run pytest tests/integration_tests
"""

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest_asyncio
import typesense
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests
from typesense.configuration import ConfigDict
from typesense.exceptions import ObjectNotFound

from langchain_typesense import TypesenseVectorStore


class TestTypesenseVectorStore(VectorStoreIntegrationTests):
    @pytest_asyncio.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:
        config: ConfigDict = {
            "nodes": [{"host": "localhost", "port": 8108, "protocol": "http"}],
            "api_key": "xyz",
            "connection_timeout_seconds": 2,
        }
        client = typesense.Client(config)
        async_client = typesense.AsyncClient(config)
        collection_name = f"langchain-typesense-test-{uuid4()}"
        store = TypesenseVectorStore(
            client=client,
            async_client=async_client,
            embedding=self.get_embeddings(),
            collection_name=collection_name,
        )
        try:
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
