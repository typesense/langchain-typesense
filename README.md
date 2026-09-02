# 🦜🔎⚡️ langchain-typesense

`langchain-typesense` is a LangChain `VectorStore` implementation for Typesense.

**Features:**

- **Sync & Async:** Supports both synchronous and asynchronous methods.
- **Search & Retrieval**: Vector similarity search, hybrid keyword/vector search, MMR (Maximal Marginal Relevance), and relevance scoring.
- **Filtering:** Metadata filtering via Typesense filter expressions.
- **Document Management:** Batch writes, ID lookups, and deletions.

## Installation

Requirements:

- Python 3.10+
- `langchain-core` 1.x
- `typesense` 2.x (Typesense Python client)
- Typesense Server 30.2
- A LangChain `Embeddings` provider.

```bash
pip install -U langchain-typesense langchain-openai
```

`langchain-openai` is used here as an example embedding provider; replace it with your provider of choice.

## Quickstart

```python
import typesense
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_typesense import TypesenseVectorStore

client = typesense.Client(
    {
        "nodes": ["http://localhost:8108"],
        "api_key": "xyz",
        "connection_timeout_seconds": 2,
    }
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = TypesenseVectorStore(
    client=client,
    embedding=embeddings,
    collection_name="my-documents",
)

store.add_documents(
    [
        Document(page_content="Typesense is fast", metadata={"source": "docs"}),
        Document(page_content="LangChain composes applications", metadata={"source": "blog"}),
    ],
    ids=["typesense", "langchain"],
)

matches = store.similarity_search(
    "Which search engine is fast?",
    k=2,
    filter={"source": "docs"},
)
print(matches)
store.close()
```

The first non-empty write creates the collection or validates its existing schema. Later
writes from the same store instance reuse the validation result.

## Configure the store

Pass clients directly when your application owns their lifecycle:

```python
store = TypesenseVectorStore(
    client=client,  # typesense.Client | None
    async_client=async_client,  # typesense.AsyncClient | None
    embedding=embeddings,
    collection_name="articles",
    text_key="body",
    vector_key="embedding",
    metadata_key="attributes",
    index_metadata=True,
    vec_dist="cosine",  # "cosine" or "ip"
)
```

At least one client is required. Sync methods require a sync client. Async methods use the
async client when supplied and otherwise run the sync method in a worker thread.

| Parameter         | Default                    | What it does                                                                                 |
| ----------------- | -------------------------- | -------------------------------------------------------------------------------------------- |
| `client`          | required unless async-only | Supplies the synchronous Typesense connection. Sync store methods require it.                |
| `async_client`    | `None`                     | Supplies a native async connection. Without it, async methods run sync methods in a worker.  |
| `embedding`       | required                   | Embeds document text for writes and query text for searches.                                 |
| `collection_name` | `"langchain-typesense"`    | Selects the Typesense collection used by this store.                                         |
| `text_key`        | `"text"`                   | Names the field that stores `Document.page_content`.                                         |
| `vector_key`      | `"vec"`                    | Names the `float[]` field that stores embeddings.                                            |
| `metadata_key`    | `"metadata"`               | Names the object field that stores `Document.metadata`.                                      |
| `index_metadata`  | `True`                     | Makes metadata filterable in collections created by the store. `False` keeps it return-only. |
| `vec_dist`        | `"cosine"`                 | Chooses cosine distance or inner-product distance (`"ip"`) for the vector field.             |

Field names must be non-empty, distinct, and different from `id`.

### Create clients from a URL

`from_client_params` creates the requested connection pools. `api_key` falls back to
`TYPESENSE_API_KEY` when omitted.

```python
store = TypesenseVectorStore.from_client_params(
    embedding=embeddings,
    typesense_url="https://example.a1.typesense.net:443",
    api_key="your-typesense-api-key",
    client_mode="both",  # "sync", "async", or "both"
    connection_timeout_seconds=2.0,
    collection_name="articles",
    text_key="body",
    vector_key="embedding",
    metadata_key="attributes",
    index_metadata=True,
    vec_dist="cosine",
)
```

The URL must be absolute HTTP(S) and cannot contain credentials, a path, query, or
fragment. Its connection-specific parameters are:

| Parameter                    | Default                 | What it does                                                                     |
| ---------------------------- | ----------------------- | -------------------------------------------------------------------------------- |
| `typesense_url`              | required                | Sets the single Typesense node, including its scheme, host, and optional port.   |
| `api_key`                    | `TYPESENSE_API_KEY` env | Authenticates requests. An explicit value takes precedence over the environment. |
| `client_mode`                | `"sync"`                | Creates a `"sync"`, `"async"`, or `"both"` client pool.                          |
| `connection_timeout_seconds` | `2.0`                   | Sets the request timeout; it must be finite and greater than zero.               |

When the URL omits a port, the URL scheme supplies its standard port: HTTP uses `80`
and HTTPS uses `443`. Port `8108` is Typesense's common direct-server port, not HTTP's
standard port, so it must be written explicitly: `http://example.com:8108`. Explicit
ports are preserved. For example, `https://example.a1.typesense.net:443` is valid and
targets the same port as `https://example.a1.typesense.net`.

### Async-only setup

```python
import typesense

config = {
    "nodes": ["http://localhost:8108"],
    "api_key": "xyz",
    "connection_timeout_seconds": 2,
}
store = TypesenseVectorStore(
    client=None,
    async_client=typesense.AsyncClient(config),
    embedding=embeddings,
    collection_name="articles",
)

await store.aadd_texts(["alpha", "beta"], ids=["alpha", "beta"])
matches = await store.asimilarity_search("alpha", k=2)
await store.aclose()
```

`close()` closes the sync pool. `aclose()` closes the async pool and then the sync pool,
if present. Do not reuse clients after the store closes them.

## Add documents

```python
ids = store.add_documents(
    [
        Document(id="doc-1", page_content="First", metadata={"group": "a"}),
        Document(page_content="Second", metadata={"group": "b"}),
    ],
    batch_size=100,
)

ids = store.add_texts(
    ["Third", "Fourth"],
    metadatas=[{"group": "a"}, {"group": "b"}],
    ids=["doc-3", "doc-4"],
    batch_size=100,
)
```

`ids` and `metadatas` must match the input length. Missing IDs come from `Document.id` or
are generated as UUIDs. IDs may contain only URL-unreserved characters. Reusing an ID
upserts the document. For multi-request imports, explicit stable IDs make retries easier
to reconcile.

Before the first import, the store retrieves or creates the collection and validates its
schema. Importing first would not safely replace that check: an existing collection can
accept a document while still using the wrong distance metric or metadata indexing. The
validated dimension is cached, so later writes skip the schema request. If someone deletes
the collection between writes, the store handles the import's not-found response, recreates
the collection, and retries once.

The async methods are `aadd_documents` and `aadd_texts`. One-step factories are also
available:

```python
store = TypesenseVectorStore.from_documents(documents, embeddings, client=client)
store = TypesenseVectorStore.from_texts(texts, embeddings, client=client)

store = await TypesenseVectorStore.afrom_documents(
    documents,
    embeddings,
    client=None,
    async_client=async_client,
)
store = await TypesenseVectorStore.afrom_texts(
    texts,
    embeddings,
    client=None,
    async_client=async_client,
)
```

Factories accept the same schema parameters as the constructor, plus `ids`, `metadatas`,
and `batch_size`.

## Search

```python
documents = store.similarity_search(
    "release notes",
    k=8,
    filter={"source": "docs"},
    distance_threshold=0.4,
    ef=100,
    flat_search_cutoff=20,
    search_parameters={
        "enable_lazy_filter": True,
        "search_cutoff_ms": 500,
        "use_cache": True,
        "cache_ttl": 60,
    },
)

documents_and_distances = store.similarity_search_with_score("release notes", k=4)
query_vector = embeddings.embed_query("release notes")
documents = store.similarity_search_by_vector(query_vector, k=4)
```

| Parameter            | Default | What it does                                                                               |
| -------------------- | ------- | ------------------------------------------------------------------------------------------ |
| `k`                  | `4`     | Caps the number of returned documents. `0` skips embedding and the Typesense request.      |
| `filter`             | `None`  | Keeps only documents matching a metadata dictionary or raw Typesense filter expression.    |
| `search_parameters`  | `None`  | Passes the safe Typesense options represented by `TypesenseSearchParameters`.              |
| `distance_threshold` | `None`  | Drops hits above this raw vector distance. Smaller distances are closer.                   |
| `ef`                 | `None`  | Expands the HNSW search. Larger values can improve recall at the cost of more server work. |
| `flat_search_cutoff` | `None`  | Switches to exact search when a filter leaves fewer than this many candidates.             |

Async equivalents are `asimilarity_search`, `asimilarity_search_with_score`, and
`asimilarity_search_by_vector`.

### Metadata filters

Dictionary filters target the configured metadata object and combine entries with `&&`:

```python
store.similarity_search(
    "release notes",
    filter={"source": "docs", "year": 2026, "published": True},
)
```

Use a raw string for ranges, joins, or other Typesense syntax:

```python
store.similarity_search(
    "release notes",
    filter="metadata.year:>=2024 && metadata.source:=docs",
)
```

Raw strings are passed through unchanged. Do not interpolate untrusted values. Dictionary
filters require indexed metadata.

### Additional Typesense search parameters

The exported [`TypesenseSearchParameters`](./langchain_typesense/_types.py) type shows which Typesense options this adapter
accepts. See the [official Typesense Search API reference](https://typesense.org/docs/30.2/api/search.html)
for their meanings, accepted values, and server defaults.

The store manages `q`, `query_by`, `vector_query`, `per_page`, `page`, `filter_by`,
`include_fields`, and `exclude_fields`; they cannot be overridden. Options that change
pagination, sorting, grouping, or the hit shape are rejected. Curation options may reorder
results.

## Hybrid search

Hybrid search combines Typesense's keyword rank with the supplied embedding rank. Use
`alpha` to choose their balance:

```python
documents = store.hybrid_search(
    "wireless keyboard",
    k=8,
    alpha=0.4,  # 40% vector, 60% keyword
    query_by=["text", "metadata.title"],
    filter={"published": True},
    distance_threshold=0.5,
    ef=100,
    flat_search_cutoff=20,
    search_parameters={
        "query_by_weights": [2, 1],
        "drop_tokens_threshold": 0,
        "num_typos": 1,
        "rerank_hybrid_matches": True,
        "enable_lazy_filter": True,
    },
)

documents_and_scores = store.hybrid_search_with_score(
    "wireless keyboard",
    k=4,
    alpha=0.3,
)
documents = await store.ahybrid_search("wireless keyboard", k=4)
documents_and_scores = await store.ahybrid_search_with_score("wireless keyboard", k=4)
```

| Parameter            | Default               | What it does                                                                              |
| -------------------- | --------------------- | ----------------------------------------------------------------------------------------- |
| `k`                  | `4`                   | Caps the number of returned documents. `0` skips embedding and the Typesense request.     |
| `alpha`              | `0.3`                 | Sets the vector rank weight from `0` (keyword) to `1` (vector).                           |
| `query_by`           | configured text field | Selects one or more indexed string fields for keyword matching.                           |
| `filter`             | `None`                | Keeps only documents matching a metadata dictionary or raw Typesense filter expression.   |
| `search_parameters`  | `None`                | Passes safe keyword, filtering, execution, and fusion options to Typesense.               |
| `distance_threshold` | `None`                | Drops vector candidates above this raw distance.                                          |
| `ef`                 | `None`                | Expands the HNSW vector search; larger values trade more server work for possible recall. |
| `flat_search_cutoff` | `None`                | Uses exact vector search when a filter leaves fewer than this many candidates.            |

`query_by` defaults to `text_key`; the adapter adds `vector_key` separately through
`vector_query`, so `query_by` must not include the vector field. `query_by_weights` must
line up with the fields in `query_by`.

Hybrid `search_parameters` accepts every safe option from
[`TypesenseSearchParameters`](./langchain_typesense/_types.py), plus the keyword and fusion options represented by the
exported [`TypesenseHybridSearchParameters`](./langchain_typesense/_types.py) type.
See the [official Typesense hybrid-search documentation](https://typesense.org/docs/30.2/api/vector-search.html#hybrid-search)
for rank-fusion behavior, keyword options, server defaults, reranking tradeoffs, and
Typesense's performance guidance for multiword queries.

`hybrid_search_with_score` returns Typesense's `rank_fusion_score`, where larger is
better. It is rank-dependent—not a raw vector distance or normalized relevance score—so
do not compare scores from separate queries.

### Distances and relevance scores

`similarity_search_with_score` returns raw Typesense distance, where lower is better.
For cosine vectors, LangChain's inherited method converts distance to relevance:

```python
documents_and_relevance = store.similarity_search_with_relevance_scores(
    "release notes",
    k=4,
    score_threshold=0.75,
)
```

The conversion is `clamp(1 - distance / 2, 0, 1)`. Use `score_threshold` for normalized
relevance and `distance_threshold` for raw Typesense distance. Inner-product collections
support raw-distance search but not bounded relevance scores or local MMR.

## Maximal marginal relevance

```python
documents = store.max_marginal_relevance_search(
    "release notes",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,
    filter={"source": "docs"},
    distance_threshold=0.5,
    ef=100,
    flat_search_cutoff=20,
    search_parameters={"enable_lazy_filter": True},
)
```

`fetch_k` candidates are fetched and up to `k` are selected locally. `lambda_mult=1`
favors query similarity; `0` favors diversity. MMR requires `vec_dist="cosine"`.

By-vector and async variants are `max_marginal_relevance_search_by_vector`,
`amax_marginal_relevance_search`, and `amax_marginal_relevance_search_by_vector`. They
accept the same filtering and search-tuning parameters.

## Retrieve, delete, and manage collections

```python
documents = store.get_by_ids(["doc-1", "missing", "doc-2"])
store.delete(ids=["doc-1", "doc-2"])

store.create_collection(num_dim=1536)
deleted = store.delete_collection()  # False if already absent
store.delete(delete_all_documents=True)  # keep schema, remove all documents
```

`get_by_ids` de-duplicates IDs, ignores missing documents, and may return fewer results
than requested. `delete(ids=[])` is a no-op. Bare `delete()` is rejected to prevent an
accidental truncate. Async equivalents are `aget_by_ids`, `adelete`,
`acreate_collection`, and `adelete_collection`.

## Use as a retriever

```python
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 8, "score_threshold": 0.75},
)
documents = retriever.invoke("release notes")
```

Supported LangChain search types are `similarity`, `mmr`, and
`similarity_score_threshold`. Hybrid search is exposed through the explicit methods above,
not through `as_retriever(search_type="hybrid")`. The inherited `search` and `asearch`
methods remain available for LangChain's standard search types.

## Schema and errors

New collections contain a string text field, a `float[]` vector field, and an optional
nested metadata object. `index_metadata=False` stores metadata without indexing it.
Existing collections must match the configured field types, vector dimension, distance,
metadata indexing requirement, and nested-field setting.

```python
from langchain_typesense import (
    TypesenseCollectionError,
    TypesenseHybridSearchParameters,
    TypesenseImportError,
    TypesenseSearchParameters,
    TypesenseVectorStoreError,
)
```

- `TypesenseCollectionError`: incompatible schema or malformed stored document.
- `TypesenseImportError`: one or more bulk-import records failed; inspect `.failures`.
- `TypesenseVectorStoreError`: malformed or unexpected Typesense response.

Bulk import may partially succeed. Successful records are not rolled back. Typesense
client, network, authentication, and embedding exceptions retain their original types.

## Migrating from `langchain-community`

```python
# Before
from langchain_community.vectorstores import Typesense

# After
from langchain_typesense import TypesenseVectorStore
```

`Typesense` remains an alias.

- Rename `typesense_client` to `client` and
  `typesense_collection_name` to `collection_name`
- Use `from_client_params` for
  `typesense_url` and `api_key`.

This package uses nested metadata, preserves `Document.id`, validates existing schemas,
rejects unknown options, and reports per-record import failures. Reindex old collections
that do not match the managed schema.

## Development

Install `uv`, then install all dependency groups:

```bash
uv sync --all-groups
```

Run formatting, linting, type checks, tests, and builds:

```bash
make format
make lint
make test
make build
```

Run integration tests against the repository's Typesense server:

```bash
# Typesense listens on http://localhost:8108; the development key is xyz.
docker compose up -d
make integration_test
docker compose down
```

Unit tests block network access. Integration tests use unique collection names and clean
them up.
