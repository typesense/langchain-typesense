# 🦜🔎⚡️ langchain-typesense

**`langchain-typesense`** is a LangChain `VectorStore` implementation for Typesense.

**Features:**

- **Sync & Async:** Supports both synchronous and asynchronous methods.
- **Search & Retrieval**: Vector similarity search, hybrid search, MMR (Maximal Marginal Relevance), and relevance scoring.
- **Filtering:** Metadata filtering via Typesense filter expressions.
- **Document Management:** Batch writes, ID lookups, and deletions.

## Installation

Requirements:

- Python 3.10+
- `langchain-core` 1.x
- `typesense` 2.x (Typsense Python client)
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
    client=client,                         # typesense.Client | None
    async_client=async_client,             # typesense.AsyncClient | None
    embedding=embeddings,
    collection_name="articles",
    text_key="body",
    vector_key="embedding",
    metadata_key="attributes",
    index_metadata=True,
    vec_dist="cosine",                    # "cosine" or "ip"
)
```

At least one client is required. Sync methods require a sync client. Async methods use the
async client when supplied and otherwise run the sync method in a worker thread.

| Parameter         | Default                    | Meaning                                                                                       |
| ----------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| `client`          | required unless async-only | Sync Typesense connection. It is required when calling methods such as `add_documents()`.     |
| `async_client`    | `None`                     | Async Typesense connection. Async methods use it instead of running sync methods in a thread. |
| `embedding`       | required                   | Converts document text and search queries into vectors.                                       |
| `collection_name` | `"langchain-typesense"`    | Name of the Typesense collection where documents are stored.                                  |
| `text_key`        | `"text"`                   | Typesense field used to store `Document.page_content`.                                        |
| `vector_key`      | `"vec"`                    | Typesense field used to store each generated embedding vector.                                |
| `metadata_key`    | `"metadata"`               | Parent object used to store `Document.metadata`.                                              |
| `index_metadata`  | `True`                     | Makes metadata filterable in new collections. Set to `False` if metadata is return-only.      |
| `vec_dist`        | `"cosine"`                 | How vector distance is calculated: `"cosine"` or inner product (`"ip"`).                      |

Field names must be non-empty, distinct, and different from `id`.

### Create clients from a URL

`from_client_params` creates the requested connection pools. `api_key` falls back to
`TYPESENSE_API_KEY` when omitted.

```python
store = TypesenseVectorStore.from_client_params(
    embedding=embeddings,
    typesense_url="https://example.a1.typesense.net:443",
    api_key="your-typesense-api-key",
    client_mode="both", # "sync", "async", or "both"
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
fragment. An omitted port follows normal URL conventions: `80` for HTTP and `443` for
HTTPS. Explicit ports are always preserved, so `https://example.a1.typesense.net:443` is valid and resolves to the same endpoint as the
URL without `:443`.

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

| Parameter            | Default | Meaning                                                                                         |
| -------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `k`                  | `4`     | Return at most this many nearest documents. `0` returns immediately without making a request.   |
| `filter`             | `None`  | Restrict results with a metadata dictionary or a raw Typesense filter expression.               |
| `search_parameters`  | `None`  | Fine-tune the Typesense request with the supported options listed below.                        |
| `distance_threshold` | `None`  | Exclude documents whose raw vector distance is greater than this value. Lower distance is best. |
| `ef`                 | `None`  | Controls HNSW search breadth. Higher values can improve recall but require more work.           |
| `flat_search_cutoff` | `None`  | Use exact brute-force search when filtering leaves fewer than this many candidate documents.    |

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

`search_parameters` accepts these keys:

| Area         | Keys                                                                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Filtering    | `max_filter_by_candidates`, `enable_lazy_filter`                                                                                                                  |
| Facets       | `facet_by`, `max_facet_values`, `facet_query`, `facet_query_num_typos`, `facet_return_parent`, `facet_sample_percent`, `facet_sample_threshold`, `facet_strategy` |
| Highlighting | `highlight_fields`, `highlight_full_fields`, `highlight_affix_num_tokens`, `highlight_start_tag`, `highlight_end_tag`, `enable_highlight_v1`, `snippet_threshold` |
| Execution    | `limit_hits`, `search_cutoff_ms`, `exhaustive_search`                                                                                                             |
| Cache        | `use_cache`, `cache_ttl`                                                                                                                                          |
| Curation     | `curation_tags`, `diversity_lambda`                                                                                                                               |

The store manages `q`, `vector_query`, `per_page`, `page`, `filter_by`, `include_fields`,
and `exclude_fields`; they cannot be overridden. Options that change pagination, sorting,
grouping, or the hit shape are rejected. Curation options may reorder results.

See the [Typesense search documentation](https://typesense.org/docs/30.2/api/search.html)
for option semantics.

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
deleted = store.delete_collection()          # False if already absent
store.delete(delete_all_documents=True)      # keep schema, remove all documents
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
`similarity_score_threshold`. The inherited `search` and `asearch` methods are available.

## Schema and errors

New collections contain a string text field, a `float[]` vector field, and an optional
nested metadata object. `index_metadata=False` stores metadata without indexing it.
Existing collections must match the configured field types, vector dimension, distance,
metadata indexing requirement, and nested-field setting.

```python
from langchain_typesense import (
    TypesenseCollectionError,
    TypesenseImportError,
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

`Typesense` remains an alias. .

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
