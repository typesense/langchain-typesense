# langchain-typesense

`langchain-typesense` is a typed LangChain `VectorStore` integration for
[Typesense](https://typesense.org). It stores LangChain documents and embeddings in a
Typesense collection and supports synchronous and asynchronous writes, vector search,
relevance scores, MMR, metadata filters, ID lookup, and deletion.

The package targets Typesense Server 30.2 and the 2.x `typesense` Python client.

## Requirements and installation

- Python 3.10 or newer
- `langchain-core` 1.x
- `typesense` 2.x
- Typesense Server 30.2
- A LangChain `Embeddings` implementation

Install the integration together with the embedding provider used by your application:

```bash
pip install -U langchain-typesense langchain-openai
```

Install the embedding provider separately.

For local development, the repository includes Typesense 30.2:

```bash
docker compose up -d
# Typesense listens on http://localhost:8108; the development key is xyz.
```

The Compose service stores data in `./typesense-data`; stopping it does not remove that
directory. Run `docker compose down` when finished.

## Quickstart

```python
import typesense
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_typesense import TypesenseVectorStore

client = typesense.Client(
    {
        "nodes": [{"host": "localhost", "port": 8108, "protocol": "http"}],
        "api_key": "xyz",
        "connection_timeout_seconds": 2,
    }
)
store = TypesenseVectorStore(
    client=client,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
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

Collections are created lazily on the first non-empty write. Existing collections are
validated against the configured text, vector, distance, metadata, and nested-field
settings before an import.

## Clients and lifecycle

Pass an existing client when the application owns connection configuration. The
constructor accepts a synchronous `typesense.Client`, an optional `typesense.AsyncClient`,
or both:

```python
store = TypesenseVectorStore(
    client=client,
    async_client=async_client,
    embedding=embeddings,
    collection_name="my-documents",
)
```

If no async client is supplied, async store methods run the synchronous operation in a
worker executor. An async-only application can pass `client=None` and use native async
operations:

```python
import typesense

store = TypesenseVectorStore(
    client=None,
    async_client=typesense.AsyncClient(config),
    embedding=embeddings,
    collection_name="my-documents",
)
await store.aadd_texts(["alpha", "beta"], ids=["alpha", "beta"])
matches = await store.asimilarity_search("alpha", k=2)
await store.aclose()
```

Use `close()` for a synchronous client and `aclose()` for an asynchronous client. When
both are supplied, `aclose()` closes both pools. Do not reuse clients after the store
closes them.

`from_client_params` is a convenience constructor for applications that want the
integration to create the Typesense client(s):

```bash
export TYPESENSE_API_KEY="your-typesense-api-key"
```

```python
store = TypesenseVectorStore.from_client_params(
    embedding=embeddings,
    typesense_url="https://example.a1.typesense.net",
    client_mode="sync",  # the default; also "async" or "both"
    collection_name="my-documents",
)
```

`typesense_url` must be an absolute HTTP(S) URL without a path, query, or fragment.
`api_key` may be passed explicitly or read from `TYPESENSE_API_KEY`. The default
`client_mode="sync"` creates one synchronous connection pool; choose `"async"` or
`"both"` when those operations are needed. Supplying both modes creates two pools and
requires cleaning up both clients.

`from_texts` and LangChain's `from_documents` are available for small one-step setups.
For application lifecycle control, constructing the store explicitly and then calling
`add_documents` or `aadd_documents` is clearer. Mixed document IDs are supported: an
explicit `ids` sequence must match the document count; otherwise each `Document.id` is
used when present and a UUID is generated for an item without one.

## Collection schema and metadata

By default, a new collection has this shape (field names are configurable):

```text
enable_nested_fields: true

text     string
vec      float[]  (num_dim=<embedding dimension>, vec_dist=<cosine or ip>)
metadata object   (indexed by default)
```

`Document.page_content` is stored in `text`, the embedding in `vec`, and LangChain
metadata under the nested `metadata` object. The Typesense document ID is preserved as
`Document.id`. Nesting keeps metadata keys from colliding with managed fields.

Set `index_metadata=False` when metadata only needs to be stored and returned. Dictionary
metadata filters require indexed metadata, and this option affects newly created
collections only. It does not alter an existing schema.

The configured field names must be non-empty, distinct, and different from Typesense's
reserved `id` field. Reusing a collection requires matching text type, vector dimension,
vector distance, metadata object type, nested-field support, and metadata indexing (when
enabled). An incompatible collection raises `TypesenseCollectionError`.

Keep the embedding model and vector dimension stable for a collection. A dimension or
distance change requires a new compatible collection and reindexing.

## Writing documents

```python
ids = store.add_documents(
    [
        Document(id="doc-1", page_content="First", metadata={"group": "a"}),
        Document(page_content="Second", metadata={"group": "b"}),
    ],
    batch_size=100,
)
```

`add_documents` embeds all content, creates or validates the collection, bulk-upserts
documents, checks every Typesense import result, and returns resolved IDs. `add_texts`
converts strings to documents; both methods have asynchronous `aadd_*` equivalents.

IDs must be non-empty and contain only URL-unreserved characters (`A-Z`, `a-z`, digits,
`-`, `.`, `_`, or `~`). Reusing an ID upserts that document. Empty input returns an empty
list without creating a collection or calling the embedding model.

Typesense bulk import can partially succeed while returning HTTP success. A rejected
record raises `TypesenseImportError`; successful records are not rolled back. Inspect
the exception's `.failures` tuple and retry or reconcile only the rejected records.

## Similarity search

Text searches embed the query. By-vector methods use the supplied embedding directly.
All methods perform vector search with `q="*"`; see the
[Typesense 30.2 vector-search documentation](https://typesense.org/docs/30.2/api/vector-search.html)
for server-side vector options:

```python
documents = store.similarity_search("release notes", k=4)
documents_and_distances = store.similarity_search_with_score("release notes", k=4)

query_vector = embeddings.embed_query("release notes")
documents = store.similarity_search_by_vector(query_vector, k=4)
```

Async equivalents are `asimilarity_search`, `asimilarity_search_with_score`, and
`asimilarity_search_by_vector`. `k` defaults to 4 and must be non-negative; zero returns
an empty list. A missing collection is reported by the Typesense client.

`*_with_score` returns the raw Typesense vector distance, where lower is better. With
cosine distance, identical vectors have distance 0 and the maximum is 2. LangChain's
`similarity_search_with_relevance_scores` converts cosine distance to a bounded score:

```text
relevance = clamp(1 - distance / 2, 0, 1)
```

Use `score_threshold` with normalized relevance, or `distance_threshold` with raw
Typesense distance; they are not interchangeable. Inner-product (`vec_dist="ip"`)
search returns raw distances but does not provide bounded relevance scores or local
cosine-based MMR.

### Filters and search options

Dictionary filters target nested metadata and combine entries with `&&`:

```python
matches = store.similarity_search(
    "release notes",
    k=8,
    filter={"source": "docs", "year": 2026, "published": True},
)
```

Use a raw Typesense `filter_by` expression for ranges or more advanced logic. The
[Typesense 30.2 search documentation](https://typesense.org/docs/30.2/api/search.html)
defines the filter syntax:

```python
matches = store.similarity_search(
    "release notes",
    filter="metadata.year:>=2024 && metadata.source:=docs",
)
```

Raw filter strings are trusted Typesense syntax. Validate or construct them safely when
values come from users.

Vector-search tuning options include `distance_threshold`, `ef`, and
`flat_search_cutoff`. `search_parameters` provides a typed way to pass additional
Typesense options, for example:

```python
matches = store.similarity_search(
    "release notes",
    k=8,
    ef=100,
    search_parameters={"enable_lazy_filter": True},
)
```

The integration owns `q`, `vector_query`, `per_page`, `page`, `filter_by`, and
`exclude_fields`; those names cannot be overridden. Other options that change the
response shape, ranking semantics, pagination, or required fields are rejected so
results remain parseable and `k` remains meaningful. The explicit `curation_tags` and
`diversity_lambda` options are the exception: they allow configured Typesense curations
to reorder results. Unsupported keyword arguments are rejected instead of silently
ignored.

## Maximal marginal relevance

MMR fetches vector candidates and reranks them in the application process using
LangChain's cosine-based implementation:

```python
documents = store.max_marginal_relevance_search(
    "release notes",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,
    filter={"source": "docs"},
)
```

`k` defaults to 4, `fetch_k` to 20, and `lambda_mult` to 0.5. Counts must be
non-negative; `lambda_mult` must be between 0 and 1. The by-vector and async variants
are available. MMR requires `vec_dist="cosine"`; use Typesense's own server-side
curation when centralized diversity rules or avoiding vector transfer is more important.

## Retrieve, delete, and manage collections

```python
documents = store.get_by_ids(["doc-1", "missing", "doc-2"])
store.delete(ids=["doc-1", "doc-2"])

store.create_collection(num_dim=1536)  # create or validate
deleted = store.delete_collection()  # False when already absent
```

`get_by_ids` ignores missing IDs and returns each requested ID at most once; the result
count need not equal the request count. `delete(ids=[])` is a no-op. Bare `delete()` is
rejected; clearing all documents requires `delete(allow_delete_all=True)`, which keeps
the collection schema. `delete_collection()` removes the collection. Async equivalents
are available for each operation.

## Use as a retriever

The store follows LangChain's `VectorStore` contract:

```python
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 8, "score_threshold": 0.75},
)
documents = retriever.invoke("release notes")
```

LangChain search types are `similarity`, `mmr`, and
`similarity_score_threshold`. The inherited `search` and `asearch` helpers are also
available.

## Errors

Integration-defined errors are exported from `langchain_typesense`:

```python
from langchain_typesense import (
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStoreError,
)
```

- `TypesenseCollectionError`: incompatible schema or malformed stored document.
- `TypesenseImportError`: one or more records rejected by bulk import; inspect
  `.failures`.
- `TypesenseVectorStoreError`: malformed or unexpected Typesense response.

Typesense client and embedding exceptions propagate unchanged, so callers can handle
authentication, network, and provider failures using their original exception types.

## Migrating from `langchain-community`

```python
# Before
from langchain_community.vectorstores import Typesense

# After
from langchain_typesense import TypesenseVectorStore
```

`Typesense` remains an alias. The main constructor changes are:

- `typesense_client` → `client`
- `typesense_collection_name` → `collection_name`
- `typesense_url` and `api_key` are used by `from_client_params`
- `async_client` enables native async I/O
- `index_metadata` controls indexing for new collections

This integration uses a managed nested metadata schema, validates existing collections,
preserves `Document.id`, rejects unknown options, and reports partial import failures.
An old collection is reusable only if it passes schema validation; otherwise create a
new collection, reindex source documents, validate representative searches, and switch
traffic after verification.

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

Unit tests block network access:

```bash
uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/
```

Run integration tests against local Typesense:

```bash
docker compose up -d
make integration_test
docker compose down
```

The integration suite uses a unique collection per test and removes those collections
during cleanup. `langchain-tests` is pinned because contract assertions can change in
minor releases.
