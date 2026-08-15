# langchain-typesense

`langchain-typesense` is a typed, standalone LangChain `VectorStore` integration for
[Typesense](https://typesense.org). It replaces the archived
`langchain_community.vectorstores.Typesense` integration and follows the current
LangChain vector-store contract, including synchronous and asynchronous operations,
relevance scores, maximal marginal relevance (MMR), ID lookup, and standard integration
tests.

## Requirements

- Python 3.10 or newer
- `langchain-core` 1.x
- `typesense` Python client 2.x
- A Typesense server with vector-search support
- A LangChain `Embeddings` implementation whose document and query vectors have the
  same, stable dimension

Install the integration and the package for your embedding provider. For example:

```bash
pip install -U langchain-typesense langchain-openai
```

This package does not install an embedding provider automatically.

## Start Typesense locally

The repository includes a Typesense 30.2 development service:

```bash
docker compose up -d
```

It listens on `http://localhost:8108` and uses the development API key `xyz`. Stop it
when finished:

```bash
docker compose down
```

The Compose configuration stores server data in `./typesense-data`. Stopping the
container does not delete that data.

## Quickstart

```python
import typesense
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_typesense import TypesenseVectorStore

config = {
    "nodes": [{"host": "localhost", "port": 8108, "protocol": "http"}],
    "api_key": "xyz",
    "connection_timeout_seconds": 2,
}

store = TypesenseVectorStore(
    client=typesense.Client(config),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my-documents",
)

documents = [
    Document(page_content="Typesense is fast", metadata={"source": "docs"}),
    Document(
        page_content="LangChain composes AI applications",
        metadata={"source": "blog"},
    ),
]
ids = store.add_documents(documents, ids=["typesense", "langchain"])

matches = store.similarity_search(
    "Which search engine is fast?",
    k=2,
    filter={"source": "docs"},
)

store.close()
```

The collection is created lazily on the first non-empty write. If it already exists,
its schema is validated before documents are imported.

## Client construction

### Synchronous client

Pass an existing `typesense.Client` when the application owns client configuration:

```python
import typesense
from langchain_openai import OpenAIEmbeddings
from langchain_typesense import TypesenseVectorStore

config = {
    "nodes": [{"host": "localhost", "port": 8108, "protocol": "http"}],
    "api_key": "xyz",
    "connection_timeout_seconds": 2,
}

store = TypesenseVectorStore(
    client=typesense.Client(config),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my-documents",
)
```

If only a synchronous client is configured, asynchronous methods remain available.
They execute their synchronous equivalents in LangChain's worker executor.

### Native asynchronous I/O

Pass both client types to use the Typesense async client and the embedding provider's
async methods:

```python
import typesense
from langchain_openai import OpenAIEmbeddings
from langchain_typesense import TypesenseVectorStore

config = {
    "nodes": [{"host": "localhost", "port": 8108, "protocol": "http"}],
    "api_key": "xyz",
    "connection_timeout_seconds": 2,
}

store = TypesenseVectorStore(
    client=typesense.Client(config),
    async_client=typesense.AsyncClient(config),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my-documents",
)

ids = await store.aadd_texts(
    ["alpha", "beta"],
    metadatas=[{"source": "a"}, {"source": "b"}],
    ids=["alpha", "beta"],
)
matches = await store.asimilarity_search("alpha", k=2)

# Closes the async client first, then the sync client.
await store.aclose()
```

`close()` closes only the synchronous client's HTTP resources. `aclose()` closes both
configured clients. Do not use either client after closing the store.

### Connection-parameter constructor and environment key

`from_client_params` creates both sync and async clients. It accepts `api_key` directly
or reads `TYPESENSE_API_KEY` when neither API-key argument is supplied:

```bash
export TYPESENSE_API_KEY="your-typesense-api-key"
```

```python
store = TypesenseVectorStore.from_client_params(
    embedding=embeddings,
    host="example.a1.typesense.net",
    port=443,
    protocol="https",
    connection_timeout_seconds=2.0,
    collection_name="my-documents",
)
```

The complete signature is:

```python
TypesenseVectorStore.from_client_params(
    embedding,
    *,
    host="localhost",
    port=8108,
    protocol="http",
    api_key=None,
    typesense_api_key=None,
    connection_timeout_seconds=2.0,
    collection_name="langchain-typesense",
    **constructor_options,
)
```

`typesense_api_key` is retained as a migration alias. Passing different non-empty
values for `api_key` and `typesense_api_key` raises `ValueError`.

### Constructor options

```python
TypesenseVectorStore(
    client,
    embedding,
    *,
    collection_name="langchain-typesense",
    async_client=None,
    text_key="text",
    vector_key="vec",
    metadata_key="metadata",
    index_metadata=True,
)
```

`text_key`, `vector_key`, and `metadata_key` must be non-empty, distinct, and different
from Typesense's reserved `id` field. These names must match an existing collection if
one is reused.

## Collection schema and metadata

The managed collection schema is based on the first embedding dimension:

```text
enable_nested_fields: true

text       string
vec        float[]  (num_dim=<embedding dimension>, vec_dist=cosine)
metadata   object   (optional, indexed by default)
```

Field names shown above are defaults and can be changed with constructor options.
Metadata is stored under one nested object, preventing metadata keys such as `id`,
`text`, or `vec` from overwriting managed Typesense fields. Returned LangChain
`Document` objects contain the nested values in `Document.metadata` and retain the
Typesense document ID in `Document.id`.

Set `index_metadata=False` when metadata only needs to be stored and returned:

```python
store = TypesenseVectorStore(
    client=client,
    embedding=embeddings,
    collection_name="unfiltered-documents",
    index_metadata=False,
)
```

Dictionary metadata filters require indexed metadata. The setting affects newly
created collections; it does not rewrite an existing schema.

For an existing collection, the integration validates:

- text field type
- vector field type and dimension
- cosine distance
- metadata object type
- nested-field support
- metadata indexing when `index_metadata=True`

An incompatible collection raises `TypesenseCollectionError` before import. A
collection creation race is handled by retrieving and validating the collection created
by the competing process.

### Document IDs and upserts

`add_documents` resolves IDs in this order:

1. Explicit `ids=` values
2. Each `Document.id`
3. Generated UUIDs

Explicit IDs must have the same count as the documents. IDs must be non-empty and use
only URL-unreserved characters: letters, digits, `-`, `.`, `_`, and `~`. Reusing an ID
upserts that Typesense document. The input `Document` objects are not mutated.

## Add and update data

### Documents

```python
ids = store.add_documents(
    [
        Document(id="doc-1", page_content="First", metadata={"group": "a"}),
        Document(id="doc-2", page_content="Second", metadata={"group": "b"}),
    ],
    batch_size=100,
)
```

`add_documents(documents, *, ids=None, batch_size=None)` embeds all document text,
creates or validates the collection, bulk-upserts the records, checks each Typesense
import result, and returns the resolved IDs. `batch_size` must be greater than zero when
provided and is passed to the Typesense import operation.

`aadd_documents` has the same arguments and return value. With an async client, it uses
`aembed_documents` and native async Typesense calls.

### Texts

```python
ids = store.add_texts(
    ["First", "Second"],
    metadatas=[{"group": "a"}, {"group": "b"}],
    ids=["doc-1", "doc-2"],
    batch_size=100,
)
```

`add_texts(texts, metadatas=None, *, ids=None, **add_options)` converts strings to
`Document` objects and delegates to `add_documents`. `aadd_texts` is its asynchronous
equivalent. Metadata and ID counts must match the number of texts.

Empty input returns an empty list without creating a collection or calling the embedding
model. Supplying non-empty IDs for empty input raises `ValueError`.

### One-step constructors

`from_texts` creates a store using a required existing sync client, inserts the texts,
and returns the store:

```python
store = TypesenseVectorStore.from_texts(
    ["First", "Second"],
    embeddings,
    metadatas=[{"group": "a"}, {"group": "b"}],
    ids=["doc-1", "doc-2"],
    client=client,
    async_client=async_client,
    collection_name="my-documents",
)
```

The inherited LangChain `from_documents` constructor delegates to `from_texts`, so it
also requires `client=...`. The inherited `afrom_texts` and `afrom_documents`
constructors run this one-step synchronous construction in a worker executor. For
native asynchronous ingestion, construct the store with `async_client` and call
`aadd_texts` or `aadd_documents` directly.

## Similarity search

All searches are pure vector searches (`q="*"`). Text-query methods embed the query;
by-vector methods use the supplied embedding directly.

```python
documents = store.similarity_search("release notes", k=4)

documents_and_distance = store.similarity_search_with_score(
    "release notes",
    k=4,
)

query_vector = embeddings.embed_query("release notes")
documents = store.similarity_search_by_vector(query_vector, k=4)
```

Asynchronous equivalents are `asimilarity_search`,
`asimilarity_search_with_score`, and `asimilarity_search_by_vector`.

`k` defaults to `4`, must be non-negative, and returns at most that many documents. A
zero value returns an empty list. Searching a collection that does not exist also
returns an empty list.

### Distance and relevance scores

`similarity_search_with_score` and `asimilarity_search_with_score` return
`(Document, distance)` pairs using raw Typesense cosine distance:

- `0` means identical direction and is best.
- `2` is the maximum cosine distance and is worst.
- Smaller values are more similar.

LangChain's inherited `similarity_search_with_relevance_scores` and async equivalent
return `(Document, relevance)` pairs. This integration converts distance with:

```text
relevance = clamp(1 - distance / 2, 0, 1)
```

The resulting range is `[0, 1]`, where `1` is most relevant. Use LangChain's
`score_threshold` to post-filter this normalized score:

```python
matches = store.similarity_search_with_relevance_scores(
    "release notes",
    k=10,
    score_threshold=0.8,
)
```

`score_threshold` and `distance_threshold` use opposite scales and are not
interchangeable: `score_threshold` is a minimum normalized relevance applied by
LangChain, while `distance_threshold` is a maximum raw vector distance sent to
Typesense.

## Filters

Dictionary filters provide equality matching against nested metadata:

```python
matches = store.similarity_search(
    "release notes",
    k=8,
    filter={
        "source": "docs",
        "year": 2026,
        "published": True,
        "category": ["guide", "reference"],
    },
)
```

Mapping entries are combined with `&&`. Scalar values use exact equality. A sequence
produces a Typesense list equality filter. Mapping keys may contain letters, digits,
underscores, hyphens, and dots, and must start with a letter or underscore. Values can
be strings, integers, finite floats, or booleans.

Pass a raw Typesense `filter_by` expression for ranges, boolean logic, geospatial
filters, or fields managed outside the nested metadata object:

```python
matches = store.similarity_search(
    "release notes",
    filter="metadata.year:>=2024 && metadata.source:=docs",
)
```

Raw strings are passed through unchanged. Treat them as trusted query syntax; validate
or construct them safely before accepting user-controlled input.

## Typesense search tuning

Every similarity and MMR method accepts the following Typesense-specific options:

- `distance_threshold`: finite, non-negative maximum cosine distance
- `ef`: positive HNSW candidate-list size
- `flat_search_cutoff`: non-negative cutoff for flat vector search
- `search_parameters`: additional Typesense document-search parameters
- `filter`: dictionary metadata filter or raw Typesense expression

Example:

```python
matches = store.similarity_search(
    "release notes",
    k=8,
    filter={"source": "docs"},
    distance_threshold=0.35,
    ef=100,
    flat_search_cutoff=20,
    search_parameters={"enable_lazy_filter": True},
)
```

The integration owns `q`, `vector_query`, `per_page`, `page`, `filter_by`, and
`exclude_fields`. Supplying any of these names through `search_parameters` raises
`ValueError`. Unsupported keyword options raise `TypeError` instead of being silently
ignored.

## Maximal marginal relevance

MMR first fetches nearest candidates, including their vectors, and then uses LangChain's
MMR implementation to balance similarity against diversity:

```python
documents = store.max_marginal_relevance_search(
    "release notes",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,
    filter={"source": "docs"},
)
```

Available methods are:

- `max_marginal_relevance_search`
- `max_marginal_relevance_search_by_vector`
- `amax_marginal_relevance_search`
- `amax_marginal_relevance_search_by_vector`

`k` defaults to `4`, `fetch_k` to `20`, and `lambda_mult` to `0.5`. The store requests
`max(k, fetch_k)` candidates. `lambda_mult=1` favors similarity; `lambda_mult=0` favors
diversity. Both counts must be non-negative, and `lambda_mult` must be in `[0, 1]`.
Passing zero for either count returns an empty list.

## Retrieve and delete by ID

```python
documents = store.get_by_ids(["doc-1", "missing", "doc-2"])
documents = await store.aget_by_ids(["doc-1", "doc-2"])

store.delete(ids=["doc-1", "doc-2"])
await store.adelete(ids=["doc-3"])
```

`get_by_ids` and `aget_by_ids` ignore missing IDs and retrieve duplicate requested IDs
only once. Callers must not assume the number of returned documents matches the number
of requested IDs.

`delete(ids=[...])` bulk-deletes selected documents. `delete(ids=[])` is a successful
no-op. `delete()` with no IDs truncates every document but preserves the collection and
schema. `delete` and `adelete` return `True`; deleting from a missing collection is also
treated as successful.

## Collection lifecycle and cleanup

Collection management is explicit when needed:

```python
store.create_collection(num_dim=1536)          # create or validate
await store.acreate_collection(num_dim=1536)

deleted = store.delete_collection()            # False if already absent
deleted = await store.adelete_collection()
```

Use `delete()` to clear documents while retaining the schema. Use
`delete_collection()` to remove the collection itself. Client cleanup is separate:

```python
store.close()
# or, when an async client was supplied:
await store.aclose()
```

## Use as a LangChain retriever

Because the class implements LangChain's `VectorStore` contract, it can be converted to
a retriever:

```python
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 8,
        "score_threshold": 0.75,
        "filter": {"source": "docs"},
    },
)

documents = retriever.invoke("release notes")
```

Supported LangChain search types are `similarity`, `mmr`, and
`similarity_score_threshold`. The same choices are available through inherited `search`
and `asearch` methods.

## Public API summary

### Properties

- `client`: configured synchronous Typesense client
- `async_client`: configured async client, or `None`
- `collection_name`: backing collection name
- `embeddings`: configured LangChain embedding model

### Construction and lifecycle

- `TypesenseVectorStore(...)`
- `from_client_params(...)`
- `from_texts(...)` and inherited `from_documents(...)`
- inherited `afrom_texts(...)` and `afrom_documents(...)`
- `create_collection(...)` and `acreate_collection(...)`
- `delete_collection()` and `adelete_collection()`
- `close()` and `aclose()`

### Writes, reads, and deletes

- `add_documents(...)` and `aadd_documents(...)`
- `add_texts(...)` and `aadd_texts(...)`
- `get_by_ids(...)` and `aget_by_ids(...)`
- `delete(...)` and `adelete(...)`

### Search

- `similarity_search(...)` and `asimilarity_search(...)`
- `similarity_search_with_score(...)` and `asimilarity_search_with_score(...)`
- `similarity_search_by_vector(...)` and `asimilarity_search_by_vector(...)`
- inherited `similarity_search_with_relevance_scores(...)` and async equivalent
- `max_marginal_relevance_search(...)` and async equivalent
- `max_marginal_relevance_search_by_vector(...)` and async equivalent
- inherited `search(...)`, `asearch(...)`, and `as_retriever(...)`

## Error handling

All integration-defined errors inherit from `TypesenseVectorStoreError`:

```python
from langchain_typesense import (
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStoreError,
)
```

- `TypesenseCollectionError` reports incompatible schemas or malformed stored
  documents.
- `TypesenseImportError` reports one or more rejected bulk-import records. Its
  `.failures` attribute is a tuple containing copies of Typesense's failure records.
- `TypesenseVectorStoreError` reports malformed or unexpected import/search responses
  and other integration-level failures.

Typesense's bulk import endpoint can return HTTP success while rejecting individual
records. This integration validates every returned result and raises
`TypesenseImportError` when any record failed:

```python
from langchain_typesense import TypesenseImportError

try:
    store.add_documents(documents, ids=ids)
except TypesenseImportError as error:
    for failure in error.failures:
        print(failure.get("code"), failure.get("error"), failure.get("document"))
```

Typesense does not roll back successful records in a partially failed batch. Recovery
logic should inspect `.failures` and reconcile or retry only rejected IDs. Network,
authentication, and other Typesense client exceptions propagate from the Typesense
client. Invalid local arguments raise `ValueError` or `TypeError` before a request where
possible.

## Troubleshooting

### Deprecation warnings from `typesense-python` 2.0.0

Tests may report warnings such as:

```text
DeprecationWarning: SyncAnalyticsV1 is deprecated on v30+.
```

`typesense-python` 2.0.0 eagerly instantiates deprecated analytics compatibility
objects while constructing a client, and deprecated overrides and synonyms objects
while constructing a collection handle. `langchain-typesense` does not call those APIs;
it uses current collection and document operations. These warnings are advisory and
originate in the upstream client, so passing integration tests do not indicate use of a
deprecated vector-store operation.

## Production guidance

- Keep the embedding model and output dimension stable for the lifetime of a
  collection. A dimension change requires a compatible new collection and reindex.
- Use explicit, stable IDs when writes may be retried; imports use upsert semantics.
- Reuse configured clients instead of creating a client per request, then close their
  HTTP resources during application shutdown.
- Supply `async_client` in async services to avoid worker-thread fallback.
- Keep metadata indexed only when filters need it. Validate filter behavior against the
  production Typesense schema.
- Handle `TypesenseImportError` as partial success, not as an atomic transaction
  failure.
- Use least-privilege Typesense API keys and do not commit keys to source control.
- Test collection-schema compatibility and representative searches before switching an
  existing production collection.

## Migrating from `langchain-community`

Change the import and preferred class name:

```python
# Before
from langchain_community.vectorstores import Typesense

# After
from langchain_typesense import TypesenseVectorStore
```

`Typesense` remains an alias for `TypesenseVectorStore`:

```python
from langchain_typesense import Typesense
```

Constructor changes:

- `typesense_client` becomes `client`.
- `typesense_collection_name` becomes `collection_name`.
- `text_key` remains supported.
- `vector_key="vec"` and `metadata_key="metadata"` are now configurable.
- `async_client` enables native async Typesense operations.
- `index_metadata` controls indexing when creating the collection.
- `from_client_params` accepts preferred `api_key` and legacy
  `typesense_api_key` names.

Behavioral changes:

- Search `k` now defaults to LangChain's standard value of `4`, rather than `10`.
- `Document.id` is preserved on reads and used on writes when explicit `ids` are absent.
- Existing collections are validated before import.
- Metadata has a managed nested object schema with nested fields enabled.
- Bulk-import partial failures raise `TypesenseImportError`.
- Dictionary metadata filters, by-vector search, relevance scoring, MMR, ID lookup,
  deletion, and native async methods are available.
- Unknown options are rejected instead of silently ignored.

The archived integration created collections with a dynamic wildcard field and did not
explicitly enable the new managed nested schema. Do not assume such a collection is
compatible. The safest migration is to create a new collection with this integration,
re-embed or copy source documents into it, validate search and filters, and then switch
application traffic. An existing collection can be reused only when it passes the
schema checks described above. Back up production data before migration.

## Development

Install `uv`, clone the repository, and install every dependency group:

```bash
uv sync --all-groups
```

Run formatting, linting, both static type checkers, unit tests, and package builds.
`make lint` runs Ruff, mypy, and ty:

```bash
make format
make lint
make test
make build
```

Unit tests block network access and use mocked Typesense clients:

```bash
uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/
```

Start Typesense and run the real LangChain contract suite:

```bash
docker compose up -d
make integration_test
docker compose down
```

Integration tests create a unique collection per test, exercise LangChain's standard
sync and async `VectorStoreIntegrationTests`, run package-specific filter and relevance
assertions, and delete their collections during fixture cleanup. `langchain-tests` is
pinned because minor releases can add new contract assertions.

CI runs on pushes to `main` and pull requests. Quality jobs cover Python 3.10 and 3.14,
including Ruff, formatting, static typing, unit tests, and distribution builds. A
separate Python 3.12 job starts Typesense 30.2 and executes the real integration suite.
