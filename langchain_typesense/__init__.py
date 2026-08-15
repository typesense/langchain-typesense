from importlib import metadata

from langchain_typesense.vectorstores import (
    Typesense,
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStore,
    TypesenseVectorStoreError,
)

try:
    __version__ = metadata.version("langchain-typesense")
except metadata.PackageNotFoundError:
    # package is not installed, e.g. running from source without `uv sync`
    __version__ = ""

__all__ = [
    "Typesense",
    "TypesenseCollectionError",
    "TypesenseImportError",
    "TypesenseVectorStore",
    "TypesenseVectorStoreError",
    "__version__",
]
