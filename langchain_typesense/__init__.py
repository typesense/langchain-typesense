from importlib import metadata

from langchain_typesense.vectorstores import (
    ClientMode,
    Typesense,
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseSearchParameters,
    TypesenseVectorStore,
    TypesenseVectorStoreError,
    VectorDistance,
)

try:
    __version__ = metadata.version("langchain-typesense")
except metadata.PackageNotFoundError:
    # package is not installed, e.g. running from source without `uv sync`
    __version__ = ""

__all__ = [
    "ClientMode",
    "Typesense",
    "TypesenseCollectionError",
    "TypesenseImportError",
    "TypesenseSearchParameters",
    "TypesenseVectorStore",
    "TypesenseVectorStoreError",
    "VectorDistance",
    "__version__",
]
