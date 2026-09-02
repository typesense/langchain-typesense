from importlib import metadata

from langchain_typesense._errors import (
    TypesenseCollectionError,
    TypesenseImportError,
    TypesenseVectorStoreError,
)
from langchain_typesense._types import (
    ClientMode,
    TypesenseHybridSearchParameters,
    TypesenseSearchParameters,
    VectorDistance,
)
from langchain_typesense.vectorstores import Typesense, TypesenseVectorStore

try:
    __version__ = metadata.version("langchain-typesense")
except metadata.PackageNotFoundError:
    # package is not installed, e.g. running from source without `uv sync`
    __version__ = ""

__all__ = [
    "ClientMode",
    "Typesense",
    "TypesenseCollectionError",
    "TypesenseHybridSearchParameters",
    "TypesenseImportError",
    "TypesenseSearchParameters",
    "TypesenseVectorStore",
    "TypesenseVectorStoreError",
    "VectorDistance",
    "__version__",
]
