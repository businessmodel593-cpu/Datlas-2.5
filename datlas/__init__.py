"""Datlas 2.5 package."""

__version__ = "2.5.0"

from datlas.cache import ResultCache
from datlas.models.transformer import DatlasModel
from datlas.vector_store import LocalVectorStore, PersistentLocalVectorStore

__all__ = ["DatlasModel", "LocalVectorStore", "PersistentLocalVectorStore", "ResultCache"]
