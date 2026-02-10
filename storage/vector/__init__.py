"""Vector storage module for storing and searching embeddings."""

from .base import BaseVectorStore
from .faiss_store import FAISSVectorStore
from .factory import create_vector_store

__all__ = ["BaseVectorStore", "FAISSVectorStore", "create_vector_store"]
