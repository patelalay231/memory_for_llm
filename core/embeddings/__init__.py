"""Embeddings module for generating text embeddings."""

from .base import EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .gemini_provider import GeminiEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from .embedding_generator import EmbeddingGenerator
from .factory import create_embedding_generator

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "EmbeddingGenerator",
    "create_embedding_generator",
]
