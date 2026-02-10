from .base import EmbeddingProvider
from huggingface_hub import InferenceClient
import numpy as np
import os
from typing import List, Optional, Any
from dotenv import load_dotenv
from logger import Logger

load_dotenv()


def _to_flat_embedding(result: Any) -> list[float]:
    """Convert API response to a flat list of floats. Handles (1,d), (d,1), [[...]], numpy."""
    if hasattr(result, "flatten"):
        return np.asarray(result).flatten().tolist()
    if isinstance(result, list):
        if len(result) == 0:
            raise ValueError("Empty embedding result")
        first = result[0]
        if isinstance(first, (int, float)):
            return [float(x) for x in result]
        if isinstance(first, list):
            return _to_flat_embedding(first)
        if hasattr(first, "flatten"):
            return np.asarray(first).flatten().tolist()
    return [float(x) for x in result]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Hugging Face embedding provider using the Inference API (feature extraction).
    Supports text embedding models such as sentence-transformers, and multimodal
    models like Qwen/Qwen3-VL-Embedding-2B (text-only input is supported).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        normalize: Optional[bool] = None,
        output_dimensionality: Optional[int] = None,
    ):
        """
        Initialize the Hugging Face embedding provider.

        Args:
            api_key: Hugging Face token. If not provided, uses HUGGINGFACE_API_KEY or HF_TOKEN env var.
            model: Model id (e.g. "sentence-transformers/all-MiniLM-L6-v2", "Qwen/Qwen3-VL-Embedding-2B").
                   If not provided, uses HUGGINGFACE_EMBEDDING_MODEL env var.
            provider: Optional inference provider (e.g. "hf-inference", "featherless-ai"). Default "auto".
            normalize: Optional. If True, normalize embeddings (useful for cosine similarity).
            output_dimensionality: Optional. Output dimension (e.g. 32-4096 for Qwen3-Embedding). Sent to API when supported; otherwise embedding is truncated. Use same value for vector store dimension.
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.model = model or os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.provider = provider if provider is not None else os.getenv("HUGGINGFACE_EMBEDDING_PROVIDER", "auto")
        self.normalize = normalize
        self.output_dimensionality = output_dimensionality

        if not self.api_key:
            raise ValueError(
                "Hugging Face API key not found. Provide api_key parameter or set HUGGINGFACE_API_KEY environment variable"
            )

        init_kwargs = {"token": self.api_key, "model": self.model}
        if self.provider and self.provider != "auto":
            init_kwargs["provider"] = self.provider
        self.client = InferenceClient(**init_kwargs)
        Logger.debug(
            f"Initialized Hugging Face embedding provider with model: {self.model}",
            "[HuggingFaceEmbeddingProvider]",
        )

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the provided text using Hugging Face Inference API.

        Args:
            text: Input text to generate embedding for.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            Exception: If embedding generation fails.
        """
        try:
            Logger.debug(f"Generating embedding for text: {text[:50]}...", "[HuggingFaceEmbeddingProvider]")

            kwargs = {"model": self.model}
            if self.normalize is not None:
                kwargs["normalize"] = self.normalize
            if self.output_dimensionality is not None:
                kwargs["extra_body"] = {"dimensions": self.output_dimensionality}

            try:
                result = self.client.feature_extraction(text, **kwargs)
            except TypeError:
                kwargs.pop("extra_body", None)
                result = self.client.feature_extraction(text, **kwargs)
            embedding = _to_flat_embedding(result)
            if self.output_dimensionality is not None and len(embedding) > self.output_dimensionality:
                embedding = embedding[: self.output_dimensionality]

            Logger.debug(
                f"Successfully generated embedding (dimensions: {len(embedding)})",
                "[HuggingFaceEmbeddingProvider]",
            )
            return embedding

        except Exception as e:
            Logger.debug(f"Failed to generate embedding: {str(e)}", "[HuggingFaceEmbeddingProvider]")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def batch_generate_embeddings(self, texts: List[str]) -> List[list[float]]:
        """
        Generate embeddings for multiple texts in one request when supported by the API.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        try:
            Logger.debug(f"Batch generating embeddings for {len(texts)} texts", "[HuggingFaceEmbeddingProvider]")

            kwargs = {"model": self.model}
            if self.normalize is not None:
                kwargs["normalize"] = self.normalize
            if self.output_dimensionality is not None:
                kwargs["extra_body"] = {"dimensions": self.output_dimensionality}

            try:
                result = self.client.feature_extraction(texts, **kwargs)
            except TypeError:
                kwargs.pop("extra_body", None)
                result = self.client.feature_extraction(texts, **kwargs)
            if isinstance(result, list) and len(result) == len(texts):
                embeddings = [_to_flat_embedding(vec) for vec in result]
            elif isinstance(result, list) and len(result) == 1 and len(texts) == 1:
                embeddings = [_to_flat_embedding(result)]
            else:
                arr = np.asarray(result)
                if arr.ndim == 2 and arr.shape[0] == len(texts):
                    embeddings = [arr[i].flatten().tolist() for i in range(arr.shape[0])]
                else:
                    embeddings = [_to_flat_embedding(result)]
            if self.output_dimensionality is not None:
                embeddings = [e[: self.output_dimensionality] if len(e) > self.output_dimensionality else e for e in embeddings]
            if len(embeddings) != len(texts):
                # Fallback to single requests
                return [self.generate_embedding(t) for t in texts]

            Logger.debug(f"Successfully generated {len(embeddings)} embeddings", "[HuggingFaceEmbeddingProvider]")
            return embeddings

        except Exception as e:
            Logger.debug(f"Batch embedding failed: {e}, falling back to single requests", "[HuggingFaceEmbeddingProvider]")
            return [self.generate_embedding(text) for text in texts]
