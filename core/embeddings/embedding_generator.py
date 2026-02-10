from .base import EmbeddingProvider
from logger import Logger
from typing import List


class EmbeddingGenerator:
    """
    Service class for generating embeddings using an embedding provider.
    """
    
    def __init__(self, provider: EmbeddingProvider):
        """
        Initialize the embedding generator.
        
        Args:
            provider: Embedding provider instance (e.g., OpenAIEmbeddingProvider)
        """
        self.provider = provider
        Logger.debug("Initialized EmbeddingGenerator", "[EmbeddingGenerator]")
    
    def generate(self, text: str) -> list[float]:
        """
        Generate an embedding for the provided text.
        
        Args:
            text: Input text to generate embedding for (typically Memory.content)
            
        Returns:
            List of floats representing the embedding vector
        """
        Logger.debug(f"Generating embedding for text: {text[:50]}...", "[EmbeddingGenerator]")
        
        embedding = self.provider.generate_embedding(text)
        
        Logger.debug(f"Successfully generated embedding (dimensions: {len(embedding)})", "[EmbeddingGenerator]")
        return embedding
    
    def generate_batch(self, texts: List[str]) -> List[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        Logger.debug(f"Generating embeddings for {len(texts)} texts", "[EmbeddingGenerator]")
        
        try:
            # Try to use provider's batch method if available
            embeddings = self.provider.batch_generate_embeddings(texts)
            Logger.debug(f"Successfully generated {len(embeddings)} embeddings", "[EmbeddingGenerator]")
            return embeddings
        except Exception as e:
            Logger.debug(f"Batch embedding generation failed: {str(e)}", "[EmbeddingGenerator]")
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
