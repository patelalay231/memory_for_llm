from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """
    
    @abstractmethod
    def __init__(self):
        """
        Initialize the embedding provider.
        """
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the provided text.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        This is an optional method that can be overridden by providers that support
        batch operations. Default implementation loops through texts and calls
        generate_embedding() for each.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        return [self.generate_embedding(text) for text in texts]
