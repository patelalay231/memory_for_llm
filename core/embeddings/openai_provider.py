from .base import EmbeddingProvider
from openai import OpenAI
import os
from typing import Optional
from dotenv import load_dotenv
from logger import Logger

load_dotenv()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            api_key: Optional API key. If not provided, will try to get from OPENAI_API_KEY env var.
            model: Optional model name (defaults to OPENAI_EMBEDDING_MODEL env var or "text-embedding-3-small")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("❌ Error: OPENAI_API_KEY not found. Provide api_key parameter or set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        Logger.debug(f"Initialized OpenAI embedding provider with model: {self.model}", "[OpenAIEmbeddingProvider]")
    
    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the provided text using OpenAI API.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            Logger.debug(f"Generating embedding for text: {text[:50]}...", "[OpenAIEmbeddingProvider]")
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            Logger.debug(f"Successfully generated embedding (dimensions: {len(embedding)})", "[OpenAIEmbeddingProvider]")
            
            return embedding
            
        except Exception as e:
            Logger.debug(f"Failed to generate embedding: {str(e)}", "[OpenAIEmbeddingProvider]")
            raise Exception(f"Failed to generate embedding: {str(e)}")
