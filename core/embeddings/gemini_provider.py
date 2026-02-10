from .base import EmbeddingProvider
from google import genai
from google.genai import types
import os
from typing import Optional
from dotenv import load_dotenv
from logger import Logger

load_dotenv()


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Gemini embedding provider implementation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        output_dimensionality: Optional[int] = None
    ):
        """
        Initialize the Gemini embedding provider.
        
        Args:
            api_key: Optional API key. If not provided, will try to get from GEMINI_API_KEY env var.
            model: Optional model name (defaults to GEMINI_EMBEDDING_MODEL env var or "gemini-embedding-001")
            task_type: Optional task type for optimization (e.g., "RETRIEVAL_DOCUMENT")
            output_dimensionality: Optional output dimension size 
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        
        if not self.api_key:
            raise ValueError("❌ Error: GEMINI_API_KEY not found. Provide api_key parameter or set GEMINI_API_KEY environment variable")
        
        self.client = genai.Client(api_key=self.api_key)
        Logger.debug(f"Initialized Gemini embedding provider with model: {self.model}", "[GeminiEmbeddingProvider]")
        if self.task_type:
            Logger.debug(f"Task type: {self.task_type}", "[GeminiEmbeddingProvider]")
        if self.output_dimensionality:
            Logger.debug(f"Output dimensionality: {self.output_dimensionality}", "[GeminiEmbeddingProvider]")
    
    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the provided text using Gemini API.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            Logger.debug(f"Generating embedding for text: {text[:50]}...", "[GeminiEmbeddingProvider]")
            
            # Build config if we have optional parameters
            config = None
            if self.task_type or self.output_dimensionality:
                config = types.EmbedContentConfig()
                if self.task_type:
                    config.task_type = self.task_type
                if self.output_dimensionality:
                    config.output_dimensionality = self.output_dimensionality
            
            # Generate embedding
            if config:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=config
                )
            else:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text
                )
            
            # Extract embedding values (result.embeddings is a list, get first item)
            if not result.embeddings or len(result.embeddings) == 0:
                raise Exception("No embeddings returned from Gemini API")
            
            embedding = list(result.embeddings[0].values)
            Logger.debug(f"Successfully generated embedding (dimensions: {len(embedding)})", "[GeminiEmbeddingProvider]")
            
            return embedding
            
        except Exception as e:
            Logger.debug(f"Failed to generate embedding: {str(e)}", "[GeminiEmbeddingProvider]")
            raise Exception(f"Failed to generate embedding: {str(e)}")
