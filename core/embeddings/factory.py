from typing import Dict, Any, Union
from pydantic import BaseModel
from .embedding_generator import EmbeddingGenerator
from .openai_provider import OpenAIEmbeddingProvider
from .gemini_provider import GeminiEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from core.utils import ConfigValidator
from logger import Logger


def create_embedding_generator(config: Union[BaseModel, Dict[str, Any]]) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator from config.
    
    Args:
        config: Configuration with provider name as key (e.g., {"openai": {...}}, {"gemini": {...}}, or {"huggingface": {...}})
        
    Returns:
        EmbeddingGenerator instance
        
    Raises:
        ValueError: If provider is not supported or config is invalid
        
    Example:
        generator = create_embedding_generator({
            "openai": {
                "api_key": "sk-your-api-key",
                "model": "text-embedding-3-small"
            }
        })
        
        generator = create_embedding_generator({
            "huggingface": {
                "api_key": "hf_...",
                "model": "Qwen/Qwen3-VL-Embedding-2B",
                "provider": "hf-inference"
            }
        })
    """
    # Extract provider name and config
    provider_name, provider_config = ConfigValidator.extract_provider_config(
        config,
        "Embedding",
        ["openai", "gemini", "huggingface"]
    )
    
    # Create provider instance
    Logger.debug(f"Creating embedding provider: {provider_name}", "[EmbeddingFactory]")
    
    if provider_name == "openai":
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=provider_config["api_key"],
            model=provider_config.get("model")
        )
        return EmbeddingGenerator(embedding_provider)
    elif provider_name == "gemini":
        embedding_provider = GeminiEmbeddingProvider(
            api_key=provider_config["api_key"],
            model=provider_config.get("model"),
            task_type=provider_config.get("task_type"),
            output_dimensionality=provider_config.get("output_dimensionality")
        )
        return EmbeddingGenerator(embedding_provider)
    elif provider_name == "huggingface":
        embedding_provider = HuggingFaceEmbeddingProvider(
            api_key=provider_config["api_key"],
            model=provider_config.get("model"),
            provider=provider_config.get("provider"),
            normalize=provider_config.get("normalize"),
            output_dimensionality=provider_config.get("output_dimensionality"),
        )
        return EmbeddingGenerator(embedding_provider)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider_name}")
