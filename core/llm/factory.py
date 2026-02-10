from typing import Dict, Any, Union
from pydantic import BaseModel
from .base import LLMProvider
from .gemini import GeminiProvider
from .huggingface import HuggingFaceProvider
from core.utils import ConfigValidator
from logger import Logger


def create_llm_provider(config: Union[BaseModel, Dict[str, Any]]) -> LLMProvider:
    """
    Factory function to create LLM provider from config.
    
    Args:
        config: Configuration with provider name as key (e.g., {"gemini": {...}} or {"huggingface": {...}})
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider is not supported or config is invalid
        
    Example:
        provider = create_llm_provider({
            "gemini": {
                "api_key": "your-api-key",
                "model": "gemini-pro"
            }
        })
        provider = create_llm_provider({
            "huggingface": {
                "api_key": "hf_...",
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "provider": "featherless-ai"
            }
        })
    """
    # Extract provider name and config
    provider_name, provider_config = ConfigValidator.extract_provider_config(
        config,
        "LLM provider",
        ["gemini", "huggingface"]
    )
    
    # Create provider instance
    Logger.debug(f"Creating LLM provider: {provider_name}", "[LLMProviderFactory]")
    
    if provider_name == "gemini":
        return GeminiProvider(
            api_key=provider_config["api_key"],
            model=provider_config["model"]
        )
    elif provider_name == "huggingface":
        return HuggingFaceProvider(
            api_key=provider_config["api_key"],
            model=provider_config["model"],
            provider=provider_config.get("provider"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
