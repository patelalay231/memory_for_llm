from typing import Dict, Any, Union
from pydantic import BaseModel
from .base import BaseVectorStore
from .faiss_store import FAISSVectorStore
from core.utils import ConfigValidator
from logger import Logger


def create_vector_store(config: Union[BaseModel, Dict[str, Any]]) -> BaseVectorStore:
    """
    Factory function to create vector store from config.
    
    Args:
        config: Configuration with vector store type as key (e.g., {"faiss": {...}})
        
    Returns:
        BaseVectorStore instance
        
    Raises:
        ValueError: If vector store type is not supported or config is invalid
        
    Example:
        vector_store = create_vector_store({
            "faiss": {
                "dimension": 768,
                "index_path": "./faiss_index",
                "index_type": "L2"  # Options: "L2", "IP", or "COSINE"
            }
        })
    """
    # Extract vector store type and config
    store_type, store_config = ConfigValidator.extract_provider_config(
        config,
        "Vector store",
        ["faiss"]
    )
    
    # Create vector store instance
    Logger.debug(f"Creating vector store: {store_type}", "[VectorStoreFactory]")
    
    if store_type == "faiss":
        return FAISSVectorStore(store_config)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
