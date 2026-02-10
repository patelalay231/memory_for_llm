from typing import Dict, Any, Union
from pydantic import BaseModel
from .base import BaseStorage
from .mongodb import MongoDBStorage
from .postgres import PostgresStorage
from core.utils import ConfigValidator
from logger import Logger


def create_storage(config: Union[BaseModel, Dict[str, Any]]) -> BaseStorage:
    """
    Factory function to create storage from config.
    
    Args:
        config: Configuration with storage type as key (e.g., {"mongodb": {...}} or {"pg": {...}})
        
    Returns:
        BaseStorage instance
        
    Raises:
        ValueError: If storage type is not supported or config is invalid
        
    Example:
        storage = create_storage({
            "mongodb": {
                "uri": "mongodb://localhost:27017",
                "database": "memory_db"
            }
        })
    """
    # Extract storage type and config
    storage_type, storage_config = ConfigValidator.extract_provider_config(
        config,
        "Storage",
        ["mongodb", "pg"]
    )
    
    # Create storage instance
    Logger.debug(f"Creating storage: {storage_type}", "[StorageFactory]")
    
    if storage_type == "mongodb":
        return MongoDBStorage(storage_config)
    elif storage_type == "pg":
        return PostgresStorage(storage_config)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
