from core.extraction.memory_extract import MemoryExtract
from core.memory.memory_store import MemoryStore
from storage.metadata.mongodb import MongoDBStorage
from storage.metadata.postgres import PostgresStorage
from core.models.Memory import Memory
from core.providers.base import LLMProvider
from logger import Logger


class MemoryAPI:
    """API for memory extraction and storage operations."""
    
    def __init__(self, config: dict):
        """
        Initialize Memory API.
        
        config = {
            "llm_provider": LLMProvider instance,
            "storage": {
                "type": "mongodb" or "postgres",
                "uri": "..." or host/port/etc,
                "database": "...",
                ...
            },
            "max_retries": 3,
            "debug": False  # Optional: Enable debug mode for verbose logging
        }
        """
        # Set debug mode if provided
        if "debug" in config:
            Logger.set_debug(config["debug"])
        
        Logger.debug("Initializing Memory API...", "[MemoryAPI]")
        
        # Initialize extractor
        Logger.debug("Setting up memory extractor...", "[MemoryAPI]")
        self.extractor = MemoryExtract(
            provider=config["llm_provider"],
            max_retries=config.get("max_retries", 3)
        )
        Logger.debug(f"Memory extractor initialized (max retries: {self.extractor.max_retries})", "[MemoryAPI]")
        
        # Initialize storage
        storage_config = config["storage"].copy()
        storage_type = storage_config.pop("type")
        
        Logger.debug(f"Initializing {storage_type} storage...", "[MemoryAPI]")
        if storage_type == "mongodb":
            storage = MongoDBStorage(storage_config)
        elif storage_type == "postgres":
            storage = PostgresStorage(storage_config)
        else:
            Logger.error(f"Unknown storage type: {storage_type}", "[MemoryAPI]")
            raise ValueError(f"Unknown storage type: {storage_type}")
        
        Logger.debug(f"Testing {storage_type} connection...", "[MemoryAPI]")
        if not storage.test_connection():
            Logger.error(f"Failed to connect to {storage_type}", "[MemoryAPI]")
            raise ConnectionError("Failed to connect to storage backend")
        Logger.debug(f"Successfully connected to {storage_type}", "[MemoryAPI]")
        
        # Initialize memory store
        self.memory_store = MemoryStore(storage)
        Logger.debug("Memory API initialized successfully", "[MemoryAPI]")
    
    def add_memory(self, recent_messages: list[dict],
                   user_message: str,
                   assistant_message: str) -> list[Memory]:
        """
        Extract and store memories from conversation.
        
        Args:
            recent_messages: List of recent conversation turns
            user_message: Current user message
            assistant_message: Current assistant response
            
        Returns:
            List of Memory objects that were stored
        """
        Logger.debug("Starting memory addition process...", "[MemoryAPI]")
        
        # Extract memories with validation/retry
        memories = self.extractor.extract_memory(
            recent_messages,
            user_message,
            assistant_message
        )
        
        if not memories:
            Logger.debug("No memories to store", "[MemoryAPI]")
            return []
        
        # Store memories (generates IDs/timestamps if needed)
        stored_memories = self.memory_store.store_memories(memories)
        
        Logger.debug(f"Memory addition process completed successfully ({len(stored_memories)} memory/memories stored)", "[MemoryAPI]")
        return stored_memories
