from core.extraction.memory_extract import MemoryExtract
from core.memory.memory_store import MemoryStore
from core.models.Memory import Memory
from core.llm.factory import create_llm_provider
from storage.metadata.factory import create_storage
from storage.vector.factory import create_vector_store
from core.embeddings.factory import create_embedding_generator
from core.api.config import MemoryAPIConfig
from logger import Logger
from typing import Optional, Union


class MemoryAPI:
    """API for memory extraction and storage operations."""
    
    def __init__(self, config: Union[MemoryAPIConfig, dict]):
        """
        Initialize Memory API.
        
        Args:
            config: MemoryAPIConfig instance or dict with configuration.
                    Use MemoryAPIConfig with typed configs for type checking.
        
        Example with typed configs:
            from core.api import MemoryAPIConfig, StorageConfig, MongoDBConfig, EmbeddingConfig, OpenAIEmbeddingConfig, LLMProviderConfig, GeminiConfig
            
            config = MemoryAPIConfig(
                llm=LLMProviderConfig(
                    gemini=GeminiConfig(
                        api_key="your-gemini-api-key",
                        model="gemini-pro"
                    )
                ),
                storage=StorageConfig(
                    mongodb=MongoDBConfig(
                        uri="mongodb://localhost:27017",
                        database="memory_db"
                    )
                ),
                embedding=EmbeddingConfig(
                    openai=OpenAIEmbeddingConfig(
                        api_key="sk-...",
                        model="text-embedding-3-small"  # Optional
                    )
                )
            )
        
        Example with dict:
            config = {
                "llm": {
                    "gemini": {
                        "api_key": "your-gemini-api-key",
                        "model": "gemini-pro"
                    }
                },
                "storage": {
                    "mongodb": {
                        "uri": "mongodb://localhost:27017",
                        "database": "memory_db"
                    }
                },
                "embedding": {
                    "openai": {
                        "api_key": "sk-...",
                        "model": "text-embedding-3-small"  # Optional
                    }
                },
                "vector": {
                    "faiss": {
                        "dimension": 768,
                        "index_path": "./faiss_index",
                        "index_type": "L2"
                    }
                }
            }
        """
        # Convert dict to MemoryAPIConfig if needed
        if isinstance(config, dict):
            config = MemoryAPIConfig(**config)
        
        # Set debug mode
        Logger.set_debug(config.debug)
        
        Logger.debug("Initializing Memory API...", "[MemoryAPI]")
        
        # Initialize LLM provider using factory
        llm_provider = create_llm_provider(config.llm)
        
        # Initialize extractor (max_retries is hardcoded to 1)
        Logger.debug("Setting up memory extractor...", "[MemoryAPI]")
        self.extractor = MemoryExtract(
            provider=llm_provider,
            max_retries=3
        )
        Logger.debug(f"Memory extractor initialized (max retries: {self.extractor.max_retries})", "[MemoryAPI]")
        
        # Initialize storage using factory
        storage = create_storage(config.storage)
        
        Logger.debug(f"Testing storage connection...", "[MemoryAPI]")
        if not storage.test_connection():
            Logger.debug(f"Failed to connect to storage", "[MemoryAPI]")
            raise ConnectionError("Failed to connect to storage backend")
        Logger.debug(f"Successfully connected to storage", "[MemoryAPI]")
        
        # Initialize embedding generator using factory
        embedding_generator = create_embedding_generator(config.embedding)
        
        # Initialize vector store using factory
        Logger.debug("Initializing vector store...", "[MemoryAPI]")
        vector_store = create_vector_store(config.vector)
        
        Logger.debug("Testing vector store connection...", "[MemoryAPI]")
        if not vector_store.test_connection():
            Logger.debug("Failed to connect to vector store", "[MemoryAPI]")
            raise ConnectionError("Failed to connect to vector store backend")
        Logger.debug("Successfully connected to vector store", "[MemoryAPI]")
        
        # Initialize memory store with all required components
        self.memory_store = MemoryStore(
            storage=storage,
            embedding_generator=embedding_generator,
            memory_extractor=self.extractor,
            vector_store=vector_store,
            llm_provider=llm_provider
        )
        Logger.debug("Memory API initialized successfully", "[MemoryAPI]")
    
    def add_memory(
        self,
        recent_messages: list[dict],
        user_message: str,
        assistant_message: str,
        user_id: Optional[str] = None,
    ) -> list[Memory]:
        """
        Extract and store memories from conversation.
        
        Args:
            recent_messages: List of recent conversation turns
            user_message: Current user message
            assistant_message: Current assistant response
            user_id: Optional user scope (e.g. for evaluation: speaker_a_0)
            
        Returns:
            List of Memory objects that were stored
        """
        Logger.debug("Starting memory addition process...", "[MemoryAPI]")
        
        # Create memories (extraction, embedding generation, and storage all happen inside)
        stored_memories = self.memory_store.create_memory(
            recent_messages,
            user_message,
            assistant_message,
            user_id=user_id,
        )
        
        Logger.debug(f"Memory addition process completed successfully ({len(stored_memories)} memory/memories stored)", "[MemoryAPI]")
        return stored_memories

    def delete_all_for_user(self, user_id: str) -> int:
        """
        Delete all memories for a given user (metadata + vector store). Used for evaluation reset.
        
        Args:
            user_id: User ID to clear
            
        Returns:
            Number of memories deleted (from metadata store)
        """
        meta_count = self.memory_store.storage.delete_all_for_user(user_id)
        if hasattr(self.memory_store.vector_store, "delete_all_for_user"):
            self.memory_store.vector_store.delete_all_for_user(user_id)
        return meta_count
