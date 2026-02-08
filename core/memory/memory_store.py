from core.models.Memory import Memory
from storage.metadata.base import BaseStorage
from logger import Logger
from datetime import datetime
from uuid import uuid4


class MemoryStore:
    """Stores memories in metadata storage with ID/timestamp generation."""
    
    def __init__(self, storage: BaseStorage):
        """
        Initialize MemoryStore.
        
        Args:
            storage: Metadata storage instance (MongoDB or PostgreSQL)
        """
        Logger.debug("Initializing memory store...", "[MemoryStore]")
        self.storage = storage
        # Ensure schema exists
        Logger.debug("Creating database schema...", "[MemoryStore]")
        if self.storage.create_schema():
            Logger.debug("Schema created/verified successfully", "[MemoryStore]")
        else:
            Logger.warning("Schema creation returned False", "[MemoryStore]")
    
    def store_memory(self, memory: Memory) -> Memory:
        """
        Store a single memory with generated ID and timestamp.
        
        Args:
            memory: Memory object (may have default or custom ID/timestamp)
            
        Returns:
            The stored Memory object
        """
        Logger.debug(f"Storing memory: {memory.type} - {memory.content[:50]}...", "[MemoryStore]")
        
        # Insert into storage
        if self.storage.insert_memory(memory):
            Logger.debug(f"Successfully stored memory (ID: {memory.memory_id})", "[MemoryStore]")
        else:
            Logger.error(f"Failed to store memory (ID: {memory.memory_id})", "[MemoryStore]")
            raise Exception(f"Failed to store memory: {memory.memory_id}")
        
        return memory
    
    def store_memories(self, memories: list[Memory]) -> list[Memory]:
        """
        Store multiple memories.
        
        Args:
            memories: List of Memory objects
            
        Returns:
            List of stored Memory objects
        """
        if not memories:
            Logger.debug("No memories to store", "[MemoryStore]")
            return []
        
        Logger.debug(f"Storing {len(memories)} memory/memories...", "[MemoryStore]")
        stored = []
        for idx, memory in enumerate(memories):
            Logger.debug(f"[{idx + 1}/{len(memories)}] Processing memory...", "[MemoryStore]")
            stored.append(self.store_memory(memory))
        
        Logger.debug(f"Successfully stored all {len(stored)} memory/memories", "[MemoryStore]")
        return stored
