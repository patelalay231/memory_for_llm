from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStorage(ABC):
    """Abstract base class for metadata storage providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the storage provider with configuration.
        
        Args:
            config: Dictionary containing storage configuration parameters
        """
        self._config = config
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return the current configuration."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is valid."""
        pass
    
    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying client instance."""
        pass
    
    @abstractmethod
    def create_schema(self) -> bool:
        """
        Create database schema (tables, collections, indexes).
        Should be idempotent (safe to call multiple times).
        
        Returns:
            True if schema created/already exists, False on error
        """
        pass
    
    @abstractmethod
    def insert_memory_metadata(self, memory: "Memory") -> bool:
        """
        Insert a memory into the database.
        
        Args:
            memory: Memory object to store
            
        Returns:
            True if inserted successfully, False on error
        """
        pass
    
    @abstractmethod
    def update_memory_metadata(self, memory: "Memory") -> bool:
        """
        Update an existing memory in the database.
        
        Args:
            memory: Memory object with updated data (must have memory_id)
            
        Returns:
            True if updated successfully, False on error
        """
        pass
    
    @abstractmethod
    def delete_memory_metadata(self, memory_id: str) -> bool:
        """
        Delete a memory from the database.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            True if deleted successfully, False on error
        """
        pass
    
    @abstractmethod
    def get_memories_by_ids(self, memory_ids: list[str]) -> list["Memory"]:
        """
        Retrieve memories by their IDs.
        
        Args:
            memory_ids: List of memory IDs to retrieve
            
        Returns:
            List of Memory objects (may be empty if IDs not found)
        """
        pass

    def delete_all_for_user(self, user_id: str) -> int:
        """
        Delete all memories for a given user_id. Used for evaluation reset.
        
        Args:
            user_id: User ID to clear
            
        Returns:
            Number of memories deleted
        """
        raise NotImplementedError("delete_all_for_user must be implemented by the storage backend")