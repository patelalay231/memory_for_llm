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
    def insert_memory(self, memory: "Memory") -> bool:
        """
        Insert a memory into the database.
        
        Args:
            memory: Memory object to store
            
        Returns:
            True if inserted successfully, False on error
        """
        pass
