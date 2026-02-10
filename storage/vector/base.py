from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """Abstract base class for vector store providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store provider with configuration.
        
        Args:
            config: Dictionary containing vector store configuration parameters
        """
        self._config = config
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return the current configuration."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection/store is valid."""
        pass
    
    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying client instance."""
        pass
    
    @abstractmethod
    def insert(self, vector_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        """
        Insert a vector with payload into the vector store.
        
        Args:
            vector_id: Unique identifier for the vector (typically memory_id)
            vector: Embedding vector (list of floats)
            payload: Metadata payload associated with the vector
            
        Returns:
            True if inserted successfully, False on error
        """
        pass
    
    @abstractmethod
    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing vector and/or payload.
        
        Args:
            vector_id: Unique identifier for the vector (typically memory_id)
            vector: Optional new embedding vector (list of floats)
            payload: Optional new metadata payload
            
        Returns:
            True if updated successfully, False on error
        """
        pass
    
    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the vector store.
        
        Args:
            vector_id: Unique identifier for the vector (typically memory_id)
            
        Returns:
            True if deleted successfully, False on error
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector (list of floats)
            top_k: Number of results to return
            filter: Optional filter criteria for payload fields
            
        Returns:
            List of dictionaries containing:
                - vector_id: The ID of the matched vector
                - score: Similarity score
                - payload: The payload associated with the vector
        """
        pass
