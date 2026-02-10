from storage.metadata.base import BaseStorage
from storage.vector.base import BaseVectorStore
from core.embeddings import EmbeddingGenerator
from core.models.Memory import Memory
from logger import Logger
from typing import Optional


class RetrievalAPI:
    """API for memory retrieval operations using semantic search."""
    
    def __init__(self, storage: BaseStorage, vector_store: BaseVectorStore, embedding_generator: EmbeddingGenerator):
        """
        Initialize Retrieval API.
        
        Args:
            storage: Metadata storage instance (MongoDB or PostgreSQL)
            vector_store: Vector store instance for similarity search
            embedding_generator: Embedding generator for converting queries to embeddings
        """
        self.storage = storage
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
        Logger.debug("Initializing Retrieval API...", "[RetrievalAPI]")
        
        # Test connections
        if not self.storage.test_connection():
            Logger.debug("Failed to connect to metadata storage", "[RetrievalAPI]")
            raise ConnectionError("Failed to connect to metadata storage")
        
        if not self.vector_store.test_connection():
            Logger.debug("Failed to connect to vector store", "[RetrievalAPI]")
            raise ConnectionError("Failed to connect to vector store")
        
        Logger.debug("Retrieval API initialized successfully", "[RetrievalAPI]")
    
    def retrieve(self, query: str, top_k: int = 10, filter: Optional[dict] = None) -> list[Memory]:
        """
        Retrieve top_k memories based on semantic similarity to the query.
        
        Args:
            query: Search query string
            top_k: Number of top memories to retrieve (default: 10)
            filter: Optional filter criteria for payload fields (e.g., {"user_id": "user123"})
            
        Returns:
            List of Memory objects sorted by similarity score (highest first)
        """
        Logger.debug(f"Starting retrieval for query: '{query[:50]}...' (top_k={top_k})", "[RetrievalAPI]")
        
        
        # Step 1: Convert query to embedding
        Logger.debug("Generating embedding for query...", "[RetrievalAPI]")
        try:
            query_embedding = self.embedding_generator.generate(query)
            Logger.debug(f"Generated query embedding (dimensions: {len(query_embedding)})", "[RetrievalAPI]")
        except Exception as e:
            Logger.debug(f"Failed to generate query embedding: {e}", "[RetrievalAPI]")
            raise Exception(f"Failed to generate query embedding: {e}")
        
        # Step 2: Search vector store for similar memories
        Logger.debug(f"Searching vector store for top {top_k} similar memories...", "[RetrievalAPI]")
        try:
            search_results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                filter=filter
            )
            Logger.debug(f"Found {len(search_results)} results from vector store", "[RetrievalAPI]")
        except Exception as e:
            Logger.debug(f"Vector search failed: {e}", "[RetrievalAPI]")
            raise Exception(f"Vector search failed: {e}")
        
        if not search_results:
            Logger.debug("No similar memories found", "[RetrievalAPI]")
            return []
        
        # Step 3: Extract memory IDs from search results
        memory_ids = [result["vector_id"] for result in search_results]
        Logger.debug(f"Retrieving {len(memory_ids)} memories from metadata storage...", "[RetrievalAPI]")
        
        # Step 4: Retrieve full memory objects from metadata storage
        try:
            memories = self.storage.get_memories_by_ids(memory_ids)
            Logger.debug(f"Retrieved {len(memories)} memories from metadata storage", "[RetrievalAPI]")
        except Exception as e:
            Logger.debug(f"Failed to retrieve memories from metadata storage: {e}", "[RetrievalAPI]")
            raise Exception(f"Failed to retrieve memories: {e}")
        
        # Step 5: Sort memories by score (from search results) and preserve order
        # Create a mapping of memory_id to score
        score_map = {result["vector_id"]: result["score"] for result in search_results}
        
        # Sort memories by score (highest first)
        memories.sort(key=lambda m: score_map.get(m.memory_id, 0.0), reverse=True)
        
        # Step 6: Return top_k memories (in case we got more than requested)
        result_memories = memories[:top_k]
        
        Logger.debug(f"Successfully retrieved {len(result_memories)} memories", "[RetrievalAPI]")
        if Logger.is_debug():
            for idx, memory in enumerate(result_memories[:3]):  # Show top 3 in debug
                score = score_map.get(memory.memory_id, 0.0)
                Logger.debug(f"  [{idx + 1}] Score: {score:.4f} | Type: {memory.type} | Content: {memory.content[:50]}...", "[RetrievalAPI]")
        
        return result_memories
    
    def retrieve_by_user(self, user_id: str, top_k: int = 10) -> list[Memory]:
        """
        Retrieve top_k memories for a specific user.
        
        Note: This method queries metadata storage directly since we're filtering by user_id.
        For semantic search, use retrieve() with a query string.
        
        Args:
            user_id: User ID to filter by
            top_k: Number of memories to retrieve
            
        Returns:
            List of Memory objects for the user
        """
        Logger.debug(f"Retrieving memories for user: {user_id}", "[RetrievalAPI]")
        # For now, return empty list - direct metadata queries can be added later
        # This is a placeholder for future implementation
        Logger.debug("retrieve_by_user not fully implemented - use retrieve() with filter instead", "[RetrievalAPI]")
        return []
    
