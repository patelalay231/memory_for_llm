from core.models.Memory import Memory
from storage.metadata.base import BaseStorage
from storage.vector.base import BaseVectorStore
from core.embeddings import EmbeddingGenerator
from core.extraction.memory_extract import MemoryExtract
from core.llm.base import LLMProvider
from core.memory.memory_operations import MemoryOperationExecutor
from logger import Logger
from datetime import datetime
from uuid import uuid4
from typing import List, Literal, Optional

MemoryType = Literal["user", "agent", "both"]
import concurrent.futures


class MemoryStore:
    """Stores memories in metadata storage and vector store with intelligent operations."""
    
    def __init__(
        self,
        storage: BaseStorage,
        embedding_generator: EmbeddingGenerator,
        memory_extractor: MemoryExtract,
        vector_store: BaseVectorStore,
        llm_provider: LLMProvider,
        max_retries: int = 3,
    ):
        """
        Initialize MemoryStore.
        
        Args:
            storage: Metadata storage instance (MongoDB or PostgreSQL)
            embedding_generator: Embedding generator instance (required)
            memory_extractor: Memory extractor instance (required)
            vector_store: Vector store instance (required)
            llm_provider: LLM provider instance for operation determination (required)
            max_retries: Max retries for LLM-based operation determination (default: 3)
        """
        Logger.debug("Initializing memory store...", "[MemoryStore]")
        self.storage = storage
        self.embedding_generator = embedding_generator
        self.memory_extractor = memory_extractor
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        
        if not self.embedding_generator:
            raise ValueError("Embedding generator is required for MemoryStore")
        
        if not self.memory_extractor:
            raise ValueError("Memory extractor is required for MemoryStore")
        
        if not self.vector_store:
            raise ValueError("Vector store is required for MemoryStore")
        
        if not self.llm_provider:
            raise ValueError("LLM provider is required for MemoryStore")
        
        Logger.debug("All components configured", "[MemoryStore]")
        
        # Initialize operation executor
        self.operation_executor = MemoryOperationExecutor(self.llm_provider, max_retries=max_retries)
        
        # Ensure schema exists
        Logger.debug("Creating database schema...", "[MemoryStore]")
        if self.storage.create_schema():
            Logger.debug("Schema created/verified successfully", "[MemoryStore]")
        else:
            Logger.debug("Schema creation returned False", "[MemoryStore]")
    
    def create_memory(
        self,
        recent_messages: list[dict],
        user_message: str,
        assistant_message: str,
        user_id: Optional[str] = None,
    ) -> list[Memory]:
        """
        Extract memories, determine operations via LLM, and execute them.
        Optimized batch flow: extract -> batch embeddings -> parallel searches -> batch LLM -> execute operations.
        
        Args:
            recent_messages: List of recent conversation turns
            user_message: Current user message
            assistant_message: Current assistant response
            user_id: Optional user scope (e.g. for evaluation: speaker_a_0)
            
        Returns:
            List of stored Memory objects
            
        Raises:
            Exception: If extraction, embedding generation, or operation execution fails
        """
        Logger.debug("Starting memory creation process...", "[MemoryStore]")
        
        # Step 1: Extract candidate memories
        Logger.debug("Extracting memories from conversation...", "[MemoryStore]")
        try:
            candidate_memories = self.memory_extractor.extract_memory(
                recent_messages,
                user_message,
                assistant_message
            )
        except Exception as e:
            Logger.debug(f"Failed to extract memories: {str(e)}", "[MemoryStore]")
            raise Exception(f"Failed to extract memories: {str(e)}")
        
        if not candidate_memories:
            Logger.debug("No memories extracted, nothing to store", "[MemoryStore]")
            return []
        
        # Scope extracted memories to user_id when provided (e.g. evaluation)
        if user_id is not None:
            for mem in candidate_memories:
                mem.user_id = user_id
        
        Logger.debug(f"Extracted {len(candidate_memories)} candidate memory/memories", "[MemoryStore]")
        
        # Step 2: Batch generate embeddings for all candidates
        Logger.debug("Batch generating embeddings...", "[MemoryStore]")
        try:
            candidate_texts = [mem.content for mem in candidate_memories]
            embeddings = self.embedding_generator.generate_batch(candidate_texts)
            Logger.debug(f"Generated {len(embeddings)} embeddings", "[MemoryStore]")
        except Exception as e:
            Logger.debug(f"Failed to generate embeddings: {str(e)}", "[MemoryStore]")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
        
        if len(embeddings) != len(candidate_memories):
            raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(candidate_memories)}")
        
        # Store embeddings in memory objects
        for memory, embedding in zip(candidate_memories, embeddings):
            memory.embedding = embedding
        
        # Step 3: Parallel vector searches (top_k=5 for each candidate); filter by user_id when set
        search_filter = {"user_id": user_id} if user_id else None
        Logger.debug("Performing parallel vector searches...", "[MemoryStore]")
        search_results = self._parallel_vector_search(embeddings, top_k=5, filter=search_filter)
        
        # Step 4: Prepare batch payload for LLM
        Logger.debug("Preparing batch payload for operation determination...", "[MemoryStore]")
        candidates_data = []
        for idx, (candidate, similar_memories) in enumerate(zip(candidate_memories, search_results)):
            candidates_data.append({
                "candidate_id": f"temp_{idx}",
                "candidate_memory": {
                    "content": candidate.content,
                    "type": candidate.type
                },
                "existing_memories": [result["payload"] for result in similar_memories]
            })
        
        # Step 5: Determine operations via batch LLM call
        Logger.debug("Determining operations via LLM...", "[MemoryStore]")
        try:
            operations = self.operation_executor.determine_operations_batch(candidates_data)
        except Exception as e:
            Logger.debug(f"Failed to determine operations: {str(e)}", "[MemoryStore]")
            raise Exception(f"Failed to determine operations: {str(e)}")
        
        # Step 6: Execute operations
        Logger.debug(f"Executing {len(operations)} operations...", "[MemoryStore]")
        stored_memories = []
        
        # Map candidate_id back to candidate memory and embedding
        candidate_map = {f"temp_{idx}": (mem, emb) for idx, (mem, emb) in enumerate(zip(candidate_memories, embeddings))}
        
        for operation in operations:
            candidate_id = operation.get("candidate_id")
            if candidate_id not in candidate_map:
                Logger.debug(f"Candidate ID {candidate_id} not found in map", "[MemoryStore]")
                continue
            
            candidate_memory, embedding = candidate_map[candidate_id]
            
            # Execute operation
            success = self.operation_executor.execute_operation(
                operation,
                candidate_memory,
                embedding,
                self.storage,
                self.vector_store
            )
            
            if success:
                # For UPDATE operations, the target_memory_id is used, so we update the memory_id
                if operation.get("operation") == "UPDATE":
                    target_id = operation.get("target_memory_id")
                    if target_id:
                        candidate_memory.memory_id = target_id
                
                # Only add to stored_memories if operation was ADD or UPDATE
                if operation.get("operation") in ["ADD", "UPDATE"]:
                    stored_memories.append(candidate_memory)
        
        Logger.debug(f"Successfully processed {len(stored_memories)} memory/memories", "[MemoryStore]")
        return stored_memories
    
    def _parallel_vector_search(
        self,
        embeddings: List[List[float]],
        top_k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[List[dict]]:
        """
        Perform parallel vector searches for multiple embeddings.
        
        Args:
            embeddings: List of embedding vectors to search with
            top_k: Number of results to return per search
            filter: Optional filter (e.g. {"user_id": "alice_0"}) for scoped search
        
        Returns:
            List of search result lists, one per embedding
        """
        def search_single(embedding: List[float]) -> List[dict]:
            return self.vector_store.search(embedding, top_k=top_k, filter=filter)
        
        # Use ThreadPoolExecutor for parallel searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(embeddings), 10)) as executor:
            results = list(executor.map(search_single, embeddings))
        
        return results
