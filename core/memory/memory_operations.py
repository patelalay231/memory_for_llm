from core.llm.base import LLMProvider
from core.llm.generation_config import GenerationConfig
from core.prompts import get_memory_operations_prompt
from core.models.Memory import Memory
from storage.metadata.base import BaseStorage
from storage.vector.base import BaseVectorStore
from logger import Logger
import json
from typing import List, Dict, Any


class MemoryOperationExecutor:
    """
    Executes memory operations (ADD, UPDATE, DELETE, NOOP) based on LLM decisions.
    Synchronizes changes across both metadata storage and vector store.
    """
    
    def __init__(self, llm_provider: LLMProvider, max_retries: int = 3):
        """
        Initialize the memory operation executor.
        
        Args:
            llm_provider: LLM provider instance for determining operations
            max_retries: Maximum number of retries for determine_operations_batch on LLM/parse failure (default: 3)
        """
        self.llm_provider = llm_provider
        self.max_retries = max(1, max_retries)
        Logger.debug(f"Initialized MemoryOperationExecutor (max_retries={self.max_retries})", "[MemoryOperationExecutor]")
    
    def determine_operations_batch(self, candidates_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine operations for multiple candidate memories in a single LLM call.
        
        Args:
            candidates_data: List of dictionaries, each containing:
                - candidate_id: Temporary ID (e.g., "temp_0")
                - candidate_memory: Dict with "content" and "type"
                - existing_memories: List of existing memory payloads from vector search
        
        Returns:
            List of operation dictionaries, each containing:
                - candidate_id: The candidate ID
                - operation: One of ["ADD", "UPDATE", "DELETE", "NOOP"]
                - target_memory_id: Memory ID for UPDATE/DELETE, None otherwise
                - confidence: Float between 0.0 and 1.0
        
        Raises:
            Exception: If LLM call fails or response is invalid
        """
        Logger.debug(f"Determining operations for {len(candidates_data)} candidates", "[MemoryOperationExecutor]")
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    Logger.debug(
                        f"Retry attempt {attempt + 1}/{self.max_retries} for operations determination",
                        "[MemoryOperationExecutor]",
                    )
                # Build prompt (handles both single and batch)
                prompt = get_memory_operations_prompt(candidates_data)
                # Call LLM with temperature=0 for deterministic operations
                Logger.debug("Sending batch operations prompt to LLM...", "[MemoryOperationExecutor]")
                generation_config = GenerationConfig(temperature=0.0)
                response = self.llm_provider.send_message(
                    prompt,
                    system_instruction=None,
                    generation_config=generation_config
                )
                # Parse JSON response
                cleaned_response = self._clean_json_output(response)
                data = json.loads(cleaned_response)
                # Validate response structure
                if "operations" not in data:
                    raise ValueError("Missing 'operations' field in LLM response")
                if not isinstance(data["operations"], list):
                    raise ValueError("'operations' field must be a list")
                if len(data["operations"]) != len(candidates_data):
                    Logger.debug(
                        f"Operations count mismatch: expected {len(candidates_data)}, got {len(data['operations'])}",
                        "[MemoryOperationExecutor]"
                    )
                # Validate each operation
                operations = []
                for idx, op in enumerate(data["operations"]):
                    if not isinstance(op, dict):
                        raise ValueError(f"Operation at index {idx} must be a dictionary")
                    required_fields = ["candidate_id", "operation", "target_memory_id", "confidence"]
                    for field in required_fields:
                        if field not in op:
                            raise ValueError(f"Operation at index {idx} missing required field: {field}")
                    if op["operation"] not in ["ADD", "UPDATE", "DELETE", "NOOP"]:
                        raise ValueError(f"Invalid operation: {op['operation']} at index {idx}")
                    if op["operation"] in ["UPDATE", "DELETE"] and op["target_memory_id"] is None:
                        Logger.debug(
                            f"Operation {op['operation']} at index {idx} has null target_memory_id",
                            "[MemoryOperationExecutor]"
                        )
                    operations.append(op)
                Logger.debug(f"Successfully determined {len(operations)} operations", "[MemoryOperationExecutor]")
                return operations
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                Logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}",
                    "[MemoryOperationExecutor]",
                )
                if attempt == self.max_retries - 1:
                    break
                continue
            except Exception as e:
                last_error = e
                Logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}",
                    "[MemoryOperationExecutor]",
                )
                if attempt == self.max_retries - 1:
                    break
                continue
        Logger.debug(
            f"Failed to determine operations after {self.max_retries} attempts",
            "[MemoryOperationExecutor]",
        )
        if isinstance(last_error, json.JSONDecodeError):
            raise Exception(f"Invalid JSON response from LLM after {self.max_retries} attempts: {last_error}")
        raise Exception(f"Failed to determine operations after {self.max_retries} attempts: {last_error}")
    
    def execute_operation(
        self,
        operation: Dict[str, Any],
        candidate_memory: Memory,
        embedding: List[float],
        metadata_store: BaseStorage,
        vector_store: BaseVectorStore
    ) -> bool:
        """
        Execute a single memory operation on both metadata and vector stores.
        
        Args:
            operation: Operation dict with "operation", "target_memory_id", etc.
            candidate_memory: The candidate memory object
            embedding: Embedding vector for the candidate memory
            metadata_store: Metadata storage instance
            vector_store: Vector store instance
        
        Returns:
            True if operation executed successfully, False otherwise
        """
        op_type = operation.get("operation")
        target_memory_id = operation.get("target_memory_id")
        confidence = operation.get("confidence", 0.0)
        
        Logger.debug(
            f"Executing operation: {op_type} for candidate {candidate_memory.memory_id} "
            f"(target: {target_memory_id}, confidence: {confidence})",
            "[MemoryOperationExecutor]"
        )
        
        try:
            if op_type == "ADD":
                return self._execute_add(candidate_memory, embedding, metadata_store, vector_store)
            elif op_type == "UPDATE":
                if not target_memory_id:
                    Logger.debug("UPDATE operation requires target_memory_id", "[MemoryOperationExecutor]")
                    return False
                return self._execute_update(
                    candidate_memory, embedding, target_memory_id, metadata_store, vector_store
                )
            elif op_type == "DELETE":
                if not target_memory_id:
                    Logger.debug("DELETE operation requires target_memory_id", "[MemoryOperationExecutor]")
                    return False
                return self._execute_delete(target_memory_id, metadata_store, vector_store)
            elif op_type == "NOOP":
                Logger.debug(f"Skipping NOOP for candidate {candidate_memory.memory_id}", "[MemoryOperationExecutor]")
                return True
            else:
                Logger.debug(f"Unknown operation type: {op_type}", "[MemoryOperationExecutor]")
                return False
        except Exception as e:
            Logger.debug(f"Error executing operation {op_type}: {e}", "[MemoryOperationExecutor]")
            return False
    
    def _execute_add(
        self,
        memory: Memory,
        embedding: List[float],
        metadata_store: BaseStorage,
        vector_store: BaseVectorStore
    ) -> bool:
        """Execute ADD operation: insert into both stores."""
        Logger.debug(f"Adding memory {memory.memory_id}", "[MemoryOperationExecutor]")
        
        # Prepare vector store payload
        payload = {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "type": memory.type,
            "source": memory.source,
            "timestamp": memory.timestamp.isoformat() if hasattr(memory.timestamp, "isoformat") else str(memory.timestamp),
            "user_id": memory.user_id,
        }
        
        # Insert into metadata store
        if not metadata_store.insert_memory_metadata(memory):
            Logger.debug(f"Failed to insert memory {memory.memory_id} into metadata store", "[MemoryOperationExecutor]")
            return False
        
        # Insert into vector store
        if not vector_store.insert(memory.memory_id, embedding, payload):
            Logger.debug(f"Failed to insert memory {memory.memory_id} into vector store", "[MemoryOperationExecutor]")
            # Try to rollback metadata store
            metadata_store.delete_memory_metadata(memory.memory_id)
            return False
        
        Logger.debug(f"Successfully added memory {memory.memory_id}", "[MemoryOperationExecutor]")
        return True
    
    def _execute_update(
        self,
        new_memory: Memory,
        new_embedding: List[float],
        target_memory_id: str,
        metadata_store: BaseStorage,
        vector_store: BaseVectorStore
    ) -> bool:
        """Execute UPDATE operation: update both stores."""
        Logger.debug(f"Updating memory {target_memory_id} with new memory {new_memory.memory_id}", "[MemoryOperationExecutor]")
        
        # Update memory_id to match target (for metadata store)
        new_memory.memory_id = target_memory_id
        
        # Prepare vector store payload (include user_id for filtered retrieval)
        payload = {
            "memory_id": target_memory_id,
            "content": new_memory.content,
            "type": new_memory.type,
            "source": new_memory.source,
            "timestamp": new_memory.timestamp.isoformat() if hasattr(new_memory.timestamp, "isoformat") else str(new_memory.timestamp),
            "user_id": new_memory.user_id,
        }
        
        # Update metadata store
        if not metadata_store.update_memory_metadata(new_memory):
            Logger.debug(f"Failed to update memory {target_memory_id} in metadata store", "[MemoryOperationExecutor]")
            return False
        
        # Update vector store (update vector and payload)
        if not vector_store.update(target_memory_id, new_embedding, payload):
            Logger.debug(f"Failed to update memory {target_memory_id} in vector store", "[MemoryOperationExecutor]")
            # Note: We can't easily rollback metadata update, but log the error
            return False
        
        Logger.debug(f"Successfully updated memory {target_memory_id}", "[MemoryOperationExecutor]")
        return True
    
    def _execute_delete(
        self,
        target_memory_id: str,
        metadata_store: BaseStorage,
        vector_store: BaseVectorStore
    ) -> bool:
        """Execute DELETE operation: delete from both stores."""
        Logger.debug(f"Deleting memory {target_memory_id}", "[MemoryOperationExecutor]")
        
        # Delete from metadata store
        metadata_success = metadata_store.delete_memory_metadata(target_memory_id)
        
        # Delete from vector store
        vector_success = vector_store.delete(target_memory_id)
        
        if not metadata_success:
            Logger.debug(f"Failed to delete memory {target_memory_id} from metadata store", "[MemoryOperationExecutor]")
        
        if not vector_success:
            Logger.debug(f"Failed to delete memory {target_memory_id} from vector store", "[MemoryOperationExecutor]")
        
        # Both must succeed
        if metadata_success and vector_success:
            Logger.debug(f"Successfully deleted memory {target_memory_id}", "[MemoryOperationExecutor]")
            return True
        else:
            Logger.debug(
                f"Partial deletion: metadata={metadata_success}, vector={vector_success}",
                "[MemoryOperationExecutor]"
            )
            return False
    
    def _clean_json_output(self, raw: str) -> str:
        """
        Remove markdown code blocks if present.
        
        Args:
            raw: Raw string output from LLM
            
        Returns:
            Cleaned string without markdown code blocks
        """
        if raw.strip().startswith("```"):
            lines = raw.strip().split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return raw
