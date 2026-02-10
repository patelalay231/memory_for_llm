from typing import Any, Dict, List, Optional
import faiss
import numpy as np
import json
import os
from pathlib import Path
from storage.vector.base import BaseVectorStore
from logger import Logger


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS vector store with configuration.
        
        Args:
            config: Dictionary containing FAISS configuration:
                - dimension: Dimension of the vectors (required)
                - index_path: Optional path to save/load FAISS index (default: "./faiss_index")
                - index_type: Optional index type (default: "L2" for L2 distance, "IP" for inner product, or "COSINE" for cosine similarity)
        """
        super().__init__(config)
        
        # Extract FAISS-specific config
        self._dimension = config.get("dimension")
        if not self._dimension:
            raise ValueError("FAISS config must include 'dimension' parameter")
        
        self._index_path = config.get("index_path", "./faiss_index")
        index_type_raw = config.get("index_type", "L2").upper()
        
        # Handle COSINE: use IP internally but normalize vectors
        if index_type_raw == "COSINE":
            self._index_type = "COSINE"
            self._use_cosine = True
        else:
            self._index_type = index_type_raw
            self._use_cosine = False
        
        # Create directory for index if it doesn't exist
        index_dir = Path(self._index_path).parent
        if index_dir and not index_dir.exists():
            index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index (COSINE uses IP internally)
        self._index = self._create_index()
        
        # Store payloads: vector_id -> payload dict
        self._payloads: Dict[str, Dict[str, Any]] = {}
        
        # Store vector_id to index mapping
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._next_index = 0
        
        # Load existing index and payloads if they exist
        self._load_index()
        
        Logger.debug(f"Initialized FAISS vector store (dimension: {self._dimension}, type: {self._index_type})", "[FAISSVectorStore]")
    
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        if self._index_type == "L2":
            # L2 (Euclidean) distance index
            index = faiss.IndexFlatL2(self._dimension)
        elif self._index_type == "IP":
            # Inner product index
            index = faiss.IndexFlatIP(self._dimension)
        elif self._index_type == "COSINE":
            # Cosine similarity: use IP index with normalized vectors
            index = faiss.IndexFlatIP(self._dimension)
        else:
            Logger.debug(f"Unknown index type: {self._index_type}, defaulting to L2", "[FAISSVectorStore]")
            index = faiss.IndexFlatL2(self._dimension)
        
        return index
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length (for cosine similarity).
        
        Args:
            vector: Numpy array of shape (1, dimension) or (dimension,)
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Return zero vector as-is
        return vector / norm
    
    def _load_index(self):
        """Load existing FAISS index and payloads from disk if they exist."""
        index_file = Path(self._index_path)
        payloads_file = Path(f"{self._index_path}.payloads")
        
        if index_file.exists() and payloads_file.exists():
            try:
                Logger.debug(f"Loading FAISS index from {index_file}", "[FAISSVectorStore]")
                self._index = faiss.read_index(str(index_file))
                
                Logger.debug(f"Loading payloads from {payloads_file}", "[FAISSVectorStore]")
                with open(payloads_file, 'r') as f:
                    data = json.load(f)
                    self._payloads = data.get("payloads", {})
                    self._id_to_index = {k: int(v) for k, v in data.get("id_to_index", {}).items()}
                    self._index_to_id = {int(k): v for k, v in data.get("index_to_id", {}).items()}
                    self._next_index = data.get("next_index", len(self._id_to_index))
                
                Logger.debug(f"Loaded {len(self._payloads)} vectors from disk", "[FAISSVectorStore]")
            except Exception as e:
                Logger.debug(f"Failed to load FAISS index: {e}", "[FAISSVectorStore]")
                # Continue with empty index
    
    def _save_index(self):
        """Save FAISS index and payloads to disk."""
        try:
            index_file = Path(self._index_path)
            payloads_file = Path(f"{self._index_path}.payloads")
            
            Logger.debug(f"Saving FAISS index to {index_file}", "[FAISSVectorStore]")
            faiss.write_index(self._index, str(index_file))
            
            Logger.debug(f"Saving payloads to {payloads_file}", "[FAISSVectorStore]")
            with open(payloads_file, 'w') as f:
                json.dump({
                    "payloads": self._payloads,
                    "id_to_index": self._id_to_index,
                    "index_to_id": self._index_to_id,
                    "next_index": self._next_index
                }, f)
            
            Logger.debug("Successfully saved FAISS index and payloads", "[FAISSVectorStore]")
        except Exception as e:
            Logger.debug(f"Failed to save FAISS index: {e}", "[FAISSVectorStore]")
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the current FAISS configuration.
        
        Returns:
            Dictionary containing FAISS configuration
        """
        return {
            "dimension": self._dimension,
            "index_path": self._index_path,
            "index_type": self._index_type,
        }
    
    def test_connection(self) -> bool:
        """
        Test if FAISS index is valid.
        
        Returns:
            True if index is valid, False otherwise
        """
        try:
            return self._index.is_trained and self._index.d == self._dimension
        except Exception:
            return False
    
    def get_client(self) -> faiss.Index:
        """
        Return the FAISS index instance.
        
        Returns:
            FAISS index instance
        """
        return self._index
    
    def insert(self, vector_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        """Insert a vector with payload into FAISS."""
        try:
            # Check if vector_id already exists
            if vector_id in self._id_to_index:
                Logger.debug(f"Vector ID {vector_id} already exists, use update() instead", "[FAISSVectorStore]")
                return False
            
            # Validate vector dimension
            if len(vector) != self._dimension:
                Logger.debug(f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}", "[FAISSVectorStore]")
                return False
            
            # Convert to numpy array and reshape for FAISS
            vector_array = np.array([vector], dtype=np.float32)
            
            # Normalize vector if using cosine similarity
            if self._use_cosine:
                vector_array = self._normalize_vector(vector_array)
            
            # Add to FAISS index
            self._index.add(vector_array)
            
            # Store mapping and payload
            index_pos = self._next_index
            self._id_to_index[vector_id] = index_pos
            self._index_to_id[index_pos] = vector_id
            self._payloads[vector_id] = payload
            self._next_index += 1
            
            # Save to disk
            self._save_index()
            
            Logger.debug(f"Inserted vector {vector_id} into FAISS", "[FAISSVectorStore]")
            return True
            
        except Exception as e:
            Logger.debug(f"Error inserting vector: {e}", "[FAISSVectorStore]")
            return False
    
    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing vector and/or payload."""
        try:
            if vector_id not in self._id_to_index:
                Logger.debug(f"Vector ID {vector_id} not found", "[FAISSVectorStore]")
                return False
            
            # Update vector if provided
            if vector is not None:
                # Validate vector dimension
                if len(vector) != self._dimension:
                    Logger.debug(f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}", "[FAISSVectorStore]")
                    return False
                
                # FAISS doesn't support direct updates, so we need to delete and reinsert
                # Get the index position
                index_pos = self._id_to_index[vector_id]
                
                # Delete old vector (we'll mark it as deleted and reinsert)
                # Note: FAISS doesn't support deletion directly, so we'll need to rebuild
                # For now, we'll use a simpler approach: delete and reinsert
                old_payload = self._payloads.get(vector_id, {})
                
                # Delete the old entry
                self.delete(vector_id)
                
                # Reinsert with new vector
                vector_array = np.array([vector], dtype=np.float32)
                
                # Normalize vector if using cosine similarity
                if self._use_cosine:
                    vector_array = self._normalize_vector(vector_array)
                
                self._index.add(vector_array)
                
                # Update mappings
                new_index_pos = self._next_index
                self._id_to_index[vector_id] = new_index_pos
                self._index_to_id[new_index_pos] = vector_id
                self._next_index += 1
                
                # Update payload (merge with old if new payload not provided)
                if payload is not None:
                    self._payloads[vector_id] = payload
                else:
                    self._payloads[vector_id] = old_payload
            
            # Update payload if provided
            if payload is not None:
                if vector_id in self._payloads:
                    # Merge with existing payload
                    self._payloads[vector_id].update(payload)
                else:
                    self._payloads[vector_id] = payload
            
            # Save to disk
            self._save_index()
            
            Logger.debug(f"Updated vector {vector_id}", "[FAISSVectorStore]")
            return True
            
        except Exception as e:
            Logger.debug(f"Error updating vector: {e}", "[FAISSVectorStore]")
            return False
    
    def delete(self, vector_id: str) -> bool:
        """Delete a vector from FAISS."""
        try:
            if vector_id not in self._id_to_index:
                Logger.debug(f"Vector ID {vector_id} not found", "[FAISSVectorStore]")
                return False
            
            # FAISS doesn't support direct deletion, so we mark it as deleted
            # by removing from our mappings and payloads
            index_pos = self._id_to_index[vector_id]
            
            # Remove from mappings
            del self._id_to_index[vector_id]
            del self._index_to_id[index_pos]
            
            # Remove payload
            if vector_id in self._payloads:
                del self._payloads[vector_id]
            
            # Note: The vector remains in FAISS index but won't be searchable
            # For a production system, you might want to rebuild the index periodically
            
            # Save to disk
            self._save_index()
            
            Logger.debug(f"Deleted vector {vector_id} from FAISS", "[FAISSVectorStore]")
            return True
            
        except Exception as e:
            Logger.debug(f"Error deleting vector: {e}", "[FAISSVectorStore]")
            return False

    def delete_all_for_user(self, user_id: str) -> int:
        """Delete all vectors whose payload has user_id matching. Returns number deleted."""
        to_delete = [
            vector_id
            for vector_id, payload in self._payloads.items()
            if payload.get("user_id") == user_id
        ]
        for vector_id in to_delete:
            self.delete(vector_id)
        Logger.debug(f"Deleted {len(to_delete)} vectors for user_id={user_id}", "[FAISSVectorStore]")
        return len(to_delete)

    def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            # Validate query vector dimension
            if len(query_vector) != self._dimension:
                Logger.debug(f"Query vector dimension mismatch: expected {self._dimension}, got {len(query_vector)}", "[FAISSVectorStore]")
                return []
            
            # Convert to numpy array and reshape for FAISS
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Normalize query vector if using cosine similarity
            if self._use_cosine:
                query_array = self._normalize_vector(query_array)
            
            # Search in FAISS
            k = min(top_k, self._index.ntotal) if self._index.ntotal > 0 else 0
            if k == 0:
                return []
            
            distances, indices = self._index.search(query_array, k)
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip if index is not in our mapping (deleted vectors)
                if idx not in self._index_to_id:
                    continue
                
                vector_id = self._index_to_id[idx]
                payload = self._payloads.get(vector_id, {})
                
                # Apply filter if provided (exclude 'type' field from filtering)
                if filter:
                    # Remove 'type' from filter if present (not filterable for now)
                    filter_without_type = {k: v for k, v in filter.items() if k != "type"}
                    if filter_without_type and not self._matches_filter(payload, filter_without_type):
                        continue
                
                # Calculate score
                if self._index_type == "L2":
                    # L2: lower distance = more similar, convert to similarity score
                    score = 1.0 / (1.0 + distance)
                elif self._index_type == "COSINE":
                    # Cosine: IP on normalized vectors gives cosine similarity (-1 to 1, typically 0 to 1)
                    # Clamp to [0, 1] range for consistency
                    score = max(0.0, float(distance))
                else:  # IP
                    # IP: higher is better, already a similarity score
                    score = float(distance)
                
                results.append({
                    "vector_id": vector_id,
                    "score": score,
                    "payload": payload
                })
            
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            
            Logger.debug(f"Found {len(results)} results for search query", "[FAISSVectorStore]")
            return results[:top_k]
            
        except Exception as e:
            Logger.debug(f"Error searching vectors: {e}", "[FAISSVectorStore]")
            return []
    
    def _matches_filter(self, payload: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Check if payload matches filter criteria.
        
        Args:
            payload: Payload to check
            filter: Filter criteria
            
        Returns:
            True if payload matches filter, False otherwise
        """
        for key, value in filter.items():
            if key not in payload:
                return False
            if payload[key] != value:
                return False
        return True
