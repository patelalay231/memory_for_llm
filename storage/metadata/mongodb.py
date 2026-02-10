from typing import Any, Dict
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from storage.metadata.base import BaseStorage
from core.models.Memory import Memory
from logger import Logger


class MongoDBStorage(BaseStorage):
    """MongoDB storage implementation for metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MongoDB storage with configuration.
        
        Args:
            config: Dictionary containing MongoDB configuration:
                - uri: MongoDB connection URI (e.g., "mongodb://localhost:27017")
                - database: Database name (optional, can be specified in URI)
                - collection: Collection name (optional, stored for convenience)
                - Additional PyMongo client options can be passed in config
        """
        super().__init__(config)
        
        # Extract MongoDB-specific config
        self._uri = config.get("uri", "mongodb://localhost:27017")
        self._database_name = config.get("database", "memory_db")
        self._collection_name = config.get("collection", "memories")
        
        # Extract any additional PyMongo client options
        client_options = {k: v for k, v in config.items() 
                         if k not in ["uri", "database", "collection"]}
        
        # Initialize MongoDB client
        self._client = MongoClient(self._uri, **client_options)
        self._database = self._client[self._database_name]
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the current MongoDB configuration.
        
        Returns:
            Dictionary containing MongoDB configuration
        """
        return {
            "uri": self._uri,
            "database": self._database_name,
            "collection": self._collection_name,
        }
    
    def test_connection(self) -> bool:
        """
        Test MongoDB connection by pinging the server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Ping the server to test connection
            self._client.admin.command('ping')
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            return False
    
    def get_client(self) -> MongoClient:
        """
        Return the PyMongo client instance.
        
        Returns:
            MongoClient instance
        """
        return self._client
    
    def create_schema(self) -> bool:
        """Create collection and indexes for memories."""
        try:
            Logger.debug("Creating schema (collection and indexes)...", "[MongoDB]")
            collection = self.get_collection()
            
            # Create indexes for efficient queries
            Logger.debug("Creating indexes...", "[MongoDB]")
            collection.create_index("memory_id", unique=True)
            collection.create_index("user_id")
            collection.create_index("type")
            collection.create_index("timestamp")
            collection.create_index([("user_id", 1), ("type", 1)])
            
            Logger.debug("Schema created successfully", "[MongoDB]")
            return True
        except Exception as e:
            Logger.debug(f"Error creating schema: {e}", "[MongoDB]")
            return False
    
    def insert_memory_metadata(self, memory: Memory) -> bool:
        """Insert a memory document."""
        try:
            collection = self.get_collection()
            collection.insert_one(memory.model_dump())
            return True
        except Exception as e:
            Logger.debug(f"Error inserting memory: {e}", "[MongoDB]")
            return False
    
    def update_memory_metadata(self, memory: Memory) -> bool:
        """Update an existing memory document."""
        try:
            collection = self.get_collection()
            result = collection.update_one(
                {"memory_id": memory.memory_id},
                {"$set": memory.model_dump()}
            )
            if result.matched_count == 0:
                Logger.debug(f"Memory {memory.memory_id} not found for update", "[MongoDB]")
                return False
            Logger.debug(f"Updated memory {memory.memory_id}", "[MongoDB]")
            return True
        except Exception as e:
            Logger.debug(f"Error updating memory: {e}", "[MongoDB]")
            return False
    
    def delete_memory_metadata(self, memory_id: str) -> bool:
        """Delete a memory document by memory_id."""
        try:
            collection = self.get_collection()
            result = collection.delete_one({"memory_id": memory_id})
            if result.deleted_count == 0:
                Logger.debug(f"Memory {memory_id} not found for deletion", "[MongoDB]")
                return False
            Logger.debug(f"Deleted memory {memory_id}", "[MongoDB]")
            return True
        except Exception as e:
            Logger.debug(f"Error deleting memory: {e}", "[MongoDB]")
            return False
    
    def get_database(self):
        """
        Get the database instance.
        
        Returns:
            Database instance
        """
        return self._database
    
    def get_collection(self):
        """
        Get the collection instance.
        
        Returns:
            Collection instance
        """
        return self._database[self._collection_name]
    
    def get_memories_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """Retrieve memories by their IDs."""
        try:
            if not memory_ids:
                return []
            
            collection = self.get_collection()
            documents = collection.find({"memory_id": {"$in": memory_ids}})
            
            memories = []
            for doc in documents:
                # Remove MongoDB's _id field if present
                doc.pop("_id", None)
                try:
                    memory = Memory(**doc)
                    memories.append(memory)
                except Exception as e:
                    Logger.debug(f"Failed to parse memory document: {e}", "[MongoDB]")
                    continue
            
            Logger.debug(f"Retrieved {len(memories)} memories from MongoDB", "[MongoDB]")
            return memories
        except Exception as e:
            Logger.debug(f"Error retrieving memories: {e}", "[MongoDB]")
            return []

    def delete_all_for_user(self, user_id: str) -> int:
        """Delete all memories for a given user_id. Returns number of documents deleted."""
        try:
            collection = self.get_collection()
            result = collection.delete_many({"user_id": user_id})
            count = result.deleted_count
            Logger.debug(f"Deleted {count} memories for user_id={user_id}", "[MongoDB]")
            return count
        except Exception as e:
            Logger.debug(f"Error deleting memories for user: {e}", "[MongoDB]")
            return 0
