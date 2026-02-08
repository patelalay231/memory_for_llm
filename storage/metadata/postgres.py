from typing import Any, Dict
import psycopg2
from psycopg2 import OperationalError, Error
from psycopg2.pool import SimpleConnectionPool
from storage.metadata.base import BaseStorage
from core.models.Memory import Memory
from logger import Logger


class PostgresStorage(BaseStorage):
    """PostgreSQL storage implementation for metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL storage with configuration.
        
        Args:
            config: Dictionary containing PostgreSQL configuration:
                - host: Database host (default: "localhost")
                - port: Database port (default: 5432)
                - database: Database name (required)
                - user: Database user (default: "postgres")
                - password: Database password (default: "postgres")
                - use_pool: Whether to use connection pooling (default: False)
                - minconn: Minimum pool connections (default: 1)
                - maxconn: Maximum pool connections (default: 10)
                - Additional psycopg2 connection parameters can be passed
        """
        super().__init__(config)
        
        # Extract PostgreSQL-specific config
        self._host = config.get("host", "localhost")
        self._port = config.get("port", 5432)
        self._database = config.get("database", "memory_db")
        self._user = config.get("user", "postgres")
        self._password = config.get("password", "postgres")
        self._use_pool = config.get("use_pool", False)
        
        # Connection parameters
        self._connection_params = {
            "host": self._host,
            "port": self._port,
            "database": self._database,
            "user": self._user,
            "password": self._password,
        }
        
        # Add any additional psycopg2 connection parameters
        additional_params = {k: v for k, v in config.items() 
                           if k not in ["host", "port", "database", "user", 
                                       "password", "use_pool", "minconn", "maxconn"]}
        self._connection_params.update(additional_params)
        
        # Initialize connection or connection pool
        if self._use_pool:
            minconn = config.get("minconn", 1)
            maxconn = config.get("maxconn", 10)
            self._pool = SimpleConnectionPool(
                minconn, maxconn, **self._connection_params
            )
            self._connection = None
        else:
            self._pool = None
            self._connection = psycopg2.connect(**self._connection_params)
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the current PostgreSQL configuration.
        
        Returns:
            Dictionary containing PostgreSQL configuration
        """
        return {
            "host": self._host,
            "port": self._port,
            "database": self._database,
            "user": self._user,
            "password": "***",  # Don't expose password in config
            "use_pool": self._use_pool,
        }
    
    def test_connection(self) -> bool:
        """
        Test PostgreSQL connection by executing SELECT 1.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            if self._pool:
                # Get connection from pool
                conn = self._pool.getconn()
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return True
                finally:
                    self._pool.putconn(conn)
            else:
                # Use direct connection
                cursor = self._connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return True
        except (OperationalError, Error):
            return False
    
    def get_client(self):
        """
        Return the psycopg2 connection or connection pool.
        
        Returns:
            psycopg2 connection instance or connection pool
        """
        if self._pool:
            return self._pool
        return self._connection
    
    def create_schema(self) -> bool:
        """Create memories table with indexes."""
        try:
            Logger.debug("Creating schema (table and indexes)...", "[PostgreSQL]")
            if self._pool:
                conn = self._pool.getconn()
            else:
                conn = self._connection
            
            cursor = conn.cursor()
            
            # Create table (memory_id instead of id)
            Logger.debug("Creating memories table...", "[PostgreSQL]")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id VARCHAR(100) PRIMARY KEY,
                    source VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    type VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    conversation_id VARCHAR(100),
                    user_id VARCHAR(100)
                )
            """)
            
            # Create indexes
            Logger.debug("Creating indexes...", "[PostgreSQL]")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON memories(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_type 
                ON memories(type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON memories(timestamp)
            """)
            
            conn.commit()
            cursor.close()
            
            if self._pool:
                self._pool.putconn(conn)
            
            Logger.debug("Schema created successfully", "[PostgreSQL]")
            return True
        except Exception as e:
            Logger.error(f"Error creating schema: {e}", "[PostgreSQL]")
            return False
    
    def insert_memory(self, memory: Memory) -> bool:
        """Insert a memory record."""
        try:
            if self._pool:
                conn = self._pool.getconn()
            else:
                conn = self._connection
            
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories 
                (memory_id, source, content, type, timestamp, conversation_id, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                memory.memory_id, memory.source, memory.content, memory.type,
                memory.timestamp, memory.conversation_id, memory.user_id
            ))
            conn.commit()
            cursor.close()
            
            if self._pool:
                self._pool.putconn(conn)
            
            return True
        except Exception as e:
            Logger.error(f"Error inserting memory: {e}", "[PostgreSQL]")
            return False
    
    def close(self):
        """
        Close the connection or connection pool.
        """
        if self._pool:
            self._pool.closeall()
        elif self._connection:
            self._connection.close()
