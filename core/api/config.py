from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any, Literal, Union


class OpenAIEmbeddingConfig(BaseModel):
    """Configuration for OpenAI embedding provider."""
    
    api_key: str = Field(
        ...,
        description="OpenAI API key"
    )
    model: Optional[str] = Field(
        "text-embedding-3-small",
        description="Optional model name. Defaults to 'text-embedding-3-small'"
    )


class GeminiEmbeddingConfig(BaseModel):
    """Configuration for Gemini embedding provider."""
    
    api_key: str = Field(
        ...,
        description="Gemini API key"
    )
    model: Optional[str] = Field(
        "gemini-embedding-001",
        description="Optional model name. Defaults to 'gemini-embedding-001'"
    )
    task_type: Optional[str] = Field(
        None,
        description="Optional task type for optimization (e.g., 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY', 'CLASSIFICATION')"
    )
    output_dimensionality: Optional[int] = Field(
        None,
        ge=128,
        le=3072,
        description="Optional output dimension size (default 3072, recommended: 768, 1536, or 3072)"
    )


class HuggingFaceEmbeddingConfig(BaseModel):
    """Configuration for Hugging Face embedding provider (Inference API feature extraction)."""
    
    api_key: str = Field(
        ...,
        description="Hugging Face API token"
    )
    model: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Model id (e.g. 'sentence-transformers/all-MiniLM-L6-v2', 'Qwen/Qwen3-VL-Embedding-2B')"
    )
    provider: Optional[str] = Field(
        None,
        description="Inference provider (e.g. 'hf-inference', 'featherless-ai'). Defaults to 'auto' if not set."
    )
    normalize: Optional[bool] = Field(
        None,
        description="If True, normalize embeddings (useful for cosine similarity)"
    )
    output_dimensionality: Optional[int] = Field(
        None,
        ge=32,
        le=4096,
        description="Output dimension (32-4096). Supported by Qwen3-Embedding (MRL). Use same value for vector store dimension."
    )


class EmbeddingConfig(BaseModel):
    """Embedding configuration wrapper that accepts provider name as key."""
    
    openai: Optional[OpenAIEmbeddingConfig] = Field(
        None,
        description="OpenAI embedding configuration (use this key for OpenAI provider)"
    )
    gemini: Optional[GeminiEmbeddingConfig] = Field(
        None,
        description="Gemini embedding configuration (use this key for Gemini provider)"
    )
    huggingface: Optional[HuggingFaceEmbeddingConfig] = Field(
        None,
        description="Hugging Face embedding configuration (Inference API, e.g. Qwen3-VL-Embedding-2B)"
    )
    
    @model_validator(mode='after')
    def validate_embedding_provider(self):
        """Validate that exactly one embedding provider is provided."""
        providers = [key for key in ["openai", "gemini", "huggingface"] if getattr(self, key) is not None]
        if len(providers) != 1:
            raise ValueError("Embedding config must have exactly one provider key (e.g., 'openai', 'gemini', or 'huggingface')")
        return self


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB storage."""
    
    uri: str = Field(
        "mongodb://localhost:27017",
        description="MongoDB connection URI (e.g., 'mongodb://localhost:27017')"
    )
    database: Optional[str] = Field(
        "memory_db",
        description="Database name (optional, can be specified in URI)"
    )
    collection: Optional[str] = Field(
        "memories",
        description="Collection name (optional, defaults to 'memories')"
    )


class PostgresConfig(BaseModel):
    """Configuration for PostgreSQL storage."""
    
    host: str = Field(
        "localhost",
        description="Database host"
    )
    port: int = Field(
        5432,
        ge=1,
        le=65535,
        description="Database port"
    )
    database: str = Field(
        "memory_db",
        description="Database name"
    )
    user: str = Field(
        "postgres",
        description="Database user"
    )
    password: str = Field(
        "postgres",
        description="Database password"
    )
    use_pool: bool = Field(
        False,
        description="Whether to use connection pooling"
    )
    minconn: Optional[int] = Field(
        1,
        ge=1,
        description="Minimum pool connections (only used if use_pool=True)"
    )
    maxconn: Optional[int] = Field(
        10,
        ge=1,
        description="Maximum pool connections (only used if use_pool=True)"
    )


class StorageConfig(BaseModel):
    """Storage configuration wrapper that accepts either mongodb or pg key."""
    
    mongodb: Optional[MongoDBConfig] = Field(
        None,
        description="MongoDB configuration (use this OR pg, not both)"
    )
    pg: Optional[PostgresConfig] = Field(
        None,
        description="PostgreSQL configuration (use this OR mongodb, not both)"
    )
    
    @model_validator(mode='after')
    def validate_storage_type(self):
        """Validate that exactly one storage type is provided."""
        if (self.mongodb is None and self.pg is None) or (self.mongodb is not None and self.pg is not None):
            raise ValueError("Storage config must have exactly one of 'mongodb' or 'pg' keys")
        return self


class GeminiConfig(BaseModel):
    """Configuration for Gemini LLM provider."""
    
    api_key: str = Field(
        ...,
        description="Gemini API key"
    )
    model: str = Field(
        ...,
        description="Gemini model name"
    )


class HuggingFaceConfig(BaseModel):
    """Configuration for Hugging Face LLM provider (Inference API)."""
    
    api_key: str = Field(
        ...,
        description="Hugging Face API token"
    )
    model: str = Field(
        ...,
        description="Model id on the Hub (e.g. meta-llama/Meta-Llama-3-8B-Instruct)"
    )
    provider: Optional[str] = Field(
        None,
        description="Inference provider (e.g. 'featherless-ai', 'hf-inference'). Defaults to 'auto' if not set."
    )


class LLMProviderConfig(BaseModel):
    """LLM provider configuration wrapper that accepts provider name as key."""
    
    gemini: Optional[GeminiConfig] = Field(
        None,
        description="Gemini LLM provider configuration (use this key for Gemini provider)"
    )
    huggingface: Optional[HuggingFaceConfig] = Field(
        None,
        description="Hugging Face LLM provider (Inference API, e.g. Meta Llama 3)"
    )
    
    @model_validator(mode='after')
    def validate_llm_provider(self):
        """Validate that exactly one LLM provider is provided."""
        providers = [key for key in ["gemini", "huggingface"] if getattr(self, key) is not None]
        if len(providers) != 1:
            raise ValueError("LLM provider config must have exactly one provider key (e.g., 'gemini' or 'huggingface')")
        return self


class FAISSConfig(BaseModel):
    """Configuration for FAISS vector store."""
    
    dimension: int = Field(
        ...,
        ge=1,
        description="Dimension of the embedding vectors (required)"
    )
    index_path: str = Field(
        "./faiss_index",
        description="Path to save/load FAISS index file"
    )
    index_type: str = Field(
        "L2",
        description="Index type: 'L2' (Euclidean distance), 'IP' (Inner Product), or 'COSINE' (Cosine similarity)"
    )


class VectorStoreConfig(BaseModel):
    """Vector store configuration wrapper that accepts vector store type as key."""
    
    faiss: Optional[FAISSConfig] = Field(
        None,
        description="FAISS vector store configuration (use this key for FAISS provider)"
    )
    
    @model_validator(mode='after')
    def validate_vector_store_type(self):
        """Validate that exactly one vector store type is provided."""
        providers = [key for key in ["faiss"] if getattr(self, key) is not None]
        if len(providers) != 1:
            raise ValueError("Vector store config must have exactly one provider key (e.g., 'faiss')")
        return self


class MemoryAPIConfig(BaseSettings):
    """
    Configuration for MemoryAPI initialization.
    
    Inherits from BaseSettings to support environment variable loading.
    Environment variables can be prefixed with MEMORY_API_ to override config values.
    
    Example .env file:
        MEMORY_API_DEBUG=true
        OPENAI_API_KEY=sk-...
        GEMINI_API_KEY=...
    """
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_API_",
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    llm: Union[LLMProviderConfig, Dict[str, Any]] = Field(
        ...,
        description="LLM provider configuration. Use LLMProviderConfig({'gemini': GeminiConfig(...)}) for type safety, or dict for flexibility."
    )
    storage: Union[StorageConfig, Dict[str, Any]] = Field(
        ...,
        description="Storage configuration. Use StorageConfig({'mongodb': MongoDBConfig(...)}) or StorageConfig({'pg': PostgresConfig(...)}) for type safety, or dict for flexibility."
    )
    embedding: Union[EmbeddingConfig, Dict[str, Any]] = Field(
        ...,
        description="Embedding configuration (required). Use EmbeddingConfig({'openai': OpenAIEmbeddingConfig(...)}) or EmbeddingConfig({'gemini': GeminiEmbeddingConfig(...)}) for type safety, or dict for flexibility."
    )
    vector: Union[VectorStoreConfig, Dict[str, Any]] = Field(
        ...,
        description="Vector store configuration (required). Use VectorStoreConfig({'faiss': FAISSConfig(...)}) for type safety, or dict for flexibility."
    )
    debug: bool = Field(
        False,
        description="Enable debug mode for verbose logging"
    )
