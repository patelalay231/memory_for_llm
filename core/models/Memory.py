from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4


class Memory(BaseModel):
    """Memory object stored in metadata storage."""
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # "conversation"
    content: str  # The actual memory content
    type: str  # Category: "dietary_preference", "personal_info", etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None  # Embedding vector for the memory content (stored in metadata)
    user_id: Optional[str] = None  # Optional scope for multi-user evaluation (e.g. speaker_a_0)
    conversation_id: Optional[str] = None  # Optional; used by some storage backends (e.g. Postgres)
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "user_message",
                "content": "The user is vegetarian",
                "type": "dietary_preference",
                "embedding": [0.1, 0.2, 0.3, ...]  
            }
        }
