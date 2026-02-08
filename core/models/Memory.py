from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from uuid import uuid4


class Memory(BaseModel):
    """Memory object stored in metadata storage."""
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # "conversation"
    content: str  # The actual memory content
    type: str  # Category: "dietary_preference", "personal_info", etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "user_message",
                "content": "The user is vegetarian",
                "type": "dietary_preference"
            }
        }
