from typing import Optional
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """
    Configuration for LLM generation parameters.
    
    Attributes:
        temperature: Controls randomness (0.0-2.0). Lower = more deterministic, higher = more creative.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling parameter.
        stop_sequences: List of sequences that stop generation.
    """
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for generation (0.0-2.0)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling threshold (0.0-1.0)")
    top_k: Optional[int] = Field(None, gt=0, description="Top-k sampling parameter")
    stop_sequences: Optional[list[str]] = Field(None, description="Sequences that stop generation")
