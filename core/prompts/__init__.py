"""Prompts module for memory extraction and other LLM operations."""

from .memory_extraction_prompt import get_memory_extraction_prompt
from .memory_operations_prompt import get_memory_operations_prompt

__all__ = ["get_memory_extraction_prompt", "get_memory_operations_prompt"]
