from abc import ABC, abstractmethod
from typing import Optional
from .generation_config import GenerationConfig

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    @abstractmethod
    def __init__(self):
        """
        Initialize the LLM provider.
        """
        pass

    @abstractmethod
    def send_message(
        self, 
        message: str, 
        system_instruction: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Send a message to the LLM provider.
        
        Args:
            message: The user message to send
            system_instruction: Optional system instruction/prompt
            generation_config: Optional generation configuration (temperature, max_tokens, etc.)
        
        Returns:
            The LLM response as a string
        """
        pass
