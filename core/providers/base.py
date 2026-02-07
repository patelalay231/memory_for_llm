from abc import ABC, abstractmethod

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
    def send_message(self, message: str, system_instruction: str) -> str:
        """
        Send a message to the LLM provider.
        """
        pass