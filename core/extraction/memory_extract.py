from core.providers.base import LLMProvider
from core.prompts import get_memory_extraction_prompt
from core.providers.gemini import GeminiProvider

class MemoryExtract():
    """
    Memory extraction class.
    """
    def __init__(self):
        """
        Initialize the memory extraction class.
        """
        self.provider = GeminiProvider()

    def extract_memory(self, conversation_summary: str, recent_messages: list[dict], user_message: str, assistant_message: str) -> str:
        """
        Extract the memory from the conversation.
        """
        prompt = get_memory_extraction_prompt(conversation_summary, recent_messages, user_message, assistant_message)
        response = self.provider.send_message(prompt)
        return response