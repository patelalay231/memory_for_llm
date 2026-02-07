from .base import LLMProvider
from google import genai
from google.genai import types
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiProvider(LLMProvider):
    """
    Gemini provider with a friendly, conversational prompt.
    """
    
    def __init__(self):
        """
        Initialize the Gemini provider.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL_NAME")
        
        if not self.api_key:
            raise ValueError("❌ Error: GEMINI_API_KEY not found in environment variables")
        if not self.model:
            raise ValueError("❌ Error: GEMINI_MODEL_NAME not found in environment variables")

        self.client = genai.Client(
            api_key=self.api_key,
        )

    # keep the system_instuction as optional parameter    
    def send_message(self, message: str, system_instruction: Optional[str] = None) -> str:
        """
        Send a message to the Gemini provider and get a response.
        """
        try:

            contents = self.get_content(message)
            response = self.get_response(contents, system_instruction)

            return response.text
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}. Could you try again?"

        
    def get_content(self, message: str) -> str:
        """
        Get the content for the message.
        """
        return [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=message),
                ],
            ),
        ]

    def get_response(self, contents, system_instruction: Optional[str] = None) -> str:
        """
        Get a response from the Gemini provider.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction if system_instruction else 'You are a helpful assistant.',
            )
        )
        return response