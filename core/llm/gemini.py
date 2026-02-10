from .base import LLMProvider
from .generation_config import GenerationConfig
from google import genai
from google.genai import types
import os
from typing import Optional
from dotenv import load_dotenv
import time

load_dotenv()

class GeminiProvider(LLMProvider):
    """
    Gemini provider with a friendly, conversational prompt.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Optional API key. If not provided, will try to get from GEMINI_API_KEY env var.
            model: Optional model name. If not provided, will try to get from GEMINI_MODEL_NAME env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL_NAME")
        
        if not self.api_key:
            raise ValueError("❌ Error: GEMINI_API_KEY not found. Provide api_key parameter or set GEMINI_API_KEY environment variable")
        if not self.model:
            raise ValueError("❌ Error: GEMINI_MODEL_NAME not found. Provide model parameter or set GEMINI_MODEL_NAME environment variable")

        self.client = genai.Client(
            api_key=self.api_key,
        )

    def send_message(
        self, 
        message: str, 
        system_instruction: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Send a message to the Gemini provider and get a response.
        
        Args:
            message: The user message to send
            system_instruction: Optional system instruction/prompt
            generation_config: Optional generation configuration (temperature, max_tokens, etc.)
        
        Returns:
            The LLM response as a string
        """
        try:
            contents = self.get_content(message)
            response = self.get_response(contents, system_instruction, generation_config)
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

    def get_response(
        self, 
        contents, 
        system_instruction: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Get a response from the Gemini provider.
        
        Args:
            contents: Message contents
            system_instruction: Optional system instruction
            generation_config: Optional generation configuration
        """
        # Build GenerateContentConfig with system instruction
        config_dict = {
            "system_instruction": system_instruction if system_instruction else 'You are a helpful assistant.',
        }
        
        # Add generation config parameters if provided
        if generation_config:
            if generation_config.temperature is not None:
                config_dict["temperature"] = generation_config.temperature
            if generation_config.max_tokens is not None:
                config_dict["max_output_tokens"] = generation_config.max_tokens
            if generation_config.top_p is not None:
                config_dict["top_p"] = generation_config.top_p
            if generation_config.top_k is not None:
                config_dict["top_k"] = generation_config.top_k
            if generation_config.stop_sequences is not None:
                config_dict["stop_sequences"] = generation_config.stop_sequences
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_dict)
        )
        return response
