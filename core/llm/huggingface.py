from .base import LLMProvider
from .generation_config import GenerationConfig
from huggingface_hub import InferenceClient
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face LLM provider using the Inference API.
    Supports Meta Llama and other models on the Hub (e.g. meta-llama/Meta-Llama-3-8B-Instruct).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the Hugging Face provider.

        Args:
            api_key: Hugging Face token. If not provided, uses HUGGINGFACE_API_KEY or HF_TOKEN env var.
            model: Model id (e.g. "meta-llama/Meta-Llama-3-8B-Instruct"). If not provided, uses HUGGINGFACE_MODEL_NAME env var.
            provider: Optional inference provider (e.g. "featherless-ai", "hf-inference"). If not set, uses "auto".
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.model = model or os.getenv("HUGGINGFACE_MODEL_NAME")
        self.provider = provider if provider is not None else os.getenv("HUGGINGFACE_INFERENCE_PROVIDER", "auto")

        if not self.api_key:
            raise ValueError(
                "Hugging Face API key not found. Provide api_key parameter or set HUGGINGFACE_API_KEY environment variable"
            )
        if not self.model:
            raise ValueError(
                "Hugging Face model not found. Provide model parameter or set HUGGINGFACE_MODEL_NAME environment variable"
            )

        init_kwargs = {"token": self.api_key, "model": self.model}
        if self.provider and self.provider != "auto":
            init_kwargs["provider"] = self.provider
        self.client = InferenceClient(**init_kwargs)

    def send_message(
        self,
        message: str,
        system_instruction: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Send a message and get a response via Hugging Face chat completion.

        Args:
            message: The user message.
            system_instruction: Optional system prompt.
            generation_config: Optional generation parameters (temperature, max_tokens, etc.).

        Returns:
            The assistant response as a string.
        """
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": message})

            kwargs = {"model": self.model, "messages": messages}
            if generation_config:
                if generation_config.max_tokens is not None:
                    kwargs["max_tokens"] = generation_config.max_tokens
                if generation_config.temperature is not None:
                    kwargs["temperature"] = generation_config.temperature
                if generation_config.top_p is not None:
                    kwargs["top_p"] = generation_config.top_p
                if generation_config.stop_sequences:
                    kwargs["stop"] = generation_config.stop_sequences[:4]

            response = self.client.chat_completion(**kwargs)
            text = response.choices[0].message.content
            return text.strip() if text else ""
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}. Could you try again?"
