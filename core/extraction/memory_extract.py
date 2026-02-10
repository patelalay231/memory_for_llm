import time
from typing import Literal, Optional

from core.llm.base import LLMProvider
from core.prompts import get_memory_extraction_prompt
from core.models.Memory import Memory
from logger import Logger
import json

MemoryType = Literal["user", "agent", "both"]


class MemoryExtract:
    """
    Memory extraction class with validation and retry logic.
    """
    
    def __init__(self, provider: LLMProvider, max_retries: int = 3):
        """
        Initialize the memory extraction class.
        
        Args:
            provider: LLM provider instance
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.provider = provider
        self.max_retries = max_retries

    def extract_memory(
        self,
        user_message: str,
        assistant_message: str,
        recent_messages: Optional[list[dict]] = None,
        memory_type: MemoryType = "both",
    ) -> list[Memory]:
        """
        Extract and validate memories with retry logic.

        Args:
            user_message: Current user message.
            assistant_message: Current assistant response.
            recent_messages: Optional list of prior turns. Default [].
            memory_type: "user" | "agent" | "both" — which side(s) to extract from.

        Returns:
            List of validated Memory objects.

        Raises:
            Exception: If extraction fails after max_retries attempts.
        """
        Logger.debug("Starting memory extraction...", "[MemoryExtract]")
        recent_messages = recent_messages or []

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    Logger.debug(f"Retry attempt {attempt + 1}/{self.max_retries}...", "[MemoryExtract]")

                Logger.debug("Sending prompt to LLM...", "[MemoryExtract]")
                prompt = get_memory_extraction_prompt(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    recent_messages=recent_messages,
                    memory_type=memory_type,
                )
                # Call provider with prompt as message (system_instruction is optional)
                response = self.provider.send_message(prompt, system_instruction=None)
                Logger.debug("Received response from LLM, validating...", "[MemoryExtract]")
                
                # Validate and parse (pass user_message as source)
                memories = self._parse_and_validate(response, user_message)
                Logger.debug(f"Successfully extracted {len(memories)} memory/memories", "[MemoryExtract]")
                return memories  # Success!
                
            except (ValueError, json.JSONDecodeError) as e:
                Logger.debug(f"Validation failed: {e}", "[MemoryExtract]")
                if attempt == self.max_retries - 1:
                    Logger.debug(f"Failed after {self.max_retries} attempts", "[MemoryExtract]")
                    raise Exception(f"Failed after {self.max_retries} attempts: {e}")
                # Retry... after a delay
                time.sleep(10)
                continue
        
        Logger.debug("No memories extracted", "[MemoryExtract]")
        return []  # No memories extracted
    
    def _parse_and_validate(self, raw_output: str, user_message: str) -> list[Memory]:
        """
        Parse JSON and validate against Memory model.
        
        Args:
            raw_output: Raw string output from LLM
            user_message: The user message that will be used as the source for all memories
            
        Returns:
            List of validated Memory objects
            
        Raises:
            ValueError: If JSON structure is invalid
            json.JSONDecodeError: If JSON parsing fails
        """
        # Strip markdown code blocks
        cleaned = self._clean_json_output(raw_output)
        
        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format: {e}", e.doc, e.pos)
        
        # Validate structure
        if "memories" not in data:
            raise ValueError("Missing 'memories' field in JSON response")
        
        if not isinstance(data["memories"], list):
            raise ValueError("'memories' field must be a list")
        
        if len(data["memories"]) == 0:
            Logger.debug("No memories found in response (empty list)", "[MemoryExtract]")
            return []

        memories = []
        for idx, m in enumerate(data["memories"]):
            try:
                source = m.get("source")
                memory = Memory(
                    source=source,
                    content=m.get("content"),
                    type=m.get("type"),
                )
                memories.append(memory)
                Logger.debug(f"Validated memory {idx + 1}: {memory.type} - {memory.content[:50]}... (source: {memory.source})", "[MemoryExtract]")
            except Exception as e:
                Logger.debug(f"Failed to validate memory {idx + 1}: {e}", "[MemoryExtract]")
                raise ValueError(f"Invalid memory structure at index {idx}: {e}")
        
        return memories
    
    def _clean_json_output(self, raw: str) -> str:
        """
        Remove markdown code blocks if present.
        
        Args:
            raw: Raw string output
            
        Returns:
            Cleaned string without markdown code blocks
        """
        # Remove ```json ... ``` or ``` ... ```
        if raw.strip().startswith("```"):
            lines = raw.strip().split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return raw
