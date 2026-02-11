import time
from core.llm.base import LLMProvider
from core.prompts.memory_extraction_prompts import (
    get_fact_retrieval_messages,
    parse_messages,
)
from core.models.Memory import Memory
from logger import Logger
import json


def _should_use_agent_memory_extraction(messages: list[dict], metadata: dict | None) -> bool:
    """Use agent extraction when metadata has agent_id and messages contain assistant role (mem0-style)."""
    if not metadata:
        return False
    if not metadata.get("agent_id"):
        return False
    return any(m.get("role") == "assistant" for m in messages)


class MemoryExtract:
    """
    Memory extraction with validation and retry logic.
    """

    def __init__(self, provider: LLMProvider, max_retries: int = 3):
        self.provider = provider
        self.max_retries = max_retries

    def extract_memory(self, messages: list[dict], metadata: dict | None = None) -> list[Memory]:
        """
        Extract memories from a list of messages.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            metadata: Optional dict; if agent_id is set and messages contain assistant, use agent extraction.

        Returns:
            List of Memory objects.
        """
        Logger.debug("Starting memory extraction...", "[MemoryExtract]")

        parsed = parse_messages(messages)
        is_agent_memory = _should_use_agent_memory_extraction(messages, metadata)
        system_prompt, user_prompt = get_fact_retrieval_messages(parsed, is_agent_memory)

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    Logger.debug(f"Retry attempt {attempt + 1}/{self.max_retries}...", "[MemoryExtract]")
                Logger.debug("Sending prompt to LLM...", "[MemoryExtract]")
                response = self.provider.send_message(user_prompt, system_instruction=system_prompt)
                Logger.debug("Received response from LLM, validating...", "[MemoryExtract]")
                memories = self._parse_and_validate(response)
                Logger.debug(f"Successfully extracted {len(memories)} memory/memories", "[MemoryExtract]")
                return memories
            except (ValueError, json.JSONDecodeError) as e:
                Logger.debug(f"Validation failed: {e}", "[MemoryExtract]")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed after {self.max_retries} attempts: {e}")
                time.sleep(5)

        return []

    def _parse_and_validate(self, raw_output: str) -> list[Memory]:
        cleaned = self._clean_json_output(raw_output)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format: {e}", e.doc, e.pos)

        if "memories" not in data:
            raise ValueError("Missing 'memories' field in JSON response")
        if not isinstance(data["memories"], list):
            raise ValueError("'memories' field must be a list")
        if len(data["memories"]) == 0:
            return []

        memories = []
        for idx, m in enumerate(data["memories"]):
            try:
                memory = Memory(
                    source=m.get("source", "conversation"),
                    content=m.get("content", ""),
                    type=m.get("type", "fact"),
                )
                memories.append(memory)
            except Exception as e:
                raise ValueError(f"Invalid memory structure at index {idx}: {e}")
        return memories

    def _clean_json_output(self, raw: str) -> str:
        if raw.strip().startswith("```"):
            lines = raw.strip().split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return raw
