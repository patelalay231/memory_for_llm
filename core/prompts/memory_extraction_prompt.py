def get_memory_extraction_prompt(
    recent_messages: list[dict],
    user_message: str,
    assistant_message: str,
) -> str:
    """
    Build a prompt that instructs the model to extract long-term user memories.
    """

    recent_messages_str = "\n".join(
        f"User: {message['user']}\nAssistant: {message['assistant']}"
        for message in recent_messages
    )

    return f"""
You are a memory extraction engine for a long-term AI assistant.

Your job is to extract ONLY stable, long-term, user-specific facts that should be remembered across conversations.

DO NOT extract:
- transient states (mood, temporary plans, one-time actions)
- conversational fluff
- questions
- assistant suggestions
- information already known unless it is corrected or refined

A memory should be:
- specific
- factual
- persistent over time
- useful in future conversations

Each memory must be written as a standalone factual statement.

Below is context from a conversation between a user and an assistant.

Recent conversation (last N turns):
{recent_messages_str}

Current exchange:
User: {user_message}
Assistant: {assistant_message}

TASK:
Extract all new long-term memories implied or explicitly stated in the conversation.

Rules:
- Extract only user-related facts (preferences, profile, habits, relationships, locations, work, goals).
- If no new long-term memory is present, return an empty list.
- Each memory must be atomic (one fact per memory).
- Do NOT infer beyond what is stated.
- Do NOT repeat existing memories unless they are clarified or corrected.

Return the output strictly as valid JSON.

EXPECTED OUTPUT FORMAT:
```json
{{
  "memories": [
    {{
      "source": "user_message",
      "content": "The user is vegetarian",
      "type": "dietary_preference"
    }}
  ]
}}
If nothing should be stored:

{{
  "memories": []
}}
"""