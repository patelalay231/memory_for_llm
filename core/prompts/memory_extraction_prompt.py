def get_memory_extraction_prompt(
    recent_messages: list[dict],
    user_message: str,
    assistant_message: str,
) -> str:
    """
    Build a prompt that instructs the model to extract long-term memories from both
    user and assistant messages that are useful for future responses.
    """

    recent_messages_str = "\n".join(
        f"User: {message['user']}\nAssistant: {message['assistant']}"
        for message in recent_messages
    )

    return f"""
You are a memory extraction engine for a long-term AI assistant.

Your job is to THINK about the conversation and decide what (if anything) is worth remembering for future conversations. Do NOT extract everything—only what will genuinely help the assistant respond better later.

STEP 1: Consider the conversation. Ask yourself: "If we have a new chat next week, what from this exchange would be useful to know?"
STEP 2: Extract ONLY those items. If nothing is worth storing, return an empty list.

What is worth remembering (be selective):
- User-related facts: preferences, profile, habits, relationships, work, goals—when they are stated clearly and will affect future answers.
- What the assistant said that should be reused: definitions you gave, facts you stated, or decisions the user agreed to (e.g. "We agreed to use weekly backups", "Project uses Python 3.11").
- Decisions or agreements that both sides relied on.

What is NOT worth remembering (do not extract):
- Greetings, thanks, small talk ("hi", "how are you", "thanks!").
- One-off questions or answers that won't matter later.
- Temporary plans or mood ("I'm tired today", "I'll try that tomorrow").
- General knowledge the assistant stated that isn't specific to this user or project.
- Anything that is only relevant to this single turn.

Each memory must be a single, standalone factual statement. Do not infer beyond what was said.

---

EXAMPLES

Example 1 — Worth storing (user fact + assistant-agreed fact):
User: "I'm vegetarian and I work from home on Mondays."
Assistant: "Got it. For your diet, you could try lentil curry. For Mondays, I'll keep that in mind."
→ Store: user preference (vegetarian), user context (works from home Mondays). The assistant's suggestion (lentil curry) is a one-off tip, not worth storing.

Example 2 — Worth storing (from assistant answer):
User: "What version of Python does our project use?"
Assistant: "Your project is set up with Python 3.11, and you deploy to production every Friday."
→ Store: fact from assistant (project uses Python 3.11, deploys Fridays). This will help future answers about the project.

Example 3 — Nothing to store:
User: "What's the weather like?"
Assistant: "I don't have access to real-time weather. You could check a weather site."
→ No lasting fact about the user or agreed context. Return empty list.

Example 4 — Nothing to store:
User: "Thanks, that helped!"
Assistant: "Glad I could help. Ask anytime."
→ Polite closing, no factual content to remember. Return empty list.

---

Below is the conversation to analyze.

Recent conversation (last N turns):
{recent_messages_str}

Current exchange:
User: {user_message}
Assistant: {assistant_message}

TASK: Think about what (if anything) from this conversation should be remembered. Extract only those items. Return valid JSON.

Return the output strictly as valid JSON.

EXPECTED OUTPUT FORMAT:
```json
{{
  "memories": [
    {{
      "source": "user_message",
      "content": "The user is vegetarian",
      "type": "user_preference"
    }},
    {{
      "source": "assistant_message",
      "content": "Project uses Python 3.11 and deploys on Fridays",
      "type": "context"
    }}
  ]
}}
```
If nothing is worth storing, return:
```json
{{
  "memories": []
}}
```

Note: "source" is "user_message", "assistant_message", or "conversation". "type" examples: user_preference, user_context, fact, definition, decision, context.
"""