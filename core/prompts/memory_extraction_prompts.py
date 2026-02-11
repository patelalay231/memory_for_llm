from datetime import datetime


def parse_messages(messages: list[dict]) -> str:
    """Turn list of {role, content} into a single string (mem0-style)."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ----- User memory extraction (facts ONLY from user messages) -----
USER_MEMORY_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.
Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.
This allows for easy retrieval and personalization in future interactions.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.
# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.

Here are some few shot examples:

User: Hi.
Assistant: Hello! How can I help today?
Output: {{"memories": []}}

User: Hi, I am looking for a restaurant in San Francisco.
Assistant: Sure, I can help with that. Any particular cuisine you're interested in?
Output: {{"memories": [{{"source": "user_message", "content": "Looking for a restaurant in San Francisco", "type": "preference"}}]}}

User: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Assistant: Sounds like a productive meeting.
Output: {{"memories": [{{"source": "user_message", "content": "Had a meeting with John at 3pm and discussed the new project", "type": "fact"}}]}}

User: Hi, my name is John. I am a software engineer.
Assistant: Nice to meet you, John! How can I help?
Output: {{"memories": [{{"source": "user_message", "content": "Name is John", "type": "personal_info"}}, {{"source": "user_message", "content": "Is a software engineer", "type": "professional"}}]}}

Return the facts and preferences in JSON format with a key "memories" and a list of objects, each with "source", "content", and "type".
- "source" must be "user_message" (you are only extracting from user messages).
- "type" examples: user_preference, personal_info, fact, plan, professional, context.

Remember:
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- If you do not find anything relevant in the conversation, return {{"memories": []}}.
- Create the facts based on the user messages only. Do not pick anything from the assistant or system messages.
- Detect the language of the user input and record the facts in the same language.

Following is the conversation. Extract relevant facts and preferences about the user from USER messages only, and return them in the JSON format above.
"""


# ----- Agent memory extraction (facts ONLY from assistant messages) -----
AGENT_MEMORY_EXTRACTION_PROMPT = f"""You are an Assistant Information Organizer, specialized in accurately storing facts, preferences, and characteristics about the AI assistant from conversations.
Your primary role is to extract relevant pieces of information about the assistant from conversations and organize them into distinct, manageable facts.
This allows for easy retrieval and characterization of the assistant in future interactions.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES. DO NOT INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.
# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.

Types of Information to Remember:

1. Assistant's Preferences: Keep track of likes, dislikes, and specific preferences the assistant mentions in various categories such as activities, topics of interest, and hypothetical scenarios.
2. Assistant's Capabilities: Note any specific skills, knowledge areas, or tasks the assistant mentions being able to perform.
3. Assistant's Hypothetical Plans or Activities: Record any hypothetical activities or plans the assistant describes engaging in.
4. Assistant's Personality Traits: Identify any personality traits or characteristics the assistant displays or mentions.
5. Assistant's Approach to Tasks: Remember how the assistant approaches different types of tasks or questions.
6. Assistant's Knowledge Areas: Keep track of subjects or fields the assistant demonstrates knowledge in.
7. Miscellaneous Information: Record any other interesting or unique details the assistant shares about itself.

Here are some few shot examples:

User: Hi, I am looking for a restaurant in San Francisco.
Assistant: Sure, I can help with that. Any particular cuisine you're interested in?
Output: {{"memories": []}}

User: Hi, my name is John. I am a software engineer.
Assistant: Nice to meet you, John! My name is Alex and I admire software engineering. How can I help?
Output: {{"memories": [{{"source": "assistant_message", "content": "Admires software engineering", "type": "preference"}}, {{"source": "assistant_message", "content": "Name is Alex", "type": "personal_info"}}]}}

User: Me favourite movies are Inception and Interstellar. What are yours?
Assistant: Great choices! Mine are The Dark Knight and The Shawshank Redemption.
Output: {{"memories": [{{"source": "assistant_message", "content": "Favourite movies are Dark Knight and Shawshank Redemption", "type": "preference"}}]}}

Return the facts and preferences in JSON format with a key "memories" and a list of objects, each with "source", "content", and "type".
- "source" must be "assistant_message" (you are only extracting from assistant messages).
- "type" examples: preference, personal_info, capability, personality, context.

Remember:
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- If you do not find anything relevant, return {{"memories": []}}.
- Create the facts based on the assistant messages only. Do not pick anything from the user or system messages.
- Detect the language of the assistant input and record the facts in the same language.

Following is the conversation. Extract relevant facts and preferences about the assistant from ASSISTANT messages only, and return them in the JSON format above.
"""


def get_fact_retrieval_messages(parsed_message: str, is_agent_memory: bool) -> tuple[str, str]:
    """
    Get (system_prompt, user_prompt) for extraction, like mem0.
    - is_agent_memory True -> agent-only extraction.
    - is_agent_memory False -> user-only extraction.
    """
    if is_agent_memory:
        system_prompt = AGENT_MEMORY_EXTRACTION_PROMPT
    else:
        system_prompt = USER_MEMORY_EXTRACTION_PROMPT
    user_prompt = f"Input:\n{parsed_message}"
    return system_prompt, user_prompt
