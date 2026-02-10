def get_memory_operations_prompt(candidates_data: list[dict]) -> str:
    """
    Generate a prompt for processing candidate memories.
    
    Args:
        candidates_data: List of dictionaries, each containing:
            - candidate_id: Temporary ID for the candidate (e.g., "temp_0")
            - candidate_memory: Dict with "content" and "type"
            - existing_memories: List of existing memory payloads from vector search
    
    Returns:
        Formatted prompt string for operation determination
    """
    import json
    
    # Format candidates for the prompt
    candidates_formatted = []
    for idx, candidate_data in enumerate(candidates_data):
        candidate_id = candidate_data.get("candidate_id", f"temp_{idx}")
        candidate_memory = candidate_data.get("candidate_memory", {})
        existing_memories = candidate_data.get("existing_memories", [])
        
        candidates_formatted.append({
            "candidate_id": candidate_id,
            "candidate_memory": {
                "content": candidate_memory.get("content", ""),
                "type": candidate_memory.get("type", "")
            },
            "existing_memories": existing_memories
        })
    
    return f"""You are a memory management engine for a long-term AI assistant.

Your task is to decide what operation should be performed for each candidate memory.

For each candidate, choose exactly one operation:
- ADD: New distinct fact that doesn't overlap with existing memories
- UPDATE: Same fact as existing memory but more specific, recent, or accurate
- DELETE: Contradicts an existing memory that should be removed
- NOOP: Semantically equivalent to existing memory or adds no new information

RULES:
- Choose UPDATE over ADD when facts describe the same real-world attribute
- Choose NOOP over ADD when information is redundant
- Choose DELETE only when there is a clear contradiction
- If multiple existing memories match, select the BEST target
- If no existing memory is relevant, choose ADD

---

INPUT DATA:

{json.dumps(candidates_formatted, indent=2)}

---

REQUIRED OUTPUT FORMAT (JSON only, no explanations):

{{
  "operations": [
    {{
      "candidate_id": "temp_0",
      "operation": "ADD | UPDATE | DELETE | NOOP",
      "target_memory_id": "string or null",
      "confidence": 0.95
    }}
  ]
}}

IMPORTANT: 
- Return ONLY valid JSON
- Include one operation per candidate
- target_memory_id is required for UPDATE/DELETE, null for ADD/NOOP
- confidence should be between 0.0 and 1.0

---

EXAMPLES:

Example 1 - UPDATE:
Candidate: "User lives in Bangalore"
Existing: [{{"memory_id": "m2", "content": "User lives in Delhi"}}]
Output: {{"candidate_id": "temp_0", "operation": "UPDATE", "target_memory_id": "m2", "confidence": 0.93}}

Example 2 - NOOP:
Candidate: "User follows a vegetarian diet"
Existing: [{{"memory_id": "m1", "content": "User is vegetarian"}}]
Output: {{"candidate_id": "temp_0", "operation": "NOOP", "target_memory_id": null, "confidence": 0.88}}

Example 3 - ADD:
Candidate: "User is lactose intolerant"
Existing: [{{"memory_id": "m1", "content": "User is vegetarian"}}]
Output: {{"candidate_id": "temp_0", "operation": "ADD", "target_memory_id": null, "confidence": 0.91}}

Example 4 - DELETE:
Candidate: "User eats chicken regularly"
Existing: [{{"memory_id": "m1", "content": "User is vegetarian"}}]
Output: {{"candidate_id": "temp_0", "operation": "DELETE", "target_memory_id": "m1", "confidence": 0.95}}
""";
