"""LLM judge: CORRECT/WRONG using same LLM as CLI (Gemini/HF)."""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import get_config
from core.api.memory_api import MemoryAPI
from core.llm.generation_config import GenerationConfig


ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic.
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references, but you should be generous - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response.

Return the label in JSON format with the key "label". Example: {{"label": "CORRECT"}}
"""


def _extract_json(text: str) -> str:
    """Extract first JSON object from text."""
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if match:
        return match.group(0)
    return "{}"


def evaluate_llm_judge(question: str, gold_answer: str, generated_answer: str, llm_provider) -> int:
    """Return 1 if CORRECT, 0 if WRONG."""
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    response = llm_provider.send_message(
        prompt,
        system_instruction=None,
        generation_config=GenerationConfig(temperature=0.0),
    )
    raw = _extract_json(response)
    try:
        data = json.loads(raw)
        label = (data.get("label") or "").strip().upper()
    except Exception:
        label = "WRONG"
    return 1 if label == "CORRECT" else 0
