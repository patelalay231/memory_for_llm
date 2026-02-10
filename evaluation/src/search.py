"""Search memories for both speakers, generate answer with LLM, record latency."""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

from jinja2 import Template
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.api.memory_api import MemoryAPI
from core.api.retrieval_api import RetrievalAPI
from evaluation.config import get_config, DATASET_FILE, SEARCH_RESULTS_FILE
from evaluation.prompts import ANSWER_PROMPT
from evaluation.src.utils import load_locomo


class MemorySearch:
    def __init__(self, output_path: Path = None, top_k: int = 30):
        self.config = get_config()
        self.memory_api = MemoryAPI(self.config)
        self.retrieval_api = RetrievalAPI(
            storage=self.memory_api.memory_store.storage,
            vector_store=self.memory_api.memory_store.vector_store,
            embedding_generator=self.memory_api.memory_store.embedding_generator,
        )
        self.llm_provider = self.memory_api.memory_store.llm_provider
        self.top_k = top_k
        self.output_path = output_path or SEARCH_RESULTS_FILE
        self.results = defaultdict(list)

    def search_memory(self, user_id: str, query: str):
        """Return (list of {memory, timestamp, score}, latency_seconds)."""
        t0 = time.perf_counter()
        memories = self.retrieval_api.retrieve(
            query=query,
            top_k=self.top_k,
            filter={"user_id": user_id},
        )
        latency = time.perf_counter() - t0
        out = [
            {
                "memory": m.content,
                "timestamp": m.timestamp.isoformat() if hasattr(m.timestamp, "isoformat") else str(m.timestamp),
                "score": 0.0,  # RetrievalAPI returns Memory without score; order reflects relevance
            }
            for m in memories
        ]
        return out, latency

    def answer_question(self, speaker_1_user_id: str, speaker_2_user_id: str, question: str):
        s1_memories, s1_time = self.search_memory(speaker_1_user_id, question)
        s2_memories, s2_time = self.search_memory(speaker_2_user_id, question)

        s1_lines = [f"{item['timestamp']}: {item['memory']}" for item in s1_memories]
        s2_lines = [f"{item['timestamp']}: {item['memory']}" for item in s2_memories]

        template = Template(ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(s1_lines, indent=4),
            speaker_2_memories=json.dumps(s2_lines, indent=4),
            question=question,
        )

        t1 = time.perf_counter()
        response = self.llm_provider.send_message(answer_prompt, system_instruction=None)
        response_time = time.perf_counter() - t1

        return (
            response.strip(),
            s1_memories,
            s2_memories,
            s1_time,
            s2_time,
            response_time,
        )

    def process_question(self, val: dict, speaker_a_user_id: str, speaker_b_user_id: str) -> dict:
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            s1_memories,
            s2_memories,
            s1_time,
            s2_time,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": s1_memories,
            "speaker_2_memories": s2_memories,
            "num_speaker_1_memories": len(s1_memories),
            "num_speaker_2_memories": len(s2_memories),
            "speaker_1_memory_time": s1_time,
            "speaker_2_memory_time": s2_time,
            "response_time": response_time,
        }
        return result

    def process_data_file(self, file_path: Path = None):
        path = file_path or DATASET_FILE
        data = load_locomo(path)
        for idx, item in enumerate(tqdm(data, desc="Conversations")):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            for q in tqdm(qa, desc=f"Questions conv {idx}", leave=False):
                r = self.process_question(q, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(r)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
