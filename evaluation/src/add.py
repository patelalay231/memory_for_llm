"""Add LOCOMO conversations to memory (per-user, with delete_all before each conversation)."""
import json
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# Add project root so we can import core and storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.api.memory_api import MemoryAPI
from evaluation.config import get_config, DATASET_FILE
from evaluation.src.utils import load_locomo


class MemoryADD:
    def __init__(self, data_path: Path = None, batch_size: int = 2):
        self.config = get_config()
        self.memory_api = MemoryAPI(self.config)
        self.batch_size = batch_size
        self.data_path = data_path or DATASET_FILE
        self.data = None
        if self.data_path.exists():
            self.load_data()

    def load_data(self):
        self.data = load_locomo(self.data_path)
        return self.data

    def add_memories_for_speaker(self, user_id: str, messages: list, desc: str):
        """messages: list of {"role": "user"|"assistant", "content": str} (alternating)."""
        # Our API takes one turn: recent_messages, user_message, assistant_message
        n = len(messages)
        for i in tqdm(range(0, n - 1, 2), desc=desc):  # pairs (user, assistant)
            user_content = messages[i]["content"]
            assistant_content = messages[i + 1]["content"]
            recent = [
                        {"user": messages[j]["content"], "assistant": messages[j + 1]["content"]}
                        for j in range(0, i, 2)
                    ]
            for attempt in range(3):
                try:
                    self.memory_api.add_memory(
                        recent_messages=recent,
                        user_message=user_content,
                        assistant_message=assistant_content,
                        user_id=user_id,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(1)

    def process_conversation(self, item: dict, idx: int):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        self.memory_api.delete_all_for_user(speaker_a_user_id)
        self.memory_api.delete_all_for_user(speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key)
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, "Adding Memories for Speaker B"),
            )
            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

    def process_all_conversations(self, max_workers: int = 1):
        if not self.data:
            raise ValueError("No data loaded. Set data_path and ensure dataset exists.")
        for idx, item in enumerate(tqdm(self.data, desc="Conversations")):
            self.process_conversation(item, idx)
