"""Run add phase: load LOCOMO, delete per-user, add memories (same config as CLI)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import DATASET_FILE
from evaluation.src.add import MemoryADD


def main():
    if not DATASET_FILE.exists():
        print(f"Dataset not found: {DATASET_FILE}")
        print("Download LOCOMO and place locomo10.json in evaluation/dataset/")
        sys.exit(1)
    add = MemoryADD(data_path=DATASET_FILE)
    add.process_all_conversations(max_workers=1)


if __name__ == "__main__":
    main()
