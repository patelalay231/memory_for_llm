"""Run search phase: for each question retrieve + LLM answer, save results + latency."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import DATASET_FILE, SEARCH_RESULTS_FILE
from evaluation.src.search import MemorySearch


def main():
    if not DATASET_FILE.exists():
        print(f"Dataset not found: {DATASET_FILE}")
        sys.exit(1)
    search = MemorySearch(output_path=SEARCH_RESULTS_FILE, top_k=30)
    search.process_data_file(DATASET_FILE)
    print(f"Results saved to {SEARCH_RESULTS_FILE}")


if __name__ == "__main__":
    main()
