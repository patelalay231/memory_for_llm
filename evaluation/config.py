"""Evaluation config: same LLM/storage/embedding as CLI (MongoDB, FAISS)."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "dataset"
DATASET_FILE = DATASET_DIR / "locomo10.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default result filenames
ADD_RESULTS_DIR = RESULTS_DIR  # add phase doesn't write a single file; search writes results
SEARCH_RESULTS_FILE = RESULTS_DIR / "memoery_search_results.json"
EVAL_METRICS_FILE = RESULTS_DIR / "evaluation_metrics.json"


def get_config():
    """Build config dict matching examples/cli (MongoDB, FAISS, same LLM/embedding)."""
    llm_provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if llm_provider_name == "huggingface":
        hf_cfg = {
            "api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
            "model": os.getenv("HUGGINGFACE_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        }
        if os.getenv("HUGGINGFACE_PROVIDER"):
            hf_cfg["provider"] = os.getenv("HUGGINGFACE_PROVIDER")
        llm_config = {"huggingface": hf_cfg}
    else:
        llm_config = {
            "gemini": {
                "api_key": os.getenv("GEMINI_API_KEY", ""),
                "model": os.getenv("GEMINI_MODEL_NAME", "gemini-pro"),
            }
        }

    return {
        "llm": llm_config,
        "storage": {
            "mongodb": {
                "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
                "database": os.getenv("MONGODB_DATABASE", "memory_db"),
                "collection": os.getenv("MONGODB_COLLECTION", "memories"),
            }
        },
        "embedding": {
            "huggingface": {
                "api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
                "model": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
                "provider": os.getenv("HUGGINGFACE_EMBEDDING_PROVIDER", "auto"),
                "normalize": True,
                "output_dimensionality": 1536,
            }
        },
        "vector": {
            "faiss": {
                "dimension": 1536,
                "index_path": os.getenv("FAISS_INDEX_PATH", str(EVAL_DIR / "faiss_eval_index")),
                "index_type": "COSINE",
            }
        },
        "debug": os.getenv("MEMORY_API_DEBUG", "false").lower() == "true",
    }
