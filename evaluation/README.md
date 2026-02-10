# Memoery-for-LLM evaluation (LOCOMO)

Same dataset and approach as mem0 evaluation: LOCOMO dataset, add → search → BLEU/F1/LLM judge, plus latency.

## Setup

1. **Dataset**: Place `locomo10.json` in `evaluation/dataset/`.  
   Download from [mem0 LOCOMO](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link).

2. **Env**: Same as CLI. From project root, copy `.env` or set:
   - `GEMINI_API_KEY` / `GEMINI_MODEL_NAME` (or `LLM_PROVIDER=huggingface` + `HUGGINGFACE_*`)
   - `HUGGINGFACE_API_KEY` (for embeddings; same as CLI)
   - `MONGODB_URI` (default `mongodb://localhost:27017`)
   - Optional: `FAISS_INDEX_PATH`, `MONGODB_DATABASE`, `MONGODB_COLLECTION`

3. **Backends**: MongoDB and FAISS (same as CLI). Start MongoDB and use the same embedding dimension (e.g. 1536 for Qwen3-Embedding-8B).

## Run (from project root: `memoery_for_llm`)

```bash
# 1. Add memories (per conversation, per user_id)
python evaluation/run_add.py

# 2. Search + answer for each question; writes results + latency
python evaluation/run_search.py

# 3. Compute BLEU, F1, LLM judge; append latency to metrics
python evaluation/run_evals.py --input_file evaluation/results/memoery_search_results.json --output_file evaluation/results/evaluation_metrics.json

# 4. Aggregate by category and overall (including mean latency)
python evaluation/generate_scores.py
```

## Metrics

- **BLEU** (e.g. BLEU-1), **F1** (token overlap), **LLM score** (0/1 from same LLM as CLI).
- **Latency**: `speaker_1_memory_time`, `speaker_2_memory_time`, `response_time` (seconds); aggregated in `generate_scores.py`.

## Layout

- `config.py`: Same LLM/storage/embedding as CLI (MongoDB, FAISS).
- `prompts.py`: Answer-generation prompt (two speakers’ memories + question → short answer).
- `src/add.py`: LOCOMO → `MemoryAPI.add_memory(..., user_id=...)` with `delete_all_for_user` before each conversation.
- `src/search.py`: `RetrievalAPI.retrieve(..., filter={"user_id": ...})` + LLM answer; records latency.
- `metrics/utils.py`: BLEU, F1.
- `metrics/llm_judge.py`: CORRECT/WRONG using same LLM as CLI.
