# Memory for LLM

Long-term memory for LLM applications: extract, store, and retrieve conversation-based memories so your assistant can respond with relevant context.

**Features:**

- **Extract** memories from user/assistant messages (preferences, facts, context) via an LLM
- **Store** memories in metadata storage (MongoDB or PostgreSQL) and vector storage (FAISS)
- **Retrieve** by semantic similarity and inject into the LLM context

**Supported providers:**

| Layer        | Options                    |
| ------------ | -------------------------- |
| LLM          | Gemini, Hugging Face (e.g. Meta Llama 3) |
| Embeddings   | OpenAI, Gemini, Hugging Face (e.g. Qwen3-VL-Embedding-2B) |
| Vector store | FAISS                      |
| Metadata     | MongoDB, PostgreSQL        |

---

## Requirements

- **Python 3.13+**
- API keys for the providers you use (e.g. Gemini, optionally OpenAI for embeddings)
- MongoDB or PostgreSQL (for metadata) — e.g. via Docker

---

## Installation

From the project root:

**Using [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
uv sync
```

**Using pip:**

```bash
pip install -e .
```

This installs the project and its dependencies. The project is run as an application (no separate package install from PyPI).

---

## Configuration

Create a `.env` file in the project root (or set environment variables) for API keys and optional overrides:

```env
# Required for LLM and (optionally) embeddings
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Optional: if using Hugging Face as LLM (e.g. Meta Llama 3)
HUGGINGFACE_API_KEY=your-hf-token
HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
HUGGINGFACE_INFERENCE_PROVIDER=featherless-ai

# Optional: if using Hugging Face for embeddings (e.g. Qwen3-VL-Embedding-2B)
HUGGINGFACE_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
HUGGINGFACE_EMBEDDING_PROVIDER=hf-inference

# Optional: if using OpenAI for embeddings
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## Basic usage

### 1. Initialize Memory API

Configure LLM, metadata storage, embeddings, and vector store; then create `MemoryAPI` and (if needed) `RetrievalAPI`:

```python
from core.api.memory_api import MemoryAPI
from core.api.retrieval_api import RetrievalAPI
import os

config = {
    "llm": {
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        }
    },
    "storage": {
        "mongodb": {
            "uri": "mongodb://localhost:27017",
            "database": "memory_db",
            "collection": "memories",
        }
    },
    "embedding": {
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": "gemini-embedding-001",
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 768,
        }
    },
    "vector": {
        "faiss": {
            "dimension": 768,
            "index_path": "./faiss_index",
            "index_type": "COSINE",
        }
    },
}

memory_api = MemoryAPI(config)
retrieval_api = RetrievalAPI(
    storage=memory_api.memory_store.storage,
    vector_store=memory_api.memory_store.vector_store,
    embedding_generator=memory_api.memory_store.embedding_generator,
)
```

To use **Hugging Face** (e.g. [Meta Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)) as the LLM, set `"llm": {"huggingface": {"api_key": os.getenv("HUGGINGFACE_API_KEY"), "model": "meta-llama/Meta-Llama-3-8B-Instruct", "provider": "featherless-ai"}}` and keep the rest of the config the same.

### 2. Add memories from a conversation turn

After each assistant response, call `add_memory` with the last few turns and the current user/assistant messages:

```python
recent = [{"user": "...", "assistant": "..."}]  # last N turns
stored = memory_api.add_memory(
    recent_messages=recent,
    user_message="I'm vegetarian and I love Italian food.",
    assistant_message="Got it! I'll keep that in mind...",
)
# stored is a list of Memory objects that were stored
```

### 3. Retrieve memories for context

Before generating a reply, retrieve relevant memories and pass them into your system prompt:

```python
memories = retrieval_api.retrieve(query=user_message, top_k=5)
context = "\n".join(m.content for m in memories)
system_prompt = f"Relevant context:\n{context}\n\n..."
response = llm_provider.send_message(user_message, system_instruction=system_prompt)
```

---

## Running the CLI example

A full chat CLI that uses memory extraction and retrieval is in `examples/cli`. See **[examples/README.md](examples/README.md)** for how to run it (env, MongoDB, and commands).

---

## Project layout

```
core/           # Memory extraction, storage, retrieval, LLM/embedding wiring
storage/        # Metadata (MongoDB, Postgres) and vector (FAISS) backends
examples/       # CLI and docker-compose for MongoDB/Postgres
main.py         # Entry point that runs the CLI
```
