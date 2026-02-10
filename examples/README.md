# Examples

## CLI chat interface

The CLI runs a chat loop that:

1. Retrieves relevant memories for each user message
2. Sends the user message to the LLM with those memories in the system prompt
3. Extracts and stores new memories from the user/assistant exchange

### Prerequisites

- Project dependencies installed from the repo root (`uv sync` or `pip install -e .`)
- **MongoDB** running on `localhost:27017` (CLI is configured for MongoDB by default)
- **Gemini API key** set in the environment or in a `.env` file at the project root

### 1. Environment variables

In the **project root** (same folder as `main.py`), create a `.env` file if you haven’t:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

The CLI reads these via `python-dotenv`.

### 2. Start MongoDB (Docker)

From the **project root**:

```bash
docker compose -f examples/docker-compose.yml up -d mongodb
```

Or from this directory:

```bash
docker compose -f docker-compose.yml up -d mongodb
```

This starts MongoDB on port 27017 with a `memory_db` database. The CLI uses `memory_db` and collection `memories`.

### 3. Run the CLI

From the **project root** (so that `core`, `storage`, and `logger` resolve):

**With uv:**

```bash
uv run python main.py
```

**With pip:**

```bash
python main.py
```

You should see a prompt like `👤 User: `. Type your message and press Enter; the assistant reply and memory summary will appear. Repeat as needed.

### 4. Exit the chat

Type one of: `exit`, `quit`, `q`, `bye`, `goodbye` (case-insensitive) and press Enter to exit.

### 5. Optional: use PostgreSQL instead of MongoDB

Edit `examples/cli/interface.py` and change the `config_dict["storage"]` section to use Postgres, and ensure Postgres is running (e.g. `docker compose -f examples/docker-compose.yml up -d postgres`). The rest of the steps are the same.

---

## Docker Compose

`docker-compose.yml` in this folder defines:

- **mongodb** — MongoDB 7.0 on port 27017, database `memory_db`
- **postgres** — PostgreSQL 15 on port 5432, user `postgres`, password `postgres`, database `memory_db`

Start one or both as needed for your config.
