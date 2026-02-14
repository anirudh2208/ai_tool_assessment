# AI Tool Assessment

## Architecture
```
config.py          — Shared: OpenAI client, token counting, cost calc, timer
chat.py            — Task 3.1: Streaming chat + cost telemetry
rag.py             — Task 3.2: RAG pipeline (ingest, query, evaluate)
agent.py           — Task 3.3: Planning agent with tool calling
healer.py          — Task 3.4: Self-healing code assistant
api.py             — FastAPI service: /rag/query and /agent/plan endpoints
dashboard.py       — Stretch: Streamlit metrics dashboard
tests/test_all.py  — Unit & integration tests
docker-compose.yml — All services: ChromaDB, API, Dashboard
```

## Setup (local)
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in credentials
```

## Run Locally
```bash
python chat.py                                    # 3.1 — streaming chat, default: 10 messages
python chat.py --history 20                       # keep last 20 messages
python chat.py --history 5                        # keep last 5 messages
python rag.py ingest                              # 3.2 — ingest PDFs (uses embedded Chroma)
python rag.py query "Who is Frodo?"               # 3.2 — query with citations
python rag.py evaluate                            # 3.2 — retrieval accuracy report
python agent.py "Plan a 2-day trip to Auckland"   # 3.3 — planning agent
python healer.py "write quicksort in Python"      # 3.4 — self-healing code
python healer.py "write a function to solve the N-Queens problem and return all solutions as a list of board configurations" # Demo for multiple iteration
uvicorn api:app --port 8080                       # API server url http://localhost:8080/docs
pytest -q                                          # run tests
```

## Docker (spins up all services)
```bash
cp .env.example .env   # fill in credentials also uncomment CHROMA_HOST

# For proper build
docker compose down
docker compose build --no-cache
# Start all long-running services (vector DB + API + dashboard)
docker compose up chromadb api dashboard

# In another terminal — ingest documents, then run all tasks:
docker compose run rag-ingest
docker compose run run-all

# Or individually:
docker compose run tests
```

### Services
| Service     | Description                  | Port  | URL |
|-------------|------------------------------|-------|-----|
| `chromadb`  | ChromaDB vector database     | 8000  | http://localhost:8000/api/v1/heartbeat
| `api`       | FastAPI (RAG + Agent endpoints) | 8080 | http://localhost:8080/docs
| api health   | API health check             | 8080  | http://localhost:8080/health
| `dashboard` | Streamlit metrics UI         | 8501  | http://localhost:8501


### API Endpoints
- `POST /rag/query`  — `{"question": "Who is Frodo?", "top_k": 5}`
- `POST /agent/plan` — `{"prompt": "Plan a 2-day trip to Auckland for under 500"}`
- `GET  /health`

## Notes
- PDF downloads automatically. `data/`.
- Locally, Chroma runs embedded. In Docker, it runs as a standalone server (set via `CHROMA_HOST`).
- Agent uses 4 mock tools (flights, weather, attractions, accommodation).
- Healer retries up to 3 times, feeding errors back to the LLM.
