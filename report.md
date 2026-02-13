# Design Decisions & Trade-offs

## Architecture
Flat structure — one file per task + shared `config.py`. Each task runs independently via CLI while sharing common utilities (OpenAI client, token counting, cost calculation, timing). A `run_all.py` entrypoint executes all tasks in sequence for quick demo.

## SDK & Framework Choices

**LangChain is used only in `rag.py`** — the one task where it genuinely earns its weight. Document loaders (`PyPDFLoader`, `TextLoader`), text splitters (`RecursiveCharacterTextSplitter`), and the Chroma vectorstore integration (`langchain_chroma`) would each be 50–100 lines of boilerplate to write from scratch. LangChain also provides `ChatOpenAI` for the QA generation step in RAG queries.

**All other tasks use the raw OpenAI SDK directly** (`openai` Python client). The trade-offs:

| File | SDK | Why |
|------|-----|-----|
| `config.py` | `openai`, `tiktoken` | Shared client init, token counting — no framework needed |
| `chat.py` | `openai` | `stream=True` gives native SSE streaming; LangChain's streaming wrapper adds latency and hides chunk-level control |
| `rag.py` | **LangChain** + `chromadb` | PDF loading, text splitting, vectorstore CRUD, and embeddings integration justify the dependency |
| `agent.py` | `openai` | Native function calling (`tools=` parameter) gives full control over the tool loop, scratchpad, and iteration logging — LangChain's `AgentExecutor` abstracts away details the assessment explicitly asks to demonstrate |
| `healer.py` | `openai` | Simple completions call + subprocess — no framework adds value here |
| `api.py` | FastAPI | Lightweight REST wrapper; imports `rag` and `agent` modules directly |
| `dashboard.py` | Streamlit | Quick metrics UI with minimal code |

This deliberate split keeps the dependency footprint small while using LangChain where its abstractions provide real value.

## Task 3.1 — Chat (`chat.py`)
Used raw OpenAI SDK with `stream=True` for token-level streaming — LangChain's wrapper adds overhead for streaming use cases. History persisted in **SQLite** (`chat_history.db`) so conversations survive restarts. The `--history N` CLI flag lets users configure the sliding window size (default 10). Cost telemetry uses `tiktoken` (`cl100k_base` encoding) for accurate token counting and GPT-4o pricing ($2.50/1M input, $10.00/1M output).

## Task 3.2 — RAG (`rag.py`)
**Embeddings:** Chroma's built-in `DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via ONNX Runtime) — runs locally with no API key or PyTorch dependency, keeping the stack lightweight. **Chunking:** 1500 chars / 300 overlap to preserve context within chunks and improve retrieval accuracy. **Corpus:** ~50 MB across multiple public domain texts (LOTR, War and Peace, Shakespeare) with auto-download. Ingestion uses batched processing (100 chunks/batch) with a `tqdm` progress bar. The loader handles both PDF and TXT files. **Evaluation:** 20 keyword-matched questions at top-10 retrieval — deterministic and fast vs. LLM-as-judge which adds cost and non-determinism. **Citations:** LLM instructed to cite inline as `[1: Source Name, p.42]` with source name and page number for traceability.

## Task 3.3 — Agent (`agent.py`)
Native OpenAI function calling over LangChain's AgentExecutor — cleaner control over the tool loop and scratchpad logging. Five mock tools (`search_flights`, `search_accommodation`, `get_weather`, `search_attractions`, `get_budget_tips`) demonstrate the architecture; swapping real APIs is a config change. Output includes both a human-readable itinerary (formatted with emojis, budget breakdown) and raw JSON for programmatic use. The `_print_itinerary` formatter defensively handles varying LLM output shapes (dicts vs lists vs strings).

## Task 3.4 — Self-Healing (`healer.py`)
Supports **Python** (pytest) and **Rust** (cargo test) — auto-detects language from the prompt. Unified generate → write → test → fix loop with max 3 retries. Language-specific config is driven by a `LANG_CONFIG` dict to avoid code duplication. Falls back to Python if `cargo` is not installed. Trade-off: generating both code and tests from the same model risks correlated failures; production would use human-authored test suites.

## Docker (`docker-compose.yml`)
Three long-lived services: **ChromaDB** (standalone vector DB, port 8000), **FastAPI** (REST API exposing RAG + agent, port 8080), and **Streamlit** (dashboard UI, port 8501). Services use `sleep 5` + `restart: on-failure` for startup ordering — simpler and more portable than healthchecks, since the Chroma image lacks `curl`/`python` for reliable health probes. Three one-off services (`rag-ingest`, `run-all`, `tests`) are available via `docker compose run`.

**Dual-mode Chroma:** `rag.py` auto-detects the `CHROMA_HOST` env var — when empty it uses embedded Chroma (local dev), when set it connects via HTTP client (Docker). The same code works in both environments without changes. `docker-compose.yml` sets `CHROMA_HOST=chromadb` automatically for all services.

## Testing (`tests/test_all.py`)
Unit tests cover config utilities (token counting, cost calculation, timing), chat SQLite persistence, RAG vectorstore creation, agent tool dispatch + schema, healer language detection + code extraction, and the FastAPI health endpoint. All tests run offline without API keys (mocked where needed).

## Stretch — Dashboard (`dashboard.py`)
Streamlit app displaying per-task latency and cost metrics from JSON logs. Runs standalone or as a Docker service on port 8501.