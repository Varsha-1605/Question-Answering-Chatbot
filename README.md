# Enterprise RAG

A document question-answering service. You upload files, they are chunked,
embedded and stored in Weaviate, and you ask questions in a chat page that
streams answers over a WebSocket. Answers come only from your documents.

## Architecture

```
Browser (frontend/)  --WebSocket /ws/chat-->  FastAPI (main.py)
        |                                          |
        +--POST /ingest-->  ingestion  ------------+
                                |                  |
                      load -> clean -> chunk       LangGraph pipeline
                                |                  (retrieval/graph.py)
                                v                       |
                       Weaviate (hybrid index)  <-------+
                                                        |
                                    LLM: OpenAI | AWS Bedrock | Ollama

Optional: Redis (embedding + answer cache) · Postgres (chat memory) · OpenRouter (LLM fallback)
```

**Ingestion** (`ai_workflow/injestion/`): a file is loaded by type (pdf, txt,
docx, csv, xlsx), cleaned, split into chunks, embedded and written to a Weaviate
index. A namespace is a Weaviate tenant. A file with unchanged content is skipped.

**Answering** (`ai_workflow/retrieval/graph.py`):

1. Rewrite the question into a search query and, in parallel, search with the raw question.
2. Route: small talk is answered directly; a standalone question reuses the raw search; a follow-up searches again with the rewritten query.
3. Hybrid (keyword + vector) search, then optional cross-encoder reranking.
4. The chat model streams an answer from the top passages.

**Code layout**

| Path | Role |
|---|---|
| `main.py` | App setup, `/health`, `/ready`, `/ui-config`, serves the chat page |
| `ai_workflow/injestion/` | Upload route, file loading and chunking |
| `ai_workflow/retrieval/` | WebSocket route, LangGraph pipeline, prompts, reranker, chat memory, answer cache |
| `ai_workflow/vector_db/` | Weaviate client, search, storing, deleting |
| `ai_workflow/cache/` | Redis client, embedding cache, semantic cache, stats |
| `ai_workflow/utils.py`, `settings.py` | LLM and embedding factories; configuration |
| `frontend/` | Chat page (`index.html`, `app.js`) and an alternative Streamlit app |
| `evals/`, `goldens/`, `baselines/` | Evaluation suite, test questions, saved results |

## Features

- Streaming answers over WebSocket, in a language you choose.
- Switchable LLM provider with `LLM_SERVICE`.
- Hybrid search with optional reranking (`RERANKER_ENABLED`).
- Chat memory in Postgres (`POSTGRES_URL`); without it the browser resends the history.
- Redis caching of embeddings, and optional reuse of answers to similar questions (`SEMANTIC_CACHE_ENABLED`). A Redis failure is treated as a cache miss.
- Optional OpenRouter fallback and per-minute OpenAI budgets (`LLM_FALLBACK_ENABLED`).
- Startup warm-up; `/ready` returns 503 until it finishes.
- Limits on upload size, question length and WebSocket connections.
- Evaluation suite covering retrieval, answers, safety, latency, cost and reliability, with a baseline comparison.

## How to run

```bash
cp .env.example .env     # set OPENAI_API_KEY, or change LLM_SERVICE
docker compose up -d weaviate        # add redis / postgres if you enable them
uv sync
uv run uvicorn main:app --reload     # chat page at http://localhost:8000
uv run pytest
```

On Windows with `POSTGRES_URL` set, start with
`uv run uvicorn main:app --loop asyncio:SelectorEventLoop`.

## How to use

Upload files in the chat page, or:

```bash
curl -F "files=@data/report.pdf" localhost:8000/ingest
```

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Multipart `files`. Stored in `INDEX_NAME` / `DEFAULT_NAMESPACE`; `index_name`, `namespace` and `delete_status` fields are ignored. |
| `WS /ws/chat` or `/ws/chat/{index}/{namespace}` | Send `{question, language?, chat_context?}`; receive `stream`, `end` and `error` messages. Optional `?thread_id=`. Send `ping` to get `pong`. |
| `GET /chat/threads/{thread_id}/messages` | Stored messages of a chat (empty without Postgres). |
| `POST /chat/title` | Send `{question}`; returns `{title}` (a short chat title, or `null`). The page calls it for a new chat's first question, in parallel with the answer. |
| `GET /health`, `GET /ready` | Liveness; readiness (warm-up done and Weaviate reachable). |
| `GET /ui-config` | Defaults and limits for the chat page. |

The delete routes and `GET /cache/stats` exist in the code but are not mounted.
There is no authentication; add your own access control and rate limiting.

## Evaluate

```bash
uv run python -m evals.run_suit --baseline    # write baselines/baseline.json
uv run python -m evals.run_suit               # write baselines/candidate.json
uv run python -m evals.compare                # PASS / REVIEW / FAIL verdict
```

Use `--only retriever,latency` to run a subset.

## Notes

- Embedding models differ per provider, so use a separate `INDEX_NAME` (or re-ingest) when you change `LLM_SERVICE`.
- `ENVIRONMENT=production` switches logs to JSON. All settings are listed in `.env.example`.
