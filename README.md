<div align="center">

# Enterprise RAG

**Ask questions about your own documents and get streamed answers grounded in them.**
Questions about current events are answered with a live web search tool.

### 🚀 Deployed at [http://107.22.242.188:8000/](http://107.22.242.188:8000/)
Running on AWS EC2, with chat memory stored in AWS RDS Postgres.

[![Build and Deploy](https://github.com/Varsha-1605/Question-Answering-Chatbot/actions/workflows/deploy.yml/badge.svg)](https://github.com/Varsha-1605/Question-Answering-Chatbot/actions/workflows/deploy.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![Weaviate](https://img.shields.io/badge/Weaviate-hybrid%20search-00B140)
![Tavily](https://img.shields.io/badge/Tavily-web%20search-5B5BD6)

[Overview](#overview) · [Architecture](#architecture) · [Evaluation](#evaluation) · [Getting started](#getting-started) · [API](#api) · [Roadmap](#roadmap)

</div>

![Demo](https://github.com/user-attachments/assets/45cce508-7d19-426c-98e7-06364494728a)

## Overview

You upload files (PDF, DOCX, TXT, CSV, XLSX). The app splits them into chunks, embeds them and stores them in Weaviate. You then ask questions in a chat page, and the answer streams back over a WebSocket. Answers about your documents come only from your documents. Questions about current events (news, who holds an office today, prices, weather) go to a web search tool instead, and the chat page shows the tool call with its sources, as in Claude or ChatGPT.

What it does:

- **Smart routing.** A classifier sends each message one of three ways: your documents, a web search, or a plain chat reply. Greetings and thanks skip the classifier entirely. Questions that might be about your documents, even with typos or unknown names, never leave the documents.
- **Web search tool.** Current-events questions are searched with [Tavily](https://tavily.com/). The chat page shows a "Searched the web · `web_search`" row with the query and source links.
- **Hybrid search and reranking.** Keyword and vector search in Weaviate, then a cross-encoder reranks the results.
- **LLM gateway.** A LiteLLM router handles automatic retries, timeouts, fallback to Llama or Gemini through OpenRouter, per-model rate limits and usage-based load balancing. See [LLM gateway and reliability](#llm-gateway-and-reliability).
- **Redis caching.** It caches embeddings and answers to similar questions (semantic cache). If Redis fails, the app treats it as a cache miss and keeps working.
- **Chat memory.** Chats are stored in Postgres on AWS RDS, with auto-generated titles and renaming. The chat page keeps a chat list, waits for the server to be ready and shows clear error messages.
- **No duplicate work.** A file whose content has not changed is skipped on re-upload.
- **Guardrails.** Strict grounding, prompt-injection defense, scope control and multilingual answers.
- **Graceful failure.** Every step has a fallback, so one broken part doesn't take the chat down.
- **Low latency.** Measured on the deployed app (AWS EC2): the first word arrives in about 0.3s for small talk, about 1.2–1.6s for document questions and about 2.0–2.3s for web questions. Full answers finish in about 0.5s, 1.7–2.5s and 2.8–3.2s.
- **Observability.** LangSmith traces, per-question latency logs, cache stats and a startup warm-up.
- **Golden-dataset evaluation.** Six golden datasets (100 test cases) drive a DeepEval suite of 10 evals that covers retrieval, generation, the full pipeline, answer quality, safety (toxicity, prompt leakage, scope, PII) and operations (latency, cost, reliability). Every run is compared to a saved baseline and gets a PASS, REVIEW or FAIL verdict. See [Evaluation](#evaluation).
- **Tested.** 152 unit tests that run without Weaviate or real API keys.
- **CI/CD.** GitHub Actions runs the tests, builds a Docker image and deploys it to AWS EC2.

## Architecture

```mermaid
flowchart LR
    UI[Chat page<br/>HTML + JS] -- POST /ingest --> ING[Ingestion<br/>load, clean, chunk]
    UI -- WebSocket /ws/chat --> API[FastAPI]
    ING --> EMB[Embeddings<br/>OpenAI]
    EMB --> WV[(Weaviate<br/>hybrid index)]
    API --> G[LangGraph pipeline]
    G --> WV
    G --> WEB[Tavily<br/>web search tool]
    G --> GW[LiteLLM gateway]
    GW --> OAI[OpenAI]
    GW -. fallback .-> OR[OpenRouter<br/>Llama / Gemini]
    API <--> R[(Redis<br/>cache + budgets)]
    API <--> PG[(Postgres on AWS RDS<br/>chat memory)]
```

### How a question is answered

```mermaid
flowchart LR
    Q[Question] --> ST{Small talk?}
    ST -- yes --> C[Reply directly]
    ST -- no --> CL[Classify<br/>DOC / WEB / None]
    ST -- no --> RS[Search documents<br/>with the question]
    CL --> D{Decide}
    RS --> D
    D -- None --> C
    D -- DOC, standalone --> A[Answer from passages]
    D -- DOC, follow-up --> S[Search with rewritten query] --> A
    D -- WEB --> W[web_search tool] --> WA[Answer from web results]
```

1. Bare greetings, thanks and goodbyes ("hey", "thanks!", "ok") are answered directly, with no classifier, search or cache lookup.
2. Otherwise a small, fast model classifies the message, and a document search with the question as typed runs at the same time.
   - **DOC:** anything that could be in your documents, typos and unknown names included. When in doubt, the classifier picks DOC.
   - **WEB:** clearly current or live information that is not about your documents, such as news, today's weather or prices, or who holds a position (president, CEO).
   - **None:** small talk, questions about the bot, and stable general knowledge ("What is the capital of France?").
3. Standalone questions are only classified and searched as typed. Follow-ups that refer back ("and what about sick leave?", "how old is he?") are also rewritten from the chat history, for both document and web searches.
4. Document search is hybrid (keyword + vector), then the top results are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`. The chat model streams the answer from numbered passages.
5. Web questions go to Tavily. A separate prompt answers only from the web results, and the sources appear in the tool row rather than in the answer text.

The semantic cache is checked alongside the graph, not before it. The graph's output waits for the lookup. A hit stops the graph and streams the cached answer. A cached answer is only reused for the same documents version, model, prompt and language.

### Web search tool

| | |
|---|---|
| **Provider** | [Tavily](https://tavily.com/) (`tavily-python`), off unless `TAVILY_API_KEY` is set |
| **When it runs** | Only when the classifier labels a message `WEB`. Document questions never trigger it. |
| **In the chat page** | A `tool` WebSocket frame is sent when the search starts and when it ends. The page shows a spinner ("Searching the web… `web_search`"), then a collapsible row ("Searched the web · `web_search` · 3 results") with the query and source links. Links must be `http`/`https`, and titles are inserted as plain text. |
| **Answer** | Answers only from the results, without listing sources in the text (the tool row shows them). Web answers are never stored in the semantic cache. |
| **Failure** | No key, an error, a timeout or no results: the message is answered as plain conversation instead. |
| **Performance** | The Tavily client is built once and warmed up at startup. |

### LLM gateway and reliability

Every LLM call goes through a [LiteLLM](https://docs.litellm.ai/) router, which handles failures and limits in one place. Classifier and answer calls use separate model groups, because OpenAI limits are counted per model.

| Feature | How it works |
|---|---|
| **Automatic retries** | Timeouts, 5xx and "service unavailable" errors are retried once. Rate-limit errors are not retried; the call falls back instead. Query embeddings retry up to 2 times. |
| **Timeouts** | Each routed LLM call has a 20 s limit, and each web search 10 s. Query embeddings have a 2 s limit, and Redis connect and read calls 0.2 s each, so a slow dependency can't stall an answer. Idle WebSocket connections close after 10 minutes. |
| **Automatic fallback** | If OpenAI fails or its budget is used up, the call moves to OpenRouter: answers go to Llama 3.3 70B, classifier calls to Llama 3.1 8B, and then both to Gemini 2.5 Flash. |
| **Caching** | Redis caches query and document embeddings and answers to similar questions (semantic cache). OpenAI prompt caching serves about half of the input tokens. A circuit breaker skips Redis for 30 s after a failure, so the app keeps answering without the cache. |
| **Rate limiting** | Each OpenAI model has a per-minute budget of 4,000 requests (RPM) and 1.6M tokens (TPM), set at 80% of the account limit. Usage is counted in Redis, so the budget is shared across all app processes. The app also limits question length, upload size and the number of open WebSocket connections. |
| **Load balancing** | `usage-based-routing-v2` sends each call to the deployment with the most room left in its budget. With `LOAD_BALANCING_ENABLED`, the same OpenAI models are also reached through OpenRouter. In `overflow` mode the extra copies only take traffic when OpenAI is full; in `split` mode traffic is spread across both. |

> [!TIP]
> A retired model does not show up as an error: the router quietly falls back to OpenRouter, and answers get slower. If LangSmith shows an OpenAI model span with no token count, check that the model in `.env` still exists.

### Answer quality and guardrails

- **Strict grounding.** The answer prompt only allows facts written in the retrieved passages. If the answer is not there, the model gives a fixed "I don't have that specific information…" reply and says what the documents do cover. When nothing was retrieved at all, it gives only the first sentence and never guesses what the documents contain.
- **Partial and conflicting answers.** If the documents cover only part of a question, the model answers that part and says what is missing. If passages disagree, it reports each version.
- **Prompt-injection defense.** Documents, web results, chat history and user messages are marked as data, not instructions, in every prompt (classifier, rewrite, answer, web answer, conversation and title). The model is told never to reveal its instructions.
- **Scope control.** Stable general-knowledge questions and small talk go to a separate conversation prompt that never claims to know the documents. Current-events questions go to the web search tool. Unrelated tasks (essays, code, stories) are politely declined.
- **Typo-safe routing.** A misspelled or unknown name ("How do I configure CyberShieldX?") stays a document question and gets a "not found" reply, instead of leaking to a web search.
- **Search vs. rerank input.** A follow-up is searched with its rewritten query, which has synonyms and resolved references, so it finds more. Reranking scores passages against the user's original question, so the order matches what was actually asked.
- **Multilingual.** Answers come in the language chosen in the chat page (English, Hindi, Spanish, French or German). The rewritten query stays in the user's language.
- **Chat history window.** Only the last 10 messages are sent to the model, to keep prompts small.

### Semantic cache rules

A cached answer is reused only when it is safe:

- Only standalone questions are cached. A follow-up such as "and what about it?" depends on the chat, so it always runs the full pipeline.
- Only answers built from retrieved passages are stored. Small talk, web answers and "not found" replies are never cached.
- Each entry is tied to the document set's version, the language, the chat model, the embedding model and a hash of the answer prompt. Uploading new content changes the version, so old answers are dropped automatically. Changing a model or the prompt does the same.

### Failure handling

Each step has a fallback, so one failing part degrades the answer instead of breaking it.

| If this fails | The app does this |
|---|---|
| Classifier | Searches the documents (never the web) |
| Query rewrite (follow-ups) | Searches with the original question |
| Raw-question search | Waits for the classifier and searches again |
| Web search (no key, error, timeout, no results) | Answers as plain conversation |
| OpenAI (error or budget used up) | Falls back to Llama or Gemini through OpenRouter |
| Redis | Circuit breaker skips the cache for 30 s; answers still work |
| Weaviate connection drops | Reconnects automatically on the next call |
| Chat title (8 s limit) | Returns no title; the chat keeps its default name |
| Search or answer step | Sends a clear error code to the chat page instead of closing the connection |

Settings are checked at startup, for example overlap smaller than chunk size, keys present for the chosen provider, and Redis database 0 for the semantic cache. A bad config fails fast with a clear message.

### Ingestion details

- Files in one upload are processed in parallel. The size limit is checked while the file streams in, so an oversized file is rejected before it is fully stored.
- Text is cleaned (extra whitespace and stray characters removed), then split with a recursive splitter (1000 characters, 200 overlap).
- Each file gets a content hash. Re-uploading an unchanged file skips embedding, which saves time and cost.
- Each namespace is a Weaviate tenant, created automatically, so different document sets stay separate in one collection. Index and namespace names are sanitized.
- Files can also be loaded from a URL (for example a pre-signed S3 link). The loader blocks private and internal addresses and does not follow redirects, which protects against SSRF.

### Observability

- **LangSmith tracing** covers every request, with custom spans for hybrid search, reranking, the web search tool, the semantic cache lookup, file ingestion, chunking and Weaviate writes. Secrets such as pre-signed URLs are left out of traces.
- **Logs** record the routing decision (`Small talk: classifier skipped`, `Transformed query: …`, `Transformed query: web search for …`), every tool call (`Web search tool called: …` and its result count), the number of passages used, time to first token and total response time for every question. Production logs are JSON.
- **Cache stats** (hits, misses, errors and Redis memory) are logged every 5 minutes.
- **Warm-up:** at startup the app opens network connections (DNS, TCP, TLS) and loads the chat, classifier and embedding models, Redis, Postgres, the reranker and the web search client, all in parallel. `/ready` returns 503 until this finishes, so the first user doesn't pay the cold-start cost.

### Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, WebSocket, Uvicorn |
| Orchestration | LangGraph, LangChain |
| LLMs | OpenAI (`gpt-4.1-mini` for answers, `gpt-4o-mini` for classification), OpenRouter fallback via LiteLLM |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Weaviate (hybrid search, one tenant per namespace) |
| Reranker | Sentence Transformers cross-encoder |
| Web search | Tavily |
| Cache | Redis Stack (vector search for the semantic cache) |
| Memory | Postgres on AWS RDS (LangGraph checkpointer) |
| Evaluation | DeepEval, LangSmith tracing |
| Deployment | Docker, Docker Compose, GitHub Actions, AWS EC2 |

## Evaluation

This project is measured against **golden datasets**, not just spot-checked. The golden datasets in [`goldens/`](goldens/) hold 100 test cases in 6 files. Each file targets one part of the system, and each has a known expected result to score against:

| Golden dataset | Cases | What it checks |
|---|---|---|
| `retriever_goldens.json` | 20 | Whether search finds the right passages (also used to test reranking) |
| `generator_goldens.json` | 18 | Whether the model answers faithfully from the given context (also used for the full pipeline eval) |
| `application_goldens.json` | 18 | End-to-end answer quality against an ideal answer |
| `toxicity_goldens.json` | 15 | Attack and benign inputs, to catch toxic output |
| `leakage_goldens.json` | 15 | Attempts to extract the system prompt or private data, plus benign cases |
| `scope_goldens.json` | 14 | Adversarial and benign requests, to check the bot stays in scope |

The suite runs **10 evals** (7 DeepEval quality and safety evals, and 3 operations evals for latency, cost and reliability) over these datasets. It saves each run as a snapshot and compares it to a saved baseline with per-metric tolerances. A drop in a safety score, a quality score, or a rise in latency or cost beyond its tolerance gives a **PASS, REVIEW or FAIL** verdict, so a change that makes the system worse shows up as a number, not a hunch.

Latest run (before the web search tool and the current models were added: `gpt-4o-mini`, chunk size 1000, overlap 200, top-k 5, reranking on):

| Area | Metric | Result |
|---|---|---|
| Retrieval | Contextual recall | 97.5% |
| | Contextual precision | 95.0% |
| Generation | Faithfulness | 99.4% |
| | Answer relevancy | 94.9% |
| Full pipeline | Faithfulness | 92.5% |
| | Answer relevancy | 92.6% |
| | Contextual relevancy | 52.0% (needs work) |
| Answer quality | Correctness / completeness / style | 90.5% / 85.8% / 90.5% |
| Safety | Toxicity | 97.3% |
| | System / prompt leakage | 100% / 100% |
| | Scope adherence | 100% |
| | PII leakage | 100% |
| Operations | p95 end-to-end latency | 2.5 s |
| | p95 time to first token | 1.5 s |
| | Cost per query | ~$0.0004 |
| | Success rate | 100% |

Run it yourself:

```bash
uv run python -m evals.run_suit --baseline    # save baselines/baseline.json
uv run python -m evals.run_suit               # save baselines/candidate.json
uv run python -m evals.compare                # PASS / REVIEW / FAIL
```

Use `--only retriever,latency` to run a subset.

> [!TIP]
> Turn the semantic cache off (`SEMANTIC_CACHE_ENABLED=false`) when measuring latency: similar test questions hit the cache, and a hit returns the whole answer at once.

## Getting started

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Docker
- An OpenAI API key (optional: Tavily for web search, OpenRouter and LangSmith keys)

### Run locally

```bash
git clone https://github.com/Varsha-1605/Question-Answering-Chatbot.git
cd Question-Answering-Chatbot

cp .env.example .env                          # add your API keys and RDS POSTGRES_URL
docker compose up -d weaviate redis
uv sync
uv run uvicorn main:app --reload
```

Open http://localhost:8000, upload a file from `data/`, and ask a question. Try a current-events question such as "Who is the CEO of Tesla?" to see the web search tool.

> [!NOTE]
> **No RDS database?** Run Postgres in Docker instead:
> ```bash
> docker run -d --name rag-postgres -p 5432:5432 \
>   -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
>   -v rag-postgres-data:/var/lib/postgresql/data postgres:16
> ```
> Then set `POSTGRES_URL=postgresql://rag:rag@localhost:5432/rag` in `.env`. The app creates its tables on first start. To skip Postgres entirely, leave `POSTGRES_URL` unset and the browser keeps the chat history.

> [!TIP]
> On Windows with `POSTGRES_URL` set, start the app with
> `uv run uvicorn main:app --loop asyncio:SelectorEventLoop`.

> [!NOTE]
> `--reload` restarts on Python changes only. After editing `.env`, stop the server and start it again.

### Tests

152 unit tests in `tests/` cover the API routes, WebSocket flow, LangGraph routing (documents, web search, small talk, classify-only and rewrite modes), the web search tool and its WebSocket tool events, the parallel cache lookup, reranker, semantic cache rules, chat titles, document loading, the ingestion pipeline, name sanitization and the follow-up and small-talk detectors. External services are mocked, so the tests run without Weaviate, Tavily or real API keys (CI uses a dummy `OPENAI_API_KEY`).

```bash
uv run pytest
```

### Command-line mode

You can try the pipeline without the web page:

```bash
uv run python -m ai_workflow.injestion.utils             # ingest data/transformer_paper.pdf
uv run python -m ai_workflow.retrieval.routes_with_graph # chat in the terminal, with timings and [tool] lines
```

### Configuration

All settings are in [`.env.example`](.env.example). The main ones:

| Setting | Purpose |
|---|---|
| `OPENAI_CHAT_MODEL`, `REWRITE_MODEL` | Answer model and the faster classifier / query-rewrite model (must be different models) |
| `REWRITE_STANDALONE_QUESTIONS` | `false` (default): standalone questions are only classified and searched as typed. `true`: they are also rewritten. Follow-ups are always rewritten. |
| `TAVILY_API_KEY` | Turns on the web search tool (empty = off) |
| `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_TIMEOUT_SECONDS` | Web results per search (default 5) and the time limit per search (default 10 s) |
| `RERANKER_ENABLED`, `RERANK_CANDIDATES` | Cross-encoder reranking |
| `LLM_FALLBACK_ENABLED` | Route calls through the LiteLLM gateway with OpenRouter fallback |
| `ANSWER_MODEL_RPM_BUDGET`, `ANSWER_MODEL_TPM_BUDGET` (and `REWRITE_*`) | Per-minute request and token budgets for each OpenAI model |
| `LOAD_BALANCING_ENABLED`, `LOAD_BALANCING_MODE` | Add OpenRouter copies of the OpenAI models (`overflow` or `split`) |
| `FALLBACK_TIMEOUT_SECONDS` | Time limit for each routed LLM call (default 20 s) |
| `CACHE_ENABLED`, `SEMANTIC_CACHE_ENABLED` | Redis caching and answer reuse |
| `POSTGRES_URL` | Chat memory in Postgres, e.g. an AWS RDS URL ending in `?sslmode=require` (leave unset to keep chats in the browser only) |

> [!IMPORTANT]
> Changing the embedding model needs a new index. Set a new `INDEX_NAME`, then upload your documents again.

### Deployment

Every push to `main` runs [`deploy.yml`](.github/workflows/deploy.yml). It runs the tests, builds the Docker image, pushes it to Docker Hub, then starts the app, Weaviate and Redis on an EC2 instance with [`docker-compose.prod.yml`](docker-compose.prod.yml). Chat memory is stored in AWS RDS Postgres, not in a container. The EC2 instance's security group (`launch-wizard-1`) must be allowed on port 5432 in the RDS security group, and the `ENV_FILE` secret must contain the RDS `POSTGRES_URL`.

The Docker image is a multi-stage `uv` build that runs as a non-root user. It has the reranker model built in, so the first start doesn't download it, and a health check on `/health`. The deploy step replaces the app container, waits for it to be healthy and fails the run if it isn't. It also clears old images first, so the server's disk doesn't fill up.

Required repository secrets: `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, and `ENV_FILE` (the full `.env` content). Include `TAVILY_API_KEY` in `ENV_FILE`, or web search stays off in production.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Upload files (multipart `files`) |
| `WS /ws/chat` | Send `{question, language?}`; receive `stream`, `tool`, `end` and `error` messages. Add `?thread_id=` to resume a chat |
| `GET /chat/threads/{thread_id}/messages` | Messages of a stored chat |
| `POST /chat/title` | Short title for a chat's first question |
| `GET /health`, `GET /ready` | Liveness; readiness (warm-up done and Weaviate reachable) |

```bash
curl -F "files=@data/transformer_paper.pdf" localhost:8000/ingest
```

A `tool` message reports a tool call, once when it starts and once when it ends:

```json
{"type": "tool", "message": "", "tool": {"name": "web_search", "status": "done",
 "input": "Who is the CEO of Tesla?", "sources": [{"title": "…", "url": "https://…"}]}}
```

`status` is `running`, `done` or `failed` (no results). Other messages carry `"tool": null`.

WebSocket error codes: `MAX_CONNECTIONS_REACHED`, `IDLE_TIMEOUT`, `INVALID_REQUEST`, `SEARCH_FAILED` and `ANSWER_FAILED`. Send `ping` to get `pong`. Without Postgres, the client sends `chat_context` (up to 50 messages) with each question.

> [!WARNING]
> There is no authentication. Add your own access control and rate limiting before exposing the app publicly.

## Project structure

```
ai_workflow/
├── injestion/        # upload route, loaders, chunking
├── retrieval/        # WebSocket route, LangGraph pipeline, prompts, classifier, reranker,
│                     # web search tool, chat memory
├── vector_db/        # Weaviate client, hybrid search, storage
├── cache/            # Redis client, embedding cache, semantic answer cache
└── utils.py          # LLM gateway and embedding factories
frontend/             # chat page (HTML, JS, CSS; Markdown and LaTeX rendering, tool-call rows)
evals/, goldens/      # evaluation suite and test questions
tests/                # unit tests
```

## Roadmap

- [x] Web search tool for recent information, with tool calls shown in the chat page
- [ ] More tools (for example a calculator) and letting the LLM pick between them
- [ ] Tavily news mode with a date range for "latest news" questions
- [ ] Llama models switchable with OpenAI for classification and answer generation
- [ ] Authentication and per-user document spaces
