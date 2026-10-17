import asyncio
import math
import time

from dotenv import load_dotenv

load_dotenv()

from ai_workflow.retrieval.graph import rag_graph
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.utils import close_vector_db
from ai_workflow.warm_up import warm_up

import warnings 
warnings.filterwarnings("ignore")

# Questions on the Transformer paper.
EASY_QUESTIONS = [
    "What is the Transformer architecture based on instead of recurrence?",
    "What BLEU score did the Transformer get on English-to-German translation?",
    "How many layers are in the encoder and decoder stacks?",
    "Which optimizer was used to train the Transformer?",
    "What is the value of d_model in the base model?",
    "How many attention heads does the base model use?",
    "Which tasks was the Transformer evaluated on?",
]

MEDIUM_QUESTIONS = [
    "What are the two sub-layers in each encoder layer?",
    "Why is masking used in the decoder's self-attention?",
    "How does the learning rate change during training?",
    "Which regularization techniques were used during training?",
    "What is the difference between additive attention and dot-product attention?",
    "Why did the authors choose sinusoidal positional encodings over learned ones?",
    "How is the position-wise feed-forward network structured?",
]

HARD_QUESTIONS = [
    "When is a self-attention layer faster than a recurrent layer, and why?",
    "How does the maximum path length between positions compare for self-attention, recurrent and convolutional layers?",
    "What are the three different ways multi-head attention is used in the Transformer?",
    "Why does label smoothing hurt perplexity but improve BLEU?",
    "How was the Transformer adapted for English constituency parsing?",
    "How does multi-head attention keep its computational cost similar to single-head attention?",
]

QUESTIONS = EASY_QUESTIONS + MEDIUM_QUESTIONS + HARD_QUESTIONS
WARMUP_QUESTION = "What is the Transformer?"

REPEATS = 1
WARMUP_RUNS = 2

# p95 latency targets
SLO_E2E_P95_MS = 3000
SLO_TTFT_P95_MS = 1500

INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE
ANSWER_NODES = {"answer", "converse"}


async def timed_run(question: str) -> dict:
    """Stream the graph once and time each stage (ms)."""
    state = {
        "question": question,
        "chat_history": "",
        "language": "English",
        "index_name": INDEX_NAME,
        "namespace": NAMESPACE,
    }
    start = time.perf_counter()
    node_done: dict[str, float] = {}
    next_step = None
    first_token = None
    answer_parts: list[str] = []

    async for mode, payload in rag_graph.astream(state, stream_mode=["messages", "updates"]):
        now = time.perf_counter()
        if mode == "updates":
            for node, update in payload.items():
                node_done[node] = now
                if node == "decide":
                    next_step = update["next_step"]
        else:
            chunk, metadata = payload
            if metadata.get("langgraph_node") in ANSWER_NODES and chunk.content:
                if first_token is None:
                    first_token = now
                answer_parts.append(chunk.content)

    end = time.perf_counter()
    decide_done = node_done["decide"]
    second_search_done = node_done.get("search")
    generation_start = second_search_done or decide_done

    def ms(a, b):
        return (b - a) * 1000

    return {
        "routed_to": next_step,
        "rewrite": ms(start, node_done["transform_query"]),
        "raw_search": ms(start, node_done["raw_search"]),
        "pre_gen": ms(start, decide_done),
        "second_search": ms(decide_done, second_search_done) if second_search_done else float("nan"),
        "generation": ms(generation_start, end),
        "ttft": ms(start, first_token) if first_token else float("nan"),
        "total": ms(start, end),
        "answer_len": len("".join(answer_parts)),
    }


def percentile(values, p):
    """The p-th percentile, with linear interpolation."""
    s = sorted(v for v in values if not math.isnan(v))
    if not s:
        return float("nan")
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


def summarize(values):
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return None
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


async def benchmark() -> list[dict]:
    print("Warming up (warm_up() + discarded runs)...")
    await warm_up()
    for _ in range(WARMUP_RUNS):
        await timed_run(WARMUP_QUESTION)

    print("Measuring...")
    runs = []
    for question in QUESTIONS:
        for _ in range(REPEATS):
            runs.append(await timed_run(question))
    return runs


def report(runs: list[dict]) -> None:
    print("\n" + "=" * 86)
    print("LATENCY (milliseconds)")
    print("=" * 86)
    print(f"{'stage':<12} | {'n':>3} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'min':>9} {'max':>9}")
    print("-" * 86)
    stats = {}
    labels = {"total": "end-to-end", "raw_search": "raw search", "pre_gen": "pre-gen",
              "second_search": "2nd search"}
    for stage in ("total", "ttft", "rewrite", "raw_search", "pre_gen", "second_search", "generation"):
        s = summarize([r[stage] for r in runs])
        stats[stage] = s
        label = labels.get(stage, stage)
        if s is None:
            print(f"{label:<12} | no samples")
            continue
        print(f"{label:<12} | {s['n']:>3} {s['mean']:9.0f} {s['p50']:9.0f} {s['p95']:9.0f} "
              f"{s['p99']:9.0f} {s['min']:9.0f} {s['max']:9.0f}")

    print("-" * 86)
    routes = {step: sum(1 for r in runs if r["routed_to"] == step) for step in ("answer", "search", "converse")}
    print(f"routes: {routes['answer']} answered from the raw search, "
          f"{routes['search']} searched again, {routes['converse']} conversation")
    conversational = routes["converse"]
    avg_len = sum(r["answer_len"] for r in runs) / len(runs)
    print(f"avg answer length: {avg_len:.0f} chars (latency grows with answer length)")
    if conversational:
        print(f"WARNING: {conversational} run(s) were routed to conversation (no search); "
              "they skew the totals.")

    print("=" * 86)
    for label, stat, budget in (
        ("full answer", stats["total"], SLO_E2E_P95_MS),
        ("first token (perceived)", stats["ttft"], SLO_TTFT_P95_MS),
    ):
        if stat is None:
            print(f"SLO: {label:<24} no samples")
            continue
        verdict = "PASS" if stat["p95"] <= budget else "FAIL"
        print(f"SLO: {label:<24} p95 <= {budget:>5} ms  ->  p95 = {stat['p95']:7.0f} ms   [{verdict}]")
    print("=" * 86)
    print(f"config: llm={config_settings.LLM_SERVICE.value} chat={config_settings.OPENAI_CHAT_MODEL} "
          f"rewrite={config_settings.REWRITE_MODEL or config_settings.OPENAI_CHAT_MODEL} "
          f"reranker={'on' if config_settings.RERANKER_ENABLED else 'off'} "
          f"candidates={config_settings.RERANK_CANDIDATES} top_k={config_settings.RETRIEVAL_TOP_K}")


async def main():
    try:
        runs = await benchmark()
    finally:
        close_vector_db()
    report(runs)


if __name__ == "__main__":
    asyncio.run(main())
