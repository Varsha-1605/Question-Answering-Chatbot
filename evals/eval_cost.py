import asyncio
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

import litellm
from langchain_core.callbacks import get_usage_metadata_callback
from litellm.integrations.custom_logger import CustomLogger

from ai_workflow.retrieval.graph import rag_graph
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.utils import close_vector_db

QUESTIONS = [
    "What is multi-head attention and why is it used?",
    "Why do we divide by the square root of dk in attention?",
    "How long did it take to train the Transformer and on what hardware?",
    "What is positional encoding and why is it needed?",
]
REPEATS = 3

# USD per 1M tokens as (input, cached input, output); matched by longest
# model-name prefix. Prices change, so re-check them.
PRICES_PER_1M = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    # OpenRouter fallbacks
    "meta-llama/llama-3.3-70b-instruct": (0.13, 0.13, 0.40),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.02, 0.05),
    "google/gemini-2.5-flash": (0.30, 0.075, 2.50),
}

QUERIES_PER_DAY = 2000
USD_TO_INR = 94.98

COST_BUDGET_PER_QUERY_USD = 0.0015

INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE


def price_for(model_name: str) -> tuple[float, float, float]:
    # Strip provider prefixes until a price matches.
    name = model_name
    while True:
        matches = [prefix for prefix in PRICES_PER_1M if name.startswith(prefix)]
        if matches:
            return PRICES_PER_1M[max(matches, key=len)]
        if "/" not in name:
            raise KeyError(f"No price configured for model {model_name!r}; add it to PRICES_PER_1M")
        name = name.split("/", 1)[1]


def cost_usd(model_name: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    price_in, price_cached, price_out = price_for(model_name)
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * price_in + cached_tokens * price_cached + output_tokens * price_out) / 1_000_000


class LiteLLMUsage(CustomLogger):
    """Records token usage of every LiteLLM call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int, int, int]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        log = kwargs.get("standard_logging_object") or {}
        usage = getattr(response_obj, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        self.calls.append((
            log.get("model") or kwargs.get("model", "unknown"),
            log.get("prompt_tokens") or 0,
            getattr(details, "cached_tokens", 0) or 0,
            log.get("completion_tokens") or 0,
        ))


litellm_usage = LiteLLMUsage()
litellm.callbacks.append(litellm_usage)


def add_usage(per_model: dict, model_name: str, input_tokens: int, cached: int, output: int) -> None:
    usage = per_model.setdefault(model_name, {"input": 0, "cached": 0, "output": 0, "cost": 0.0})
    usage["input"] += input_tokens
    usage["cached"] += cached
    usage["output"] += output
    usage["cost"] += cost_usd(model_name, input_tokens, cached, output)


async def measure(question: str) -> dict:
    """Run the graph once and return token usage and cost per model."""
    state = {
        "question": question,
        "chat_history": "",
        "language": "English",
        "index_name": INDEX_NAME,
        "namespace": NAMESPACE,
    }
    litellm_usage.calls.clear()
    with get_usage_metadata_callback() as usage_callback:
        await rag_graph.ainvoke(state)

    per_model: dict = {}
    if config_settings.LLM_FALLBACK_ENABLED:
        for model_name, input_tokens, cached, output in litellm_usage.calls:
            add_usage(per_model, model_name, input_tokens, cached, output)
    else:
        for model_name, usage in usage_callback.usage_metadata.items():
            cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
            add_usage(per_model, model_name, usage.get("input_tokens", 0), cached,
                      usage.get("output_tokens", 0))

    if not per_model:
        raise RuntimeError(f"No LLM token usage was recorded for {question!r}")
    return per_model


async def benchmark() -> list[dict]:
    print("Measuring token usage...")
    runs = []
    for question in QUESTIONS:
        for _ in range(REPEATS):
            runs.append(await measure(question))
    return runs


def report(runs: list[dict]) -> None:
    n = len(runs)
    totals = defaultdict(lambda: {"input": 0, "cached": 0, "output": 0, "cost": 0.0})
    for run in runs:
        for model_name, usage in run.items():
            for key in ("input", "cached", "output", "cost"):
                totals[model_name][key] += usage[key]

    per_query_costs = [sum(u["cost"] for u in run.values()) for run in runs]
    avg_cost = sum(per_query_costs) / n

    print("\n" + "=" * 94)
    print(f"COST  ({n} queries)")
    print("=" * 94)
    print(f"{'model':<44} {'avg in':>8} {'cached':>8} {'avg out':>8} {'avg $/query':>13} {'share':>7}")
    print("-" * 94)
    for model_name, t in sorted(totals.items(), key=lambda item: -item[1]["cost"]):
        share = 100 * (t["cost"] / n) / avg_cost if avg_cost else 0
        print(f"{model_name:<44} {t['input'] / n:8.0f} {t['cached'] / n:8.0f} {t['output'] / n:8.0f} "
              f"{t['cost'] / n:13.6f} {share:6.0f}%")
    print("-" * 94)
    print(f"avg cost / query : ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"   min / max     : ${min(per_query_costs):.6f} / ${max(per_query_costs):.6f}")

    daily = avg_cost * QUERIES_PER_DAY
    print(f"projection @ {QUERIES_PER_DAY}/day : ${daily:.2f}/day (Rs {daily * USD_TO_INR:.2f}), "
          f"${daily * 30:.2f}/month (Rs {daily * 30 * USD_TO_INR:.2f})")
    print("=" * 94)
    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  ${avg_cost:.6f}   [{verdict}]")
    print("=" * 94)
    print("not included: query embedding (negligible) and the local reranker (no API cost).")


async def main():
    try:
        runs = await benchmark()
    finally:
        close_vector_db()
    report(runs)


if __name__ == "__main__":
    asyncio.run(main())