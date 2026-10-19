import argparse
import asyncio
import importlib
import json
import math
import runpy
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import deepeval  # noqa: E402
from deepeval.evaluate.configs import CacheConfig  # noqa: E402

# eval name -> (module, metric namespace)
DEEPEVAL_EVALS = {
    "retriever": ("evals.eval_retriever", "retriever"),
    "generator": ("evals.eval_generator", "generator"),
    "pipeline": ("evals.eval_rag_pipeline", "pipeline"),
    "application": ("evals.eval_application", "application"),
    "toxicity": ("evals.eval_toxicity", "safety.toxicity"),
    "leakage": ("evals.eval_leakage", "safety.leakage"),
    "scope": ("evals.eval_safety_scope", "safety.scope"),
}
OPS_EVALS = {
    "latency": "evals.eval_latency",
    "cost": "evals.eval_cost",
    "reliability": "evals.eval_reliability",
}
ALL_EVALS = list(DEEPEVAL_EVALS) + list(OPS_EVALS)


def run_deepeval_file(module: str) -> list:
    results = []
    original = deepeval.evaluate

    def keep_result(*args, **kwargs):
        # DeepEval's write cache is not safe under parallel test cases (a locked cache
        # file makes a task get None and crash), and use_cache is off anyway.
        kwargs.setdefault("cache_config", CacheConfig(write_cache=False, use_cache=False))
        result = original(*args, **kwargs)
        results.append(result)
        return result

    deepeval.evaluate = keep_result
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        deepeval.evaluate = original
    return results


def summarize_by_metric(results: list) -> dict:
    """Summarize DeepEval results per metric (count, pass rate, avg/min/max score)."""
    buckets: dict = {}
    for result in results:
        test_results = getattr(result, "test_results", None) or []
        for test in test_results:
            for metric in getattr(test, "metrics_data", None) or []:
                bucket = buckets.setdefault(metric.name, {"scores": [], "passed": 0, "total": 0})
                bucket["total"] += 1
                if metric.score is not None:
                    bucket["scores"].append(metric.score)
                if metric.success:
                    bucket["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "scored": len(scores),
            "pass_rate": 100 * b["passed"] / b["total"] if b["total"] else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else math.nan,
            "min_score": min(scores) if scores else math.nan,
            "max_score": max(scores) if scores else math.nan,
        }
    return summary


def slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def flatten(namespace: str, summary: dict) -> dict:
    return {
        f"{namespace}.{slug(metric)}.{stat}": value
        for metric, stats in summary.items()
        for stat, value in stats.items()
    }


def latency_metrics(mod) -> dict:
    try:
        runs = asyncio.run(mod.benchmark())
    finally:
        mod.close_vector_db()
    mod.report(runs)

    def stat(stage: str, key: str) -> float:
        s = mod.summarize([r[stage] for r in runs])
        return s[key] if s else math.nan

    metrics = {
        "ops.latency.e2e_p95_ms": stat("total", "p95"),
        "ops.latency.ttft_p95_ms": stat("ttft", "p95"),
        "ops.latency.e2e_p50_ms": stat("total", "p50"),
        "ops.latency.ttft_p50_ms": stat("ttft", "p50"),
        "ops.latency.rewrite_p50_ms": stat("rewrite", "p50"),
        "ops.latency.raw_search_p50_ms": stat("raw_search", "p50"),
        "ops.latency.pre_gen_p50_ms": stat("pre_gen", "p50"),
        "ops.latency.generation_p50_ms": stat("generation", "p50"),
        "ops.latency.n": len(runs),
    }
    metrics["ops.latency.slo_e2e_pass"] = metrics["ops.latency.e2e_p95_ms"] <= mod.SLO_E2E_P95_MS
    metrics["ops.latency.slo_ttft_pass"] = metrics["ops.latency.ttft_p95_ms"] <= mod.SLO_TTFT_P95_MS
    return metrics


def cost_metrics(mod) -> dict:
    try:
        runs = asyncio.run(mod.benchmark())
    finally:
        mod.close_vector_db()
    mod.report(runs)

    per_query = [sum(usage["cost"] for usage in run.values()) for run in runs]
    avg_cost = sum(per_query) / len(per_query)

    def avg_tokens(key: str) -> float:
        return sum(usage[key] for run in runs for usage in run.values()) / len(runs)

    return {
        "ops.cost.cost_per_query_usd": avg_cost,
        "ops.cost.budget_pass": avg_cost <= mod.COST_BUDGET_PER_QUERY_USD,
        "ops.cost.input_tokens_per_query": avg_tokens("input"),
        "ops.cost.cached_tokens_per_query": avg_tokens("cached"),
        "ops.cost.output_tokens_per_query": avg_tokens("output"),
        "ops.cost.models_used": len({model for run in runs for model in run}),
    }


def reliability_metrics(mod) -> dict:
    try:
        results = asyncio.run(mod.benchmark())
    finally:
        mod.close_vector_db()
    mod.report(results)

    n = len(results)
    first_try = 100 * sum(r["first_try_ok"] for r in results) / n
    eventually = 100 * sum(r["eventually_ok"] for r in results) / n
    return {
        "ops.reliability.success_rate": first_try,
        "ops.reliability.error_rate": 100 - eventually,
        "ops.reliability.retried_rate": 100 * sum(r["retries"] > 0 for r in results) / n,
        "ops.reliability.slo_success_pass": first_try >= mod.SLO_SUCCESS_RATE,
        "ops.reliability.n": n,
    }


OPS_COLLECTORS = {"latency": latency_metrics, "cost": cost_metrics, "reliability": reliability_metrics}


def collect(name: str) -> dict:
    if name in DEEPEVAL_EVALS:
        module, namespace = DEEPEVAL_EVALS[name]
        results = run_deepeval_file(module)
        if not results:
            raise RuntimeError(f"{module} returned no DeepEval results")
        return flatten(namespace, summarize_by_metric(results))
    return OPS_COLLECTORS[name](importlib.import_module(OPS_EVALS[name]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one eval and save its metrics as JSON.")
    parser.add_argument("name", choices=ALL_EVALS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    metrics = collect(args.name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"metrics": metrics, "seconds": round(time.perf_counter() - started, 1)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()