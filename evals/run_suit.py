import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from app.settings import config_settings  # noqa: E402
from evals.collect_metrics import ALL_EVALS  # noqa: E402
from evals.metric_registry import rule_for  # noqa: E402

BASELINE_PATH = Path("baselines/baseline.json")
CANDIDATE_PATH = Path("baselines/candidate.json")
WORK_DIR = Path("baselines/tmp")
LOG_DIR = Path("baselines/logs")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prompt_hashes() -> dict:
    """Short hashes of each prompt, to detect edits."""
    from app.retrieval import prompts

    return {
        "answer": _hash(prompts.initialise_doc_search_prompt_template().template),
        "rewrite": _hash(prompts.initialise_rewrite_prompt_template().template),
        "conversation": _hash(prompts.initialise_conversation_prompt_template().template),
    }


def pipeline_config() -> dict:
    """Settings that affect results."""
    s = config_settings
    return {
        "llm_service": s.LLM_SERVICE.value,
        "chat_model": s.OPENAI_CHAT_MODEL,
        "rewrite_model": s.REWRITE_MODEL or s.OPENAI_CHAT_MODEL,
        "temperature": s.LLM_TEMPERATURE,
        "embedding_model": s.OPENAI_EMBEDDING_MODEL,
        "hybrid_alpha": s.WEAVIATE_HYBRID_ALPHA,
        "top_k": s.RETRIEVAL_TOP_K,
        "reranker_enabled": s.RERANKER_ENABLED,
        "reranker_model": s.RERANKER_MODEL,
        "rerank_candidates": s.RERANK_CANDIDATES,
        "chunk_size": s.CHUNK_SIZE,
        "chunk_overlap": s.CHUNK_OVERLAP,
        "llm_fallback_enabled": s.LLM_FALLBACK_ENABLED,
        "index/namespace": f"{s.INDEX_NAME}/{s.DEFAULT_NAMESPACE}",
    }


def build_metadata(label: str) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        # Counts only edits to tracked files, excluding baselines.
        "git_dirty": _git(
            "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)baselines"
        ) not in ("", "unknown"),
        "prompt_hashes": prompt_hashes(),
        "config": pipeline_config(),
    }


def run_one(name: str, quiet: bool) -> tuple[dict, str | None]:
    """Run one eval in a subprocess; returns (metrics, error)."""
    out = WORK_DIR / f"{name}.json"
    out.unlink(missing_ok=True)
    command = [sys.executable, "-m", "evals.collect_metrics", name, "--out", str(out)]

    if quiet:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    else:
        code = subprocess.call(command)

    if code != 0 or not out.exists():
        where = f" (see {LOG_DIR / f'{name}.log'})" if quiet else ""
        return {}, f"exit code {code}{where}"
    return json.loads(out.read_text(encoding="utf-8"))["metrics"], None


def run_suite(names: list[str], label: str, quiet: bool, full: bool) -> dict:
    started = time.perf_counter()
    metrics, failed, timings = {}, {}, {}

    for step, name in enumerate(names, start=1):
        print(f"\n[{step}/{len(names)}] {name} ...", flush=True)
        t0 = time.perf_counter()
        result, error = run_one(name, quiet)
        timings[name] = round(time.perf_counter() - t0, 1)
        if error:
            failed[name] = error
            print(f"    FAILED: {error}")
        else:
            metrics.update(result)
            print(f"    done in {timings[name]}s ({len(result)} metrics)")

    if not full:
        metrics = {mid: v for mid, v in metrics.items() if rule_for(mid)["kind"] != "info"}

    return {
        "metadata": {
            **build_metadata(label),
            "evals": names,
            "failed_evals": failed,
            "complete": not failed,
            "eval_seconds": timings,
            "suite_seconds": round(time.perf_counter() - started, 1),
            "full": full,
            "n_metrics": len(metrics),
        },
        "metrics": metrics,
    }


def print_snapshot(snapshot: dict) -> None:
    meta, metrics = snapshot["metadata"], snapshot["metrics"]
    print("\n" + "=" * 78)
    print("SNAPSHOT")
    print("=" * 78)
    print(f"label       : {meta['label'] or '(none)'}")
    print(f"created_at  : {meta['created_at']}")
    print(f"git         : {meta['git_sha']}{' (uncommitted changes)' if meta['git_dirty'] else ''}")
    print(f"prompts     : {meta['prompt_hashes']}")
    print(f"suite_time  : {meta['suite_seconds']}s     metrics: {len(metrics)}")
    if meta["failed_evals"]:
        print(f"INCOMPLETE  : failed evals -> {meta['failed_evals']}")
    print("-" * 78)
    for key in sorted(metrics):
        value = metrics[key]
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:<56} {shown}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval suite and write a snapshot.")
    parser.add_argument("--baseline", action="store_true", help=f"write to {BASELINE_PATH}")
    parser.add_argument("--out", default=None, help="custom output path")
    parser.add_argument("--label", default="", help="what change this snapshot represents")
    parser.add_argument("--only", default="", help=f"comma-separated subset of: {','.join(ALL_EVALS)}")
    parser.add_argument("--quiet", action="store_true", help="send each eval's output to baselines/logs/")
    parser.add_argument("--full", action="store_true", help="also keep info metrics")
    args = parser.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or ALL_EVALS
    unknown = sorted(set(names) - set(ALL_EVALS))
    if unknown:
        parser.error(f"unknown evals {unknown}; choose from {ALL_EVALS}")

    snapshot = run_suite(names, args.label, args.quiet, args.full)
    print_snapshot(snapshot)

    out = Path(args.out) if args.out else (BASELINE_PATH if args.baseline else CANDIDATE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote snapshot -> {out}")
    if snapshot["metadata"]["failed_evals"]:
        sys.exit(1)


if __name__ == "__main__":
    main()