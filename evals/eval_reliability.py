import asyncio
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from ai_workflow.retrieval.graph import rag_graph
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.utils import close_vector_db

QUESTIONS = [
    "What is multi-head attention and why is it used?",
    "Why do we divide by the square root of dk in attention?",
    "How long did it take to train the Transformer and on what hardware?",
    "What is positional encoding and why is it needed?",
    "hi",
]
REPEATS = 5

MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5
REQUEST_TIMEOUT_S = 60

SLO_SUCCESS_RATE = 99.0

INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE


class EmptyAnswerError(RuntimeError):
    """The graph finished but produced no answer text."""


async def run_once(question: str) -> str:
    state = await asyncio.wait_for(
        rag_graph.ainvoke(
            {
                "question": question,
                "chat_history": "",
                "language": "English",
                "index_name": INDEX_NAME,
                "namespace": NAMESPACE,
            }
        ),
        timeout=REQUEST_TIMEOUT_S,
    )
    answer = (state.get("answer") or "").strip()
    if not answer:
        raise EmptyAnswerError("empty answer")
    return answer


async def call_with_retries(question: str) -> dict:
    """Run a question with retries; report success flags, retry count and error types."""
    errors = []
    for attempt in range(MAX_RETRIES + 1):
        try:
            await run_once(question)
            return {"first_try_ok": attempt == 0, "eventually_ok": True, "retries": attempt, "errors": errors}
        except Exception as error:  # noqa: BLE001
            errors.append("Timeout" if isinstance(error, asyncio.TimeoutError) else type(error).__name__)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_BASE_S * (2 ** attempt))
    print(f"FAILED after {MAX_RETRIES} retries: {question!r} -> {errors}")
    return {"first_try_ok": False, "eventually_ok": False, "retries": MAX_RETRIES, "errors": errors}


async def benchmark() -> list[dict]:
    print("Measuring reliability...")
    results = []
    for question in QUESTIONS:
        for _ in range(REPEATS):
            results.append(await call_with_retries(question))
    return results


def report(results: list[dict]) -> None:
    n = len(results)
    first_try = sum(r["first_try_ok"] for r in results)
    eventually = sum(r["eventually_ok"] for r in results)
    retried = sum(1 for r in results if r["retries"] > 0)
    error_types = Counter(e for r in results for e in r["errors"])

    first_try_rate = 100 * first_try / n
    print("\n" + "=" * 62)
    print("RELIABILITY")
    print("=" * 62)
    print(f"requests               : {n}")
    print(f"first-try success      : {first_try:>4}  ({first_try_rate:6.2f}%)")
    print(f"success with retries   : {eventually:>4}  ({100 * eventually / n:6.2f}%)")
    print(f"hard failures          : {n - eventually:>4}  ({100 * (n - eventually) / n:6.2f}%)")
    print(f"requests that retried  : {retried:>4}  ({100 * retried / n:6.2f}%)")
    print("-" * 62)
    if error_types:
        print("errors by type (all attempts):")
        for name, count in error_types.most_common():
            print(f"   {name:<30} {count}")
    else:
        print("errors by type         : none")
    print("=" * 62)
    verdict = "PASS" if first_try_rate >= SLO_SUCCESS_RATE else "FAIL"
    print(f"SLO: first-try success >= {SLO_SUCCESS_RATE}%  ->  {first_try_rate:.2f}%   [{verdict}]")
    print("=" * 62)


async def main():
    try:
        results = await benchmark()
    finally:
        close_vector_db()
    report(results)


if __name__ == "__main__":
    asyncio.run(main())