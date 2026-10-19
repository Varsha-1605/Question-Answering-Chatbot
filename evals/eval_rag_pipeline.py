import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from app.retrieval.graph import rag_graph
from app.settings import config_settings
from app.vector_db.utils import close_vector_db

GOLDEN_PATH = "goldens/generator_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7
LANGUAGE = "English"

INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE


with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


async def run_all():
    results = []
    for g in goldens:
        state = await rag_graph.ainvoke(
            {
                "question": g["query"],
                "chat_history": "",
                "language": LANGUAGE,
                "index_name": INDEX_NAME,
                "namespace": NAMESPACE,
            }
        )
        # "passages" is missing for small talk.
        passages = state.get("passages", [])
        results.append(
            {
                "answer": state["answer"],
                "context": [doc.page_content for doc, _score in passages],
            }
        )
    return results


try:
    outputs = asyncio.run(run_all())
finally:
    close_vector_db()

test_cases = [
    LLMTestCase(
        input=g["query"],
        actual_output=out["answer"],
        retrieval_context=out["context"],
    )
    for g, out in zip(goldens, outputs)
]


metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=3, throttle_value=1),
    hyperparameters={
        "pipeline": "rewrite+hybrid+rerank+answer" if config_settings.RERANKER_ENABLED else "rewrite+hybrid+answer",
        "llm_service": config_settings.LLM_SERVICE.value,
        "chat_model": config_settings.OPENAI_CHAT_MODEL,
        "hybrid_alpha": config_settings.WEAVIATE_HYBRID_ALPHA,
        "rerank_candidates": config_settings.RERANK_CANDIDATES,
        "top_k": config_settings.RETRIEVAL_TOP_K,
        "chunk_size": config_settings.CHUNK_SIZE,
        "chunk_overlap": config_settings.CHUNK_OVERLAP,
        "index/namespace": f"{INDEX_NAME}/{NAMESPACE}",
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)