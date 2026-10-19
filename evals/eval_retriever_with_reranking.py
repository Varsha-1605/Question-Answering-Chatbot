import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from app.retrieval.pre_grouding import transform_user_query_for_retrieval
from app.retrieval.reranker import RerankingRetriever
from app.settings import config_settings
from app.vector_db.utils import close_vector_db

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE


with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


async def retrieve_all():
    retriever = RerankingRetriever.from_settings()
    contexts = []
    for g in goldens:
        # The rewrite drives the search; the original question drives reranking.
        search_query = await transform_user_query_for_retrieval(question=g["query"])
        passages = await retriever.retrieve(
            question=g["query"],
            search_query=search_query or g["query"],
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
        )
        contexts.append([doc.page_content for doc, _score in passages])
    return contexts


try:
    retrieval_contexts = asyncio.run(retrieve_all())
finally:
    close_vector_db()

test_cases = [
    LLMTestCase(
        input=g["query"],
        expected_output=g["ideal_answer"],
        retrieval_context=context,
        actual_output="(generator not evaluated in this run)",
    )
    for g, context in zip(goldens, retrieval_contexts)
]


metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "rewrite+hybrid+rerank" if config_settings.RERANKER_ENABLED else "rewrite+hybrid",
        "llm_service": config_settings.LLM_SERVICE.value,
        "hybrid_alpha": config_settings.WEAVIATE_HYBRID_ALPHA,
        "rerank_candidates": config_settings.RERANK_CANDIDATES,
        "reranker_model": config_settings.RERANKER_MODEL,
        "top_k": config_settings.RETRIEVAL_TOP_K,
        "chunk_size": config_settings.CHUNK_SIZE,
        "chunk_overlap": config_settings.CHUNK_OVERLAP,
        "index/namespace": f"{INDEX_NAME}/{NAMESPACE}",
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)