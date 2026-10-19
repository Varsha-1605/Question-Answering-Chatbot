import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.retrieval.graph import rag_graph
from app.settings import config_settings
from app.vector_db.utils import close_vector_db

GOLDEN_PATH = "goldens/mm_application_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7
LANGUAGE = "English"

INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE


# Load questions and ideal answers
with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


# Run the full pipeline per question
async def answer_all():
    answers = []
    for g in goldens:
        state = await rag_graph.ainvoke(
            {
                "question": g["question"],
                "chat_history": "",
                "language": LANGUAGE,
                "index_name": INDEX_NAME,
                "namespace": NAMESPACE,
            }
        )
        answers.append(state["answer"])
    return answers


try:
    answers = asyncio.run(answer_all())
finally:
    close_vector_db()

test_cases = [
    LLMTestCase(
        input=g["question"],
        actual_output=answer,
        expected_output=g["ideal_answer"],
    )
    for g, answer in zip(goldens, answers)
]


# Correctness: judges truth against the expected output
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "Pay special attention to numbers, formulas and units (e.g. 10^-9 vs 10^9, d_model = 512, 8 heads): a wrong number is a factual error.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
        Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
        Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

# Completeness: judges coverage of the expected output
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

# Style: judges tone and structure, no expected output
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward answers that sound natural and confident, like someone who knows the material well, and are easy to understand.",
        "Penalize answers that open with filler such as 'According to the provided documents' or 'Based on the documents'.",
        "Penalize answers that mention source file names or page numbers, since the question did not ask where the information comes from.",
        "Reward clear structure: concise paragraphs, or a list or steps when they genuinely make the answer clearer. Do NOT penalize a list that helps clarity.",
        "Penalize answers that are padded, repetitive, robotic, or use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length --- only on style and tone.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Natural, confident and clear; no filler openers or unrequested citations; structure helps the reader."),
        Rubric(score_range=(7, 8),  expected_outcome="Clear and well organized, with at most one small style slip."),
        Rubric(score_range=(4, 6),  expected_outcome="Understandable but stiff, padded, or opens with filler / cites sources unprompted."),
        Rubric(score_range=(0, 3),  expected_outcome="Robotic, confusing, repetitive or badly structured."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)


evaluate(
    test_cases=test_cases,
    metrics=[correctness, completeness, style],
    async_config=AsyncConfig(max_concurrent=3, throttle_value=1),
    error_config=ErrorConfig(ignore_errors=True),
    hyperparameters={
        "pipeline": "rewrite+hybrid+rerank+answer" if config_settings.RERANKER_ENABLED else "rewrite+hybrid+answer",
        "llm_service": config_settings.LLM_SERVICE.value,
        "chat_model": config_settings.OPENAI_CHAT_MODEL,
        "temperature": config_settings.LLM_TEMPERATURE,
        "top_k": config_settings.RETRIEVAL_TOP_K,
        "chunk_size": config_settings.CHUNK_SIZE,
        "chunk_overlap": config_settings.CHUNK_OVERLAP,
        "index/namespace": f"{INDEX_NAME}/{NAMESPACE}",
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)