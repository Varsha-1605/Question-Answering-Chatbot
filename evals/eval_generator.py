import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document

from ai_workflow.retrieval.graph import answer
from ai_workflow.settings import config_settings

GOLDEN_PATH = "goldens/generator_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7
LANGUAGE = "English"


with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


def to_passages(golden):
    """Convert golden context to (Document, score) pairs."""
    return [
        (Document(page_content=c["text"], metadata={"file_name": golden["source"], "page": c["page"]}), 1.0)
        for c in golden["ideal_context"]
    ]


async def generate_all():
    answers = []
    for g in goldens:
        state = {
            "question": g["query"],
            "chat_history": "",
            "language": LANGUAGE,
            "passages": to_passages(g),
        }
        result = await answer(state)
        answers.append(result["answer"])
    return answers


generated = asyncio.run(generate_all())

test_cases = [
    LLMTestCase(
        input=g["query"],
        actual_output=output,
        retrieval_context=[c["text"] for c in g["ideal_context"]],
    )
    for g, output in zip(goldens, generated)
]


metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "component": "generator (golden context)",
        "llm_service": config_settings.LLM_SERVICE.value,
        "chat_model": config_settings.OPENAI_CHAT_MODEL,
        "temperature": config_settings.LLM_TEMPERATURE,
        "language": LANGUAGE,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)