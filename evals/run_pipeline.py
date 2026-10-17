import asyncio

from ai_workflow.retrieval.graph import rag_graph
from ai_workflow.settings import config_settings
from ai_workflow.vector_db.utils import close_vector_db

LANGUAGE = "English"
INDEX_NAME = config_settings.INDEX_NAME
NAMESPACE = config_settings.DEFAULT_NAMESPACE


async def _answer_all(questions: list[str]) -> list[str]:
    answers = []
    for question in questions:
        state = await rag_graph.ainvoke(
            {
                "question": question,
                "chat_history": "",
                "language": LANGUAGE,
                "index_name": INDEX_NAME,
                "namespace": NAMESPACE,
            }
        )
        answers.append(state["answer"])
    return answers


def answer_questions(questions: list[str]) -> list[str]:
    """Answer each question, in order."""
    try:
        return asyncio.run(_answer_all(questions))
    finally:
        close_vector_db()