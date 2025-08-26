from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_community.vectorstores.faiss import FAISS

from .config import get_query_llm, get_answer_llm
from .prompts import rewrite_prompt, answer_prompt, followup_prompt
from .schemas import TransformedQuery

# --- LLMs ---
query_llm = get_query_llm()
answer_llm = get_answer_llm()
structured_query_llm = query_llm.with_structured_output(TransformedQuery)

rewrite_chain = rewrite_prompt | structured_query_llm
parse_answer = StrOutputParser()

_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    hist = _session_histories.get(session_id)
    if not hist:
        hist = InMemoryChatMessageHistory()
        _session_histories[session_id] = hist
    return hist


def _format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"[SOURCE: {src}] {d.page_content}")
    return "---".join(parts)


def make_rag_chain(vs: FAISS):
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={
                                "k": 6, "fetch_k": 12, "lambda_mult": 0.6})

    def _retrieve(inputs: Dict[str, Any]):
        q = inputs["question"]
        # Return a *list of Documents* directly (so RunnableParallel assigns it to the "docs" key)
        # Using the new Retriever API to avoid deprecation warnings
        return retriever.invoke(q)

    retrieve = RunnableLambda(_retrieve)

    def _prep(inputs: Dict[str, Any]):
        docs: List[Document] = inputs["docs"]
        q: str = inputs["question"]
        return {"question": q, "context": _format_context(docs)}

    prep = RunnableLambda(_prep)

    rag = (
        RunnableParallel({"question": RunnableLambda(
            lambda x: x["question"]), "docs": retrieve})
        | prep
        | answer_prompt
        | answer_llm
        | parse_answer
    )

    # Wrap with memory; caller will pass session_id via config
    rag_with_memory = RunnableWithMessageHistory(
        rag,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return rag_with_memory, retriever


followup_chain = followup_prompt | answer_llm | parse_answer
