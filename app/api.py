from __future__ import annotations
import uuid
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import DOCS_DIR, INDEX_DIR
from .vectorstore import load_or_build_index, build_faiss_from_docs
from .pipeline import rewrite_chain, make_rag_chain, followup_chain
from .schemas import IngestResponse, ChatRequest, ChatResponse, TransformedQuery

# Globals (lazy initialized)
_vectorstore = None
_rag_with_memory = None
_retriever = None


def _ensure_index():
    global _vectorstore, _rag_with_memory, _retriever
    if _vectorstore is None:
        _vectorstore = load_or_build_index(DOCS_DIR, INDEX_DIR)
        _rag_with_memory, _retriever = make_rag_chain(_vectorstore)


def create_app() -> FastAPI:
    app = FastAPI(title="Sri Lanka Legal LLM — RAG (Gemini + FAISS)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest():
        global _vectorstore, _rag_with_memory, _retriever
        _vectorstore = build_faiss_from_docs(DOCS_DIR, INDEX_DIR)
        _rag_with_memory, _retriever = make_rag_chain(_vectorstore)
        num_chunks = _vectorstore.index.ntotal if hasattr(
            _vectorstore, "index") else 0
        return IngestResponse(built=True, chunks=num_chunks)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        _ensure_index()
        session_id = req.session_id or str(uuid.uuid4())

        # Stage 1: structured query rewrite
        tq: TransformedQuery = rewrite_chain.invoke(
            {"user_question": req.message})
        if not tq.is_legal:
            answer = followup_chain.invoke({"user_message": req.message})
            return ChatResponse(session_id=session_id, is_legal=False, answer=answer, citations=[], used_k=0)

        # Stage 2: RAG answer using refined query
        candidates = [q for q in tq.transformed_queries if q.strip()]
        refined = candidates[0] if candidates else req.message

        top_docs = _retriever.invoke(refined)
        answer_text = _rag_with_memory.invoke(
            {"question": refined},
            config={"configurable": {"session_id": session_id}},
        )

        cites: List[str] = []
        seen = set()
        for d in top_docs:
            src = d.metadata.get("source")
            if src and src not in seen:
                cites.append(src)
                seen.add(src)

        return ChatResponse(session_id=session_id, is_legal=True, answer=answer_text, citations=cites, used_k=len(top_docs))

    @app.get("/")
    def root():
        return {"ok": True}

    return app
