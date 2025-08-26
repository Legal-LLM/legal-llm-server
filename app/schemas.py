from typing import List, Optional
from pydantic import BaseModel, Field


class TransformedQuery(BaseModel):
    """LLM output used to route queries to legal RAG or a follow‑up."""
    is_legal: bool = Field(
        description="Whether it's about Sri Lankan business/corporate/tax/labor law")
    transformed_queries: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    built: bool
    chunks: int


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    is_legal: bool
    answer: str
    citations: List[str] = []
    used_k: int = 0
