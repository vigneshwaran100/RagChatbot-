from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float = Field(..., description="Cosine similarity score (0-1)")
    excerpt: str = Field(..., description="Raw text of the retrieved chunk")
    page_number: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str | None = Field(
        default=None, description="Override default collection name"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    is_answerable: bool = Field(
        ..., description="False when documents don't contain enough info"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer",
    )
    reformulated_query: str | None = Field(
        default=None, description="Query after LLM reformulation"
    )
    provider_used: str = Field(..., description="LLM provider that generated the answer")
