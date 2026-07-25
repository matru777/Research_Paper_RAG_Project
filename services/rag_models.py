from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import MessagesState

class PipelineState(MessagesState):
    """Clean state for the RAG pipeline graph."""
    session_id: str
    route: Literal["qa", "verification"]
    rewritten_query: str
    metadata_filter: dict
    needs_web_search: bool
    expanded_queries: list[str]
    retrieved_docs: list[Document]
    is_relevant: bool
    generated_answer: str
    is_grounded: bool
    grounding_feedback: str
    hallucination_retries: int
    extracted_claims: list[str]
    web_evidence: str
    local_evidence: str
    cache_hit: bool

class MetadataFilter(BaseModel):
    year: int | None = Field(default=None, description="The publication year to filter by, if mentioned.")
    author: str | None = Field(default=None, description="The author name to filter by, if mentioned.")

class QueryRewrite(BaseModel):
    rewritten_query: str = Field(
        description="The user's query rewritten to be perfectly standalone and highly specific, incorporating conversation history."
    )
    metadata_filter: MetadataFilter | None = Field(
        default=None,
        description="Optional metadata filters. Output null if no filter is specified."
    )
    needs_web_search: bool = Field(
        default=False,
        description="Set to True ONLY if the user explicitly asks for recent, current, or external information that requires a live Google Web Search. If the query is strictly about the uploaded paper, set to False."
    )

class RouteDecision(BaseModel):
    route: Literal["qa", "verification"] = Field(
        description="'qa' for questions about the papers. 'verification' for fact-checking claims."
    )

class MultiQueryExpansion(BaseModel):
    queries: list[str] = Field(
        min_length=3, max_length=3,
        description="3 distinct semantic search queries covering different angles."
    )

class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(
        description="True if the retrieved documents contain information relevant to the query."
    )
    reasoning: str

class GroundingCheck(BaseModel):
    is_grounded: bool = Field(
        description="True if every claim in the answer is supported by the provided context."
    )
    feedback: str = Field(
        description="If ungrounded, list the specific sentences that lack evidence."
    )

class ClaimExtraction(BaseModel):
    claims: list[str] = Field(
        description="Individual factual claims extracted from the user's query."
    )
