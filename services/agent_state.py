from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import MessagesState

class RAGState(MessagesState):
    """
    The state that gets passed from node to node in LangGraph.
    MessagesState automatically handles the chat history (list of messages).
    """
    session_id: str
    route: str
    parent_docs: list[Document]
    is_relevant: bool
    is_grounded: bool
    hallucination_retries: int
    retrieval_attempts: int
    final_status: str
    answer: str

class RouterDecision(BaseModel):
    """LLM-structured output that decides how to handle an incoming query."""
    route: Literal["retrieve", "verify_claim", "direct_answer"] = Field(
        description=(
            "'retrieve' - answer requires fetching from the vector store; "
            "'verify_claim' - user wants to check whether a paper's claim still holds; "
            "'direct_answer' - conversational / no retrieval needed."
        )
    )

class MultiQueryExpansion(BaseModel):
    """Three semantically-varied reformulations of the user's original query."""
    queries: list[str] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 distinct search queries that cover different angles of the original question.",
    )

class RelevancyDecision(BaseModel):
    """Grades whether retrieved context is sufficient to answer the query."""
    is_relevant: bool = Field(
        description="True if the retrieved chunks adequately address the query."
    )
    reason: str = Field(
        description="One-sentence explanation of the relevancy verdict."
    )

class HallucinationDecision(BaseModel):
    """Grades whether the generated answer is grounded in the supplied context."""
    is_grounded: bool = Field(
        description="True if every factual claim in the answer is supported by the provided context."
    )
    reason: str = Field(
        description="One-sentence explanation of the grounding verdict."
    )

class SupersedingPaper(BaseModel):
    """Metadata for a paper that supersedes or challenges a claim."""
    title: str
    url: str
    summary: str = Field(description="One-sentence summary of how this paper supersedes the claim.")

class ClaimVerificationResult(BaseModel):
    """Result of checking whether a paper's claim is still current."""
    is_superseded: bool
    verdict_summary: str = Field(description="High-level verdict in one or two sentences.")
    superseding_papers: list[SupersedingPaper] = Field(default_factory=list)

class BtwRouteDecision(BaseModel):
    """LLM-structured output deciding if a general question needs the internet."""
    needs_web_search: bool = Field(
        description="True if the question requires recent news, current prices, or live web data."
    )