"""
rag_graph.py
────────────
Agentic CRAG (Corrective RAG) pipeline for Papeer, implemented as a
LangGraph StateGraph.

Node flow
─────────
cache_check → router ──┬─→ multi_query → async_retrieval → relevancy_check
                        │       └── (web_fallback if irrelevant) ──────────┐
                        ├─→ direct_answer                                   │
                        └─→ verify_claim                                    │
                                                        generate ←──────────┘
                                                           │
                                                    hallucination_check
                                                           │
                                                    (loop back or END)

LLMs
────
• llama-3.3-70b-versatile  – answer generation   (quality)
• llama3-8b-8192           – routing / grading    (speed)

Free-tier dependencies
──────────────────────
• langchain-groq      – LLM calls
• fastembed           – local embeddings  (see vector_store.py)
• flashrank           – local re-ranker
• tavily-python       – web fallback search
• langgraph[sqlite]   – checkpointer
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient
from typing_extensions import TypedDict

from models import (
    ClaimVerificationResult,
    HallucinationDecision,
    MultiQueryExpansion,
    RelevancyDecision,
    RouterDecision,
)
from paper_loader import fetch_parent
from vector_store import check_cache, save_to_cache, search

load_dotenv()

# ── LLMs ─────────────────────────────────────────────────────────────────────
_llm_fast = ChatGroq(model="llama3-8b-8192", temperature=0)
_llm_strong = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# ── Re-ranker ─────────────────────────────────────────────────────────────────

from flashrank import Ranker, RerankRequest  # type: ignore

_reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# ── Web search ────────────────────────────────────────────────────────────────

_tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
# ── State ─────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    # Core conversation
    messages: Annotated[list[Any], add_messages]
    session_id: str

    # Routing
    route: str                          # "retrieve" | "direct_answer" | "verify_claim"

    # Multi-query
    queries: list[str]                  # three expanded queries

    # Retrieval
    retrieved_docs: list[Document]      # child docs after vector search
    parent_docs: list[Document]         # full parent text (post-swap)

    # Grading
    is_relevant: bool
    is_grounded: bool

    # Final answer
    answer: str
    final_status: str                   # "cache_hit" | "generated" | "web_fallback" | "direct"

    # Hallucination retry counter
    hallucination_retries: int

# ── Helpers ───────────────────────────────────────────────────────────────────
def _last_user_message(state: RAGState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def _rerank_docs(query: str, docs: list[Document], top_k: int = 4) -> list[Document]:
    """Re-rank documents using FlashRank; return the top-k."""
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    results = _reranker.rerank(request)
    ranked_indices = [r["id"] for r in results[:top_k]]
    return [docs[i] for i in ranked_indices]

def _parent_swap(child_docs: list[Document], db_path: str = "doc_store.db") -> list[Document]:
    """
    Replace each child doc's page_content with its full parent text.
    Falls back to the child text if no parent_id is found.
    """
    swapped: list[Document] = []
    for doc in child_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_text = fetch_parent(parent_id, db_path)
            if parent_text:
                doc = Document(page_content=parent_text, metadata=doc.metadata)
        swapped.append(doc)
    return swapped

# ── Nodes ─────────────────────────────────────────────────────────────────────
def cache_check_node(state: RAGState) -> dict:
    """Check the semantic cache before doing any heavy lifting."""
    query = _last_user_message(state)
    cached_answer = check_cache(query, state["session_id"])
    if cached_answer:
        return {
            "answer": cached_answer,
            "final_status": "cache_hit",
            "messages": [AIMessage(content=cached_answer)],
        }
    return {"final_status": ""}

def router_node(state: RAGState) -> dict:
    """Classify the user's intent to decide which branch to take."""
    query = _last_user_message(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query router for a research paper assistant.\n"
                "Classify the user's message into exactly one of:\n"
                "  - 'retrieve'      – needs facts from the indexed papers\n"
                "  - 'verify_claim'  – user wants to check if a paper's claim is still current\n"
                "  - 'direct_answer' – conversational or general knowledge; no retrieval needed",
            ),
            ("human", "{query}"),
        ]
    )
    decision: RouterDecision = (prompt | _llm_fast.with_structured_output(RouterDecision)).invoke(
        {"query": query}
    )
    return {"route": decision.route}

def direct_answer_node(state: RAGState) -> dict:
    """Answer conversational queries directly without retrieval."""
    query = _last_user_message(state)
    response = _llm_strong.invoke(query)
    return {
        "answer": response.content,
        "final_status": "direct",
        "messages": [AIMessage(content=response.content)],
    }

def verify_claim_node(state: RAGState) -> dict:
    """
    Use Tavily to search for papers that supersede or contradict the
    user's cited claim, then return a structured verdict.
    """
    query = _last_user_message(state)
    results = _tavily.search(query, max_results=5)
    context = "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}"
        for r in results.get("results", [])
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant verifying whether a scientific claim is still current.\n"
                "Use the web search results below to decide if the claim has been superseded.\n\n"
                "Search results:\n{context}",
            ),
            ("human", "Claim to verify: {query}"),
        ]
    )
    verdict: ClaimVerificationResult = (
        prompt | _llm_strong.with_structured_output(ClaimVerificationResult)
    ).invoke({"context": context, "query": query})

    summary = verdict.verdict_summary
    if verdict.superseding_papers:
        papers_text = "\n".join(
            f"• [{p.title}]({p.url}) – {p.summary}" for p in verdict.superseding_papers
        )
        summary += f"\n\nRelated papers:\n{papers_text}"

    return {
        "answer": summary,
        "final_status": "generated",
        "messages": [AIMessage(content=summary)],
    }

def multi_query_node(state: RAGState) -> dict:
    """Expand the user query into three semantically varied reformulations."""
    query = _last_user_message(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query expansion assistant. Given one search query, produce "
                "exactly 3 distinct reformulations that approach the topic from different angles. "
                "The goal is maximum recall from a vector store.",
            ),
            ("human", "Original query: {query}"),
        ]
    )
    expansion: MultiQueryExpansion = (
        prompt | _llm_fast.with_structured_output(MultiQueryExpansion)
    ).invoke({"query": query})
    return {"queries": expansion.queries}

async def async_retrieval_node(state: RAGState) -> dict:
    """
    Concurrently run hybrid vector searches for all expanded queries,
    pool & deduplicate child chunks, re-rank, then swap in full parent text.
    """
    session_id = state["session_id"]
    query = _last_user_message(state)
    queries = state.get("queries") or [query]

    async def _search_one(q: str) -> list[Document]:
        return await asyncio.to_thread(search, q, session_id, k=6)

    results = await asyncio.gather(*[_search_one(q) for q in queries])

    # Pool and deduplicate by page_content identity
    seen: set[str] = set()
    pooled: list[Document] = []
    for doc_list in results:
        for doc in doc_list:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                pooled.append(doc)

    # Re-rank and keep top-4
    top_children = _rerank_docs(query, pooled, top_k=4)

    # Swap children for their full parent text
    parent_docs = await asyncio.to_thread(_parent_swap, top_children)

    return {"retrieved_docs": top_children, "parent_docs": parent_docs}


def relevancy_check_node(state: RAGState) -> dict:
    """Grade whether the retrieved parent docs actually answer the query."""
    query = _last_user_message(state)
    context = "\n\n---\n\n".join(d.page_content for d in state.get("parent_docs", []))
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict relevancy grader for a RAG pipeline.\n"
                "Context:\n{context}",
            ),
            (
                "human",
                "Query: {query}\n\nDoes this context adequately answer the query?",
            ),
        ]
    )
    decision: RelevancyDecision = (
        prompt | _llm_fast.with_structured_output(RelevancyDecision)
    ).invoke({"context": context, "query": query})
    return {"is_relevant": decision.is_relevant}


def web_fallback_node(state: RAGState) -> dict:
    """
    CRAG fallback: when retrieved docs are irrelevant, replace them with
    fresh Tavily web search results.
    """
    query = _last_user_message(state)
    results = _tavily.search(query, max_results=5)
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"title": r["title"], "source": r["url"]},
        )
        for r in results.get("results", [])
    ]
    return {
        "parent_docs": web_docs,
        "is_relevant": True,   # trust web results and proceed to generate
        "final_status": "web_fallback",
    }


def generate_node(state: RAGState) -> dict:
    """
    Generate a grounded answer with [Doc N] citations.
    Also persists the Q&A pair to the semantic cache.
    """
    query = _last_user_message(state)
    parent_docs = state.get("parent_docs", [])

    numbered_context = "\n\n".join(
        f"[Doc {i + 1}] {doc.page_content}" for i, doc in enumerate(parent_docs)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise research assistant. Answer the user's question using ONLY "
                "the provided context. Cite sources inline as [Doc 1], [Doc 2], etc. "
                "If the context is insufficient, say so honestly.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{query}"),
        ]
    )
    response = _llm_strong.invoke(
        prompt.format_messages(context=numbered_context, query=query)
    )
    answer = response.content

    # Persist to semantic cache
    save_to_cache(query, answer, state["session_id"])

    return {
        "answer": answer,
        "final_status": state.get("final_status") or "generated",
        "messages": [AIMessage(content=answer)],
    }


def hallucination_check_node(state: RAGState) -> dict:
    """
    Verify that the generated answer is grounded in the retrieved context.
    Increments a retry counter so the graph can loop back to generate_node.
    """
    answer = state.get("answer", "")
    parent_docs = state.get("parent_docs", [])
    context = "\n\n".join(d.page_content for d in parent_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grounding checker. Determine if the following answer is fully "
                "supported by the provided context. Flag it as NOT grounded only if it "
                "contains facts that cannot be traced back to the context.\n\n"
                "Context:\n{context}",
            ),
            ("human", "Answer to check:\n{answer}"),
        ]
    )
    decision: HallucinationDecision = (
        prompt | _llm_fast.with_structured_output(HallucinationDecision)
    ).invoke({"context": context, "answer": answer})

    retries = state.get("hallucination_retries", 0)
    return {
        "is_grounded": decision.is_grounded,
        "hallucination_retries": retries + 1,
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def _after_cache_check(state: RAGState) -> str:
    return "end" if state.get("final_status") == "cache_hit" else "router"


def _after_router(state: RAGState) -> str:
    route = state.get("route", "retrieve")
    return {
        "retrieve": "multi_query",
        "direct_answer": "direct_answer",
        "verify_claim": "verify_claim",
    }.get(route, "multi_query")


def _after_relevancy(state: RAGState) -> str:
    return "generate" if state.get("is_relevant") else "web_fallback"


def _after_hallucination(state: RAGState) -> str:
    if state.get("is_grounded"):
        return "end"
    # Allow at most 2 regeneration attempts before giving up
    if state.get("hallucination_retries", 0) >= 2:
        return "end"
    return "generate"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph(db_path: str = "checkpoints.db") -> Any:
    """
    Compile and return the executable LangGraph CompiledGraph.

    Parameters
    ----------
    db_path : str
        Path for the SQLite checkpointer database.
    """
    builder = StateGraph(RAGState)

    # Register nodes
    builder.add_node("cache_check", cache_check_node)
    builder.add_node("router", router_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("verify_claim", verify_claim_node)
    builder.add_node("multi_query", multi_query_node)
    builder.add_node("async_retrieval", async_retrieval_node)
    builder.add_node("relevancy_check", relevancy_check_node)
    builder.add_node("web_fallback", web_fallback_node)
    builder.add_node("generate", generate_node)
    builder.add_node("hallucination_check", hallucination_check_node)

    # Entry point
    builder.set_entry_point("cache_check")

    # Edges
    builder.add_conditional_edges(
        "cache_check",
        _after_cache_check,
        {"end": END, "router": "router"},
    )
    builder.add_conditional_edges(
        "router",
        _after_router,
        {
            "multi_query": "multi_query",
            "direct_answer": "direct_answer",
            "verify_claim": "verify_claim",
        },
    )
    builder.add_edge("direct_answer", END)
    builder.add_edge("verify_claim", END)
    builder.add_edge("multi_query", "async_retrieval")
    builder.add_edge("async_retrieval", "relevancy_check")
    builder.add_conditional_edges(
        "relevancy_check",
        _after_relevancy,
        {"generate": "generate", "web_fallback": "web_fallback"},
    )
    builder.add_edge("web_fallback", "generate")
    builder.add_edge("generate", "hallucination_check")
    builder.add_conditional_edges(
        "hallucination_check",
        _after_hallucination,
        {"end": END, "generate": "generate"},
    )

    # SQLite checkpointer for persistence & resumability
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)


# ── Convenience runner ────────────────────────────────────────────────────────

async def run_query(query: str, session_id: str, graph=None) -> str:
    """
    High-level entry point.  Invoke the graph for a single user query and
    return the final answer string.

    Parameters
    ----------
    query      : The user's raw question.
    session_id : Identifies the Qdrant collection and cache partition.
    graph      : Pre-compiled graph (optional; builds a default one if None).
    """
    if graph is None:
        graph = build_graph()

    config = {"configurable": {"thread_id": session_id}}
    initial_state: RAGState = {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "route": "",
        "queries": [],
        "retrieved_docs": [],
        "parent_docs": [],
        "is_relevant": False,
        "is_grounded": False,
        "answer": "",
        "final_status": "",
        "hallucination_retries": 0,
    }

    final_state = await graph.ainvoke(initial_state, config=config)

    # If the graph ended, but the answer was never fully grounded...
    if not final_state.get("is_grounded") and final_state.get("hallucination_retries", 0) >= 2:
        return (
            "I apologize, but I could not generate an answer fully supported by "
            "the provided documents. Please try rephrasing your question or uploading "
            "more relevant papers."
        )
    return final_state.get("answer", "No answer generated.")


