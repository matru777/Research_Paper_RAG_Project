"""
handle_btw.py
─────────────
Off-topic side channel for general chat and live web queries.
Bypasses the main Qdrant vector store completely to save time and compute.
"""

import os
from typing import Generator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient

from models import BtwRouteDecision

load_dotenv()

# ─── Dual-Model Architecture ───
# Fast model for instant true/false routing decisions
_llm_fast = ChatGroq(model="llama3-8b-8192", temperature=0)

# Strong model for generating high-quality conversational responses
_llm_strong = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    
    # 1. Routing Decision
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    
    # Use the small, fast 8B model to make the boolean decision
    decision: BtwRouteDecision = (route_prompt | _llm_fast.with_structured_output(BtwRouteDecision)).invoke({"query": query})

    # 2. Context Gathering
    if decision.needs_web_search:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        results = client.search(query, max_results=3)
        context = "\n\n".join(r["content"] for r in results.get("results", []))
        sources = "\n".join(f"- {r['url']}" for r in results.get("results", []))

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using the web search results below. Be concise.\n\n"
             f"Results:\n{context}\n\nSources:\n{sources}"),
            ("human", "{query}"),
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    # 3. Streaming the Final Answer
    # Use the larger 70B model to write the actual response to the user
    for chunk in (answer_prompt | _llm_strong).stream({"query": query}):
        if chunk.content:
            yield chunk.content