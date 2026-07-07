import asyncio
from typing import Annotated, Any
import sqlite3
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from core.config import settings, get_fast_llm, get_strong_llm
from services.agent_state import (
    RAGState, RouterDecision, RelevancyDecision, HallucinationDecision
)
from services.paper_loader import fetch_parent
from services.vector_store import search, check_cache, save_to_cache


llm_fast = get_fast_llm()
llm_strong = get_strong_llm()

from flashrank import Ranker, RerankRequest  
reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

from tavily import TavilyClient
tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)

from pydantic import BaseModel, Field

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(query: str):
    """Search the uploaded research paper vector store for relevant passages."""
    pass

class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query optimized for web search")

@tool(args_schema=WebSearchInput)
def web_search(optimized_query: str):
    """Search the web for current or supplementary information."""
    pass

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm_strong.bind_tools(RETRIEVAL_TOOLS)

def _rerank_docs(query: str, docs: list[Document], top_k: int = 4) -> list[Document]:
    """Re-ranks retrieved documents for higher relevance."""
    if not docs:
        return []
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    results = reranker.rerank(request)
    ranked_indices = [r["id"] for r in results[:top_k]]
    return [docs[i] for i in ranked_indices]

def _parent_swap(child_docs: list[Document], session_id: str, db_path: str = "doc_store.db") -> list[Document]:
    """Swaps tiny children for their massive parents securely."""
    swapped: list[Document] = []
    for doc in child_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            # Passing session_id for Multi-Tenancy security!
            parent_text = fetch_parent(parent_id, session_id, db_path) 
            if parent_text:
                doc = Document(page_content=parent_text, metadata=doc.metadata)
        swapped.append(doc)
    return swapped

def cache_check_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    cached_answer = check_cache(query, state["session_id"])
    if cached_answer:
        return {"answer": cached_answer, "final_status": "cache_hit"}
    return {"final_status": ""}

def router_node(state: RAGState) -> dict:
    current_query = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the intent router for a Research Assistant.\n"
            "ROUTING CATEGORIES:\n"
            " - 'retrieve': Question about research, papers, summaries, or technical data.\n"
            " - 'verify_claim': User explicitly asks to fact-check or search the live web.\n"
            " - 'direct_answer': Casual greetings (hi, thanks).\n"
        ),
        ("human", "Latest User Query: {query}"),
    ])

    decision: RouterDecision = (prompt | llm_fast.with_structured_output(RouterDecision)).invoke(
        {"query": current_query}
    )
    return {"route": decision.route}

def agent_node(state: RAGState) -> dict:
    """The agent that decides whether to search the Vector DB or the Web."""
    current_attempts = state.get("retrieval_attempts", 0)
    lm = llm_strong if current_attempts >= 3 else retrieval_llm
    
    system_msg = (
        "You are a research assistant gathering context. "
        "DO NOT answer the user's question directly here. ONLY call tools to collect context."
    )
    messages = [{"role": "system", "content": system_msg}] + state["messages"]
    response = lm.invoke(messages)
    
    updates = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates

def execute_tools_node(state: RAGState) -> dict:
    """Executes Qdrant or Tavily based on what the Agent decided."""
    last_msg = state["messages"][-1]
    session_id = state["session_id"]
    current_docs = list(state.get("parent_docs", []))
    
    tool_messages = []
    for tc in last_msg.tool_calls:
        if tc["name"] == "retrieve_from_vectorstore":
            query = tc["args"]["query"]
            raw_docs = search(query, session_id, k=6)
            
            top_children = _rerank_docs(query, raw_docs, top_k=4)
            # Fetch parents securely!
            parent_docs = _parent_swap(top_children, session_id)
            
            if parent_docs:
                current_docs.extend(parent_docs)
                msg = f"Retrieved {len(parent_docs)} chunks from vector store."
            else:
                msg = "No relevant documents found."
                
            tool_messages.append(ToolMessage(content=msg, tool_call_id=tc["id"], name=tc["name"]))
            
        elif tc["name"] == "web_search":
            query = tc["args"]["optimized_query"]
            results = tavily.search(query, max_results=3)
            
            if results.get("results"):
                web_docs = [Document(page_content=r["content"], metadata={"source": r["url"]}) for r in results["results"]]
                current_docs.extend(web_docs)
                msg = f"Found {len(web_docs)} web result(s)."
            else:
                msg = "No web results found."
                
            tool_messages.append(ToolMessage(content=msg, tool_call_id=tc["id"], name=tc["name"]))
            
    return {"messages": tool_messages, "parent_docs": current_docs}

def grade_documents_node(state: RAGState) -> dict:
    """Checks if the retrieved documents are actually relevant to the question."""
    query = state["messages"][0].content
    docs = state.get("parent_docs", [])
    if not docs:
        return {"is_relevant": False}
        
    context = "\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade whether the provided context contains the answer to the user's question."),
        ("human", "Question: {query}\n\nContext:\n{context}")
    ])
    
    decision: RelevancyDecision = (prompt | llm_fast.with_structured_output(RelevancyDecision)).invoke(
        {"query": query, "context": context}
    )
    return {"is_relevant": decision.is_relevant}

def generate_node(state: RAGState) -> dict:
    """Writes the final answer based on the collected context."""
    query = state["messages"][0].content
    docs = state.get("parent_docs", [])
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Answer the user's question using ONLY the provided context.\n\nContext:\n{context}"),
        ("human", "{query}")
    ])
    
    response = (prompt | llm_strong).invoke({"query": query, "context": context})
    return {"answer": response.content, "final_status": "generated"}

def hallucination_check_node(state: RAGState) -> dict:
    """The Strict Auditor: Checks if the LLM made anything up."""
    answer = state["answer"]
    docs = state.get("parent_docs", [])
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an auditor. Check if the generated answer contains ANY factual claims not present in the Context."),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
    ])
    
    decision: HallucinationDecision = (prompt | llm_fast.with_structured_output(HallucinationDecision)).invoke(
        {"context": context, "answer": answer}
    )
    return {"is_grounded": decision.is_grounded}


def route_after_cache(state: RAGState) -> str:
    return "end" if state.get("final_status") == "cache_hit" else "router"

def route_after_router(state: RAGState) -> str:
    return "agent" if state.get("route") == "retrieve" else "end"

def route_after_agent(state: RAGState) -> str:
    last_msg = state["messages"][-1]
    return "execute_tools" if getattr(last_msg, "tool_calls", None) else "grade_documents"

def route_after_grading(state: RAGState) -> str:
    return "generate" if state.get("is_relevant") else "agent"

def route_after_hallucination_check(state: RAGState) -> str:
    retries = state.get("hallucination_retries", 0)
    if state.get("is_grounded") or retries >= 2:
        return "end"
    return "generate" 

workflow = StateGraph(RAGState)
workflow.add_node("cache_check", cache_check_node)
workflow.add_node("router", router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("execute_tools", execute_tools_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_check", hallucination_check_node)

workflow.set_entry_point("cache_check")
workflow.add_conditional_edges("cache_check", route_after_cache, {"end": END, "router": "router"})
workflow.add_conditional_edges("router", route_after_router, {"agent": "agent", "end": END})
workflow.add_conditional_edges("agent", route_after_agent, {"execute_tools": "execute_tools", "grade_documents": "grade_documents"})

workflow.add_edge("execute_tools", "agent") # Tools always feed back to the agent
workflow.add_conditional_edges("grade_documents", route_after_grading, {"generate": "generate", "agent": "agent"})
workflow.add_edge("generate", "hallucination_check")
workflow.add_conditional_edges("hallucination_check", route_after_hallucination_check, {"end": END, "generate": "generate"})

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
rag_app = workflow.compile(checkpointer=memory)

def run_query(query: str, session_id: str) -> str:
    """The single function your FastAPI endpoint will call."""
    config = {"configurable": {"thread_id": session_id}}
    
    state_input = {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "hallucination_retries": 0,
        "retrieval_attempts": 0
    }

    final_state = rag_app.invoke(state_input, config=config)

    if final_state.get("final_status") == "generated":
        save_to_cache(query, final_state["answer"], session_id)
        
    return final_state.get("answer", "No answer could be generated.")

