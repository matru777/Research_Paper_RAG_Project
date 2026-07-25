import asyncio
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
import urllib.parse
import urllib.request
import re

from core.config import settings, get_fast_llm, get_strong_llm
from services.rag_models import (
    PipelineState, RouteDecision, MultiQueryExpansion, 
    RelevanceCheck, GroundingCheck, ClaimExtraction, QueryRewrite
)
from services.paper_loader import fetch_parent
from services.vector_store import hybrid_search, check_cache, save_to_cache
from flashrank import Ranker, RerankRequest

llm_fast = get_fast_llm()
llm_strong = get_strong_llm()
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

ROUTER_SYSTEM = (
    "You are a routing assistant for a research paper Q&A system. "
    "Classify the user query into exactly one of two categories:\n\n"
    "  qa — Use this for questions about the content of uploaded research papers.\n"
    "  verification — The user wants to check whether a specific factual claim "
    "or finding is accurate.\n\n"
    "When in doubt, prefer 'qa'. Return only the route field."
)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewrite assistant. Given the conversation history, rewrite the latest user query to be perfectly standalone and highly specific for vector search. "
    "If the query references previous entities (e.g. 'it', 'they', 'the model'), resolve them using the history.\n"
    "Also, extract any metadata filters if specified by the user."
)

QUERY_EXPANDER_SYSTEM = "Generate 3 distinct search queries to find the best information for the user's question."

RELEVANCE_GATE_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. Be lenient: if there is any substantive overlap, return true."
)

VERDICT_GENERATOR_SYSTEM = (
    "You are a research fact-checker. Given a claim and a set of local uploaded papers and recent web search results "
    "(including arXiv and PubMed), determine its accuracy.\n\n"
    "Rules:\n"
    "1. Output a verdict of exactly: [SUPPORTED], [REFUTED], or [INSUFFICIENT EVIDENCE].\n"
    "2. Write a detailed explanation justifying your verdict based ONLY on the provided evidence. Prioritize local uploaded papers.\n"
    "3. You must cite your sources using the exact URLs or Local Document references provided in the evidence block.\n"
    "4. If a claim has been superseded or updated by more recent work, explicitly mention the newer findings."
)

async def cache_check_node(state: PipelineState) -> dict:
    query = state["messages"][-1].content
    cached = await asyncio.to_thread(check_cache, query, state["session_id"])
    if cached:
        return {"generated_answer": cached, "is_grounded": True, "cache_hit": True}
    return {"cache_hit": False}

async def router_node(state: PipelineState) -> dict:
    query = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("user", "{query}")
    ])
    chain = prompt | llm_fast.with_structured_output(RouteDecision)
    res = await chain.ainvoke({"query": query})
    return {"route": res.route}

async def query_rewrite_node(state: PipelineState) -> dict:
    conversation_history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-6:]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_SYSTEM),
        ("user", "History:\n{history}")
    ])
    chain = prompt | llm_fast.with_structured_output(QueryRewrite)
    res = await chain.ainvoke({"history": conversation_history})
    filter_dict = res.metadata_filter.model_dump(exclude_none=True) if res.metadata_filter else {}
    return {"rewritten_query": res.rewritten_query, "metadata_filter": filter_dict, "needs_web_search": res.needs_web_search}

async def query_expander_node(state: PipelineState) -> dict:
    query = state["rewritten_query"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_EXPANDER_SYSTEM),
        ("user", "{query}")
    ])
    chain = prompt | llm_fast.with_structured_output(MultiQueryExpansion)
    res = await chain.ainvoke({"query": query})
    return {"expanded_queries": res.queries}

async def retriever_node(state: PipelineState) -> dict:
    queries = state["expanded_queries"]
    session_id = state["session_id"]
    query = state["rewritten_query"]
    metadata_filter = state.get("metadata_filter")

    all_docs = []
    for q in queries:
        docs = await asyncio.to_thread(hybrid_search, q, session_id, 5, metadata_filter)
        all_docs.extend(docs)
        
    if state.get("needs_web_search"):
        try:
            # Also run a web search to augment QA with recent real-world knowledge
            web_res = await asyncio.to_thread(tavily_client.search, query, search_depth="basic", max_results=3)
            for r in web_res.get("results", []):
                all_docs.append(Document(
                    page_content=r["content"], 
                    metadata={"title": f"Web Source: {r.get('url', 'URL')}", "parent_id": "web"}
                ))
        except Exception:
            pass # Silently ignore web search failures in QA route
        
    unique_ids = set()
    unique_docs = []
    for doc in all_docs:
        doc_hash = hash(doc.page_content)
        if doc_hash not in unique_ids:
            unique_ids.add(doc_hash)
            unique_docs.append(doc)
            
    if not unique_docs:
        return {"retrieved_docs": []}

    passages = [{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(unique_docs)]
    rerankreq = RerankRequest(query=query, passages=passages)
    reranked = await asyncio.to_thread(ranker.rerank, rerankreq)

    top_children = []
    for hit in reranked[:5]:
        meta = hit["meta"]
        top_children.append(Document(page_content=hit["text"], metadata=meta))

    parent_docs = []
    parent_ids_seen = set()
    for child in top_children:
        pid = child.metadata.get("parent_id")
        if pid in ("table_standalone", "web"):
            parent_docs.append(child)
        elif pid and pid not in parent_ids_seen:
            parent_ids_seen.add(pid)
            parent_text = await fetch_parent(pid, session_id)
            if parent_text:
                parent_docs.append(Document(page_content=parent_text, metadata=child.metadata))

    return {"retrieved_docs": parent_docs}


async def relevance_gate_node(state: PipelineState) -> dict:
    query = state["rewritten_query"]
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"is_relevant": False}
        
    context = "\n\n".join([f"Document {i}:\n{d.page_content}" for i, d in enumerate(docs)])
    prompt = ChatPromptTemplate.from_messages([
        ("system", RELEVANCE_GATE_SYSTEM),
        ("user", "Query: {query}\n\nContext: {context}")
    ])
    chain = prompt | llm_fast.with_structured_output(RelevanceCheck)
    res = await chain.ainvoke({"query": query, "context": context})
    return {"is_relevant": res.is_relevant}

async def answer_generator_node(state: PipelineState, config: RunnableConfig) -> dict:
    query = state["rewritten_query"]
    docs = state["retrieved_docs"]
    is_relevant = state.get("is_relevant", True)
    feedback = state.get("grounding_feedback")
    
    conf = config.copy()
    conf["tags"] = conf.get("tags", []) + ["stream_me"]
    
    if not is_relevant or not docs:
        sys_prompt = "You are a helpful AI assistant. The user just said something conversational, or their query is not answerable using the uploaded documents. Respond politely using ONLY the context of the chat history. Note: The user DOES have documents uploaded, they just weren't relevant to this specific message. DO NOT claim you cannot access their documents."
        chat_history = [SystemMessage(content=sys_prompt)] + state["messages"][-6:]
        res_content = ""
        async for chunk in llm_strong.astream(chat_history, config=conf):
            res_content += chunk.content
        return {"generated_answer": res_content, "messages": [AIMessage(content=res_content)]}
    
    context = "\n\n".join([f"[{d.metadata.get('title', 'Paper')}]:\n{d.page_content}" for d in docs])
    sys_prompt = "Answer the user's question using ONLY the provided context. Include inline citations like [Paper Title, Page X]."
    if feedback:
        sys_prompt += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ATTEMPT: {feedback}. Ensure all claims are supported."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "Context:\n{context}\n\nQuery:\n{query}")
    ])
    res_content = ""
    async for chunk in llm_strong.astream(prompt.format_messages(context=context, query=query), config=conf):
        res_content += chunk.content
    return {"generated_answer": res_content, "messages": [AIMessage(content=res_content)]}

async def grounding_check_node(state: PipelineState) -> dict:
    answer = state["generated_answer"]
    docs = state["retrieved_docs"]
    retries = state.get("hallucination_retries", 0)
    
    context = "\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify if EVERY factual claim in the answer is supported by the context. If not, mark is_grounded=False and provide feedback. General transitional text is acceptable."),
        ("user", "Context:\n{context}\n\nAnswer:\n{answer}")
    ])
    chain = prompt | llm_fast.with_structured_output(GroundingCheck)
    res = await chain.ainvoke({"context": context, "answer": answer})
    
    if res.is_grounded:
        save_to_cache(state["messages"][-1].content, answer, state["session_id"])
        
    return {
        "is_grounded": res.is_grounded, 
        "grounding_feedback": res.feedback,
        "hallucination_retries": retries + 1
    }

async def claim_extractor_node(state: PipelineState) -> dict:
    query = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract individual factual claims that the user wants to verify."),
        ("user", "{query}")
    ])
    chain = prompt | llm_fast.with_structured_output(ClaimExtraction)
    res = await chain.ainvoke({"query": query})
    return {"extracted_claims": res.claims}


async def local_evidence_node(state: PipelineState) -> dict:
    claims = state["extracted_claims"]
    session_id = state["session_id"]
    search_query = " ".join(claims)
    docs = await asyncio.to_thread(hybrid_search, search_query, session_id, 5, None)
    evidence = "\n".join([f"[Local Document]: {d.page_content}" for d in docs])
    return {"local_evidence": evidence}

async def web_evidence_node(state: PipelineState) -> dict:
    claims = state["extracted_claims"]
    search_query = " ".join(claims)
    results = await asyncio.to_thread(
        tavily_client.search,
        search_query, 
        search_depth="advanced", 
        include_domains=["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "ieee.org", "semanticscholar.org"]
    )
    evidence = "\n".join([f"[{res['url']}] {res['content']}" for res in results.get("results", [])])
    return {"web_evidence": evidence}

async def verdict_generator_node(state: PipelineState, config: RunnableConfig) -> dict:
    claims = state["extracted_claims"]
    local_evidence = state.get("local_evidence", "")
    web_evidence = state.get("web_evidence", "")
    
    conf = config.copy()
    conf["tags"] = conf.get("tags", []) + ["stream_me"]
    
    combined_evidence = f"--- LOCAL UPLOADED PAPERS ---\n{local_evidence}\n\n--- EXTERNAL WEB EVIDENCE ---\n{web_evidence}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERDICT_GENERATOR_SYSTEM),
        ("user", "Claims to Verify:\n{claims}\n\nSearch Results / Evidence:\n{evidence}")
    ])
    res_content = ""
    async for chunk in llm_strong.astream(prompt.format_messages(claims="\n".join(claims), evidence=combined_evidence), config=conf):
        res_content += chunk.content
    save_to_cache(state["messages"][-1].content, res_content, state["session_id"])
    return {"generated_answer": res_content, "messages": [AIMessage(content=res_content)]}

def route_cache(state: PipelineState):
    if state.get("cache_hit"): return END
    return "router"
def route_after_router(state: PipelineState):
    if state["route"] == "qa": return "query_rewrite"
    return "claim_extractor"
def route_after_generation(state: PipelineState):
    if not state.get("is_relevant", True): return END
    return "grounding_check"
def route_grounding(state: PipelineState):
    if state["is_grounded"] or state.get("hallucination_retries", 0) >= 2:
        return END
    return "answer_generator"

workflow = StateGraph(PipelineState)
workflow.add_node("cache_check", cache_check_node)
workflow.add_node("router", router_node)
workflow.add_node("query_rewrite", query_rewrite_node)
workflow.add_node("query_expander", query_expander_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("relevance_gate", relevance_gate_node)
workflow.add_node("answer_generator", answer_generator_node)
workflow.add_node("grounding_check", grounding_check_node)
workflow.add_node("claim_extractor", claim_extractor_node)
workflow.add_node("local_evidence", local_evidence_node)
workflow.add_node("web_evidence", web_evidence_node)
workflow.add_node("verdict_generator", verdict_generator_node)
workflow.set_entry_point("cache_check")
workflow.add_conditional_edges("cache_check", route_cache)
workflow.add_conditional_edges("router", route_after_router)

workflow.add_edge("query_rewrite", "query_expander")
workflow.add_edge("query_expander", "retriever")
workflow.add_edge("retriever", "relevance_gate")
workflow.add_edge("relevance_gate", "answer_generator")
workflow.add_conditional_edges("answer_generator", route_after_generation)
workflow.add_conditional_edges("grounding_check", route_grounding)

workflow.add_edge("claim_extractor", "local_evidence")
workflow.add_edge("local_evidence", "web_evidence")
workflow.add_edge("web_evidence", "verdict_generator")
workflow.add_edge("verdict_generator", END)

async def run_pipeline_stream(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """Runs the pipeline and streams the final answer token by token."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from core.database import DATABASE_URL
    
    pg_url = DATABASE_URL.replace("+asyncpg", "")
    
    cached = await asyncio.to_thread(check_cache, query, session_id)
    if cached:
        yield cached
        return
        
    async with AsyncPostgresSaver.from_conn_string(pg_url) as checkpointer:
        await checkpointer.setup()
        app_graph = workflow.compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": session_id}}
        
        state = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "hallucination_retries": 0
        }
        
        is_not_relevant = False
        
        async for event in app_graph.astream_events(state, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream" and "stream_me" in event.get("tags", []):
                content = event["data"]["chunk"].content
                if content:
                    yield content