import asyncio
import json
import os
import uuid
import pandas as pd
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# We need an OpenAI wrapper for Ragas
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

async def evaluate_pipeline():
    console = Console()
    console.print("[bold blue]Starting Ragas Evaluation...[/bold blue]")
    
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    if not os.path.exists(dataset_path):
        console.print("[red]Dataset not found! Please run generate_dataset.py first.[/red]")
        return
        
    with open(dataset_path, "r") as f:
        synthetic_data = json.load(f)
        
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from services.rag_pipeline import workflow
    from langchain_core.messages import HumanMessage
    
    app_graph = workflow.compile()
    
    # We must use a valid session_id that actually exists in Qdrant, otherwise hybrid_search will filter out everything!
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333")
    res = client.scroll(collection_name="research_papers_v2", limit=1, with_payload=True)[0]
    valid_session_id = res[0].payload.get("session_id")
    console.print(f"[green]Using valid session_id: {valid_session_id}[/green]")
    
    eval_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }
    
    for item in track(synthetic_data, description="Running RAG Pipeline on Test Set..."):
        query = item["user_input"]
        session_id = valid_session_id
        
        try:
            res = await app_graph.ainvoke({"messages": [HumanMessage(content=query)], "session_id": session_id})
            
            answer = res.get("generated_answer", "")
            docs = res.get("retrieved_docs", [])
            contexts = [d.page_content for d in docs] if docs else ["No context retrieved."]
            
            eval_data["user_input"].append(query)
            eval_data["response"].append(answer)
            eval_data["retrieved_contexts"].append(contexts)
            eval_data["reference"].append(item["reference"])
        except Exception as e:
            console.print(f"[red]Error on query '{query}': {e}[/red]")
            
    hf_dataset = Dataset.from_dict(eval_data)
    
    console.print("[bold yellow]Running Ragas LLM-as-a-Judge Evaluation...[/bold yellow]")
    
    # In Ragas 0.2+, evaluate expects a Dataset and a list of metrics
    metrics = [
        context_precision,
        faithfulness,
        answer_relevancy
    ]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    # Print beautiful table
    df = result.to_pandas()
    
    table = Table(title="Ragas Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Average Score", style="magenta")
    
    # Calculate means
    mean_cp = df['context_precision'].mean() if 'context_precision' in df else 0
    mean_f = df['faithfulness'].mean() if 'faithfulness' in df else 0
    mean_ar = df['answer_relevancy'].mean() if 'answer_relevancy' in df else 0
    
    table.add_row("Context Precision", f"{mean_cp:.3f}")
    table.add_row("Faithfulness", f"{mean_f:.3f}")
    table.add_row("Answer Relevancy", f"{mean_ar:.3f}")
    
    console.print(table)
    
    out_path = os.path.join(os.path.dirname(__file__), "metrics.csv")
    df.to_csv(out_path, index=False)
    console.print(f"[bold green]Saved detailed metrics to {out_path}[/bold green]")

if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())
