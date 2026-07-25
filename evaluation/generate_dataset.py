import asyncio
import json
import os
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import random
from rich.console import Console
from rich.progress import track

class SyntheticQA(BaseModel):
    question: str = Field(description="A highly specific question answerable by the text.")
    ground_truth: str = Field(description="The comprehensive ground truth answer.")

async def generate_qa_pair(llm, text: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert examiner. Read the text and generate a complex, specific question that requires reasoning, and provide the ground-truth answer. The question must not be trivial."),
        ("user", "Text:\n{text}")
    ])
    chain = prompt | llm.with_structured_output(SyntheticQA)
    try:
        res = await chain.ainvoke({"text": text})
        return {"user_input": res.question, "reference": res.ground_truth, "context": text}
    except Exception as e:
        return None

async def main():
    console = Console()
    console.print("[bold blue]Starting Synthetic Dataset Generation...[/bold blue]")
    
    # We must import from our codebase to get the right collection
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from services.vector_store import qdrant_client, MAIN_COLLECTION
    
    # Fetch random chunks
    scroll_res, _ = qdrant_client.scroll(
        collection_name=MAIN_COLLECTION,
        limit=500,
        with_payload=True
    )
    
    chunks = [p.payload.get("page_content", "") for p in scroll_res if p.payload]
    chunks = [c for c in chunks if len(c) > 300]
    
    if not chunks:
        console.print("[red]No documents found in Qdrant! Have you uploaded a paper?[/red]")
        return
        
    random.shuffle(chunks)
    target_chunks = chunks[:25] # We will generate 25 questions
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    dataset = []
    
    for i, chunk in enumerate(track(target_chunks, description="Generating Q&A Pairs...")):
        res = await generate_qa_pair(llm, chunk)
        if res:
            dataset.append(res)
            
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "test_dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)
        
    console.print(f"[bold green]Successfully generated {len(dataset)} Q&A pairs![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
