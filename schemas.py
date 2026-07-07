from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or prompt.")
    session_id: str = Field(..., description="A unique ID for the user's chat session.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The AI generated answer.")
    status: str = Field(..., description="The routing status (e.g., 'cache_hit', 'generated').")

class LoadDocumentRequest(BaseModel):
    source: str = Field(..., description="The source to load (ArXiv ID, HTTPS URL, or local file path).")
    session_id: str = Field(..., description="The session ID to store the document under.")

class LoadDocumentResponse(BaseModel):
    status: str = Field(..., description="Status of the operation.")
    message: str = Field(..., description="Details about the loaded document.")
                            
