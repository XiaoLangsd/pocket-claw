from pydantic import BaseModel, Field
from typing import Optional, Annotated
from datetime import datetime

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    chunks: int
    status: str

class ChatRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=5000)]
    session_id: Optional[str] = None
    top_k: int = Field(4, gt=1, le=10)
    stream: bool = False

class SourceChunk(BaseModel):
    content: str
    source: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    source: list[SourceChunk]
    session_id: str
    tokens_used: int

class DocumentInfo(BaseModel):
    file_id: str
    filename: str
    chunks: int
    uploaded_at: datetime

