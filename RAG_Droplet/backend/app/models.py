from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    session_id: str