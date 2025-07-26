from pydantic import BaseModel
from typing import List, Optional

class TicketRequest(BaseModel):
    text: str

class ExtractedEntities(BaseModel):
    product: Optional[str] = None
    date: Optional[str] = None
    complaint_keywords: List[str] = []

class AnalysisResponse(BaseModel):
    issue_type: str
    urgency_level: str
    entities: ExtractedEntities