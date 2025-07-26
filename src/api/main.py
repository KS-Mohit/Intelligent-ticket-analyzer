from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TicketRequest, AnalysisResponse, ExtractedEntities
from .dependencies import (
    TOKENIZER, CLASSIFIER, ISSUE_MAP, URGENCY_MAP, PRODUCT_LIST,
    rule_based_extract
)
import torch

app = FastAPI(title="Intelligent Ticket Analyzer")

# --- ADD THIS CORS MIDDLEWARE BLOCK ---
# This allows your frontend (running on localhost:5173) to make requests to this backend.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# ------------------------------------

@app.post("/analyze_ticket", response_model=AnalysisResponse)
async def analyze_ticket(request: TicketRequest):
    """
    Analyzes a ticket text to classify its issue type and urgency,
    and extracts key entities.
    """
    text = request.text

    # --- 1. Classification ---
    # Tokenize the input text for the model
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=192)
    
    # Get predictions from the model
    with torch.no_grad():
        outputs = CLASSIFIER(**inputs)
    
    issue_logits = outputs['logits_issue']
    urgency_logits = outputs['logits_urgency']

    # Get the predicted class IDs by finding the index with the highest score
    predicted_issue_id = torch.argmax(issue_logits, dim=1).item()
    predicted_urgency_id = torch.argmax(urgency_logits, dim=1).item()

    # Convert the IDs back to their text labels using the mapping files
    issue_type = ISSUE_MAP.get(predicted_issue_id, "Unknown")
    urgency_level = URGENCY_MAP.get(predicted_urgency_id, "Unknown")

    # --- 2. Entity Extraction ---
    # Use our rule-based extractor to find entities
    entities = rule_based_extract(text, PRODUCT_LIST)
    
    # --- 3. Format and Return the Response ---
    return AnalysisResponse(
        issue_type=issue_type,
        urgency_level=urgency_level,
        entities=ExtractedEntities(**entities)
    )

@app.get("/")
def read_root():
    return {"message": "Ticket Analyzer API is running."}
