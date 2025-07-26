import torch
import pandas as pd
import os
import json
from transformers import RobertaTokenizer
from src.models.multi_task_model import RobertaForMultiTaskClassification
from src.entity_extraction import extract_entities as rule_based_extract

# --- Build Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'roberta-multi-task-classifier')
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Support_Tickets.xlsx')

# --- Load Resources ---

def load_classification_model():
    """Loads the fine-tuned classification model and tokenizer."""
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    
    # --- THIS IS THE CORRECTED LOGIC ---
    # Load label mappings first to get the counts
    issue_map_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'issue_type_mapping.json')
    urgency_map_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'urgency_level_mapping.json')
    
    with open(issue_map_path, 'r') as f:
        issue_mapping = {int(k): v for k, v in json.load(f).items()}
    with open(urgency_map_path, 'r') as f:
        urgency_mapping = {int(k): v for k, v in json.load(f).items()}
        
    # Get the number of labels from the length of the mapping dictionaries
    num_issue_labels = len(issue_mapping)
    num_urgency_labels = len(urgency_mapping)
    
    # Load the model and explicitly pass the number of labels
    model = RobertaForMultiTaskClassification.from_pretrained(
        MODEL_PATH,
        num_issue_labels=num_issue_labels,
        num_urgency_labels=num_urgency_labels
    )
    model.eval() # Set model to evaluation mode
            
    return model, tokenizer, issue_mapping, urgency_mapping

def load_product_list():
    """Loads the unique product list for entity extraction."""
    df = pd.read_excel(DATA_FILE_PATH)
    return df['product'].dropna().unique().tolist()

# --- Global Variables ---
# These are loaded once when the application starts
CLASSIFIER, TOKENIZER, ISSUE_MAP, URGENCY_MAP = load_classification_model()
PRODUCT_LIST = load_product_list()