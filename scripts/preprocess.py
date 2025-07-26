import pandas as pd
import torch
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json

# --- Configuration ---
DATA_FILE_PATH = 'data/raw/Support_Tickets.xlsx' # Make sure this name is correct
PROCESSED_DATA_DIR = 'data/processed/'
MODEL_NAME = 'roberta-base'
MAX_LENGTH = 192  # Based on our EDA
TEST_SIZE = 0.2
RANDOM_STATE = 42

def run_preprocessing():
    """
    Loads, preprocesses, and tokenizes the ticket data for multi-task learning.
    """
    print("--- Starting Preprocessing ---")

    # 1. Load and Clean Data
    try:
        df = pd.read_excel(DATA_FILE_PATH)
        df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level'], inplace=True)
        df['ticket_text'] = df['ticket_text'].astype(str).str.lower()
        print(f"Loaded and cleaned {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_FILE_PATH}")
        return

    # 2. Encode Labels
    issue_encoder = LabelEncoder()
    urgency_encoder = LabelEncoder()

    df['issue_type_encoded'] = issue_encoder.fit_transform(df['issue_type'])
    df['urgency_level_encoded'] = urgency_encoder.fit_transform(df['urgency_level'])

    # Save label mappings
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DATA_DIR, 'issue_type_mapping.json'), 'w') as f:
        json.dump(dict(zip(range(len(issue_encoder.classes_)), issue_encoder.classes_)), f)
    with open(os.path.join(PROCESSED_DATA_DIR, 'urgency_level_mapping.json'), 'w') as f:
        json.dump(dict(zip(range(len(urgency_encoder.classes_)), urgency_encoder.classes_)), f)
    
    print("Encoded labels and saved mappings.")
    print(f"Issue Types: {list(issue_encoder.classes_)}")
    print(f"Urgency Levels: {list(urgency_encoder.classes_)}")

    # 3. Stratified Split
    # Create a combined column for stratification to balance both labels
    df['stratify_col'] = df['issue_type'].astype(str) + '_' + df['urgency_level'].astype(str)
    
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['stratify_col']
    )
    print(f"Data split into {len(train_df)} training and {len(val_df)} validation samples.")

    # 4. Tokenize Text
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

    train_encodings = tokenize(train_df['ticket_text'].tolist())
    val_encodings = tokenize(val_df['ticket_text'].tolist())
    print("Tokenization complete.")

    # 5. Save Processed Data
    processed_data = {
        'train_encodings': train_encodings,
        'train_issue_labels': torch.tensor(train_df['issue_type_encoded'].values),
        'train_urgency_labels': torch.tensor(train_df['urgency_level_encoded'].values),
        'val_encodings': val_encodings,
        'val_issue_labels': torch.tensor(val_df['issue_type_encoded'].values),
        'val_urgency_labels': torch.tensor(val_df['urgency_level_encoded'].values),
        'num_issue_labels': len(issue_encoder.classes_),
        'num_urgency_labels': len(urgency_encoder.classes_)
    }
    
    torch.save(processed_data, os.path.join(PROCESSED_DATA_DIR, 'multi_task_data.pt'))
    print(f"Processed multi-task data saved to {os.path.join(PROCESSED_DATA_DIR, 'multi_task_data.pt')}")


if __name__ == '__main__':
    run_preprocessing()