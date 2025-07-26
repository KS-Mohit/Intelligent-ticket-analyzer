import sys
import os
import torch
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add the project root to the Python path for robust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.multi_task_model import RobertaForMultiTaskClassification

# --- Configuration ---
# Build paths relative to this script's location for stability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'multi_task_data.pt')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'roberta-multi-task-classifier')

MODEL_NAME = 'roberta-base'
NUM_EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# --- Custom Dataset ---
class MultiTaskDataset(Dataset):
    """A custom dataset to handle our multi-task data."""
    def __init__(self, data):
        self.encodings = data['encodings']
        self.labels_issue = data['issue_labels']
        self.labels_urgency = data['urgency_labels']

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels_issue'] = self.labels_issue[idx].clone().detach()
        item['labels_urgency'] = self.labels_urgency[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels_issue)

# --- Metrics Calculation ---
def compute_metrics(p):
    """Computes and returns a dictionary of metrics for both tasks."""
    preds_issue = p.predictions[0].argmax(-1)
    preds_urgency = p.predictions[1].argmax(-1)
    labels_issue = p.label_ids[0]
    labels_urgency = p.label_ids[1]

    precision_issue, recall_issue, f1_issue, _ = precision_recall_fscore_support(labels_issue, preds_issue, average='weighted', zero_division=0)
    acc_issue = accuracy_score(labels_issue, preds_issue)

    precision_urgency, recall_urgency, f1_urgency, _ = precision_recall_fscore_support(labels_urgency, preds_urgency, average='weighted', zero_division=0)
    acc_urgency = accuracy_score(labels_urgency, preds_urgency)

    return {
        'accuracy_issue': acc_issue, 'f1_issue': f1_issue,
        'precision_issue': precision_issue, 'recall_issue': recall_issue,
        'accuracy_urgency': acc_urgency, 'f1_urgency': f1_urgency,
        'precision_urgency': precision_urgency, 'recall_urgency': recall_urgency,
    }

# --- Custom Trainer to handle two sets of labels ---
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels_issue = inputs.pop("labels_issue")
        labels_urgency = inputs.pop("labels_urgency")
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = torch.tensor(outputs['loss'].item()) if hasattr(outputs['loss'], 'item') else torch.tensor(outputs['loss'])
            logits_issue = outputs['logits_issue']
            logits_urgency = outputs['logits_urgency']

        labels = (labels_issue, labels_urgency)
        logits = (logits_issue, logits_urgency)
        
        return (loss, logits, labels)

# --- Main Training Function ---
def train():
    """Main function to load data, set up, and run the training process."""
    print("--- Loading Preprocessed Data ---")
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False)

    train_dataset = MultiTaskDataset({
        'encodings': data['train_encodings'], 'issue_labels': data['train_issue_labels'], 'urgency_labels': data['train_urgency_labels']
    })
    val_dataset = MultiTaskDataset({
        'encodings': data['val_encodings'], 'issue_labels': data['val_issue_labels'], 'urgency_labels': data['val_urgency_labels']
    })
    
    print("--- Initializing Model & Tokenizer ---")
    # Load the tokenizer that matches the base model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    model = RobertaForMultiTaskClassification.from_pretrained(
        MODEL_NAME, num_issue_labels=data['num_issue_labels'], num_urgency_labels=data['num_urgency_labels']
    )

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs'),
        logging_steps=10,
        eval_strategy="epoch",      # Using older API name for compatibility
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_issue",
        report_to="none",
    )
    
    trainer = MultiTaskTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, compute_metrics=compute_metrics,
    )
    
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")
    
    # Save the final model and the tokenizer
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    print(f"Best model and tokenizer saved to {MODEL_OUTPUT_DIR}")

if __name__ == '__main__':
    train()