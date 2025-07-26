import torch
import onnxruntime
import numpy as np
import os
import json
import sys

# Add the project root to the Python path for robust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.multi_task_model import RobertaForMultiTaskClassification
from transformers import RobertaTokenizer

# --- Configuration ---
# Build paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

PYTORCH_MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'roberta-multi-task-classifier')
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'ticket_classifier.onnx')
MAPPING_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def export_to_onnx():
    """
    Loads the fine-tuned PyTorch model and exports it to the ONNX format.
    """
    print("--- Starting ONNX Export Process ---")

    # 1. Load the fine-tuned PyTorch model and tokenizer
    print(f"Loading model from: {PYTORCH_MODEL_PATH}")
    tokenizer = RobertaTokenizer.from_pretrained(PYTORCH_MODEL_PATH)

    # Get label counts from saved mapping files
    with open(os.path.join(MAPPING_DIR, 'issue_type_mapping.json'), 'r') as f:
        num_issue_labels = len(json.load(f))
    with open(os.path.join(MAPPING_DIR, 'urgency_level_mapping.json'), 'r') as f:
        num_urgency_labels = len(json.load(f))

    model = RobertaForMultiTaskClassification.from_pretrained(
        PYTORCH_MODEL_PATH,
        num_issue_labels=num_issue_labels,
        num_urgency_labels=num_urgency_labels
    )
    model.eval()
    print("PyTorch model loaded successfully.")

    # 2. Create a dummy input for tracing the model
    dummy_text = "This is a dummy ticket for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, padding=True, max_length=192)
    dummy_input_ids = inputs['input_ids']
    dummy_attention_mask = inputs['attention_mask']
    print(f"Created dummy input with shape: {dummy_input_ids.shape}")

    # 3. Export the model to ONNX
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask), # Model inputs
        ONNX_MODEL_PATH,                         # Where to save the model
        input_names=['input_ids', 'attention_mask'], # Input names
        output_names=['logits_issue', 'logits_urgency'], # Output names
        dynamic_axes={ # Allows for variable-length inputs
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits_issue': {0: 'batch_size'},
            'logits_urgency': {0: 'batch_size'}
        },
        opset_version=14 # A commonly used ONNX opset version
    )
    print(f"Model successfully exported to: {ONNX_MODEL_PATH}")

    # 4. Validate the ONNX model
    print("\n--- Validating ONNX Model ---")
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    
    # Get PyTorch outputs
    with torch.no_grad():
        pytorch_outputs = model(dummy_input_ids, dummy_attention_mask)
    pytorch_logits_issue = pytorch_outputs['logits_issue'].numpy()
    pytorch_logits_urgency = pytorch_outputs['logits_urgency'].numpy()

    # Get ONNX outputs
    ort_inputs = {
        'input_ids': dummy_input_ids.numpy(),
        'attention_mask': dummy_attention_mask.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_logits_issue = ort_outputs[0]
    onnx_logits_urgency = ort_outputs[1]

    # Compare outputs
    np.testing.assert_allclose(pytorch_logits_issue, onnx_logits_issue, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(pytorch_logits_urgency, onnx_logits_urgency, rtol=1e-3, atol=1e-5)
    
    print("Validation successful: PyTorch and ONNX model outputs match.")

if __name__ == '__main__':
    export_to_onnx()