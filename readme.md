Intelligent Ticket Analyzer
An end-to-end, full-stack application that uses a fine-tuned RoBERTa model to classify customer support tickets by issue type and urgency, and extracts key entities from the text. The system is served via a high-performance FastAPI backend and features a responsive React (TypeScript) frontend.

(Recommendation: Record a short GIF of you using the app and upload it to a site like Imgur to replace the link above.)

Features
Multi-Task Classification: A single transformer model predicts both issue_type and urgency_level simultaneously.

Rule-Based Entity Extraction: Identifies products, dates, and complaint keywords using regex and keyword matching.

High-Performance API: Built with FastAPI and optimized for production with an ONNX-exported model for fast inference.

Interactive Frontend: A clean, professional UI built with React and TypeScript.

Theme Toggle: A dynamic theme switcher for a seamless user experience in both dark and light modes.

Modular Project Structure: Organized for scalability and maintainability.

Tech Stack
Category

Technology

Backend

Python, FastAPI, Uvicorn

Frontend

React, TypeScript, Vite, Axios, CSS

ML / Ops

PyTorch, Hugging Face Transformers, Scikit-learn, ONNX, ONNX Runtime, Pandas, Git

Project Structure
.
├── data/                 # Contains raw and processed datasets
├── frontend/             # React (TypeScript) single-page application
├── notebooks/            # Jupyter notebook for Exploratory Data Analysis (EDA)
├── scripts/              # Python scripts for preprocessing, training, and ONNX export
├── src/                  # Main Python source code
│   ├── api/              # FastAPI application logic, schemas, and dependencies
│   ├── models/           # Custom model architecture and saved model files
│   └── entity_extraction.py
└── requirements.txt      # Python dependencies

Setup and Installation
Prerequisites
Python 3.8+

Node.js and npm

Git

1. Clone the Repository
git clone https://github.com/KS-Mohit/Intelligent-ticket-analyzer.git
cd Intelligent-ticket-analyzer

2. Backend Setup
# Create and activate a Python virtual environment
python -m venv sta_venv
source sta_venv/bin/activate  # On Windows: sta_venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install the correct PyTorch version for your system (CPU example)
# For GPU, visit https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

3. Frontend Setup
# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

Usage
1. Run the Backend API
Open a terminal in the project root and run:

uvicorn src.api.main:app --reload

The API will be available at http://127.0.0.1:8000. You can access the interactive documentation at http://127.0.0.1:8000/docs.

2. Run the Frontend Application
Open a second terminal and navigate to the frontend directory:

cd frontend
npm run dev

The application will be running at http://localhost:5173.

API Endpoint
POST /analyze_ticket
Analyzes a raw ticket text and returns the classification and extracted entities.

Request Body:

{
  "text": "My new Vision LED TV arrived today, 2025-07-20, but it's defective and completely broken."
}

Success Response (200 OK):

{
  "issue_type": "Product Defect",
  "urgency_level": "High",
  "entities": {
    "product": "Vision LED TV",
    "date": "2025-07-20",
    "complaint_keywords": [
      "defective",
      "broken"
    ]
  }
}

Model Training & Export
To retrain the model with new or updated data:

Update Data: Modify the data/raw/Support_Tickets.xlsx file.

Run Preprocessing: python scripts/preprocess.py

Run Training: python scripts/train.py

Export to ONNX: python scripts/export_onnx.py