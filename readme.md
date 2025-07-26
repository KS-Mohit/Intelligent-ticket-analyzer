# Intelligent Ticket Analyzer

This project is a full-stack, AI-powered application designed to automate the analysis of customer support tickets. It features a **React (TypeScript) frontend** and a **FastAPI backend** that leverages a fine-tuned RoBERTa model for classification and entity extraction.

---

## Implemented Features & API

The backend API provides a comprehensive analysis of raw ticket text through a single, optimized endpoint.

* **Multi-Task Classification**
    * The model simultaneously predicts the `issue_type` (e.g., "Billing Problem", "Product Defect") and the `urgency_level` ("Low", "Medium", "High").

* **Entity Extraction**
    * A rule-based system identifies and extracts key entities from the text, including `product` names, `dates`, and specific `complaint_keywords`.

* **Core API Endpoint**
    * `POST /analyze_ticket`: Accepts a raw text string and returns a structured JSON object containing the full analysis.

---

## Tech Stack

* **Frontend**: React 18 with TypeScript & Vite
* **Backend**: Python 3.10 with FastAPI
* **ML Framework**: PyTorch, Hugging Face Transformers
* **Inference Optimization**: ONNX, ONNX Runtime
* **Data Handling**: Pandas, Scikit-learn
* **Containerization**: Docker (optional, for deployment)

---

## Setup and Installation

### Prerequisites

* Python 3.10+ & `pip`
* Node.js & `npm`
* Git

### Local Development Setup

* **Run the Backend**:
    * Navigate to the project root directory.
    * Create and activate a virtual environment: `python -m venv sta_venv` then `.\sta_venv\Scripts\activate`.
    * Install dependencies: `pip install -r requirements.txt`.
    * Start the server: `uvicorn src.api.main:app --reload`. It will run on `http://localhost:8000`.

* **Run the Frontend**:
    * Open a **new terminal**.
    * Navigate to the `frontend` directory.
    * Get dependencies: `npm install`.
    * Start the app: `npm run dev`. The app will be available at `http://localhost:5173`.
