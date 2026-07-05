# SHL Assessment Recommender API

A conversational AI assistant designed to recommend SHL job assessments based on user inputs. The app combines dense semantic vector retrieval (hybrid search) with the Google Gemini 1.5 Flash model inside a stateless FastAPI server.

---

## 🚀 Key Features

* **Hybrid Retrieval Search**: Combines Dense Vector Search (Cosine Similarity via SentenceTransformers) with rare-word boosting and substring exact-word guardrails to prevent partial match issues (e.g., separating "Java" from "JavaScript").
* **Low-Resource & Fast Startup Optimization**:
  * **Offline Model Caching**: SentenceTransformer weights are cached locally inside the repository, preventing network download timeouts on boot.
  * **Pre-computed Embeddings**: Loads catalog vector embeddings from a pre-calculated `assessment_embeddings.npy` file, reducing RAM usage by 80% and startup time to under 0.1 seconds.
* **LLM Grounding Layer**: Automatically verifies and maps all LLM recommended items back to the local database to eliminate hallucinations and broken URLs.
* **State-Free Conversational Loop**: Implements a 3-turn internal ReAct loop to resolve queries and run database search tools in a single HTTP request-response cycle.

---

## 🛠️ Architecture Overview

```
                  ┌──────────────────────┐
                  │     FastAPI App      │
                  │      (main.py)       │
                  └──────────┬───────────┘
                             │ (runs conversation history)
                             ▼
                  ┌──────────────────────┐
                  │      run_agent       │
                  │      (agent.py)      │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
     (If search tool call)              (If final response)
            ▼                                 ▼
┌───────────────────────┐            ┌──────────────────┐
│  semantic_recommend   │            │ Grounding Layer  │
│(semantic_recommender) │            │ (verify URL/name)│
└──────────┬────────────┘            └────────┬─────────┘
           │                                  │
    ┌──────┴───────────────┐                  ▼
    │  Cosine Similarity   │           [ChatResponse]
    │    (embeddings)      │
    └──────────────────────┘
```

---

## 📦 Installation & Setup

### Prerequisites
* Python 3.9+
* A Google Gemini API Key

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd shl-assessment

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Set your Gemini API Key in your shell environment:
```bash
# On Windows (PowerShell):
$env:GEMINI_API_KEY="your-gemini-api-key"

# On macOS/Linux:
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## 🏃 Running the Application

To start the FastAPI development server:
```bash
uvicorn main:app --reload
```
The server will start on `http://127.0.0.1:8000`.

---

## 📍 API Endpoints

### 1. Health Check
* **Endpoint**: `GET /health`
* **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Conversational Agent Recommendation
* **Endpoint**: `POST /chat`
* **Request Body**:
  ```json
  {
    "messages": [
      {
        "role": "user",
        "content": "I need to test an entry-level Java Developer on core coding skills."
      }
    ]
  }
  ```
* **Response**:
  ```json
  {
    "reply": "Based on your requirements, here are the best core Java assessments for entry-level developers...",
    "recommendations": [
      {
        "name": "Core Java (Entry Level) (New)",
        "url": "https://www.shl.com/products/product-catalog/view/core-java-entry-level-new/",
        "test_type": "K"
      }
    ],
    "end_of_conversation": true
  }
  ```

---

## 🧪 Running Tests

The test suite runs queries through the agent simulation to verify the vague/specific queries handling and grounding logic.

To run the local test suite using the virtual environment:
```bash
# On Windows:
.\venv\Scripts\python.exe test_agent.py

# On macOS/Linux:
python test_agent.py
```
