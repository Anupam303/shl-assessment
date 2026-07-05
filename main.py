from fastapi import FastAPI
from catalogue import load_catalogue
from models import ChatRequest, ChatResponse
from agent import run_agent
from embeddings import build_assessment_embeddings

app = FastAPI()

# Load the catalogue and compute embeddings once at startup
catalogue = load_catalogue()
assessment_embeddings = build_assessment_embeddings(catalogue)

@app.get("/")
def home():
    return{
        "message": "SHL assessment running"
    }

@app.get("/health")
def health():
    return{
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def get_recommendation(request: ChatRequest):
    """
    Handles conversational messages and returns agent reply with recommendations when ready
    """
    return run_agent(request.messages, catalogue, assessment_embeddings)