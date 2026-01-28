from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
import os
import requests
import time

# Import Tools
from tools.model_inspector import ModelInspector
from tools.drift_auditor import DriftAuditor
from tools.news_scraper import NewsScraper

app = FastAPI(title="Stock Insight Agent", version="1.0")

# --- CONFIGURATION ---
# Using Host Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# --- AUTO-PULL MODEL ---
def ensure_model_exists():
    try:
        print(f"Checking for model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}...")
        for _ in range(5):
            try:
                res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                if res.status_code == 200:
                    break
            except:
                print("Waiting for Ollama service...")
                time.sleep(2)
        
        # Guard: if res is not defined due to loop failure
        if 'res' not in locals():
             print("Could not connect to Ollama.")
             return

        existing_models = [m['name'] for m in res.json().get('models', [])]
        found = any(OLLAMA_MODEL in m for m in existing_models)
        
        if not found:
            print(f"Model {OLLAMA_MODEL} not found. Pulling... (This may take a while)")
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": OLLAMA_MODEL}, stream=True)
            for line in resp.iter_lines():
                if line:
                    print(f"Pulling: {line.decode('utf-8')}")
            print("Model pulled successfully.")
        else:
            print(f"Model {OLLAMA_MODEL} is ready.")
            
    except Exception as e:
        print(f"Warning: Failed to check/pull model: {e}. Agent may fail if model is missing.")

# Run check on startup
ensure_model_exists()

# --- INITIALIZE AGENT ---
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


# --- TOOLS ---
model_inspector = ModelInspector()
drift_auditor = DriftAuditor()
news_scraper = NewsScraper()

class AnalysisRequest(BaseModel):
    ticker: str

@app.get("/")
def home():
    return {"status": "Agent Service Ready"}

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest):
    """
    Deterministic Chain using In-Process GGUF Model.
    """
    ticker = req.ticker.upper()
    
    # Step 1: Technicals
    try:
        tech_data = model_inspector.run(ticker)
    except Exception as e:
        tech_data = f"Error fetching technicals: {e}"

    # Step 2: Drift
    try:
        drift_data = drift_auditor.run({"ignore": ""})
    except Exception as e:
        drift_data = f"Error checking drift: {e}"

    # Step 3: News
    try:
        news_context = news_scraper.run(f"{ticker} stock news financial analysis")
    except Exception as e:
        news_context = "News search unavailable. Rely on technicals."

    # Step 4: Synthesis
    # Mistral Instruct Prompt Format
    prompt = (
        f"Tu es un Analyste Financier Expert francophone. Synthétise les données suivantes pour {ticker} en un paragraphe clair et concis en FRANÇAIS.\n\n"
        f"--- SOURCES DE DONNÉES ---\n"
        f"1. MODÈLE TECHNIQUE : {tech_data}\n"
        f"2. VÉRIFICATION FIABILITÉ (DRIFT) : {drift_data}\n"
        f"3. ACTUALITÉS MARCHÉ : {news_context}\n"
        f"--------------------\n\n"
        f"Ton analyse doit :\n"
        f"- Commencer par le signal du modèle et sa confiance.\n"
        f"- Mentionner si le modèle est fiable (Drift).\n"
        f"- Ajouter du contexte provenant des news.\n"
        f"- Fournir une recommandation finale : ACHETER, VENDRE ou ATTENDRE.\n"
        f"- Expliquer pourquoi.\n"
        f"Réponds UNIQUEMENT en Français. Maximum 150 mots."
    )
    
    try:
        # ChatOllama returns a BaseMessage usually when invoked, access .content
        from langchain.schema import HumanMessage
        print(f"--- PROMPT ---\n{prompt}\n--- END PROMPT ---", flush=True)
        
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"--- RAW RESPONSE TYPE: {type(response)} ---", flush=True)
        print(f"--- RAW CONTENT repr() ---\n{repr(response.content)}\n--- END CONTENT ---", flush=True)
        
        return {"analysis": response.content}
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"analysis": f"Analysis failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
