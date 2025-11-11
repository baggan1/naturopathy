# ==============================
# api/app.py (Self-Healing Render Version)
# ==============================
from fastapi import FastAPI, Request
import httpx, os
from pathlib import Path
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ----------------------------------------------------
# Environment Variables
# ----------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# ----------------------------------------------------
# Model Path Configuration
# ----------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILE = MODEL_DIR / "0_Transformer" / "model.safetensors"

# ----------------------------------------------------
# Auto-Healing Model Loader
# ----------------------------------------------------
def load_or_download_model():
    """
    Attempts to load local SentenceTransformer model.
    If missing or incomplete, downloads from Hugging Face.
    """
    try:
        if MODEL_FILE.exists():
            print(f"✅ Found model file: {MODEL_FILE}")
            model = SentenceTransformer(str(MODEL_DIR))
        else:
            print("⚠️ model.safetensors not found — downloading fresh model from Hugging Face...")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            model.save(str(MODEL_DIR))
            print(f"✅ Model downloaded and saved to {MODEL_DIR}")
        return model
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        raise e

encoder = load_or_download_model()

# ----------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Naturopathy API is running on Render."}

# ----------------------------------------------------
# Fetch Naturopathy Results
# ----------------------------------------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    """
    Fetches top matching naturopathy remedies from Supabase.
    If Supabase or model fails, returns detailed error message.
    """
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            return {"error": "Missing 'query' field."}

        # Create embedding vector
        query_embedding = encoder.encode([query])[0].tolist()

        # Query Supabase function
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_documents_v2",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query_embedding": query_embedding,
                    "match_threshold": body.get("match_threshold", 0.4),
                    "match_count": body.get("match_count", 3),
                },
            )

        if response.status_code != 200:
            return {
                "error": "Failed to query Supabase",
                "status_code": response.status_code,
                "details": response.text,
            }

        data = response.json()
        if not data:
            return {
                "message": "No matching naturopathy results found in database.",
                "query": query,
            }

        return {
            "query": query,
            "results": data,
            "note": (
                "⚠️ These are natural remedies from Naturopathy sources. "
                "In case of serious or emergency conditions, please consult "
                "a qualified healthcare professional."
            ),
        }

    except Exception as e:
        return {"error": f"Server exception: {e}"}
