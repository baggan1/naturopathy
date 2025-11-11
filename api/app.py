# =====================================
# api/app.py — Hugging Face hosted version
# =====================================
from fastapi import FastAPI, Request
import httpx, os
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ----------------------------------------------------
# Environment Variables
# ----------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional: your Hugging Face Access Token
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------------------
# Load model from Hugging Face (hosted repo)
# ----------------------------------------------------
def load_hf_model():
    """
    Load the SentenceTransformer model directly from Hugging Face.
    Uses your own repo if defined in HF_MODEL_REPO.
    """
    print(f"🔹 Loading model from Hugging Face repo: {HF_MODEL_REPO}")
    try:
        model = SentenceTransformer(HF_MODEL_REPO, use_auth_token=HF_TOKEN)
        print("✅ SentenceTransformer model loaded successfully from Hugging Face.")
        return model
    except Exception as e:
        print(f"❌ Model load failed from Hugging Face: {e}")
        raise e

encoder = load_hf_model()

# ----------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_repo": HF_MODEL_REPO,
        "message": "Naturopathy API is running on Render using Hugging Face model.",
    }

# ----------------------------------------------------
# Main Endpoint for GPT or Clients
# ----------------------------------------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    """
    Fetches embeddings for a query, and matches top remedies from Supabase.
    """
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            return {"error": "Missing 'query' field."}

        # Encode query
        query_embedding = encoder.encode([query])[0].tolist()

        # Query Supabase RPC
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
