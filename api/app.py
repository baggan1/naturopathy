# ==============================
# api/app.py  (FINAL VERIFIED)
# ==============================

from fastapi import FastAPI, Request
import httpx, os
from pathlib import Path
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ----------------------------------------------------
# Environment Variables for Supabase
# ----------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# ----------------------------------------------------
# Load or Auto-Download the Model
# ----------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

if not MODEL_DIR.exists():
    print("📦 Model folder not found — downloading fresh copy from Hugging Face...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save(str(MODEL_DIR))
else:
    print(f"✅ Found local model at: {MODEL_DIR}")

# Load the local model
encoder = SentenceTransformer(str(MODEL_DIR))
print("✅ SentenceTransformer model loaded successfully.")

# ----------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Naturopathy API is running on Render."}

# ----------------------------------------------------
# Main Endpoint for GPT or External Calls
# ----------------------------------------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    """
    Takes a JSON body with:
      - query: the natural-language question or ailment
      - match_threshold (optional): float
      - match_count (optional): int
    Returns top matching naturopathy remedies from Supabase.
    """
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            return {"error": "Missing 'query' field."}

        query_embedding = encoder.encode([query])[0].tolist()

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
