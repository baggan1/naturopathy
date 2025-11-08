from fastapi import FastAPI, Request
import httpx, os
from sentence_transformers import SentenceTransformer

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# ✅ Load embedding model once (MiniLM-L6-v2 → 384-dim)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    """
    Secure proxy: receives raw text query from GPT or Postman,
    embeds it into a 384-d vector, and forwards to Supabase.
    """
    body = await request.json()

    # If user sent text, embed it
    if "query" in body and isinstance(body["query"], str):
        query_embedding = encoder.encode([body["query"]])[0].tolist()
    else:
        # Backward compatibility: accept direct embedding too
        query_embedding = body.get("query_embedding", [])

    payload = {
        "query_embedding": query_embedding,
        "match_threshold": body.get("match_threshold", 0.5),
        "match_count": body.get("match_count", 3),
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_documents_v2",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )

    return res.json()
@app.get("/")
def health():
    return {"status": "ok", "message": "Naturopathy proxy is running!"}

