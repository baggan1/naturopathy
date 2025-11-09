from fastapi import FastAPI, Request
import httpx, os
from sentence_transformers import SentenceTransformer

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

encoder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

@app.get("/")
def root():
    return {"status": "ok", "message": "Naturopathy Proxy API is running."}

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return {"error": "Missing 'query' field."}

    # Generate embedding
    query_embedding = encoder.encode([query])[0].tolist()

    async with httpx.AsyncClient() as client:
        res = await client.post(
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

    return res.json()
