# Inside your /api/app.py
from fastapi import FastAPI, Request
import httpx, os

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")

# Your Hugging Face Space URL
EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed"
)

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    try:
        body = await request.json()
        if body.get("auth_key") != SECRET:
           return {"error": "Unauthorized"}
        query = body.get("query")
        if not query:
            return {"error": "Missing 'query' field."}

        # 🔹 Call your Hugging Face Space for embeddings
        async with httpx.AsyncClient(timeout=60.0) as client:
            emb_res = await client.post(
                EMBEDDING_API,
                json={"query": query}
            )
            emb_res.raise_for_status()
            query_embedding = emb_res.json()["embedding"]

        # 🔹 Now query Supabase using the embedding
        rpc_payload = {
            "query_embedding": query_embedding,
            "match_threshold": body.get("match_threshold", 0.4),
            "match_count": body.get("match_count", 3),
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_documents_v2",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json=rpc_payload,
            )

        if response.status_code != 200:
            return {
                "error": "Supabase query failed",
                "status_code": response.status_code,
                "details": response.text,
            }

        data = response.json()
        if not data:
            return {"message": "No matches found for that query."}

        return {
            "query": query,
            "results": data,
            "note": (
                "⚠️ These are natural remedies from Naturopathy sources. "
                "For serious or emergency conditions, please consult a healthcare professional."
            ),
        }

    except Exception as e:
        return {"error": f"Server exception: {str(e)}"}
