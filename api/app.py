# Inside your /api/app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, os

app = FastAPI()

# --------------------
# CORS SETTINGS
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://arkayoga.com", "https://www.arkayoga.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")

# Hugging Face embedder URL
EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed"
)

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    try:
        body = await request.json()
        
        # Auth
        if body.get("auth_key") != SECRET:
           return {"error": "Unauthorized"}
            
        # Extract & clean query
        query = body.get("query", "").strip()
        if not query:
            return {"error": "Missing 'query' field."}

        # --------------------
        # 1. Call HuggingFace for Embedding
        # --------------------
        async with httpx.AsyncClient(timeout=60.0) as client:
            emb_res = await client.post(
                EMBEDDING_API,
                json={"query": query}
            )
        # Do NOT use raise_for_status()    
        if emb_res.status_code != 200:
            return {"error": "Embedding API failed", "details": emb_res.text}
            
        query_embedding = emb_res.json()["embedding"]

        # --------------------
        # 2. Now query Supabase RPC (match_documents_v2)
        # --------------------
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
