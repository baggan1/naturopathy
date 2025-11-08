from fastapi import FastAPI, Request
import httpx, os, traceback
from sentence_transformers import SentenceTransformer

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    try:
        body = await request.json()
        print("📥 Incoming body:", body)

        if "query" in body and isinstance(body["query"], str):
            query_embedding = encoder.encode([body["query"]])[0].tolist()
            print(f"✅ Embedding created (len={len(query_embedding)})")
        else:
            query_embedding = body.get("query_embedding", [])
            print(f"⚠️ Using provided embedding, len={len(query_embedding)}")

        payload = {
            "query_embedding": query_embedding,
            "match_threshold": body.get("match_threshold", 0.5),
            "match_count": body.get("match_count", 3),
        }

        print("🚀 Sending to Supabase:", payload.keys())

        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_documents_v2",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        print("📤 Supabase response:", res.status_code, res.text)
        return res.json()

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        return {"error": str(e)}
@app.get("/")
def health():
    return {"status": "ok", "message": "Naturopathy proxy is running!"}

