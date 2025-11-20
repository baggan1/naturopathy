# Inside your /api/app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json
import time, asyncio
from openai import OpenAI

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# --------------------
# CORS SETTINGS
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # TEMPORARY — restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Preflight for Squarespace
@app.options("/fetch_naturopathy_results")
async def preflight_handler():
    return {"status": "ok"}


# --------------------
# SECRETS & CONFIGS
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")
EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed"
)


# --------------------
# ANALYTICS LOGGING
# --------------------
async def log_analytics(data):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/analytics_logs",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json=data
            )
    except Exception as e:
        print("Analytics logging failed:", str(e))


# --------------------
# MAIN ENDPOINT
# --------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):

    start_time = time.time()

    try:
        # ------- AUTH FIRST -------
        auth = request.headers.get("X-API-KEY")
        if auth != SECRET:
            return {"error": "Unauthorized"}

        # Only read JSON once
        body = await request.json()

        query = body.get("query", "").strip()
        if not query:
            return {"error": "Missing 'query' field."}

        # ----------------------------
        # 1. Embedding
        # ----------------------------
        async with httpx.AsyncClient(timeout=60.0) as client:
            emb_res = await client.post(
                EMBEDDING_API,
                json={"query": query}
            )

        if emb_res.status_code != 200:
            return {"error": "Embedding API failed", "details": emb_res.text}

        embed_json = emb_res.json()

        if "embedding" in embed_json:
            query_embedding = embed_json["embedding"]
        elif "data" in embed_json and "embedding" in embed_json["data"]:
            query_embedding = embed_json["data"]["embedding"]
        else:
            return {"error": "Invalid embedding response from HF", "raw": embed_json}

        # Flatten nested vectors
        if isinstance(query_embedding, list) and len(query_embedding) == 1 and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]


        # ----------------------------
        # 2. Supabase Vector Search
        # ----------------------------
        rpc_payload = {
            "query_embedding": json.dumps(query_embedding),
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

        matches = response.json()

        if not matches:
            return {"message": "No matches found."}

        # ----------------------------
        # 3. LLM Summarization
        # ----------------------------
        chunks_text = "\n\n".join([f"- {m['chunk']}" for m in matches])

        prompt = f"""
You are Nani-AI, a warm and clear naturopathy assistant.

Summarize helpful remedies for: {query}

Source Text:
{chunks_text}

Instructions:
- Give 4–6 specific remedies
- Use bullet points
- Be concise but friendly
- Include diet, lifestyle, hydrotherapy, or home remedies
- Do NOT mention documents or sources
"""

        ai_res = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_summary = ai_res.choices[0].message.content

        DISCLAIMER = (
            "⚠️ Disclaimer: Nani-AI provides general wellness suggestions derived from "
            "Naturopathy and Ayurvedic principles. This is not medical advice. For severe, "
            "persistent, or emergency symptoms, consult a licensed healthcare professional."
        )

        summary = raw_summary + "\n\n---\n" + DISCLAIMER


        # ----------------------------
        # 4. ANALYTICS (async)
        # ----------------------------
        latency = int((time.time() - start_time) * 1000)

        analytics_payload = {
            "query": query,
            "used_embeddings": True,
            "used_google_search": False,
            "sources": [m["source"] for m in matches],
            "match_count": len(matches),
            "latency_ms": latency
        }

        asyncio.create_task(log_analytics(analytics_payload))


        # ----------------------------
        # RETURN
        # ----------------------------
        return {
            "query": query,
            "summary": summary,
            "sources": [m["source"] for m in matches]
        }

    except Exception as e:
        return {"error": f"Server exception: {str(e)}"}

