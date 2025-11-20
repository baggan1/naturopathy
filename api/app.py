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
    allow_origins=["*"],     # TEMPORARY (enable per-domain later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Handle Squarespace preflight
@app.options("/fetch_naturopathy_results")
async def preflight_handler():
    return {"status": "ok"}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")

# Hugging Face embedder URL
EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed"
)

# --------------------
# ANALYTICS LOGGING FUNCTION
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

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    start_time = time.time()
    try:
        body = await request.json()

        # Auth
        if body.get("auth_key") != SECRET:
            return {"error": "Unauthorized"}

        query = body.get("query", "").strip()
        if not query:
            return {"error": "Missing 'query' field."}

        # --------------------
        # 1. HF Embedding
        # --------------------
        async with httpx.AsyncClient(timeout=60.0) as client:
            emb_res = await client.post(
                EMBEDDING_API,
                json={"query": query}
            )

        if emb_res.status_code != 200:
            return {"error": "Embedding API failed", "details": emb_res.text}

        embed_json = emb_res.json()
        # HF Spaces sometimes return {"embedding": [...]} or {"data": {"embedding": [...]}}
        if "embedding" in embed_json:
            query_embedding = embed_json["embedding"]
        elif "data" in embed_json and "embedding" in embed_json["data"]:
            query_embedding = embed_json["data"]["embedding"]
        else:
            return {"error": "Invalid embedding response from HF", "raw": embed_json}
        # Flatten if nested
        if isinstance(query_embedding, list) and len(query_embedding) == 1 and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
        print("Embedding length:", len(query_embedding))

        # --------------------
        # 2. Supabase RPC
        # --------------------
        rpc_payload = {
            "query_embedding": json.dumps(query_embedding),  # ← stringify the vector
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
       
        print("Sending vector type:", type(query_embedding), "length:", len(query_embedding))
      
        # --------------------
        # 3. LLM Summarization Layer (Nani-AI Magic)
        # --------------------
        chunks_text = "\n\n".join(
            [f"- {m['chunk']}" for m in matches]
        )

        prompt = f"""
You are Nani-AI, a warm, clear Naturopathy assistant.

Summarize the following naturopathy text into **simple, actionable remedies** for: {query}

Text:
{chunks_text}

Instructions:
- Give 4–6 specific remedies
- Use bullet points
- Be concise but friendly
- Include diet, lifestyle, hydrotherapy or home remedies
- Do NOT mention 'chunks' or 'documents'
- Make it easy to follow

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
        
        # --------------------
        # Analytics Logging
        # --------------------
        latency = int((time.time() - start_time) * 1000)

        # Extract sources
        matched_sources = [m["source"] for m in matches] if matches else []

        # Determine what was used
        used_embeddings = True if matches else False
        used_google_search = False  # You can flip this if you add fallback

        # Analytics payload
        analytics_payload = {
            "query": query,
            "used_embeddings": used_embeddings,
            "used_google_search": used_google_search,
            "sources": matched_sources,
            "match_count": len(matches) if matches else 0,
            "latency_ms": latency
        }

        # Fire and forget logging
        asyncio.create_task(log_analytics(analytics_payload))

        # Now return your response
        return {
            "query": query,
            "summary": summary,
            "sources": matched_sources
        }

    except Exception as e:
        return {"error": f"Server exception: {str(e)}"}
