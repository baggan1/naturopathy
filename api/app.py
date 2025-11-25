# /api/app.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json
import time, asyncio
from datetime import datetime, timezone
from openai import OpenAI

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# --------------------
# CORS SETTINGS
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Preflight for Squarespace / PWA
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
async def log_analytics(data: dict):
    """
    Fire-and-forget logging into analytics_logs.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/analytics_logs",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=data,
            )
    except Exception as e:
        print("Analytics logging failed:", str(e))


# --------------------
# PROFILE LOOKUP HELPER
# --------------------
async def get_profile(email: str):
    """
    Fetch a single profile row by email from Supabase.
    Relies on your `profiles` table and handle_new_user() trigger.
    """
    if not email:
        return None

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={
                    "email": f"eq.{email}",
                    "select": "*",
                    "limit": 1,
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )

        if resp.status_code != 200:
            print("Profile lookup failed:", resp.status_code, resp.text)
            return None

        rows = resp.json()
        if not rows:
            return None
        return rows[0]

    except Exception as e:
        print("Profile lookup exception:", str(e))
        return None


# --------------------
# AUTH STATUS ENDPOINT
# --------------------
@app.get("/auth/status")
async def auth_status(email: str):
    """
    Return trial/subscription status for the given email.
    Used by the Account panel in the PWA.
    """
    profile = await get_profile(email)
    if not profile:
        return {"error": "no-profile"}

    trial_end_str = profile.get("trial_end")
    subscribed = bool(profile.get("subscribed"))
    now = datetime.now(timezone.utc)

    trial_active = False
    days_left = 0

    if trial_end_str:
        try:
            trial_end = datetime.fromisoformat(trial_end_str.replace("Z", "+00:00"))
            if trial_end > now:
                trial_active = True
                days_left = max(0, (trial_end - now).days)
        except Exception as e:
            print("Error parsing trial_end:", e)

    # Subscription always overrides trial
    if subscribed:
        trial_active = True

    return {
        "email": email,
        "trial_active": trial_active,
        "days_left": days_left,
        "subscribed": subscribed,
    }


# --------------------
# MAIN NANI-AI ENDPOINT
# --------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    start_time = time.time()

    try:
        # ------- AUTH HEADER -------
        auth = request.headers.get("X-API-KEY")
        if auth != SECRET:
            return {"error": "Unauthorized"}

        # ------- BODY -------
        body = await request.json()
        email = body.get("email", "").strip()
        query = body.get("query", "").strip()

        if not email:
            return {"error": "Missing email. Please sign in again."}
        if not query:
            return {"error": "Missing 'query' field."}

        # ------- PROFILE & TRIAL CHECK -------
        profile = await get_profile(email)
        if not profile:
            return {"error": "No profile found. Please sign out and sign in again."}

        trial_end_str = profile.get("trial_end")
        subscribed = bool(profile.get("subscribed"))
        now = datetime.now(timezone.utc)

        trial_expired = False
        if trial_end_str:
            try:
                trial_end = datetime.fromisoformat(trial_end_str.replace("Z", "+00:00"))
                if trial_end <= now:
                    trial_expired = True
            except Exception as e:
                print("Error parsing trial_end in main handler:", e)

        if trial_expired and not subscribed:
            return {
                "error": "Your free trial has expired. Please subscribe to continue using Nani-AI.",
                "trial_expired": True,
            }

        # ----------------------------
        # 1. HF Embedding
        # ----------------------------
        async with httpx.AsyncClient(timeout=60.0) as client:
            emb_res = await client.post(
                EMBEDDING_API,
                json={"query": query},
            )

        if emb_res.status_code != 200:
            return {
                "error": "Embedding API failed",
                "details": emb_res.text,
            }

        embed_json = emb_res.json()
        if "embedding" in embed_json:
            query_embedding = embed_json["embedding"]
        elif "data" in embed_json and "embedding" in embed_json["data"]:
            query_embedding = embed_json["data"]["embedding"]
        else:
            return {"error": "Invalid embedding response", "raw": embed_json}

        # Flatten nested vectors
        if (
            isinstance(query_embedding, list)
            and len(query_embedding) == 1
            and isinstance(query_embedding[0], list)
        ):
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

        # ----------------------------
        # 3. HYBRID LOGIC (RAG + LLM)
        # ----------------------------
        similarities = [m["similarity"] for m in matches] if matches else []
        max_similarity = max(similarities) if similarities else 0.0

        rag_used = False
        llm_used = False
        mode = "LLM_ONLY"

        chunks_text = (
            "\n\n".join([f"- {m['chunk']}" for m in matches]) if matches else ""
        )

        # --- CASE 1: HIGH CONFIDENCE → RAG-ONLY SUMMARY ---
        if matches and max_similarity >= 0.70:
            mode = "RAG_ONLY"
            rag_used = True
            llm_used = True   # summarizer is still LLM

            final_prompt = f"""
You are Nani-AI, a warm, clear Naturopathy & Ayurveda guide.

User query:
{query}

Use the following retrieved naturopathy text as your primary source:
{chunks_text}

Instructions:
- Summarize the retrieved content first
- Provide 4–6 clearly separated bullet-point remedies
- Include food, herbs, lifestyle, and simple home practices
- Keep the language gentle, simple, and practical
"""

        # --- CASE 2: MEDIUM CONFIDENCE → HYBRID (RAG + LLM) ---
        elif matches and 0.40 <= max_similarity < 0.70:
            mode = "HYBRID"
            rag_used = True
            llm_used = True

            final_prompt = f"""
You are Nani-AI, a naturopathy + ayurveda assistant.

User query:
{query}

We found related but not perfect matches. Blend them with your own clinical-style reasoning.

Retrieved text:
{chunks_text}

Instructions:
- Start from the best RAG insights you see
- Add your own ayurvedic & naturopathic reasoning to fill gaps
- Provide 4–6 bullet-point remedies
- Include food, herbs, daily routine, and home treatments
- Keep it safe, non-alarming, and easy to follow
"""

        # --- CASE 3: LOW OR NO MATCHES → PURE LLM ---
        else:
            mode = "LLM_ONLY"
            rag_used = False
            llm_used = True

            final_prompt = f"""
You are Nani-AI, an Ayurvedic + Naturopathy wellness guide.

No reliable matches were found in the database for:
{query}

Generate a fresh answer from your ayurveda + naturopathy knowledge.

Instructions:
- Provide 4–6 bullet-point remedies
- Include diet, herbs, lifestyle, and home treatments
- Focus on gentle, preventive, non-emergency guidance
- Avoid mentioning 'database' or 'documents'
"""

        # --- LLM CALL ---
        ai_res = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}],
        )
        raw_summary = ai_res.choices[0].message.content

        DISCLAIMER = (
            "⚠️ Disclaimer: Nani-AI provides general wellness suggestions derived from "
            "Naturopathy and Ayurvedic principles. This is not medical advice. For severe, "
            "persistent, or emergency symptoms, consult a licensed healthcare professional."
        )

        summary = raw_summary + "\n\n---\n" + DISCLAIMER

        # --------------------
        # 4. Analytics Logging
        # --------------------
        latency = int((time.time() - start_time) * 1000)
        matched_sources = [m["source"] for m in matches] if matches else []

        analytics_payload = {
            "query": query,
            "match_count": len(matches) if matches else 0,
            "max_similarity": float(max_similarity),
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
            "sources": matched_sources,
            "latency_ms": latency,
        }

        asyncio.create_task(log_analytics(analytics_payload))

        # --------------------
        # 5. Final Response
        # --------------------
        return {
            "query": query,
            "summary": summary,
            "sources": matched_sources,
            "match_count": len(matches) if matches else 0,
            "max_similarity": float(max_similarity),
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
        }

    except Exception as e:
        return {"error": f"Server exception: {str(e)}"}
