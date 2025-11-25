# /api/app.py  (Render)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json
import time, asyncio
from datetime import datetime, timezone

from openai import OpenAI

# -----------------------
# CLIENTS & CONFIG
# -----------------------
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")   # for GPT Actions / legacy X-API-KEY

EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed"
)

# -----------------------
# HELPER: Supabase Auth
# -----------------------

async def get_supabase_user(access_token: str):
    """
    Validate a Supabase JWT and return {id, email} or None.
    """
    if not access_token:
        return None

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {access_token}",
            },
        )

    if resp.status_code != 200:
        print("Auth user fetch failed:", resp.status_code, resp.text)
        return None

    data = resp.json()
    return {
        "id": data.get("id"),
        "email": data.get("email"),
    }


async def get_profile(user_id: str):
    """
    Fetch profile row from Supabase 'profiles' table by user id.
    """
    if not user_id:
        return None

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
            params={
                "id": f"eq.{user_id}",
                "select": "id,email,trial_start,trial_end,subscribed,stripe_customer_id",
            },
        )

    if resp.status_code != 200:
        print("Profile fetch failed:", resp.status_code, resp.text)
        return None

    rows = resp.json()
    if not rows:
        return None
    return rows[0]


def evaluate_access(profile: dict):
    """
    Given a profile row, decide access:
      - trial_active
      - days_left
      - subscribed
      - allowed (bool)
      - message (if blocked)
    """
    if not profile:
        return {
            "allowed": False,
            "trial_active": False,
            "days_left": 0,
            "subscribed": False,
            "message": "No profile found. Please sign out and sign in again.",
        }

    subscribed = bool(profile.get("subscribed", False))
    trial_end_raw = profile.get("trial_end")
    trial_active = False
    days_left = 0

    if trial_end_raw:
        # handle ISO string, possibly with Z
        try:
            trial_end_str = str(trial_end_raw)
            if trial_end_str.endswith("Z"):
                trial_end_str = trial_end_str.replace("Z", "+00:00")
            trial_end = datetime.fromisoformat(trial_end_str)
            now = datetime.now(timezone.utc)
            delta = trial_end - now
            days_left = max(delta.days, 0)
            trial_active = days_left > 0
        except Exception as e:
            print("Error parsing trial_end:", e)

    # Option A logic:
    # allow IF (trial_active OR subscribed), otherwise block
    if subscribed or trial_active:
        return {
            "allowed": True,
            "trial_active": trial_active,
            "days_left": days_left,
            "subscribed": subscribed,
            "message": None,
        }

    return {
        "allowed": False,
        "trial_active": False,
        "days_left": 0,
        "subscribed": subscribed,
        "message": "Your free trial has ended. Please subscribe to continue using Nani-AI.",
    }


# -----------------------
# ANALYTICS LOGGING
# -----------------------
async def log_analytics(data):
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


# -----------------------
# PREFLIGHT (Squarespace/PWA)
# -----------------------
@app.options("/fetch_naturopathy_results")
async def preflight_handler():
    return {"status": "ok"}


# -----------------------
# AUTH STATUS ENDPOINT
# -----------------------
@app.get("/auth/status")
async def auth_status(request: Request):
    """
    Used by the PWA account panel to display:
    - email
    - trial_active
    - days_left
    - subscribed
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"error": "Missing or invalid Authorization header"}

    access_token = auth_header.split(" ", 1)[1].strip()
    supa_user = await get_supabase_user(access_token)
    if not supa_user:
        return {"error": "Invalid token"}

    profile = await get_profile(supa_user["id"])
    eval_result = evaluate_access(profile)

    return {
        "email": supa_user.get("email"),
        "trial_active": eval_result["trial_active"],
        "days_left": eval_result["days_left"],
        "subscribed": eval_result["subscribed"],
    }


# -----------------------
# MAIN NANI-AI ENDPOINT
# -----------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    start_time = time.time()

    try:
        # 1. AUTHENTICATION
        auth_header = request.headers.get("Authorization", "")
        x_api_key = request.headers.get("X-API-KEY")

        supa_user = None
        profile = None
        user_id = None
        user_email = None

        if auth_header.startswith("Bearer "):
            # Normal PWA user path (Supabase Auth)
            access_token = auth_header.split(" ", 1)[1].strip()
            supa_user = await get_supabase_user(access_token)
            if not supa_user:
                return {"error": "Invalid or expired session. Please log in again."}

            user_id = supa_user["id"]
            user_email = supa_user.get("email")

            # gate via trial/subscription
            profile = await get_profile(user_id)
            eval_result = evaluate_access(profile)

            if not eval_result["allowed"]:
                return {"error": eval_result["message"]}

        elif x_api_key and x_api_key == SECRET:
            # GPT Action / internal use
            user_id = "system"
            user_email = None
            # no trial gating here
        else:
            return {"error": "Unauthorized"}

        # 2. BODY / QUERY
        body = await request.json()
        query = (body.get("query") or "").strip()
        if not query:
            return {"error": "Missing 'query' field."}

        # --------------------
        # 3. HF Embedding
        # --------------------
        async with httpx.AsyncClient(timeout=60.0) as client:
            emb_res = await client.post(
                EMBEDDING_API,
                json={"query": query},
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

        # Flatten nested vector
        if (
            isinstance(query_embedding, list)
            and len(query_embedding) == 1
            and isinstance(query_embedding[0], list)
        ):
            query_embedding = query_embedding[0]

        # --------------------
        # 4. Supabase Vector Search (RAG)
        # --------------------
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

        matches = response.json() or []
        similarities = [m.get("similarity", 0.0) for m in matches]
        max_similarity = max(similarities) if similarities else 0.0

        # ----------------------------
        # 5. HYBRID LOGIC
        # ----------------------------
        chunks_text = "\n\n".join([f"- {m['chunk']}" for m in matches]) if matches else ""

        rag_used = False
        llm_used = False
        mode = "LLM_ONLY"

        # High confidence => RAG primary
        if matches and max_similarity >= 0.70:
            rag_used = True
            llm_used = True
            mode = "RAG_ONLY"
            final_prompt = f"""
You are Nani-AI.

Provide clear, actionable naturopathy + ayurveda guidance for the query:
{query}

Use the following retrieved text:
{chunks_text}

Instructions:
- Summarize the retrieved content
- Provide 4–6 bullet-point remedies
- Include diet, lifestyle, hydrotherapy or simple home practices
- Keep the tone warm and simple
"""
        # Medium confidence => Hybrid
        elif matches and 0.40 <= max_similarity < 0.70:
            rag_used = True
            llm_used = True
            mode = "HYBRID"
            final_prompt = f"""
You are Nani-AI.

User query:
{query}

We found related information but confidence is moderate.
Blend retrieved naturopathy text with your own ayurvedic reasoning.

Retrieved text:
{chunks_text}

Instructions:
- Start with the best information from the retrieved text
- Add LLM reasoning to fill gaps
- Provide 4–6 simple, safe remedies
- Include food, herbs, habits and daily practices
"""
        # Low/no RAG => Pure LLM
        else:
            rag_used = False
            llm_used = True
            mode = "LLM_ONLY"
            final_prompt = f"""
You are Nani-AI.

No reliable text matches were found in the RAG database for:
{query}

Generate an ayurveda + naturopathy–based answer from scratch.

Instructions:
- Provide 4–6 bullet-point remedies
- Include diet, herbs, lifestyle, and gentle home treatments
- Keep it simple, safe, and practical
"""

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
        # 6. Analytics Logging
        # --------------------
        latency = int((time.time() - start_time) * 1000)
        matched_sources = [m["source"] for m in matches] if matches else []

        analytics_payload = {
            "query": query,
            "match_count": len(matches),
            "max_similarity": float(max_similarity),
            "sources": matched_sources,
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
            "latency_ms": latency,
            "user_id": user_id,
            "user_email": user_email,
        }

        asyncio.create_task(log_analytics(analytics_payload))

        return {
            "query": query,
            "summary": summary,
            "sources": matched_sources,
            "match_count": len(matches),
            "max_similarity": max_similarity,
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
        }

    except Exception as e:
        print("Server exception:", str(e))
        return {"error": f"Server exception: {str(e)}"}
