# /api/app.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json, time, asyncio
from datetime import datetime, timezone
from openai import OpenAI

# -------------------------
# OPENAI CLIENT
# -------------------------
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# FASTAPI
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/{path:path}")
async def preflight():
    return {"status": "ok"}


# -------------------------
# ENV VARS
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")

EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed",
)


# ====================================================================================
#                               SUPABASE AUTH HELPERS
# ====================================================================================
async def get_supabase_user(access_token: str):
    """
    Given the Supabase access token (from frontend), return user info.
    """
    if not access_token:
        return None

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {access_token}",
            },
        )

    if resp.status_code != 200:
        print("Supabase auth user error:", resp.text)
        return None

    return resp.json()


async def get_or_create_profile(user_id: str, email: str):
    """
    Fetch profile row from Supabase.
    If missing, auto-create it → this activates 7-day free trial.
    """
    if not user_id:
        return None

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Check existing profile
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"select": "*", "id": f"eq.{user_id}", "limit": 1},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )

        rows = resp.json() if resp.status_code == 200 else []
        if rows:
            return rows[0]

        # 2. Create new profile → auto trial
        insert_resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={"id": user_id, "email": email},
        )

        if insert_resp.status_code not in (200, 201):
            print("Profile insert error:", insert_resp.text)
            return None

        created = insert_resp.json()
        return created[0] if created else None


def compute_trial_status(profile: dict):
    """Calculate trial_active, days_left, subscribed."""
    subscribed = bool(profile.get("subscribed", False))
    trial_active = False
    days_left = 0

    trial_end_str = profile.get("trial_end")
    if trial_end_str:
        try:
            if trial_end_str.endswith("Z"):
                trial_end = datetime.fromisoformat(trial_end_str.replace("Z", "+00:00"))
            else:
                trial_end = datetime.fromisoformat(trial_end_str)

            now = datetime.now(timezone.utc)
            delta = trial_end - now

            trial_active = delta.total_seconds() > 0
            days_left = max(0, delta.days)

        except Exception as e:
            print("trial_end parse error:", e)

    if subscribed:
        trial_active = True

    return {
        "trial_active": trial_active,
        "days_left": days_left,
        "subscribed": subscribed,
        "trial_start": profile.get("trial_start"),
        "trial_end": profile.get("trial_end"),
    }


# ====================================================================================
#                           AUTH STATUS ENDPOINT (FRONTEND USES THIS)
# ====================================================================================
@app.get("/auth/status")
async def auth_status(request: Request):
    """
    Frontend calls this using:
    - Authorization: Bearer <supabase access_token>
    - X-API-KEY: internal API secret

    Returns:
    - trial_active
    - days_left
    - subscribed
    """
    # Internal API security
    api_key = request.headers.get("X-API-KEY")
    if api_key != SECRET:
        return {"error": "Unauthorized"}

    # Supabase token check
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"error": "Missing auth token"}

    access_token = auth_header.split(" ", 1)[1]

    # Get user from Supabase
    user = await get_supabase_user(access_token)
    if not user:
        return {"error": "Invalid token"}

    user_id = user.get("id")
    email = user.get("email") or user.get("user_metadata", {}).get("email")

    # Ensure profile exists (auto-create trial on first login)
    profile = await get_or_create_profile(user_id, email)
    if not profile:
        return {"error": "Profile error"}

    trial_info = compute_trial_status(profile)

    return {"email": email, **trial_info}


# ====================================================================================
#                                 ANALYTICS LOGGING
# ====================================================================================
async def log_analytics(data: dict):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/analytics_logs",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                json=data,
            )
    except Exception as e:
        print("Analytics error:", e)


# ====================================================================================
#                               MAIN NANI-AI ENDPOINT
# ====================================================================================
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    """
    Uses:
    - Supabase Auth token (in Authorization header)
    - X-API-KEY internal secret
    - Checks trial/subscription status
    """
    start = time.time()
    print("STEP: Start")

    # --- X-API-KEY security ---
    if request.headers.get("X-API-KEY") != SECRET:
        return {"error": "Unauthorized"}

    # --- Body ---
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return {"error": "Missing query"}

    # ---------------------------
    # AUTHENTICATE USER
    # ---------------------------
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"error": "Missing auth token"}

    token = auth_header.split(" ", 1)[1]
    user = await get_supabase_user(token)

    if not user:
        return {"error": "Invalid session. Please sign in again."}

    user_id = user.get("id")
    email = user.get("email")

    # Ensure profile exists and get trial/sub status
    profile = await get_or_create_profile(user_id, email)
    trial_info = compute_trial_status(profile)

    if not (trial_info["trial_active"] or trial_info["subscribed"]):
        return {
            "error": "Trial expired",
            **trial_info
        }

    # ====================================================================================
    # 1. GENERATE EMBEDDING
    # ====================================================================================
    async with httpx.AsyncClient(timeout=60) as client:
        emb = await client.post(EMBEDDING_API, json={"query": query})

    if emb.status_code != 200:
        return {"error": "Embedding API failure", "details": emb.text}

    emb_json = emb.json()
    embedding = (
        emb_json.get("embedding")
        or emb_json.get("data", {}).get("embedding")
    )

    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
        embedding = embedding[0]
    
    print("STEP: Embedding", time.time() - start_step)
    # ====================================================================================
    # 2. VECTOR SEARCH
    # ====================================================================================
    rpc_payload = {
        "query_embedding": json.dumps(embedding),
        "match_threshold": body.get("match_threshold", 0.4),
        "match_count": body.get("match_count", 3),
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_documents_v2",
            json=rpc_payload,
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            },
        )

    if resp.status_code != 200:
        return {"error": "Supabase vector failure", "details": resp.text}

    matches = resp.json()
    similarities = [m["similarity"] for m in matches] if matches else []
    max_sim = max(similarities) if similarities else 0

    rag_used = False
    llm_used = True
    mode = "LLM_ONLY"

    chunks_text = "\n\n".join([m["chunk"][:800] for m in matches]) if matches else ""
    
    print("STEP: Vector Search", time.time() - start_step)
    # ====================================================================================
    # 3. PROMPT SELECTION
    # ====================================================================================
    if matches and max_sim >= 0.70:
        mode = "RAG_ONLY"
        rag_used = True
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
- Instead of using Vata, Pitta, Kapha terms in Ayurveda, use Wind, Fire, Water or Earth energy instead.

"""
    elif matches and max_sim >= 0.40:
        mode = "HYBRID"
        rag_used = True
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
- Instead of using Vata, Pitta, Kapha terms in Ayurveda, use Wind, Fire, Water or Earth energy instead.
- Keep it safe, non-alarming, and easy to follow
"""
    else:
        mode = "LLM_ONLY"
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
- Instead of using Vata, Pitta, Kapha terms in Ayurveda, use Wind, Fire, Water or Earth energy instead.
"""

    # ====================================================================================
    # 4. GENERATE SUMMARY
    # ====================================================================================
    ai = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
    )
    summary = ai.choices[0].message.content

    summary += "\n\n⚠️ Disclaimer: Nani-AI provides general wellness guidance..."
    
    print("STEP: OpenAI Completion", time.time() - start_step)
    # ====================================================================================
    # 5. ANALYTICS
    # ====================================================================================
    latency = int((time.time() - start) * 1000)

    asyncio.create_task(
        log_analytics({
            "query": query,
            "match_count": len(matches),
            "max_similarity": max_sim,
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
            "sources": [m["source"] for m in matches] if matches else [],
            "latency_ms": latency,
        })
    )

    # ====================================================================================
    # 6. RETURN
    # ====================================================================================
    return {
        "query": query,
        "summary": summary,
        "match_count": len(matches),
        "max_similarity": max_sim,
        "rag_used": rag_used,
        "llm_used": llm_used,
        "mode": mode,
    }
