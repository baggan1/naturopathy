# /api/app.py (Optimized Version)

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json, time, asyncio
from datetime import datetime, timezone
from openai import OpenAI

# ----------------------------------------------
# OPENAI INIT
# ----------------------------------------------
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------
# FASTAPI + CORS
# ----------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ----------------------------------------------
# ENV VARS
# ----------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")


# ================================================================
# SUPABASE HELPERS
# ================================================================
async def get_supabase_user(access_token: str):
    if not access_token:
        return None
    
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {access_token}"},
        )

    if resp.status_code != 200:
        print("Supabase auth error:", resp.text)
        return None

    return resp.json()


async def get_or_create_profile(user_id: str, email: str):
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"select": "*", "id": f"eq.{user_id}", "limit": 1},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
        )

        rows = resp.json() if resp.status_code == 200 else []

        if rows:
            return rows[0]

        # Create new profile
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

        created = insert_resp.json()
        return created[0] if created else None


def compute_trial_status(profile: dict):
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
            print("Trial parse error:", e)

    if subscribed:
        trial_active = True

    return {
        "trial_active": trial_active,
        "days_left": days_left,
        "subscribed": subscribed,
        "trial_start": profile.get("trial_start"),
        "trial_end": profile.get("trial_end"),
    }


# ================================================================
# AUTH STATUS ENDPOINT
# ================================================================
@app.get("/auth/status")
async def auth_status(request: Request):

    if request.headers.get("X-API-KEY") != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    access_token = auth_header.split(" ", 1)[1]
    user = await get_supabase_user(access_token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    profile = await get_or_create_profile(user["id"], user["email"])
    trial_info = compute_trial_status(profile)

    return {"email": user["email"], **trial_info}


# ================================================================
# ANALYTICS (background task)
# ================================================================
async def log_analytics(data: dict):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
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


# ================================================================
# MAIN NANI-AI ENDPOINT
# ================================================================
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):

    total_start = time.time()
    print("\n==============================")
    print("STEP 1: Start fetch_naturopathy_results")

    # ------------------------------------
    # SECURITY CHECK
    # ------------------------------------
    if request.headers.get("X-API-KEY") != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    # ---------------------------------------
    # AUTH + PROFILE
    # ---------------------------------------
    step_auth = time.time()

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = auth_header.split(" ", 1)[1]
    user = await get_supabase_user(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")

    profile = await get_or_create_profile(user["id"], user["email"])
    trial_info = compute_trial_status(profile)

    print(f"STEP 2: Auth/Profile: {time.time() - step_auth:.2f} sec")

    if not (trial_info["trial_active"] or trial_info["subscribed"]):
        return {"error": "Trial expired", **trial_info}

    # ==================================================
    # OPTIMIZED — OPENAI EMBEDDING (150ms)
    # ==================================================
    step_emb = time.time()
    try:
        emb_resp = client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding = emb_resp.data[0].embedding
    except Exception as e:
        print("Embedding Error:", e)
        raise HTTPException(status_code=500, detail="Embedding failure")

    print(f"STEP 3: Embedding: {time.time() - step_emb:.2f} sec")

    # ==================================================
    # VECTOR SEARCH (optimized)
    # ==================================================
    step_vec = time.time()

    rpc_payload = {
        "query_embedding": json.dumps(embedding),
        "match_threshold": body.get("match_threshold", 0.40),
        "match_count": 2,  # faster + still relevant
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_documents_v2",
                json=rpc_payload,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
            )
    except Exception as e:
        print("Vector search error:", e)
        raise HTTPException(status_code=500, detail="Vector search failure")

    matches = resp.json() if resp.status_code == 200 else []
    similarities = [m["similarity"] for m in matches] if matches else []
    max_sim = max(similarities) if similarities else 0

    print(f"STEP 4: Vector Search: {time.time() - step_vec:.2f} sec")

    # ----------------------------------------------
    # TRIM CHUNKS FOR SPEED
    # ----------------------------------------------
    chunks_text = "\n\n".join([m["chunk"][:600] for m in matches]) if matches else ""

    # ----------------------------------------------
    # PROMPT GENERATION (optimized)
    # ----------------------------------------------
    if matches and max_sim >= 0.70:
        mode = "RAG_ONLY"
        final_prompt = f"""
You are Nani-AI, a warm, clear Naturopathy & Ayurveda guide.

User query:
{query}

Primary text to reference:
{chunks_text}

Instructions:
- Start with a gentle summary of the matched text
- Provide 4–6 short, practical bullet remedies
- Include diet, herbs, lifestyle & home treatments
- Use Wind/Fire/Water/Earth energies instead of Vata/Pitta/Kapha
- Keep tone simple, soothing, and preventative
"""

    elif matches and max_sim >= 0.40:
        mode = "HYBRID"
        final_prompt = f"""
You are Nani-AI, a naturopathy + ayurveda assistant.

User query:
{query}

We found related (but not perfect) text:
{chunks_text}

Instructions:
- Start from RAG content
- Add your own Ayurvedic reasoning
- Give 4–6 bullet remedies
- Include food, herbs, routines, and simple home therapy
- Use Wind/Fire/Water/Earth energies
- Keep it friendly, safe, and actionable
"""

    else:
        mode = "LLM_ONLY"
        final_prompt = f"""
You are Nani-AI, an Ayurveda + Naturopathy guide.

No RAG matches were found for:
{query}

Please produce:
- 4–6 bullet-point natural remedies
- Include diet, herbs, lifestyle, home practices
- Use Wind/Fire/Water/Earth energies
- Keep it gentle, non-medical, supportive
"""

    # Trim giant prompt if needed
    final_prompt = final_prompt[:5000]

    # ==================================================
    # FASTER LLM — gpt-4o-mini-quick
    # ==================================================
    step_llm = time.time()

    try:
        ai = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.25,
            max_tokens=330,
            top_p=1,
            messages=[{"role": "user", "content": final_prompt}],
        )
    except Exception as e:
        print("LLM ERROR:", e)
        raise HTTPException(status_code=500, detail="LLM failure")

    summary = ai.choices[0].message.content
    summary += "\n\n⚠️ Disclaimer: Nani-AI provides general wellness guidance only."

    print(f"STEP 5: LLM Completion: {time.time() - step_llm:.2f} sec")

    # ==================================================
    # ANALYTICS (async, non-blocking)
    # ==================================================
    asyncio.create_task(log_analytics({
        "query": query,
        "match_count": len(matches),
        "max_similarity": max_sim,
        "mode": mode,
    }))

    # ==================================================
    # TOTAL TIME
    # ==================================================
    print(f"STEP 6: Total Response: {time.time() - total_start:.2f} sec")
    print("==============================\n")

    return {
        "query": query,
        "summary": summary,
        "match_count": len(matches),
        "max_similarity": max_sim,
        "mode": mode,
    }
