# /api/app.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json, time, asyncio
from datetime import datetime, timezone
from openai import OpenAI

# ----------------------------------------------
# OPENAI CLIENT
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

@app.options("/fetch_naturopathy_results")
async def preflight_handler():
    return {"status": "ok"}
# ----------------------------------------------
# ENV VARS
# ----------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")

EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed",
)

# ----------------------------------------------
# SUPABASE HELPERS
# ----------------------------------------------
async def get_supabase_user(access_token: str):
    if not access_token:
        return None

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {access_token}"},
        )

    if resp.status_code != 200:
        print("Supabase auth error:", resp.text)
        return None

    return resp.json()


async def get_or_create_profile(user_id: str, email: str):
    async with httpx.AsyncClient(timeout=30) as client:

        # Check existing
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"select": "*", "id": f"eq.{user_id}", "limit": 1},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
        )

        rows = resp.json() if resp.status_code == 200 else []
        if rows:
            return rows[0]

        # Create profile (start trial)
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

# ----------------------------------------------
# AUTH STATUS
# ----------------------------------------------
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


# ----------------------------------------------
# ANALYTICS
# ----------------------------------------------
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


# ----------------------------------------------
# NANI-AI MAIN ENDPOINT (WITH TIMING LOGS + NEW PROMPTS + TEMP CONTROL)
# ----------------------------------------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):

    total_start = time.time()
    print("\n==============================")
    print("STEP 1: Start fetch_naturopathy_results")

    # -------------------------
    # SECURITY CHECK
    # -------------------------
    if request.headers.get("X-API-KEY") != SECRET:
       raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    # -------------------------
    # AUTH + PROFILE
    # -------------------------
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

    # -------------------------
    # EMBEDDING
    # -------------------------
    step_emb = time.time()

# ---------------------------
# EMBEDDING SAFE WRAPPER
# ---------------------------
    step_emb = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            emb = await client.post(EMBEDDING_API, json={"query": query})

        if emb.status_code != 200:
            print("Embedding service error:", emb.text)
            raise HTTPException(status_code=500, detail="Embedding API failure")

        emb_json = emb.json()

    except Exception as e:
        print("Embedding crashed:", str(e))
        raise HTTPException(status_code=500, detail="Embedding crashed")

    embedding = (
        emb_json.get("embedding")
        or emb_json.get("data", {}).get("embedding")
    )

    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
        embedding = embedding[0]

    print(f"STEP 3: Embedding: {time.time() - step_emb:.2f} sec")

    # -------------------------
    # VECTOR SEARCH
    # -------------------------
    step_vec = time.time()

    # Ensure embedding is list[float]
    if not isinstance(embedding, list):
        raise HTTPException(status_code=500, detail="Embedding format invalid")

    print("Embedding size:", len(embedding))

    rpc_payload = {
        "query_embedding": embedding,  # MUST be float[]
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
        print("RPC Error:", resp.text)
        raise HTTPException(status_code=500, detail="Vector search failure")

    matches = resp.json()
    print("Vector search returned", len(matches), "matches")

    similarities = [m["similarity"] for m in matches] if matches else []
    max_sim = max(similarities) if similarities else 0
    
    print("Max similarity:", max_sim)
    print(f"STEP 4: Vector Search: {time.time() - step_vec:.2f} sec")

     # ---------------------------------------------- 
    # PROMPT GENERATION (optimized & de-templated)
    # ----------------------------------------------
    # The content must come from RAG + the user query.
    if matches and max_sim >= 0.55:
        mode = "RAG_ONLY"
        rag_used = True
        chunks_text = "\n\n".join([m["chunk"][:650] for m in matches]) if matches else ""
        final_prompt = f"""
You are Nani-AI, a warm, clear Naturopathy & Ayurveda guide.

The user asked:
{query}

Below is highly relevant naturopathy text from the knowledgebase.
You MUST base your answer primarily on this text and NOT on generic health advice:

<<<CHUNKS_TEXT>>>
{chunks_text}
<<<END_CHUNKS_TEXT>>>

Your job:
- Use the retrieved text as the PRIMARY source of truth.
- Make the answer clearly specific to the user's query (mention {query} explicitly).
- Do NOT reuse the same remedies for every condition.
- Do NOT invent herbs or treatments that are not compatible with the retrieved text.

Respond using EXACTLY this structure (same headings + emojis, but YOU must create the bullet content):

🌿 Nani-AI Wellness Guidance  
✨ What’s Happening in Your Body  
(Write 2–3 short lines summarizing the situation for {query}, grounded in the retrieved text.)
---

💚 Your Personalized Natural Remedies  

🥗 1. Nourishing Food Support  
- 3–5 bullet points about food/diet **specifically helpful for {query}**, grounded in the retrieved text.  
- Avoid generic “drink more water” unless the retrieved text clearly supports it.

🌿 2. Herbal & Home Remedies  
- 3–5 bullet points of herbs, decoctions, powders, or home practices that are relevant to {query}.  
- Prefer remedies explicitly or implicitly supported by the retrieved text.

🛁 3. Simple Home Therapy  
- 2–4 bullet points of simple, safe home practices aligned with the retrieved text and {query}.  

🧘‍♀️ 4. Lifestyle & Routine Balance  
- 3–5 bullet points of realistic routines, movement, sleep, and habits that support healing for {query}.
---

🌬️ Ayurveda Explanation
Describe the energy imbalance in {query} using Vata as Air, Space, Gas, Constipation, Anxiety  / Pitta as Excess Heat, Inflammation, Acidity, Rashes / Kapha as Water, Mucus, Heaviness, Sluggishness, Congestion, Lethargy

Make the reasoning feel personal to {query}.

Rules:
- DO NOT copy/paste the same remedies for different conditions.
- DO NOT output example text from this prompt; generate fresh, condition-specific bullets.
- All advice must feel tailored to {query} and consistent with the retrieved text.
"""

    elif matches and max_sim >= 0.25:
        mode = "HYBRID"
        rag_used = True
        chunks_text = "\n\n".join([m["chunk"][:650] for m in matches]) if matches else ""
        final_prompt = f"""
You are Nani-AI, a warm, clear Naturopathy & Ayurveda guide.

The user asked:
{query}

Below is somewhat related naturopathy text from your knowledgebase:

<<<CHUNKS_TEXT>>>
{chunks_text}
<<<END_CHUNKS_TEXT>>>

Your job:
- Use the retrieved text as an ANCHOR whenever it’s relevant.
- Fill gaps with your own Ayurvedic + naturopathic reasoning for {query}.
- Make the answer clearly specific to {query}, not generic.

Respond using EXACTLY this structure (same headings + emojis; you create the content):

🌿 Nani-AI Wellness Guidance  
✨ What’s Happening in Your Body  
(2–3 lines describing what might be happening in the body for {query}, referencing the retrieved text where possible.)
---

💚 Your Personalized Natural Remedies  

🥗 1. Nourishing Food Support  
- 3–5 condition-specific diet bullets for {query}, using RAG text where helpful.

🌿 2. Herbal & Home Remedies  
- 3–5 bullets combining RAG-based herbs + your Ayurvedic reasoning for {query}.

🛁 3. Simple Home Therapy  
- 2–4 practical at-home steps that are safe and relevant to {query}.

🧘‍♀️ 4. Lifestyle & Routine Balance  
- 3–5 realistic changes in routine that support healing for {query}.
---

🌬️ Ayurveda Explanation
Describe the energy imbalance in {query} using Vata as Air, Space, Gas, Constipation, Anxiety  / Pitta as Excess Heat, Inflammation, Acidity, Rashes / Kapha as Water, Mucus, Heaviness, Sluggishness, Congestion, Lethargy

Rules:
- Do NOT reuse the same remedies across unrelated conditions.
- Ground as much as possible in the CHUNKS_TEXT, but adapt to the specific query.
"""

    else:
        mode = "LLM_ONLY"
        rag_used = False
        final_prompt = f"""
You are Nani-AI, a warm Ayurvedic + Naturopathy wellness guide.

No RAG text was found for:
{query}

You must answer from your own Ayurvedic + naturopathy knowledge, but the response
must still feel UNIQUE to {query} (do not repeat the same generic template for every condition).

Respond using THIS structure (you create the content):

🌿 Nani-AI Wellness Guidance  
✨ What’s Happening in Your Body  
(Explain {query} in 2–3 soothing lines.)
---

💚 Your Personalized Natural Remedies  

🥗 1. Nourishing Food Support  
- 3–5 bullets describing specific diet patterns helpful for {query}.

🌿 2. Herbal & Home Remedies  
- 3–5 bullets of herbs and home remedies suitable for {query}.

🛁 3. Simple Home Therapy  
- 2–4 at-home practices that are safe and simple for {query}.

🧘‍♀️ 4. Lifestyle & Routine Balance  
- 3–5 bullets around movement, rest, work, and daily rhythm tailored to {query}.
---

🌬️ Ayurveda Explanation
Describe the energy imbalance in {query} using Vata as Air, Space, Gas, Constipation, Anxiety  / Pitta as Excess Heat, Inflammation, Acidity, Rashes / Kapha as Water, Mucus, Heaviness, Sluggishness, Congestion, Lethargy

Rules:
- Do NOT copy remedies used for entirely different conditions.
- The suggestions must clearly match the nature of {query}.
"""

    # Trim giant prompt if needed
    final_prompt = final_prompt[:5000]
 
 
    # ---------------------------------------------------
    # LLM Completion (WITH TEMPERATURE TUNING)
    # ---------------------------------------------------
    step_llm = time.time()

    ai = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,            # ⭐ tuned temperature
        max_tokens=380,              # ⭐ stable response length
        messages=[{"role": "user", "content": final_prompt}],
    )

    summary = ai.choices[0].message.content
    summary += "\n\n⚠️ Disclaimer: Nani-AI provides general wellness guidance, not medical care."

    print(f"STEP 5: LLM Completion: {time.time() - step_llm:.2f} sec")

    # ---------------------------------------------------
    # ANALYTICS (async)
    # ---------------------------------------------------
    asyncio.create_task(log_analytics({
        "query": query,
        "match_count": len(matches),
        "max_similarity": max_sim,
        "rag_used": rag_used,
        "mode": mode,
        "sources": [m["source"] for m in matches] if matches else [],
        "latency_ms": int((time.time() - total_start) * 1000)
    }))
    # ---------------------------------------------------
    # TOTAL RUNTIME
    # ---------------------------------------------------
    print(f"STEP 6: Total Response: {time.time() - total_start:.2f} sec")
    print("==============================\n")

    return {
        "query": query,
        "summary": summary,
        "match_count": len(matches),
        "max_similarity": max_sim,
        "mode": mode,
        "rag_used": rag_used
    }
