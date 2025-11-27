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
        chunks_text = chunks_text[:3500]
        final_prompt = f"""
You are Nani-AI, a warm Naturopathy + Ayurveda–informed wellness guide.

USER QUERY:
{query}

HIGH-RELEVANCE KNOWLEDGE (PRIMARY SOURCE):
<<<RAG>>>
{chunks_text}
<<<END-RAG>>>

Your instructions:
• Base your guidance mainly on the retrieved RAG text.  
• Make the explanation specific to {query}.  
• Use calm, simple body-science: circulation, hydration, digestion, hormone shifts, liver load, tension, inflammation.  
• Integrate Ayurveda by identifying energy imbalance using the plain-English patterns below (NO Sanskrit terms):

    • Vata > Air & Space → movement, gas, bloating, dryness, constipation, anxiety  
    • Pitta > Fire & Water → heat, inflammation, acidity, irritation, rashes  
    • Kapha > Water & Earth → heaviness, mucus, sluggishness, congestion, lethargy  

---------------------------------
EXAMPLE (Few-Shot — Follow this style):
✨ What’s Happening in Your Body  
Mild bloating can occur when digestion slows or extra gas forms in the gut.  
This often happens with irregular meals, eating quickly, or foods that are harder to break down.  
From an Ayurveda perspective, this reflects a rise in Vata(Air + Space) qualities  
(gas, movement, dryness), which can create a feeling of expansion or mild discomfort.

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Herbal Support  
- Choose warm, easy-to-digest meals  
- Sip ginger–fennel tea after eating  
- Add cumin or ajwain to support smoother digestion  
- Avoid cold drinks or heavy raw foods if you're feeling bloated  

2️⃣ Lifestyle & Routine Balance  
- Take a 10–15 min gentle walk after meals  
- Slow down chewing to support the digestive process  
- Place a warm compress on the abdomen if it feels tight  
- Keep meal times steady to support digestive rhythm  

---------------------------------

Now respond in THIS structure, fully tailored to {query}:

✨ What’s Happening in Your Body
(2–4 lines summarizing what the RAG text suggests about {query}.  
Explain using simple physiology and include energy imbalance using the plain-English mappings above.)

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Herbal Support  
- 3–5 food, drink, or gentle herbal suggestions clearly connected to RAG and helpful for {query}.  
- Focus on soothing, cooling, grounding, or digestively supportive choices—whatever matches the energy imbalance.

2️⃣ Lifestyle & Routine Balance  
- 3–5 practical lifestyle shifts for {query} (movement, rest, warm/cool applications, simple home practices).  
- Keep everything non-alarming, supportive, and easy to follow.

RULES:
• Stay grounded in RAG.  
• No repetitive generic advice across conditions.  
• No medical claims.  
"""


    elif matches and max_sim >= 0.25:
        mode = "HYBRID"
        rag_used = True
        chunks_text = "\n\n".join([m["chunk"][:650] for m in matches]) if matches else ""
        chunks_text = chunks_text[:3500]
        final_prompt = f"""
You are Nani-AI, a warm Naturopathy + Ayurveda–informed wellness guide.

USER QUERY:
{query}

HIGH-RELEVANCE KNOWLEDGE (PRIMARY SOURCE):
<<<RAG>>>
{chunks_text}
<<<END-RAG>>>

Your instructions:
• Base your guidance mainly on the retrieved RAG text.  
• Make the explanation specific to {query}.  
• Use calm, simple body-science: circulation, hydration, digestion, hormone shifts, liver load, tension, inflammation.  
• Integrate Ayurveda by identifying energy imbalance using the plain-English patterns below (NO Sanskrit terms):

    • Vata > Air & Space → movement, gas, bloating, dryness, constipation, anxiety  
    • Pitta > Fire & Water → heat, inflammation, acidity, irritation, rashes  
    • Kapha > Water & Earth → heaviness, mucus, sluggishness, congestion, lethargy  

---------------------------------
EXAMPLE (Few-Shot — Follow this style):
✨ What’s Happening in Your Body  
Mild bloating can occur when digestion slows or extra gas forms in the gut.  
This often happens with irregular meals, eating quickly, or foods that are harder to break down.  
From an Ayurveda perspective, this reflects a rise in Vata(Air + Space) qualities  
(gas, movement, dryness), which can create a feeling of expansion or mild discomfort.

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Herbal Support  
- Choose warm, easy-to-digest meals  
- Sip ginger–fennel tea after eating  
- Add cumin or ajwain to support smoother digestion  
- Avoid cold drinks or heavy raw foods if you're feeling bloated  

2️⃣ Lifestyle & Routine Balance  
- Take a 10–15 min gentle walk after meals  
- Slow down chewing to support the digestive process  
- Place a warm compress on the abdomen if it feels tight  
- Keep meal times steady to support digestive rhythm  

---------------------------------

Now respond in THIS structure, fully tailored to {query}:

✨ What’s Happening in Your Body
(2–4 lines summarizing what the RAG text suggests about {query}.  
Explain using simple physiology and include energy imbalance using the plain-English mappings above.)

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Herbal Support  
- 3–5 food, drink, or gentle herbal suggestions clearly connected to RAG and helpful for {query}.  
- Focus on soothing, cooling, grounding, or digestively supportive choices—whatever matches the energy imbalance.

2️⃣ Lifestyle & Routine Balance  
- 3–5 practical lifestyle shifts for {query} (movement, rest, warm/cool applications, simple home practices).  
- Keep everything non-alarming, supportive, and easy to follow.

RULES:
• Avoid generic or repeated remedies across conditions.  
• Do not exaggerate.  
• Keep tone soft, grounded, and supportive.  
"""


    else:
        mode = "LLM_ONLY"
        rag_used = False
        final_prompt = f"""
You are Nani-AI, a warm Ayurveda + Naturopathy–informed wellness guide.

We found no RAG text for this query:
{query}

Your instructions:
• Use calm, simple physiology to explain what’s happening (circulation, hydration, digestion, liver load, hormones, inflammation).  
• Keep tone gentle and non-alarming.  
• Identify energy imbalance using ONLY these plain-English mappings:

    • Vata > Air & Space → movement, gas, bloating, dryness, constipation, anxiety  
    • Pitta > Fire & Water → heat, inflammation, acidity, irritation, rashes  
    • Kapha > Water & Earth → heaviness, mucus, sluggishness, congestion, lethargy  

---------------------------------
EXAMPLE (Few-Shot — Follow this style):
✨ What’s Happening in Your Body  
Mild bloating can occur when digestion slows or extra gas forms in the gut.  
This often happens with irregular meals, eating quickly, or foods that are harder to break down.  
From an energy perspective, this reflects a rise in Air + Space qualities  
(gas, movement, dryness), which can create a feeling of expansion or mild discomfort.

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Herbal Support  
- Choose warm, easy-to-digest meals  
- Sip ginger–fennel tea after eating  
- Add cumin or ajwain to support smoother digestion  
- Avoid cold drinks or heavy raw foods if you're feeling bloated  

2️⃣ Lifestyle & Routine Balance  
- Take a 10–15 min gentle walk after meals  
- Slow down chewing to support the digestive process  
- Place a warm compress on the abdomen if it feels tight  
- Keep meal times steady to support digestive rhythm  
---------------------------------

Now respond using THIS structure:

✨ What’s Happening in Your Body  
(2–4 soothing lines explaining {query} using simple body-science +  
a clear energy imbalance based ONLY on the plain-English mapping.)

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Herbal Support  
- 3–5 food & herbal suggestions suitable for {query}, grounded in Ayurvedic energetics.

2️⃣ Lifestyle & Routine Balance  
- 3–5 practical lifestyle + home-based practices that support comfort and recovery for {query}.

RULES:
• No generic repetition across conditions.  
• No Sanskrit dosha words.  
• No medical claims.  
• Keep tone gentle, helpful, supportive.  
"""




# Avoid over-long prompts but keep them intact
    if len(final_prompt) > 20000:   # 24k chars safe for GPT-4o
        final_prompt = final_prompt[:20000]

 
    # ---------------------------------------------------
    # LLM Completion (WITH TEMPERATURE TUNING)
    # ---------------------------------------------------
    step_llm = time.time()

    ai = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,            # ⭐ tuned temperature
        max_tokens=650,              # ⭐ stable response length
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
