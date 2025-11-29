# /api/app.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, json, stripe, time, asyncio
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

#--------------------------------
#STRIPE API KEY
#-----------------------------------
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

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
You are Nani-AI, a warm Naturopathy + Ayurveda–inspired wellness guide.

USER QUERY:
{query}

HIGH-RELEVANCE KNOWLEDGE (PRIMARY SOURCE):
<<<RAG>>>
{chunks_text}
<<<END-RAG>>>

Your role:
• Use the RAG text as your main knowledge source.  
• “What’s Happening in Your Body” must explain only the **physiology** of the symptom  
  (digestion, circulation, inflammation, hormones, hydration, tension, gut motility).  
• End that section with ONE short Ayurveda interpretation in plain English  
  (e.g., “Ayurveda sees this as excess internal heat/dryness/heaviness”).  
• No Sanskrit dosha names.  
• Supplement Rule (Semi-Strict):
  - Prefer supplements or Ayurvedic herbs explicitly found in the RAG text.
  - If RAG has none, you may use safe, widely trusted natural supplements or Ayurvedic herbs.
  - Never imply the RAG text included something it didn’t.
• Remedies MUST include these three sections:
  1. Nourishing Food & Drinks  
  2. Lifestyle, Routine & Movement (include 1–2 simple yoga asanas)  
  3. Natural Supplements & Ayurvedic Herbs  
• Keep tone supportive, warm, and non-medical.

---------------------------------
FEW-SHOT EXAMPLE (Follow tone + structure)

✨ What’s Happening in Your Body

Bloating can occur when digestion slows, allowing gas to accumulate in the intestines.  
It may be triggered by eating quickly, irregular meals, or foods that ferment easily.  
A warmer digestive environment generally helps move gas more smoothly.  
Ayurveda sees this pattern as increased dryness and lightness in the system.

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Drinks  
- Warm meals like soups or lightly spiced lentils  
- Ginger–fennel tea  
- Add a pinch of cumin or ajwain to meals  
- Avoid cold drinks and heavy raw foods  

2️⃣ Lifestyle, Routine & Movement  
- Take a slow 10–15 minute walk after meals  
- Chew food thoroughly  
- Use a warm compress on the abdomen  
- Gentle yoga poses: **Wind-Relieving Pose**, **Cat–Cow**  

3️⃣ Natural Supplements & Ayurvedic Herbs  
- Magnesium glycinate for smoother digestion  
- Triphala at night  
- Ginger or fennel capsules  
- A pinch of hing (asafoetida) in warm water if very gassy  

---------------------------------

NOW RESPOND IN THIS FORMAT FOR: {query}

✨ What’s Happening in Your Body  
(2–4 lines explaining physiology behind {query}, then ONE Ayurveda line.)

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Drinks  
(3–5 items: foods, drinks, teas connected to RAG.)

2️⃣ Lifestyle, Routine & Movement  
(3–6 daily practices + include 1–2 yoga poses appropriate for {query}.)

3️⃣ Natural Supplements & Ayurvedic Herbs  
(3–6 items: vitamins, minerals, natural supplements, Ayurvedic herbs.)
- Prefer supplements/herbs found directly in RAG.  
- If none are present in RAG, suggest safe universal options (e.g., magnesium glycinate, ginger capsules, triphala, amla, turmeric, ashwagandha).  

RULES:
• Stay grounded in RAG.  
• ONE short Ayurveda line.  
• No Sanskrit dosha names.  
• No medical claims.    
"""

    elif matches and max_sim >= 0.25:
        mode = "HYBRID"
        rag_used = True
        chunks_text = "\n\n".join([m["chunk"][:650] for m in matches]) if matches else ""
        chunks_text = chunks_text[:3500]
        final_prompt = f"""
You are Nani-AI, a warm Naturopathy + Ayurveda–inspired wellness guide.

USER QUERY:
{query}

PARTIAL RAG (use when relevant):
<<<RAG>>>
{chunks_text}
<<<END-RAG>>>

Guidelines:
• Blend RAG with simple physiology-based reasoning (digestion, circulation, inflammation, tension, hydration, hormones).  
• “What’s Happening in Your Body” must describe physiology only, then add ONE short Ayurveda line in plain English (e.g., heat, dryness, heaviness).  
• Do NOT use Sanskrit dosha names.  
• Supplement Rule (Semi-Strict):
  - Use supplements/Ayurvedic herbs mentioned in RAG when they appear.
  - If RAG includes none, you may add safe, gentle, universally known options.
  - Do not suggest RAG-specific supplements unless they exist in the RAG text.
• Remedies MUST include these 3 sections:
  1. Nourishing Food & Drinks  
  2. Lifestyle, Routine & Movement  
  3. Natural Supplements & Ayurvedic Herbs  
• Tone must stay warm, supportive, and non-medical.

---------------------------------
FEW-SHOT EXAMPLE (Follow structure + tone)

✨ What’s Happening in Your Body

Bloating can occur when digestion slows and gas builds up in the intestines.  
This may happen from eating quickly, irregular meals, or foods that ferment easily.  
Warmth generally helps the gut move more smoothly.  
Ayurveda sees this as a pattern of dryness and lightness in the system.

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Drinks  
- Warm meals like soups or lightly spiced lentils  
- Ginger–fennel tea  
- Add a pinch of cumin or ajwain  
- Avoid cold or raw foods if digestion feels sluggish  

2️⃣ Lifestyle, Routine & Movement  
- Slow 10–15 minute walk after meals  
- Chew food more thoroughly  
- Apply a warm compress  
- Gentle yoga: **Wind-Relieving Pose**, **Cat–Cow**  

3️⃣ Natural Supplements & Ayurvedic Herbs  
- Magnesium glycinate  
- Triphala at night  
- Fennel or ginger capsules  
- A pinch of hing (asafoetida) in warm water  

---------------------------------

Now create a UNIQUE response for {query} in this format:

✨ What’s Happening in Your Body  
(2–4 lines blending RAG + physiology, then ONE Ayurveda line.)

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Drinks  
(3–5 food + drink suggestions directly relevant to {query}.)

2️⃣ Lifestyle, Routine & Movement  
(3–6 lifestyle practices + include 1–2 yoga poses.)

3️⃣ Natural Supplements & Ayurvedic Herbs  
(3–6 natural supplements + Ayurvedic herbs.)
- Use supplements/herbs from RAG if available.  
- If not, include safe, general naturopathy or Ayurveda options.  

RULES:
• Avoid generic repetition.  
• No Sanskrit dosha names.  
• Stay gentle and non-medical.  
"""

    else:
        mode = "LLM_ONLY"
        rag_used = False
        final_prompt = f"""
You are Nani-AI, a warm naturopathy + Ayurveda–inspired wellness guide.

No RAG was found for: {query}

Guidelines:
• In “What’s Happening in Your Body,” explain the symptom using simple physiology only  
  (digestion, circulation, inflammation, hydration, hormones, nerve tension, gut motility).  
• Add ONE short line of plain-English Ayurveda interpretation at the end  
  (e.g., “Ayurveda sees this as excess internal heat/heaviness/dryness”).  
• Do NOT use Sanskrit dosha names. 
• Supplement Rule (Semi-Strict):
  - If no RAG is present, you may offer safe, widely known natural supplements and gentle Ayurvedic herbs.
  - Keep them non-medical and low-risk. 
• Remedies must include:
  1. Nourishing Food & Drinks  
  2. Lifestyle, Routine & Movement  
  3. Natural Supplements & Ayurvedic Herbs  
• Keep tone warm, safe, and non-medical.

---------------------------------
FEW-SHOT EXAMPLE (Follow structure + tone)

✨ What’s Happening in Your Body

Bloating can happen when digestion slows and gas gets trapped.  
It may arise from eating too fast, irregular meals, or foods that ferment easily.  
Warmth supports smoother movement of the gut.  
Ayurveda views this as a pattern of dryness and lightness.

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Drinks  
- Warm easy-to-digest meals  
- Ginger–fennel tea  
- Cumin or ajwain in cooking  
- Avoid cold/raw meals  

2️⃣ Lifestyle, Routine & Movement  
- Gentle walking after meals  
- Slow chewing  
- Warm compress  
- Yoga poses: **Wind-Relieving Pose**, **Cat–Cow**  

3️⃣ Natural Supplements & Ayurvedic Herbs  
- Triphala  
- Magnesium glycinate  
- Amla or ginger capsules  
- Small pinch of hing in warm water  

---------------------------------

Now answer for {query} in this format:

✨ What’s Happening in Your Body  
(2–4 soothing lines explaining physiology of {query} + ONE Ayurveda line.)

💚 Personalized Natural Remedies  

1️⃣ Nourishing Food & Drinks  
(3–5 food + drink suggestions tailored to {query}.)

2️⃣ Lifestyle, Routine & Movement  
(3–6 lifestyle steps + include 1–2 yoga poses relevant to {query}.)

3️⃣ Natural Supplements & Ayurvedic Herbs  
(3–6 supplement + herbal options.)
- Use safe and widely trusted supplements/herbs appropriate for {query}.

RULES:
• No Sanskrit dosha names.  
• No medical claims.  
• Must feel personalized to {query}.    
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
# ============================================================
# STRIPE SUBSCRIPTION CHECKOUT
# ============================================================

def verify_internal_key(request: Request):
    """ Shared small helper to secure sensitive routes """
    if request.headers.get("X-API-KEY") != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/create_checkout_session")
async def create_checkout_session(request: Request):
    """
    Creates a Stripe subscription checkout session.
    User must already be authenticated in Supabase Auth.
    """
    verify_internal_key(request)
    body = await request.json()

    price_id = body.get("price_id")
    user_email = body.get("email")
    user_id = body.get("user_id")

    if not price_id:
        raise HTTPException(status_code=400, detail="Missing price_id")
    if not user_email:
        raise HTTPException(status_code=400, detail="Missing email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            customer_email=user_email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url="https://nani.arkayoga.com/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://nani.arkayoga.com/cancel",
            metadata={"user_id": user_id}
        )
        return {"checkout_url": session.url}

    except Exception as e:
        print("Stripe session error:", str(e))
        raise HTTPException(status_code=500, detail="Stripe session creation failed")


# ============================================================
# STRIPE WEBHOOK — updates Supabase profile after payment
# ============================================================

@app.post("/stripe_webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook: called after successful payment.
    Updates:
        subscribed = true
        trial_end = null
    """
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")

    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except Exception as e:
        print("Webhook signature error:", e)
        raise HTTPException(status_code=400, detail="Invalid signature")

    # When the user pays for subscription:
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"].get("user_id")

        if user_id:
            async with httpx.AsyncClient(timeout=30) as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/profiles",
                    params={"id": f"eq.{user_id}"},
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=representation"
                    },
                    json={"subscribed": True, "trial_end": None}
                )

            print("✔ Updated user subscription in Supabase:", user_id)

    return {"status": "success"}
