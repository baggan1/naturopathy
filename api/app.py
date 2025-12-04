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
SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL")
CANCEL_URL = os.getenv("STRIPE_CANCEL_URL")

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

    # Simple role logic
    if trial_info["subscribed"]:
        role = "premium"
    elif trial_info["trial_active"]:
        role = "trial"
    else:
        role = "free"

    return {
        "user_email": user["email"],
        "role": role,              # 👈 new field
        **trial_info
    }


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
# NANI-AI MAIN ENDPOINT (CONVERSATIONAL)
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
    history = body.get("history") or []

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    # Build short conversational snippet from recent turns
    history_snippet = ""
    if isinstance(history, list) and history:
        trimmed = history[-6:]  # last few messages only
        lines = []
        for m in trimmed:
            role = (m.get("role") or "").upper()
            if role not in ("USER", "ASSISTANT"):
                continue
            content = (m.get("content") or "")[:350]
            lines.append(f"{role}: {content}")
        history_snippet = "\n".join(lines)

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

    if not isinstance(embedding, list):
        raise HTTPException(status_code=500, detail="Embedding format invalid")

    print("Embedding size:", len(embedding))

    rpc_payload = {
        "query_embedding": embedding,
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

    # -------------------------------------------------------
    # PROMPT GENERATION — CONVERSATIONAL + FOLLOW-UP LOGIC
    # -------------------------------------------------------

followup_rule = """
FOLLOW-UP DETECTION LOGIC (STRICT):

A message IS a follow-up to the SAME ailment when ANY of the following are true:
• The user asks for alternatives, substitutions, replacements, healthier options,
  examples, additional tips, timing, quantities, “what else can I do?”, “what should I avoid?”,
  or similar detail-expanding questions.
• The user stays within the same general problem domain (e.g., weight management,
  digestion, bloating, immunity, sleep, stress).
• The user does NOT introduce a new ailment, new symptom category, or unrelated condition.

A message IS a NEW ailment ONLY IF:
• It introduces a new physical problem or symptom unrelated to the prior issue.
  Examples: “I also have headaches”, “Now I feel nausea”, “My joints hurt too.”

BEHAVIOR RULES:

IF this is a FOLLOW-UP to the SAME AILMENT:
→ Respond ONLY with a warm, conversational paragraph (2–5 sentences).
→ You may include 1–2 tiny bullet items ONLY if the user explicitly asks for “more foods”,
   “more supplements”, “more items”, etc.
→ DO NOT produce the structured 4-part format.
→ DO NOT output section headers (“What’s happening…”, “Action Steps”, etc.).
→ DO NOT repeat the original main advice unless summarizing briefly.
→ Provide a natural, human-like continuation of the conversation.

IF this is a NEW AILMENT:
→ Use the FULL structured format with bullet points:
   1. What’s Happening in Your Body (paragraph)
   2. Action Steps
      • Nourishing Food & Drinks (bullet points)
      • Lifestyle, Routine & Movement (bullet points; yoga allowed but no pose names)
      • Natural Supplements & Ayurvedic Herbs (bullet points)
   3. Follow-up question (paragraph)

MANDATORY SAFETY & STYLE RULES:
• Tone must be warm, supportive, non-alarming.
• No medical diagnoses. Keep physiology gentle (digestion, circulation, hydration, tension, stress).
• Ayurveda reasoning only if user explicitly asks.
• NO naming yoga poses; only general references (“gentle yoga”, “light stretching”).
• Bullet points only appear in structured (new ailment) mode or tiny lists when user asks for “more items.”
"""

    # -------------------------------------------------------
    # RAG_ONLY (high confidence)
    # -------------------------------------------------------
    if matches and max_sim >= 0.55:
        mode = "RAG_ONLY"
        rag_used = True
        chunks_text = "\n\n".join([m["chunk"][:650] for m in matches])
        chunks_text = chunks_text[:3500]

        final_prompt = f"""
You are Nani-AI, a warm, naturopathy + Ayurveda–inspired wellness guide.

USER QUERY:
{query}

PRIOR CONVERSATION:
{history_snippet if history_snippet else "None."}

HIGH-RELEVANCE KNOWLEDGE (RAG):
<<<RAG>>>
{chunks_text}
<<<END-RAG>>>

{followup_rule}

RESPONSE RULES:
• Maintain a calming tone.
• Keep physiology gentle (digestion, circulation, hydration, tension, mild inflammation).
• Include ONE short Ayurveda interpretation unless user requests full dosha analysis.
• *When following full structured format*, ensure all remedies are presented as bullet points.
• Yoga is allowed only as “gentle yoga” or “simple strengthening exercises” (no pose names).

NOW DETERMINE FORMAT:
• If this is a follow-up to the SAME ailment → return ONLY a paragraph (optionally tiny bullets if user asks for “more items”).
• If this is a NEW ailment → return the full structured 3-section bullet list format.
"""

    # -------------------------------------------------------
    # HYBRID (medium similarity)
    # -------------------------------------------------------
    elif matches and max_sim >= 0.25:
        mode = "HYBRID"
        rag_used = True
        chunks_text = "\n\n".join([m["chunk"][:650] for m in matches])
        chunks_text = chunks_text[:3500]

        final_prompt = f"""
You are Nani-AI, a warm naturopathy + Ayurveda–inspired wellness guide.

USER QUERY:
{query}

PRIOR CONVERSATION:
{history_snippet if history_snippet else "None."}

PARTIAL RAG:
<<<RAG>>>
{chunks_text}
<<<END-RAG>>>

{followup_rule}

HYBRID RULES:
• Blend partial RAG + physiology.
• Keep tone gentle and non-alarming.
• When using structured mode, remedies must be bullet points.
• Yoga allowed only generally (“gentle yoga”, “light stretching”, “general strengthening”).
• Ayurveda mode activates only when user explicitly asks.

NOW CHOOSE THE CORRECT FORMAT:
• Follow-up = paragraph only.  
• New ailment = full structured 4-section response with bullet points.
"""

    # -------------------------------------------------------
    # LLM_ONLY (no RAG available)
    # -------------------------------------------------------
    else:
        mode = "LLM_ONLY"
        rag_used = False

        final_prompt = f"""
You are Nani-AI, a warm naturopathy + Ayurveda–inspired wellness guide.

No RAG knowledge found for:
{query}

PRIOR CONVERSATION:
{history_snippet if history_snippet else "None."}

{followup_rule}

LLM-ONLY RULES:
• Physiology must remain gentle and safe.
• Ayurveda reasoning only if user requests.
• When using structured mode, all remedies must be bullet points.
• Yoga is allowed only as general movement (“gentle yoga”, no pose names).

NOW DETERMINE RESPONSE STYLE:
• If follow-up about SAME ailment → paragraph only.  
• If NEW ailment → full structured response with bullet points in Action Steps.
"""

    # Ensure safety: prevent overly long prompts
    if len(final_prompt) > 20000:
        final_prompt = final_prompt[:20000]

    # ---------------------------------------------------
    # LLM Completion
    # ---------------------------------------------------
    step_llm = time.time()

    ai = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=650,
        messages=[{"role": "user", "content": final_prompt}],
    )

    summary = ai.choices[0].message.content
    summary += "\n\n⚠️ Disclaimer: Nani-AI provides general wellness guidance, not medical care."

    print(f"STEP 5: LLM Completion: {time.time() - step_llm:.2f} sec")

    # ---------------------------------------------------
    # ANALYTICS (async)
    # ---------------------------------------------------
    asyncio.create_task(log_analytics({
        "user_id": user["id"],
        "user_email": user["email"],
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
    
    print("🔵 Creating checkout session:", user_id, user_email, price_id)
    
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            customer_email=user_email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{SUCCESS_URL}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=CANCEL_URL,
            metadata={"user_id": user_id}
        )
        print("🟢 Checkout session created:", session.id)
        return {"checkout_url": session.url}

    except Exception as e:
        print("Stripe session error:", str(e))
        raise HTTPException(status_code=500, detail="Stripe session creation failed")


# ============================================================
# STRIPE WEBHOOK — updates Supabase profile after payment
# ============================================================

@app.post("/stripe_webhook")
async def stripe_webhook(request: Request):
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

    event_type = event["type"]
    session_obj = event["data"]["object"]   # ← FIXED

    print(f"🔔 Received Stripe event: {event_type}")

    # ------------------------------------------------------------
    # CHECKOUT COMPLETED → FIRST PAYMENT
    # ------------------------------------------------------------
    if event_type == "checkout.session.completed":
        user_id = session_obj.get("metadata", {}).get("user_id")
        stripe_customer_id = (
            session_obj.get("customer")
            or session_obj.get("customer_details", {}).get("id")
        )

        print("Parsed webhook values:", user_id, stripe_customer_id)

        if not user_id:
            print("⚠️ Missing `user_id` in session metadata.")
            return {"status": "ignored"}

        if not stripe_customer_id:
            print("⚠️ Webhook missing customer ID.")
            return {"status": "ignored"}

        # Update Supabase user row
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
                json={
                    "subscribed": True,
                    "trial_end": None,
                    "stripe_customer_id": stripe_customer_id
                }
            )

        print(f"✅ Updated Supabase profile for user: {user_id}")
        return {"status": "success"}

    # ------------------------------------------------------------
    # OTHER EVENTS IGNORED FOR NOW
    # ------------------------------------------------------------
    return {"status": "ignored"}


#---------------------------------------------
#CUSTOMER PORTAL TO MANAGE BILLING-------------
#-----------------------------------------------
@app.post("/create_customer_portal")
async def create_customer_portal(request: Request):
    """
    Creates a Stripe billing portal session for the logged-in user.
    Allows them to update payment method, cancel subscription, view invoices, etc.
    """
    if request.headers.get("X-API-KEY") != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    user_id = body.get("user_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    # 1️⃣ Look up Stripe customer ID from Supabase profile
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"select": "*", "id": f"eq.{user_id}", "limit": 1},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )
    rows = resp.json()
    if not rows:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = rows[0]
    stripe_customer_id = profile.get("stripe_customer_id")

    if not stripe_customer_id:
        raise HTTPException(
            status_code=400,
            detail="No Stripe customer found — is this user subscribed?"
        )

    # 2️⃣ Create Stripe portal session
    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url="https://nani.arkayoga.com/",  # Adjust as needed
        )
        return {"url": portal_session.url}

    except Exception as e:
        print("Portal creation error:", e)
        raise HTTPException(status_code=500, detail="Failed to create billing portal session")

#------------------------------------
#USER HISTORY ENDPOINT---------------
#-------------------------------------
@app.get("/user/history")
async def user_history(request: Request):

    if request.headers.get("X-API-KEY") != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    access_token = auth_header.split(" ", 1)[1]
    user = await get_supabase_user(access_token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/analytics_logs",
            params={
                "select": "query,created_at",
                "user_id": f"eq.{user['id']}",
                "order": "created_at.desc",
                "limit": 25
            },
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            }
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="History fetch failed")

    return resp.json()
