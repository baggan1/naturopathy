# Inside /api/app.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import json
import time
import asyncio
from datetime import datetime, timedelta, timezone

from openai import OpenAI
import stripe

# ------------
# CLIENTS & CONFIG
# ------------
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # TODO: restrict to your domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Preflight for Squarespace / browsers
@app.options("/fetch_naturopathy_results")
async def preflight_handler():
    return {"status": "ok"}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SECRET = os.getenv("APP_SECRET")

EMBEDDING_API = os.getenv(
    "EMBEDDING_API",
    "https://mystiqspice-naturopathy-embedder.hf.space/embed",
)

# Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


# ------------
# HELPERS: Supabase REST
# ------------
async def supabase_post(path: str, json_body: dict, params: dict | None = None, prefer: str | None = None):
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SUPABASE_URL}{path}",
            headers=headers,
            json=json_body,
            params=params,
        )
    return resp


async def supabase_get(path: str, params: dict | None = None):
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SUPABASE_URL}{path}",
            headers=headers,
            params=params,
        )
    return resp


# ------------
# ANALYTICS LOGGING
# ------------
async def log_analytics(data: dict):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
        print("Analytics logging failed:", str(e))


# ------------
# FREE TRIAL ENDPOINT (Email-only)
# ------------
@app.post("/start_trial")
async def start_trial(request: Request):
    # Auth via X-API-KEY
    auth = request.headers.get("X-API-KEY")
    if auth != SECRET:
        return {"error": "Unauthorized"}

    body = await request.json()
    email = body.get("email")

    if not email:
        return {"error": "Email is required"}

    # 15-day trial
    trial_ends_at = (datetime.now(timezone.utc) + timedelta(days=15)).isoformat()

    payload = {
        "email": email,
        "active": True,
        "trial_ends_at": trial_ends_at,
    }

    # Upsert into user_access (on_conflict=email)
    resp = await supabase_post(
        "/rest/v1/user_access",
        json_body=payload,
        params={"on_conflict": "email"},
        prefer="resolution=merge-duplicates",
    )

    if resp.status_code not in (200, 201, 204):
        print("start_trial error:", resp.status_code, resp.text)
        return {"error": "Failed to start trial"}

    return {"success": True, "trial_ends": trial_ends_at}


# ------------
# MAIN NANI-AI ENDPOINT (RAG + LLM + GATING)
# ------------
@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    start_time = time.time()

    try:
        # ---- AUTH HEADER ----
        auth = request.headers.get("X-API-KEY")
        if auth != SECRET:
            return {"error": "Unauthorized"}

        body = await request.json()

        email = body.get("email")
        query = body.get("query", "").strip()

        if not email:
            return {"error": "EMAIL_REQUIRED"}

        if not query:
            return {"error": "Missing 'query' field."}

        # ----------------------------
        # 1. GATEKEEP: Check user_access in Supabase
        # ----------------------------
        access_resp = await supabase_get(
            "/rest/v1/user_access",
            params={"email": f"eq.{email}", "select": "*"},
        )

        if access_resp.status_code != 200:
            return {"error": "ACCESS_CHECK_FAILED"}

        rows = access_resp.json()
        if not rows:
            return {"error": "NO_ACCESS"}

        record = rows[0]

        # Trial expiry check (if set)
        trial_ends_str = record.get("trial_ends_at")
        if trial_ends_str:
            # Normalize possible 'Z' suffix
            trial_ends_str = trial_ends_str.replace("Z", "+00:00")
            trial_ends = datetime.fromisoformat(trial_ends_str)
            now = datetime.now(timezone.utc)
            if trial_ends < now:
                return {"error": "TRIAL_EXPIRED"}

        if not record.get("active", False):
            return {"error": "SUBSCRIPTION_REQUIRED"}

        # ----------------------------
        # 2. Embedding (HF Space)
        # ----------------------------
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

        if (
            isinstance(query_embedding, list)
            and len(query_embedding) == 1
            and isinstance(query_embedding[0], list)
        ):
            query_embedding = query_embedding[0]

        # ----------------------------
        # 3. Supabase Vector Search via RPC
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

        matches = response.json() or []
        similarities = [m["similarity"] for m in matches] if matches else []
        max_similarity = max(similarities) if similarities else 0.0

        chunks_text = "\n\n".join([f"- {m['chunk']}" for m in matches]) if matches else ""

        # ----------------------------
        # 4. HYBRID LOGIC — RAG_ONLY / HYBRID / LLM_ONLY
        # ----------------------------
        rag_used = False
        llm_used = False
        mode = "LLM_ONLY"

        # High confidence → RAG_ONLY (summarize mostly from matches)
        if matches and max_similarity >= 0.70:
            mode = "RAG_ONLY"
            rag_used = True
            llm_used = True

            final_prompt = f"""
You are Nani-AI.

Provide clear, actionable naturopathy + ayurveda guidance for the query:
{query}

Use the following retrieved text:
{chunks_text}

Instructions:
- Summarize RAG content first
- Provide 4–6 bullet-point remedies
- Include diet, lifestyle, hydrotherapy or home remedies
- Keep the tone warm, simple, and nurturing
"""

        # Medium confidence → HYBRID
        elif matches and 0.40 <= max_similarity < 0.70:
            mode = "HYBRID"
            rag_used = True
            llm_used = True

            final_prompt = f"""
You are Nani-AI.

User query:
{query}

We found related information but confidence is moderate.
Blend retrieved naturopathy text with your own ayurveda reasoning.

Retrieved text:
{chunks_text}

Instructions:
- Start with the best RAG insights
- Add LLM enhancements to fill gaps
- Provide 4–6 simple remedies
- Include food, herbs, habits, and home practices
"""

        # Low / no matches → LLM_ONLY
        else:
            mode = "LLM_ONLY"
            rag_used = False
            llm_used = True

            final_prompt = f"""
You are Nani-AI.

No reliable matches were found in the database for:
{query}

Generate an ayurveda + naturopathy–based answer from scratch.

Instructions:
- Provide 4–6 bullet-point remedies
- Include diet, herbs, lifestyle, and home treatments
- Keep it simple, safe, and practical
"""

        # ----------------------------
        # 5. Call OpenAI LLM
        # ----------------------------
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

        # ----------------------------
        # 6. Analytics Logging
        # ----------------------------
        latency_ms = int((time.time() - start_time) * 1000)
        matched_sources = [m["source"] for m in matches] if matches else []

        analytics_payload = {
            "query": query,
            "match_count": len(matches),
            "max_similarity": float(max_similarity),
            "sources": matched_sources,
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
            "latency_ms": latency_ms,
        }

        asyncio.create_task(log_analytics(analytics_payload))

        return {
            "query": query,
            "summary": summary,
            "sources": matched_sources,
            "match_count": len(matches),
            "max_similarity": float(max_similarity),
            "rag_used": rag_used,
            "llm_used": llm_used,
            "mode": mode,
        }

    except Exception as e:
        return {"error": f"Server exception: {str(e)}"}


# ------------
# STRIPE CHECKOUT (for future paid subscription)
# ------------
@app.post("/create_checkout_session")
async def create_checkout_session(request: Request):
    auth = request.headers.get("X-API-KEY")
    if auth != SECRET:
        return {"error": "Unauthorized"}

    body = await request.json()
    price_id = body.get("price_id")
    email = body.get("email")  # optional

    if not price_id:
        return {"error": "Missing price_id"}

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            billing_address_collection="auto",
            customer_email=email,
            subscription_data={
                # You can add paid trial here later if desired
                # "trial_period_days": 15,
            },
            line_items=[{"price": price_id, "quantity": 1}],
            allow_promotion_codes=True,
            success_url="https://nani-ai-pwa.vercel.app/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://nani-ai-pwa.vercel.app/cancel",
        )
        return {"checkout_url": session.url}
    except Exception as e:
        return {"error": str(e)}


# ------------
# STRIPE WEBHOOK (mark users active after payment)
# ------------
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()

    try:
        event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
    except Exception as e:
        return {"error": f"Webhook parse error: {str(e)}"}

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        email = session_obj.get("customer_details", {}).get("email")

        if email:
            # Mark user as active paid user
            payload = {
                "email": email,
                "active": True,
            }
            # Upsert on email
            resp = await supabase_post(
                "/rest/v1/user_access",
                json_body=payload,
                params={"on_conflict": "email"},
                prefer="resolution=merge-duplicates",
            )
            if resp.status_code not in (200, 201, 204):
                print("stripe webhook user_access upsert failed:", resp.status_code, resp.text)

    return {"status": "ok"}
