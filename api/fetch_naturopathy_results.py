from fastapi import FastAPI, Request
import httpx, os

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

@app.post("/fetch_naturopathy_results")
async def fetch_results(request: Request):
    """
    Proxy endpoint to securely forward GPT Action calls to Supabase RPC.
    """
    body = await request.json()

    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/fetch_naturopathy_results",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json"
            },
            json=body
        )
    return res.json()
