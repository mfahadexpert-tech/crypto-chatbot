from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.chatbot import generate_chat_response
from backend.services.coingecko import get_crypto_price

app = FastAPI(
    title="Crypto Chatbot API",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/price/{coin_id}")
async def current_price(
    coin_id: str,
    currency: str = Query(default="usd"),
):
    return await get_crypto_price(
        coin_id=coin_id,
        currency=currency,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_history = getattr(request, "history", [])

    history = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request_history
    ]

    return await generate_chat_response(
        message=request.message,
        history=history,
    )