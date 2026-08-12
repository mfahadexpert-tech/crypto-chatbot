import json
import os
import re
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

from backend.services.coingecko import (
    get_coins_near_price,
    get_crypto_price,
    search_coin,
)
from backend.services.currency import convert_currency

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

POPULAR_COINS = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "ether": "ethereum", "eth": "ethereum",
    "etherum": "ethereum", "etherium": "ethereum",
    "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "cardano": "cardano", "ada": "cardano",
    "xrp": "ripple", "ripple": "ripple",
    "bnb": "binancecoin", "litecoin": "litecoin", "ltc": "litecoin",
    "polkadot": "polkadot", "dot": "polkadot",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
    "chainlink": "chainlink", "link": "chainlink",
}

FIAT_ALIASES = {
    "dollar": "USD", "dollars": "USD", "us dollar": "USD",
    "pakistani rupee": "PKR", "pakistani rupees": "PKR", "rupees": "PKR",
    "pound": "GBP", "pounds": "GBP", "british pound": "GBP",
    "euro": "EUR", "euros": "EUR",
    "riyal": "SAR", "riyals": "SAR", "saudi riyal": "SAR",
    "dirham": "AED", "dirhams": "AED", "uae dirham": "AED",
    "indian rupee": "INR", "indian rupees": "INR",
    "yen": "JPY", "yuan": "CNY", "canadian dollar": "CAD",
    "australian dollar": "AUD", "swiss franc": "CHF",
}

ISO_CURRENCIES = set("""
AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND
BOB BRL BSD BTN BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF
DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD
HNL HRK HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KRW
KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR
MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN
PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN
SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND
VUV WST XAF XCD XOF XPF YER ZAR ZMW
""".split())

MARKET_WORDS = (
    "price", "rate", "current", "today", "live", "market cap", "volume",
    "24h", "24 hour", "supply", "convert", "conversion", "worth",
)

SYSTEM_PROMPT = """
You are a cryptocurrency and global currency market assistant.

Answer about cryptocurrencies, fiat currencies, currency conversions,
blockchains, wallets, mining, staking, exchanges and market concepts.

Rules:
1. Support any cryptocurrency found through CoinGecko, not only featured coins.
2. Use supplied live data for current prices, rates, volume and supply.
3. Never invent live values. Say when verified data is unavailable.
4. Clearly distinguish fiat currencies from cryptocurrencies.
5. Never claim a cryptocurrency is legal in every country. For legal questions,
   ask for the country and explain that regulations can change.
6. Do not promise profits or provide personalized financial advice.
7. Mention the data source and update time when live data is supplied.
8. Keep answers clear and concise.
"""


def previous_user_messages(history: list[dict]) -> list[str]:
    return [
        item.get("content", "") for item in history
        if item.get("role") == "user"
    ]


def find_fiat_mentions(text: str) -> list[tuple[int, str]]:
    lowered = text.lower()
    found: list[tuple[int, str]] = []

    for alias, code in sorted(FIAT_ALIASES.items(), key=lambda item: -len(item[0])):
        match = re.search(rf"\b{re.escape(alias)}\b", lowered)
        if match and code not in [item[1] for item in found]:
            found.append((match.start(), code))

    for match in re.finditer(r"\b[A-Za-z]{3}\b", text):
        code = match.group().upper()
        if code in ISO_CURRENCIES and code not in [item[1] for item in found]:
            found.append((match.start(), code))

    return sorted(found)


def find_amount(text: str) -> float:
    match = re.search(r"(?<![A-Za-z])([0-9][0-9,]*(?:\.[0-9]+)?)", text)
    return float(match.group(1).replace(",", "")) if match else 1.0


def find_target_price(text: str) -> float | None:
    """Return a price when the user asks for coins near a value."""
    lowered = text.lower()
    search_phrases = (
        "near",
        "near value",
        "around",
        "close to",
        "approximately",
        "approx",
        "similar price",
    )

    if not any(phrase in lowered for phrase in search_phrases):
        return None

    match = re.search(
        r"(?:[$£€]\s*)?([0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
    )
    if not match:
        return None

    try:
        price = float(match.group(1).replace(",", ""))
        return price if price > 0 else None
    except ValueError:
        return None


def needs_market_data(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in MARKET_WORDS)


def find_popular_coin_ids(text: str) -> list[str]:
    lowered = text.lower()
    result = []
    for name, coin_id in POPULAR_COINS.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered) and coin_id not in result:
            result.append(coin_id)
    return result


def extract_coin_query(text: str) -> str | None:
    """Extract an unknown coin name from common natural-language questions."""
    patterns = [
        r"(?:price|value|market cap|volume|supply)\s+(?:of\s+)?([a-z0-9][a-z0-9 .-]{0,35})",
        r"(?:tell me about|information about|info about)\s+([a-z0-9][a-z0-9 .-]{0,35})",
        r"what is\s+(?:the\s+)?(?:current\s+)?([a-z0-9][a-z0-9 .-]{0,35}?)(?:\s+price|\s+worth|\?|$)",
        r"([a-z0-9][a-z0-9 .-]{0,25})\s+(?:price|value|market cap|volume|supply)",
    ]
    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        candidate = re.split(
            r"\s+(?:in|to|today|now|currently|please)\b",
            match.group(1),
        )[0].strip(" .?!")
        candidate = re.sub(r"^(?:the|current|live)\s+", "", candidate)
        if candidate and not find_fiat_mentions(candidate):
            return candidate
    return None


async def resolve_coin_ids(message: str, history: list[dict]) -> list[str]:
    known = find_popular_coin_ids(message)
    if known:
        return known

    query = extract_coin_query(message)
    if query:
        match = await search_coin(query)
        return [match["id"]] if match else []

    # Support follow-ups such as "in PKR".
    for previous in reversed(previous_user_messages(history)):
        known = find_popular_coin_ids(previous)
        if known:
            return known
        query = extract_coin_query(previous)
        if query:
            match = await search_coin(query)
            if match:
                return [match["id"]]
    return []


async def collect_crypto_data(coin_ids: list[str], currency: str) -> list[dict]:
    results = []
    for coin_id in coin_ids:
        try:
            results.append(await get_crypto_price(coin_id, currency.lower()))
        except HTTPException as error:
            results.append({"coin_id": coin_id, "error": error.detail})
    return results


async def generate_chat_response(
    message: str,
    history: list[dict] | None = None,
) -> dict:
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY is missing from .env.")

    recent_history = (history or [])[-10:]
    fiat_mentions = find_fiat_mentions(message)
    fiat_codes = [item[1] for item in fiat_mentions]
    coin_ids = await resolve_coin_ids(message, recent_history)
    target_price = find_target_price(message)
    price_search_requested = target_price is not None

    live_data: dict = {
        "cryptocurrencies": [],
        "coins_near_price": None,
        "fiat_conversion": None,
    }

    if price_search_requested:
        search_currency = fiat_codes[-1] if fiat_codes else "USD"
        live_data["coins_near_price"] = await get_coins_near_price(
            target_price=target_price,
            currency=search_currency.lower(),
            limit=5,
        )

    # Two fiat currencies means a currency conversion/rate request.
    if not price_search_requested and len(fiat_codes) >= 2:
        live_data["fiat_conversion"] = await convert_currency(
            fiat_codes[0], fiat_codes[1], find_amount(message)
        )
    elif (
        not price_search_requested
        and len(fiat_codes) == 1
        and not coin_ids
        and needs_market_data(message)
    ):
        # For a single foreign-currency rate, use PKR as the practical default.
        target = "PKR" if fiat_codes[0] != "PKR" else "USD"
        live_data["fiat_conversion"] = await convert_currency(
            fiat_codes[0], target, find_amount(message)
        )

    if coin_ids and needs_market_data(message):
        quote_currency = fiat_codes[-1] if fiat_codes else "USD"
        live_data["cryptocurrencies"] = await collect_crypto_data(
            coin_ids, quote_currency
        )

    conversation = []
    for item in recent_history:
        role = "model" if item.get("role") == "assistant" else "user"
        conversation.append({
            "role": role,
            "parts": [{"text": item.get("content", "")}],
        })

    prompt = f"""
Latest user question:
{message}

Verified live data:
{json.dumps(live_data, ensure_ascii=False)}

Answer the latest question using conversation context. Only state a current
price or rate when it appears in verified live data. If coins_near_price is
present, list the closest matches with symbol, live price and difference from
the requested target. Explain that the search covers the ranked coins returned
by CoinGecko, not every cryptocurrency ever created.
"""
    conversation.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": conversation,
        "generationConfig": {"maxOutputTokens": 800},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            502,
            f"Gemini API error: {error.response.text}",
        ) from error
    except (KeyError, IndexError) as error:
        raise HTTPException(502, "Gemini returned an unexpected response.") from error
    except httpx.RequestError as error:
        raise HTTPException(
            503,
            f"Could not connect to Gemini: {error}",
        ) from error

    return {
        "reply": reply,
        "response_id": f"gemini-{uuid.uuid4()}",
    }