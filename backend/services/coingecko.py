import os
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")


def _headers() -> dict[str, str]:
    if COINGECKO_API_KEY:
        return {"x-cg-demo-api-key": COINGECKO_API_KEY}
    return {}


async def search_coin(query: str) -> dict | None:
    """Find a CoinGecko coin dynamically by name or ticker symbol."""
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(
                f"{COINGECKO_API_URL}/search",
                params={"query": query.strip()},
                headers=_headers(),
            )
            response.raise_for_status()
            coins = response.json().get("coins", [])
    except httpx.TimeoutException as error:
        raise HTTPException(504, "CoinGecko search timed out.") from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            error.response.status_code,
            "CoinGecko search returned an error.",
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(503, "Could not connect to CoinGecko.") from error

    if not coins:
        return None

    normalized = query.strip().lower()
    exact = [
        coin for coin in coins
        if coin.get("name", "").lower() == normalized
        or coin.get("symbol", "").lower() == normalized
        or coin.get("id", "").lower() == normalized
    ]
    candidates = exact or coins
    candidates.sort(key=lambda coin: coin.get("market_cap_rank") or 10**9)
    coin = candidates[0]
    return {
        "id": coin["id"],
        "name": coin["name"],
        "symbol": coin["symbol"].upper(),
        "market_cap_rank": coin.get("market_cap_rank"),
    }


async def get_crypto_price(coin_id: str, currency: str = "usd") -> dict:
    """Retrieve current market data, including circulating supply."""
    normalized_currency = currency.lower()
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(
                f"{COINGECKO_API_URL}/coins/markets",
                params={
                    "vs_currency": normalized_currency,
                    "ids": coin_id.lower(),
                    "price_change_percentage": "24h",
                    "sparkline": "false",
                },
                headers=_headers(),
            )
            response.raise_for_status()
            rows = response.json()
    except httpx.TimeoutException as error:
        raise HTTPException(504, "CoinGecko did not respond in time.") from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            error.response.status_code,
            "CoinGecko returned an error.",
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(503, "Could not connect to CoinGecko.") from error

    if not rows:
        raise HTTPException(404, f"Cryptocurrency '{coin_id}' was not found.")

    coin = rows[0]
    return {
        "coin_id": coin["id"],
        "name": coin["name"],
        "symbol": coin["symbol"].upper(),
        "currency": normalized_currency.upper(),
        "price": coin.get("current_price"),
        "market_cap": coin.get("market_cap"),
        "market_cap_rank": coin.get("market_cap_rank"),
        "volume_24h": coin.get("total_volume"),
        "high_24h": coin.get("high_24h"),
        "low_24h": coin.get("low_24h"),
        "change_24h_percent": coin.get("price_change_percentage_24h"),
        "circulating_supply": coin.get("circulating_supply"),
        "total_supply": coin.get("total_supply"),
        "max_supply": coin.get("max_supply"),
        "source_last_updated_at": coin.get("last_updated"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "CoinGecko",
    }


async def get_coins_near_price(
    target_price: float,
    currency: str = "usd",
    limit: int = 5,
) -> dict:
    """Find ranked cryptocurrencies closest to a requested live price."""
    if target_price <= 0:
        raise HTTPException(400, "Target price must be greater than zero.")
    if limit < 1 or limit > 20:
        raise HTTPException(400, "Result limit must be between 1 and 20.")

    normalized_currency = currency.strip().lower()
    all_coins = []

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Scan the top 500 cryptocurrencies by market capitalization.
            for page in range(1, 3):
                response = await client.get(
                    f"{COINGECKO_API_URL}/coins/markets",
                    params={
                        "vs_currency": normalized_currency,
                        "order": "market_cap_desc",
                        "per_page": 250,
                        "page": page,
                        "sparkline": "false",
                        "price_change_percentage": "24h",
                    },
                    headers=_headers(),
                )
                response.raise_for_status()
                page_coins = response.json()
                if not page_coins:
                    break
                all_coins.extend(page_coins)
    except httpx.TimeoutException as error:
        raise HTTPException(504, "CoinGecko price search timed out.") from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            error.response.status_code,
            f"CoinGecko price search returned an error: {error.response.text}",
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(503, "Could not connect to CoinGecko.") from error

    matches = []
    for coin in all_coins:
        try:
            current_price = float(coin.get("current_price"))
        except (TypeError, ValueError):
            continue

        difference = abs(current_price - target_price)
        matches.append({
            "coin_id": coin.get("id"),
            "name": coin.get("name"),
            "symbol": coin.get("symbol", "").upper(),
            "price": current_price,
            "currency": normalized_currency.upper(),
            "difference": difference,
            "difference_percent": (difference / target_price) * 100,
            "market_cap_rank": coin.get("market_cap_rank"),
            "market_cap": coin.get("market_cap"),
            "volume_24h": coin.get("total_volume"),
            "change_24h_percent": coin.get("price_change_percentage_24h"),
            "last_updated": coin.get("last_updated"),
        })

    matches.sort(key=lambda coin: (
        coin["difference"],
        coin["market_cap_rank"] or 10**9,
    ))

    return {
        "target_price": target_price,
        "currency": normalized_currency.upper(),
        "matches": matches[:limit],
        "coins_checked": len(all_coins),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "CoinGecko",
    }