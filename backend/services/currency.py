from datetime import datetime, timezone

import httpx
from fastapi import HTTPException


EXCHANGE_API_URL = "https://open.er-api.com/v6/latest"


async def convert_currency(
    base: str,
    target: str,
    amount: float = 1.0,
) -> dict:
    """
    Convert one fiat currency into another.

    Examples:
    USD to PKR
    GBP to PKR
    SAR to PKR
    AED to USD
    """

    base = base.upper()
    target = target.upper()

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(
                f"{EXCHANGE_API_URL}/{base}"
            )

            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Exchange-rate service timed out.",
        ) from error

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail="Exchange-rate service returned an error.",
        ) from error

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the exchange-rate service.",
        ) from error

    if data.get("result") != "success":
        raise HTTPException(
            status_code=502,
            detail="Exchange-rate service returned invalid data.",
        )

    rate = data.get("rates", {}).get(target)

    if rate is None:
        raise HTTPException(
            status_code=404,
            detail=f"Currency '{target}' is not supported.",
        )

    return {
        "base": base,
        "target": target,
        "amount": amount,
        "rate": rate,
        "converted_amount": amount * rate,
        "source_last_updated_at": data.get(
            "time_last_update_utc"
        ),
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": "ExchangeRate-API open endpoint",
    }