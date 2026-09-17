"""Currency-conversion logic for our custom travel MCP server.

Run from the project root: python mcp_server/currency.py
Example: python mcp_server/currency.py --amount 100 --from-currency USD --to-currency SGD

This file checks the provider directly. The MCP server/client connection is a
separate integration step. No Gemini call, API key, or automatic retry is used.

Provider documentation: https://frankfurter.dev/
We request the latest ECB daily reference rate via Frankfurter's v2 REST API
and calculate the converted amount ourselves using Decimal arithmetic.
"""

import argparse
import asyncio
import json
from datetime import date, datetime, timezone
from decimal import Decimal, DecimalException, ROUND_HALF_UP, localcontext
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.frankfurter.dev/v2/rate"
ECB_URL = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"
# Currency minor units determine how many decimal places we display.
SUPPORTED_CURRENCIES = {
    "INR": 2, "SGD": 2, "USD": 2, "EUR": 2, "GBP": 2,
    "AUD": 2, "CAD": 2, "CHF": 2, "JPY": 0,
}
MAX_AMOUNT = Decimal("1000000000000")
# Application policy: do not present older data as a current travel estimate.
MAX_RATE_AGE_DAYS = 7


class CurrencyToolError(Exception):
    """A clear failure that the MCP wrapper can report to its client."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def currency_code(value: str) -> str:
    if not isinstance(value, str):
        raise CurrencyToolError("invalid_currency", "Use a supported three-letter currency code.")
    code = value.strip().upper()
    if code not in SUPPORTED_CURRENCIES:
        raise CurrencyToolError(
            "unsupported_currency",
            "Supported currencies: " + ", ".join(SUPPORTED_CURRENCIES) + ".",
        )
    return code


def money_amount(amount, source_currency: str) -> Decimal:
    """Validate non-negative finite amounts without silently rounding the input."""
    if isinstance(amount, bool) or len(str(amount)) > 80:
        raise CurrencyToolError("invalid_amount", "Enter a numeric amount, such as 10000.")
    try:
        value = Decimal(str(amount).strip())
        if not value.is_finite() or value < 0 or value > MAX_AMOUNT:
            raise CurrencyToolError("invalid_amount", f"The amount must be between 0 and {MAX_AMOUNT}.")
        places = SUPPORTED_CURRENCIES[source_currency]
        quantum = Decimal(1).scaleb(-places)
        rounded = value.quantize(quantum)
        if value != rounded:
            raise CurrencyToolError("invalid_amount", f"{source_currency} amounts must have at most {places} decimal places.")
        return abs(rounded) if rounded == 0 else rounded
    except DecimalException:
        raise CurrencyToolError("invalid_amount", "Enter a valid numeric amount without currency symbols or commas.") from None


def validate_rate(payload: dict, source: str, target: str) -> tuple[Decimal, date, int]:
    """Check the requested pair, rate value, and publication date."""
    if not isinstance(payload, dict) or payload.get("base") != source or payload.get("quote") != target:
        raise CurrencyToolError("invalid_provider_response", "The provider returned an unexpected currency pair.")
    raw_rate = payload.get("rate")
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float, Decimal)):
        raise CurrencyToolError("invalid_provider_response", "The provider did not return a numeric exchange rate.")
    try:
        rate = Decimal(str(raw_rate))
        if not rate.is_finite() or rate <= 0:
            raise ValueError
        raw_date = payload["date"]
        rate_date = date.fromisoformat(raw_date)
        if raw_date != rate_date.isoformat():
            raise ValueError
    except (ValueError, TypeError, KeyError, DecimalException):
        raise CurrencyToolError("invalid_provider_response", "Invalid exchange rate or publication date.") from None
    age = (datetime.now(timezone.utc).date() - rate_date).days
    if age < 0:
        raise CurrencyToolError("invalid_provider_response", "The provider returned a future publication date.")
    if age > MAX_RATE_AGE_DAYS:
        raise CurrencyToolError("stale_rate", f"The available rate is dated {rate_date.isoformat()}, more than {MAX_RATE_AGE_DAYS} days old. A recent rate is unavailable.")
    return rate, rate_date, age


async def convert_currency(
    amount: float, source_currency: str = "INR", target_currency: str = "SGD"
) -> dict:
    """Convert a travel budget using the latest ECB reference rate via Frankfurter.

    Supported codes: INR, SGD, USD, EUR, GBP, AUD, CAD, CHF, JPY. The amount must
    be non-negative and respect the source currency's minor units. Returned
    money and rates are decimal strings. This estimates a conversion; it does
    not exchange money or provide a bank quote. Report rate_date to the user.
    """
    source = currency_code(source_currency)
    target = currency_code(target_currency)
    value = money_amount(amount, source)
    request_url = None
    rate_date = None
    rate_age = None

    if source == target:
        # A same-currency amount is unchanged; no external rate is needed.
        rate = Decimal("1")
        rate_kind = "same_currency_identity"
    else:
        url = f"{API_ROOT}/{source.lower()}/{target.lower()}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
                response = await client.get(url, params={"providers": "ecb"})
                response.raise_for_status()
                # Preserve decimal digits from JSON instead of parsing them as binary floats.
                payload = response.json(parse_float=Decimal)
        except httpx.TimeoutException:
            raise CurrencyToolError("provider_timeout", "Frankfurter timed out. The conversion is unavailable; try again later.") from None
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise CurrencyToolError("rate_unavailable", "No ECB rate is available for this currency pair.") from None
            raise CurrencyToolError("provider_http_error", f"Frankfurter returned HTTP {error.response.status_code}. The conversion is unavailable.") from None
        except httpx.RequestError:
            raise CurrencyToolError("provider_connection_error", "Could not connect to Frankfurter. Check your internet connection and try again.") from None
        except ValueError:
            raise CurrencyToolError("invalid_provider_response", "Frankfurter returned an unreadable response.") from None
        rate, rate_date, rate_age = validate_rate(payload, source, target)
        request_url = str(response.request.url)
        rate_kind = "daily_reference_rate"

    places = SUPPORTED_CURRENCIES[target]
    try:
        with localcontext() as context:
            context.prec = 40
            converted = (value * rate).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    except DecimalException:
        raise CurrencyToolError("invalid_provider_response", "The returned rate could not be used to calculate a valid amount.") from None

    fetched = source != target
    return {
        "status": "ok",
        "amount": format(value, "f"),
        "source_currency": source,
        "target_currency": target,
        "exchange_rate": str(rate),
        "rate_definition": f"1 {source} = {rate} {target}",
        "converted_amount": format(converted, "f"),
        "calculation": f"{value} {source} x {rate} = {converted} {target} (rounded)",
        "rounding": {"method": "ROUND_HALF_UP", "decimal_places": places},
        "rate_kind": rate_kind,
        "rate_date": rate_date.isoformat() if rate_date else None,
        "rate_age_days": rate_age,
        "calculated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat() if fetched else None,
        "provider": "Frankfurter" if fetched else None,
        "rate_source": "European Central Bank (ECB)" if fetched else None,
        "source_url": ECB_URL if fetched else None,
        "documentation_url": "https://frankfurter.dev/" if fetched else None,
        "request_url": request_url,
        "external_rate_fetched": fetched,
        "notes": [
            "This is the latest published daily reference rate returned by the API. The publication date may be an earlier business day.",
            "The conversion is an estimate; provider fees and bank or card exchange-rate margins are not included.",
        ] if fetched else ["The source and target currencies are the same. No external exchange rate was requested."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amount", default="10000")
    parser.add_argument("--from-currency", default="INR")
    parser.add_argument("--to-currency", default="SGD")
    args = parser.parse_args()
    print("Checking the currency-conversion component...")
    try:
        result = asyncio.run(convert_currency(args.amount, args.from_currency, args.to_currency))
    except CurrencyToolError as error:
        print(f"Conversion unavailable ({error.code}): {error}")
        print("No new currency result was saved.")
        raise SystemExit(1) from None

    output = PROJECT_ROOT / "data" / "processed" / "currency_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"check_type": "direct_currency_component", "mcp_protocol_tested": False, "currency": result}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{result['amount']} {result['source_currency']} = {result['converted_amount']} {result['target_currency']}")
    print("Rate:", result["rate_definition"])
    if result["external_rate_fetched"]:
        print("Rate date:", result["rate_date"])
        print("Source:", result["rate_source"], "via", result["provider"])
        print("Reference-rate estimate; fees and provider margins are excluded.")
    else:
        print("Same currency: no external rate needed.")
    print("Saved data/processed/currency_check.json")
    print("This checked the currency provider directly. MCP integration and the app are tested separately.")


if __name__ == "__main__":
    main()
