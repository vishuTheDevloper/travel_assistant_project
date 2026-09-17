"""Our custom MCP server exposing the weather and currency components.

The client starts this server as a local subprocess using the stdio transport.
Run python src/mcp_client.py from the project root to check the connection.
Keep stdout exclusively for MCP messages; diagnostics belong on stderr.
"""

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, StrictStr

if __package__:
    from .currency import CurrencyToolError, convert_currency as fetch_conversion
    from .weather import WeatherToolError, get_weather as fetch_weather
else:
    from currency import CurrencyToolError, convert_currency as fetch_conversion
    from weather import WeatherToolError, get_weather as fetch_weather


# Custom travel tools
# This project implements its own MCP server with weather and currency handlers.
# Weather comes from Open-Meteo; currency uses ECB reference rates through Frankfurter.
# Each handler returns source details and a clear error when data is unavailable.
SERVER_NAME = "singapore_travel"
mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "Travel tools for Singapore weather and currency conversion. "
        "Use the knowledge base separately for attractions and other static travel facts. "
        "Report weather dates, units and provider; current conditions have their own timestamp. "
        "Currency rates are daily reference rates: report the rate date and source. "
        "An unavailable tool result must not be replaced with invented data."
    ),
    log_level="WARNING",
)

READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


@mcp.tool(annotations=READ_ONLY_TOOL, structured_output=True)
async def get_weather(
    city: Annotated[StrictStr, Field(description="Singapore or SG; this project covers Singapore only.")] = "Singapore",
    start_date: Annotated[StrictStr | None, Field(description="First forecast date, YYYY-MM-DD in Singapore local time; defaults to today.")] = None,
    end_date: Annotated[StrictStr | None, Field(description="Last forecast date, inclusive; defaults to two days after the start date.")] = None,
) -> dict[str, Any]:
    """Fetch Singapore current weather estimates and a daily forecast from Open-Meteo.

    Dates must be within today through today+15 in Singapore local time. For a
    next-week trip, supply its actual dates. Preserve source, timestamps and
    units. Daily precipitation probabilities do not specify exact rain hours.
    This tool fetches weather data; it does not retrieve attraction suggestions.
    """
    try:
        result = await fetch_weather(city, start_date, end_date)
    except WeatherToolError as error:
        raise ToolError(f"{error.code}: {error}") from None
    return {**result, "provenance": {"channel": "mcp", "server": SERVER_NAME, "tool": "get_weather"}}


@mcp.tool(annotations=READ_ONLY_TOOL, structured_output=True)
async def convert_currency(
    amount: Annotated[float, Field(strict=True, ge=0, le=1000000000000, allow_inf_nan=False, description="Non-negative amount in the source currency, using its allowed decimal places.")],
    source_currency: Annotated[StrictStr, Field(description="Source code: INR, SGD, USD, EUR, GBP, AUD, CAD, CHF, or JPY.")] = "INR",
    target_currency: Annotated[StrictStr, Field(description="Target code from the same supported currency list.")] = "SGD",
) -> dict[str, Any]:
    """Estimate a currency conversion using the latest ECB reference rate via Frankfurter.

    Report the returned publication date, exchange rate and converted amount.
    This is a daily reference-rate estimate, excluding bank fees and margins.
    Money and rates are returned as decimal strings. Equal currencies use an
    identity rate without an external API request. This tool does not move money.
    """
    try:
        result = await fetch_conversion(amount, source_currency, target_currency)
    except CurrencyToolError as error:
        raise ToolError(f"{error.code}: {error}") from None
    return {**result, "provenance": {"channel": "mcp", "server": SERVER_NAME, "tool": "convert_currency"}}


if __name__ == "__main__":
    mcp.run(transport="stdio")
