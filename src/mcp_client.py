"""Our MCP client: start our server, discover tools, and call them through MCP.

Run from the project root: python src/mcp_client.py
The client automatically starts and closes mcp_server/server.py using the same
Python interpreter as this process. No separate server terminal is required.

This checks MCP communication and LangChain tool conversion. Gemini routing,
conversation memory, and the user interface are later integration steps.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Implementation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"
SERVER_NAME = "singapore_travel"
EXPECTED_TOOLS = {"get_weather", "convert_currency"}
SINGAPORE_TIMEZONE = timezone(timedelta(hours=8))


def server_parameters() -> StdioServerParameters:
    for name in ("server.py", "weather.py", "currency.py"):
        if not (SERVER_PATH.parent / name).is_file():
            raise FileNotFoundError(f"Place {name} inside the project's mcp_server folder.")
    # The SDK inherits standard OS variables. Add network settings when present,
    # rather than passing every environment variable to the subprocess.
    network_variables = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    )
    environment = {key: os.environ[key] for key in network_variables if key in os.environ}
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
        env=environment,
        encoding="utf-8",
    )


@asynccontextmanager
async def open_travel_session():
    """Keep the initialized MCP session alive while its tools are in use."""
    async with stdio_client(server_parameters()) as (read, write):
        async with ClientSession(
            read, write,
            read_timeout_seconds=timedelta(seconds=45),
            client_info=Implementation(name="travel_assignment_client", version="1.0.0"),
        ) as session:
            initialization = await session.initialize()
            if initialization.serverInfo.name != SERVER_NAME:
                raise RuntimeError("The connected server has an unexpected name.")
            yield session, initialization


# MCP connection and discovery
# Start the project server through the active Python interpreter and initialize stdio.
# Discover its tools and adapt them for LangChain while the MCP session stays open.
@asynccontextmanager
async def open_travel_tools():
    """Reusable entry point for the later LangChain assistant.

    Keep agent execution inside this async context. Returned tools use the live
    session and must not be stored or called after this context closes.
    """
    async with open_travel_session() as (session, _):
        tools = await load_mcp_tools(session, server_name=SERVER_NAME, handle_tool_errors=True)
        if {tool.name for tool in tools} != EXPECTED_TOOLS:
            raise RuntimeError("The expected weather and currency tools were not discovered.")
        yield tools


def checked_tool_data(result: CallToolResult, tool_name: str) -> dict:
    """Accept successful structured MCP results with the expected provenance."""
    if result.isError:
        text = " ".join(block.text for block in result.content if block.type == "text")
        raise RuntimeError(f"{tool_name} failed: {text}")
    data = result.structuredContent
    if not isinstance(data, dict) or data.get("status") != "ok":
        raise RuntimeError(f"{tool_name} did not return a successful structured result.")
    expected = {"channel": "mcp", "server": SERVER_NAME, "tool": tool_name}
    if data.get("provenance") != expected:
        raise RuntimeError(f"{tool_name} returned unexpected provenance.")
    return data


async def check_connection() -> dict:
    today = datetime.now(SINGAPORE_TIMEZONE).date()
    next_monday = today + timedelta(days=7 - today.weekday())
    next_wednesday = next_monday + timedelta(days=2)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "stdio",
        "server_file": "mcp_server/server.py",
        "mcp_protocol_tested": True,
        "llm_routing_tested": False,
        "package_versions": {name: version(name) for name in ("mcp", "langchain-mcp-adapters", "langchain-core")},
        "checks": [],
    }
    print("Starting our MCP server and initializing the connection...")
    async with open_travel_session() as (session, initialization):
        report["initialization"] = initialization.model_dump(mode="json", exclude_none=True)
        print("Connected to:", initialization.serverInfo.name)
        print("Negotiated MCP protocol:", initialization.protocolVersion)

        listed = await session.list_tools()
        if listed.nextCursor is not None:
            raise RuntimeError("Unexpected pagination from our two-tool server.")
        if {tool.name for tool in listed.tools} != EXPECTED_TOOLS:
            raise RuntimeError("Our server did not advertise both expected tools.")
        report["discovered_tools"] = [tool.model_dump(mode="json", exclude_none=True) for tool in listed.tools]
        print("Discovered tools:", ", ".join(sorted(EXPECTED_TOOLS)))

        weather_args = {"city": "Singapore", "start_date": next_monday.isoformat(), "end_date": next_wednesday.isoformat()}
        print("Calling get_weather through MCP for", next_monday, "to", next_wednesday, "...")
        result = await session.call_tool("get_weather", weather_args)
        weather = checked_tool_data(result, "get_weather")
        expected_dates = [(next_monday + timedelta(days=i)).isoformat() for i in range(3)]
        if [day["date"] for day in weather["forecast"]] != expected_dates:
            raise RuntimeError("The weather tool returned different forecast dates.")
        report["checks"].append({"name": "weather_via_mcp", "passed": True, "arguments": weather_args, "result": result.model_dump(mode="json", exclude_none=True)})
        print("PASS: three requested forecast dates received through MCP.")

        currency_args = {"amount": 10000, "source_currency": "INR", "target_currency": "SGD"}
        print("Calling convert_currency through MCP...")
        result = await session.call_tool("convert_currency", currency_args)
        currency = checked_tool_data(result, "convert_currency")
        if currency["source_currency"] != "INR" or currency["target_currency"] != "SGD" or not currency["external_rate_fetched"]:
            raise RuntimeError("The currency tool returned a different conversion.")
        report["checks"].append({"name": "currency_via_mcp", "passed": True, "arguments": currency_args, "result": result.model_dump(mode="json", exclude_none=True)})
        print(f"PASS: {currency['amount']} INR = {currency['converted_amount']} SGD; rate date {currency['rate_date']}.")

        # These intentionally invalid inputs must produce MCP error responses.
        failure_checks = [
            ("unsupported_weather_destination", "get_weather", {"city": "Tokyo"}, "unsupported_destination"),
            ("negative_currency_amount", "convert_currency", {"amount": -5, "source_currency": "INR", "target_currency": "SGD"}, "greater_than_equal"),
        ]
        for name, tool_name, arguments, expected_text in failure_checks:
            result = await session.call_tool(tool_name, arguments)
            details = " ".join(block.text for block in result.content if block.type == "text")
            if not result.isError or expected_text not in details:
                raise RuntimeError(f"The {name} check did not produce the expected MCP error.")
            report["checks"].append({"name": name, "passed": True, "arguments": arguments, "expected_error": True, "result": result.model_dump(mode="json", exclude_none=True)})
            print("PASS: MCP rejected", name.replace("_", " ") + ".")

        tools = await load_mcp_tools(session, server_name=SERVER_NAME, handle_tool_errors=True)
        by_name = {tool.name: tool for tool in tools}
        if set(by_name) != EXPECTED_TOOLS:
            raise RuntimeError("LangChain did not load the expected tools.")
        report["langchain_tool_names"] = sorted(by_name)
        # Identity conversion proves the LangChain adapter calls the MCP server
        # without making another external exchange-rate request.
        message = await by_name["convert_currency"].ainvoke({
            "name": "convert_currency", "type": "tool_call", "id": "adapter_check_1",
            "args": {"amount": 1, "source_currency": "SGD", "target_currency": "SGD"},
        })
        if not isinstance(message, ToolMessage) or message.status != "success":
            raise RuntimeError("The LangChain tool invocation did not succeed.")
        artifact = message.artifact or {}
        data = artifact.get("structured_content", {})
        if data.get("converted_amount") != "1.00" or data.get("external_rate_fetched") is not False or data.get("provenance", {}).get("channel") != "mcp":
            raise RuntimeError("Unexpected result from the LangChain-to-MCP call.")
        report["checks"].append({"name": "langchain_adapter_call", "passed": True, "result": message.model_dump(mode="json", exclude_none=True)})
        print("PASS: LangChain successfully called our MCP currency tool.")

    report["all_checks_passed"] = True
    report["connection_closed"] = True
    return report


def error_messages(error: BaseException) -> list[str]:
    """Unwrap async task errors into short, readable messages for the CLI."""
    if isinstance(error, BaseExceptionGroup):
        return [message for child in error.exceptions for message in error_messages(child)]
    return [str(error) or type(error).__name__]


def main() -> None:
    try:
        report = asyncio.run(check_connection())
    except Exception as error:
        print("MCP check failed:", " | ".join(dict.fromkeys(error_messages(error))))
        print("No new MCP result was saved.")
        raise SystemExit(1) from None
    output = PROJECT_ROOT / "data" / "processed" / "mcp_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved data/processed/mcp_check.json")
    print("The MCP client connection and server subprocess have closed.")
    print("Next step: connect the discovered tools to the LangChain assistant.")


if __name__ == "__main__":
    main()
