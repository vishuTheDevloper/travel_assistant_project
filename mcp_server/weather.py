"""Fetch Singapore weather for the custom MCP server we will build next.
Run from the project root: python mcp_server/weather.py
Optional dates: python mcp_server/weather.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

Provider documentation: https://open-meteo.com/en/docs
Data attribution: Open-Meteo, https://open-meteo.com/ (CC BY 4.0).
"""

import argparse
import asyncio
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_NAME = "Asia/Singapore"
# Singapore has a fixed UTC+8 offset; this also works on Windows without tzdata.
SINGAPORE_TIMEZONE = timezone(timedelta(hours=8), name=TIMEZONE_NAME)
LATITUDE = 1.3521
LONGITUDE = 103.8198
MAX_FORECAST_DAYS = 16
CURRENT_FIELDS = {
    "temperature_2m": "°C",
    "weather_code": "wmo code",
    "precipitation": "mm",
    "wind_speed_10m": "km/h",
}
DAILY_FIELDS = {
    "weather_code": "wmo code",
    "temperature_2m_max": "°C",
    "temperature_2m_min": "°C",
    "precipitation_sum": "mm",
    "precipitation_probability_max": "%",
    "wind_speed_10m_max": "km/h",
}
WEATHER_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherToolError(Exception):
    """An explicit failure that the MCP wrapper can report to its client."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def singapore_today() -> date:
    return datetime.now(SINGAPORE_TIMEZONE).date()


def parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
        if value != parsed.isoformat():
            raise ValueError
        return parsed
    except (ValueError, TypeError):
        raise WeatherToolError("invalid_dates", "Use dates in YYYY-MM-DD format.") from None


def resolve_dates(start_date: str | None, end_date: str | None) -> tuple[date, date, date]:
    today = singapore_today()
    start = today if start_date is None else parse_date(start_date)
    end = start + timedelta(days=2) if end_date is None else parse_date(end_date)
    last_allowed = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    if end < start:
        raise WeatherToolError("invalid_dates", "The end date must be on or after the start date.")
    if start < today or end > last_allowed:
        raise WeatherToolError(
            "dates_outside_forecast_window",
            f"Request dates between {today.isoformat()} and {last_allowed.isoformat()} "
            f"inclusive, using Singapore local time. Forecasts beyond this window are unavailable.",
        )
    return today, start, end


def checked_number(value, field: str):
    """Preserve missing values as None, never as zero or made-up estimates."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WeatherToolError("invalid_provider_response", f"Invalid value for {field}.")
    if field == "weather_code" and value not in WEATHER_DESCRIPTIONS:
        raise WeatherToolError("invalid_provider_response", "Unrecognised weather condition code.")
    if field == "precipitation_probability_max" and not 0 <= value <= 100:
        raise WeatherToolError("invalid_provider_response", "Invalid precipitation probability.")
    if field in {"precipitation", "precipitation_sum", "wind_speed_10m", "wind_speed_10m_max"} and value < 0:
        raise WeatherToolError("invalid_provider_response", f"Negative value for {field}.")
    return value


def check_units(payload: dict, section: str, expected: dict) -> None:
    units = payload.get(f"{section}_units")
    if not isinstance(units, dict) or any(units.get(key) != unit for key, unit in expected.items()):
        raise WeatherToolError("invalid_provider_response", f"Unexpected {section} units from Open-Meteo.")


def normalise_response(payload: dict, start: date, end: date) -> dict:
    """Validate dates, units and values, then select exactly the requested days."""
    if not isinstance(payload, dict) or payload.get("error"):
        raise WeatherToolError("invalid_provider_response", "Open-Meteo did not return usable weather data.")
    if payload.get("timezone") != TIMEZONE_NAME or payload.get("utc_offset_seconds") != 28800:
        raise WeatherToolError("invalid_provider_response", "The provider returned an unexpected timezone.")

    daily = payload.get("daily")
    if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
        raise WeatherToolError("invalid_provider_response", "Daily forecast dates are missing.")
    dates = daily["time"]
    if not dates or any(not isinstance(day, str) for day in dates) or len(set(dates)) != len(dates):
        raise WeatherToolError("invalid_provider_response", "Invalid or duplicate forecast dates.")
    for field in DAILY_FIELDS:
        if not isinstance(daily.get(field), list) or len(daily[field]) != len(dates):
            raise WeatherToolError("invalid_provider_response", f"Missing or incomplete forecast field: {field}.")
    check_units(payload, "daily", DAILY_FIELDS)
    requested_dates = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    date_indexes = {day: index for index, day in enumerate(dates)}
    if any(day not in date_indexes for day in requested_dates):
        raise WeatherToolError("forecast_unavailable", "The provider did not return every requested forecast date.")

    forecast = []
    missing_fields = []
    for day in requested_dates:
        index = date_indexes[day]
        values = {field: checked_number(daily[field][index], field) for field in DAILY_FIELDS}
        if all(value is None for value in values.values()):
            raise WeatherToolError("forecast_unavailable", f"No forecast values are available for {day}.")
        low, high = values["temperature_2m_min"], values["temperature_2m_max"]
        if low is not None and high is not None and low > high:
            raise WeatherToolError("invalid_provider_response", "Minimum temperature exceeds maximum temperature.")
        missing_fields.extend(f"{day}: {field}" for field, value in values.items() if value is None)
        forecast.append({"date": day, **values, "condition": WEATHER_DESCRIPTIONS.get(values["weather_code"])})

    # Current conditions have their own timestamp; they do not describe future dates.
    raw_current = payload.get("current")
    current = None
    if raw_current is not None:
        if not isinstance(raw_current, dict):
            raise WeatherToolError("invalid_provider_response", "Invalid current weather section.")
        check_units(payload, "current", CURRENT_FIELDS)
        try:
            current_time = datetime.fromisoformat(raw_current["time"])
            if current_time.tzinfo is not None:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise WeatherToolError("invalid_provider_response", "Invalid current weather timestamp.") from None
        values = {field: checked_number(raw_current.get(field), field) for field in CURRENT_FIELDS}
        if any(value is not None for value in values.values()):
            interval = raw_current.get("interval")
            if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
                raise WeatherToolError("invalid_provider_response", "Invalid current weather interval.")
            current = {
                "time": current_time.replace(tzinfo=SINGAPORE_TIMEZONE).isoformat(),
                "interval_seconds": interval,
                **values,
                "condition": WEATHER_DESCRIPTIONS.get(values["weather_code"]),
                "data_kind": "model_based_current_conditions",
            }
            missing_fields.extend(f"current: {field}" for field, value in values.items() if value is None)
    if current is None:
        missing_fields.append("Current weather conditions are unavailable.")

    return {
        "status": "ok",
        "provider": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "documentation_url": "https://open-meteo.com/en/docs",
        "attribution": "Weather data by Open-Meteo (CC BY 4.0); dates selected and WMO codes labelled by this application.",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": "Singapore",
        "coordinates_requested": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "timezone": TIMEZONE_NAME,
        "requested_start_date": start.isoformat(),
        "requested_end_date": end.isoformat(),
        "current": current,
        "current_units": CURRENT_FIELDS,
        "forecast": forecast,
        "forecast_units": DAILY_FIELDS,
        "missing_fields": missing_fields,
        "notes": [
            "Current conditions are weather-model estimates, with their own valid timestamp.",
            "The forecast represents a Singapore location, not separate forecasts for every neighbourhood.",
            "Daily conditions and precipitation probabilities do not specify exactly when rain will occur or mean it will rain all day.",
            "Forecasts can change; null values mean unavailable information.",
        ],
    }


async def get_weather(
    city: str = "Singapore", start_date: str | None = None, end_date: str | None = None
) -> dict:
    """Get Singapore current conditions and a daily forecast for inclusive dates.

    Use YYYY-MM-DD dates within today through today+15 in Singapore local time.
    Omitted dates mean today and the next two days. For next-week requests,
    supply the actual trip dates. This project supports Singapore only.
    """
    if not isinstance(city, str) or city.strip().casefold() not in {"singapore", "sg"}:
        raise WeatherToolError("unsupported_destination", "This travel assistant supports Singapore weather only.")
    today, start, end = resolve_dates(start_date, end_date)
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE, "timezone": TIMEZONE_NAME,
        "current": ",".join(CURRENT_FIELDS), "daily": ",".join(DAILY_FIELDS),
        "forecast_days": (end - today).days + 1,
        "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
            response = await client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        raise WeatherToolError("provider_timeout", "Open-Meteo timed out. Weather information is unavailable; try again later.") from None
    except httpx.HTTPStatusError as error:
        raise WeatherToolError("provider_http_error", f"Open-Meteo returned HTTP {error.response.status_code}. Weather information is unavailable.") from None
    except httpx.RequestError:
        raise WeatherToolError("provider_connection_error", "Could not connect to Open-Meteo. Check your internet connection and try again.") from None
    except ValueError:
        raise WeatherToolError("invalid_provider_response", "Open-Meteo returned an unreadable response.") from None
    result = normalise_response(payload, start, end)
    result["request_url"] = str(response.request.url)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="Singapore")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    print("Checking the weather component directly with Open-Meteo...")
    try:
        result = asyncio.run(get_weather(args.city, args.start_date, args.end_date))
    except WeatherToolError as error:
        print(f"Weather unavailable ({error.code}): {error}")
        print("No new weather result was saved.")
        raise SystemExit(1) from None
    output = PROJECT_ROOT / "data" / "processed" / "weather_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"check_type": "direct_weather_component", "mcp_protocol_tested": False, "weather": result}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Provider:", result["provider"])
    print("Timezone:", result["timezone"])
    print("Requested dates:", result["requested_start_date"], "to", result["requested_end_date"])
    if result["current"] is not None:
        current = result["current"]
        print("Current model estimate:", current["temperature_2m"], "C;", current["condition"], "at", current["time"])
    for day in result["forecast"]:
        print(f"{day['date']}: {day['condition']}; {day['temperature_2m_min']} to {day['temperature_2m_max']} C; "
              f"precipitation {day['precipitation_sum']} mm; maximum precipitation probability {day['precipitation_probability_max']}%")
    if result["missing_fields"]:
        print("Some fields are unavailable; see missing_fields in the saved result.")
    print("Saved data/processed/weather_check.json")
    print("This checked the weather component. The custom MCP server and client are the next integration steps.")


if __name__ == "__main__":
    main()
