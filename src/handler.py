"""
forecast-weather-tool-lambda

Deterministic tool Lambda for the Agentic Weather App (Style 3: Bedrock
AgentCore). Registered as an AWS Lambda target behind an AgentCore Gateway.
Fetches a 10-day daily forecast from Open-Meteo for a given
latitude/longitude. No AI, no reasoning, no loop -- it's called by the
Orchestrator Runtime's MCP client, through the Gateway.

Structurally identical to its sibling, current-weather-tool-lambda: same
invocation contract, same dependency-free design, same CI/CD pattern. Only
the Open-Meteo query (`daily` instead of `current`) and the response shape
(a 10-entry `days` array instead of a single reading) differ.

Invocation contract (AgentCore Gateway Lambda target):
    event   -- a flat dict of the tool's inputSchema properties, e.g.
               {"latitude": 40.7128, "longitude": -74.0060}
    context -- AgentCore populates context.client_context.custom with
               bedrockAgentCore* metadata (tool name, gateway id, target id,
               etc). See schema/gateway-tool-schema.json for the registered
               inputSchema/outputSchema, and events/ for sample payloads.

Returns the forecast dict directly. Gateway maps a Lambda target's return
value straight back to the MCP tool result -- there is no statusCode/body
HTTP-style envelope here, unlike a Bedrock Agent Action Group or a Lambda
Function URL.

Only the Python standard library is used (urllib), so the deployment
package is just this file zipped up -- no dependency layer, no
`pip install -t`, no vendoring step in CI.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 8
FORECAST_DAYS = 10

DAILY_FIELDS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
)


class WeatherError(Exception):
    """Raised when Open-Meteo can't be reached or returns an unusable payload."""


class InputError(Exception):
    """Raised when the incoming event is missing or has invalid coordinates."""


def fetch_forecast(latitude: float, longitude: float) -> dict[str, Any]:
    """Call Open-Meteo's daily-forecast endpoint and return a 10-day summary."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_FIELDS),
        "forecast_days": FORECAST_DAYS,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WeatherError(f"Open-Meteo request failed: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WeatherError(f"Open-Meteo returned unparseable JSON: {exc}") from exc

    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        raise WeatherError("Open-Meteo response had no 'daily' block")

    days = []
    for i, date in enumerate(daily["time"]):
        days.append(
            {
                "date": date,
                "weather_code": _at(daily, "weather_code", i),
                "temp_max_f": _at(daily, "temperature_2m_max", i),
                "temp_min_f": _at(daily, "temperature_2m_min", i),
                "feels_like_max_f": _at(daily, "apparent_temperature_max", i),
                "feels_like_min_f": _at(daily, "apparent_temperature_min", i),
                "precipitation_in": _at(daily, "precipitation_sum", i),
                "precipitation_probability_pct": _at(daily, "precipitation_probability_max", i),
                "wind_speed_max_mph": _at(daily, "wind_speed_10m_max", i),
                "wind_direction_deg": _at(daily, "wind_direction_10m_dominant", i),
            }
        )

    return {
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone"),
        "forecast_days": len(days),
        "days": days,
    }


def _at(daily: dict[str, Any], field: str, index: int) -> Any:
    """Safely index into one of Open-Meteo's parallel daily-field arrays."""
    values = daily.get(field)
    if not values or index >= len(values):
        return None
    return values[index]


def _parse_coordinates(event: dict[str, Any]) -> tuple[float, float]:
    """
    Gateway passes a flat dict of inputSchema properties. Numbers usually
    arrive as JSON numbers, but accept numeric strings defensively too --
    different MCP clients serialize tool-call arguments slightly
    differently. Both keys must be present and convertible to float.
    """
    if not isinstance(event, dict):
        raise InputError("Event must be a JSON object")

    try:
        latitude = float(event["latitude"])
        longitude = float(event["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError("Missing or invalid 'latitude'/'longitude'") from exc

    return latitude, longitude


def _tool_name_from_context(context: Any) -> str | None:
    """
    Best-effort extraction of the Gateway-assigned tool name, for logging
    only. AgentCore sets context.client_context.custom when the Gateway
    invokes this function; a direct boto3 `invoke()` call (no
    ClientContext param) leaves client_context unset, so this safely
    returns None outside the Gateway path.
    """
    client_context = getattr(context, "client_context", None)
    if not client_context:
        return None
    custom = getattr(client_context, "custom", None) or {}
    return custom.get("bedrockAgentCoreToolName")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    tool_name = _tool_name_from_context(context)
    if tool_name:
        logger.info("Invoked via AgentCore Gateway as tool=%s", tool_name)

    try:
        latitude, longitude = _parse_coordinates(event)
    except InputError as exc:
        logger.warning("Input error: %s", exc)
        return {"error": str(exc)}

    try:
        return fetch_forecast(latitude, longitude)
    except WeatherError as exc:
        logger.error("Upstream error: %s", exc)
        return {"error": str(exc)}
