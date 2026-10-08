import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src import handler


def _make_daily_block(num_days: int = 10) -> dict:
    return {
        "time": [f"2026-10-{3 + i:02d}" for i in range(num_days)],
        "weather_code": [1 + i for i in range(num_days)],
        "temperature_2m_max": [70.0 + i for i in range(num_days)],
        "temperature_2m_min": [55.0 + i for i in range(num_days)],
        "apparent_temperature_max": [71.0 + i for i in range(num_days)],
        "apparent_temperature_min": [53.0 + i for i in range(num_days)],
        "precipitation_sum": [0.0 for _ in range(num_days)],
        "precipitation_probability_max": [10 + i for i in range(num_days)],
        "wind_speed_10m_max": [8.0 + i for i in range(num_days)],
        "wind_direction_10m_dominant": [200 + i for i in range(num_days)],
    }


SAMPLE_OPEN_METEO_RESPONSE = {
    "latitude": 40.71,
    "longitude": -74.01,
    "timezone": "America/New_York",
    "daily": _make_daily_block(),
}


def _mock_urlopen_returning(payload_bytes: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


class TestFetchForecast:
    @patch("src.handler.urllib.request.urlopen")
    def test_success_returns_ten_days(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_returning(
            json.dumps(SAMPLE_OPEN_METEO_RESPONSE).encode("utf-8")
        )

        result = handler.fetch_forecast(40.71, -74.01)

        assert result["forecast_days"] == 10
        assert len(result["days"]) == 10
        assert result["timezone"] == "America/New_York"

    @patch("src.handler.urllib.request.urlopen")
    def test_success_maps_first_day_fields(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_returning(
            json.dumps(SAMPLE_OPEN_METEO_RESPONSE).encode("utf-8")
        )

        result = handler.fetch_forecast(40.71, -74.01)
        first_day = result["days"][0]

        assert first_day["date"] == "2026-10-03"
        assert first_day["temp_max_f"] == 70.0
        assert first_day["temp_min_f"] == 55.0
        assert first_day["feels_like_max_f"] == 71.0
        assert first_day["precipitation_probability_pct"] == 10
        assert first_day["wind_speed_max_mph"] == 8.0
        assert first_day["wind_direction_deg"] == 200
        assert first_day["weather_code"] == 1

    @patch("src.handler.urllib.request.urlopen")
    def test_network_error_raises_weather_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        with pytest.raises(handler.WeatherError):
            handler.fetch_forecast(40.71, -74.01)

    @patch("src.handler.urllib.request.urlopen")
    def test_timeout_raises_weather_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")

        with pytest.raises(handler.WeatherError):
            handler.fetch_forecast(40.71, -74.01)

    @patch("src.handler.urllib.request.urlopen")
    def test_bad_json_raises_weather_error(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_returning(b"not valid json")

        with pytest.raises(handler.WeatherError):
            handler.fetch_forecast(40.71, -74.01)

    @patch("src.handler.urllib.request.urlopen")
    def test_missing_daily_block_raises_weather_error(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_returning(
            json.dumps({"latitude": 1, "longitude": 2}).encode("utf-8")
        )

        with pytest.raises(handler.WeatherError):
            handler.fetch_forecast(1, 2)

    @patch("src.handler.urllib.request.urlopen")
    def test_empty_time_array_raises_weather_error(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_returning(
            json.dumps({"latitude": 1, "longitude": 2, "daily": {"time": []}}).encode("utf-8")
        )

        with pytest.raises(handler.WeatherError):
            handler.fetch_forecast(1, 2)


class TestAt:
    def test_returns_value_at_index(self):
        assert handler._at({"field": [1, 2, 3]}, "field", 1) == 2

    def test_missing_field_returns_none(self):
        assert handler._at({}, "field", 0) is None

    def test_index_out_of_range_returns_none(self):
        assert handler._at({"field": [1, 2]}, "field", 5) is None


class TestParseCoordinates:
    def test_valid_numbers(self):
        lat, lon = handler._parse_coordinates({"latitude": 40.71, "longitude": -74.01})
        assert lat == 40.71
        assert lon == -74.01

    def test_valid_numeric_strings(self):
        lat, lon = handler._parse_coordinates({"latitude": "40.71", "longitude": "-74.01"})
        assert lat == 40.71
        assert lon == -74.01

    def test_missing_latitude_raises(self):
        with pytest.raises(handler.InputError):
            handler._parse_coordinates({"longitude": -74.01})

    def test_missing_longitude_raises(self):
        with pytest.raises(handler.InputError):
            handler._parse_coordinates({"latitude": 40.71})

    def test_non_numeric_value_raises(self):
        with pytest.raises(handler.InputError):
            handler._parse_coordinates({"latitude": "abc", "longitude": -74.01})

    def test_non_dict_event_raises(self):
        with pytest.raises(handler.InputError):
            handler._parse_coordinates(["not", "a", "dict"])


class TestToolNameFromContext:
    def test_no_client_context_returns_none(self):
        assert handler._tool_name_from_context(SimpleNamespace()) is None

    def test_client_context_with_custom_returns_tool_name(self):
        context = SimpleNamespace(
            client_context=SimpleNamespace(
                custom={"bedrockAgentCoreToolName": "weather-tools___get_forecast"}
            )
        )
        assert handler._tool_name_from_context(context) == "weather-tools___get_forecast"

    def test_client_context_without_custom_returns_none(self):
        context = SimpleNamespace(client_context=SimpleNamespace(custom=None))
        assert handler._tool_name_from_context(context) is None


class TestLambdaHandler:
    @patch("src.handler.fetch_forecast")
    def test_success_returns_raw_forecast_dict(self, mock_fetch):
        mock_fetch.return_value = {"forecast_days": 10, "days": []}

        result = handler.lambda_handler({"latitude": 40.71, "longitude": -74.01}, SimpleNamespace())

        assert result == {"forecast_days": 10, "days": []}

    def test_missing_coordinates_returns_error_dict(self):
        result = handler.lambda_handler({}, SimpleNamespace())
        assert "error" in result

    @patch("src.handler.fetch_forecast")
    def test_upstream_failure_returns_error_dict(self, mock_fetch):
        mock_fetch.side_effect = handler.WeatherError("Open-Meteo request failed: boom")

        result = handler.lambda_handler({"latitude": 40.71, "longitude": -74.01}, SimpleNamespace())

        assert "error" in result

    @patch("src.handler.fetch_forecast")
    def test_gateway_invocation_logs_tool_name(self, mock_fetch, caplog):
        mock_fetch.return_value = {"forecast_days": 10, "days": []}
        context = SimpleNamespace(
            client_context=SimpleNamespace(
                custom={"bedrockAgentCoreToolName": "weather-tools___get_forecast"}
            )
        )

        with caplog.at_level("INFO"):
            result = handler.lambda_handler({"latitude": 40.71, "longitude": -74.01}, context)

        assert result == {"forecast_days": 10, "days": []}
        assert "get_forecast" in caplog.text
