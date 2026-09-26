from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from client import Candle

logger = logging.getLogger(__name__)


@dataclass
class KronosForecast:
    timestamp: int
    horizon: int
    predicted_close: Decimal
    current_close: Decimal
    expected_change_pct: Decimal


class KronosClient:
    """Thin client for a Kronos K-line forecasting endpoint (HF Space or self-hosted).

    Posts OHLCV history, gets back a forecast of future candles. Returns None
    when unconfigured or unreachable so callers degrade gracefully.
    """

    def __init__(self, base_url: str = "", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def forecast(self, candles: list[Candle], horizon: int = 5) -> KronosForecast | None:
        if not self.enabled:
            return None
        lookback = candles[-500:]  # Kronos-small/base context is 512
        if len(lookback) < 50:
            return None
        payload = {
            "timestamp": [c.timestamp for c in lookback],
            "open": [float(c.open) for c in lookback],
            "high": [float(c.high) for c in lookback],
            "low": [float(c.low) for c in lookback],
            "close": [float(c.close) for c in lookback],
            "volume": [float(c.volume) for c in lookback],
            "horizon": horizon,
        }
        try:
            resp = self._client.post(f"{self.base_url}/predict", json=payload)
            resp.raise_for_status()
            closes = _extract_predicted_closes(resp.json())
        except Exception as e:
            # degrade gracefully, but never silently — an outage that
            # vanishes from the score with no log line is undebuggable
            logger.warning("Kronos forecast failed (%s): %s", self.base_url, e)
            return None
        if not closes:
            logger.warning(
                "Kronos responded 200 but no 'close' fields parsed from /predict: %s",
                self.base_url,
            )
            return None

        current = Decimal(str(lookback[-1].close))
        predicted = Decimal(str(closes[-1]))
        change = (predicted - current) / current * 100 if current else Decimal("0")
        return KronosForecast(
            timestamp=int(time.time()),
            horizon=len(closes),
            predicted_close=predicted,
            current_close=current,
            expected_change_pct=change,
        )

    def close(self) -> None:
        self._client.close()


def _extract_predicted_closes(data: Any) -> list[float]:
    """Tolerate forecast shapes: [{"close": ...}], {"forecast": [{"close": ...}]},
    {"close": [...]}, or {"forecast": {"close": [...]}}. Anything else -> []."""
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        blob = data.get("forecast", data)
        if isinstance(blob, list):
            rows = [r for r in blob if isinstance(r, dict)]
        elif isinstance(blob, dict):
            raw = blob.get("close", blob.get("closes"))
            if isinstance(raw, list):
                return [float(x) for x in raw if isinstance(x, (int, float))]
            return []
        else:
            return []
    else:
        return []

    closes: list[float] = []
    for row in rows:
        c = row.get("close")
        if isinstance(c, (int, float)):
            closes.append(float(c))
    return closes


def forecast_direction(forecast: KronosForecast | None, threshold_pct: float = 0.5) -> int:
    """Vote: +1 bullish, -1 bearish, 0 no edge. Kronos is one factor, not truth."""
    if forecast is None:
        return 0
    threshold = Decimal(str(threshold_pct))
    if forecast.expected_change_pct >= threshold:
        return 1
    if forecast.expected_change_pct <= -threshold:
        return -1
    return 0


def demo() -> None:
    assert _extract_predicted_closes({"forecast": {"close": [1, 2, 3]}}) == [1.0, 2.0, 3.0]
    assert _extract_predicted_closes({"close": [5, 6]}) == [5.0, 6.0]
    assert _extract_predicted_closes([{"close": 7}, {"close": 8}]) == [7.0, 8.0]
    assert _extract_predicted_closes("junk") == []
    assert _extract_predicted_closes({}) == []

    up = KronosForecast(0, 3, Decimal("110"), Decimal("100"), Decimal("10"))
    down = KronosForecast(0, 3, Decimal("99"), Decimal("100"), Decimal("-1"))
    flat = KronosForecast(0, 3, Decimal("100.3"), Decimal("100"), Decimal("0.3"))
    assert forecast_direction(up) == 1
    assert forecast_direction(down) == -1
    assert forecast_direction(flat) == 0
    assert forecast_direction(None) == 0
    print("kronos demo OK")


if __name__ == "__main__":
    demo()
