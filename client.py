from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import httpx


@dataclass
class Ticker:
    pair: str
    ask: Decimal
    bid: Decimal
    last_trade: Decimal
    timestamp: int
    status: str
    rolling_24_hour_volume: str = ""


@dataclass
class Candle:
    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class Trade:
    timestamp: int
    price: Decimal
    volume: Decimal
    is_buy: bool


class LunoClient:
    def __init__(
        self,
        base_url: str = "https://api.luno.com",
        api_key_id: str = "",
        api_key_secret: str = "",
    ) -> None:
        self.base = base_url
        auth = httpx.BasicAuth(api_key_id, api_key_secret) if api_key_id else None
        self._client = httpx.Client(base_url=base_url, timeout=15, auth=auth)

    def get_ticker(self, pair: str) -> Ticker:
        resp = self._client.get("/api/1/ticker", params={"pair": pair})
        resp.raise_for_status()
        d = resp.json()
        return Ticker(
            pair=d["pair"],
            ask=Decimal(d["ask"]),
            bid=Decimal(d["bid"]),
            last_trade=Decimal(d["last_trade"]),
            timestamp=d["timestamp"],
            status=d["status"],
        )

    def get_candles(self, pair: str, duration: int, since: int) -> list[Candle]:
        resp = self._client.get(
            "/api/exchange/1/candles",
            params={"pair": pair, "duration": duration, "since": since},
        )
        resp.raise_for_status()
        data = resp.json().get("candles", [])
        return [
            Candle(
                timestamp=c["timestamp"],
                open=Decimal(c["open"]),
                high=Decimal(c["high"]),
                low=Decimal(c["low"]),
                close=Decimal(c["close"]),
                volume=Decimal(c["volume"]),
            )
            for c in data
        ]

    def get_recent_trades(self, pair: str, since: int | None = None) -> list[Trade]:
        params = {"pair": pair}
        if since is not None:
            params["since"] = since
        resp = self._client.get("/api/1/trades", params=params)
        resp.raise_for_status()
        data = resp.json().get("trades", [])
        return [
            Trade(
                timestamp=t["timestamp"],
                price=Decimal(t["price"]),
                volume=Decimal(t["volume"]),
                is_buy=t["is_buy"],
            )
            for t in data
        ]

    def close(self) -> None:
        self._client.close()
