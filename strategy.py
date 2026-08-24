from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from config import Config
from client import Candle, Ticker
from governance import GovernanceParams
from indicators import compute_rsi, compute_macd, compute_sma
from kronos import KronosClient, KronosForecast, forecast_direction
from paper_trader import PaperTrader


@dataclass
class Signal:
    action: str
    confidence: float
    reason: str
    price: Decimal | None = None


@dataclass
class MarketContext:
    ticker: Ticker
    candles: list[Candle]
    rsi: Decimal | None
    macd_histogram: Decimal | None
    sma_short: Decimal | None
    sma_long: Decimal | None
    price_above_sma: bool | None
    atr: Decimal | None = None


def compute_atr(candles: list[Candle], period: int = 14) -> Decimal | None:
    if len(candles) < period + 1:
        return None
    tr_sum = Decimal("0")
    for i in range(-period, 0):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_sum += tr
    return tr_sum / Decimal(str(period))


class Strategy:
    def __init__(self, config: Config, trader: PaperTrader) -> None:
        self.config = config
        self.trader = trader
        self.governance: GovernanceParams = GovernanceParams()
        self.kronos = KronosClient(config.kronos_url)
        self._kronos_cache: tuple[float, KronosForecast | None] | None = None

    def update_governance(self, params: GovernanceParams) -> None:
        self.governance = params

    def build_context(self, ticker: Ticker, candles: list[Candle]) -> MarketContext:
        closes = [c.close for c in candles]
        rsi = compute_rsi(closes, self.config.rsi_period)
        _, _, histogram = compute_macd(
            closes, self.config.ema_fast, self.config.ema_slow, self.config.macd_signal
        )
        sma_short_list = compute_sma(closes, self.config.ema_fast)
        sma_long_list = compute_sma(closes, self.config.ema_slow)
        atr = compute_atr(candles)

        current_price = ticker.last_trade
        sma_short = sma_short_list[-1] if sma_short_list else None
        sma_long = sma_long_list[-1] if sma_long_list else None
        price_above_sma = None
        if sma_short is not None:
            price_above_sma = current_price > sma_short

        macd_hist = histogram[-1] if histogram else None
        return MarketContext(
            ticker=ticker,
            candles=candles,
            rsi=rsi,
            macd_histogram=macd_hist,
            sma_short=sma_short,
            sma_long=sma_long,
            price_above_sma=price_above_sma,
            atr=atr,
        )

    def _rules_signal(
        self,
        ctx: MarketContext,
        forecast: KronosForecast | None = None,
        kronos_reason: str | None = None,
    ) -> Signal | None:
        price = ctx.ticker.last_trade
        gp = self.governance

        if self.trader.state.open_position is not None:
            entry = Decimal(self.trader.state.open_position["entry_price"])
            change = (price - entry) / entry

            if change >= Decimal(str(gp.take_profit_pct)):
                return Signal("SELL", 0.85, f"Trim take profit: +{change * 100:.2f}%", price)
            if change <= -Decimal(str(gp.stop_loss_pct)):
                return Signal("SELL", 0.95, f"Trim stop loss: {change * 100:.2f}%", price)

        score = 0.0
        reasons = []

        if forecast is not None:
            direction = forecast_direction(forecast, self.config.kronos_threshold_pct)
            if direction != 0:
                mag = min(float(abs(forecast.expected_change_pct)) / 2, 1.0)
                score += direction * 0.35 * mag
                reasons.append(kronos_reason or f"Kronos: {forecast.expected_change_pct:+.2f}%")

        if ctx.rsi is not None:
            if ctx.rsi < Decimal(str(gp.rsi_oversold)):
                score += 0.3
                reasons.append(f"RSI oversold ({ctx.rsi:.1f})")
            elif ctx.rsi > Decimal(str(gp.rsi_overbought)):
                score -= 0.3
                reasons.append(f"RSI overbought ({ctx.rsi:.1f})")

        if ctx.macd_histogram is not None:
            if ctx.macd_histogram > 0:
                score += 0.2
                reasons.append("MACD histogram positive")
            else:
                score -= 0.2
                reasons.append("MACD histogram negative")

        if ctx.price_above_sma is not None:
            if ctx.price_above_sma:
                score += 0.15
                reasons.append("Price above SMA")
            else:
                score -= 0.15
                reasons.append("Price below SMA")

        threshold = max(0.3, 0.4 - (gp.max_position_size_pct * 0.3))

        if score >= threshold:
            return Signal("BUY", min(score / 0.9, 0.9), "; ".join(reasons), price)

        return None

    async def _kronos_forecast(self, ctx: MarketContext) -> tuple[KronosForecast | None, str | None]:
        if not self.kronos.enabled:
            return None, None
        if self._kronos_cache is None or time.time() - self._kronos_cache[0] > self.config.kronos_refresh_seconds:
            forecast = await asyncio.to_thread(self.kronos.forecast, ctx.candles, self.config.kronos_horizon)
            self._kronos_cache = (time.time(), forecast)
        forecast = self._kronos_cache[1]
        if forecast is None:
            return None, "Kronos: forecast unavailable"
        return forecast, f"Kronos: {forecast.expected_change_pct:+.2f}% over {forecast.horizon} bars"

    async def analyze(self, ctx: MarketContext) -> Signal:
        forecast, kronos_reason = await self._kronos_forecast(ctx)
        rules_signal = self._rules_signal(ctx, forecast, kronos_reason)

        if self.config.llm_provider != "none" and self.config.llm_api_key:
            return await self._llm_decision(ctx, rules_signal)

        if rules_signal is not None:
            return rules_signal
        return Signal("HOLD", 0.0, "No clear signal from technical analysis")

    async def _llm_decision(
        self, ctx: MarketContext, rules_signal: Signal | None
    ) -> Signal:
        provider = self.config.llm_provider
        key = self.config.llm_api_key
        model = self.config.llm_model

        prompt = self._build_llm_prompt(ctx, rules_signal)

        if provider == "openai":
            return await self._call_openai(prompt, model, key, ctx)
        elif provider == "anthropic":
            return await self._call_anthropic(prompt, model, key, ctx)
        return rules_signal or Signal("HOLD", 0.0, "LLM not configured properly")

    def _build_llm_prompt(self, ctx: MarketContext, rules_signal: Signal | None) -> str:
        pos = self.trader.state.open_position
        pos_info = (
            f"Open position: entry={pos['entry_price']}, volume={pos['volume']}, "
            f"unrealized P&L={pos.get('unrealized_pnl_pct', 'N/A')}%"
            if pos
            else "No open position"
        )
        gp = self.governance

        return f"""You are a cryptocurrency trading advisor for Luno exchange ({self.config.trading_pair}).

Current market state:
- Ask: {ctx.ticker.ask}
- Bid: {ctx.ticker.bid}
- Last trade: {ctx.ticker.last_trade}
- 24h volume: {ctx.ticker.rolling_24_hour_volume}
- RSI (period {self.config.rsi_period}): {ctx.rsi}
- MACD histogram: {ctx.macd_histogram}
- SMA short: {ctx.sma_short}
- SMA long: {ctx.sma_long}
- ATR: {ctx.atr}
- Regime: {gp.regime}

Portfolio state:
- Balance (counter): {self.trader.state.balance_counter}
- Balance (base): {self.trader.state.balance_base}
- {pos_info}
- Trades today: {self.trader.state.trades_today} / {gp.max_daily_trades}

Active governance parameters:
- Stop loss: {gp.stop_loss_pct*100:.1f}%
- Take profit: {gp.take_profit_pct*100:.1f}%
- Max position size: {gp.max_position_size_pct*100:.1f}%
- RSI overbought: {gp.rsi_overbought}
- RSI oversold: {gp.rsi_oversold}
- Governor notes: {gp.governor_notes}

Rules-based signal: {rules_signal}

Respond with a JSON object only:
{{"action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reason": "brief justification"}}"""

    async def _call_openai(self, prompt: str, model: str, key: str, ctx: MarketContext) -> Signal:
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 200,
                    },
                    timeout=20,
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_llm_response(content, ctx)
            except Exception as e:
                return Signal("HOLD", 0.0, f"LLM error: {e}")

    async def _call_anthropic(self, prompt: str, model: str, key: str, ctx: MarketContext) -> Signal:
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "max_tokens": 200,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=20,
                )
                data = resp.json()
                content = data["content"][0]["text"]
                return self._parse_llm_response(content, ctx)
            except Exception as e:
                return Signal("HOLD", 0.0, f"LLM error: {e}")

    def _parse_llm_response(self, content: str, ctx: MarketContext) -> Signal:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
            action = result.get("action", "HOLD").upper()
            confidence = float(result.get("confidence", 0.5))
            reason = result.get("reason", "LLM decision")
            return Signal(action, confidence, reason, ctx.ticker.last_trade)
        except (json.JSONDecodeError, KeyError, ValueError):
            return Signal("HOLD", 0.0, f"Failed to parse LLM response: {content[:100]}")
