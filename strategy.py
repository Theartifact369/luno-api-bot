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
    hard_exit: bool = False  # capital-preservation exits bypass the LLM
    family: str = ""         # stable entry-signal id for the decision log


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

    def _reason_feedback(self, family: str) -> tuple[float, str]:
        """Decision-log memory: how has this ENTRY SIGNAL family performed?

        Every buy stores its signal family on the position and, at exit, on the
        completed trade. Before the next buy, that record is consulted: a family
        that has never won (>= 2 samples) is vetoed, one that has won less than
        half its trades gets its confidence halved. This is the 'why did we
        think this' log feeding back — not momentary indicators, but the bot's
        own history of reasons.
        (ponytail: exact-family matching only. Regime repeats one stable family;
        rules mode aggregates every rules-buy under one family. Good enough at
        these trade rates, but per-factor families would sharpen it.)"""
        same = [
            Decimal(str(t["pnl_pct"])) for t in self.trader.completed_log.trades
            if t.get("entry_reason") == family and t.get("pnl_pct") is not None
        ]
        if len(same) < 2:
            return 1.0, ""
        if all(p <= 0 for p in same):
            return 0.0, f"family '{family}' lost {len(same)}/{len(same)}"
        if sum(1 for p in same if p > 0) / len(same) < 0.5:
            return 0.5, f"family '{family}' won under half ({len(same)} trades)"
        return 1.0, ""

    def _order_book_check(self, price: Decimal) -> str:
        """Order book imbalance: bid volume vs ask volume near price."""
        try:
            from client import LunoClient
            client = LunoClient(self.config.luno_api_base, self.config.luno_api_key_id, self.config.luno_api_secret)
            ob = client.get_order_book(self.trader.pair)
            client.close()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if not bids or not asks:
                return ""
            bid_vol = sum(Decimal(b["volume"]) for b in bids[:10])
            ask_vol = sum(Decimal(a["volume"]) for a in asks[:10])
            if bid_vol == 0 or ask_vol == 0:
                return ""
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
            if imbalance > Decimal("0.2"):
                return f"OB: bid-heavy ({float(imbalance)*100:.0f}%)"
            if imbalance < Decimal("-0.2"):
                return f"OB: ask-heavy ({float(imbalance)*100:.0f}%)"
            return ""
        except Exception:
            return ""

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

    def _regime_signal(
        self, ctx: MarketContext, daily_candles: list[Candle] | None = None
    ) -> Signal:
        """Daily SMA50/200 regime: all-in on golden cross, cash on death cross.

        VERDICT — backtest.py --sma-regime --daily, 2021-07..2026-08, 5.1yr:
        this LOSES to buy-and-hold. +95.3% vs +130.3%. Its only edge is
        drawdown (47% vs 73%). Sharpe is ~0.7 on 5 trades, i.e. no measurable
        risk-adjusted edge (it printed 12.18 until the annualiser was fixed —
        per-trade returns were being scaled by sqrt(365)). The backtest prints
        "VERDICT: LOSES to buy-and-hold" directly under these numbers.

        Shape: all-in/all-out, no trims, no SL/TP, ~1 trade per 160 days.
        Aimed at a pair with deep history; Luno silently truncates candle
        requests past 1000 bars from the OLD end, so keep the daily window
        under that or the SMA goes stale with no warning.

        Do not fund this until it beats holding on its own PAPER record over
        months. Do not quote PF 5.18 / maxDD 47% without the verdict above.
        """
        closes = [d.close for d in daily_candles] if daily_candles else []
        price = ctx.ticker.last_trade
        if len(closes) < 200:
            return Signal("HOLD", 0.0, f"Regime: warm-up ({len(closes)}/200 daily bars)", price)
        fast = sum(closes[-50:]) / Decimal("50")
        slow = sum(closes[-200:]) / Decimal("200")  # >= 200 bars, guard above
        bullish = fast > slow
        if self.trader.state.open_position is None:
            if bullish:
                mult, fb = self._reason_feedback("golden_cross")
                if mult == 0:
                    return Signal("HOLD", 0.0, f"Regime: golden cross skipped — {fb}", price)
                ob_reason = self._order_book_check(price)
                reason = "Regime: SMA50>200 golden — go all-in"
                if ob_reason:
                    reason += f" [{ob_reason}]"
                return Signal("BUY", 0.7 * mult, reason + (f" [{fb}]" if fb else ""),
                              price, hard_exit=True, family="golden_cross")
            return Signal("HOLD", 0.0, "Regime: SMA50<200 — staying in cash", price)
        if not bullish:
            return Signal("SELL", 0.8, "Regime: SMA50<200 death cross — exit all", price, hard_exit=True)
        return Signal("HOLD", 0.0, "Regime: bullish — holding full position", price)

    def _rules_signal(
        self,
        ctx: MarketContext,
        forecast: KronosForecast | None = None,
        kronos_reason: str | None = None,
    ) -> Signal | None:
        price = ctx.ticker.last_trade
        gp = self.governance

        if self.trader.state.open_position is not None:
            pos = self.trader.state.open_position
            entry = Decimal(pos["entry_price"])
            # ponytail: peak tracked at scan-tick granularity (60s) — fine for 1h candles
            peak = Decimal(str(pos.get("peak_price", entry)))
            if price > peak:
                peak = price
                pos["peak_price"] = str(price)
            change = (price - entry) / entry

            if change >= Decimal(str(gp.take_profit_pct)):
                return Signal("SELL", 0.85, f"Trim take profit: +{change * 100:.2f}%", price, hard_exit=True)
            if change <= -Decimal(str(gp.stop_loss_pct)):
                return Signal("SELL", 0.95, f"Trim stop loss: {change * 100:.2f}%", price, hard_exit=True)
            if peak > entry:
                trail = (peak - price) / peak
                if trail >= Decimal(str(gp.trailing_stop_pct)):
                    return Signal("SELL", 0.9, f"Trailing stop: {trail * 100:.2f}% off peak", price, hard_exit=True)

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
            mult, fb = self._reason_feedback("rules_buy")
            if mult == 0:
                return Signal("HOLD", 0.0, f"Buy skipped — {fb}", price)
            return Signal("BUY", min(score / 0.9, 0.9) * mult,
                          "; ".join(reasons) + (f" [{fb}]" if fb else ""),
                          price, family="rules_buy")

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

    async def analyze(self, ctx: MarketContext, daily_candles: list[Candle] | None = None) -> Signal:
        if self.config.strategy == "regime":
            return self._regime_signal(ctx, daily_candles)
        forecast, kronos_reason = await self._kronos_forecast(ctx)
        rules_signal = self._rules_signal(ctx, forecast, kronos_reason)

        # ponytail: hard exits (SL/TP/trail) never consult the LLM
        if rules_signal is not None and rules_signal.hard_exit:
            return rules_signal

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
