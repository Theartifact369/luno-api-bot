from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from performance import PerformanceMetrics


@dataclass
class GovernanceParams:
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.03
    sell_fraction_pct: float = 0.3
    max_position_size_pct: float = 0.3
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    max_daily_trades: int = 5
    regime: str = "unknown"
    governor_notes: str = ""


@dataclass
class GovernanceState:
    params: GovernanceParams = field(default_factory=GovernanceParams)
    last_reflection_time: float = 0.0
    reflection_count: int = 0
    reflection_history: list[dict[str, Any]] = field(default_factory=list)
    consecutive_losses_at_last_reflection: int = 0

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> GovernanceState:
        try:
            data = json.loads(Path(path).read_text())
            gp = GovernanceParams(**data.get("params", {}))
            return cls(
                params=gp,
                last_reflection_time=data.get("last_reflection_time", 0.0),
                reflection_count=data.get("reflection_count", 0),
                reflection_history=data.get("reflection_history", []),
                consecutive_losses_at_last_reflection=data.get("consecutive_losses_at_last_reflection", 0),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return cls()


class GovernorAgent:
    def __init__(
        self,
        llm_provider: str = "none",
        llm_api_key: str = "",
        llm_model: str = "gpt-4",
        reflection_interval_trades: int = 10,
        reflection_interval_seconds: int = 3600,
    ) -> None:
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.reflection_interval_trades = reflection_interval_trades
        self.reflection_interval_seconds = reflection_interval_seconds

    def should_reflect(
        self, state: GovernanceState, total_trades: int, now: float
    ) -> bool:
        if self.llm_provider == "none" or not self.llm_api_key:
            return False
        trades_since = total_trades - (state.consecutive_losses_at_last_reflection if state.reflection_count > 0 else 0)
        time_since = now - state.last_reflection_time
        return (
            trades_since >= self.reflection_interval_trades
            or time_since >= self.reflection_interval_seconds
        )

    async def reflect(
        self,
        state: GovernanceState,
        metrics: PerformanceMetrics,
        current_params: GovernanceParams,
        recent_trades_summary: str,
        market_summary: str,
    ) -> GovernanceParams:
        prompt = self._build_reflection_prompt(
            metrics, current_params, recent_trades_summary, market_summary
        )

        new_params = await self._call_llm(prompt)
        new_params.governor_notes = new_params.governor_notes or "No changes proposed"
        new_params = self._clamp_params(new_params)

        state.last_reflection_time = time.time()
        state.reflection_count += 1
        state.reflection_history.append({
            "time": datetime.now(tz=timezone.utc).isoformat(),
            "metrics": {
                "win_rate": metrics.win_rate,
                "total_pnl": str(metrics.total_pnl),
                "sharpe": metrics.sharpe_ratio,
                "drawdown": metrics.current_drawdown_pct,
            },
            "old_params": asdict(current_params),
            "new_params": asdict(new_params),
        })

        return new_params

    def _clamp_params(self, p: GovernanceParams) -> GovernanceParams:
        p.stop_loss_pct = max(0.01, min(0.20, p.stop_loss_pct))
        p.take_profit_pct = max(0.005, min(0.15, p.take_profit_pct))
        p.sell_fraction_pct = max(0.05, min(0.8, p.sell_fraction_pct))
        p.max_position_size_pct = max(0.05, min(0.9, p.max_position_size_pct))
        p.rsi_overbought = max(60, min(90, p.rsi_overbought))
        p.rsi_oversold = max(10, min(40, p.rsi_oversold))
        p.max_daily_trades = max(1, min(20, p.max_daily_trades))
        return p

    def _build_reflection_prompt(
        self,
        metrics: PerformanceMetrics,
        params: GovernanceParams,
        recent_trades_summary: str,
        market_summary: str,
    ) -> str:
        return f"""You are the governor of an autonomous Bitcoin trading agent on Luno (XBTZAR).
Your job is to review recent performance and market conditions, then propose strategy adjustments.

=== CURRENT PARAMETERS ===
- Stop loss: {params.stop_loss_pct*100:.1f}%
- Take profit: {params.take_profit_pct*100:.1f}%
- Sell fraction (portion of position trimmed per exit): {params.sell_fraction_pct*100:.1f}%
- Max position size: {params.max_position_size_pct*100:.1f}%
- RSI overbought: {params.rsi_overbought}
- RSI oversold: {params.rsi_oversold}
- Max daily trades: {params.max_daily_trades}
- Market regime: {params.regime}

=== PERFORMANCE METRICS ===
- Total trades: {metrics.total_trades}
- Win rate: {metrics.win_rate*100:.1f}%
- Total P&L: {metrics.total_pnl}
- Sharpe ratio: {metrics.sharpe_ratio:.2f}
- Max drawdown: {metrics.max_drawdown_pct:.1f}%
- Current drawdown: {metrics.current_drawdown_pct:.1f}%
- Profit factor: {metrics.profit_factor:.2f}
- Consecutive losses: {metrics.consecutive_losses}
- Avg return per trade: {metrics.avg_return_per_trade:.2f}%

=== RECENT TRADES ===
{recent_trades_summary}

=== MARKET CONDITIONS ===
{market_summary}

Respond with a JSON object only. Adjust parameters to improve performance.
Be conservative — small incremental changes are better than drastic ones.
Set "regime" to one of: "trending_up", "trending_down", "ranging", "volatile", "unknown".
Include a brief "governor_notes" explaining your reasoning.

{{
  "stop_loss_pct": 0.05,
  "take_profit_pct": 0.03,
  "sell_fraction_pct": 0.3,
  "max_position_size_pct": 0.3,
  "rsi_overbought": 70.0,
  "rsi_oversold": 30.0,
  "max_daily_trades": 5,
  "regime": "ranging",
  "governor_notes": "brief reasoning for changes"
}}"""

    async def _call_llm(self, prompt: str) -> GovernanceParams:
        import httpx

        if self.llm_provider == "openai":
            return await self._call_openai(prompt)
        elif self.llm_provider == "anthropic":
            return await self._call_anthropic(prompt)
        return GovernanceParams()

    async def _call_openai(self, prompt: str) -> GovernanceParams:
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.llm_api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.4,
                        "max_tokens": 500,
                    },
                    timeout=30,
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_response(content)
            except Exception as e:
                return GovernanceParams(governor_notes=f"LLM error: {e}")

    async def _call_anthropic(self, prompt: str) -> GovernanceParams:
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": self.llm_api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                    json={
                        "model": self.llm_model,
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                data = resp.json()
                content = data["content"][0]["text"]
                return self._parse_response(content)
            except Exception as e:
                return GovernanceParams(governor_notes=f"LLM error: {e}")

    def _parse_response(self, content: str) -> GovernanceParams:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
            return GovernanceParams(
                stop_loss_pct=float(result.get("stop_loss_pct", 0.05)),
                take_profit_pct=float(result.get("take_profit_pct", 0.03)),
                sell_fraction_pct=float(result.get("sell_fraction_pct", 0.3)),
                max_position_size_pct=float(result.get("max_position_size_pct", 0.3)),
                rsi_overbought=float(result.get("rsi_overbought", 70.0)),
                rsi_oversold=float(result.get("rsi_oversold", 30.0)),
                max_daily_trades=int(result.get("max_daily_trades", 5)),
                regime=str(result.get("regime", "unknown")),
                governor_notes=str(result.get("governor_notes", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return GovernanceParams(governor_notes=f"Parse error: {content[:100]}")
