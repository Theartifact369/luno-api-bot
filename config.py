from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _load_dotenv(path: str = ".env") -> None:
    # ponytail: stdlib dotenv, ~10 lines. Add python-dotenv when we need
    # interpolation, overrides, or export handling.
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Config:
    luno_api_base: str = "https://api.luno.com"
    # ponytail: creds come from .env / env vars only; never hardcode here
    luno_api_key_id: str = ""
    luno_api_secret: str = ""
    trading_pair: str = "XBTZAR"
    candle_duration: int = 3600

    initial_balance_base: float = 0.0
    initial_balance_counter: float = 100000.0

    scan_interval_seconds: int = 60
    max_daily_trades: int = 5
    max_position_size_pct: float = 0.3
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.03

    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9

    llm_provider: Literal["openai", "anthropic", "none"] = "none"
    llm_api_key: str = ""
    llm_model: str = "gpt-4"

    kronos_url: str = ""
    kronos_horizon: int = 5
    kronos_refresh_seconds: int = 900
    kronos_threshold_pct: float = 0.5

    state_file: str = "trading_state.json"

    governance_state_file: str = "governance_state.json"
    completed_trades_file: str = ""

    risk_max_daily_loss_pct: float = 3.0
    risk_max_drawdown_pct: float = 15.0
    risk_max_consecutive_losses: int = 5
    risk_cooldown_hours: float = 4.0
    risk_volatility_sizing: bool = True

    governor_reflection_interval_trades: int = 10
    governor_reflection_interval_seconds: int = 3600

    log_level: str = "INFO"
    log_dir: str = "logs"
    data_dir: str = ""
    headless: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv()
        c = cls()
        c.log_level = os.getenv("LOG_LEVEL", c.log_level)
        c.log_dir = os.getenv("LOG_DIR", c.log_dir)
        c.data_dir = os.getenv("DATA_DIR", c.data_dir)
        c.headless = os.getenv("HEADLESS", "").lower() in ("1", "true", "yes")
        c.luno_api_base = os.getenv("LUNO_API_BASE", c.luno_api_base)
        c.luno_api_key_id = os.getenv("LUNO_API_KEY_ID", c.luno_api_key_id)
        c.luno_api_secret = os.getenv("LUNO_API_SECRET", c.luno_api_secret)
        c.trading_pair = os.getenv("LUNO_PAIR", c.trading_pair)
        c.initial_balance_counter = float(os.getenv("INITIAL_BALANCE", str(c.initial_balance_counter)))
        c.scan_interval_seconds = int(os.getenv("SCAN_INTERVAL", str(c.scan_interval_seconds)))
        c.max_daily_trades = int(os.getenv("MAX_DAILY_TRADES", str(c.max_daily_trades)))
        c.llm_provider = os.getenv("LLM_PROVIDER", c.llm_provider)
        c.llm_api_key = os.getenv("LLM_API_KEY", c.llm_api_key)
        c.llm_model = os.getenv("LLM_MODEL", c.llm_model)
        c.kronos_url = os.getenv("KRONOS_URL", c.kronos_url)
        c.kronos_horizon = int(os.getenv("KRONOS_HORIZON", str(c.kronos_horizon)))
        c.kronos_refresh_seconds = int(os.getenv("KRONOS_REFRESH_SECONDS", str(c.kronos_refresh_seconds)))
        c.kronos_threshold_pct = float(os.getenv("KRONOS_THRESHOLD_PCT", str(c.kronos_threshold_pct)))
        c.state_file = os.getenv("STATE_FILE", c.state_file)
        c.governance_state_file = os.getenv("GOVERNANCE_FILE", c.governance_state_file)
        c.completed_trades_file = os.getenv("COMPLETED_TRADES_FILE", c.completed_trades_file)
        c.risk_max_daily_loss_pct = float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", str(c.risk_max_daily_loss_pct)))
        c.risk_max_drawdown_pct = float(os.getenv("RISK_MAX_DRAWDOWN_PCT", str(c.risk_max_drawdown_pct)))
        c.risk_max_consecutive_losses = int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", str(c.risk_max_consecutive_losses)))
        c.risk_cooldown_hours = float(os.getenv("RISK_COOLDOWN_HOURS", str(c.risk_cooldown_hours)))
        c.governor_reflection_interval_trades = int(os.getenv("GOVERNOR_REFLECTION_TRADES", str(c.governor_reflection_interval_trades)))
        c.governor_reflection_interval_seconds = int(os.getenv("GOVERNOR_REFLECTION_SECONDS", str(c.governor_reflection_interval_seconds)))
        return c
