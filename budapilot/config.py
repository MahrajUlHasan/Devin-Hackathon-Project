"""Runtime configuration. Every tunable in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------

WATCHLIST: list[str] = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "LTC/USD",
    "AVAX/USD",
    "DOGE/USD",
]

BAR_MINUTES = 5
MAX_CANDIDATES = 3

# --------------------------------------------------------------------------------------
# Models -- one line per agent so the allocation is auditable at a glance
# --------------------------------------------------------------------------------------

MODEL_HAIKU = "claude-haiku-4-5"
MODEL_SONNET = "claude-sonnet-5"
MODEL_OPUS = "claude-opus-5"

MODELS: dict[str, str] = {
    "scout": MODEL_HAIKU,  # A1 ranking is arithmetic; model writes justification
    "headline_scorer": MODEL_HAIKU,  # A3 first stage, high volume
    "news": MODEL_SONNET,  # A3 synthesis into a risk rating
    "technical": MODEL_SONNET,  # A2 per-symbol reasoning, hot path
    "regime": MODEL_HAIKU,  # A4 cached 30m, cheap labelling
    "risk_analyst": MODEL_SONNET,  # A5 advisory judgment
    "pm": MODEL_OPUS,  # A6 the one call that becomes an order
    "reflection": MODEL_SONNET,  # A7 off the hot path
    "deep": MODEL_OPUS,  # A8 human-triggered showpiece, effort=high
}

# Per-agent timeouts (seconds). On expiry the agent degrades rather than raising.
TIMEOUTS: dict[str, float] = {
    "scout": 8.0,
    "headline_scorer": 8.0,
    "news": 10.0,
    "technical": 10.0,
    "regime": 8.0,
    "risk_analyst": 10.0,
    "pm": 15.0,  # then DeterministicArbiter, not a stub
    "reflection": 20.0,
    "deep": 120.0,
}

# --------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------

NEWS_TTL_S = 900  # 15 min
REGIME_TTL_S = 1800  # 30 min
LESSON_INJECT_K = 5

# --------------------------------------------------------------------------------------
# Risk -- mirrored in risk/engine.py as the single source of truth for limits
# --------------------------------------------------------------------------------------

MAX_POSITION_PCT = 0.10
MAX_TOTAL_EXPOSURE_PCT = 0.50
MAX_OPEN_POSITIONS = 3
RISK_PER_TRADE = 0.01
K_STOP = 2.0
K_TAKE = 3.0
MIN_CONVICTION = 0.60

# Reject positions too small to matter. Without this the engine happily spends the last
# few dollars of cap headroom on a $6 order that pays fees and moves no needle.
MIN_ORDER_NOTIONAL_PCT = 0.005  # 0.5% of equity

# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


@dataclass
class Settings:
    alpaca_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))
    alpaca_paper: bool = field(
        default_factory=lambda: os.getenv("ALPACA_PAPER", "true").lower() != "false"
    )
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    db_path: str = field(default_factory=lambda: os.getenv("BUDAPILOT_DB", "budapilot.db"))
    host: str = field(default_factory=lambda: os.getenv("BUDAPILOT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("BUDAPILOT_PORT", "8000")))

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_key)

    def assert_paper_only(self) -> None:
        """The live-trading path is unimplemented by construction. (NFR4)"""
        if not self.alpaca_paper:
            raise NotImplementedError(
                "Live trading is deliberately not implemented. Set ALPACA_PAPER=true. "
                "This system has never been validated with real money and must not be."
            )


settings = Settings()
