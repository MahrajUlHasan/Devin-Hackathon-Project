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

# Agents are assigned a *tier*, not a model id. The reasoning behind each assignment
# is about the job, not the vendor -- "ranking six symbols is nearly mechanical" is
# true whoever serves the tokens -- so the tier survives a provider swap and only the
# lookup table below changes.
#
#   FAST  high volume, near-mechanical: ranking, labelling, scoring headlines
#   MID   per-symbol reasoning in the hot path, where nuance pays
#   DEEP  the one call that becomes an order, and the on-demand showpiece
FAST, MID, DEEP = "fast", "mid", "deep"

AGENT_TIERS: dict[str, str] = {
    "scout": FAST,  # A1 ranking is arithmetic; model writes justification
    "headline_scorer": FAST,  # A3 first stage, high volume
    "news": MID,  # A3 synthesis into a risk rating
    "technical": MID,  # A2 per-symbol reasoning, hot path
    "regime": FAST,  # A4 cached 30m, cheap labelling
    "risk_analyst": MID,  # A5 advisory judgment
    "pm": DEEP,  # A6 the one call that becomes an order
    "reflection": MID,  # A7 off the hot path
    "deep": DEEP,  # A8 human-triggered showpiece, effort=high
    "bull": MID,  # A9 advocacy needs reasoning, not just labelling
    "bear": MID,  # A10 ditto -- a weak bear case is worthless
}

PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        FAST: os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-4-5"),
        MID: os.getenv("CLAUDE_MODEL_MID", "claude-sonnet-5"),
        DEEP: os.getenv("CLAUDE_MODEL_DEEP", "claude-opus-5"),
    },
    "gemini": {
        # Rolling aliases rather than pinned versions: a hackathon demo that breaks
        # because a dated snapshot was retired is a bad trade for reproducibility we
        # are not otherwise relying on. Override per tier via env if you need a pin.
        FAST: os.getenv("GEMINI_MODEL_FAST", "gemini-flash-lite-latest"),
        MID: os.getenv("GEMINI_MODEL_MID", "gemini-flash-latest"),
        DEEP: os.getenv("GEMINI_MODEL_DEEP", "gemini-pro-latest"),
    },
}

DEFAULT_PROVIDER = "anthropic"


def models_for(provider: str) -> dict[str, str]:
    """Agent name -> model id for one provider."""
    tiers = PROVIDER_MODELS[provider]
    return {agent: tiers[tier] for agent, tier in AGENT_TIERS.items()}


# The Claude allocation, kept as a module constant so existing imports and the docs
# that quote it still read correctly. Live lookups go through the runtime's provider.
MODELS: dict[str, str] = models_for(DEFAULT_PROVIDER)

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
    "bull": 12.0,
    "bear": 12.0,
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

# --- Daily drawdown kill-switch ---
# Measured from the session's peak equity, not its opening equity: a desk that is up 8%
# and gives back 5% has lost control of the day just as much as one that started flat.
MAX_DAILY_DRAWDOWN_PCT = 0.05
# A halt stops NEW ENTRIES only. Open positions keep their stops and take-profits,
# because force-liquidating into whatever caused the drawdown is usually how a bad day
# becomes a catastrophic one. Set true to flatten instead.
HALT_FLATTENS_POSITIONS = False

# --- Per-symbol cooldown ---
# Bars to wait after closing a position before re-entering the same symbol. Stops the
# desk from immediately re-buying what just stopped it out.
COOLDOWN_BARS = 6  # 30 minutes at 5-minute bars
# A take-profit is not evidence the thesis was wrong, so it cools down for less time.
COOLDOWN_BARS_AFTER_TAKE = 2

# --- Bull/Bear debate (A9/A10) ---
# Off by default: it roughly doubles tokens and latency per candidate. Worth it when
# you want to watch the argument; not worth it in the hot path by default.
ENABLE_DEBATE = False
DEBATE_ROUNDS = 1  # 1 = opening statements only, 2 = adds a rebuttal round

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
    gemini_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower())
    db_path: str = field(default_factory=lambda: os.getenv("BUDAPILOT_DB", "budapilot.db"))
    host: str = field(default_factory=lambda: os.getenv("BUDAPILOT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("BUDAPILOT_PORT", "8000")))

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_key)

    def key_for(self, provider: str) -> str:
        return {"anthropic": self.anthropic_key, "gemini": self.gemini_key}.get(provider, "")

    def resolve_provider(self, requested: str | None = None) -> tuple[str, str]:
        """Pick a provider and its key. Returns ``(provider, key)``; key is "" offline.

        An explicit request is honoured even when its key is missing, rather than
        quietly falling through to the other vendor. Asking for Claude and silently
        getting Gemini -- or vice versa -- is the kind of surprise that has you
        debugging prompt quality when the real problem is that you are talking to a
        different model than you think.
        """
        want = (requested or self.llm_provider or "auto").lower()
        if want != "auto":
            if want not in PROVIDER_MODELS:
                raise ValueError(
                    f"unknown provider {want!r}; expected one of {sorted(PROVIDER_MODELS)}"
                )
            return want, self.key_for(want)
        for provider in (DEFAULT_PROVIDER, *PROVIDER_MODELS):
            if self.key_for(provider):
                return provider, self.key_for(provider)
        return DEFAULT_PROVIDER, ""

    def assert_paper_only(self) -> None:
        """The live-trading path is unimplemented by construction. (NFR4)"""
        if not self.alpaca_paper:
            raise NotImplementedError(
                "Live trading is deliberately not implemented. Set ALPACA_PAPER=true. "
                "This system has never been validated with real money and must not be."
            )


settings = Settings()
