"""Every Pydantic model and Protocol in the system.

This module is the contract boundary. Once frozen, no module may change it; a module
that believes the contract is wrong stops and reports rather than editing.

Two invariants are encoded here and enforced again downstream in the risk engine:

1. Alpaca crypto supports ``order_class=simple`` only -- no bracket, no OCO, no OTO.
   Stops and take-profits are therefore carried on ``ProtectedPosition`` and managed
   in-process by the StopManager, never attached to the entry order.
2. Alpaca crypto is long-only. ``Action.SELL`` always means reduce or close an existing
   long. It never opens a short.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

TOut = TypeVar("TOut", bound=BaseModel)


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"  # reduce/close only -- crypto is long-only
    HOLD = "HOLD"


class NewsRisk(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"  # hard reject in the risk engine

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[self.value]


class Regime(StrEnum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOL_EXPANSION = "VOL_EXPANSION"
    RISK_OFF = "RISK_OFF"


class AgentStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"  # real agent failed, stub output substituted
    STUB = "stub"  # stub agent by configuration (demo-safe)
    FALLBACK = "fallback"  # DeterministicArbiter stood in for the PM


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"  # the only protective type crypto supports


class OrderStatus(StrEnum):
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExitReason(StrEnum):
    STOP = "STOP"
    TAKE = "TAKE"
    AGENT = "AGENT"
    MANUAL = "MANUAL"


# --------------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------------


class Bar(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: float | None = None


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    ts: datetime
    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    symbols: list[str] = Field(default_factory=list)


class FeatureBundle(BaseModel):
    """Deterministic indicator output. The only market representation an agent ever sees.

    Agents never receive raw price series. Keeping the bundle small is what keeps the
    prompt cheap, the latency low and the reasoning auditable.
    """

    symbol: str
    ts: datetime
    close: float

    sma_20: float | None = None
    sma_50: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    ema_cross: Literal["GOLDEN", "DEATH", "NONE"] = "NONE"
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None  # atr/close, the scale-free version
    vol_zscore_20: float | None = None
    ret_1: float | None = None
    ret_5: float | None = None
    ret_20: float | None = None
    dist_from_high_20: float | None = None
    dist_from_low_20: float | None = None

    trend_score: float = 0.0  # composite, computed in pandas -- not by an LLM

    def is_tradeable(self) -> bool:
        """The risk engine refuses anything missing an ATR; check it early too."""
        return self.atr_14 is not None and self.atr_14 > 0 and self.close > 0


# --------------------------------------------------------------------------------------
# Agent outputs (A1-A8)
# --------------------------------------------------------------------------------------


class ScoutCandidate(BaseModel):
    symbol: str
    trend_score: float
    thesis: str = Field(description="One line, max 20 words, on why this symbol is interesting.")
    vetoed: bool = Field(
        default=False,
        description="True if chart structure looks broken in a way the score missed.",
    )
    veto_reason: str = ""


class ScoutOutput(BaseModel):
    """A1. Ranking is arithmetic; the model supplies justification and structural veto."""

    candidates: list[ScoutCandidate] = Field(default_factory=list)
    market_note: str = ""


class TechnicalOutput(BaseModel):
    """A2."""

    symbol: str
    direction: Action
    conviction: float = Field(ge=0.0, le=1.0)
    horizon_bars: int = Field(ge=1, le=288)
    support: float | None = None
    resistance: float | None = None
    invalidation: float | None = Field(
        default=None, description="Price at which this thesis is simply wrong."
    )
    rationale: str
    key_factors: list[str] = Field(default_factory=list, max_length=4)


class ScoredHeadline(BaseModel):
    headline: str
    score: float = Field(ge=-1.0, le=1.0)
    reason: str = ""


class NewsOutput(BaseModel):
    """A3."""

    symbol: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    news_risk: NewsRisk
    catalysts: list[str] = Field(default_factory=list, max_length=5)
    cited: list[ScoredHeadline] = Field(default_factory=list)
    rationale: str = ""
    headline_count: int = 0


class RegimeOutput(BaseModel):
    """A4."""

    regime: Regime
    btc_dominance_drift: float = 0.0
    realized_vol_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    correlation_cluster: str = ""
    rationale: str = ""


class RiskAnalystOutput(BaseModel):
    """A5. Advisory only -- this agent cannot approve anything."""

    symbol: str
    size_multiplier: float = Field(ge=0.0, le=1.0)
    suggested_stop_atr: float = Field(default=2.0, ge=0.5, le=5.0)
    concerns: list[str] = Field(default_factory=list, max_length=4)
    rationale: str = ""


class TradeProposal(BaseModel):
    """A6 output. The single thing the risk engine is asked to rule on."""

    symbol: str
    action: Action
    conviction: float = Field(ge=0.0, le=1.0)
    size_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_atr: float = Field(default=2.0, ge=0.5, le=5.0)
    rationale: str
    overrode: list[str] = Field(
        default_factory=list,
        description="Which analysts were overruled and why, one string each.",
    )


class Lesson(BaseModel):
    """A7. Written on position close, injected into A6's next prompt."""

    symbol: str
    ts: datetime = Field(default_factory=utcnow)
    outcome_pct: float
    exit_reason: ExitReason
    lesson: str
    tags: list[str] = Field(default_factory=list, max_length=4)


class DeepAnalysis(BaseModel):
    """A8. Human-triggered showpiece."""

    symbol: str
    thesis: str
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    verdict: Action
    conviction: float = Field(ge=0.0, le=1.0)


class DebateCase(BaseModel):
    """A9 / A10. One side of the argument for a single symbol."""

    symbol: str
    side: Literal["BULL", "BEAR"]
    claims: list[str] = Field(default_factory=list, max_length=4)
    strongest_point: str
    rebuttal: str = Field(
        default="", description="Response to the other side. Empty in round 1."
    )
    conceded: str = Field(
        default="",
        description="The strongest point against this side that the advocate accepts.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Debate(BaseModel):
    symbol: str
    bull: DebateCase
    bear: DebateCase
    rounds: int = 1


class Disagreement(BaseModel):
    """How far apart the desk was on one symbol. Drives the dashboard heatmap.

    Scored rather than eyeballed because 'the agents disagreed' is the single most
    interesting thing a multi-agent system produces, and it is invisible unless you
    measure it.
    """

    symbol: str
    technical_signed: float  # +conviction for BUY, -conviction for SELL, 0 for HOLD
    news_sentiment: float
    risk_multiplier: float
    pm_action: Action
    pm_conviction: float
    overrode_count: int = 0
    score: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------------------
# Agent envelope
# --------------------------------------------------------------------------------------


class AgentResult(BaseModel, Generic[TOut]):
    """Wraps every agent call with the telemetry the dashboard and journal need."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: str
    model: str
    output: TOut
    status: AgentStatus = AgentStatus.OK
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None
    symbol: str | None = None


class SymbolOpinions(BaseModel):
    """Everything the specialists said about one candidate, handed to the PM."""

    symbol: str
    features: FeatureBundle
    technical: TechnicalOutput
    news: NewsOutput
    risk: RiskAnalystOutput
    debate: Debate | None = None


class BarDecision(BaseModel):
    """The complete record of one turn of the loop."""

    bar_id: str
    ts: datetime
    regime: RegimeOutput
    scout: ScoutOutput
    opinions: list[SymbolOpinions] = Field(default_factory=list)
    proposal: TradeProposal | None = None
    results: list[AgentResult[Any]] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------------------


class PortfolioState(BaseModel):
    equity: float
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        return self.gross_exposure / self.equity if self.equity > 0 else 0.0


class Position(BaseModel):
    symbol: str
    qty: float
    avg_entry: float
    market_value: float
    unrealized_pl: float = 0.0
    current_price: float = 0.0


class AssetSpec(BaseModel):
    """From GET /v2/assets/{symbol}. Ignoring these is how you get 422s all day."""

    symbol: str
    min_order_size: float = 0.0
    min_trade_increment: float = 0.0
    price_increment: float = 0.01
    tradable: bool = True


class SessionState(BaseModel):
    """Mutable risk state that must survive a restart.

    The kill-switch in particular: if a crash reset the drawdown counter, a bad day
    could halt, restart, and cheerfully resume losing money. So this is persisted to
    the journal on every bar and reloaded at startup.

    Equity is deliberately NOT cached here. A stale copy is how you get a kill-switch
    reading the wrong number; callers pass live equity to ``risk.session.drawdown_from``.
    """

    trading_day: date
    day_open_equity: float
    day_peak_equity: float
    bar_index: int = 0
    halted: bool = False
    halt_reason: str = ""
    # symbol -> the bar_index at which this symbol becomes tradeable again
    cooldown_until: dict[str, int] = Field(default_factory=dict)

    def cooling_down(self, symbol: str) -> bool:
        return self.cooldown_until.get(symbol, -1) > self.bar_index

    def bars_remaining(self, symbol: str) -> int:
        return max(0, self.cooldown_until.get(symbol, -1) - self.bar_index)


class RiskDecision(BaseModel):
    approved: bool
    symbol: str
    action: Action
    qty: float = 0.0
    notional: float = 0.0
    entry_ref: float = 0.0
    stop_px: float | None = None
    take_px: float | None = None
    reason: str = ""
    checks: dict[str, bool] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: Literal["gtc", "ioc"] = "gtc"
    client_order_id: str | None = None


class Order(BaseModel):
    id: str
    client_order_id: str | None = None
    symbol: str
    side: OrderSide
    qty: float
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.NEW
    submitted_at: datetime = Field(default_factory=utcnow)
    stop_price: float | None = None
    limit_price: float | None = None


class ProtectedPosition(BaseModel):
    """A position plus the stop/take the broker will not hold for us.

    ``broker_stop_order_id`` is the resting stop_limit that survives process death.
    ``closing`` guards the cancel-then-close sequence against re-entry.
    """

    symbol: str
    qty: float
    entry: float
    stop_px: float
    take_px: float
    opened_at: datetime = Field(default_factory=utcnow)
    broker_stop_order_id: str | None = None
    closing: bool = False


# --------------------------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------------------------


@runtime_checkable
class MarketDataPort(Protocol):
    async def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]: ...

    async def latest_price(self, symbol: str) -> float: ...

    async def get_asset(self, symbol: str) -> AssetSpec: ...


@runtime_checkable
class NewsPort(Protocol):
    async def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]: ...


@runtime_checkable
class BrokerPort(Protocol):
    async def get_portfolio(self) -> PortfolioState: ...

    async def submit(self, req: OrderRequest) -> Order: ...

    async def cancel(self, order_id: str) -> bool: ...

    async def get_order(self, order_id: str) -> Order | None: ...

    async def list_open_orders(self) -> list[Order]: ...


PortfolioState.model_rebuild()
