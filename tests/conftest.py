from __future__ import annotations

import pytest

from budapilot.contracts import (
    Action,
    AssetSpec,
    FeatureBundle,
    NewsOutput,
    NewsRisk,
    PortfolioState,
    Position,
    RiskAnalystOutput,
    SymbolOpinions,
    TechnicalOutput,
    TradeProposal,
    utcnow,
)

SYMBOL = "BTC/USD"


@pytest.fixture
def asset() -> AssetSpec:
    return AssetSpec(
        symbol=SYMBOL, min_order_size=1.0, min_trade_increment=1e-8, tradable=True
    )


@pytest.fixture
def features() -> FeatureBundle:
    return FeatureBundle(
        symbol=SYMBOL,
        ts=utcnow(),
        close=60_000.0,
        atr_14=600.0,
        atr_pct=0.01,
        rsi_14=58.0,
        macd_hist=12.0,
        ema_12=60_100.0,
        ema_26=59_800.0,
        ret_5=0.004,
        ret_20=0.011,
        trend_score=1.4,
    )


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(equity=100_000.0, cash=100_000.0, positions={})


@pytest.fixture
def proposal() -> TradeProposal:
    return TradeProposal(
        symbol=SYMBOL,
        action=Action.BUY,
        conviction=0.75,
        size_multiplier=1.0,
        stop_atr=2.0,
        rationale="test",
    )


def make_position(symbol: str = SYMBOL, qty: float = 0.1, entry: float = 60_000.0) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry=entry,
        market_value=qty * entry,
        current_price=entry,
    )


def make_opinion(
    symbol: str = SYMBOL,
    *,
    direction: Action = Action.BUY,
    conviction: float = 0.8,
    sentiment: float = 0.2,
    news_risk: NewsRisk = NewsRisk.LOW,
    size_multiplier: float = 1.0,
    features: FeatureBundle | None = None,
) -> SymbolOpinions:
    f = features or FeatureBundle(
        symbol=symbol, ts=utcnow(), close=100.0, atr_14=1.0, atr_pct=0.01, trend_score=1.0
    )
    return SymbolOpinions(
        symbol=symbol,
        features=f,
        technical=TechnicalOutput(
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            horizon_bars=12,
            rationale="t",
        ),
        news=NewsOutput(symbol=symbol, sentiment=sentiment, news_risk=news_risk),
        risk=RiskAnalystOutput(symbol=symbol, size_multiplier=size_multiplier),
    )
