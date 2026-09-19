"""The PM's fallback. Not a stub.

If Opus times out and we fall back to a canned HOLD, the system stops trading and the
demo is dead on stage. Instead we degrade to a quant: a weighted vote over the same
specialist opinions the PM would have read, emitting the same `TradeProposal` shape and
a rationale that says plainly that it is the fallback.

The system degrades to a quant, not to a corpse.
"""

from __future__ import annotations

from budapilot.config import MAX_POSITION_PCT, MIN_CONVICTION
from budapilot.contracts import (
    Action,
    NewsRisk,
    PortfolioState,
    SymbolOpinions,
    TradeProposal,
)

# Sentiment nudges conviction but must not dominate a technical read.
SENTIMENT_WEIGHT = 0.15


def score_opinion(op: SymbolOpinions) -> float:
    """Blend technical conviction with news sentiment, then scale by the risk multiplier."""
    base = op.technical.conviction
    if op.technical.direction is Action.BUY:
        adjusted = base + SENTIMENT_WEIGHT * op.news.sentiment
    elif op.technical.direction is Action.SELL:
        adjusted = base - SENTIMENT_WEIGHT * op.news.sentiment
    else:
        return 0.0
    return max(0.0, min(1.0, adjusted)) * op.risk.size_multiplier


def has_headroom(op: SymbolOpinions, portfolio: PortfolioState | None) -> bool:
    """False if a BUY here would just collect a per-symbol-cap rejection."""
    if portfolio is None or op.technical.direction is not Action.BUY:
        return True
    position = portfolio.positions.get(op.symbol)
    if position is None or portfolio.equity <= 0:
        return True
    return position.market_value / portfolio.equity < MAX_POSITION_PCT - 0.005


def deterministic_arbitrate(
    opinions: list[SymbolOpinions], portfolio: PortfolioState | None = None
) -> TradeProposal:
    """Pick the best actionable opinion. Always returns a proposal, never raises."""
    actionable = [op for op in opinions if has_headroom(op, portfolio)]
    if not actionable:
        return TradeProposal(
            symbol=opinions[0].symbol if opinions else "-",
            action=Action.HOLD,
            conviction=0.0,
            rationale=(
                "Fallback arbiter: no candidates this bar."
                if not opinions
                else "Fallback arbiter: every candidate is already at its position cap."
            ),
        )

    ranked = sorted(actionable, key=score_opinion, reverse=True)
    best = ranked[0]
    score = score_opinion(best)

    if best.technical.direction is Action.HOLD or score < MIN_CONVICTION:
        return TradeProposal(
            symbol=best.symbol,
            action=Action.HOLD,
            conviction=round(score, 2),
            rationale=(
                f"Fallback arbiter (the PM model was unavailable): best blended score "
                f"{score:.2f} is below the {MIN_CONVICTION:.2f} floor. Standing down."
            ),
            overrode=["portfolio_manager: unavailable, deterministic vote substituted"],
        )

    if best.news.news_risk is NewsRisk.CRITICAL:
        return TradeProposal(
            symbol=best.symbol,
            action=Action.HOLD,
            conviction=0.0,
            rationale=(
                "Fallback arbiter: news risk is CRITICAL on the leading candidate. "
                "Standing down."
            ),
            overrode=["portfolio_manager: unavailable, deterministic vote substituted"],
        )

    return TradeProposal(
        symbol=best.symbol,
        action=best.technical.direction,
        conviction=round(score, 2),
        size_multiplier=best.risk.size_multiplier,
        stop_atr=best.risk.suggested_stop_atr,
        rationale=(
            f"Fallback arbiter (the PM model was unavailable). Weighted vote: technical "
            f"conviction {best.technical.conviction:.2f}, news sentiment "
            f"{best.news.sentiment:+.2f}, risk multiplier "
            f"{best.risk.size_multiplier:.2f} -> blended {score:.2f}."
        ),
        overrode=["portfolio_manager: unavailable, deterministic vote substituted"],
    )
