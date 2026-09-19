"""A5 -- Risk Analyst. Model: claude-sonnet-5. **Advisory only.**

This agent cannot approve anything. Its entire authority is to shrink a position via
`size_multiplier` and to widen or tighten the stop. The deterministic risk engine holds
the veto, and it will reject a trade this agent loved.

That split is deliberate: the decision that must be *correct* is code with 100% branch
coverage; the decision that must be *judged* is a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from budapilot.agents.base import Agent
from budapilot.agents.fmt import features_block, portfolio_block
from budapilot.config import MAX_OPEN_POSITIONS, MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE_PCT
from budapilot.contracts import (
    FeatureBundle,
    NewsOutput,
    NewsRisk,
    PortfolioState,
    RegimeOutput,
    RiskAnalystOutput,
    TechnicalOutput,
)

SYSTEM = f"""You are the Risk Analyst on a crypto trading desk. You advise on sizing. \
You do NOT approve trades -- a deterministic risk engine downstream holds the veto and \
will reject trades you were comfortable with. Your job is to shrink exposure where the \
setup deserves less than full size.

`size_multiplier` in [0, 1] scales the position the risk engine would otherwise take:
  1.0   clean setup, calm tape, no news overhang
  0.5   something is off: elevated volatility, conflicting signals, thin conviction
  0.25  you want the desk in small or not at all
  0.0   do not take this trade

Shrink hard for: news_risk HIGH, RISK_OFF or VOL_EXPANSION regime, ATR above ~3% of \
price on a 5-minute chart, RSI above 75 on a BUY, a portfolio already near its limits.

`suggested_stop_atr` in [0.5, 5.0] is the stop distance in ATR multiples. Default 2.0. \
Widen it in high volatility so you are not stopped out by noise; tighten it when the \
invalidation level is close.

Hard limits already enforced downstream (do not restate them as concerns unless the \
trade is genuinely near one): max {MAX_POSITION_PCT:.0%} of equity per symbol, \
max {MAX_TOTAL_EXPOSURE_PCT:.0%} gross exposure, max {MAX_OPEN_POSITIONS} open positions.

`concerns` is at most four short, specific strings. Not "market is risky"."""


@dataclass
class RiskAnalystContext:
    features: FeatureBundle
    technical: TechnicalOutput
    news: NewsOutput
    portfolio: PortfolioState
    regime: RegimeOutput | None = None

    @property
    def symbol(self) -> str:
        return self.features.symbol


class RiskAnalystAgent(Agent[RiskAnalystOutput]):
    name = "risk_analyst"
    output_model = RiskAnalystOutput
    system = SYSTEM
    max_tokens = 800

    def user_prompt(self, ctx: RiskAnalystContext) -> str:
        t, n = ctx.technical, ctx.news
        parts = [
            features_block(ctx.features),
            "",
            f"Technical Analyst: {t.direction.value} conviction={t.conviction:.2f} "
            f"horizon={t.horizon_bars} bars, invalidation={t.invalidation}",
            f"  {t.rationale}",
            "",
            f"News Analyst: sentiment={n.sentiment:+.2f} risk={n.news_risk.value} "
            f"({n.headline_count} headlines)",
            f"  catalysts: {', '.join(n.catalysts) if n.catalysts else 'none'}",
            f"  {n.rationale}",
            "",
        ]
        if ctx.regime:
            parts.append(f"Regime: {ctx.regime.regime.value} -- {ctx.regime.rationale}")
            parts.append("")
        parts += [portfolio_block(ctx.portfolio), "", "Advise on sizing."]
        return "\n".join(parts)

    def stub(self, ctx: RiskAnalystContext) -> RiskAnalystOutput:
        mult = 1.0
        concerns: list[str] = []
        atr_pct = ctx.features.atr_pct or 0.0
        if atr_pct > 0.03:
            mult *= 0.5
            concerns.append(f"ATR is {atr_pct * 100:.1f}% of price")
        if ctx.news.news_risk.rank >= NewsRisk.HIGH.rank:
            mult *= 0.5
            concerns.append(f"news risk {ctx.news.news_risk.value}")
        if ctx.portfolio.exposure_pct > MAX_TOTAL_EXPOSURE_PCT * 0.8:
            mult *= 0.5
            concerns.append("portfolio near exposure cap")
        return RiskAnalystOutput(
            symbol=ctx.symbol,
            size_multiplier=round(max(mult, 0.0), 2),
            suggested_stop_atr=2.5 if atr_pct > 0.03 else 2.0,
            concerns=concerns[:4],
            rationale="Deterministic sizing heuristic (no model).",
        )
