"""A4 -- Regime Analyst. Model: claude-haiku-4-5, cached 30 minutes.

One call for the whole market, not per symbol. The statistics (realized vol percentile,
BTC-vs-alt dispersion) are computed in pandas; the model supplies the label and the
one-paragraph read that the Technical Analyst and PM both condition on.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from budapilot.agents.base import Agent
from budapilot.agents.fmt import features_line
from budapilot.contracts import FeatureBundle, Regime, RegimeOutput

SYSTEM = """You are the Regime Analyst on a crypto trading desk. Once every 30 minutes you \
label the tape so the rest of the desk can condition on it.

Choose exactly one regime:
  TRENDING        directional persistence, most of the book moving together with the trend
  RANGING         chop, mean reversion, no persistent direction
  VOL_EXPANSION   realized volatility breaking out of its recent range, ranges widening
  RISK_OFF        broad drawdown, correlations going to 1, alts underperforming BTC

`btc_dominance_drift` is the BTC 20-bar return minus the average alt 20-bar return. \
Positive means BTC is outperforming (defensive rotation).
`realized_vol_pct` is given to you; pass it through.
`correlation_cluster` is a short phrase describing what is moving together.
Keep the rationale to two sentences."""


@dataclass
class RegimeContext:
    features: list[FeatureBundle] = field(default_factory=list)
    realized_vol_pct: float = 50.0


def btc_dominance_drift(features: list[FeatureBundle]) -> float:
    btc = next((f for f in features if f.symbol.startswith("BTC")), None)
    alts = [f for f in features if not f.symbol.startswith("BTC") and f.ret_20 is not None]
    if btc is None or btc.ret_20 is None or not alts:
        return 0.0
    return round(btc.ret_20 - statistics.fmean(f.ret_20 or 0.0 for f in alts), 5)


class RegimeAgent(Agent[RegimeOutput]):
    name = "regime"
    output_model = RegimeOutput
    system = SYSTEM
    max_tokens = 600

    def symbol_of(self, ctx: RegimeContext) -> str | None:
        return None

    def user_prompt(self, ctx: RegimeContext) -> str:
        lines = ["Watchlist snapshot:"]
        lines += [f"  {features_line(f)}" for f in ctx.features]
        lines += [
            "",
            f"Realized vol percentile (20-bar, vs 30d): {ctx.realized_vol_pct:.0f}",
            f"BTC dominance drift (BTC ret20 - mean alt ret20): "
            f"{btc_dominance_drift(ctx.features) * 100:+.2f}%",
            "",
            "Label the regime.",
        ]
        return "\n".join(lines)

    def stub(self, ctx: RegimeContext) -> RegimeOutput:
        drift = btc_dominance_drift(ctx.features)
        rets = [f.ret_20 for f in ctx.features if f.ret_20 is not None]
        mean_ret = statistics.fmean(rets) if rets else 0.0
        if ctx.realized_vol_pct > 80:
            regime = Regime.VOL_EXPANSION
        elif mean_ret < -0.02:
            regime = Regime.RISK_OFF
        elif abs(mean_ret) > 0.01:
            regime = Regime.TRENDING
        else:
            regime = Regime.RANGING
        return RegimeOutput(
            regime=regime,
            btc_dominance_drift=drift,
            realized_vol_pct=ctx.realized_vol_pct,
            correlation_cluster="n/a (offline)",
            rationale="Deterministic regime label from mean 20-bar return and vol percentile.",
        )
