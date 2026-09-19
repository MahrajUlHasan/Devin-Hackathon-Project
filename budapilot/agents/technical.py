"""A2 -- Technical Analyst. Model: claude-sonnet-5.

One call per shortlisted candidate. Reads the indicator bundle, returns a directional
thesis with an explicit invalidation price. The invalidation field matters more than it
looks: an analyst who cannot say what would prove them wrong is not an analyst.
"""

from __future__ import annotations

from dataclasses import dataclass

from budapilot.agents.base import Agent
from budapilot.agents.fmt import features_block
from budapilot.contracts import Action, FeatureBundle, RegimeOutput, TechnicalOutput

SYSTEM = """You are the Technical Analyst on a crypto trading desk. You are given a \
computed indicator bundle for one symbol on a 5-minute chart and must return a directional \
read.

Rules:
- This desk is LONG ONLY. BUY means open or add to a long. SELL means reduce or close an \
existing long. SELL NEVER means open a short. If the read is bearish and there is no \
position to reduce, the correct answer is HOLD.
- `conviction` is your honest probability that this direction plays out over `horizon_bars` \
5-minute bars. Calibrate it. A desk that reports 0.9 on everything is useless. Most reads \
on a 5-minute crypto chart deserve 0.4-0.7.
- `invalidation` is the price at which your thesis is simply wrong. Always set it.
- `key_factors` names at most four specific indicators with their values. No vague phrases \
like "momentum looks good".
- Be willing to say HOLD. Most bars are not trades."""


@dataclass
class TechnicalContext:
    features: FeatureBundle
    regime: RegimeOutput | None = None
    has_position: bool = False

    @property
    def symbol(self) -> str:
        return self.features.symbol


class TechnicalAgent(Agent[TechnicalOutput]):
    name = "technical"
    output_model = TechnicalOutput
    system = SYSTEM
    max_tokens = 900

    def user_prompt(self, ctx: TechnicalContext) -> str:
        parts = [features_block(ctx.features)]
        if ctx.regime:
            parts.append(
                f"\nMarket regime: {ctx.regime.regime.value} "
                f"(realized vol percentile {ctx.regime.realized_vol_pct:.0f}). "
                f"{ctx.regime.rationale}"
            )
        held = (
            "OPEN (SELL would reduce it)"
            if ctx.has_position
            else "NONE (SELL is not available)"
        )
        parts.append(f"\nCurrent position in {ctx.symbol}: {held}")
        parts.append("\nGive your read.")
        return "\n".join(parts)

    def stub(self, ctx: TechnicalContext) -> TechnicalOutput:
        f = ctx.features
        score = f.trend_score
        if score > 0.5:
            action, conv = Action.BUY, min(0.5 + score / 4, 0.8)
        elif score < -0.5 and ctx.has_position:
            action, conv = Action.SELL, min(0.5 + abs(score) / 4, 0.8)
        else:
            action, conv = Action.HOLD, 0.5
        atr = f.atr_14 or 0.0
        return TechnicalOutput(
            symbol=f.symbol,
            direction=action,
            conviction=round(conv, 2),
            horizon_bars=12,
            support=round(f.close - 2 * atr, 6) if atr else None,
            resistance=round(f.close + 2 * atr, 6) if atr else None,
            invalidation=round(f.close - 2 * atr, 6) if atr else None,
            rationale=f"Deterministic read from composite trend score {score:.2f}.",
            key_factors=[f"trend_score={score:.2f}", f"rsi={f.rsi_14 or 0:.1f}"],
        )
