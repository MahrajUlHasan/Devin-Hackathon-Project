"""A1 -- Scout (screener). Model: claude-haiku-4-5.

Honest division of labour: **ranking six symbols by trend strength is arithmetic**, done
in pandas by ``rank_symbols`` below. The model never does the maths. It is asked for two
things a number cannot give: a one-line human justification, and a structural veto on a
candidate whose chart is broken in a way the composite score missed (a vertical blow-off
that scores as "strong trend", say).
"""

from __future__ import annotations

from dataclasses import dataclass

from budapilot.agents.base import Agent
from budapilot.agents.fmt import features_line
from budapilot.config import MAX_CANDIDATES
from budapilot.contracts import FeatureBundle, RegimeOutput, ScoutCandidate, ScoutOutput

SYSTEM = """You are the Scout on a crypto trading desk. You screen a small watchlist and \
hand the desk a shortlist.

The numeric ranking has ALREADY been computed deterministically and is given to you. Do \
not re-rank and do not recompute anything. Your job is exactly two things:

1. Write a one-line thesis (max 20 words) for each candidate, in plain desk English.
2. Set `vetoed: true` on any candidate whose structure is broken in a way a trend score \
cannot see -- a parabolic blow-off, a single-bar spike on thin volume, a symbol pinned \
at a 20-bar high with RSI above 80 and no pullback. Give a short veto_reason.

Veto sparingly. A veto with a weak reason is worse than no veto.

This desk is LONG ONLY. Never suggest shorting.
Keep `market_note` to one sentence on the overall tape."""


@dataclass
class ScoutContext:
    features: list[FeatureBundle]
    regime: RegimeOutput | None = None


def rank_symbols(features: list[FeatureBundle], top_n: int = MAX_CANDIDATES) -> list[FeatureBundle]:
    """Deterministic shortlist. Untradeable bundles (no ATR) are dropped here, not later."""
    tradeable = [f for f in features if f.is_tradeable()]
    return sorted(tradeable, key=lambda f: f.trend_score, reverse=True)[:top_n]


class ScoutAgent(Agent[ScoutOutput]):
    name = "scout"
    output_model = ScoutOutput
    system = SYSTEM
    max_tokens = 800

    def symbol_of(self, ctx: ScoutContext) -> str | None:
        return None

    def user_prompt(self, ctx: ScoutContext) -> str:
        ranked = rank_symbols(ctx.features)
        regime = ctx.regime.regime.value if ctx.regime else "UNKNOWN"
        lines = [f"Market regime: {regime}", "", "Full watchlist:"]
        lines += [f"  {features_line(f)}" for f in ctx.features]
        lines += ["", f"Pre-ranked shortlist (top {len(ranked)} by composite trend score):"]
        lines += [f"  {i + 1}. {features_line(f)}" for i, f in enumerate(ranked)]
        lines += [
            "",
            "Return one candidate object per shortlisted symbol, in the same order.",
        ]
        return "\n".join(lines)

    def stub(self, ctx: ScoutContext) -> ScoutOutput:
        ranked = rank_symbols(ctx.features)
        return ScoutOutput(
            candidates=[
                ScoutCandidate(
                    symbol=f.symbol,
                    trend_score=f.trend_score,
                    thesis=f"Composite trend score {f.trend_score:.2f}; ranked #{i + 1}.",
                )
                for i, f in enumerate(ranked)
            ],
            market_note="Deterministic shortlist (no model).",
        )
