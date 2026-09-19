"""A8 -- Deep Analysis. Model: claude-opus-5 at effort=high. Human-triggered only.

Deliberately out of the hot path: one click, one long call, a written thesis. Putting
Opus-at-high-effort latency inside a 5-minute decision loop would be a mistake; putting
it behind a button is a good demo beat and a genuinely useful desk tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from budapilot.agents.base import Agent
from budapilot.agents.fmt import features_block, news_block
from budapilot.contracts import (
    Action,
    DeepAnalysis,
    FeatureBundle,
    NewsItem,
    RegimeOutput,
)

SYSTEM = """You are the desk's senior analyst. A PM has asked you for a full written view \
on one crypto asset. Take your time and think it through properly.

Produce:
- `thesis`: three to five sentences. Your actual view, written for a professional. No \
hedging boilerplate, no "crypto is volatile" filler.
- `bull_case`: the strongest specific arguments for being long. Not generic optimism.
- `bear_case`: the strongest specific arguments against. Argue it as if you believed it. \
A bear case you do not take seriously is worthless to the PM.
- `verdict`: BUY, SELL (reduce an existing long) or HOLD. This desk is long only.
- `conviction`: calibrated in [0, 1].

Ground every claim in the data you were given. Where you are reasoning beyond it, say so \
explicitly. You are allowed to conclude that there is no edge here."""


@dataclass
class DeepContext:
    features: FeatureBundle
    news: list[NewsItem] = field(default_factory=list)
    regime: RegimeOutput | None = None

    @property
    def symbol(self) -> str:
        return self.features.symbol


class DeepAgent(Agent[DeepAnalysis]):
    name = "deep"
    output_model = DeepAnalysis
    system = SYSTEM
    max_tokens = 4000
    effort = "high"

    def user_prompt(self, ctx: DeepContext) -> str:
        parts = [f"Full analysis requested: {ctx.symbol}", "", features_block(ctx.features)]
        if ctx.regime:
            parts += [
                "",
                f"Market regime: {ctx.regime.regime.value} -- {ctx.regime.rationale}",
            ]
        parts += ["", "Recent headlines:", news_block(ctx.news, limit=20), "", "Write it up."]
        return "\n".join(parts)

    def stub(self, ctx: DeepContext) -> DeepAnalysis:
        return DeepAnalysis(
            symbol=ctx.symbol,
            thesis=(
                f"Deep analysis is unavailable offline. Composite trend score for "
                f"{ctx.symbol} is {ctx.features.trend_score:+.2f} with RSI "
                f"{ctx.features.rsi_14 or 0:.1f}."
            ),
            bull_case=["(offline)"],
            bear_case=["(offline)"],
            verdict=Action.HOLD,
            conviction=0.0,
        )
