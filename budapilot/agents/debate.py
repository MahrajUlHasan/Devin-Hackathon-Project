"""A9 Bull and A10 Bear. Model: claude-sonnet-5 each.

Adversarial review before the PM rules. Two advocates argue one symbol, optionally
rebut each other, and the PM reads both cases alongside the specialists.

Why adversaries rather than one more analyst: the failure mode of a pipeline where
every agent sees the same features is that they all agree, and unanimity reads as
confidence when it is really just correlation. Assigning a side forces the strongest
available argument for each direction onto the page, so the PM's job becomes weighing
a real disagreement instead of rubber-stamping a consensus.

The `conceded` field is the one that keeps this honest. An advocate who cannot name the
best point against them is not arguing, they are cheerleading, and the PM should
discount them accordingly.

Off by default (`ENABLE_DEBATE`): it roughly doubles tokens and latency per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

from budapilot.agents.base import Agent
from budapilot.agents.fmt import features_block, news_block
from budapilot.contracts import (
    DebateCase,
    FeatureBundle,
    NewsItem,
    NewsOutput,
    RegimeOutput,
    TechnicalOutput,
)

_SHARED = """You are on a crypto trading desk that runs adversarial review before \
committing capital. You have been assigned a side. Argue it as well as it can honestly \
be argued.

Rules for both advocates:
- Ground every claim in the data you were given. Cite actual indicator values and \
actual headlines. An argument that would read the same for any symbol is worthless.
- `claims` is at most four specific points.
- `strongest_point` is the single best argument for your side. One sentence.
- `conceded` is the strongest point AGAINST your side that you accept is real. This is \
mandatory and it is not a formality. If you cannot name one, you are not thinking.
- `confidence` is how strong your side's case actually is, in [0, 1] -- NOT how \
strongly you are arguing it. Being assigned the bull side of a bad setup should \
produce a low number. Inflating it destroys your usefulness to the PM.

This desk is LONG ONLY: the bear case argues against entering or for reducing, never \
for opening a short."""

BULL_SYSTEM = (
    _SHARED
    + """

You are the BULL. Argue for entering or adding to a long position in this symbol."""
)

BEAR_SYSTEM = (
    _SHARED
    + """

You are the BEAR. Argue against entering, or for reducing an existing long. The desk's \
default is to do nothing, so you do not need to prove disaster is coming -- only that \
the edge is not there. "This setup is unremarkable and the fees are certain" is a \
legitimate bear case."""
)

REBUTTAL_NOTE = """

You have now seen the opposing case. Fill `rebuttal` with your direct response to their \
strongest point. Update `confidence` if they genuinely moved you -- a debate where \
neither side ever updates is theatre."""


@dataclass
class DebateContext:
    features: FeatureBundle
    technical: TechnicalOutput
    news: NewsOutput
    headlines: list[NewsItem] | None = None
    regime: RegimeOutput | None = None
    has_position: bool = False
    opposing: DebateCase | None = None  # set in round 2

    @property
    def symbol(self) -> str:
        return self.features.symbol


class _Advocate(Agent[DebateCase]):
    output_model = DebateCase
    max_tokens = 900
    side: str = "BULL"

    def user_prompt(self, ctx: DebateContext) -> str:
        parts = [features_block(ctx.features), ""]
        if ctx.regime:
            parts += [f"Regime: {ctx.regime.regime.value} -- {ctx.regime.rationale}", ""]
        parts += [
            f"Technical Analyst: {ctx.technical.direction.value} "
            f"conviction={ctx.technical.conviction:.2f}, "
            f"invalidation={ctx.technical.invalidation}",
            f"  {ctx.technical.rationale}",
            "",
            f"News Analyst: sentiment={ctx.news.sentiment:+.2f} "
            f"risk={ctx.news.news_risk.value}",
            f"  {ctx.news.rationale}",
            "",
            f"Position currently open: {'yes' if ctx.has_position else 'no'}",
        ]
        if ctx.headlines:
            parts += ["", "Headlines:", news_block(ctx.headlines, limit=10)]
        if ctx.opposing:
            parts += [
                "",
                f"THE {ctx.opposing.side} CASE:",
                f"  strongest point: {ctx.opposing.strongest_point}",
                *[f"  - {c}" for c in ctx.opposing.claims],
                f"  they conceded: {ctx.opposing.conceded or '(nothing)'}",
                f"  their confidence: {ctx.opposing.confidence:.2f}",
                "",
                "Rebut it.",
            ]
        else:
            parts += ["", f"Make the {self.side.lower()} case."]
        return "\n".join(parts)

    def stub(self, ctx: DebateContext) -> DebateCase:
        """Deterministic case derived from the trend score, so offline runs still show
        a real disagreement rather than two blank cards."""
        score = ctx.features.trend_score
        bullish = score > 0
        strength = min(abs(score) / 3.0, 1.0)
        confidence = strength if (bullish == (self.side == "BULL")) else 1.0 - strength
        direction = "supports" if bullish == (self.side == "BULL") else "works against"
        return DebateCase(
            symbol=ctx.symbol,
            side=self.side,  # type: ignore[arg-type]
            claims=[
                f"Composite trend score {score:+.2f} {direction} this side.",
                f"RSI {ctx.features.rsi_14 or 0:.1f}, ATR "
                f"{(ctx.features.atr_pct or 0) * 100:.2f}% of price.",
            ],
            strongest_point=f"Trend score {score:+.2f} with EMA {ctx.features.ema_cross}.",
            conceded="Offline stub: no adversarial review performed.",
            confidence=round(max(0.0, min(1.0, confidence)), 2),
        )


class BullAgent(_Advocate):
    name = "bull"
    system = BULL_SYSTEM
    side = "BULL"


class BearAgent(_Advocate):
    name = "bear"
    system = BEAR_SYSTEM
    side = "BEAR"


class BullRebuttalAgent(BullAgent):
    system = BULL_SYSTEM + REBUTTAL_NOTE


class BearRebuttalAgent(BearAgent):
    system = BEAR_SYSTEM + REBUTTAL_NOTE
