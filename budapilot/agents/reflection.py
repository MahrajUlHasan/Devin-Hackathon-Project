"""A7 -- Reflection. Model: claude-sonnet-5. Runs on position close, off the hot path.

Reads the full decision chain that led to a trade against what actually happened, and
writes one lesson. Lessons are persisted and the most recent K are injected into A6's
prompt, so the desk's own history becomes an input to its next decision.

This is the only component that closes the loop. It is also the cheapest thing in the
system to get wrong in an impressive-looking way, so the prompt pushes hard against
hindsight narration.
"""

from __future__ import annotations

from dataclasses import dataclass

from budapilot.agents.base import Agent
from budapilot.contracts import ExitReason, Lesson, NewsOutput, TechnicalOutput, TradeProposal

SYSTEM = """You review closed trades on a crypto desk and write one lesson.

You are given the reasoning that led to the trade and what actually happened. Write the \
lesson the desk should carry forward.

Discipline:
- A single trade is one sample. Do not infer a strategy from it. "RSI above 70 always \
reverses" is not a lesson, it is overfitting to noise.
- A trade that lost money after sound reasoning is not a mistake. Say so when it is true. \
The useful lesson there is often "the process was right, the outcome was variance".
- A trade that made money after sloppy reasoning IS a mistake. Say that too.
- Prefer lessons about the decision process over lessons about price.
- One or two sentences. Concrete. Something a PM could actually apply on the next bar.

`tags` are at most four short keywords for retrieval, e.g. "high-atr", "news-override", \
"stopped-out", "trend-continuation"."""


@dataclass
class ReflectionContext:
    symbol: str
    proposal: TradeProposal
    technical: TechnicalOutput | None
    news: NewsOutput | None
    entry: float
    exit_price: float
    outcome_pct: float
    exit_reason: ExitReason
    bars_held: int


class ReflectionAgent(Agent[Lesson]):
    name = "reflection"
    output_model = Lesson
    system = SYSTEM
    max_tokens = 700

    def user_prompt(self, ctx: ReflectionContext) -> str:
        parts = [
            f"CLOSED TRADE: {ctx.symbol}",
            f"  entry {ctx.entry:.4f} -> exit {ctx.exit_price:.4f} "
            f"({ctx.outcome_pct:+.2f}%) after {ctx.bars_held} bars",
            f"  exit reason: {ctx.exit_reason.value}",
            "",
            "THE REASONING AT ENTRY",
            f"  PM: {ctx.proposal.action.value} conviction={ctx.proposal.conviction:.2f} "
            f"size_mult={ctx.proposal.size_multiplier:.2f}",
            f"    {ctx.proposal.rationale}",
        ]
        if ctx.proposal.overrode:
            parts += [f"    overrode: {'; '.join(ctx.proposal.overrode)}"]
        if ctx.technical:
            parts += [
                f"  Technical: {ctx.technical.direction.value} "
                f"conv={ctx.technical.conviction:.2f} -- {ctx.technical.rationale}",
                f"    invalidation was {ctx.technical.invalidation}",
            ]
        if ctx.news:
            parts += [
                f"  News: sentiment={ctx.news.sentiment:+.2f} "
                f"risk={ctx.news.news_risk.value} -- {ctx.news.rationale}"
            ]
        parts += ["", "Write the lesson."]
        return "\n".join(parts)

    def stub(self, ctx: ReflectionContext) -> Lesson:
        verdict = "hit its target" if ctx.outcome_pct > 0 else "was stopped out"
        return Lesson(
            symbol=ctx.symbol,
            outcome_pct=round(ctx.outcome_pct, 3),
            exit_reason=ctx.exit_reason,
            lesson=(
                f"{ctx.symbol} {verdict} at {ctx.outcome_pct:+.2f}% after "
                f"{ctx.bars_held} bars. Recorded without model review (offline)."
            ),
            tags=[ctx.exit_reason.value.lower()],
        )
