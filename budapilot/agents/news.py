"""A3 -- News & Catalyst Analyst. Two stages.

Stage 1 (``HeadlineScorer``, claude-haiku-4-5): batch-score every headline to [-1, +1]
with a one-line reason. High volume, nearly mechanical, so it goes to the cheapest model.

Stage 2 (``NewsAgent``, claude-sonnet-5): synthesise the scored headlines into an
aggregate sentiment and -- the output that actually has teeth -- a ``news_risk`` rating.
``CRITICAL`` is a hard reject in the risk engine, so the prompt is explicit about what
earns it.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import timedelta

from pydantic import BaseModel

from budapilot.agents.base import Agent
from budapilot.agents.fmt import news_block
from budapilot.contracts import NewsItem, NewsOutput, NewsRisk, ScoredHeadline, utcnow

SCORER_SYSTEM = """You score crypto news headlines for trading sentiment.

For each headline return a score in [-1, +1] and a reason of at most 12 words.
  +1.0  unambiguously bullish for this asset (major adoption, ETF approval, supply shock)
  +0.3  mildly positive
   0.0  noise, price commentary, or not actually about this asset
  -0.3  mildly negative
  -1.0  unambiguously bearish (exchange hack, enforcement action, delisting, depeg)

Score the asset named, not the general crypto mood. Price-recap headlines ("BTC rises 2%") \
are noise: score them near 0. Return one entry per headline, in order."""

SYNTH_SYSTEM = """You are the News & Catalyst Analyst on a crypto trading desk. You are \
given headlines that have already been scored individually. Synthesise them.

`sentiment` is a time-weighted aggregate in [-1, +1]. Recent headlines dominate.

`news_risk` is the important field. It gates real money, so calibrate it honestly:
  LOW      routine flow, price commentary, nothing pending
  MEDIUM   a real story with unclear direction, or a scheduled event soon
  HIGH     a significant unresolved catalyst: regulatory decision pending, large \
unlock, exchange under stress, credible security concern
  CRITICAL an active, confirmed, asset-specific crisis: exchange hack in progress, \
enforcement action filed against the asset, depeg underway, imminent delisting

CRITICAL is a hard veto on all trading in this symbol. Do not reach for it because the \
market is merely volatile or a headline is loud. It requires a confirmed, ongoing, \
asset-specific event. If you are unsure between HIGH and CRITICAL, choose HIGH.

With no headlines at all, return sentiment 0.0 and news_risk LOW.

`catalysts` lists concrete upcoming or ongoing events, not vibes."""


class ScoredHeadlineList(BaseModel):
    scores: list[ScoredHeadline] = []


def _stable_pseudo_score(headline: str) -> float:
    """A repeatable [-1, 1] score for offline use.

    Deliberately not ``hash()``: Python randomises string hashing per process, so that
    would give a different "deterministic" answer on every run and quietly destroy the
    reproducibility that --demo-safe is supposed to guarantee.
    """
    digest = zlib.crc32(headline.encode("utf-8"))
    return round(((digest % 2001) - 1000) / 1000.0, 3)


@dataclass
class NewsContext:
    symbol: str
    items: list[NewsItem] = field(default_factory=list)
    scored: list[ScoredHeadline] = field(default_factory=list)


class HeadlineScorer(Agent[ScoredHeadlineList]):
    name = "headline_scorer"
    output_model = ScoredHeadlineList
    system = SCORER_SYSTEM
    max_tokens = 1500

    def user_prompt(self, ctx: NewsContext) -> str:
        return f"Asset: {ctx.symbol}\n\nHeadlines:\n{news_block(ctx.items, limit=20)}"

    def stub(self, ctx: NewsContext) -> ScoredHeadlineList:
        """Deterministic pseudo-scores derived from the headline text.

        Returning 0.0 for everything would be more "honest", but it makes the offline
        system degenerate: neutral sentiment everywhere means no agent can ever
        disagree with another, the disagreement heatmap is uniformly blank, and
        --demo-safe stops being representative of what the real desk does. These
        scores are stable for a given headline, so runs remain reproducible.
        """
        return ScoredHeadlineList(
            scores=[
                ScoredHeadline(
                    headline=n.headline,
                    score=_stable_pseudo_score(n.headline),
                    reason="deterministic offline pseudo-score, not real analysis",
                )
                for n in ctx.items[:20]
            ]
        )


def time_decayed_sentiment(
    items: list[NewsItem], scored: list[ScoredHeadline], half_life_h: float = 8.0
) -> float:
    """Deterministic fallback aggregate, also used to sanity-check the model."""
    if not scored:
        return 0.0
    now = utcnow()
    total_w = 0.0
    total = 0.0
    for item, s in zip(items, scored, strict=False):
        age_h = max((now - item.ts) / timedelta(hours=1), 0.0)
        w = 0.5 ** (age_h / half_life_h)
        total += w * s.score
        total_w += w
    return round(total / total_w, 3) if total_w else 0.0


class NewsAgent(Agent[NewsOutput]):
    name = "news"
    output_model = NewsOutput
    system = SYNTH_SYSTEM
    max_tokens = 900

    def user_prompt(self, ctx: NewsContext) -> str:
        if not ctx.scored:
            return f"Asset: {ctx.symbol}\n\nNo headlines in the last 24 hours."
        lines = [f"Asset: {ctx.symbol}", "", "Scored headlines (newest first):"]
        for item, s in zip(ctx.items, ctx.scored, strict=False):
            lines.append(f"  [{s.score:+.2f}] {item.ts:%m-%d %H:%M} {s.headline} -- {s.reason}")
        lines += [
            "",
            f"Deterministic time-decayed aggregate: "
            f"{time_decayed_sentiment(ctx.items, ctx.scored):+.3f} "
            f"(a reference point, not a constraint)",
            "",
            "Synthesise.",
        ]
        return "\n".join(lines)

    def stub(self, ctx: NewsContext) -> NewsOutput:
        sentiment = time_decayed_sentiment(ctx.items, ctx.scored)
        # Strongly negative aggregate flow reads as elevated risk. Never CRITICAL:
        # that is a hard veto on real money and no offline heuristic has earned it.
        if sentiment <= -0.6:
            risk = NewsRisk.HIGH
        elif sentiment <= -0.25:
            risk = NewsRisk.MEDIUM
        else:
            risk = NewsRisk.LOW
        return NewsOutput(
            symbol=ctx.symbol,
            sentiment=sentiment,
            news_risk=risk,
            catalysts=[],
            cited=ctx.scored[:3],
            rationale="Deterministic time-decayed aggregate (no model).",
            headline_count=len(ctx.items),
        )
