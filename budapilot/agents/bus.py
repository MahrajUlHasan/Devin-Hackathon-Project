"""The orchestrator. One `run_bar` per closed 5-minute bar.

Sequencing is dictated by data dependencies, not by taste:

    regime (cached 30m)
        -> scout ranks, shortlists 3
        -> per candidate, in parallel: technical || news (cached 15m)
        -> per candidate, in parallel: risk analyst (needs technical + news)
        -> PM arbitrates, once

Everything that can be concurrent is concurrent, so wall-clock per bar is roughly
regime + scout + technical + risk + PM, not the sum of eleven calls.

`run_bar` never raises. Individual agents degrade to stubs; the PM degrades to the
deterministic arbiter.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from budapilot.agents.arbiter import deterministic_arbitrate
from budapilot.agents.base import AgentRuntime
from budapilot.agents.debate import (
    BearAgent,
    BearRebuttalAgent,
    BullAgent,
    BullRebuttalAgent,
    DebateContext,
)
from budapilot.agents.deep import DeepAgent, DeepContext
from budapilot.agents.news import HeadlineScorer, NewsAgent, NewsContext
from budapilot.agents.pm import PMAgent, PMContext
from budapilot.agents.reflection import ReflectionAgent, ReflectionContext
from budapilot.agents.regime import RegimeAgent, RegimeContext
from budapilot.agents.risk_analyst import RiskAnalystAgent, RiskAnalystContext
from budapilot.agents.scout import ScoutAgent, ScoutContext
from budapilot.agents.technical import TechnicalAgent, TechnicalContext
from budapilot.config import (
    DEBATE_ROUNDS,
    ENABLE_DEBATE,
    MAX_CANDIDATES,
    NEWS_TTL_S,
    REGIME_TTL_S,
)
from budapilot.contracts import (
    Action,
    AgentResult,
    AgentStatus,
    BarDecision,
    Debate,
    DeepAnalysis,
    Disagreement,
    FeatureBundle,
    Lesson,
    NewsItem,
    NewsOutput,
    PortfolioState,
    RegimeOutput,
    SymbolOpinions,
    TechnicalOutput,
    TradeProposal,
    utcnow,
)


class _TTLCache:
    def __init__(self, ttl_s: float) -> None:
        self.ttl_s = ttl_s
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        hit = self._store.get(key)
        if hit is None:
            return None
        ts, value = hit
        if time.monotonic() - ts > self.ttl_s:
            del self._store[key]
            return None
        return value

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)


def disagreement_score(op: SymbolOpinions, proposal: TradeProposal | None) -> Disagreement:
    """Quantify how far apart the desk was on one symbol.

    Three axes, averaged: technical direction vs news sentiment, how hard the risk
    analyst cut size, and whether the PM overrode anyone. Unanimity reads as confidence
    when it is often just correlation, so measuring the spread is what makes a
    multi-agent system legible rather than merely plural.
    """
    signed = {
        Action.BUY: op.technical.conviction,
        Action.SELL: -op.technical.conviction,
    }.get(op.technical.direction, 0.0)

    # Technical vs news, each mapped to [-1, 1]; a full inversion scores 1.0.
    tech_vs_news = abs(signed - op.news.sentiment) / 2.0
    # A risk analyst cutting to 0.2x is loudly disagreeing with a confident technical.
    risk_cut = (1.0 - op.risk.size_multiplier) * abs(signed)

    overrode = 0
    pm_gap = 0.0
    if proposal is not None and proposal.symbol == op.symbol:
        overrode = len(proposal.overrode)
        pm_signed = {
            Action.BUY: proposal.conviction,
            Action.SELL: -proposal.conviction,
        }.get(proposal.action, 0.0)
        pm_gap = abs(signed - pm_signed) / 2.0

    score = max(0.0, min(1.0, (tech_vs_news + risk_cut + pm_gap) / 3.0))

    return Disagreement(
        symbol=op.symbol,
        technical_signed=round(signed, 3),
        news_sentiment=op.news.sentiment,
        risk_multiplier=op.risk.size_multiplier,
        pm_action=proposal.action if proposal else Action.HOLD,
        pm_conviction=proposal.conviction if proposal else 0.0,
        overrode_count=overrode,
        score=round(score, 3),
    )


class AgentBus:
    def __init__(self, runtime: AgentRuntime, *, enable_debate: bool | None = None) -> None:
        self.runtime = runtime
        self.enable_debate = ENABLE_DEBATE if enable_debate is None else enable_debate
        self.scout = ScoutAgent(runtime)
        self.technical = TechnicalAgent(runtime)
        self.headline_scorer = HeadlineScorer(runtime)
        self.news = NewsAgent(runtime)
        self.regime = RegimeAgent(runtime)
        self.risk_analyst = RiskAnalystAgent(runtime)
        self.pm = PMAgent(runtime)
        self.reflection = ReflectionAgent(runtime)
        self.deep = DeepAgent(runtime)
        self.bull = BullAgent(runtime)
        self.bear = BearAgent(runtime)
        self.bull_rebuttal = BullRebuttalAgent(runtime)
        self.bear_rebuttal = BearRebuttalAgent(runtime)

        self._regime_cache = _TTLCache(REGIME_TTL_S)
        self._news_cache = _TTLCache(NEWS_TTL_S)

    # -- stages --------------------------------------------------------------------

    async def _get_regime(
        self, features: list[FeatureBundle], realized_vol_pct: float, sink: list[AgentResult[Any]]
    ) -> RegimeOutput:
        cached = self._regime_cache.get("regime")
        if cached is not None:
            return cached
        res = await self.regime.run(
            RegimeContext(features=features, realized_vol_pct=realized_vol_pct)
        )
        sink.append(res)
        self._regime_cache.put("regime", res.output)
        return res.output

    async def _get_news(
        self, symbol: str, items: list[NewsItem], sink: list[AgentResult[Any]]
    ) -> NewsOutput:
        cached = self._news_cache.get(symbol)
        if cached is not None:
            return cached

        ctx = NewsContext(symbol=symbol, items=items)
        if items:
            scored = await self.headline_scorer.run(ctx)
            sink.append(scored)
            ctx.scored = scored.output.scores

        res = await self.news.run(ctx)
        sink.append(res)
        self._news_cache.put(symbol, res.output)
        return res.output

    async def _debate(
        self,
        features: FeatureBundle,
        technical: TechnicalOutput,
        news: NewsOutput,
        headlines: list[NewsItem],
        regime: RegimeOutput,
        has_position: bool,
        sink: list[AgentResult[Any]],
    ) -> Debate:
        """Run the adversarial round for one symbol.

        Round 1's advocates do not see each other -- otherwise whoever went first
        anchors the other, and the second case becomes a reaction rather than an
        argument. Round 2 is where they respond.
        """
        ctx = DebateContext(
            features=features,
            technical=technical,
            news=news,
            headlines=headlines,
            regime=regime,
            has_position=has_position,
        )
        bull_res, bear_res = await asyncio.gather(
            self.bull.run(ctx), self.bear.run(ctx)
        )
        sink.extend([bull_res, bear_res])
        bull, bear = bull_res.output, bear_res.output

        if DEBATE_ROUNDS > 1:
            bull_ctx = DebateContext(**{**ctx.__dict__, "opposing": bear})
            bear_ctx = DebateContext(**{**ctx.__dict__, "opposing": bull})
            r2_bull, r2_bear = await asyncio.gather(
                self.bull_rebuttal.run(bull_ctx), self.bear_rebuttal.run(bear_ctx)
            )
            sink.extend([r2_bull, r2_bear])
            bull, bear = r2_bull.output, r2_bear.output

        return Debate(symbol=features.symbol, bull=bull, bear=bear, rounds=DEBATE_ROUNDS)

    # -- the loop ------------------------------------------------------------------

    async def run_bar(
        self,
        *,
        bar_id: str,
        features: list[FeatureBundle],
        news_by_symbol: dict[str, list[NewsItem]],
        portfolio: PortfolioState,
        lessons: list[Lesson] | None = None,
        realized_vol_pct: float = 50.0,
    ) -> BarDecision:
        results: list[AgentResult[Any]] = []

        regime = await self._get_regime(features, realized_vol_pct, results)

        scout_res = await self.scout.run(ScoutContext(features=features, regime=regime))
        results.append(scout_res)
        scout = scout_res.output

        by_symbol = {f.symbol: f for f in features}
        candidates = [
            by_symbol[c.symbol]
            for c in scout.candidates
            if not c.vetoed and c.symbol in by_symbol and by_symbol[c.symbol].is_tradeable()
        ][:MAX_CANDIDATES]

        decision = BarDecision(
            bar_id=bar_id, ts=utcnow(), regime=regime, scout=scout, results=results
        )
        if not candidates:
            decision.proposal = TradeProposal(
                symbol="-",
                action=Action.HOLD,
                conviction=0.0,
                rationale="No tradeable candidates survived the scout's screen this bar.",
            )
            return decision

        # Technical and news are independent -- run every candidate's pair concurrently.
        tech_results, news_outputs = await asyncio.gather(
            asyncio.gather(
                *(
                    self.technical.run(
                        TechnicalContext(
                            features=f,
                            regime=regime,
                            has_position=f.symbol in portfolio.positions,
                        )
                    )
                    for f in candidates
                )
            ),
            asyncio.gather(
                *(
                    self._get_news(f.symbol, news_by_symbol.get(f.symbol, []), results)
                    for f in candidates
                )
            ),
        )
        results.extend(tech_results)

        # Risk analysis depends on both, so it is a second wave.
        risk_results = await asyncio.gather(
            *(
                self.risk_analyst.run(
                    RiskAnalystContext(
                        features=f,
                        technical=t.output,
                        news=n,
                        portfolio=portfolio,
                        regime=regime,
                    )
                )
                for f, t, n in zip(candidates, tech_results, news_outputs, strict=True)
            )
        )
        results.extend(risk_results)

        # Optional adversarial round, after the specialists and before the PM.
        debates: list[Debate | None] = [None] * len(candidates)
        if self.enable_debate:
            debates = list(
                await asyncio.gather(
                    *(
                        self._debate(
                            f,
                            t.output,
                            n,
                            news_by_symbol.get(f.symbol, []),
                            regime,
                            f.symbol in portfolio.positions,
                            results,
                        )
                        for f, t, n in zip(
                            candidates, tech_results, news_outputs, strict=True
                        )
                    )
                )
            )

        opinions = [
            SymbolOpinions(
                symbol=f.symbol,
                features=f,
                technical=t.output,
                news=n,
                risk=r.output,
                debate=d,
            )
            for f, t, n, r, d in zip(
                candidates, tech_results, news_outputs, risk_results, debates, strict=True
            )
        ]
        decision.opinions = opinions

        pm_res = await self.pm.run(
            PMContext(
                opinions=opinions,
                portfolio=portfolio,
                regime=regime,
                lessons=lessons or [],
            )
        )
        # The PM's stub IS the deterministic arbiter, so a degraded PM still trades.
        if pm_res.status is AgentStatus.DEGRADED:
            pm_res.status = AgentStatus.FALLBACK
        results.append(pm_res)
        decision.proposal = pm_res.output
        decision.results = results
        decision.disagreements = [
            disagreement_score(op, pm_res.output) for op in opinions
        ]
        return decision

    # -- off the hot path ----------------------------------------------------------

    async def reflect(self, ctx: ReflectionContext) -> AgentResult[Lesson]:
        return await self.reflection.run(ctx)

    async def deep_analysis(self, ctx: DeepContext) -> AgentResult[DeepAnalysis]:
        return await self.deep.run(ctx)


__all__ = ["AgentBus", "AgentRuntime", "deterministic_arbitrate", "disagreement_score"]
