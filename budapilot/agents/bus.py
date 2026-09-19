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
from budapilot.agents.deep import DeepAgent, DeepContext
from budapilot.agents.news import HeadlineScorer, NewsAgent, NewsContext
from budapilot.agents.pm import PMAgent, PMContext
from budapilot.agents.reflection import ReflectionAgent, ReflectionContext
from budapilot.agents.regime import RegimeAgent, RegimeContext
from budapilot.agents.risk_analyst import RiskAnalystAgent, RiskAnalystContext
from budapilot.agents.scout import ScoutAgent, ScoutContext
from budapilot.agents.technical import TechnicalAgent, TechnicalContext
from budapilot.config import MAX_CANDIDATES, NEWS_TTL_S, REGIME_TTL_S
from budapilot.contracts import (
    Action,
    AgentResult,
    AgentStatus,
    BarDecision,
    DeepAnalysis,
    FeatureBundle,
    Lesson,
    NewsItem,
    NewsOutput,
    PortfolioState,
    RegimeOutput,
    SymbolOpinions,
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


class AgentBus:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self.scout = ScoutAgent(runtime)
        self.technical = TechnicalAgent(runtime)
        self.headline_scorer = HeadlineScorer(runtime)
        self.news = NewsAgent(runtime)
        self.regime = RegimeAgent(runtime)
        self.risk_analyst = RiskAnalystAgent(runtime)
        self.pm = PMAgent(runtime)
        self.reflection = ReflectionAgent(runtime)
        self.deep = DeepAgent(runtime)

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

        opinions = [
            SymbolOpinions(
                symbol=f.symbol, features=f, technical=t.output, news=n, risk=r.output
            )
            for f, t, n, r in zip(
                candidates, tech_results, news_outputs, risk_results, strict=True
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
        return decision

    # -- off the hot path ----------------------------------------------------------

    async def reflect(self, ctx: ReflectionContext) -> AgentResult[Lesson]:
        return await self.reflection.run(ctx)

    async def deep_analysis(self, ctx: DeepContext) -> AgentResult[DeepAnalysis]:
        return await self.deep.run(ctx)


__all__ = ["AgentBus", "AgentRuntime", "deterministic_arbitrate"]
