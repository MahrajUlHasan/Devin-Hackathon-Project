"""Agent tests.

The property under test throughout: **an agent never raises.** Whatever the API does --
time out, 500, return garbage -- the agent returns a valid structured output with a
degraded status and the loop keeps trading. A dead news feed must not stop the desk.
"""

from __future__ import annotations

import asyncio

import pytest

from budapilot.agents.arbiter import deterministic_arbitrate, score_opinion
from budapilot.agents.base import Agent, AgentRuntime
from budapilot.agents.bus import AgentBus
from budapilot.agents.news import NewsContext, time_decayed_sentiment
from budapilot.agents.regime import RegimeContext, btc_dominance_drift
from budapilot.agents.scout import ScoutContext, rank_symbols
from budapilot.agents.technical import TechnicalContext
from budapilot.config import MIN_CONVICTION
from budapilot.contracts import (
    Action,
    AgentStatus,
    FeatureBundle,
    NewsItem,
    NewsRisk,
    PortfolioState,
    ScoredHeadline,
    utcnow,
)
from tests.conftest import make_opinion

OFFLINE = AgentRuntime(offline=True)


def bundle(symbol: str, *, trend: float = 1.0, atr: float = 1.0, **kw) -> FeatureBundle:
    return FeatureBundle(
        symbol=symbol,
        ts=utcnow(),
        close=kw.pop("close", 100.0),
        atr_14=atr,
        atr_pct=kw.pop("atr_pct", 0.01),
        trend_score=trend,
        **kw,
    )


# -- offline behaviour ------------------------------------------------------------------


async def test_every_agent_returns_stub_output_offline():
    bus = AgentBus(OFFLINE)
    features = [bundle("BTC/USD", trend=2.0), bundle("ETH/USD", trend=0.5)]

    decision = await bus.run_bar(
        bar_id="b1",
        features=features,
        news_by_symbol={},
        portfolio=PortfolioState(equity=100_000, cash=100_000),
    )

    assert decision.proposal is not None
    assert all(r.status is AgentStatus.STUB for r in decision.results)
    assert {r.agent for r in decision.results} >= {"scout", "technical", "news", "pm"}


async def test_run_bar_with_no_tradeable_candidates_holds():
    bus = AgentBus(OFFLINE)
    broken = FeatureBundle(symbol="BTC/USD", ts=utcnow(), close=100.0)  # no ATR
    decision = await bus.run_bar(
        bar_id="b1",
        features=[broken],
        news_by_symbol={},
        portfolio=PortfolioState(equity=100_000, cash=100_000),
    )
    assert decision.proposal.action is Action.HOLD


async def test_vetoed_candidates_are_dropped():
    bus = AgentBus(OFFLINE)
    features = [bundle("BTC/USD", trend=2.0), bundle("ETH/USD", trend=1.0)]
    decision = await bus.run_bar(
        bar_id="b1",
        features=features,
        news_by_symbol={},
        portfolio=PortfolioState(equity=100_000, cash=100_000),
    )
    # The stub scout vetoes nothing, so both survive.
    assert len(decision.opinions) == 2


# -- failure handling ---------------------------------------------------------------------


class Boom(Agent):
    name = "technical"
    system = "x"

    def __init__(self, runtime, exc: Exception):
        from budapilot.contracts import TechnicalOutput

        self.output_model = TechnicalOutput
        super().__init__(runtime)
        self._exc = exc

    def user_prompt(self, ctx):
        return "x"

    def stub(self, ctx):
        from budapilot.contracts import TechnicalOutput

        return TechnicalOutput(
            symbol=ctx.symbol,
            direction=Action.HOLD,
            conviction=0.5,
            horizon_bars=1,
            rationale="stub",
        )

    async def _call(self, ctx):
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("api exploded"),
        ValueError("schema violation"),
        ConnectionError("network gone"),
    ],
)
async def test_agent_degrades_instead_of_raising(exc):
    runtime = AgentRuntime(offline=False, api_key="sk-fake-not-used")
    agent = Boom(runtime, exc)
    result = await agent.run(TechnicalContext(features=bundle("BTC/USD")))

    assert result.status is AgentStatus.DEGRADED
    assert result.error and type(exc).__name__ in result.error
    assert result.output.direction is Action.HOLD  # usable output, not None


async def test_agent_degrades_on_timeout():
    class Slow(Boom):
        async def _call(self, ctx):
            await asyncio.sleep(5)

    runtime = AgentRuntime(offline=False, api_key="sk-fake-not-used")
    agent = Slow(runtime, RuntimeError())
    agent.timeout_s = 0.01

    result = await agent.run(TechnicalContext(features=bundle("BTC/USD")))
    assert result.status is AgentStatus.DEGRADED
    assert "timeout" in result.error


async def test_runtime_without_a_key_is_offline():
    assert AgentRuntime(api_key="").offline is True


# -- the PM fallback is a quant, not a corpse -----------------------------------------------


async def test_degraded_pm_is_marked_fallback_and_still_trades():
    bus = AgentBus(AgentRuntime(offline=False, api_key="sk-fake-not-used"))

    # Force every agent to fail; the PM's stub is the deterministic arbiter.
    async def explode(ctx):
        raise RuntimeError("down")

    for agent in (bus.scout, bus.technical, bus.news, bus.regime, bus.risk_analyst, bus.pm):
        agent._call = explode

    decision = await bus.run_bar(
        bar_id="b1",
        features=[bundle("BTC/USD", trend=2.0)],
        news_by_symbol={},
        portfolio=PortfolioState(equity=100_000, cash=100_000),
    )

    pm_result = next(r for r in decision.results if r.agent == "pm")
    assert pm_result.status is AgentStatus.FALLBACK
    assert decision.proposal is not None


def test_arbiter_holds_below_the_conviction_floor():
    p = deterministic_arbitrate([make_opinion(conviction=0.3)])
    assert p.action is Action.HOLD
    assert "below" in p.rationale


def test_arbiter_acts_above_the_floor():
    p = deterministic_arbitrate([make_opinion(conviction=0.9, sentiment=0.5)])
    assert p.action is Action.BUY
    assert p.conviction >= MIN_CONVICTION
    assert p.overrode  # always names that it stood in for the PM


def test_arbiter_respects_critical_news():
    p = deterministic_arbitrate(
        [make_opinion(conviction=0.95, news_risk=NewsRisk.CRITICAL)]
    )
    assert p.action is Action.HOLD


def test_arbiter_handles_no_opinions():
    assert deterministic_arbitrate([]).action is Action.HOLD


def test_arbiter_skips_a_symbol_already_at_its_cap():
    """Otherwise it proposes the same capped symbol every bar and collects rejections."""
    from budapilot.config import MAX_POSITION_PCT
    from tests.conftest import make_position

    portfolio = PortfolioState(equity=100_000.0, cash=50_000.0)
    portfolio.positions["BTC/USD"] = make_position(
        "BTC/USD", qty=MAX_POSITION_PCT * 100_000.0 / 100.0, entry=100.0
    )
    opinions = [
        make_opinion("BTC/USD", conviction=0.95),
        make_opinion("ETH/USD", conviction=0.70),
    ]
    assert deterministic_arbitrate(opinions, portfolio).symbol == "ETH/USD"


def test_arbiter_holds_when_everything_is_capped():
    from budapilot.config import MAX_POSITION_PCT
    from tests.conftest import make_position

    portfolio = PortfolioState(equity=100_000.0, cash=50_000.0)
    portfolio.positions["BTC/USD"] = make_position(
        "BTC/USD", qty=MAX_POSITION_PCT * 100_000.0 / 100.0, entry=100.0
    )
    p = deterministic_arbitrate([make_opinion("BTC/USD", conviction=0.95)], portfolio)
    assert p.action is Action.HOLD and "position cap" in p.rationale


def test_arbiter_still_allows_a_sell_on_a_capped_symbol():
    """Capped means no more buying, not that we cannot get out."""
    from budapilot.config import MAX_POSITION_PCT
    from tests.conftest import make_position

    portfolio = PortfolioState(equity=100_000.0, cash=50_000.0)
    portfolio.positions["BTC/USD"] = make_position(
        "BTC/USD", qty=MAX_POSITION_PCT * 100_000.0 / 100.0, entry=100.0
    )
    p = deterministic_arbitrate(
        [make_opinion("BTC/USD", direction=Action.SELL, conviction=0.9)], portfolio
    )
    assert p.action is Action.SELL


def test_pm_prompt_warns_when_a_candidate_is_capped():
    from budapilot.agents.pm import PMContext
    from budapilot.config import MAX_POSITION_PCT
    from tests.conftest import make_position

    portfolio = PortfolioState(equity=100_000.0, cash=50_000.0)
    portfolio.positions["BTC/USD"] = make_position(
        "BTC/USD", qty=MAX_POSITION_PCT * 100_000.0 / 100.0, entry=100.0
    )
    prompt = AgentBus(OFFLINE).pm.user_prompt(
        PMContext(opinions=[make_opinion("BTC/USD")], portfolio=portfolio)
    )
    assert "AT THE 10% CAP" in prompt


def test_arbiter_zero_size_multiplier_kills_the_score():
    assert score_opinion(make_opinion(conviction=1.0, size_multiplier=0.0)) == 0.0


def test_arbiter_hold_opinion_scores_zero():
    assert score_opinion(make_opinion(direction=Action.HOLD)) == 0.0


def test_arbiter_picks_the_best_candidate():
    p = deterministic_arbitrate(
        [
            make_opinion("BTC/USD", conviction=0.65),
            make_opinion("ETH/USD", conviction=0.95),
        ]
    )
    assert p.symbol == "ETH/USD"


def test_arbiter_sentiment_penalises_a_buy_into_bad_news():
    good = score_opinion(make_opinion(conviction=0.8, sentiment=0.8))
    bad = score_opinion(make_opinion(conviction=0.8, sentiment=-0.8))
    assert good > bad


# -- deterministic helpers ------------------------------------------------------------------


def test_rank_symbols_orders_by_trend_and_drops_untradeable():
    ranked = rank_symbols(
        [
            bundle("A/USD", trend=0.5),
            bundle("B/USD", trend=3.0),
            FeatureBundle(symbol="C/USD", ts=utcnow(), close=1.0),  # no ATR
        ],
        top_n=5,
    )
    assert [f.symbol for f in ranked] == ["B/USD", "A/USD"]


def test_rank_symbols_respects_top_n():
    assert len(rank_symbols([bundle(f"{i}/USD") for i in range(10)], top_n=3)) == 3


def test_time_decayed_sentiment_weights_recent_news_more():
    now = utcnow()
    from datetime import timedelta

    items = [
        NewsItem(id="new", ts=now, headline="h1"),
        NewsItem(id="old", ts=now - timedelta(hours=48), headline="h2"),
    ]
    scored = [
        ScoredHeadline(headline="h1", score=1.0),
        ScoredHeadline(headline="h2", score=-1.0),
    ]
    assert time_decayed_sentiment(items, scored) > 0.5


def test_time_decayed_sentiment_empty():
    assert time_decayed_sentiment([], []) == 0.0


def test_btc_dominance_drift():
    features = [
        bundle("BTC/USD", ret_20=0.05),
        bundle("ETH/USD", ret_20=0.01),
        bundle("SOL/USD", ret_20=0.01),
    ]
    assert btc_dominance_drift(features) == pytest.approx(0.04)


def test_btc_dominance_drift_without_btc():
    assert btc_dominance_drift([bundle("ETH/USD", ret_20=0.01)]) == 0.0


# -- prompts render ---------------------------------------------------------------------------


def test_prompts_render_without_error():
    """Cheap guard: a KeyError in an f-string would only surface in production."""
    bus = AgentBus(OFFLINE)
    features = [bundle("BTC/USD", rsi_14=60.0), bundle("ETH/USD", rsi_14=40.0)]
    portfolio = PortfolioState(equity=100_000, cash=100_000)

    assert bus.scout.user_prompt(ScoutContext(features=features))
    assert bus.technical.user_prompt(TechnicalContext(features=features[0]))
    assert bus.regime.user_prompt(RegimeContext(features=features))
    assert bus.news.user_prompt(NewsContext(symbol="BTC/USD"))

    from budapilot.agents.pm import PMContext

    prompt = bus.pm.user_prompt(
        PMContext(opinions=[make_opinion()], portfolio=portfolio)
    )
    assert "PORTFOLIO" in prompt and "CANDIDATES" in prompt


def test_pm_prompt_includes_lessons():
    from budapilot.agents.pm import PMContext
    from budapilot.contracts import ExitReason, Lesson

    bus = AgentBus(OFFLINE)
    prompt = bus.pm.user_prompt(
        PMContext(
            opinions=[make_opinion()],
            portfolio=PortfolioState(equity=1000, cash=1000),
            lessons=[
                Lesson(
                    symbol="BTC/USD",
                    outcome_pct=-1.2,
                    exit_reason=ExitReason.STOP,
                    lesson="Do not chase a vertical move.",
                )
            ],
        )
    )
    assert "Do not chase a vertical move." in prompt


def test_model_allocation_is_what_the_plan_says():
    """Guards against a quiet model downgrade on the call that becomes an order."""
    bus = AgentBus(OFFLINE)
    assert bus.pm.model == "claude-opus-5"
    assert bus.deep.model == "claude-opus-5"
    assert bus.deep.effort == "high"
    assert bus.scout.model == "claude-haiku-4-5"
    assert bus.regime.model == "claude-haiku-4-5"
    assert bus.headline_scorer.model == "claude-haiku-4-5"
    assert bus.technical.model == "claude-sonnet-5"
    assert bus.risk_analyst.model == "claude-sonnet-5"
    assert bus.news.model == "claude-sonnet-5"
