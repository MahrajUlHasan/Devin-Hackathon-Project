"""Bull/Bear debate (A9/A10) and the disagreement metric.

The debate's value depends on the two advocates being genuinely independent in round 1.
If the bear can see the bull's case before writing its own, it anchors, and what looks
like adversarial review is really one argument plus a reaction to it. That property is
asserted here rather than assumed.
"""

from __future__ import annotations

import pytest

from budapilot.agents.base import AgentRuntime
from budapilot.agents.bus import AgentBus, disagreement_score
from budapilot.agents.debate import (
    BearAgent,
    BearRebuttalAgent,
    BullAgent,
    BullRebuttalAgent,
    DebateContext,
)
from budapilot.contracts import (
    Action,
    AgentStatus,
    DebateCase,
    FeatureBundle,
    NewsOutput,
    NewsRisk,
    PortfolioState,
    TechnicalOutput,
    utcnow,
)
from tests.conftest import make_opinion

OFFLINE = AgentRuntime(offline=True)


def ctx(trend: float = 1.5, *, opposing: DebateCase | None = None) -> DebateContext:
    f = FeatureBundle(
        symbol="BTC/USD",
        ts=utcnow(),
        close=100.0,
        atr_14=1.0,
        atr_pct=0.01,
        rsi_14=60.0,
        trend_score=trend,
    )
    return DebateContext(
        features=f,
        technical=TechnicalOutput(
            symbol="BTC/USD",
            direction=Action.BUY,
            conviction=0.7,
            horizon_bars=12,
            rationale="uptrend",
        ),
        news=NewsOutput(symbol="BTC/USD", sentiment=0.3, news_risk=NewsRisk.LOW),
        opposing=opposing,
    )


# -- the advocates ---------------------------------------------------------------------


def test_models_are_sonnet_for_both_sides():
    assert BullAgent(OFFLINE).model == "claude-sonnet-5"
    assert BearAgent(OFFLINE).model == "claude-sonnet-5"


def test_each_side_is_told_which_side_it_is_on():
    assert "You are the BULL" in BullAgent(OFFLINE).system
    assert "You are the BEAR" in BearAgent(OFFLINE).system


def test_both_advocates_are_required_to_concede_something():
    for agent in (BullAgent(OFFLINE), BearAgent(OFFLINE)):
        assert "conceded" in agent.system
        assert "mandatory" in agent.system


def test_confidence_is_defined_as_case_strength_not_advocacy_strength():
    """Otherwise every advocate returns 0.9 and the field carries no information."""
    system = BullAgent(OFFLINE).system
    assert "NOT how" in system and "strongly you are arguing" in system


def test_both_advocates_are_told_the_desk_is_long_only():
    for agent in (BullAgent(OFFLINE), BearAgent(OFFLINE)):
        assert "LONG ONLY" in agent.system


def test_bear_may_argue_merely_no_edge():
    """A bear case should not have to prove disaster to be useful."""
    assert "the edge is not there" in BearAgent(OFFLINE).system


def test_rebuttal_agents_add_the_rebuttal_instruction():
    assert "rebuttal" in BullRebuttalAgent(OFFLINE).system
    assert "theatre" in BearRebuttalAgent(OFFLINE).system
    assert "rebuttal" not in BullAgent(OFFLINE).system.lower().split("conceded")[0]


# -- stub behaviour --------------------------------------------------------------------


async def test_stubs_produce_an_actual_disagreement():
    """Offline the two sides must still differ, or the demo shows two blank cards."""
    bull = await BullAgent(OFFLINE).run(ctx(trend=2.0))
    bear = await BearAgent(OFFLINE).run(ctx(trend=2.0))

    assert bull.output.side == "BULL" and bear.output.side == "BEAR"
    assert bull.output.confidence > bear.output.confidence  # bullish tape


async def test_stub_confidence_flips_with_the_trend():
    bull_up = (await BullAgent(OFFLINE).run(ctx(trend=2.0))).output.confidence
    bull_down = (await BullAgent(OFFLINE).run(ctx(trend=-2.0))).output.confidence
    assert bull_up > bull_down


async def test_advocates_never_raise():
    class Boom(BullAgent):
        async def _call(self, c):
            raise RuntimeError("model down")

    agent = Boom(AgentRuntime(offline=False, api_key="sk-fake"))
    result = await agent.run(ctx())
    assert result.status is AgentStatus.DEGRADED
    assert result.output.side == "BULL"


# -- prompt construction -----------------------------------------------------------------


def test_round_one_prompt_has_no_opposing_case():
    """Independence in round 1: whoever goes first would otherwise anchor the other."""
    prompt = BullAgent(OFFLINE).user_prompt(ctx())
    assert "BEAR CASE" not in prompt
    assert "Make the bull case." in prompt


def test_round_two_prompt_includes_the_opposing_case():
    opposing = DebateCase(
        symbol="BTC/USD",
        side="BEAR",
        claims=["Volume is thin"],
        strongest_point="The move is unconfirmed by volume.",
        conceded="The trend is genuinely intact.",
        confidence=0.6,
    )
    prompt = BullRebuttalAgent(OFFLINE).user_prompt(ctx(opposing=opposing))
    assert "THE BEAR CASE" in prompt
    assert "The move is unconfirmed by volume." in prompt
    assert "they conceded: The trend is genuinely intact." in prompt
    assert "Rebut it." in prompt


# -- bus wiring ------------------------------------------------------------------------------


async def test_debate_is_off_by_default():
    bus = AgentBus(OFFLINE)
    assert bus.enable_debate is False

    decision = await bus.run_bar(
        bar_id="b1",
        features=[ctx().features],
        news_by_symbol={},
        portfolio=PortfolioState(equity=100_000, cash=100_000),
    )
    assert {r.agent for r in decision.results}.isdisjoint({"bull", "bear"})
    assert all(op.debate is None for op in decision.opinions)


async def test_debate_runs_when_enabled_and_reaches_the_pm():
    bus = AgentBus(OFFLINE, enable_debate=True)

    decision = await bus.run_bar(
        bar_id="b1",
        features=[ctx(trend=2.0).features],
        news_by_symbol={},
        portfolio=PortfolioState(equity=100_000, cash=100_000),
    )

    agents = {r.agent for r in decision.results}
    assert {"bull", "bear"} <= agents
    assert decision.opinions[0].debate is not None
    assert decision.opinions[0].debate.bull.side == "BULL"


def test_pm_prompt_renders_the_debate():
    from budapilot.agents.pm import PMContext
    from budapilot.contracts import Debate

    op = make_opinion("BTC/USD")
    op.debate = Debate(
        symbol="BTC/USD",
        bull=DebateCase(
            symbol="BTC/USD",
            side="BULL",
            claims=["EMA cross is fresh"],
            strongest_point="Momentum just turned.",
            conceded="Volume has not confirmed.",
            confidence=0.7,
        ),
        bear=DebateCase(
            symbol="BTC/USD",
            side="BEAR",
            claims=["Thin volume"],
            strongest_point="No volume behind the move.",
            conceded="The trend is intact.",
            confidence=0.5,
        ),
    )
    prompt = AgentBus(OFFLINE).pm.user_prompt(
        PMContext(opinions=[op], portfolio=PortfolioState(equity=1000, cash=1000))
    )
    assert "BULL (confidence 0.70)" in prompt
    assert "BEAR (confidence 0.50)" in prompt
    assert "concedes: Volume has not confirmed." in prompt


def test_pm_system_prompt_warns_that_advocates_are_not_neutral():
    system = AgentBus(OFFLINE).pm.system
    assert "assigned a side" in system
    assert "conceded" in system


# -- the disagreement metric -------------------------------------------------------------------


def test_unanimous_desk_scores_near_zero():
    from budapilot.contracts import TradeProposal

    op = make_opinion(conviction=0.8, sentiment=0.8, size_multiplier=1.0)
    proposal = TradeProposal(
        symbol=op.symbol, action=Action.BUY, conviction=0.8, rationale="agreed"
    )
    assert disagreement_score(op, proposal).score < 0.1


def test_technical_against_news_raises_the_score():
    agree = disagreement_score(make_opinion(conviction=0.9, sentiment=0.9), None)
    clash = disagreement_score(make_opinion(conviction=0.9, sentiment=-0.9), None)
    assert clash.score > agree.score


def test_a_risk_analyst_cutting_size_raises_the_score():
    full = disagreement_score(make_opinion(conviction=0.9, size_multiplier=1.0), None)
    cut = disagreement_score(make_opinion(conviction=0.9, size_multiplier=0.1), None)
    assert cut.score > full.score


def test_pm_overriding_the_technical_raises_the_score():
    from budapilot.contracts import TradeProposal

    op = make_opinion(conviction=0.9, sentiment=0.0)
    agreeing = TradeProposal(
        symbol=op.symbol, action=Action.BUY, conviction=0.9, rationale="x"
    )
    overruling = TradeProposal(
        symbol=op.symbol,
        action=Action.HOLD,
        conviction=0.0,
        rationale="x",
        overrode=["technical: too stretched"],
    )
    assert (
        disagreement_score(op, overruling).score
        > disagreement_score(op, agreeing).score
    )
    assert disagreement_score(op, overruling).overrode_count == 1


def test_score_is_bounded():
    for conv in (0.0, 0.5, 1.0):
        for sent in (-1.0, 0.0, 1.0):
            for mult in (0.0, 0.5, 1.0):
                d = disagreement_score(
                    make_opinion(conviction=conv, sentiment=sent, size_multiplier=mult),
                    None,
                )
                assert 0.0 <= d.score <= 1.0


def test_hold_technical_has_no_signed_direction():
    d = disagreement_score(make_opinion(direction=Action.HOLD), None)
    assert d.technical_signed == 0.0


def test_sell_is_signed_negative():
    d = disagreement_score(make_opinion(direction=Action.SELL, conviction=0.8), None)
    assert d.technical_signed == pytest.approx(-0.8)


def test_proposal_for_another_symbol_is_ignored():
    from budapilot.contracts import TradeProposal

    op = make_opinion("BTC/USD", conviction=0.9)
    other = TradeProposal(
        symbol="ETH/USD", action=Action.BUY, conviction=0.2, rationale="x",
        overrode=["a", "b"],
    )
    assert disagreement_score(op, other).overrode_count == 0


# -- journalling -------------------------------------------------------------------------------


def test_offline_pseudo_scores_are_stable_across_processes():
    """hash() is per-process randomised; --demo-safe must reproduce exactly."""
    import subprocess
    import sys

    code = (
        "from budapilot.agents.news import _stable_pseudo_score;"
        "print(_stable_pseudo_score('Bitcoin ETF inflows hit a record'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"pseudo-score varied with PYTHONHASHSEED: {runs}"


def test_offline_news_produces_a_spread_of_sentiment():
    """Uniform 0.0 would make the disagreement heatmap permanently blank."""
    from budapilot.agents.news import _stable_pseudo_score

    scores = [
        _stable_pseudo_score(f"Some crypto headline number {i}") for i in range(30)
    ]
    assert min(scores) < -0.2 and max(scores) > 0.2
    assert len({round(s, 1) for s in scores}) > 5


def test_offline_news_risk_never_reaches_critical():
    """CRITICAL is a hard veto on real money; no offline heuristic has earned it."""
    from budapilot.agents.news import NewsAgent, NewsContext
    from budapilot.contracts import NewsItem, ScoredHeadline

    agent = NewsAgent(OFFLINE)
    items = [NewsItem(id=str(i), ts=utcnow(), headline=f"h{i}") for i in range(5)]
    scored = [ScoredHeadline(headline=f"h{i}", score=-1.0) for i in range(5)]
    out = agent.stub(NewsContext(symbol="BTC/USD", items=items, scored=scored))
    assert out.sentiment < -0.9
    assert out.news_risk is NewsRisk.HIGH  # elevated, but not a veto


async def test_disagreements_are_journalled_and_form_a_grid(tmp_path):
    from budapilot.journal.store import Journal

    journal = Journal(str(tmp_path / "d.db"))
    bus = AgentBus(OFFLINE)

    for i in range(3):
        decision = await bus.run_bar(
            bar_id=f"bar-{i}",
            features=[ctx(trend=2.0).features],
            news_by_symbol={},
            portfolio=PortfolioState(equity=100_000, cash=100_000),
        )
        journal.log_decision(decision)

    grid = journal.disagreement_grid(24)
    assert len(grid) == 3
    assert {r["symbol"] for r in grid} == {"BTC/USD"}
    assert all(0.0 <= r["score"] <= 1.0 for r in grid)
    journal.close()
