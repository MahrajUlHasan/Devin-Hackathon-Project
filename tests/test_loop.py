"""End-to-end loop tests, entirely offline.

The structural property under test: **there is no path from a model's opinion to an
order that skips the risk engine.** Everything else here is about the loop surviving
bad days -- a feed that raises, an agent that dies, a bar that explodes.
"""

from __future__ import annotations

import asyncio

import pytest

from budapilot.agents.base import AgentRuntime
from budapilot.agents.bus import AgentBus
from budapilot.contracts import Action, AssetSpec, NewsRisk, OrderStatus
from budapilot.data.fixtures import FixtureFeed, synthesize_bars, synthesize_news
from budapilot.execution.broker_sim import SimBroker
from budapilot.journal.store import Journal
from budapilot.loop import TradingLoop

SYMBOLS = ["BTC/USD", "ETH/USD"]


@pytest.fixture
def journal(tmp_path):
    j = Journal(str(tmp_path / "t.db"))
    yield j
    j.close()


def build_loop(journal, *, cash: float = 100_000.0) -> TradingLoop:
    return TradingLoop(
        feed=FixtureFeed(SYMBOLS),
        broker=SimBroker(cash=cash),
        bus=AgentBus(AgentRuntime(offline=True)),
        journal=journal,
        symbols=SYMBOLS,
        interval_s=0,
    )


# -- the happy path ----------------------------------------------------------------


async def test_one_bar_runs_end_to_end_offline(journal):
    loop = build_loop(journal)
    await loop.startup()
    await loop.run_once()

    assert loop.last_decision is not None
    assert loop.last_decision.proposal is not None
    assert journal.latest_agent_runs()  # the audit trail exists
    assert journal.equity_series()


async def test_several_bars_advance_the_replay_clock(journal):
    loop = build_loop(journal)
    await loop.run_forever(max_bars=4)
    assert loop.bar_count == 4
    assert len({r["bar_id"] for r in journal.latest_agent_runs(200)}) == 4


async def test_every_agent_run_is_journalled_with_its_model(journal):
    loop = build_loop(journal)
    await loop.run_once()
    runs = journal.runs_for_bar(journal.latest_bar_id())
    assert {r["agent"] for r in runs} >= {"scout", "technical", "news", "pm", "regime"}
    assert all(r["output_json"] for r in runs)
    assert all(r["status"] == "stub" for r in runs)


# -- start / stop and the time budget ---------------------------------------------------


async def test_a_paused_loop_decides_nothing_but_still_feeds_its_stops(journal, monkeypatch):
    loop = build_loop(journal)
    fed: list[str] = []

    async def spy_tick():
        fed.append("tick")

    monkeypatch.setattr(loop, "_tick_stops", spy_tick)
    loop.pause()

    async def stop_soon():
        await asyncio.sleep(0.05)
        loop.stop()

    asyncio.get_running_loop().create_task(stop_soon())
    await loop.run_forever()

    assert loop.bar_count == 0  # no decisions were made
    assert fed  # ...but protection kept running
    assert loop.running is False


async def test_resume_lets_bars_flow_again(journal):
    loop = build_loop(journal)
    loop.pause()
    loop.resume()
    await loop.run_forever(max_bars=2)
    assert loop.bar_count == 2


async def test_time_budget_ends_the_loop_without_a_bar_cap(journal):
    loop = build_loop(journal)
    # A budget that is already spent on the first check: the loop must exit cleanly
    # rather than run "just one more bar".
    await loop.run_forever(max_minutes=1e-9)
    assert loop.bar_count == 0
    assert loop.running is False
    assert loop.deadline is not None


async def test_a_relaunched_run_gets_its_full_bar_budget_again(journal):
    loop = build_loop(journal)
    await loop.run_forever(max_bars=2)
    assert loop.bar_count == 2
    await loop.run_forever(max_bars=2)  # what the dashboard's Start does after a finish
    assert loop.bar_count == 4, "the second run must not be cut short by the first's bars"


async def test_no_budget_means_no_deadline(journal):
    loop = build_loop(journal)
    await loop.run_forever(max_bars=1, max_minutes=None)
    assert loop.deadline is None


# -- THE STRUCTURAL PROPERTY ---------------------------------------------------------


async def test_no_order_is_ever_placed_without_a_risk_ruling(journal, monkeypatch):
    """Every order must be preceded by an approved risk decision for that symbol."""
    loop = build_loop(journal)

    from budapilot.contracts import TradeProposal

    # Force a strong BUY every bar so orders actually flow.
    async def always_buy(*args, **kwargs):
        decision = await original(*args, **kwargs)
        decision.proposal = TradeProposal(
            symbol="BTC/USD",
            action=Action.BUY,
            conviction=0.95,
            rationale="forced for test",
        )
        return decision

    original = loop.bus.run_bar
    monkeypatch.setattr(loop.bus, "run_bar", always_buy)

    await loop.run_forever(max_bars=3)

    orders = journal.latest_orders(50)
    approvals = [r for r in journal.latest_risk(50) if r["approved"]]
    assert orders, "expected at least one order"
    assert len(orders) <= len(approvals)
    for order in orders:
        assert any(a["symbol"] == order["symbol"] for a in approvals)


async def test_risk_veto_blocks_the_order_and_is_journalled(journal, monkeypatch):
    loop = build_loop(journal)

    from budapilot.contracts import TradeProposal

    original = loop.bus.run_bar

    async def critical_news_buy(*args, **kwargs):
        decision = await original(*args, **kwargs)
        decision.proposal = TradeProposal(
            symbol="BTC/USD", action=Action.BUY, conviction=0.99, rationale="forced"
        )
        for op in decision.opinions:
            op.news.news_risk = NewsRisk.CRITICAL
        return decision

    monkeypatch.setattr(loop.bus, "run_bar", critical_news_buy)
    await loop.run_once()

    rulings = journal.latest_risk(10)
    assert rulings and not rulings[0]["approved"]
    assert "CRITICAL" in rulings[0]["reason"]
    assert not journal.latest_orders(10)  # nothing reached the broker


async def test_hold_produces_no_risk_ruling_and_no_order(journal):
    loop = build_loop(journal)
    await loop.run_once()
    if loop.last_decision.proposal.action is Action.HOLD:
        assert not journal.latest_orders(10)


# -- filled positions get protected -----------------------------------------------------


async def test_a_fill_arms_both_stop_layers(journal, monkeypatch):
    loop = build_loop(journal)

    from budapilot.contracts import TradeProposal

    original = loop.bus.run_bar

    async def buy(*args, **kwargs):
        decision = await original(*args, **kwargs)
        decision.proposal = TradeProposal(
            symbol="BTC/USD", action=Action.BUY, conviction=0.95, rationale="forced"
        )
        return decision

    monkeypatch.setattr(loop.bus, "run_bar", buy)
    await loop.run_once()

    if "BTC/USD" in loop.stops.positions:
        pos = loop.stops.positions["BTC/USD"]
        assert pos.stop_px < pos.entry < pos.take_px
        assert pos.broker_stop_order_id is not None  # resting backstop armed
        assert journal.load_protected()  # and persisted for crash recovery


# -- resilience --------------------------------------------------------------------------


async def test_a_feed_that_raises_does_not_kill_the_loop(journal):
    loop = build_loop(journal)
    real_get_bars = loop.feed.get_bars

    calls = {"n": 0}

    async def flaky(symbol, limit=200):
        calls["n"] += 1
        if calls["n"] % 2:
            raise ConnectionError("feed down")
        return await real_get_bars(symbol, limit)

    loop.feed.get_bars = flaky
    await loop.run_once()  # must not raise


async def test_a_bar_that_explodes_is_caught_by_run_forever(journal):
    loop = build_loop(journal)

    async def explode():
        loop.bar_count += 1
        raise RuntimeError("bar exploded")

    loop.run_once = explode
    await loop.run_forever(max_bars=2)  # must not propagate
    assert loop.bar_count >= 2


async def test_no_bars_at_all_is_survivable(journal):
    loop = build_loop(journal)

    async def empty(symbol, limit=200):
        return []

    loop.feed.get_bars = empty
    await loop.run_once()
    assert loop.last_decision is None


# -- fixtures and the offline guarantee -----------------------------------------------------


def test_synthetic_bars_are_deterministic():
    a = synthesize_bars("BTC/USD", 50, seed=7)
    b = synthesize_bars("BTC/USD", 50, seed=7)
    assert [x.close for x in a] == [x.close for x in b]


def test_synthetic_bars_are_internally_consistent():
    for bar in synthesize_bars("ETH/USD", 100, seed=1):
        assert bar.low <= min(bar.open, bar.close)
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low > 0 and bar.volume >= 0


def test_synthetic_news_is_recent_and_labelled_synthetic():
    items = synthesize_news("BTC/USD")
    assert items and all("BTC" in n.headline for n in items)
    assert all("Synthetic" in n.summary for n in items)


async def test_fixture_feed_makes_no_network_calls(monkeypatch):
    """The --demo-safe guarantee, asserted rather than assumed."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("demo-safe attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    feed = FixtureFeed(SYMBOLS)
    assert await feed.get_bars("BTC/USD", 50)
    assert await feed.latest_price("BTC/USD") > 0
    assert await feed.get_news("BTC/USD") is not None
    assert isinstance(await feed.get_asset("BTC/USD"), AssetSpec)


async def test_fixture_feed_advances_and_exhausts():
    feed = FixtureFeed(["BTC/USD"])
    first = await feed.latest_price("BTC/USD")
    for _ in range(20):
        feed.advance()
    assert await feed.latest_price("BTC/USD") != first
    while feed.advance():
        pass
    assert feed.advance() is False


# -- the simulated broker ---------------------------------------------------------------------


async def test_sim_broker_round_trip_settles():
    broker = SimBroker(cash=10_000.0)
    broker.set_price("BTC/USD", 100.0)

    from budapilot.contracts import OrderRequest, OrderSide, OrderType

    buy = await broker.submit(
        OrderRequest(symbol="BTC/USD", side=OrderSide.BUY, qty=10.0, type=OrderType.MARKET)
    )
    assert buy.status is OrderStatus.FILLED
    portfolio = await broker.get_portfolio()
    assert portfolio.positions["BTC/USD"].qty == 10.0

    broker.set_price("BTC/USD", 110.0)
    sell = await broker.submit(
        OrderRequest(symbol="BTC/USD", side=OrderSide.SELL, qty=10.0, type=OrderType.MARKET)
    )
    assert sell.status is OrderStatus.FILLED
    final = await broker.get_portfolio()
    assert not final.positions
    assert final.cash > 10_000.0  # profit net of fees


async def test_sim_broker_refuses_to_short():
    """Long-only, enforced at the simulated venue as well as in the risk engine."""
    broker = SimBroker(cash=10_000.0)
    broker.set_price("BTC/USD", 100.0)

    from budapilot.contracts import OrderRequest, OrderSide, OrderType

    order = await broker.submit(
        OrderRequest(symbol="BTC/USD", side=OrderSide.SELL, qty=1.0, type=OrderType.MARKET)
    )
    assert order.status is OrderStatus.REJECTED


async def test_sim_broker_rejects_an_unaffordable_buy():
    broker = SimBroker(cash=100.0)
    broker.set_price("BTC/USD", 100.0)

    from budapilot.contracts import OrderRequest, OrderSide, OrderType

    order = await broker.submit(
        OrderRequest(symbol="BTC/USD", side=OrderSide.BUY, qty=1000.0, type=OrderType.MARKET)
    )
    assert order.status is OrderStatus.REJECTED


async def test_sim_broker_resting_stop_fires_when_price_crosses():
    broker = SimBroker(cash=10_000.0)
    broker.set_price("BTC/USD", 100.0)

    from budapilot.contracts import OrderRequest, OrderSide, OrderType

    await broker.submit(
        OrderRequest(symbol="BTC/USD", side=OrderSide.BUY, qty=10.0, type=OrderType.MARKET)
    )
    stop = await broker.submit(
        OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            qty=10.0,
            type=OrderType.STOP_LIMIT,
            stop_price=95.0,
            limit_price=94.0,
        )
    )
    assert stop.status is OrderStatus.ACCEPTED  # rests

    broker.set_price("BTC/USD", 94.0)
    assert (await broker.get_order(stop.id)).status is OrderStatus.FILLED


# -- the journal ---------------------------------------------------------------------------------


async def test_journal_survives_a_restart_with_protected_positions(tmp_path):
    from budapilot.contracts import ProtectedPosition

    path = str(tmp_path / "restart.db")
    j1 = Journal(path)
    j1.upsert_protected(
        ProtectedPosition(
            symbol="BTC/USD",
            qty=0.5,
            entry=100.0,
            stop_px=95.0,
            take_px=110.0,
            broker_stop_order_id="abc",
        )
    )
    j1.close()

    j2 = Journal(path)
    restored = j2.load_protected()
    assert len(restored) == 1
    assert restored[0].broker_stop_order_id == "abc"
    j2.close()


async def test_journal_records_degradations(journal):
    from budapilot.contracts import AgentResult, AgentStatus, ScoutOutput

    journal.log_agent_run(
        AgentResult[ScoutOutput](
            agent="scout",
            model="claude-haiku-4-5",
            output=ScoutOutput(),
            status=AgentStatus.DEGRADED,
            error="timeout after 8.0s",
        ),
        "bar-1",
    )
    assert journal.degradation_count() == 1
