"""Dashboard API, entirely offline.

Two properties matter here. First, the dashboard is scoped to *this* session: a
journal that already holds last week's failed runs against another vendor must not
surface them as today's errors. Second, the only write the dashboard can make is the
watchlist, and it is validated against the universe -- no route here reaches the broker.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from budapilot.agents.base import AgentRuntime
from budapilot.agents.bus import AgentBus
from budapilot.config import UNIVERSE
from budapilot.contracts import (
    Action,
    AgentResult,
    AgentStatus,
    TechnicalOutput,
)
from budapilot.data.fixtures import FixtureFeed
from budapilot.execution.broker_sim import SimBroker
from budapilot.journal.store import Journal
from budapilot.loop import TradingLoop
from budapilot.web.app import create_app

SYMBOLS = ["BTC/USD", "ETH/USD"]


@pytest.fixture
def journal(tmp_path):
    j = Journal(str(tmp_path / "t.db"))
    yield j
    j.close()


def build_loop(journal) -> TradingLoop:
    return TradingLoop(
        feed=FixtureFeed(SYMBOLS),
        broker=SimBroker(cash=100_000.0),
        bus=AgentBus(AgentRuntime(offline=True)),
        journal=journal,
        symbols=list(SYMBOLS),
        interval_s=0,
    )


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def _stale_failure(symbol: str = "BTC/USD") -> AgentResult:
    return AgentResult[TechnicalOutput](
        agent="technical",
        model="gemini-flash-latest",
        output=TechnicalOutput(
            symbol=symbol, direction=Action.HOLD, conviction=0.5, horizon_bars=1, rationale="x"
        ),
        status=AgentStatus.DEGRADED,
        error="ClientError: 429 RESOURCE_EXHAUSTED",
        symbol=symbol,
    )


# -- session scoping -----------------------------------------------------------------


async def test_errors_from_a_previous_process_do_not_appear_as_todays(journal):
    """The bug this pins: stale Gemini 429s showing on a Claude run's dashboard."""
    journal.log_agent_run(_stale_failure(), "old-bar-00001")
    journal.log_agent_run(_stale_failure("ETH/USD"), "old-bar-00001")

    loop = build_loop(journal)
    app = create_app(journal, loop)  # started_at is *after* the stale rows

    async with _client(app) as c:
        snap = (await c.get("/api/snapshot")).json()

    assert snap["bar_id"] is None  # the old bar is history, not "latest"
    assert snap["degradations"] == 0
    assert all(not card["error"] for g in snap["panel"] for card in g["cards"])


async def test_after_a_bar_the_panel_shows_this_sessions_runs(journal):
    journal.log_agent_run(_stale_failure(), "old-bar-00001")
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.startup()
    await loop.run_once()

    async with _client(app) as c:
        snap = (await c.get("/api/snapshot")).json()

    assert snap["bar_id"] == journal.latest_bar_id()
    keys = {g["key"] for g in snap["panel"]}
    assert {"regime", "scout", "technical", "news", "risk_analyst", "pm"} <= keys
    assert "bull" not in keys  # debate off -> the advocates are absent, not "failed"
    tech = next(g for g in snap["panel"] if g["key"] == "technical")
    assert tech["cards"] and all(c["status"] == "stub" for c in tech["cards"])
    assert all("gemini" not in (c["model"] or "") for c in tech["cards"])


async def test_cached_agents_show_their_last_value_not_an_empty_card(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.startup()
    await loop.run_once()
    await loop.run_once()  # regime is cached 30m -> does not run on bar 2

    async with _client(app) as c:
        snap = (await c.get("/api/snapshot")).json()

    regime = next(g for g in snap["panel"] if g["key"] == "regime")
    assert regime["cards"], "a cached agent must still show its last output"
    assert regime["cards"][0]["cached"] is True
    assert regime["note"] == "cached 30m"


async def test_snapshot_carries_provider_and_analytics(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.startup()
    await loop.run_once()

    async with _client(app) as c:
        snap = (await c.get("/api/snapshot")).json()

    assert snap["provider"]["stubbed"] is True
    assert snap["provider"]["broker"] == "SimBroker"
    assert snap["watchlist"] == SYMBOLS
    assert snap["universe"] == UNIVERSE
    a = snap["analytics"]
    assert {r["agent"] for r in a["agents"]} >= {"scout", "pm"}
    assert sum(x["n"] for x in a["actions"]) == 1
    assert a["errors"] == []  # stubs never fail


async def test_analytics_groups_failure_reasons_so_the_dashboard_can_say_why(journal):
    """A wall of red 'degraded' bars is useless without the reason next to it."""
    loop = build_loop(journal)
    app = create_app(journal, loop)
    for sym in ("BTC/USD", "ETH/USD", "SOL/USD"):
        journal.log_agent_run(_stale_failure(sym), "bar-1")

    async with _client(app) as c:
        errs = (await c.get("/api/snapshot")).json()["analytics"]["errors"]

    assert len(errs) == 1
    assert errs[0]["n"] == 3
    assert "429" in errs[0]["error"] and errs[0]["agents"] == "technical"


def test_error_kinds_ignore_request_ids_and_retry_hints():
    """105 identical billing failures must be one group of 105, not 105 groups of one."""
    from budapilot.journal.store import _error_kind

    a = ("BadRequestError: Error code: 400 - {'type': 'error', 'error': {'message': "
         "'Your credit balance is too low'}, 'request_id': 'req_011CfCyi2qTeZmL7Tn8fFWnb'}")
    b = a.replace("req_011CfCyi2qTeZmL7Tn8fFWnb", "req_011CfCyhzJero2ewpCRYdNHS")
    assert _error_kind(a) == _error_kind(b)

    g1 = "429 RESOURCE_EXHAUSTED ... Please retry in 23.588662194s. 'retryDelay': '23s'"
    g2 = "429 RESOURCE_EXHAUSTED ... Please retry in 0.546s. 'retryDelay': '0s'"
    assert _error_kind(g1) == _error_kind(g2)

    assert _error_kind("timeout after 10.0s") == _error_kind("timeout after 8.0s") == "timeout"


# -- market tape and suggestions -----------------------------------------------------


async def test_market_returns_a_row_per_watched_symbol_with_live_price(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.startup()
    await loop.run_once()

    async with _client(app) as c:
        m = (await c.get("/api/market")).json()

    assert [r["symbol"] for r in m["symbols"]] == SYMBOLS
    for r in m["symbols"]:
        assert r["price"] > 0
        assert r["bars"] > 0 and len(r["spark"]) > 1
        assert r["features"]["tradeable"] in (True, False)


async def test_suggest_ranks_the_universe_and_recommends_positive_trend_only(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)

    async with _client(app) as c:
        s = (await c.get("/api/suggest")).json()

    ranked = s["ranked"]
    assert ranked, "synthetic bars exist for every universe symbol"
    scores = [r["trend_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(r["why"] for r in ranked)
    for sym in s["recommended"]:
        row = next(r for r in ranked if r["symbol"] == sym)
        assert row["trend_score"] > 0 and row["tradeable"]
    assert len(s["recommended"]) <= 3


# -- the one write -------------------------------------------------------------------


async def test_watchlist_update_applies_to_the_loop_and_is_validated(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)

    async with _client(app) as c:
        ok = await c.post("/api/watchlist", json={"symbols": ["sol/usd", "BTC/USD", "SOL/USD"]})
        assert ok.status_code == 200
        assert ok.json()["symbols"] == ["SOL/USD", "BTC/USD"]  # upper-cased, de-duplicated
        assert loop.symbols == ["SOL/USD", "BTC/USD"]

        bad = await c.post("/api/watchlist", json={"symbols": ["TSLA/USD"]})
        assert bad.status_code == 400 and "not in universe" in bad.json()["error"]

        empty = await c.post("/api/watchlist", json={"symbols": []})
        assert empty.status_code == 400

        too_many = await c.post("/api/watchlist", json={"symbols": UNIVERSE})
        assert too_many.status_code == 400

        # A rejected write must not have touched the loop.
        assert loop.symbols == ["SOL/USD", "BTC/USD"]


async def test_watchlist_change_is_what_the_next_bar_trades(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.startup()

    async with _client(app) as c:
        await c.post("/api/watchlist", json={"symbols": ["ETH/USD"]})
    await loop.run_once()

    scout = next(r for r in loop.last_decision.results if r.agent == "scout")
    assert {c.symbol for c in scout.output.candidates} == {"ETH/USD"}


# -- start / stop --------------------------------------------------------------------


async def test_stop_pauses_decisions_and_start_resumes_them(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)
    loop.running = True  # as run_forever would have set it

    async with _client(app) as c:
        r = await c.post("/api/loop/stop")
        assert r.json() == {"running": True, "paused": True, "relaunched": False}
        assert (await c.get("/api/snapshot")).json()["paused"] is True

        r = await c.post("/api/loop/start")
        assert r.json() == {"running": True, "paused": False, "relaunched": False}

        assert (await c.post("/api/loop/flatten")).status_code == 400


async def test_start_is_refused_once_the_loop_has_finished(journal):
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.run_forever(max_bars=1)
    assert loop.running is False

    async with _client(app) as c:
        r = await c.post("/api/loop/start")
    assert r.status_code == 409


async def test_start_relaunches_a_finished_loop_when_the_process_offers_to(journal):
    """Hosted: the budget ran out, the URL is still up, and Start begins a fresh run."""
    loop = build_loop(journal)
    app = create_app(journal, loop)
    await loop.run_forever(max_bars=1)
    assert loop.running is False

    launched: list[str] = []

    def relaunch():
        launched.append("go")
        loop.running = True  # what run_forever's first tick does

    app.state.relaunch = relaunch
    async with _client(app) as c:
        r = await c.post("/api/loop/start")
    assert r.status_code == 200 and r.json()["relaunched"] is True
    assert launched == ["go"]


# -- the token gate --------------------------------------------------------------------


async def test_writes_need_the_token_when_one_is_configured_and_reads_do_not(
    journal, monkeypatch
):
    from budapilot.web import app as web

    monkeypatch.setattr(web.settings, "dashboard_token", "s3cret")
    loop = build_loop(journal)
    loop.running = True
    app = create_app(journal, loop)

    async with _client(app) as c:
        assert (await c.get("/api/snapshot")).json()["locked"] is True
        assert (await c.get("/api/market")).status_code == 200

        assert (await c.post("/api/loop/stop")).status_code == 401
        assert loop.paused is False  # the refused call changed nothing
        assert (await c.post("/api/watchlist", json={"symbols": ["BTC/USD"]})).status_code == 401
        assert (await c.post("/api/deep/BTC/USD")).status_code == 401

        wrong = {"X-Dashboard-Token": "nope"}
        assert (await c.post("/api/loop/stop", headers=wrong)).status_code == 401

        right = {"X-Dashboard-Token": "s3cret"}
        assert (await c.post("/api/loop/stop", headers=right)).status_code == 200
        assert loop.paused is True


async def test_no_token_configured_means_no_gate(journal):
    loop = build_loop(journal)
    loop.running = True
    app = create_app(journal, loop)
    async with _client(app) as c:
        assert (await c.get("/api/snapshot")).json()["locked"] is False
        assert (await c.post("/api/loop/stop")).status_code == 200


async def test_index_renders(journal):
    app = create_app(journal, build_loop(journal))
    async with _client(app) as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "GOLDFISH" in r.text and "/api/market" in r.text
