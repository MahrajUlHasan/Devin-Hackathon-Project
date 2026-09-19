"""Kill-switch and cooldown tests.

These are the controls that stop a bad day becoming a very bad one, so they are held
to the same bar as the risk engine. Two behaviours get the most attention because they
are the ones most likely to be quietly wrong:

- the halt must never block an EXIT, only an entry
- the halt must survive a restart, or a crash becomes a way to reset the loss limit
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from budapilot.config import (
    COOLDOWN_BARS,
    COOLDOWN_BARS_AFTER_TAKE,
    MAX_DAILY_DRAWDOWN_PCT,
)
from budapilot.contracts import Action, ExitReason, SessionState
from budapilot.journal.store import Journal
from budapilot.risk.engine import evaluate
from budapilot.risk.session import (
    advance_bar,
    drawdown_from,
    new_session,
    observe_equity,
    roll_day,
    start_cooldown,
)
from tests.conftest import SYMBOL, make_position

START = 100_000.0


@pytest.fixture
def session() -> SessionState:
    return new_session(START, day=date(2026, 9, 19))


# -- drawdown accounting --------------------------------------------------------------


def test_new_session_starts_flat(session):
    assert drawdown_from(session, START) == 0.0
    assert not session.halted


def test_drawdown_is_measured_from_the_peak_not_the_open(session):
    """Up 10% then back to flat is a 9.1% drawdown, not 0%."""
    observe_equity(session, 110_000.0)
    assert session.day_peak_equity == 110_000.0
    assert drawdown_from(session, 100_000.0) == pytest.approx(10_000 / 110_000)


def test_profit_does_not_fund_later_losses(session):
    """Measuring from the open would let a morning gain mask an afternoon collapse."""
    observe_equity(session, 120_000.0)  # +20%
    observe_equity(session, 114_500.0)  # -4.6% from peak, still +14.5% on the day
    assert not session.halted
    observe_equity(session, 113_000.0)  # -5.8% from peak
    assert session.halted


def test_gain_raises_the_peak(session):
    observe_equity(session, 101_000.0)
    assert session.day_peak_equity == 101_000.0
    assert drawdown_from(session, 101_000.0) == 0.0


def test_drawdown_with_zero_peak_is_zero():
    state = new_session(0.0)
    assert drawdown_from(state, 0.0) == 0.0


def test_drawdown_never_negative(session):
    assert drawdown_from(session, 200_000.0) == 0.0


# -- the kill-switch ------------------------------------------------------------------


def test_kill_switch_trips_at_the_limit(session):
    observe_equity(session, START * (1 - MAX_DAILY_DRAWDOWN_PCT))
    assert session.halted
    assert "Daily drawdown" in session.halt_reason
    assert "open positions keep" in session.halt_reason


def test_kill_switch_does_not_trip_just_below_the_limit(session):
    observe_equity(session, START * (1 - MAX_DAILY_DRAWDOWN_PCT) + 1)
    assert not session.halted


def test_halt_is_sticky_and_does_not_reset_on_recovery(session):
    """An oscillating kill-switch is worse than none."""
    observe_equity(session, 94_000.0)
    assert session.halted
    reason = session.halt_reason

    observe_equity(session, 105_000.0)  # fully recovered and then some
    assert session.halted
    assert session.halt_reason == reason  # unchanged, not re-armed


def test_halt_reason_is_written_once(session):
    observe_equity(session, 90_000.0)
    first = session.halt_reason
    observe_equity(session, 80_000.0)
    assert session.halt_reason == first


# -- day rollover ----------------------------------------------------------------------


def test_roll_day_clears_the_halt(session):
    observe_equity(session, 90_000.0)
    assert session.halted

    tomorrow = datetime(2026, 9, 20, 0, 5, tzinfo=None).replace(tzinfo=None)
    rolled = roll_day(session, 90_000.0, now=datetime(2026, 9, 20, 0, 5))
    assert not rolled.halted
    assert rolled.day_open_equity == 90_000.0
    assert rolled.day_peak_equity == 90_000.0
    assert rolled.trading_day == tomorrow.date()


def test_roll_day_is_a_noop_within_the_same_day(session):
    same = roll_day(session, 50_000.0, now=datetime(2026, 9, 19, 23, 59))
    assert same is session
    assert same.day_open_equity == START  # not rewritten


def test_cooldowns_survive_the_day_boundary(session):
    """A symbol that stopped out before midnight is still cooling after it."""
    start_cooldown(session, SYMBOL, ExitReason.STOP)
    rolled = roll_day(session, START, now=datetime(2026, 9, 20, 0, 1))
    assert rolled.cooling_down(SYMBOL)
    assert rolled.bar_index == session.bar_index


# -- cooldowns --------------------------------------------------------------------------


def test_stop_out_cools_longer_than_a_take_profit(session):
    start_cooldown(session, "A/USD", ExitReason.STOP)
    start_cooldown(session, "B/USD", ExitReason.TAKE)
    assert session.bars_remaining("A/USD") == COOLDOWN_BARS
    assert session.bars_remaining("B/USD") == COOLDOWN_BARS_AFTER_TAKE
    assert COOLDOWN_BARS > COOLDOWN_BARS_AFTER_TAKE


def test_cooldown_expires_after_the_right_number_of_bars(session):
    start_cooldown(session, SYMBOL, ExitReason.STOP)
    for _ in range(COOLDOWN_BARS - 1):
        advance_bar(session)
        assert session.cooling_down(SYMBOL)
    advance_bar(session)
    assert not session.cooling_down(SYMBOL)


def test_advance_bar_evicts_expired_cooldowns(session):
    """Otherwise the dict grows unbounded across a long session."""
    start_cooldown(session, SYMBOL, ExitReason.TAKE)
    for _ in range(COOLDOWN_BARS_AFTER_TAKE + 1):
        advance_bar(session)
    assert SYMBOL not in session.cooldown_until


def test_unknown_symbol_is_not_cooling(session):
    assert not session.cooling_down("NOPE/USD")
    assert session.bars_remaining("NOPE/USD") == 0


# -- enforcement in the risk engine ------------------------------------------------------


def test_halt_blocks_a_buy(proposal, features, portfolio, asset, session):
    observe_equity(session, 90_000.0)
    d = evaluate(proposal, features, portfolio, asset, session=session)
    assert not d.approved
    assert "halted" in d.reason.lower()


def test_halt_does_not_block_an_exit(proposal, features, portfolio, asset, session):
    """Refusing to let the desk out of a position is not a risk control, it is a trap."""
    observe_equity(session, 90_000.0)
    portfolio.positions[SYMBOL] = make_position(qty=0.25)
    proposal.action = Action.SELL

    d = evaluate(proposal, features, portfolio, asset, session=session)

    assert d.approved
    assert d.qty == pytest.approx(0.25)


def test_cooldown_blocks_a_re_entry(proposal, features, portfolio, asset, session):
    start_cooldown(session, SYMBOL, ExitReason.STOP)
    d = evaluate(proposal, features, portfolio, asset, session=session)
    assert not d.approved
    assert "cooldown" in d.reason


def test_cooldown_does_not_block_an_exit(proposal, features, portfolio, asset, session):
    start_cooldown(session, SYMBOL, ExitReason.STOP)
    portfolio.positions[SYMBOL] = make_position(qty=0.25)
    proposal.action = Action.SELL
    assert evaluate(proposal, features, portfolio, asset, session=session).approved


def test_cooldown_does_not_block_other_symbols(
    proposal, features, portfolio, asset, session
):
    start_cooldown(session, "ETH/USD", ExitReason.STOP)
    assert evaluate(proposal, features, portfolio, asset, session=session).approved


def test_buy_allowed_once_the_cooldown_expires(
    proposal, features, portfolio, asset, session
):
    start_cooldown(session, SYMBOL, ExitReason.STOP)
    for _ in range(COOLDOWN_BARS):
        advance_bar(session)
    assert evaluate(proposal, features, portfolio, asset, session=session).approved


def test_no_session_means_no_session_checks(proposal, features, portfolio, asset):
    """Session state is optional; the engine must work without it."""
    d = evaluate(proposal, features, portfolio, asset, session=None)
    assert d.approved
    assert "not_halted" not in d.checks


# -- persistence ---------------------------------------------------------------------------


def test_session_survives_a_restart(tmp_path):
    """A crash must not be a way to reset the daily loss limit."""
    path = str(tmp_path / "s.db")
    j1 = Journal(path)
    state = new_session(START)
    observe_equity(state, 90_000.0)
    start_cooldown(state, SYMBOL, ExitReason.STOP)
    j1.save_session(state)
    j1.close()

    j2 = Journal(path)
    restored = j2.load_session()
    assert restored is not None
    assert restored.halted
    assert restored.halt_reason == state.halt_reason
    assert restored.cooling_down(SYMBOL)
    assert restored.day_peak_equity == START
    j2.close()


def test_load_session_on_a_fresh_db_returns_none(tmp_path):
    j = Journal(str(tmp_path / "fresh.db"))
    assert j.load_session() is None
    j.close()


def test_corrupt_session_row_does_not_block_startup(tmp_path):
    j = Journal(str(tmp_path / "bad.db"))
    j._write(
        "INSERT INTO session_state (id, state_json, updated_at) VALUES (1,?,?)",
        ("{not json", "now"),
    )
    assert j.load_session() is None  # degrades rather than crashing the process
    j.close()


def test_save_session_is_idempotent(tmp_path):
    j = Journal(str(tmp_path / "i.db"))
    state = new_session(START)
    j.save_session(state)
    observe_equity(state, 90_000.0)
    j.save_session(state)
    assert len(j.query("SELECT * FROM session_state")) == 1
    assert j.load_session().halted
    j.close()


# -- loop integration -----------------------------------------------------------------------


async def test_loop_halts_and_stops_entering(tmp_path, monkeypatch):
    from budapilot.agents.base import AgentRuntime
    from budapilot.agents.bus import AgentBus
    from budapilot.contracts import TradeProposal
    from budapilot.data.fixtures import FixtureFeed
    from budapilot.execution.broker_sim import SimBroker
    from budapilot.loop import TradingLoop

    journal = Journal(str(tmp_path / "loop.db"))
    loop = TradingLoop(
        feed=FixtureFeed(["BTC/USD"]),
        broker=SimBroker(cash=100_000.0),
        bus=AgentBus(AgentRuntime(offline=True)),
        journal=journal,
        symbols=["BTC/USD"],
        interval_s=0,
    )
    original = loop.bus.run_bar

    async def buy(*a, **kw):
        decision = await original(*a, **kw)
        decision.proposal = TradeProposal(
            symbol="BTC/USD", action=Action.BUY, conviction=0.95, rationale="forced"
        )
        return decision

    monkeypatch.setattr(loop.bus, "run_bar", buy)

    await loop.startup()
    assert loop.session is not None and not loop.session.halted

    # Wipe out 6% of equity behind the loop's back.
    loop.broker.cash = 94_000.0
    await loop.run_once()

    assert loop.session.halted
    assert journal.load_session().halted  # persisted
    rulings = journal.latest_risk(5)
    assert rulings and not rulings[0]["approved"]
    assert "halted" in rulings[0]["reason"].lower()
    journal.close()


async def test_loop_starts_a_cooldown_on_exit(tmp_path):
    from budapilot.agents.base import AgentRuntime
    from budapilot.agents.bus import AgentBus
    from budapilot.contracts import ProtectedPosition
    from budapilot.data.fixtures import FixtureFeed
    from budapilot.execution.broker_sim import SimBroker
    from budapilot.loop import TradingLoop

    journal = Journal(str(tmp_path / "cd.db"))
    loop = TradingLoop(
        feed=FixtureFeed(["BTC/USD"]),
        broker=SimBroker(cash=100_000.0),
        bus=AgentBus(AgentRuntime(offline=True)),
        journal=journal,
        symbols=["BTC/USD"],
        interval_s=0,
    )
    await loop.startup()

    pos = ProtectedPosition(
        symbol="BTC/USD", qty=0.1, entry=100.0, stop_px=95.0, take_px=110.0
    )
    await loop._on_exit(pos, 95.0, ExitReason.STOP)

    assert loop.session.cooling_down("BTC/USD")
    assert loop.session.bars_remaining("BTC/USD") == COOLDOWN_BARS
    assert journal.load_session().cooling_down("BTC/USD")
    journal.close()
