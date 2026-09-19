"""Session risk state: the daily drawdown kill-switch and per-symbol cooldowns.

Pure functions over ``SessionState``, held to the same coverage bar as the risk engine
for the same reason -- these are the controls that stop a bad day becoming a very bad
one, and they are worthless if they are subtly wrong.

Two deliberate choices worth stating:

**Drawdown is measured from the session peak, not the open.** A desk that is up 8% and
gives back 5% has lost control of the day exactly as much as one that started flat and
fell 5%. Measuring from the open lets a morning profit fund an afternoon of losses.

**A halt blocks new entries, it does not flatten.** Force-liquidating into whatever
caused the drawdown is how a bad day becomes a catastrophic one -- you sell the bottom
and pay the spread to do it. Open positions keep their stops and take-profits, which
were sized before the trouble started. ``HALT_FLATTENS_POSITIONS`` inverts this for
anyone who disagrees.
"""

from __future__ import annotations

from datetime import date, datetime

from budapilot.config import (
    COOLDOWN_BARS,
    COOLDOWN_BARS_AFTER_TAKE,
    MAX_DAILY_DRAWDOWN_PCT,
)
from budapilot.contracts import ExitReason, SessionState, utcnow


def new_session(equity: float, day: date | None = None) -> SessionState:
    return SessionState(
        trading_day=day or utcnow().date(),
        day_open_equity=equity,
        day_peak_equity=equity,
    )


def drawdown_from(state: SessionState, equity: float) -> float:
    """Fractional drawdown from the session peak. Never negative."""
    if state.day_peak_equity <= 0:
        return 0.0
    return max(0.0, (state.day_peak_equity - equity) / state.day_peak_equity)


def roll_day(state: SessionState, equity: float, now: datetime | None = None) -> SessionState:
    """Start a fresh session at the UTC day boundary, clearing the halt.

    The bar index and cooldowns carry over: a symbol that stopped out four bars before
    midnight should still be cooling down four bars after it.
    """
    today = (now or utcnow()).date()
    if today == state.trading_day:
        return state
    return SessionState(
        trading_day=today,
        day_open_equity=equity,
        day_peak_equity=equity,
        bar_index=state.bar_index,
        halted=False,
        halt_reason="",
        cooldown_until=dict(state.cooldown_until),
    )


def observe_equity(state: SessionState, equity: float) -> SessionState:
    """Update the peak and trip the kill-switch if the drawdown limit is breached.

    Once halted the session stays halted until the day rolls; recovering above the
    threshold does not un-halt. A desk that hit its loss limit is done for the day,
    and an oscillating kill-switch is worse than none.
    """
    if equity > state.day_peak_equity:
        state.day_peak_equity = equity

    if not state.halted:
        drawdown = drawdown_from(state, equity)
        if drawdown >= MAX_DAILY_DRAWDOWN_PCT:
            state.halted = True
            state.halt_reason = (
                f"Daily drawdown {drawdown:.2%} reached the "
                f"{MAX_DAILY_DRAWDOWN_PCT:.0%} limit "
                f"(peak ${state.day_peak_equity:,.0f} -> ${equity:,.0f}). "
                f"New entries are blocked until the next UTC day; open positions keep "
                f"their stops."
            )
    return state


def start_cooldown(
    state: SessionState, symbol: str, reason: ExitReason
) -> SessionState:
    """Cool a symbol off after an exit.

    A take-profit gets a shorter cooldown than a stop-out: hitting your target is not
    evidence the thesis was wrong, whereas being stopped out is evidence you were early
    or wrong, and re-entering immediately is how one bad read becomes five.
    """
    bars = COOLDOWN_BARS_AFTER_TAKE if reason is ExitReason.TAKE else COOLDOWN_BARS
    state.cooldown_until[symbol] = state.bar_index + bars
    return state


def advance_bar(state: SessionState) -> SessionState:
    state.bar_index += 1
    # Drop expired entries so the dict does not grow without bound over a long session.
    state.cooldown_until = {
        sym: until for sym, until in state.cooldown_until.items() if until > state.bar_index
    }
    return state
