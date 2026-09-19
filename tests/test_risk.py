"""Risk engine tests. This is the module held to 100% branch coverage.

Every rejection path gets a test, because every rejection path is a place where the
system would otherwise have placed a trade it should not have.
"""

from __future__ import annotations

import math

import pytest

from budapilot.config import MAX_OPEN_POSITIONS, MAX_POSITION_PCT, RISK_PER_TRADE
from budapilot.contracts import Action, AssetSpec, NewsRisk, PortfolioState
from budapilot.risk.engine import evaluate, snap_to_increment
from tests.conftest import SYMBOL, make_position

# -- snapping -------------------------------------------------------------------------


def test_snap_floors_never_rounds_up():
    # Rounding up could breach a cap that was just checked.
    assert snap_to_increment(1.9999, 1.0) == 1.0
    assert snap_to_increment(0.9, 1.0) == 0.0


def test_snap_passthrough_when_no_increment():
    assert snap_to_increment(1.2345, 0.0) == 1.2345


def test_snap_handles_fractional_increment():
    assert snap_to_increment(0.123456789, 0.0001) == pytest.approx(0.1234)


# -- approvals ------------------------------------------------------------------------


def test_risk_budget_binds_when_atr_is_large(proposal, features, portfolio, asset):
    """When the stop is wide, risk-per-trade is what sizes the position."""
    # 2 x ATR must exceed MAX_POSITION_PCT of price for the risk budget to bind.
    features.atr_14 = features.close * 0.08  # 2xATR = 16% of price > 10% cap
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved
    assert "risk_budget" in d.reason
    loss_at_stop = (features.close - d.stop_px) * d.qty
    assert loss_at_stop == pytest.approx(portfolio.equity * RISK_PER_TRADE, rel=0.02)


def test_position_cap_binds_at_realistic_crypto_volatility(
    proposal, features, portfolio, asset
):
    """The interaction that decides real sizing, pinned so it cannot drift silently.

    On a 5-minute crypto chart ATR is typically well under 5% of price. Risking
    RISK_PER_TRADE to a 2xATR stop then implies a position far larger than
    MAX_POSITION_PCT allows, so the per-symbol cap binds and the desk's true risk per
    trade is *below* the nominal RISK_PER_TRADE. That is intentionally conservative,
    but the journal must say so rather than claiming the nominal number.
    """
    assert features.atr_pct == 0.01  # 1% ATR, a normal 5-minute crypto bar
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved
    assert "position_cap" in d.reason
    assert d.notional == pytest.approx(portfolio.equity * MAX_POSITION_PCT, rel=1e-6)

    actual_risk = (features.close - d.stop_px) * d.qty / portfolio.equity
    assert actual_risk < RISK_PER_TRADE  # less risk than nominal, never more


def test_stop_and_take_give_1_5_reward_risk(proposal, features, portfolio, asset):
    d = evaluate(proposal, features, portfolio, asset)
    risk_dist = features.close - d.stop_px
    reward_dist = d.take_px - features.close
    assert reward_dist / risk_dist == pytest.approx(1.5, rel=1e-6)


def test_size_multiplier_scales_the_position(proposal, features, portfolio, asset):
    """Only meaningful while the risk budget is the binding constraint."""
    features.atr_14 = features.close * 0.08
    full = evaluate(proposal, features, portfolio, asset)
    proposal.size_multiplier = 0.5
    half = evaluate(proposal, features, portfolio, asset)
    assert half.qty == pytest.approx(full.qty * 0.5, rel=1e-6)


def test_size_multiplier_is_powerless_once_the_cap_binds(
    proposal, features, portfolio, asset
):
    """A consequence worth knowing: at normal ATR the risk analyst's 0.9x does nothing.

    Only a multiplier small enough to pull the risk-budget quantity under the cap
    changes the order. The risk analyst can still veto with 0.0.
    """
    full = evaluate(proposal, features, portfolio, asset)
    proposal.size_multiplier = 0.9
    nudged = evaluate(proposal, features, portfolio, asset)
    assert nudged.qty == pytest.approx(full.qty, rel=1e-9)

    proposal.size_multiplier = 0.1  # small enough to bind
    cut = evaluate(proposal, features, portfolio, asset)
    assert cut.qty < full.qty


def test_position_cap_clamps_an_oversized_entry(proposal, features, portfolio, asset):
    # A tiny ATR would otherwise size an enormous position.
    features.atr_14 = 1.0
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved
    assert d.notional <= portfolio.equity * MAX_POSITION_PCT * 1.0001


def test_sell_closes_the_whole_position(proposal, features, portfolio, asset):
    portfolio.positions[SYMBOL] = make_position(qty=0.25)
    proposal.action = Action.SELL
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved and d.qty == pytest.approx(0.25)


# -- rejections -----------------------------------------------------------------------


def test_hold_is_not_executable(proposal, features, portfolio, asset):
    proposal.action = Action.HOLD
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "HOLD" in d.reason


def test_untradable_asset_rejected(proposal, features, portfolio, asset):
    asset.tradable = False
    assert not evaluate(proposal, features, portfolio, asset).approved


def test_critical_news_is_an_absolute_veto(proposal, features, portfolio, asset):
    proposal.conviction = 0.99
    d = evaluate(proposal, features, portfolio, asset, news_risk=NewsRisk.CRITICAL)
    assert not d.approved and "CRITICAL" in d.reason


def test_high_news_risk_alone_does_not_block(proposal, features, portfolio, asset):
    assert evaluate(proposal, features, portfolio, asset, news_risk=NewsRisk.HIGH).approved


def test_low_conviction_rejected(proposal, features, portfolio, asset):
    proposal.conviction = 0.4
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "below" in d.reason


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_bad_atr_fails_closed(proposal, features, portfolio, asset, bad):
    features.atr_14 = bad
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved


def test_nan_equity_fails_closed(proposal, features, portfolio, asset):
    portfolio.equity = float("nan")
    assert not evaluate(proposal, features, portfolio, asset).approved


def test_zero_equity_rejected(proposal, features, portfolio, asset):
    portfolio.equity = 0.0
    assert not evaluate(proposal, features, portfolio, asset).approved


def test_negative_equity_rejected(proposal, features, portfolio, asset):
    portfolio.equity = -5.0
    portfolio.cash = -5.0
    assert not evaluate(proposal, features, portfolio, asset).approved


def test_zero_price_fails_closed(proposal, features, portfolio, asset):
    features.close = 0.0
    assert not evaluate(proposal, features, portfolio, asset).approved


# -- the long-only invariant ------------------------------------------------------------


def test_sell_with_no_position_is_rejected_never_shorted(proposal, features, portfolio, asset):
    """The invariant that keeps us from sending a short to a venue that forbids it."""
    proposal.action = Action.SELL
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved
    assert d.qty == 0.0
    assert "long-only" in d.reason


def test_sell_with_zero_qty_position_rejected(proposal, features, portfolio, asset):
    portfolio.positions[SYMBOL] = make_position(qty=0.0)
    proposal.action = Action.SELL
    assert not evaluate(proposal, features, portfolio, asset).approved


def test_sell_below_min_increment_rejected(proposal, features, portfolio, asset):
    portfolio.positions[SYMBOL] = make_position(qty=0.5)
    proposal.action = Action.SELL
    asset.min_trade_increment = 1.0  # position is smaller than one increment
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "minimum increment" in d.reason


# -- portfolio caps -----------------------------------------------------------------------


def test_max_open_positions_blocks_a_new_symbol(proposal, features, portfolio, asset):
    for i in range(MAX_OPEN_POSITIONS):
        portfolio.positions[f"ALT{i}/USD"] = make_position(f"ALT{i}/USD", qty=0.01, entry=100.0)
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "max" in d.reason


def test_adding_to_an_existing_position_ignores_the_slot_cap(
    proposal, features, portfolio, asset
):
    for i in range(MAX_OPEN_POSITIONS - 1):
        portfolio.positions[f"ALT{i}/USD"] = make_position(f"ALT{i}/USD", qty=0.01, entry=100.0)
    portfolio.positions[SYMBOL] = make_position(qty=0.001)
    assert evaluate(proposal, features, portfolio, asset).approved


def test_symbol_already_at_its_cap(proposal, features, portfolio, asset):
    portfolio.positions[SYMBOL] = make_position(
        qty=portfolio.equity * MAX_POSITION_PCT / 60_000.0, entry=60_000.0
    )
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "per-symbol cap" in d.reason


def test_gross_exposure_cap(proposal, features, portfolio, asset):
    portfolio.positions["ETH/USD"] = make_position("ETH/USD", qty=25.0, entry=2_000.0)
    portfolio.positions["SOL/USD"] = make_position("SOL/USD", qty=1.0, entry=1.0)
    assert portfolio.exposure_pct >= 0.5
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "exposure" in d.reason


def test_no_cash_rejected(proposal, features, portfolio, asset):
    portfolio.cash = 0.0
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "cash" in d.reason.lower()


def test_zero_size_multiplier_means_the_risk_analyst_said_no(
    proposal, features, portfolio, asset
):
    proposal.size_multiplier = 0.0
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "sized this to nothing" in d.reason


def test_below_min_order_size_rejected(proposal, features, portfolio, asset):
    portfolio.equity = 100.0
    portfolio.cash = 100.0
    asset.min_order_size = 50.0
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "minimum" in d.reason


def test_coarse_increment_can_reject_a_small_order(proposal, features, portfolio, asset):
    asset.min_trade_increment = 1.0  # 1 whole BTC per step
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "minimum" in d.reason


def test_dust_order_rejected(proposal, features, portfolio, asset):
    """Filling the last sliver of cap headroom pays fees for nothing."""
    # Leave only ~$20 of headroom under the per-symbol cap on $100k equity.
    held = (portfolio.equity * MAX_POSITION_PCT - 20.0) / features.close
    portfolio.positions[SYMBOL] = make_position(qty=held, entry=features.close)
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved
    assert "too small to be worth the fees" in d.reason


def test_order_just_above_the_dust_floor_is_allowed(proposal, features, portfolio, asset):
    from budapilot.config import MIN_ORDER_NOTIONAL_PCT

    floor = portfolio.equity * MIN_ORDER_NOTIONAL_PCT
    held = (portfolio.equity * MAX_POSITION_PCT - floor * 1.5) / features.close
    portfolio.positions[SYMBOL] = make_position(qty=held, entry=features.close)
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved and d.notional >= floor


def test_huge_atr_makes_the_stop_non_positive(proposal, features, portfolio, asset):
    features.atr_14 = features.close  # 2x ATR stop lands below zero
    proposal.stop_atr = 2.0
    d = evaluate(proposal, features, portfolio, asset)
    assert not d.approved and "at or below zero" in d.reason


def test_default_stop_used_when_proposal_has_none(proposal, features, portfolio, asset):
    object.__setattr__(proposal, "stop_atr", 0.0)  # bypass validation for the branch
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved


def test_engine_fails_closed_on_an_unexpected_error(proposal, features, portfolio, asset):
    class Exploding(PortfolioState):
        @property
        def gross_exposure(self) -> float:
            raise RuntimeError("boom")

    bad = Exploding(equity=100_000.0, cash=100_000.0, positions={})
    d = evaluate(proposal, features, bad, asset)
    assert not d.approved and "failing closed" in d.reason


def test_approved_quantity_is_always_a_whole_increment(proposal, features, portfolio, asset):
    asset.min_trade_increment = 0.0001
    d = evaluate(proposal, features, portfolio, asset)
    assert d.approved
    steps = d.qty / asset.min_trade_increment
    assert math.isclose(steps, round(steps), abs_tol=1e-6)


def test_min_increment_default_asset_spec_does_not_crash(proposal, features, portfolio):
    d = evaluate(proposal, features, portfolio, AssetSpec(symbol=SYMBOL))
    assert d.approved
