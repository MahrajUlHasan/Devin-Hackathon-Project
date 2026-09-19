"""The deterministic risk engine. Pure functions, no I/O, hard veto.

This is the only module in the system that a language model cannot influence. Agents
propose; this decides. It is also the only module held to 100% branch coverage, because
it is the one place where a bug costs money rather than embarrassment.

**It fails closed.** Any exception, any missing input, any NaN produces a rejection with
a stated reason and no order. A visibly blocked trade is a better outcome than a
silently allowed one.

Two venue facts are enforced here rather than trusted to a prompt:
- Crypto is long-only: SELL with no position is rejected, never flipped into a short.
- Quantities are snapped to the asset's ``min_trade_increment``, without which Alpaca
  rejects the order with a 422.
"""

from __future__ import annotations

import math

from budapilot.config import (
    K_STOP,
    K_TAKE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_CONVICTION,
    MIN_ORDER_NOTIONAL_PCT,
    RISK_PER_TRADE,
)
from budapilot.contracts import (
    Action,
    AssetSpec,
    FeatureBundle,
    NewsRisk,
    PortfolioState,
    RiskDecision,
    TradeProposal,
)


def _finite(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def snap_to_increment(qty: float, increment: float) -> float:
    """Floor to a whole multiple of the venue's trade increment.

    Floor, never round: rounding up can push a position over a cap that was just
    checked, and can exceed available cash on a full-size entry.
    """
    if increment <= 0:
        return qty
    steps = math.floor(qty / increment + 1e-9)
    return round(steps * increment, 12)


def _reject(symbol: str, action: Action, reason: str, checks: dict[str, bool]) -> RiskDecision:
    return RiskDecision(
        approved=False, symbol=symbol, action=action, reason=reason, checks=checks
    )


def evaluate(
    proposal: TradeProposal,
    features: FeatureBundle,
    portfolio: PortfolioState,
    asset: AssetSpec,
    *,
    news_risk: NewsRisk = NewsRisk.LOW,
) -> RiskDecision:
    """Rule on one proposal. Returns an approval with a sized order, or a rejection."""
    checks: dict[str, bool] = {}
    symbol = proposal.symbol
    action = proposal.action

    try:
        # -- Trivial and structural rejections -----------------------------------------
        if action is Action.HOLD:
            checks["actionable"] = False
            return _reject(symbol, action, "Proposal is HOLD; nothing to execute.", checks)
        checks["actionable"] = True

        if not asset.tradable:
            checks["tradable"] = False
            return _reject(symbol, action, f"{symbol} is not tradable at the venue.", checks)
        checks["tradable"] = True

        if news_risk is NewsRisk.CRITICAL:
            checks["news_risk"] = False
            return _reject(
                symbol,
                action,
                "News risk is CRITICAL; trading in this symbol is vetoed regardless "
                "of conviction.",
                checks,
            )
        checks["news_risk"] = True

        if proposal.conviction < MIN_CONVICTION:
            checks["conviction"] = False
            return _reject(
                symbol,
                action,
                f"Conviction {proposal.conviction:.2f} is below the "
                f"{MIN_CONVICTION:.2f} floor.",
                checks,
            )
        checks["conviction"] = True

        # -- Data integrity: fail closed on anything non-finite ------------------------
        atr = features.atr_14
        price = features.close
        if not _finite(atr, price, portfolio.equity) or atr <= 0 or price <= 0:
            checks["inputs_finite"] = False
            return _reject(
                symbol,
                action,
                "Missing or non-finite inputs (ATR, price or equity); failing closed.",
                checks,
            )
        checks["inputs_finite"] = True

        if portfolio.equity <= 0:
            checks["equity_positive"] = False
            return _reject(symbol, action, "Equity is zero or negative.", checks)
        checks["equity_positive"] = True

        position = portfolio.positions.get(symbol)

        # -- SELL: reduce or close only. Crypto is long-only. --------------------------
        if action is Action.SELL:
            if position is None or position.qty <= 0:
                checks["has_position"] = False
                return _reject(
                    symbol,
                    action,
                    "SELL with no open long. Crypto is long-only here; this would be "
                    "a short, which the venue does not permit.",
                    checks,
                )
            checks["has_position"] = True

            qty = snap_to_increment(position.qty, asset.min_trade_increment)
            if qty <= 0:
                checks["min_size"] = False
                return _reject(
                    symbol,
                    action,
                    f"Position {position.qty} is below the venue minimum increment "
                    f"{asset.min_trade_increment}; cannot be closed by size.",
                    checks,
                )
            checks["min_size"] = True
            return RiskDecision(
                approved=True,
                symbol=symbol,
                action=action,
                qty=qty,
                notional=qty * price,
                entry_ref=price,
                reason=f"Closing {qty} {symbol} at ~{price:.4f}.",
                checks=checks,
            )

        # -- BUY: portfolio caps, then sizing ------------------------------------------
        if position is None and portfolio.open_count >= MAX_OPEN_POSITIONS:
            checks["open_slots"] = False
            return _reject(
                symbol,
                action,
                f"Already holding {portfolio.open_count} positions "
                f"(max {MAX_OPEN_POSITIONS}).",
                checks,
            )
        checks["open_slots"] = True

        stop_mult = proposal.stop_atr if proposal.stop_atr > 0 else K_STOP

        # Validate the stop level before sizing. It depends only on price and ATR, and
        # a stop at or below zero means the setup is unworkable regardless of size.
        stop_px = price - stop_mult * atr
        take_px = price + (K_TAKE / K_STOP) * stop_mult * atr
        if stop_px <= 0:
            checks["stop_valid"] = False
            return _reject(
                symbol,
                action,
                f"Computed stop {stop_px:.4f} is at or below zero; ATR "
                f"{atr:.4f} is too large relative to price {price:.4f}.",
                checks,
            )
        checks["stop_valid"] = True

        risk_budget = portfolio.equity * RISK_PER_TRADE * proposal.size_multiplier
        if risk_budget <= 0:
            checks["risk_budget"] = False
            return _reject(
                symbol,
                action,
                f"Risk budget is zero (size_multiplier="
                f"{proposal.size_multiplier:.2f}); the risk analyst sized this to nothing.",
                checks,
            )
        checks["risk_budget"] = True

        # Size so that a stop-out costs exactly RISK_PER_TRADE of equity.
        raw_qty = risk_budget / (stop_mult * atr)

        # Clamp 1: per-symbol cap, counting any existing position.
        held_value = position.market_value if position else 0.0
        symbol_headroom = max(portfolio.equity * MAX_POSITION_PCT - held_value, 0.0)
        if symbol_headroom <= 0:
            checks["position_cap"] = False
            return _reject(
                symbol,
                action,
                f"{symbol} is already at the {MAX_POSITION_PCT:.0%} per-symbol cap.",
                checks,
            )
        checks["position_cap"] = True

        # Clamp 2: remaining gross exposure headroom.
        exposure_headroom = max(
            portfolio.equity * MAX_TOTAL_EXPOSURE_PCT - portfolio.gross_exposure, 0.0
        )
        if exposure_headroom <= 0:
            checks["exposure_cap"] = False
            return _reject(
                symbol,
                action,
                f"Gross exposure is at the {MAX_TOTAL_EXPOSURE_PCT:.0%} cap.",
                checks,
            )
        checks["exposure_cap"] = True

        # Clamp 3: cash on hand.
        cash_headroom = max(portfolio.cash, 0.0)
        if cash_headroom <= 0:
            checks["cash"] = False
            return _reject(symbol, action, "No cash available.", checks)
        checks["cash"] = True

        # Which constraint actually bound? On a 5-minute crypto chart ATR is usually
        # well under 5% of price, which makes the risk-budget quantity larger than the
        # per-symbol cap allows -- so the cap binds and true risk-per-trade lands below
        # RISK_PER_TRADE. Reporting the wrong binding constraint would make the journal
        # claim a risk the desk is not actually taking.
        caps = {
            "position_cap": symbol_headroom / price,
            "exposure_cap": exposure_headroom / price,
            "cash": cash_headroom / price,
            "risk_budget": raw_qty,
        }
        binding = min(caps, key=lambda k: caps[k])
        qty = snap_to_increment(min(caps.values()), asset.min_trade_increment)

        if qty <= 0 or qty * price < asset.min_order_size:
            checks["min_size"] = False
            return _reject(
                symbol,
                action,
                f"Sized quantity {qty} is below the venue minimum "
                f"(increment {asset.min_trade_increment}, min notional "
                f"{asset.min_order_size}).",
                checks,
            )
        checks["min_size"] = True

        # Dust guard. Filling the last sliver of cap headroom pays fees for a position
        # too small to affect the book.
        floor = portfolio.equity * MIN_ORDER_NOTIONAL_PCT
        if qty * price < floor:
            checks["meaningful_size"] = False
            return _reject(
                symbol,
                action,
                f"Order notional ${qty * price:,.2f} is below the "
                f"{MIN_ORDER_NOTIONAL_PCT:.1%} of equity floor (${floor:,.2f}); "
                f"too small to be worth the fees.",
                checks,
            )
        checks["meaningful_size"] = True

        return RiskDecision(
            approved=True,
            symbol=symbol,
            action=action,
            qty=qty,
            notional=qty * price,
            entry_ref=price,
            stop_px=round(stop_px, 8),
            take_px=round(take_px, 8),
            reason=(
                f"Approved {qty} {symbol} (~${qty * price:,.2f}, "
                f"{qty * price / portfolio.equity:.1%} of equity), bound by "
                f"{binding}. Stop {stop_mult:.1f}xATR at {stop_px:.4f} risks "
                f"{(price - stop_px) * qty / portfolio.equity:.2%} of equity."
            ),
            checks=checks,
        )

    except Exception as exc:  # noqa: BLE001 -- fails closed, always
        return _reject(
            symbol,
            action,
            f"Risk engine error, failing closed: {type(exc).__name__}: {exc}",
            checks,
        )
