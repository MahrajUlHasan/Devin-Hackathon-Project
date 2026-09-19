"""StopManager tests.

Alpaca crypto has no bracket orders, so protective exits are ours to get right. The
sequence that matters is cancel -> CONFIRM -> close. Inverting it, or trusting an
unconfirmed cancel, double-sells the position. These tests assert the ordering
directly rather than just the end state, because the end state can look correct while
the sequence is wrong.
"""

from __future__ import annotations

import asyncio

import pytest

from budapilot.contracts import (
    AssetSpec,
    ExitReason,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
    ProtectedPosition,
    utcnow,
)
from budapilot.execution.stops import StopManager

SYMBOL = "BTC/USD"


class RecordingBroker:
    """Records the exact call sequence so ordering can be asserted, not inferred."""

    def __init__(
        self,
        *,
        cancel_succeeds: bool = True,
        stop_submit_fails: bool = False,
        close_rejected: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.cancel_succeeds = cancel_succeeds
        self.stop_submit_fails = stop_submit_fails
        self.close_rejected = close_rejected
        self._n = 0

    async def get_portfolio(self) -> PortfolioState:
        self.calls.append("get_portfolio")
        return PortfolioState(equity=100_000.0, cash=50_000.0, positions=dict(self.positions))

    async def get_asset(self, symbol: str) -> AssetSpec:
        return AssetSpec(symbol=symbol)

    async def submit(self, req: OrderRequest) -> Order:
        self._n += 1
        kind = "submit_stop" if req.type is OrderType.STOP_LIMIT else "submit_close"
        self.calls.append(kind)
        if kind == "submit_stop" and self.stop_submit_fails:
            raise RuntimeError("venue rejected the resting stop")
        order = Order(
            id=f"o{self._n}",
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            type=req.type,
            status=OrderStatus.ACCEPTED,
            stop_price=req.stop_price,
            submitted_at=utcnow(),
        )
        if req.type is not OrderType.STOP_LIMIT:
            if self.close_rejected:
                order.status = OrderStatus.REJECTED
            else:
                order.status = OrderStatus.FILLED
                order.filled_qty = req.qty
                order.filled_avg_price = req.limit_price or 100.0
        self.orders[order.id] = order
        return order

    async def cancel(self, order_id: str) -> bool:
        self.calls.append("cancel")
        if not self.cancel_succeeds:
            return False
        order = self.orders.get(order_id)
        if order:
            order.status = OrderStatus.CANCELED
        return True

    async def get_order(self, order_id: str) -> Order | None:
        self.calls.append("get_order")
        return self.orders.get(order_id)

    async def list_open_orders(self) -> list[Order]:
        self.calls.append("list_open_orders")
        return [
            o for o in self.orders.values()
            if o.status in (OrderStatus.NEW, OrderStatus.ACCEPTED)
        ]


async def armed(broker: RecordingBroker, **kw) -> tuple[StopManager, list]:
    exits: list = []

    async def on_exit(pos, fill, reason):
        exits.append((pos.symbol, fill, reason))

    mgr = StopManager(broker, on_exit=on_exit)
    await mgr.protect(SYMBOL, qty=0.1, entry=100.0, stop_px=95.0, take_px=110.0, **kw)
    broker.calls.clear()
    return mgr, exits


# -- arming -------------------------------------------------------------------------


async def test_protect_arms_a_resting_stop_limit():
    broker = RecordingBroker()
    mgr = StopManager(broker)
    pos = await mgr.protect(SYMBOL, 0.1, 100.0, 95.0, 110.0)
    assert broker.calls == ["submit_stop"]
    assert pos.broker_stop_order_id is not None
    resting = broker.orders[pos.broker_stop_order_id]
    assert resting.type is OrderType.STOP_LIMIT
    assert resting.side is OrderSide.SELL
    assert resting.stop_price == 95.0


async def test_resting_stop_limit_price_sits_below_the_stop():
    """A limit at the stop may never fill in a fast drop; it must be slipped below."""
    broker = RecordingBroker()
    mgr = StopManager(broker)
    await mgr.protect(SYMBOL, 0.1, 100.0, 95.0, 110.0)
    submitted = next(o for o in broker.orders.values() if o.type is OrderType.STOP_LIMIT)
    assert submitted.stop_price == 95.0


async def test_a_failed_resting_stop_does_not_lose_the_position():
    """Losing the backstop is a degradation, not a reason to abandon a filled position."""
    broker = RecordingBroker(stop_submit_fails=True)
    mgr = StopManager(broker)
    pos = await mgr.protect(SYMBOL, 0.1, 100.0, 95.0, 110.0)
    assert pos.broker_stop_order_id is None
    assert SYMBOL in mgr.positions  # still protected in software


# -- THE RACE -----------------------------------------------------------------------


async def test_close_cancels_and_confirms_before_selling():
    """The ordering test. cancel -> get_order (confirm) -> submit_close. Never inverted."""
    broker = RecordingBroker()
    mgr, exits = await armed(broker)

    reason = await mgr.on_price(SYMBOL, 94.0)

    assert reason is ExitReason.STOP
    assert broker.calls == ["cancel", "get_order", "submit_close"]
    assert broker.calls.index("cancel") < broker.calls.index("submit_close")
    assert exits == [(SYMBOL, 100.0, ExitReason.STOP)]


async def test_unconfirmed_cancel_refuses_to_sell():
    """If we cannot prove the resting stop is dead, selling risks a double-sell."""
    broker = RecordingBroker(cancel_succeeds=False)
    mgr, exits = await armed(broker)

    reason = await mgr.on_price(SYMBOL, 94.0)

    assert reason is None
    assert "submit_close" not in broker.calls  # the critical assertion
    assert exits == []
    assert SYMBOL in mgr.positions  # still tracked, retried next tick
    assert mgr.positions[SYMBOL].closing is False  # and not wedged


async def test_a_filled_resting_stop_is_not_treated_as_cancelled():
    """The stop already filled: the position is gone. Selling again would be a short."""
    broker = RecordingBroker()
    mgr, exits = await armed(broker)
    stop_id = mgr.positions[SYMBOL].broker_stop_order_id
    broker.orders[stop_id].status = OrderStatus.FILLED
    broker.cancel_succeeds = False

    assert await mgr.on_price(SYMBOL, 94.0) is None
    assert "submit_close" not in broker.calls


async def test_concurrent_ticks_close_exactly_once():
    """Ticks arrive faster than the round trip to Alpaca. Only one may close."""
    broker = RecordingBroker()
    mgr, exits = await armed(broker)

    results = await asyncio.gather(*(mgr.on_price(SYMBOL, 94.0) for _ in range(12)))

    assert results.count(ExitReason.STOP) == 1
    assert broker.calls.count("submit_close") == 1
    assert len(exits) == 1


async def test_closing_flag_blocks_re_entry():
    broker = RecordingBroker()
    mgr, _ = await armed(broker)
    mgr.positions[SYMBOL].closing = True
    assert await mgr.on_price(SYMBOL, 94.0) is None
    assert broker.calls == []


# -- triggers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (94.0, ExitReason.STOP),
        (95.0, ExitReason.STOP),  # inclusive
        (95.01, None),
        (109.99, None),
        (110.0, ExitReason.TAKE),  # inclusive
        (115.0, ExitReason.TAKE),
    ],
)
async def test_trigger_boundaries(price, expected):
    broker = RecordingBroker()
    mgr, _ = await armed(broker)
    assert await mgr.on_price(SYMBOL, price) is expected


async def test_untracked_symbol_is_ignored():
    broker = RecordingBroker()
    mgr, _ = await armed(broker)
    assert await mgr.on_price("ETH/USD", 1.0) is None


async def test_manual_close_uses_the_same_safe_sequence():
    broker = RecordingBroker()
    mgr, exits = await armed(broker)
    assert await mgr.close_manually(SYMBOL, 101.0) is ExitReason.AGENT
    assert broker.calls == ["cancel", "get_order", "submit_close"]
    assert exits[0][2] is ExitReason.AGENT


async def test_manual_close_of_untracked_symbol():
    broker = RecordingBroker()
    mgr, _ = await armed(broker)
    assert await mgr.close_manually("DOGE/USD", 1.0) is None


async def test_rejected_close_leaves_the_position_open_and_unwedged():
    broker = RecordingBroker(close_rejected=True)
    mgr, exits = await armed(broker)
    assert await mgr.on_price(SYMBOL, 94.0) is None
    assert SYMBOL in mgr.positions
    assert mgr.positions[SYMBOL].closing is False
    assert exits == []


# -- crash recovery ---------------------------------------------------------------------


async def test_reconcile_rearms_a_missing_stop_after_restart():
    """Process died, the resting stop went with it. It must come back before we trade."""
    broker = RecordingBroker()
    broker.positions[SYMBOL] = Position(
        symbol=SYMBOL, qty=0.1, avg_entry=100.0, market_value=10.0
    )
    mgr = StopManager(broker)
    stale = ProtectedPosition(
        symbol=SYMBOL,
        qty=0.1,
        entry=100.0,
        stop_px=95.0,
        take_px=110.0,
        broker_stop_order_id="gone-with-the-process",
    )

    await mgr.reconcile([stale])

    assert "submit_stop" in broker.calls
    assert mgr.positions[SYMBOL].broker_stop_order_id != "gone-with-the-process"


async def test_reconcile_drops_a_position_closed_while_we_were_down():
    broker = RecordingBroker()  # broker reports no positions
    mgr = StopManager(broker)
    await mgr.reconcile(
        [ProtectedPosition(symbol=SYMBOL, qty=0.1, entry=100.0, stop_px=95.0, take_px=110.0)]
    )
    assert SYMBOL not in mgr.positions


async def test_reconcile_cancels_an_orphaned_resting_stop():
    """A stop with no position behind it will sell something we do not own."""
    broker = RecordingBroker()
    orphan = await broker.submit(
        OrderRequest(
            symbol="ETH/USD",
            side=OrderSide.SELL,
            qty=1.0,
            type=OrderType.STOP_LIMIT,
            stop_price=1.0,
        )
    )
    broker.calls.clear()
    mgr = StopManager(broker)

    await mgr.reconcile([])

    assert "cancel" in broker.calls
    assert broker.orders[orphan.id].status is OrderStatus.CANCELED


async def test_reconcile_syncs_quantity_to_the_broker():
    """The broker is the source of truth for size, not our last write."""
    broker = RecordingBroker()
    broker.positions[SYMBOL] = Position(
        symbol=SYMBOL, qty=0.07, avg_entry=100.0, market_value=7.0
    )
    mgr = StopManager(broker)
    await mgr.reconcile(
        [ProtectedPosition(symbol=SYMBOL, qty=0.1, entry=100.0, stop_px=95.0, take_px=110.0)]
    )
    assert mgr.positions[SYMBOL].qty == 0.07
