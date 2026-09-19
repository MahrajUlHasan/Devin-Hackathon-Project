"""StopManager -- protective exits for a venue that will not hold them for us.

Alpaca crypto accepts ``order_class=simple`` only. No bracket, no OCO, no OTO. So the
plan's "attach stop and take-profit as bracket legs" is not buildable and the protection
has to live here. Two layers, deliberately redundant:

1. **Resting stop_limit at the broker.** Submitted immediately after entry. It survives
   this process dying, which is the failure the software stop cannot cover. It is the
   backstop, not the primary.
2. **Software stop in-process.** ``on_price`` is driven by the trade stream and checks
   stop and take on every tick. It is the primary because it is precise and because the
   venue has no take-profit order type for us at all.

The race that matters
---------------------
Both layers can fire on the same tick. If we submit the closing market order first and
then cancel the resting stop, the stop can fill in between and we sell the position
twice -- the second sell being a short the venue rejects, or worse, an unintended
position. So the order is always:

    cancel resting stop -> CONFIRM the cancel -> submit the close

and if the cancel cannot be confirmed we do **not** close. We leave the broker's stop to
do its job and try again on the next tick. Refusing to act on an unconfirmed cancel is
the whole point; ``test_stops.py`` asserts this sequence explicitly.

``closing`` guards re-entry, because ticks arrive faster than the round trip to Alpaca.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from budapilot.contracts import (
    BrokerPort,
    ExitReason,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    ProtectedPosition,
)

log = logging.getLogger("budapilot.stops")

# Resting stop_limit needs a limit below the stop or it may never fill in a fast drop.
STOP_LIMIT_SLIP_BPS = 50.0  # 0.50%

ExitCallback = Callable[[ProtectedPosition, float, ExitReason], Awaitable[None]]


class StopManager:
    def __init__(
        self,
        broker: BrokerPort,
        *,
        on_exit: ExitCallback | None = None,
        journal: object | None = None,
    ) -> None:
        self.broker = broker
        self.on_exit = on_exit
        self.journal = journal
        self.positions: dict[str, ProtectedPosition] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, symbol: str) -> asyncio.Lock:
        return self._locks.setdefault(symbol, asyncio.Lock())

    def _persist(self, pos: ProtectedPosition) -> None:
        if self.journal is not None:
            self.journal.upsert_protected(pos)  # type: ignore[attr-defined]

    def _forget(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        if self.journal is not None:
            self.journal.delete_protected(symbol)  # type: ignore[attr-defined]

    # -- arming -------------------------------------------------------------------------

    async def protect(
        self, symbol: str, qty: float, entry: float, stop_px: float, take_px: float
    ) -> ProtectedPosition:
        """Register a filled position and arm the broker-side backstop."""
        pos = ProtectedPosition(
            symbol=symbol, qty=qty, entry=entry, stop_px=stop_px, take_px=take_px
        )
        self.positions[symbol] = pos
        await self._arm_broker_stop(pos)
        self._persist(pos)
        return pos

    async def _arm_broker_stop(self, pos: ProtectedPosition) -> None:
        """Submit the resting stop_limit. A failure here is logged, not raised.

        If the venue refuses the resting order we still have the software stop; losing
        the backstop is a degradation, not a reason to abandon a filled position.
        """
        limit = pos.stop_px * (1 - STOP_LIMIT_SLIP_BPS / 10_000)
        try:
            order = await self.broker.submit(
                OrderRequest(
                    symbol=pos.symbol,
                    side=OrderSide.SELL,
                    qty=pos.qty,
                    type=OrderType.STOP_LIMIT,
                    stop_price=pos.stop_px,
                    limit_price=limit,
                    time_in_force="gtc",
                    client_order_id=f"stop-{pos.symbol.replace('/', '')}-{int(pos.entry * 1e6)}",
                )
            )
            if order.status is OrderStatus.REJECTED:
                log.warning("Resting stop rejected for %s; software stop only.", pos.symbol)
                return
            pos.broker_stop_order_id = order.id
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not arm resting stop for %s: %s", pos.symbol, exc)

    # -- the hot path -------------------------------------------------------------------

    async def on_price(self, symbol: str, price: float) -> ExitReason | None:
        """Called on every tick. Returns the reason if this tick closed the position."""
        pos = self.positions.get(symbol)
        if pos is None or pos.closing:
            return None

        if price <= pos.stop_px:
            reason = ExitReason.STOP
        elif price >= pos.take_px:
            reason = ExitReason.TAKE
        else:
            return None

        return await self._close(pos, price, reason)

    async def close_manually(self, symbol: str, price: float) -> ExitReason | None:
        pos = self.positions.get(symbol)
        if pos is None or pos.closing:
            return None
        return await self._close(pos, price, ExitReason.AGENT)

    async def _close(
        self, pos: ProtectedPosition, price: float, reason: ExitReason
    ) -> ExitReason | None:
        async with self._lock(pos.symbol):
            # Re-check under the lock: another tick may have closed this already.
            if pos.closing or pos.symbol not in self.positions:
                return None
            pos.closing = True
            self._persist(pos)

            try:
                # 1. Cancel the resting stop and CONFIRM it before selling anything.
                if pos.broker_stop_order_id:
                    if not await self._cancel_confirmed(pos.broker_stop_order_id):
                        # Unconfirmed cancel: the resting order may still be live and
                        # could fill. Selling now risks a double-sell. Back off and let
                        # the broker's own stop handle it; we retry on the next tick.
                        log.warning(
                            "Could not confirm cancel of resting stop %s for %s; "
                            "refusing to close to avoid a double-sell.",
                            pos.broker_stop_order_id,
                            pos.symbol,
                        )
                        pos.closing = False
                        self._persist(pos)
                        return None

                # 2. Only now submit the close.
                order = await self.broker.submit(
                    OrderRequest(
                        symbol=pos.symbol,
                        side=OrderSide.SELL,
                        qty=pos.qty,
                        type=OrderType.MARKET,
                        time_in_force="gtc",
                    )
                )
                if order.status is OrderStatus.REJECTED:
                    log.error("Close order rejected for %s; position still open.", pos.symbol)
                    pos.closing = False
                    self._persist(pos)
                    return None

                fill = order.filled_avg_price or price
                self._forget(pos.symbol)
                if self.on_exit:
                    await self.on_exit(pos, fill, reason)
                return reason

            except Exception as exc:  # noqa: BLE001 -- never kill the price loop
                log.exception("Exit failed for %s: %s", pos.symbol, exc)
                pos.closing = False
                self._persist(pos)
                return None

    async def _cancel_confirmed(self, order_id: str) -> bool:
        """Cancel, then verify. A `True` from cancel() is not proof the order is dead.

        An order that already filled is also 'not live' -- but that means the position
        is gone, so reporting failure here is correct: we must not sell again.
        """
        try:
            await self.broker.cancel(order_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel(%s) raised: %s", order_id, exc)

        order: Order | None = await self.broker.get_order(order_id)
        if order is None:
            return False
        return order.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED)

    # -- crash recovery -------------------------------------------------------------------

    async def reconcile(self, protected: list[ProtectedPosition]) -> None:
        """Rebuild state after a restart and re-arm any stop that went missing.

        Three cases: the broker has a position we tracked (re-arm if its stop is gone),
        the broker has no position we thought we had (it closed while we were down, so
        drop it), and a resting stop with no position (cancel the orphan).
        """
        portfolio = await self.broker.get_portfolio()
        open_orders = await self.broker.list_open_orders()
        live_stop_ids = {o.id for o in open_orders if o.type is OrderType.STOP_LIMIT}

        for pos in protected:
            broker_pos = portfolio.positions.get(pos.symbol)
            if broker_pos is None or broker_pos.qty <= 0:
                log.info("%s closed while we were down; dropping tracked stop.", pos.symbol)
                self._forget(pos.symbol)
                continue

            pos.qty = broker_pos.qty
            pos.closing = False
            self.positions[pos.symbol] = pos

            if pos.broker_stop_order_id not in live_stop_ids:
                log.warning("Resting stop missing for %s after restart; re-arming.", pos.symbol)
                await self._arm_broker_stop(pos)
            self._persist(pos)

        tracked_ids = {p.broker_stop_order_id for p in self.positions.values()}
        for order in open_orders:
            if order.type is OrderType.STOP_LIMIT and order.id not in tracked_ids:
                log.warning("Orphaned resting stop %s (%s); cancelling.", order.id, order.symbol)
                await self.broker.cancel(order.id)
