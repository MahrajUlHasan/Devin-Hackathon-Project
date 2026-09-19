"""ExecutionEngine -- turns an approved RiskDecision into a live order and arms its stop.

Deterministic on purpose. There is no language model in this path; by the time we are
here the decision has been made and ruled on, and the only remaining questions have
exact answers.
"""

from __future__ import annotations

import logging

from budapilot.contracts import (
    Action,
    BrokerPort,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskDecision,
)
from budapilot.execution.stops import StopManager

log = logging.getLogger("budapilot.execution")


class ExecutionEngine:
    def __init__(self, broker: BrokerPort, stops: StopManager, journal: object | None = None):
        self.broker = broker
        self.stops = stops
        self.journal = journal

    async def execute(self, decision: RiskDecision) -> Order | None:
        """Submit an approved decision. Returns the order, or None if it did not fill."""
        if not decision.approved or decision.qty <= 0:
            return None

        side = OrderSide.BUY if decision.action is Action.BUY else OrderSide.SELL

        # A SELL here is an agent-initiated close. Route it through the StopManager so
        # the resting stop is cancelled first -- bypassing it would double-sell.
        if side is OrderSide.SELL and decision.symbol in self.stops.positions:
            await self.stops.close_manually(decision.symbol, decision.entry_ref)
            return None

        order = await self.broker.submit(
            OrderRequest(
                symbol=decision.symbol,
                side=side,
                qty=decision.qty,
                type=OrderType.MARKET,
                time_in_force="gtc",
            )
        )
        if self.journal is not None:
            self.journal.log_order(order, intent="entry")  # type: ignore[attr-defined]

        if order.status is OrderStatus.REJECTED:
            log.error("Entry rejected for %s.", decision.symbol)
            return order

        # Arm protection only on an actual fill. An accepted-but-unfilled order has no
        # position to protect, and arming a stop against it would sell what we do not own.
        if (
            order.status is OrderStatus.FILLED
            and side is OrderSide.BUY
            and decision.stop_px
            and decision.take_px
        ):
            entry = order.filled_avg_price or decision.entry_ref
            await self.stops.protect(
                symbol=decision.symbol,
                qty=order.filled_qty,
                entry=entry,
                stop_px=decision.stop_px,
                take_px=decision.take_px,
            )
        elif order.status is not OrderStatus.FILLED:
            log.warning(
                "Order %s is %s, not filled; no stop armed.", order.id, order.status.value
            )

        return order
