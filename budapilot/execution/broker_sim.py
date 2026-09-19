"""In-process simulated broker. Used by --demo-safe, by every test, and as the
fallback when Alpaca credentials are absent.

It mimics the parts of Alpaca's crypto behaviour that matter to us, including the
restrictions: simple order class only, long-only, and resting stop_limit orders that
sit until the price crosses them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from budapilot.contracts import (
    AssetSpec,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
    utcnow,
)

DEFAULT_ASSETS: dict[str, AssetSpec] = {
    "BTC/USD": AssetSpec(symbol="BTC/USD", min_order_size=1.0, min_trade_increment=1e-9),
    "ETH/USD": AssetSpec(symbol="ETH/USD", min_order_size=1.0, min_trade_increment=1e-9),
    "SOL/USD": AssetSpec(symbol="SOL/USD", min_order_size=1.0, min_trade_increment=1e-9),
    "LTC/USD": AssetSpec(symbol="LTC/USD", min_order_size=1.0, min_trade_increment=1e-9),
    "AVAX/USD": AssetSpec(symbol="AVAX/USD", min_order_size=1.0, min_trade_increment=1e-9),
    "DOGE/USD": AssetSpec(symbol="DOGE/USD", min_order_size=1.0, min_trade_increment=1.0),
}


@dataclass
class SimBroker:
    """A long-only crypto broker with instant market fills and resting stop_limits."""

    cash: float = 100_000.0
    fee_bps: float = 10.0  # 0.10%, roughly Alpaca's crypto taker fee
    prices: dict[str, float] = field(default_factory=dict)
    _positions: dict[str, Position] = field(default_factory=dict)
    _orders: dict[str, Order] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # -- price feed -------------------------------------------------------------------

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price
        pos = self._positions.get(symbol)
        if pos:
            pos.current_price = price
            pos.market_value = pos.qty * price
            pos.unrealized_pl = (price - pos.avg_entry) * pos.qty
        self._trigger_resting(symbol, price)

    def _trigger_resting(self, symbol: str, price: float) -> None:
        """Fire any resting stop_limit whose stop has been crossed."""
        for order in list(self._orders.values()):
            if (
                order.symbol == symbol
                and order.type is OrderType.STOP_LIMIT
                and order.status in (OrderStatus.NEW, OrderStatus.ACCEPTED)
                and order.stop_price is not None
                and order.side is OrderSide.SELL
                and price <= order.stop_price
            ):
                self._fill(order, price)

    # -- BrokerPort --------------------------------------------------------------------

    async def get_portfolio(self) -> PortfolioState:
        equity = self.cash + sum(p.market_value for p in self._positions.values())
        return PortfolioState(
            equity=equity, cash=self.cash, positions=dict(self._positions)
        )

    async def get_asset(self, symbol: str) -> AssetSpec:
        return DEFAULT_ASSETS.get(
            symbol, AssetSpec(symbol=symbol, min_order_size=1.0, min_trade_increment=1e-9)
        )

    async def submit(self, req: OrderRequest) -> Order:
        order = Order(
            id=f"sim-{next(self._ids)}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            type=req.type,
            status=OrderStatus.ACCEPTED,
            stop_price=req.stop_price,
            limit_price=req.limit_price,
            submitted_at=utcnow(),
        )
        self._orders[order.id] = order

        if req.type is OrderType.STOP_LIMIT:
            return order  # rests until triggered

        price = req.limit_price or self.prices.get(req.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            return order
        self._fill(order, price)
        return order

    def _fill(self, order: Order, price: float) -> None:
        fee = price * order.qty * self.fee_bps / 10_000

        if order.side is OrderSide.BUY:
            cost = price * order.qty + fee
            if cost > self.cash:
                order.status = OrderStatus.REJECTED
                return
            self.cash -= cost
            existing = self._positions.get(order.symbol)
            if existing:
                total_qty = existing.qty + order.qty
                existing.avg_entry = (
                    existing.avg_entry * existing.qty + price * order.qty
                ) / total_qty
                existing.qty = total_qty
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    qty=order.qty,
                    avg_entry=price,
                    market_value=order.qty * price,
                    current_price=price,
                )
        else:
            pos = self._positions.get(order.symbol)
            if pos is None or pos.qty <= 0:
                order.status = OrderStatus.REJECTED  # long-only: nothing to sell
                return
            qty = min(order.qty, pos.qty)
            self.cash += price * qty - fee
            pos.qty -= qty
            if pos.qty <= 1e-12:
                del self._positions[order.symbol]
            order.qty = qty

        order.filled_qty = order.qty
        order.filled_avg_price = price
        order.status = OrderStatus.FILLED

    async def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status not in (OrderStatus.NEW, OrderStatus.ACCEPTED):
            return False
        order.status = OrderStatus.CANCELED
        return True

    async def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    async def list_open_orders(self) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.NEW, OrderStatus.ACCEPTED)
        ]
