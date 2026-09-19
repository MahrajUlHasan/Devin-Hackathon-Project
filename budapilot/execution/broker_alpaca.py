"""Alpaca paper broker adapter.

Every venue restriction that bit us is enforced here rather than trusted to a prompt:
``order_class`` is never set (crypto is simple-only), TIF is gtc or ioc only, and the
live-trading path raises by construction.

alpaca-py is synchronous, so each call is pushed to a thread to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from budapilot.config import Settings
from budapilot.config import settings as default_settings
from budapilot.contracts import (
    AssetSpec,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)

log = logging.getLogger("budapilot.broker.alpaca")

_STATUS_MAP = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "pending_cancel": OrderStatus.ACCEPTED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "done_for_day": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELED,
    "suspended": OrderStatus.ACCEPTED,
    "stopped": OrderStatus.ACCEPTED,
    "calculated": OrderStatus.ACCEPTED,
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class AlpacaPaperBroker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.settings.assert_paper_only()  # NFR4: the live path raises
        if not self.settings.has_alpaca:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET are not set.")

        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key=self.settings.alpaca_key,
            secret_key=self.settings.alpaca_secret,
            paper=True,
        )
        self._asset_cache: dict[str, AssetSpec] = {}

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    # -- BrokerPort ---------------------------------------------------------------------

    async def get_portfolio(self) -> PortfolioState:
        account = await self._run(self._client.get_account)
        raw_positions = await self._run(self._client.get_all_positions)
        positions: dict[str, Position] = {}
        for p in raw_positions:
            symbol = getattr(p, "symbol", "")
            # Alpaca returns crypto positions as "BTCUSD"; the rest of the system
            # speaks "BTC/USD". Normalise at the boundary, nowhere else.
            if "/" not in symbol and symbol.endswith("USD"):
                symbol = f"{symbol[:-3]}/USD"
            positions[symbol] = Position(
                symbol=symbol,
                qty=_f(p.qty),
                avg_entry=_f(p.avg_entry_price),
                market_value=_f(p.market_value),
                unrealized_pl=_f(p.unrealized_pl),
                current_price=_f(p.current_price),
            )
        return PortfolioState(
            equity=_f(account.equity),
            cash=_f(account.cash),
            positions=positions,
        )

    async def get_asset(self, symbol: str) -> AssetSpec:
        """Fetch and cache the venue's size rules. Skipping this is a 422 generator."""
        if symbol in self._asset_cache:
            return self._asset_cache[symbol]
        try:
            asset = await self._run(self._client.get_asset, symbol)
            spec = AssetSpec(
                symbol=symbol,
                min_order_size=_f(getattr(asset, "min_order_size", None), 0.0),
                min_trade_increment=_f(getattr(asset, "min_trade_increment", None), 1e-9),
                price_increment=_f(getattr(asset, "price_increment", None), 0.01),
                tradable=bool(getattr(asset, "tradable", True)),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("get_asset(%s) failed (%s); using conservative defaults.", symbol, exc)
            spec = AssetSpec(symbol=symbol, min_order_size=1.0, min_trade_increment=1e-9)
        self._asset_cache[symbol] = spec
        return spec

    async def submit(self, req: OrderRequest) -> Order:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
        )

        side = AlpacaSide.BUY if req.side is OrderSide.BUY else AlpacaSide.SELL
        tif = TimeInForce.GTC if req.time_in_force == "gtc" else TimeInForce.IOC
        common = {
            "symbol": req.symbol,
            "qty": req.qty,
            "side": side,
            "time_in_force": tif,
            "client_order_id": req.client_order_id,
        }
        # Note: order_class is deliberately never set. Crypto is simple-only.
        if req.type is OrderType.MARKET:
            payload: Any = MarketOrderRequest(**common)
        elif req.type is OrderType.LIMIT:
            payload = LimitOrderRequest(limit_price=req.limit_price, **common)
        else:
            payload = StopLimitOrderRequest(
                stop_price=req.stop_price, limit_price=req.limit_price, **common
            )

        raw = await self._run(self._client.submit_order, payload)
        return self._to_order(raw)

    async def cancel(self, order_id: str) -> bool:
        try:
            await self._run(self._client.cancel_order_by_id, order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel(%s) failed: %s", order_id, exc)
            return False

    async def get_order(self, order_id: str) -> Order | None:
        try:
            return self._to_order(await self._run(self._client.get_order_by_id, order_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("get_order(%s) failed: %s", order_id, exc)
            return None

    async def list_open_orders(self) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            raw = await self._run(
                self._client.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            return [self._to_order(o) for o in raw]
        except Exception as exc:  # noqa: BLE001
            log.warning("list_open_orders failed: %s", exc)
            return []

    # -- mapping -------------------------------------------------------------------------

    def _to_order(self, raw: Any) -> Order:
        symbol = str(getattr(raw, "symbol", ""))
        if "/" not in symbol and symbol.endswith("USD"):
            symbol = f"{symbol[:-3]}/USD"
        status_raw = str(getattr(getattr(raw, "status", None), "value", raw.status)).lower()
        type_raw = str(getattr(getattr(raw, "order_type", None), "value", "market")).lower()
        side_raw = str(getattr(getattr(raw, "side", None), "value", "buy")).lower()
        return Order(
            id=str(raw.id),
            client_order_id=getattr(raw, "client_order_id", None),
            symbol=symbol,
            side=OrderSide.BUY if side_raw == "buy" else OrderSide.SELL,
            qty=_f(getattr(raw, "qty", 0)),
            filled_qty=_f(getattr(raw, "filled_qty", 0)),
            filled_avg_price=(
                _f(raw.filled_avg_price) if getattr(raw, "filled_avg_price", None) else None
            ),
            type=(
                OrderType(type_raw)
                if type_raw in {t.value for t in OrderType}
                else OrderType.MARKET
            ),
            status=_STATUS_MAP.get(status_raw, OrderStatus.ACCEPTED),
            stop_price=_f(raw.stop_price) if getattr(raw, "stop_price", None) else None,
            limit_price=_f(raw.limit_price) if getattr(raw, "limit_price", None) else None,
        )
