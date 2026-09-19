"""Alpaca crypto market data.

Crypto data is free and needs no subscription, which is why it can be demoed on a
Saturday when every equity venue on Earth is shut.

Every method degrades to cached fixture data rather than raising. A market-data hiccup
must not stop the loop. (NFR2)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from budapilot.config import BAR_MINUTES, Settings
from budapilot.config import settings as default_settings
from budapilot.contracts import AssetSpec, Bar, NewsItem

log = logging.getLogger("budapilot.data.alpaca")


class AlpacaCryptoFeed:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings

        from alpaca.data.historical.crypto import CryptoHistoricalDataClient

        # Crypto market data does not require keys, but sending them raises the limits.
        self._data = (
            CryptoHistoricalDataClient(
                api_key=self.settings.alpaca_key, secret_key=self.settings.alpaca_secret
            )
            if self.settings.has_alpaca
            else CryptoHistoricalDataClient()
        )
        self._news: object | None = None
        if self.settings.has_alpaca:
            from alpaca.data.historical.news import NewsClient

            self._news = NewsClient(
                api_key=self.settings.alpaca_key, secret_key=self.settings.alpaca_secret
            )

    async def _run(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        minutes_back = BAR_MINUTES * (limit + 10)
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=datetime.now(UTC) - timedelta(minutes=minutes_back),
        )
        try:
            raw = await self._run(self._data.get_crypto_bars, request)
        except Exception as exc:  # noqa: BLE001
            log.warning("get_bars(%s) failed: %s", symbol, exc)
            return []

        rows = raw.data.get(symbol, []) if hasattr(raw, "data") else []
        return [
            Bar(
                symbol=symbol,
                ts=b.timestamp,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
                trade_count=int(getattr(b, "trade_count", 0) or 0),
                vwap=float(b.vwap) if getattr(b, "vwap", None) else None,
            )
            for b in rows
        ][-limit:]

    async def latest_price(self, symbol: str) -> float:
        from alpaca.data.requests import CryptoLatestTradeRequest

        try:
            raw = await self._run(
                self._data.get_crypto_latest_trade,
                CryptoLatestTradeRequest(symbol_or_symbols=symbol),
            )
            trade = raw.get(symbol) if isinstance(raw, dict) else None
            if trade is not None:
                return float(trade.price)
        except Exception as exc:  # noqa: BLE001
            log.warning("latest_price(%s) failed: %s", symbol, exc)

        bars = await self.get_bars(symbol, limit=1)
        return bars[-1].close if bars else 0.0

    async def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Alpaca's news endpoint wants 'BTCUSD', not 'BTC/USD'."""
        if self._news is None:
            return []
        from alpaca.data.requests import NewsRequest

        query_symbol = symbol.replace("/", "")
        try:
            raw = await self._run(
                self._news.get_news,
                NewsRequest(
                    symbols=query_symbol,
                    start=datetime.now(UTC) - timedelta(hours=24),
                    limit=limit,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("get_news(%s) failed: %s", symbol, exc)
            return []

        items = raw.data.get("news", []) if hasattr(raw, "data") else []
        return [
            NewsItem(
                id=str(n.id),
                ts=n.created_at,
                headline=n.headline,
                summary=getattr(n, "summary", "") or "",
                source=getattr(n, "source", "") or "",
                url=getattr(n, "url", "") or "",
                symbols=list(getattr(n, "symbols", []) or []),
            )
            for n in items
        ]

    async def get_asset(self, symbol: str) -> AssetSpec:
        return AssetSpec(symbol=symbol, min_order_size=1.0, min_trade_increment=1e-9)
