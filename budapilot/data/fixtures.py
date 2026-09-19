"""Frozen fixtures: the data layer that makes --demo-safe real.

The highest-leverage fifteen minutes in the build was spent producing these. A rate
limit, a network outage or a venue hiccup at hour four cannot touch a demo that reads
from disk, and "verified with the wifi off" is a claim you can only make if this exists.

``SyntheticFeed`` is the last line of defence: if there are no fixtures either, it
generates a deterministic random walk so the loop still runs and the dashboard still
renders. Seeded, so two runs produce identical output.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from budapilot.config import BAR_MINUTES, WATCHLIST
from budapilot.contracts import AssetSpec, Bar, NewsItem

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"

# Rough spot levels, only used to make synthetic data look plausible on screen.
SEED_PRICES: dict[str, float] = {
    "BTC/USD": 64_000.0,
    "ETH/USD": 3_100.0,
    "SOL/USD": 145.0,
    "LTC/USD": 82.0,
    "AVAX/USD": 27.0,
    "DOGE/USD": 0.14,
}
ANNUAL_VOL: dict[str, float] = {
    "BTC/USD": 0.55,
    "ETH/USD": 0.70,
    "SOL/USD": 0.95,
    "LTC/USD": 0.75,
    "AVAX/USD": 0.95,
    "DOGE/USD": 1.10,
}


def _bars_path(symbol: str) -> Path:
    return FIXTURE_DIR / "bars" / f"{symbol.replace('/', '-')}.json"


def _news_path(symbol: str) -> Path:
    return FIXTURE_DIR / "news" / f"{symbol.replace('/', '-')}.json"


def save_bars(symbol: str, bars: list[Bar]) -> Path:
    path = _bars_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([b.model_dump(mode="json") for b in bars], indent=1))
    return path


def save_news(symbol: str, items: list[NewsItem]) -> Path:
    path = _news_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([n.model_dump(mode="json") for n in items], indent=1))
    return path


def load_bars(symbol: str) -> list[Bar]:
    path = _bars_path(symbol)
    if not path.exists():
        return []
    return [Bar.model_validate(r) for r in json.loads(path.read_text())]


def load_news(symbol: str) -> list[NewsItem]:
    path = _news_path(symbol)
    if not path.exists():
        return []
    return [NewsItem.model_validate(r) for r in json.loads(path.read_text())]


def synthesize_bars(symbol: str, count: int = 400, seed: int | None = None) -> list[Bar]:
    """Deterministic GBM random walk. Same seed, same series, every time."""
    rng = random.Random(seed if seed is not None else hash(symbol) & 0xFFFF)
    price = SEED_PRICES.get(symbol, 100.0)
    # Convert annualised vol to per-bar vol.
    bars_per_year = 365 * 24 * 60 / BAR_MINUTES
    sigma = ANNUAL_VOL.get(symbol, 0.8) / math.sqrt(bars_per_year)
    drift = rng.uniform(-0.6, 0.6) * sigma

    start = datetime.now(UTC) - timedelta(minutes=BAR_MINUTES * count)
    out: list[Bar] = []
    for i in range(count):
        shock = rng.gauss(drift, sigma)
        open_px = price
        close_px = max(price * (1 + shock), 1e-8)
        wick = abs(rng.gauss(0, sigma)) * price
        out.append(
            Bar(
                symbol=symbol,
                ts=start + timedelta(minutes=BAR_MINUTES * i),
                open=round(open_px, 8),
                high=round(max(open_px, close_px) + wick, 8),
                low=round(max(min(open_px, close_px) - wick, 1e-9), 8),
                close=round(close_px, 8),
                volume=round(abs(rng.gauss(1000, 300)), 2),
                trade_count=rng.randint(50, 500),
            )
        )
        price = close_px
    return out


SYNTHETIC_HEADLINES = [
    ("{asset} network activity hits a three-month high", "desk-wire"),
    ("Analysts split on {asset} after the weekend range", "desk-wire"),
    ("{asset} open interest climbs as funding turns positive", "desk-wire"),
    ("Regulatory clarity for {asset} still pending, says counsel", "desk-wire"),
    ("{asset} spot volumes thin into the weekend", "desk-wire"),
]


def synthesize_news(symbol: str, count: int = 5) -> list[NewsItem]:
    asset = symbol.split("/")[0]
    now = datetime.now(UTC)
    return [
        NewsItem(
            id=f"syn-{symbol}-{i}",
            ts=now - timedelta(hours=2 * i + 1),
            headline=template.format(asset=asset),
            summary="Synthetic fixture headline; not real news.",
            source=source,
            symbols=[symbol],
        )
        for i, (template, source) in enumerate(SYNTHETIC_HEADLINES[:count])
    ]


class FixtureFeed:
    """Offline MarketDataPort + NewsPort. Never touches the network. (FR11)"""

    def __init__(self, symbols: list[str] | None = None, *, synthesize: bool = True) -> None:
        self.symbols = symbols or WATCHLIST
        self._bars: dict[str, list[Bar]] = {}
        self._news: dict[str, list[NewsItem]] = {}
        self.synthetic = False

        for symbol in self.symbols:
            bars = load_bars(symbol)
            if not bars and synthesize:
                bars = synthesize_bars(symbol)
                self.synthetic = True
            self._bars[symbol] = bars

            news = load_news(symbol)
            if not news and synthesize:
                news = synthesize_news(symbol)
            self._news[symbol] = news

        # Replay cursor: start partway in so indicators have history from bar one.
        self._cursor = min((len(b) for b in self._bars.values() if b), default=0)
        self._cursor = max(self._cursor - 60, 60)

    def advance(self) -> bool:
        """Step the replay clock one bar. False when the fixtures run out."""
        if self._cursor >= max((len(b) for b in self._bars.values()), default=0):
            return False
        self._cursor += 1
        return True

    async def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        return self._bars.get(symbol, [])[: self._cursor][-limit:]

    async def latest_price(self, symbol: str) -> float:
        window = self._bars.get(symbol, [])[: self._cursor]
        return window[-1].close if window else 0.0

    async def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        return self._news.get(symbol, [])[:limit]

    async def get_asset(self, symbol: str) -> AssetSpec:
        from budapilot.execution.broker_sim import DEFAULT_ASSETS

        return DEFAULT_ASSETS.get(
            symbol, AssetSpec(symbol=symbol, min_order_size=1.0, min_trade_increment=1e-9)
        )
