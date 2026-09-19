"""Freeze real market data to disk so --demo-safe is real.

Run this once while the network is good:

    python scripts/fetch_fixtures.py

It pulls 5-minute bars and 24h of headlines for the whole watchlist into fixtures/ and
commits nothing -- you do that. After this, a rate limit, an outage or a hostile venue
cannot touch the demo, because the demo never asks them anything.

Without Alpaca keys the bars still work (crypto market data is unauthenticated); only
the news needs credentials.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budapilot.config import WATCHLIST, settings  # noqa: E402
from budapilot.data.fixtures import save_bars, save_news  # noqa: E402

BAR_LIMIT = 2000  # ~7 days of 5-minute bars


async def main() -> int:
    from budapilot.data.alpaca_crypto import AlpacaCryptoFeed

    feed = AlpacaCryptoFeed()
    if not settings.has_alpaca:
        print("! No Alpaca keys: bars will be fetched, news will be empty.\n")

    failures = 0
    for symbol in WATCHLIST:
        try:
            bars = await feed.get_bars(symbol, limit=BAR_LIMIT)
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol:<10} BARS FAILED: {exc}")
            failures += 1
            continue

        if not bars:
            print(f"  {symbol:<10} no bars returned")
            failures += 1
            continue

        save_bars(symbol, bars)
        news = await feed.get_news(symbol, limit=50)
        save_news(symbol, news)
        print(
            f"  {symbol:<10} {len(bars):>5} bars "
            f"({bars[0].ts:%m-%d %H:%M} -> {bars[-1].ts:%m-%d %H:%M}), "
            f"{len(news):>3} headlines"
        )

    print(
        "\nDone." if not failures else f"\nDone with {failures} failure(s).",
        "Verify with: python -m budapilot --demo-safe",
    )
    return 1 if failures == len(WATCHLIST) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
