"""Compact renderers for agent prompts.

Agents never see a raw price series. They see these lines. Keeping the representation
tight is what keeps latency and cost down, and it is also what makes the reasoning
auditable -- what the model saw is exactly what is written to the journal.
"""

from __future__ import annotations

from budapilot.contracts import FeatureBundle, NewsItem, PortfolioState


def _n(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def features_line(f: FeatureBundle) -> str:
    return (
        f"{f.symbol} | px={_n(f.close, 4)} trend={_n(f.trend_score)} "
        f"rsi={_n(f.rsi_14, 1)} macd_h={_n(f.macd_hist, 4)} ema={f.ema_cross} "
        f"atr%={_n((f.atr_pct or 0) * 100, 2)} volz={_n(f.vol_zscore_20)} "
        f"ret1/5/20={_n((f.ret_1 or 0) * 100)}%/{_n((f.ret_5 or 0) * 100)}%/"
        f"{_n((f.ret_20 or 0) * 100)}% "
        f"fromHi20={_n((f.dist_from_high_20 or 0) * 100)}% "
        f"fromLo20={_n((f.dist_from_low_20 or 0) * 100)}%"
    )


def features_block(f: FeatureBundle) -> str:
    return "\n".join(
        [
            f"Symbol: {f.symbol}",
            f"Price: {_n(f.close, 4)}",
            f"SMA20/50: {_n(f.sma_20, 4)} / {_n(f.sma_50, 4)}",
            f"EMA12/26: {_n(f.ema_12, 4)} / {_n(f.ema_26, 4)}  cross={f.ema_cross}",
            f"RSI(14): {_n(f.rsi_14, 1)}",
            f"MACD: {_n(f.macd, 4)}  signal={_n(f.macd_signal, 4)}  hist={_n(f.macd_hist, 4)}",
            f"ATR(14): {_n(f.atr_14, 4)}  ({_n((f.atr_pct or 0) * 100, 2)}% of price)",
            f"Volume z-score(20): {_n(f.vol_zscore_20)}",
            f"Returns 1/5/20 bars: {_n((f.ret_1 or 0) * 100)}% / "
            f"{_n((f.ret_5 or 0) * 100)}% / {_n((f.ret_20 or 0) * 100)}%",
            f"Distance from 20-bar high: {_n((f.dist_from_high_20 or 0) * 100)}%",
            f"Distance from 20-bar low: {_n((f.dist_from_low_20 or 0) * 100)}%",
            f"Composite trend score: {_n(f.trend_score)}",
        ]
    )


def news_block(items: list[NewsItem], limit: int = 15) -> str:
    if not items:
        return "(no headlines in the last 24h)"
    return "\n".join(
        f"[{i + 1}] {n.ts:%Y-%m-%d %H:%M} ({n.source or 'unknown'}) {n.headline}"
        for i, n in enumerate(items[:limit])
    )


def portfolio_block(p: PortfolioState) -> str:
    lines = [
        f"Equity: ${p.equity:,.2f}",
        f"Cash: ${p.cash:,.2f}",
        f"Open positions: {p.open_count}",
        f"Gross exposure: {p.exposure_pct * 100:.1f}% of equity",
    ]
    for pos in p.positions.values():
        lines.append(
            f"  - {pos.symbol}: qty={pos.qty:.6f} entry={pos.avg_entry:.4f} "
            f"mv=${pos.market_value:,.2f} upl=${pos.unrealized_pl:,.2f}"
        )
    return "\n".join(lines)
