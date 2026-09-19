"""Hand-rolled indicators. No pandas-ta.

That library breaks on numpy/pandas version bumps and a dependency resolution failure
at hour one of a five-hour build is unrecoverable. These are forty lines of pandas and
they are testable against hand-checked values.

Conventions:
- RSI and ATR use Wilder's smoothing (``ewm(alpha=1/n, adjust=False)``), which is what
  every charting package means by "RSI(14)". Using a simple mean here is the classic
  way to get numbers that disagree with TradingView by a few points.
- Every function returns a Series aligned to the input index, NaN-padded at the front.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from budapilot.contracts import Bar, FeatureBundle


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        {
            "ts": [b.ts for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    ).set_index("ts")
    return frame.sort_index()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average: seed with the SMA of the first n values, then smooth.

    This is NOT the same as ``ewm(alpha=1/n, adjust=False)``, which seeds the recursion
    at the first observation. The two converge on long histories but disagree sharply
    on short ones -- which is precisely the case where a cold-started agent would read
    the number. Every charting package means *this* by RSI(14) and ATR(14).
    """
    values = series.to_numpy(dtype="float64")
    out = np.full(values.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size < n:
        return pd.Series(out, index=series.index)

    start = valid[n - 1]
    prev = float(np.mean(values[valid[:n]]))
    out[start] = prev
    for i in range(start + 1, len(values)):
        if np.isnan(values[i]):
            continue
        prev = (prev * (n - 1) + values[i]) / n
        out[i] = prev
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0.0), n)
    avg_loss = wilder_smooth(-delta.clip(upper=0.0), n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # All-gain windows give avg_loss == 0 -> RSI is 100 by definition, not NaN.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # The first true range has no previous close, so drop it before smoothing --
    # including it biases the seed on short histories.
    return wilder_smooth(true_range(high, low, close).iloc[1:], n).reindex(close.index)


def volume_zscore(volume: pd.Series, n: int = 20) -> pd.Series:
    mean = volume.rolling(n, min_periods=n).mean()
    std = volume.rolling(n, min_periods=n).std(ddof=0)
    return (volume - mean) / std.replace(0.0, np.nan)


def realized_vol_percentile(close: pd.Series, window: int = 20, lookback: int = 2000) -> float:
    """Where current realized vol sits in its own recent distribution, 0-100."""
    returns = close.pct_change()
    vol = returns.rolling(window, min_periods=window).std(ddof=0)
    recent = vol.tail(lookback).dropna()
    if len(recent) < 2:
        return 50.0
    current = recent.iloc[-1]
    return float((recent <= current).mean() * 100.0)


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def composite_trend_score(
    *,
    ema_12: float | None,
    ema_26: float | None,
    close: float,
    rsi_14: float | None,
    macd_hist: float | None,
    ret_5: float | None,
    ret_20: float | None,
    atr_pct: float | None,
) -> float:
    """Deterministic trend strength in roughly [-3, +3]. The Scout agent ranks on this.

    Each term is normalised by ATR or bounded, so a high-volatility symbol does not
    dominate the ranking simply by moving more.
    """
    score = 0.0
    scale = atr_pct if atr_pct and atr_pct > 0 else 0.01

    if ema_12 is not None and ema_26 is not None and close > 0:
        score += float(np.clip((ema_12 - ema_26) / close / scale, -1.5, 1.5))
    if macd_hist is not None and close > 0:
        score += float(np.clip(macd_hist / close / scale, -1.0, 1.0))
    if ret_5 is not None:
        score += float(np.clip(ret_5 / scale / 5, -1.0, 1.0)) * 0.5
    if ret_20 is not None:
        score += float(np.clip(ret_20 / scale / 10, -1.0, 1.0)) * 0.5
    if rsi_14 is not None:
        # Reward strength, but penalise a stretched tape rather than rewarding it.
        score += 0.5 if 55 <= rsi_14 <= 70 else (-0.5 if rsi_14 > 80 or rsi_14 < 30 else 0.0)

    return round(score, 4)


def compute_features(symbol: str, bars: list[Bar]) -> FeatureBundle:
    """Build the one representation any agent ever sees of the market.

    Short histories are not an error: every indicator degrades to None and
    ``FeatureBundle.is_tradeable()`` returns False, which the risk engine treats as a
    hard reject. Nothing downstream has to special-case a cold start.
    """
    frame = bars_to_frame(bars)
    if frame.empty:
        return FeatureBundle(symbol=symbol, ts=pd.Timestamp.utcnow().to_pydatetime(), close=0.0)

    close = frame["close"]
    high, low, volume = frame["high"], frame["low"], frame["volume"]

    ema_12, ema_26 = _last(ema(close, 12)), _last(ema(close, 26))
    macd_line, macd_signal, macd_hist = macd(close)
    atr_14 = _last(atr(high, low, close, 14))
    last_close = float(close.iloc[-1])
    rsi_14 = _last(rsi(close, 14))

    def ret(n: int) -> float | None:
        if len(close) <= n or close.iloc[-1 - n] == 0:
            return None
        return float(close.iloc[-1] / close.iloc[-1 - n] - 1.0)

    high_20 = _last(high.rolling(20, min_periods=20).max())
    low_20 = _last(low.rolling(20, min_periods=20).min())
    atr_pct = (atr_14 / last_close) if (atr_14 and last_close) else None

    cross = "NONE"
    if ema_12 is not None and ema_26 is not None:
        cross = "GOLDEN" if ema_12 > ema_26 else "DEATH"

    ret_5, ret_20 = ret(5), ret(20)

    return FeatureBundle(
        symbol=symbol,
        ts=frame.index[-1].to_pydatetime(),
        close=last_close,
        sma_20=_last(sma(close, 20)),
        sma_50=_last(sma(close, 50)),
        ema_12=ema_12,
        ema_26=ema_26,
        ema_cross=cross,  # type: ignore[arg-type]
        rsi_14=rsi_14,
        macd=_last(macd_line),
        macd_signal=_last(macd_signal),
        macd_hist=_last(macd_hist),
        atr_14=atr_14,
        atr_pct=atr_pct,
        vol_zscore_20=_last(volume_zscore(volume, 20)),
        ret_1=ret(1),
        ret_5=ret_5,
        ret_20=ret_20,
        dist_from_high_20=(last_close / high_20 - 1.0) if high_20 else None,
        dist_from_low_20=(last_close / low_20 - 1.0) if low_20 else None,
        trend_score=composite_trend_score(
            ema_12=ema_12,
            ema_26=ema_26,
            close=last_close,
            rsi_14=rsi_14,
            macd_hist=_last(macd_hist),
            ret_5=ret_5,
            ret_20=ret_20,
            atr_pct=atr_pct,
        ),
    )
