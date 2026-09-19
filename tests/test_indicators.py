"""Indicator tests against hand-checked values.

Wilder's smoothing is the thing most likely to be silently wrong: using a simple mean
for RSI or ATR produces numbers that look plausible but disagree with every charting
package. The canonical RSI series below is the standard worked example.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from budapilot.contracts import Bar
from budapilot.features.indicators import (
    atr,
    bars_to_frame,
    composite_trend_score,
    compute_features,
    ema,
    macd,
    realized_vol_percentile,
    rsi,
    sma,
    true_range,
    volume_zscore,
)

# Wilder's own worked example from "New Concepts in Technical Trading Systems".
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
]


def make_bars(closes, symbol="BTC/USD", volume=1000.0):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Bar(
            symbol=symbol,
            ts=start + timedelta(minutes=5 * i),
            open=c,
            high=c * 1.002,
            low=c * 0.998,
            close=c,
            volume=volume,
        )
        for i, c in enumerate(closes)
    ]


# -- individual indicators ----------------------------------------------------------


# Published RSI(14) values for the closes above, from Wilder's table.
WILDER_RSI = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97]


def test_rsi_matches_wilders_published_table():
    """The whole tail, not one value -- a wrong seed still matches at one point."""
    series = rsi(pd.Series(WILDER_CLOSES), 14)
    computed = series.dropna().tolist()
    assert len(computed) == len(WILDER_RSI)
    for got, expected in zip(computed, WILDER_RSI, strict=True):
        assert got == pytest.approx(expected, abs=0.1)


def test_rsi_seeding_is_wilder_not_a_plain_ewm():
    """Regression guard. ewm(alpha=1/n) seeds at the first observation and is wrong.

    On this 20-bar series the two disagree by ~15 RSI points, which is the difference
    between 'overbought' and 'neutral' for anything reading the number.
    """
    series = pd.Series(WILDER_CLOSES)
    delta = series.diff()
    naive_gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    naive_loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    naive = (100 - 100 / (1 + naive_gain / naive_loss)).iloc[-1]

    assert rsi(series, 14).iloc[-1] == pytest.approx(57.97, abs=0.1)
    assert abs(naive - 57.97) > 10  # the bug this replaced


def test_wilder_smooth_seeds_with_the_simple_mean():
    from budapilot.features.indicators import wilder_smooth

    values = pd.Series([2.0] * 14 + [16.0])
    smoothed = wilder_smooth(values, 14)
    assert smoothed.iloc[13] == pytest.approx(2.0)  # seed is the SMA
    assert smoothed.iloc[14] == pytest.approx((2.0 * 13 + 16.0) / 14)


def test_wilder_smooth_too_short():
    from budapilot.features.indicators import wilder_smooth

    assert wilder_smooth(pd.Series([1.0, 2.0]), 14).isna().all()


def test_rsi_is_100_when_every_bar_gains():
    """avg_loss is zero here; a naive implementation divides by zero and yields NaN."""
    value = rsi(pd.Series([float(i) for i in range(1, 30)]), 14).iloc[-1]
    assert value == pytest.approx(100.0)


def test_rsi_is_low_when_every_bar_loses():
    value = rsi(pd.Series([float(i) for i in range(30, 1, -1)]), 14).iloc[-1]
    assert value == pytest.approx(0.0, abs=1e-6)


def test_rsi_is_nan_before_the_window_fills():
    assert pd.isna(rsi(pd.Series([1.0, 2.0, 3.0]), 14).iloc[-1])


def test_sma_and_ema_basics():
    series = pd.Series([float(i) for i in range(1, 21)])
    assert sma(series, 20).iloc[-1] == pytest.approx(10.5)
    assert pd.isna(sma(series, 21).iloc[-1])
    # EMA weights recent values more, so it leads the SMA in an uptrend.
    assert ema(series, 12).iloc[-1] > sma(series, 12).iloc[-1]


def test_ema_of_a_constant_series_is_the_constant():
    assert ema(pd.Series([5.0] * 40), 12).iloc[-1] == pytest.approx(5.0)


def test_macd_histogram_is_line_minus_signal():
    series = pd.Series(np.linspace(100, 140, 80))
    line, signal, hist = macd(series)
    assert hist.iloc[-1] == pytest.approx(line.iloc[-1] - signal.iloc[-1])
    assert line.iloc[-1] > 0  # uptrend


def test_true_range_uses_the_previous_close():
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.0])
    close = pd.Series([9.5, 11.5])
    # Second bar gaps up: TR is high - prev_close = 12 - 9.5 = 2.5, not 12 - 11 = 1.
    assert true_range(high, low, close).iloc[-1] == pytest.approx(2.5)


def test_atr_of_a_constant_range_equals_that_range():
    high = pd.Series([101.0] * 40)
    low = pd.Series([99.0] * 40)
    close = pd.Series([100.0] * 40)
    assert atr(high, low, close, 14).iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_atr_is_positive_and_scales_with_volatility():
    calm = make_bars([100.0 + 0.1 * (i % 3) for i in range(60)])
    wild = make_bars([100.0 + 5.0 * (i % 3) for i in range(60)])
    f_calm = compute_features("A/USD", calm)
    f_wild = compute_features("A/USD", wild)
    assert f_wild.atr_14 > f_calm.atr_14 > 0


def test_volume_zscore_flags_a_spike():
    volume = pd.Series([1000.0] * 25 + [5000.0])
    assert volume_zscore(volume, 20).iloc[-1] > 3


def test_volume_zscore_is_nan_when_volume_is_flat():
    assert pd.isna(volume_zscore(pd.Series([1000.0] * 30), 20).iloc[-1])


def test_realized_vol_percentile_bounds():
    rng = np.random.default_rng(0)
    series = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    assert 0.0 <= realized_vol_percentile(series) <= 100.0


def test_realized_vol_percentile_with_too_little_data():
    assert realized_vol_percentile(pd.Series([1.0, 2.0])) == 50.0


# -- the bundle -----------------------------------------------------------------------


def test_compute_features_on_a_clean_uptrend():
    f = compute_features("BTC/USD", make_bars(list(np.linspace(100, 130, 80))))

    assert f.symbol == "BTC/USD"
    assert f.close == pytest.approx(130.0)
    assert f.ema_cross == "GOLDEN"
    assert f.rsi_14 > 70
    assert f.macd_hist is not None
    assert f.atr_14 > 0
    assert f.ret_20 > 0
    assert f.trend_score > 0
    assert f.is_tradeable()


def test_compute_features_on_a_downtrend():
    f = compute_features("BTC/USD", make_bars(list(np.linspace(130, 100, 80))))
    assert f.ema_cross == "DEATH"
    assert f.rsi_14 < 30
    assert f.trend_score < 0


def test_compute_features_with_no_bars_is_not_tradeable():
    f = compute_features("BTC/USD", [])
    assert f.close == 0.0
    assert not f.is_tradeable()


def test_compute_features_with_a_short_history_degrades_to_none():
    """A cold start must not raise; it must produce an untradeable bundle."""
    f = compute_features("BTC/USD", make_bars([100.0, 101.0, 102.0]))
    assert f.sma_20 is None
    assert f.rsi_14 is None
    assert f.atr_14 is None
    assert not f.is_tradeable()


def test_distance_from_20_bar_high_is_zero_at_the_high():
    f = compute_features("BTC/USD", make_bars(list(np.linspace(100, 130, 60))))
    assert f.dist_from_high_20 == pytest.approx(0.0, abs=0.01)
    assert f.dist_from_low_20 > 0


def test_bars_to_frame_sorts_by_timestamp():
    bars = make_bars([100.0, 101.0, 102.0])
    frame = bars_to_frame([bars[2], bars[0], bars[1]])
    assert list(frame["close"]) == [100.0, 101.0, 102.0]


def test_bars_to_frame_empty():
    assert bars_to_frame([]).empty


# -- the composite score ------------------------------------------------------------------


def test_trend_score_is_volatility_normalised():
    """A high-vol symbol must not outrank a low-vol one purely by moving more."""
    common = dict(close=100.0, rsi_14=60.0, ret_5=None, ret_20=None)
    calm = composite_trend_score(
        ema_12=101.0, ema_26=100.0, macd_hist=0.5, atr_pct=0.005, **common
    )
    wild = composite_trend_score(
        ema_12=110.0, ema_26=100.0, macd_hist=5.0, atr_pct=0.05, **common
    )
    assert calm == pytest.approx(wild, abs=0.3)


def test_trend_score_penalises_an_overbought_tape():
    base = dict(
        ema_12=101.0, ema_26=100.0, close=100.0, macd_hist=0.1,
        ret_5=0.01, ret_20=0.02, atr_pct=0.01,
    )
    healthy = composite_trend_score(rsi_14=62.0, **base)
    stretched = composite_trend_score(rsi_14=85.0, **base)
    assert stretched < healthy


def test_trend_score_handles_all_none():
    assert composite_trend_score(
        ema_12=None, ema_26=None, close=100.0, rsi_14=None,
        macd_hist=None, ret_5=None, ret_20=None, atr_pct=None,
    ) == 0.0


def test_trend_score_is_negative_in_a_downtrend():
    score = composite_trend_score(
        ema_12=99.0, ema_26=100.0, close=100.0, rsi_14=35.0,
        macd_hist=-0.5, ret_5=-0.02, ret_20=-0.05, atr_pct=0.01,
    )
    assert score < 0
