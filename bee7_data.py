"""
bee7_data.py
=============
OHLCV loading and WaveTrend indicator preparation for bee7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bee7_params import (
    ATR_LEN,
    HTF_EMA_INTERVAL,
    HTF_EMA_LEN,
    WT_CHANNEL_LEN,
    WT_AVG_LEN,
    WT_SIGNAL_LEN,
    WT_CHANNEL_LEN_GRID,
    WT_AVG_LEN_GRID,
    WT_SIGNAL_LEN_GRID,
    WT_EMA_FILTER_LEN,
    WT_EMA_FILTER_LEN_OPTIONS,
    WT_H4_FILTER_INTERVAL,
)


def wt1_column(channel_len: int, avg_len: int) -> str:
    return f"wt1_c{int(channel_len)}_a{int(avg_len)}"


def wt2_column(channel_len: int, avg_len: int, signal_len: int) -> str:
    return f"wt2_c{int(channel_len)}_a{int(avg_len)}_s{int(signal_len)}"


def wt_columns(channel_len: int, avg_len: int, signal_len: int) -> tuple[str, str]:
    return wt1_column(channel_len, avg_len), wt2_column(channel_len, avg_len, signal_len)


def htf_wt1_column(channel_len: int, avg_len: int, interval: str = WT_H4_FILTER_INTERVAL) -> str:
    clean_interval = str(interval).replace(" ", "").replace(":", "")
    return f"wt1_{clean_interval}_c{int(channel_len)}_a{int(avg_len)}"


def htf_wt2_column(
    channel_len: int,
    avg_len: int,
    signal_len: int,
    interval: str = WT_H4_FILTER_INTERVAL,
) -> str:
    clean_interval = str(interval).replace(" ", "").replace(":", "")
    return f"wt2_{clean_interval}_c{int(channel_len)}_a{int(avg_len)}_s{int(signal_len)}"


def _bars_since_flag(flag: pd.Series) -> pd.Series:
    values: list[float] = []
    last_idx: int | None = None
    for idx, is_true in enumerate(flag.fillna(False).astype(bool).to_list()):
        if is_true:
            last_idx = idx
        values.append(np.nan if last_idx is None else float(idx - last_idx))
    return pd.Series(values, index=flag.index, dtype="float64")


def _estimate_base_tf_minutes(df: pd.DataFrame) -> float:
    if "time" not in df.columns or len(df) < 2:
        return 0.0
    diffs = (
        pd.to_datetime(df["time"], utc=True, errors="coerce")
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )
    if diffs.empty:
        return 0.0
    return float(diffs.median())


def _higher_timeframe_ema(
    df: pd.DataFrame,
    target_interval: str = HTF_EMA_INTERVAL,
    ema_len: int = HTF_EMA_LEN,
) -> pd.Series:
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    base_minutes = _estimate_base_tf_minutes(df)
    target_minutes = pd.Timedelta(target_interval).total_seconds() / 60.0

    if base_minutes <= 0 or base_minutes >= target_minutes:
        return close.ewm(span=int(ema_len), adjust=False).mean()

    resampled = (
        pd.DataFrame({"time": times, "close": close})
        .dropna(subset=["time"])
        .set_index("time")["close"]
        .resample(target_interval)
        .last()
        .dropna()
    )
    ema_htf = resampled.ewm(span=int(ema_len), adjust=False).mean()
    aligned = ema_htf.reindex(times, method="ffill")
    return pd.Series(aligned.to_numpy(), index=df.index, dtype="float64")


def _higher_timeframe_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    base_minutes = _estimate_base_tf_minutes(df)
    target_minutes = pd.Timedelta(target_interval).total_seconds() / 60.0

    if base_minutes <= 0 or base_minutes >= target_minutes:
        out = df[["time", "open", "high", "low", "close", "volume"]].copy()
        out["time"] = times
        return out.dropna(subset=["time"]).reset_index(drop=True)

    ohlcv = (
        pd.DataFrame(
            {
                "time": times,
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
            }
        )
        .dropna(subset=["time"])
        .set_index("time")
        .resample(target_interval)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return ohlcv


def _atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()


def compute_wave_trend(
    df: pd.DataFrame,
    channel_len: int,
    avg_len: int,
    signal_len: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Classic WaveTrend oscillator:
      ap  = HLC3
      esa = EMA(ap, channel_len)
      d   = EMA(abs(ap - esa), channel_len)
      ci  = (ap - esa) / (0.015 * d)
      wt1 = EMA(ci, avg_len)
      wt2 = SMA(wt1, signal_len)
    """
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ap.ewm(span=int(channel_len), adjust=False).mean()
    d = (ap - esa).abs().ewm(span=int(channel_len), adjust=False).mean()
    d = d.replace(0.0, np.nan)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=int(avg_len), adjust=False).mean()
    wt2 = wt1.rolling(int(signal_len)).mean()
    return wt1, wt2


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add precomputed WaveTrend columns for all WFO combinations plus default aliases:
      - hlc3
      - wt1_cX_aY
      - wt2_cX_aY_sZ
      - ema_X filter columns
      - wt1, wt2, wt_delta
      - wt_green_dot, wt_red_dot
      - bars_since_wt_green_dot, bars_since_wt_red_dot
    """
    df = df.copy()
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0

    channel_lens = sorted(set([WT_CHANNEL_LEN] + list(WT_CHANNEL_LEN_GRID)))
    avg_lens = sorted(set([WT_AVG_LEN] + list(WT_AVG_LEN_GRID)))
    signal_lens = sorted(set([WT_SIGNAL_LEN] + list(WT_SIGNAL_LEN_GRID)))

    for channel_len in channel_lens:
        ap = df["hlc3"]
        esa = ap.ewm(span=int(channel_len), adjust=False).mean()
        d = (ap - esa).abs().ewm(span=int(channel_len), adjust=False).mean().replace(0.0, np.nan)
        ci = (ap - esa) / (0.015 * d)

        for avg_len in avg_lens:
            wt1_col = wt1_column(channel_len, avg_len)
            wt1 = ci.ewm(span=int(avg_len), adjust=False).mean()
            df[wt1_col] = wt1

            for signal_len in signal_lens:
                df[wt2_column(channel_len, avg_len, signal_len)] = wt1.rolling(int(signal_len)).mean()

    h4_df = _higher_timeframe_ohlcv(df, WT_H4_FILTER_INTERVAL)
    h4_times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    h4_time_index = pd.to_datetime(h4_df["time"], utc=True, errors="coerce")
    for channel_len in channel_lens:
        for avg_len in avg_lens:
            wt1_h4, _ = compute_wave_trend(h4_df, channel_len, avg_len, signal_lens[0])
            h4_wt1_col = htf_wt1_column(channel_len, avg_len, WT_H4_FILTER_INTERVAL)
            wt1_series = pd.Series(wt1_h4.to_numpy(), index=h4_time_index, dtype="float64")
            df[h4_wt1_col] = wt1_series.reindex(h4_times, method="ffill").to_numpy()

            for signal_len in signal_lens:
                wt1_h4, wt2_h4 = compute_wave_trend(h4_df, channel_len, avg_len, signal_len)
                h4_wt2_col = htf_wt2_column(channel_len, avg_len, signal_len, WT_H4_FILTER_INTERVAL)
                wt2_series = pd.Series(wt2_h4.to_numpy(), index=h4_time_index, dtype="float64")
                df[h4_wt2_col] = wt2_series.reindex(h4_times, method="ffill").to_numpy()

    default_wt1_col, default_wt2_col = wt_columns(WT_CHANNEL_LEN, WT_AVG_LEN, WT_SIGNAL_LEN)
    default_h4_wt1_col = htf_wt1_column(WT_CHANNEL_LEN, WT_AVG_LEN, WT_H4_FILTER_INTERVAL)
    default_h4_wt2_col = htf_wt2_column(WT_CHANNEL_LEN, WT_AVG_LEN, WT_SIGNAL_LEN, WT_H4_FILTER_INTERVAL)
    ema_lens = sorted(set([WT_EMA_FILTER_LEN] + list(WT_EMA_FILTER_LEN_OPTIONS)))
    for ema_len in ema_lens:
        df[f"ema_{int(ema_len)}"] = df["close"].ewm(span=int(ema_len), adjust=False).mean()
    df["ema20"] = df["ema_20"]
    df["htf_ema200"] = _higher_timeframe_ema(df, target_interval=HTF_EMA_INTERVAL, ema_len=HTF_EMA_LEN)
    df["atr_14"] = _atr(df, ATR_LEN)
    df["atr"] = df["atr_14"]
    df["wt1"] = df[default_wt1_col]
    df["wt2"] = df[default_wt2_col]
    df["wt_delta"] = df["wt1"] - df["wt2"]
    df["h4_wt1"] = df[default_h4_wt1_col]
    df["h4_wt2"] = df[default_h4_wt2_col]
    df["h4_wt_delta"] = df["h4_wt1"] - df["h4_wt2"]
    df["h4_prev_wt1"] = df["h4_wt1"].shift(1)
    df["h4_prev_wt2"] = df["h4_wt2"].shift(1)
    df["h4_prev_wt_delta"] = df["h4_wt_delta"].shift(1)
    df["wt_green_dot"] = (df["wt1"].shift(1) <= df["wt2"].shift(1)) & (df["wt1"] > df["wt2"])
    df["wt_red_dot"] = (df["wt1"].shift(1) >= df["wt2"].shift(1)) & (df["wt1"] < df["wt2"])
    df["bars_since_wt_green_dot"] = _bars_since_flag(df["wt_green_dot"])
    df["bars_since_wt_red_dot"] = _bars_since_flag(df["wt_red_dot"])

    return df


def load_klines(csv_path: str) -> pd.DataFrame:
    """
    Load candle CSV.
    Supports open_time as milliseconds, seconds or string timestamps.
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

    if "open_time" in df.columns:
        col = df["open_time"]
        try:
            is_numeric = np.issubdtype(col.dtype, np.number)
        except TypeError:
            is_numeric = False
        if is_numeric:
            unit = "ms" if col.max() > 1e12 else "s"
            df["time"] = pd.to_datetime(col, unit=unit, utc=True)
        else:
            df["time"] = pd.to_datetime(col, utc=True, errors="coerce")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        raise ValueError("Missing open_time / time column in CSV.")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in CSV.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def format_ts(ts) -> str:
    """Pretty UTC timestamp string."""
    if pd.isna(ts):
        return "NaT"
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")

