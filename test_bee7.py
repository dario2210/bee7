"""
test_bee7.py
============
Pytest suite for the BEE7_2 WaveTrend H1 + H4 branch.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bee7_binance import interval_to_ms, wfo_bars
from bee7_data import (
    htf_wt1_column,
    htf_wt2_column,
    load_klines,
    prepare_indicators,
    wt_columns,
)
from bee7_engine import (
    BarData,
    PositionState,
    apply_slippage,
    bar_from_row,
    generate_entry_signal,
    generate_exit_signal,
)
from bee7_strategy import Bee7Strategy
from bee7_wfo import get_latest_best_params, walk_forward_optimization


BASE_PARAMS = {
    "wt_channel_len": 10,
    "wt_avg_len": 21,
    "wt_signal_len": 4,
    "wt_min_signal_level": 0.0,
    "wt_zero_line": 0.0,
    "trade_direction": "both",
    "allow_longs": True,
    "allow_shorts": True,
    "wt_long_entry_window_bars": 0,
    "wt_long_entry_max_above_zero": -30.0,
    "wt_long_exit_min_level": 0.0,
    "wt_long_require_ema20_reclaim": False,
    "wt_long_require_htf_trend": False,
    "wt_short_entry_window_bars": 0,
    "wt_short_entry_min_below_zero": 30.0,
    "wt_short_exit_max_level": 0.0,
    "wt_short_require_ema20_reject": False,
    "wt_short_require_htf_trend": False,
    "wt_ema_filter_len": 20,
    "wt_h4_filter_interval": "4h",
    "wt_h4_long_filter_max": -20.0,
    "wt_h4_short_filter_min": 50.0,
    "wt_h4_invalidation_exit_enabled": True,
    "atr_stop_enabled": False,
    "atr_stop_multiplier": 2.0,
    "breakeven_trigger_atr": 0.0,
    "trailing_trigger_atr": 0.0,
    "trailing_distance_atr": 1.5,
    "wt_exhaustion_exit_enabled": False,
    "wt_exhaustion_min_level": 50.0,
    "fee_rate": 0.00035,
    "slippage_bps": 0.0,
    "spread_bps": 0.0,
    "max_bars_in_trade": 0,
}

LONG_ONLY_PARAMS = {
    **BASE_PARAMS,
    "trade_direction": "long",
    "allow_longs": True,
    "allow_shorts": False,
}

SHORT_ONLY_PARAMS = {
    **BASE_PARAMS,
    "trade_direction": "short",
    "allow_longs": False,
    "allow_shorts": True,
}


def _make_bar(
    close=1800.0,
    wt1=-40.0,
    wt2=-45.0,
    h4_wt1=-28.0,
    h4_wt2=-22.0,
    h4_prev_wt1=-34.0,
    h4_prev_wt2=-22.0,
):
    return BarData(
        time=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        open=close,
        high=close + 5.0,
        low=close - 5.0,
        close=close,
        wt1=wt1,
        wt2=wt2,
        wt_delta=wt1 - wt2,
        h4_wt1=h4_wt1,
        h4_wt2=h4_wt2,
        h4_wt_delta=h4_wt1 - h4_wt2,
        h4_prev_wt1=h4_prev_wt1,
        h4_prev_wt2=h4_prev_wt2,
        h4_prev_wt_delta=h4_prev_wt1 - h4_prev_wt2,
        ema20=np.nan,
        ema_filter_len=20,
        htf_ema200=np.nan,
        atr=25.0,
        wt_green_dot=False,
        wt_red_dot=False,
        bars_since_wt_green_dot=np.nan,
        bars_since_wt_red_dot=np.nan,
    )


def _signal_df() -> pd.DataFrame:
    wt1_col, wt2_col = wt_columns(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_signal_len"],
    )
    h4_wt1_col = htf_wt1_column(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_h4_filter_interval"],
    )
    h4_wt2_col = htf_wt2_column(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_signal_len"],
        BASE_PARAMS["wt_h4_filter_interval"],
    )
    times = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    closes = [1800.0, 1810.0, 1830.0, 1820.0]
    wt1_vals = [-48.0, -34.0, 44.0, 34.0]
    wt2_vals = [-42.0, -44.0, 36.0, 44.0]
    h4_wt1_vals = [-34.0, -28.0, 62.0, 58.0]
    h4_wt2_vals = [-22.0, -22.0, 50.0, 52.0]

    df = pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": [c + 6.0 for c in closes],
            "low": [c - 6.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            wt1_col: wt1_vals,
            wt2_col: wt2_vals,
            h4_wt1_col: h4_wt1_vals,
            h4_wt2_col: h4_wt2_vals,
            "wt1": wt1_vals,
            "wt2": wt2_vals,
            "wt_delta": np.array(wt1_vals) - np.array(wt2_vals),
            "h4_wt1": h4_wt1_vals,
            "h4_wt2": h4_wt2_vals,
        }
    )
    df["h4_wt_delta"] = df["h4_wt1"] - df["h4_wt2"]
    df["h4_prev_wt1"] = df["h4_wt1"].shift(1)
    df["h4_prev_wt2"] = df["h4_wt2"].shift(1)
    df["h4_prev_wt_delta"] = df["h4_wt_delta"].shift(1)
    df["ema20"] = pd.Series(closes).ewm(span=20, adjust=False).mean()
    df["ema_20"] = df["ema20"]
    df["htf_ema200"] = pd.Series(closes).ewm(span=200, adjust=False).mean()
    df["atr"] = [20.0] * len(df)
    df["wt_green_dot"] = (df["wt1"].shift(1) <= df["wt2"].shift(1)) & (df["wt1"] > df["wt2"])
    df["wt_red_dot"] = (df["wt1"].shift(1) >= df["wt2"].shift(1)) & (df["wt1"] < df["wt2"])
    df["bars_since_wt_green_dot"] = [np.nan, 0.0, 1.0, 2.0]
    df["bars_since_wt_red_dot"] = [np.nan, np.nan, np.nan, 0.0]
    return df


def _write_csv(path: Path, column: str, values: list[str]) -> Path:
    df = pd.DataFrame(
        {
            column: values,
            "open": [100.0] * len(values),
            "high": [101.0] * len(values),
            "low": [99.0] * len(values),
            "close": [100.0] * len(values),
            "volume": [1.0] * len(values),
        }
    )
    df.to_csv(path, index=False)
    return path


class TestWaveTrendPreparation:
    def test_prepare_indicators_adds_h4_columns(self):
        times = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
        close = 1800.0 + np.sin(np.linspace(0, 30, len(times))) * 60.0
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 4.0,
                "low": close - 4.0,
                "close": close,
                "volume": 1000.0,
            }
        )

        out = prepare_indicators(df)
        wt1_col, wt2_col = wt_columns(10, 21, 4)
        h4_wt1_col = htf_wt1_column(10, 21, "4h")
        h4_wt2_col = htf_wt2_column(10, 21, 4, "4h")

        for col in [
            "wt1",
            "wt2",
            "wt_delta",
            "h4_wt1",
            "h4_wt2",
            "h4_wt_delta",
            "h4_prev_wt_delta",
            wt1_col,
            wt2_col,
            h4_wt1_col,
            h4_wt2_col,
        ]:
            assert col in out.columns

        assert out["wt1"].dropna().shape[0] > 0
        assert out["h4_wt1"].dropna().shape[0] > 0


class TestEntrySignals:
    def test_open_long_on_h1_green_dot_with_h4_filter(self):
        prev = _make_bar(wt1=-48.0, wt2=-42.0)
        bar = _make_bar(wt1=-34.0, wt2=-44.0)

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "open_long"
        assert sig.reason == "WT_H1_GREEN_DOT_H4_FILTER"

    def test_no_long_without_h4_convergence(self):
        prev = _make_bar(wt1=-48.0, wt2=-42.0)
        bar = _make_bar(
            wt1=-34.0,
            wt2=-44.0,
            h4_wt1=-34.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-28.0,
            h4_prev_wt2=-22.0,
        )

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "none"

    def test_no_long_when_h1_zone_is_too_shallow(self):
        prev = _make_bar(wt1=-20.0, wt2=-18.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0)

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "none"

    def test_open_short_on_h1_red_dot_with_h4_filter(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "open_short"
        assert sig.reason == "WT_H1_RED_DOT_H4_FILTER"

    def test_direction_filter_blocks_short_in_long_only_mode(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )

        sig = generate_entry_signal(bar, prev, LONG_ONLY_PARAMS, None)

        assert sig.action == "none"


class TestExitSignals:
    def test_reverse_long_to_short_on_opposite_signal(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_reverse"
        assert sig.reason == "REVERSE_TO_SHORT"

    def test_force_exit_long_in_long_only_mode(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, LONG_ONLY_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_H1_RED_DOT_H4_FILTER_EXIT_LONG"

    def test_emergency_exit_long_when_h4_turns_against_position(self):
        prev = _make_bar(
            wt1=-35.0,
            wt2=-42.0,
            h4_wt1=-28.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-24.0,
            h4_prev_wt2=-22.0,
        )
        bar = _make_bar(
            wt1=-32.0,
            wt2=-40.0,
            h4_wt1=-34.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-28.0,
            h4_prev_wt2=-22.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "H4_LONG_INVALIDATION_EXIT"

    def test_emergency_exit_short_when_h4_turns_against_position(self):
        prev = _make_bar(
            wt1=64.0,
            wt2=54.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=54.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=60.0,
            wt2=55.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=58.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="short", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "H4_SHORT_INVALIDATION_EXIT"


class TestBarHelpers:
    def test_bar_from_row_reads_h1_h4_and_ema_columns(self):
        wt1_col, wt2_col = wt_columns(10, 21, 4)
        h4_wt1_col = htf_wt1_column(10, 21, "4h")
        h4_wt2_col = htf_wt2_column(10, 21, 4, "4h")
        row = pd.Series(
            {
                "time": pd.Timestamp("2024-01-01", tz="UTC"),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                wt1_col: -34.0,
                wt2_col: -44.0,
                h4_wt1_col: -28.0,
                h4_wt2_col: -22.0,
                "h4_prev_wt1": -34.0,
                "h4_prev_wt2": -22.0,
                "h4_prev_wt_delta": -12.0,
                "ema20": 90.0,
                "ema_10": 95.0,
                "htf_ema200": 85.0,
                "atr": 4.0,
            }
        )

        bar = bar_from_row(row, dict(BASE_PARAMS, wt_ema_filter_len=10))

        assert bar.wt1 == pytest.approx(-34.0)
        assert bar.wt2 == pytest.approx(-44.0)
        assert bar.h4_wt1 == pytest.approx(-28.0)
        assert bar.h4_wt2 == pytest.approx(-22.0)
        assert bar.ema20 == pytest.approx(95.0)
        assert bar.ema_filter_len == 10
        assert bar.htf_ema200 == pytest.approx(85.0)
        assert bar.atr == pytest.approx(4.0)

    def test_load_klines_string(self):
        path = Path(".test_load_klines_string.csv")
        try:
            times_str = [f"2024-01-0{i + 1} 00:00:00+00:00" for i in range(5)]
            csv_path = _write_csv(path, "open_time", times_str)
            df = load_klines(str(csv_path))
            assert len(df) == 5
        finally:
            path.unlink(missing_ok=True)


class TestSlippage:
    def test_zero_slippage_returns_price(self):
        assert apply_slippage(1800.0, "long", "open", 0.0, 0.0) == 1800.0

    def test_long_open_increases_price(self):
        assert apply_slippage(1800.0, "long", "open", 5.0, 2.0) > 1800.0

    def test_short_open_decreases_price(self):
        assert apply_slippage(1800.0, "short", "open", 5.0, 2.0) < 1800.0


class TestWFOHelpers:
    def test_wfo_bars_1h(self):
        ob, lb = wfo_bars("1h", 90, 14)
        assert ob == 2160
        assert lb == 336

    def test_interval_to_ms(self):
        assert interval_to_ms("1h") == 3_600_000

    def test_save_and_load_params_json(self):
        from bee7_params import save_params

        path = Path(".test_params.json")
        try:
            save_params(dict(BASE_PARAMS), str(path))
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["wt_channel_len"] == BASE_PARAMS["wt_channel_len"]
            assert loaded["wt_h4_long_filter_max"] == BASE_PARAMS["wt_h4_long_filter_max"]
        finally:
            path.unlink(missing_ok=True)

    def test_get_latest_best_params_includes_h4_filters_and_direction(self):
        windows_df = pd.DataFrame(
            {
                "best_wt_channel_len": [10, 10, 12],
                "best_wt_avg_len": [21, 21, 28],
                "best_wt_signal_len": [4, 4, 3],
                "best_wt_min_signal_level": [0.0, 0.0, 0.0],
                "best_wt_reentry_window_bars": [0, 0, 0],
                "best_wt_use_ema_filter": [False, False, False],
                "best_wt_use_htf_filter": [False, False, False],
                "best_wt_ema_filter_len": [20, 20, 20],
                "best_wt_long_entry_max_above_zero": [-30.0, -30.0, -40.0],
                "best_wt_short_entry_min_below_zero": [30.0, 30.0, 40.0],
                "best_wt_h4_long_filter_max": [-20.0, -20.0, -30.0],
                "best_wt_h4_short_filter_min": [50.0, 50.0, 60.0],
                "allow_longs": [True, True, True],
                "allow_shorts": [False, False, False],
                "n_trades_live": [2, 1, 1],
            }
        )

        best = get_latest_best_params(windows_df)

        assert best["trade_direction"] == "long"
        assert best["allow_longs"] is True
        assert best["allow_shorts"] is False
        assert best["wt_channel_len"] == 10
        assert best["wt_avg_len"] == 21
        assert best["wt_signal_len"] == 4
        assert best["wt_long_entry_max_above_zero"] == pytest.approx(-30.0)
        assert best["wt_short_entry_min_below_zero"] == pytest.approx(30.0)
        assert best["wt_h4_long_filter_max"] == pytest.approx(-20.0)
        assert best["wt_h4_short_filter_min"] == pytest.approx(50.0)

    def test_wfo_accepts_bee7_2_grid_overrides(self):
        times = pd.date_range("2024-01-01", periods=160, freq="1h", tz="UTC")
        wave = np.sin(np.linspace(0, 18, len(times)))
        close = 1800.0 + wave * 90.0 + np.linspace(0.0, 60.0, len(times))
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 6.0,
                "low": close - 6.0,
                "close": close,
                "volume": 1000.0,
            }
        )
        df = prepare_indicators(df)

        grid_overrides = {
            "wt_channel_len": [10],
            "wt_avg_len": [21],
            "wt_signal_len": [4],
            "wt_min_signal_level": [0.0],
            "wt_reentry_window_bars": [0],
            "wt_use_ema_filter": [False],
            "wt_use_htf_filter": [False],
            "wt_ema_filter_len": [20],
            "wt_long_entry_max_above_zero": [-30.0],
            "wt_short_entry_min_below_zero": [30.0],
            "wt_h4_long_filter_max": [-20.0],
            "wt_h4_short_filter_min": [50.0],
        }

        _trades, _equity, windows_df, _final_cap, stopped = walk_forward_optimization(
            df,
            interval="1h",
            verbose=False,
            fee_rate=0.0,
            opt_days=2,
            live_days=1,
            initial_capital=10_000.0,
            base_params=dict(BASE_PARAMS, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0),
            grid_overrides=grid_overrides,
        )

        assert stopped is False
        assert not windows_df.empty
        assert set(windows_df["best_wt_channel_len"]) == {10}
        assert set(windows_df["best_wt_avg_len"]) == {21}
        assert set(windows_df["best_wt_signal_len"]) == {4}
        assert set(windows_df["best_wt_long_entry_max_above_zero"]) == {-30.0}
        assert set(windows_df["best_wt_short_entry_min_below_zero"]) == {30.0}
        assert set(windows_df["best_wt_h4_long_filter_max"]) == {-20.0}
        assert set(windows_df["best_wt_h4_short_filter_min"]) == {50.0}

    def test_wfo_can_stop_during_first_window(self):
        times = pd.date_range("2024-01-01", periods=160, freq="1h", tz="UTC")
        wave = np.sin(np.linspace(0, 18, len(times)))
        close = 1800.0 + wave * 90.0 + np.linspace(0.0, 60.0, len(times))
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 6.0,
                "low": close - 6.0,
                "close": close,
                "volume": 1000.0,
            }
        )
        df = prepare_indicators(df)

        checks = {"count": 0}

        def should_stop():
            checks["count"] += 1
            return checks["count"] >= 5

        _trades, _equity, windows_df, _final_cap, stopped = walk_forward_optimization(
            df,
            interval="1h",
            verbose=False,
            fee_rate=0.0,
            opt_days=2,
            live_days=1,
            initial_capital=10_000.0,
            base_params=dict(BASE_PARAMS, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0),
            should_stop=should_stop,
        )

        assert stopped is True
        assert windows_df.empty


class TestClosedCandleOnly:
    def test_runner_uses_last_closed_candle(self):
        import bee7_live_runner as runner_module

        source = inspect.getsource(runner_module.main_loop)
        assert "df.iloc[-2]" in source
        assert "df.iloc[-3]" in source
        assert "len(df) < 3" in source


class TestBacktestRunnerParity:
    def _simulate_runner(self, df: pd.DataFrame, params: dict) -> list[dict]:
        import copy
        import bee7_live_runner as runner_module
        from bee7_live_runner import process_bar

        state = {
            "position": None,
            "entry_price": None,
            "entry_time": None,
            "bars_in_position": 0,
            "capital": 10_000.0,
            "capital_at_open": None,
            "stop_price": None,
            "entry_atr": None,
            "last_bar_time": None,
            "daily_loss_usd": 0.0,
            "daily_date": None,
            "paused": False,
            "mode": "paper",
        }

        collected = []
        original_log_trade = runner_module.log_trade
        original_save_state = runner_module.save_state
        runner_module.log_trade = lambda t: collected.append(copy.deepcopy(t))
        runner_module.save_state = lambda s: None

        try:
            for i in range(1, len(df)):
                bar = bar_from_row(df.iloc[i], params)
                prev = bar_from_row(df.iloc[i - 1], params)
                process_bar(bar, prev, params, state, mode="paper")
        finally:
            runner_module.log_trade = original_log_trade
            runner_module.save_state = original_save_state

        return collected

    def test_same_closed_trades_as_runner(self):
        df = _signal_df()
        strat = Bee7Strategy(LONG_ONLY_PARAMS)
        bt_trades, _equity, _final_cap = strat.run(df, 10_000.0)
        bt_real = bt_trades[bt_trades["reason"] != "FORCE_EXIT_END"].reset_index(drop=True)

        runner_trades = self._simulate_runner(df, LONG_ONLY_PARAMS)

        assert len(bt_real) == len(runner_trades)
        assert list(bt_real["side"]) == [t["side"] for t in runner_trades]
        assert list(bt_real["reason"]) == [t["reason"] for t in runner_trades]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
