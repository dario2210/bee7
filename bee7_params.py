"""
bee7_params.py
===============
Central configuration for the first bee7 WaveTrend strategy.

Project structure stays aligned with bee1 on purpose, but the trading logic
and WFO grids are now built around WaveTrend crossover signals.
"""

from __future__ import annotations

import json
import os

# Input data
CSV_PATH = "ethusdt_1h.csv"
INITIAL_CAPITAL = 10_000.0

# Binance data source
BINANCE_SYMBOL = "ETHUSDT"
BINANCE_INTERVAL = "1h"
BINANCE_MARKET = "spot"
BINANCE_START_DATE = "2021-01-01"
BINANCE_CSV_CACHE = None

# Default WaveTrend setup
WT_CHANNEL_LEN = 10
WT_AVG_LEN = 21
WT_SIGNAL_LEN = 3
WT_MIN_SIGNAL_LEVEL = 0.0
WT_ZERO_LINE = 0.0
TRADE_DIRECTION = "both"
ALLOW_LONGS = True
ALLOW_SHORTS = True
SHORT_TRADING_ENABLED = True
WT_LONG_ENTRY_WINDOW_BARS = 3
WT_LONG_ENTRY_MAX_ABOVE_ZERO = -30.0
WT_LONG_CLOSE_MIN_LEVEL = 40.0
WT_LONG_EXIT_MIN_LEVEL = WT_LONG_CLOSE_MIN_LEVEL
WT_LONG_REQUIRE_EMA20_RECLAIM = False
WT_LONG_REQUIRE_HTF_TREND = False
WT_SHORT_ENTRY_WINDOW_BARS = WT_LONG_ENTRY_WINDOW_BARS
WT_SHORT_ENTRY_MIN_BELOW_ZERO = 30.0
WT_SHORT_CLOSE_MAX_LEVEL = -WT_LONG_CLOSE_MIN_LEVEL
WT_SHORT_EXIT_MAX_LEVEL = WT_SHORT_CLOSE_MAX_LEVEL
WT_SHORT_REQUIRE_EMA20_REJECT = False
WT_SHORT_REQUIRE_HTF_TREND = False
WT_EMA_FILTER_LEN = 20
HTF_EMA_LEN = 200
HTF_EMA_INTERVAL = "4h"
WT_H4_FILTER_INTERVAL = "4h"
WT_H4_LONG_FILTER_MAX = 20.0
WT_H4_LONG_CLOSE_MIN = 10.0
WT_H4_SHORT_FILTER_MIN = abs(WT_H4_LONG_FILTER_MAX)
WT_H4_SHORT_CLOSE_MAX = -WT_H4_LONG_CLOSE_MIN
WT_LONG_TP1_ENABLED = True
WT_LONG_TP1_PCT = 0.01
WT_LONG_TP1_FRACTION = 1.0 / 3.0
WT_LONG_TP1_BREAKEVEN_ENABLED = True
WT_LONG_TP2_ENABLED = True
WT_LONG_TP2_PCT = 0.02
WT_LONG_TP2_FRACTION = 1.0 / 3.0
WT_LONG_EMERGENCY_SL_ENABLED = False
WT_LONG_EMERGENCY_SL_CAPITAL_PCT = 0.0
WT_SHORT_TP1_ENABLED = True
WT_SHORT_TP1_PCT = 0.01
WT_SHORT_TP1_FRACTION = 1.0 / 3.0
WT_SHORT_TP1_BREAKEVEN_ENABLED = True
WT_SHORT_TP2_ENABLED = True
WT_SHORT_TP2_PCT = 0.02
WT_SHORT_TP2_FRACTION = 1.0 / 3.0
ATR_LEN = 14
ATR_STOP_ENABLED = False
ATR_STOP_MULTIPLIER = 2.0
BREAKEVEN_TRIGGER_ATR = 0.0
TRAILING_TRIGGER_ATR = 0.0
TRAILING_DISTANCE_ATR = 1.5
WT_EXHAUSTION_EXIT_ENABLED = False
WT_EXHAUSTION_MIN_LEVEL = 50.0

# Fees and execution friction
FEE_RATE = 0.00035

# WFO windows
OPT_DAYS = 120
LIVE_DAYS = 14

# WFO parameter grid
WT_CHANNEL_LEN_GRID = [10]
WT_AVG_LEN_GRID = [21]
WT_SIGNAL_LEN_GRID = [3]
WT_MIN_SIGNAL_LEVEL_GRID = [0.0]
WT_MIN_SIGNAL_LEVEL_OPTIONS = [0.0]
WT_REENTRY_WINDOW_GRID = [0, 1, 2, 3]
WT_USE_EMA_FILTER_GRID = [False]
WT_USE_HTF_TREND_FILTER_GRID = [False]
WT_EMA_FILTER_LEN_GRID = [20]
WT_EMA_FILTER_LEN_OPTIONS = [8, 10, 15, 20]
WT_LONG_ENTRY_MAX_ABOVE_ZERO_GRID = [-70.0, -60.0, -50.0, -40.0, -30.0]
WT_LONG_ENTRY_MAX_ABOVE_ZERO_OPTIONS = [-70.0, -60.0, -50.0, -40.0, -30.0]
WT_LONG_CLOSE_MIN_LEVEL_GRID = [40.0, 50.0, 60.0, 70.0]
WT_LONG_CLOSE_MIN_LEVEL_OPTIONS = [40.0, 50.0, 60.0, 70.0]
WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID = [30.0, 40.0, 50.0, 60.0]
WT_SHORT_ENTRY_MIN_BELOW_ZERO_OPTIONS = [30.0, 40.0, 50.0, 60.0]
WT_SHORT_CLOSE_MAX_LEVEL_GRID = [-70.0, -60.0, -50.0, -40.0]
WT_SHORT_CLOSE_MAX_LEVEL_OPTIONS = [-70.0, -60.0, -50.0, -40.0]
WT_H4_LONG_FILTER_MAX_GRID = [-70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0]
WT_H4_LONG_FILTER_MAX_OPTIONS = [-70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0]
WT_H4_LONG_CLOSE_MIN_GRID = [30.0, 40.0, 50.0, 60.0, 70.0]
WT_H4_LONG_CLOSE_MIN_OPTIONS = [30.0, 40.0, 50.0, 60.0, 70.0]
WT_H4_SHORT_FILTER_MIN_GRID = [30.0, 40.0, 50.0, 60.0]
WT_H4_SHORT_FILTER_MIN_OPTIONS = [30.0, 40.0, 50.0, 60.0]
WT_H4_SHORT_CLOSE_MAX_GRID = [-70.0, -60.0, -50.0, -40.0, -30.0]
WT_H4_SHORT_CLOSE_MAX_OPTIONS = [-70.0, -60.0, -50.0, -40.0, -30.0]
WT_LONG_EMERGENCY_SL_CAPITAL_PCT_GRID = [0.0, 0.01, 0.02, 0.05, 0.10]
WT_LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS = [0.0, 0.01, 0.02, 0.05, 0.10]

# Compatibility aliases kept so the bee1 dashboard structure can stay intact
TP_GRID = WT_CHANNEL_LEN_GRID
SL_GRID = WT_AVG_LEN_GRID
TRAIL_DROP_GRID = WT_SIGNAL_LEN_GRID
EMA_LEN_GRID = WT_EMA_FILTER_LEN_GRID

# Visual guide bands for the dashboard signal panel
TMA_LOW_MIN = -60.0
TMA_LOW_MAX = -53.0
TMA_HIGH_MIN = 53.0
TMA_HIGH_MAX = 60.0

# Fallback params used when WFO JSON is missing
DEFAULT_PARAMS = {
    "wt_channel_len": WT_CHANNEL_LEN,
    "wt_avg_len": WT_AVG_LEN,
    "wt_signal_len": WT_SIGNAL_LEN,
    "wt_min_signal_level": WT_MIN_SIGNAL_LEVEL,
    "wt_zero_line": WT_ZERO_LINE,
    "trade_direction": TRADE_DIRECTION,
    "allow_longs": ALLOW_LONGS,
    "allow_shorts": ALLOW_SHORTS,
    "short_trading_enabled": SHORT_TRADING_ENABLED,
    "wt_long_entry_window_bars": WT_LONG_ENTRY_WINDOW_BARS,
    "wt_long_entry_max_above_zero": WT_LONG_ENTRY_MAX_ABOVE_ZERO,
    "wt_long_close_min_level": WT_LONG_CLOSE_MIN_LEVEL,
    "wt_long_exit_min_level": WT_LONG_EXIT_MIN_LEVEL,
    "wt_long_require_ema20_reclaim": WT_LONG_REQUIRE_EMA20_RECLAIM,
    "wt_long_require_htf_trend": WT_LONG_REQUIRE_HTF_TREND,
    "wt_short_entry_window_bars": WT_SHORT_ENTRY_WINDOW_BARS,
    "wt_short_entry_min_below_zero": WT_SHORT_ENTRY_MIN_BELOW_ZERO,
    "wt_short_close_max_level": WT_SHORT_CLOSE_MAX_LEVEL,
    "wt_short_exit_max_level": WT_SHORT_EXIT_MAX_LEVEL,
    "wt_short_require_ema20_reject": WT_SHORT_REQUIRE_EMA20_REJECT,
    "wt_short_require_htf_trend": WT_SHORT_REQUIRE_HTF_TREND,
    "wt_ema_filter_len": WT_EMA_FILTER_LEN,
    "wt_h4_filter_interval": WT_H4_FILTER_INTERVAL,
    "wt_h4_long_filter_max": WT_H4_LONG_FILTER_MAX,
    "wt_h4_long_close_min": WT_H4_LONG_CLOSE_MIN,
    "wt_h4_short_filter_min": WT_H4_SHORT_FILTER_MIN,
    "wt_h4_short_close_max": WT_H4_SHORT_CLOSE_MAX,
    "wt_long_tp1_enabled": WT_LONG_TP1_ENABLED,
    "wt_long_tp1_pct": WT_LONG_TP1_PCT,
    "wt_long_tp1_fraction": WT_LONG_TP1_FRACTION,
    "wt_long_tp1_breakeven_enabled": WT_LONG_TP1_BREAKEVEN_ENABLED,
    "wt_long_tp2_enabled": WT_LONG_TP2_ENABLED,
    "wt_long_tp2_pct": WT_LONG_TP2_PCT,
    "wt_long_tp2_fraction": WT_LONG_TP2_FRACTION,
    "wt_long_emergency_sl_enabled": WT_LONG_EMERGENCY_SL_ENABLED,
    "wt_long_emergency_sl_capital_pct": WT_LONG_EMERGENCY_SL_CAPITAL_PCT,
    "wt_short_tp1_enabled": WT_SHORT_TP1_ENABLED,
    "wt_short_tp1_pct": WT_SHORT_TP1_PCT,
    "wt_short_tp1_fraction": WT_SHORT_TP1_FRACTION,
    "wt_short_tp1_breakeven_enabled": WT_SHORT_TP1_BREAKEVEN_ENABLED,
    "wt_short_tp2_enabled": WT_SHORT_TP2_ENABLED,
    "wt_short_tp2_pct": WT_SHORT_TP2_PCT,
    "wt_short_tp2_fraction": WT_SHORT_TP2_FRACTION,
    "htf_ema_len": HTF_EMA_LEN,
    "htf_ema_interval": HTF_EMA_INTERVAL,
    "atr_len": ATR_LEN,
    "atr_stop_enabled": ATR_STOP_ENABLED,
    "atr_stop_multiplier": ATR_STOP_MULTIPLIER,
    "breakeven_trigger_atr": BREAKEVEN_TRIGGER_ATR,
    "trailing_trigger_atr": TRAILING_TRIGGER_ATR,
    "trailing_distance_atr": TRAILING_DISTANCE_ATR,
    "wt_exhaustion_exit_enabled": WT_EXHAUSTION_EXIT_ENABLED,
    "wt_exhaustion_min_level": WT_EXHAUSTION_MIN_LEVEL,
    "fee_rate": FEE_RATE,
    "slippage_bps": 2.0,
    "spread_bps": 1.0,
    "max_bars_in_trade": 0,
}

WFO_BEST_PARAMS_PATH = "bee7_wfo_best_params.json"


def load_params() -> dict:
    """
    Return strategy params:
    1) start from DEFAULT_PARAMS,
    2) override with WFO_BEST_PARAMS_PATH when present.
    """
    params = dict(DEFAULT_PARAMS)

    json_path = os.path.join(os.path.dirname(__file__), WFO_BEST_PARAMS_PATH)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in DEFAULT_PARAMS:
                    if key in data:
                        params[key] = data[key]
            print(f"[params] Loaded WFO params from {WFO_BEST_PARAMS_PATH}")
        except Exception as exc:
            print(f"[params] Could not load {WFO_BEST_PARAMS_PATH}: {exc!r}")
    else:
        print(f"[params] Missing {WFO_BEST_PARAMS_PATH} - using DEFAULT_PARAMS")

    # BEE7 runs mirrored long/short signals with the long grid as the source
    # of paired inverse short thresholds.
    params["trade_direction"] = "both"
    params["allow_longs"] = True
    params["allow_shorts"] = True
    params["short_trading_enabled"] = True
    sl_pct = float(params.get("wt_long_emergency_sl_capital_pct", 0.0) or 0.0)
    params["wt_long_emergency_sl_capital_pct"] = sl_pct
    params["wt_long_emergency_sl_enabled"] = bool(
        params.get("wt_long_emergency_sl_enabled", False)
    ) and sl_pct > 0.0
    params["atr_stop_enabled"] = False

    return params


def save_params(params: dict, path: str = WFO_BEST_PARAMS_PATH) -> None:
    """Persist strategy params to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[params] Saved params -> {path}")

