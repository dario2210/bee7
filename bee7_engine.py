"""
bee7_engine.py
===============
Shared decision layer for the first bee7 WaveTrend strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from bee7_data import htf_wt1_column, htf_wt2_column, wt_columns
from bee7_params import WT_ZERO_LINE

Side = Literal["long", "short"]
Action = Literal["none", "open_long", "open_short", "close_reverse", "close_force"]


@dataclass
class BarData:
    """Single-candle input for the strategy engine."""

    time: object
    open: float
    high: float
    low: float
    close: float
    wt1: float
    wt2: float
    wt_delta: float
    h4_wt1: float = np.nan
    h4_wt2: float = np.nan
    h4_wt_delta: float = np.nan
    h4_prev_wt1: float = np.nan
    h4_prev_wt2: float = np.nan
    h4_prev_wt_delta: float = np.nan
    ema20: float = np.nan
    ema_filter_len: int = 20
    htf_ema200: float = np.nan
    atr: float = np.nan
    wt_green_dot: bool = False
    wt_red_dot: bool = False
    bars_since_wt_green_dot: float = np.nan
    bars_since_wt_red_dot: float = np.nan


@dataclass
class Signal:
    action: Action
    reason: str = ""
    exit_price: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass
class PositionState:
    side: Side
    entry_price: float
    entry_time: object
    bars_in_position: int = 0
    entry_meta: dict = field(default_factory=dict)
    stop_price: float = np.nan
    entry_atr: float = np.nan


def compute_trade_close(
    entry_price: float,
    exit_price: float,
    side: Side,
    fee_rate: float,
    capital_at_open: float,
) -> dict:
    """Shared trade close accounting used by backtest and live runner."""
    if side == "long":
        gross_ret = (exit_price / entry_price) - 1.0
    else:
        gross_ret = (entry_price / exit_price) - 1.0

    net_ret = (1.0 + gross_ret) * (1.0 - fee_rate) ** 2 - 1.0
    fee_ret = net_ret - gross_ret
    fee_usd = abs(fee_ret) * capital_at_open
    pnl = capital_at_open * net_ret

    return {
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "fee_ret": fee_ret,
        "fee_usd": fee_usd,
        "pnl": pnl,
    }


def build_position_state(
    side: Side,
    entry_price: float,
    entry_time: object,
    bar: BarData,
    params: dict,
    entry_meta: Optional[dict] = None,
) -> PositionState:
    stop_price = np.nan
    entry_atr = np.nan
    if bool(params.get("atr_stop_enabled", False)) and not np.isnan(bar.atr):
        atr_mult = float(params.get("atr_stop_multiplier", 2.0))
        offset = bar.atr * atr_mult
        stop_price = entry_price - offset if side == "long" else entry_price + offset
        entry_atr = float(bar.atr)

    return PositionState(
        side=side,
        entry_price=entry_price,
        entry_time=entry_time,
        entry_meta=entry_meta or {},
        stop_price=stop_price,
        entry_atr=entry_atr,
    )


def _zone(wt1: float, wt2: float, zero_line: float) -> str:
    if wt1 < zero_line and wt2 < zero_line:
        return "below_zero"
    if wt1 > zero_line and wt2 > zero_line:
        return "above_zero"
    return "mixed"


def _cross_up(bar: BarData, prev_bar: BarData) -> bool:
    return prev_bar.wt_delta <= 0.0 and bar.wt_delta > 0.0


def _cross_down(bar: BarData, prev_bar: BarData) -> bool:
    return prev_bar.wt_delta >= 0.0 and bar.wt_delta < 0.0


def _signal_level(bar: BarData) -> float:
    return min(abs(bar.wt1), abs(bar.wt2))


def _has_recent_signal(value: float, max_bars: int) -> bool:
    return not np.isnan(value) and value <= float(max_bars)


def _flag_value(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(value)


def _float_or_nan(value) -> float:
    if value is None:
        return np.nan
    try:
        return np.nan if np.isnan(value) else float(value)
    except TypeError:
        return float(value)


def _ema_filter_ok(bar: BarData, side: Side, required: bool) -> bool:
    if not required:
        return True
    if np.isnan(bar.ema20):
        return False
    return bar.close > bar.ema20 if side == "long" else bar.close < bar.ema20


def _htf_trend_ok(bar: BarData, side: Side, required: bool) -> bool:
    if not required:
        return True
    if np.isnan(bar.htf_ema200):
        return False
    return bar.close > bar.htf_ema200 if side == "long" else bar.close < bar.htf_ema200


def _direction_flags(params: dict) -> tuple[bool, bool]:
    allow_longs = bool(params.get("allow_longs", True))
    allow_shorts = bool(params.get("allow_shorts", True))
    return allow_longs, allow_shorts


def _h4_converging(bar: BarData, side: Side) -> bool:
    if any(
        np.isnan(v)
        for v in [bar.h4_wt1, bar.h4_wt2, bar.h4_wt_delta, bar.h4_prev_wt_delta]
    ):
        return False

    current_gap = abs(bar.h4_wt_delta)
    prev_gap = abs(bar.h4_prev_wt_delta)
    if current_gap >= prev_gap:
        return False

    if side == "long":
        return bar.h4_wt_delta > bar.h4_prev_wt_delta
    return bar.h4_wt_delta < bar.h4_prev_wt_delta


def _h4_filter_ok(bar: BarData, side: Side, params: dict) -> bool:
    long_filter_max = float(params.get("wt_h4_long_filter_max", -20.0))
    short_filter_min = float(params.get("wt_h4_short_filter_min", 20.0))

    if side == "long":
        return (
            not np.isnan(bar.h4_wt1)
            and not np.isnan(bar.h4_wt2)
            and bar.h4_wt1 <= long_filter_max
            and bar.h4_wt2 <= long_filter_max
            and _h4_converging(bar, "long")
        )

    return (
        not np.isnan(bar.h4_wt1)
        and not np.isnan(bar.h4_wt2)
        and bar.h4_wt1 >= short_filter_min
        and bar.h4_wt2 >= short_filter_min
        and _h4_converging(bar, "short")
    )


def _h4_invalidated(bar: BarData, side: Side, params: dict) -> bool:
    if not bool(params.get("wt_h4_invalidation_exit_enabled", True)):
        return False
    if any(
        np.isnan(v)
        for v in [bar.h4_wt1, bar.h4_wt2, bar.h4_wt_delta, bar.h4_prev_wt1, bar.h4_prev_wt_delta]
    ):
        return False

    wt1_slope = bar.h4_wt1 - bar.h4_prev_wt1
    gap_widening = abs(bar.h4_wt_delta) > abs(bar.h4_prev_wt_delta)

    if side == "long":
        bearish_spread_widens = bar.h4_wt_delta < 0.0 and gap_widening
        return wt1_slope < 0.0 or bearish_spread_widens

    bullish_spread_widens = bar.h4_wt_delta > 0.0 and gap_widening
    return wt1_slope > 0.0 or bullish_spread_widens


def generate_entry_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: Optional[PositionState],
) -> Signal:
    """
    Entry logic for BEE7_2:
      - long immediately on fresh H1 bullish cross in a deep negative H1 zone
      - short immediately on fresh H1 bearish cross in a deep positive H1 zone
      - both entries are filtered by H4 WaveTrend zone + convergence
    """
    if position is not None:
        return Signal(action="none")

    if any(np.isnan(v) for v in [bar.wt1, bar.wt2, prev_bar.wt1, prev_bar.wt2]):
        return Signal(action="none")

    zero_line = float(params.get("wt_zero_line", WT_ZERO_LINE))
    min_level = float(params.get("wt_min_signal_level", 0.0))
    level_now = _signal_level(bar)

    allow_longs, allow_shorts = _direction_flags(params)
    long_entry_max_above_zero = float(params.get("wt_long_entry_max_above_zero", 0.0))
    short_entry_min_below_zero = float(params.get("wt_short_entry_min_below_zero", 0.0))
    long_h4_ok = _h4_filter_ok(bar, "long", params)
    short_h4_ok = _h4_filter_ok(bar, "short", params)

    fresh_long_cross = (
        allow_longs
        and _cross_up(bar, prev_bar)
        and bar.wt1 <= long_entry_max_above_zero
        and bar.wt2 <= long_entry_max_above_zero
        and level_now >= min_level
        and long_h4_ok
    )
    fresh_short_cross = (
        allow_shorts
        and _cross_down(bar, prev_bar)
        and bar.wt1 >= short_entry_min_below_zero
        and bar.wt2 >= short_entry_min_below_zero
        and level_now >= min_level
        and short_h4_ok
    )
    long_cond = fresh_long_cross
    short_cond = fresh_short_cross

    meta = {
        "entry_wt1": round(bar.wt1, 4),
        "entry_wt2": round(bar.wt2, 4),
        "entry_delta": round(bar.wt_delta, 4),
        "entry_zone": _zone(bar.wt1, bar.wt2, zero_line),
        "entry_signal_level": round(level_now, 4),
        "prev_wt1": round(prev_bar.wt1, 4),
        "prev_wt2": round(prev_bar.wt2, 4),
        "entry_ema_filter": round(bar.ema20, 4) if not np.isnan(bar.ema20) else np.nan,
        "entry_ema_filter_len": int(bar.ema_filter_len),
        "entry_htf_ema200": round(bar.htf_ema200, 4) if not np.isnan(bar.htf_ema200) else np.nan,
        "entry_atr": round(bar.atr, 4) if not np.isnan(bar.atr) else np.nan,
        "entry_h4_wt1": round(bar.h4_wt1, 4) if not np.isnan(bar.h4_wt1) else np.nan,
        "entry_h4_wt2": round(bar.h4_wt2, 4) if not np.isnan(bar.h4_wt2) else np.nan,
        "entry_h4_delta": round(bar.h4_wt_delta, 4) if not np.isnan(bar.h4_wt_delta) else np.nan,
    }

    if long_cond:
        return Signal(
            action="open_long",
            reason="WT_H1_GREEN_DOT_H4_FILTER",
            meta={
                **meta,
                "cross_type": "bullish",
                "bars_since_green_dot": round(bar.bars_since_wt_green_dot, 4)
                if not np.isnan(bar.bars_since_wt_green_dot)
                else np.nan,
            },
        )
    if short_cond:
        return Signal(
            action="open_short",
            reason="WT_H1_RED_DOT_H4_FILTER",
            meta={
                **meta,
                "cross_type": "bearish",
                "bars_since_red_dot": round(bar.bars_since_wt_red_dot, 4)
                if not np.isnan(bar.bars_since_wt_red_dot)
                else np.nan,
            },
        )
    return Signal(action="none")


def generate_exit_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    """
    Exit logic for BEE7_2:
      - no stop-loss / break-even / trailing by default
      - emergency close if the H4 WaveTrend setup is invalidated
      - close or reverse only when the opposite H1 cross + H4 filter appears
    """
    position.bars_in_position += 1

    max_bars = int(params.get("max_bars_in_trade", 0) or 0)
    zero_line = float(params.get("wt_zero_line", WT_ZERO_LINE))

    def _meta(trigger: str) -> dict:
        return {
            "exit_wt1": round(bar.wt1, 4),
            "exit_wt2": round(bar.wt2, 4),
            "exit_delta": round(bar.wt_delta, 4),
            "exit_zone": _zone(bar.wt1, bar.wt2, zero_line),
            "exit_signal_level": round(_signal_level(bar), 4),
            "bars_in_position": position.bars_in_position,
            "exit_trigger": trigger,
            "stop_price": round(position.stop_price, 4) if not np.isnan(position.stop_price) else np.nan,
            "exit_h4_wt1": round(bar.h4_wt1, 4) if not np.isnan(bar.h4_wt1) else np.nan,
            "exit_h4_wt2": round(bar.h4_wt2, 4) if not np.isnan(bar.h4_wt2) else np.nan,
            "exit_h4_delta": round(bar.h4_wt_delta, 4) if not np.isnan(bar.h4_wt_delta) else np.nan,
        }

    if max_bars > 0 and position.bars_in_position >= max_bars:
        return Signal(action="close_force", reason="TIME_STOP", meta=_meta("TIME_STOP"))

    allow_longs, allow_shorts = _direction_flags(params)
    opposite_params = dict(params)
    opposite_params["allow_longs"] = True
    opposite_params["allow_shorts"] = True
    opposite_signal = generate_entry_signal(bar, prev_bar, opposite_params, None)
    if position.side == "long":
        if opposite_signal.action == "open_short":
            if allow_shorts:
                return Signal(
                    action="close_reverse",
                    reason="REVERSE_TO_SHORT",
                    meta=_meta("REVERSE_TO_SHORT"),
                )
            return Signal(
                action="close_force",
                reason="WT_H1_RED_DOT_H4_FILTER_EXIT_LONG",
                meta=_meta("WT_H1_RED_DOT_H4_FILTER_EXIT_LONG"),
            )

    if position.side == "short":
        if opposite_signal.action == "open_long":
            if allow_longs:
                return Signal(
                    action="close_reverse",
                    reason="REVERSE_TO_LONG",
                    meta=_meta("REVERSE_TO_LONG"),
                )
            return Signal(
                action="close_force",
                reason="WT_H1_GREEN_DOT_H4_FILTER_EXIT_SHORT",
                meta=_meta("WT_H1_GREEN_DOT_H4_FILTER_EXIT_SHORT"),
            )

    if position.side == "long" and _h4_invalidated(bar, "long", params):
        return Signal(
            action="close_force",
            reason="H4_LONG_INVALIDATION_EXIT",
            meta=_meta("H4_LONG_INVALIDATION_EXIT"),
        )

    if position.side == "short" and _h4_invalidated(bar, "short", params):
        return Signal(
            action="close_force",
            reason="H4_SHORT_INVALIDATION_EXIT",
            meta=_meta("H4_SHORT_INVALIDATION_EXIT"),
        )

    return Signal(action="none")


def apply_slippage(
    price: float,
    side: Side,
    action: str,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
) -> float:
    """Adjust execution price by slippage and half-spread."""
    if slippage_bps == 0.0 and spread_bps == 0.0:
        return price

    total_bps = slippage_bps + spread_bps / 2.0
    factor = total_bps / 10_000.0
    opening = action == "open"

    if (side == "long" and opening) or (side == "short" and not opening):
        return price * (1.0 + factor)
    return price * (1.0 - factor)


def bar_from_row(row, params: dict) -> BarData:
    """Build BarData for the parameter-specific WaveTrend columns."""
    wt1_col, wt2_col = wt_columns(
        int(params["wt_channel_len"]),
        int(params["wt_avg_len"]),
        int(params["wt_signal_len"]),
    )
    h4_interval = str(params.get("wt_h4_filter_interval", "4h"))
    h4_wt1_col = htf_wt1_column(
        int(params["wt_channel_len"]),
        int(params["wt_avg_len"]),
        h4_interval,
    )
    h4_wt2_col = htf_wt2_column(
        int(params["wt_channel_len"]),
        int(params["wt_avg_len"]),
        int(params["wt_signal_len"]),
        h4_interval,
    )

    wt1 = row[wt1_col] if wt1_col in row.index else row.get("wt1", np.nan)
    wt2 = row[wt2_col] if wt2_col in row.index else row.get("wt2", np.nan)
    wt1 = float(wt1) if not np.isnan(wt1) else np.nan
    wt2 = float(wt2) if not np.isnan(wt2) else np.nan
    h4_wt1 = row[h4_wt1_col] if h4_wt1_col in row.index else row.get("h4_wt1", np.nan)
    h4_wt2 = row[h4_wt2_col] if h4_wt2_col in row.index else row.get("h4_wt2", np.nan)
    h4_wt1 = float(h4_wt1) if not np.isnan(h4_wt1) else np.nan
    h4_wt2 = float(h4_wt2) if not np.isnan(h4_wt2) else np.nan
    bars_since_wt_green_dot = row.get("bars_since_wt_green_dot", np.nan)
    bars_since_wt_red_dot = row.get("bars_since_wt_red_dot", np.nan)
    ema_filter_len = int(params.get("wt_ema_filter_len", 20) or 20)
    ema_col = f"ema_{ema_filter_len}"
    ema_filter_value = row.get(ema_col, row.get("ema20", np.nan))

    return BarData(
        time=row["time"],
        open=float(row.get("open", row["close"])),
        high=float(row.get("high", row["close"])),
        low=float(row.get("low", row["close"])),
        close=float(row["close"]),
        wt1=wt1,
        wt2=wt2,
        wt_delta=float(wt1 - wt2) if not np.isnan(wt1) and not np.isnan(wt2) else np.nan,
        h4_wt1=h4_wt1,
        h4_wt2=h4_wt2,
        h4_wt_delta=float(h4_wt1 - h4_wt2) if not np.isnan(h4_wt1) and not np.isnan(h4_wt2) else np.nan,
        h4_prev_wt1=_float_or_nan(row.get("h4_prev_wt1", np.nan)),
        h4_prev_wt2=_float_or_nan(row.get("h4_prev_wt2", np.nan)),
        h4_prev_wt_delta=_float_or_nan(row.get("h4_prev_wt_delta", np.nan)),
        ema20=_float_or_nan(ema_filter_value),
        ema_filter_len=ema_filter_len,
        htf_ema200=_float_or_nan(row.get("htf_ema200", np.nan)),
        atr=_float_or_nan(row.get("atr", row.get("atr_14", np.nan))),
        wt_green_dot=_flag_value(row.get("wt_green_dot", False)),
        wt_red_dot=_flag_value(row.get("wt_red_dot", False)),
        bars_since_wt_green_dot=(
            _float_or_nan(bars_since_wt_green_dot)
        ),
        bars_since_wt_red_dot=(
            _float_or_nan(bars_since_wt_red_dot)
        ),
    )

