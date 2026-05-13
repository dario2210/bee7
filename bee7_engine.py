"""
bee7_engine.py
===============
Shared decision layer for the first bee7 WaveTrend strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from bee7_data import (
    htf_prev_wt1_column,
    htf_prev_wt2_column,
    htf_wt1_column,
    htf_wt2_column,
    wt_columns,
)
from bee7_params import SHORT_TRADING_ENABLED, WT_ZERO_LINE

Side = Literal["long", "short"]
Action = Literal["none", "open_long", "open_short", "close_force", "close_partial"]


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
    remaining_fraction: float = 1.0
    tp1_taken: bool = False
    tp2_taken: bool = False
    tp1_protection_after_bars: int = 0
    h1_red_close_count: int = 0
    h1_green_close_count: int = 0
    trade_id: int = 0


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
    elif side == "long" and bool(params.get("wt_long_emergency_sl_enabled", True)):
        emergency_pct = float(
            params.get("wt_long_emergency_sl_capital_pct", params.get("wt_long_emergency_sl_pct", 0.02)) or 0.0
        )
        if emergency_pct > 0.0:
            stop_price = entry_price * (1.0 - emergency_pct)

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


def _h4_value_changed(bar: BarData, prev_bar: BarData) -> bool:
    if any(np.isnan(v) for v in [bar.h4_wt1, bar.h4_wt2, prev_bar.h4_wt1, prev_bar.h4_wt2]):
        return False
    return bar.h4_wt1 != prev_bar.h4_wt1 or bar.h4_wt2 != prev_bar.h4_wt2


def _h4_cross_down(bar: BarData, prev_bar: BarData) -> bool:
    if any(np.isnan(v) for v in [bar.h4_wt_delta, prev_bar.h4_wt_delta]):
        return False
    prev_delta = bar.h4_prev_wt_delta if not np.isnan(bar.h4_prev_wt_delta) else prev_bar.h4_wt_delta
    if np.isnan(prev_delta):
        return False
    return _h4_value_changed(bar, prev_bar) and prev_delta >= 0.0 and bar.h4_wt_delta < 0.0


def _h4_cross_up(bar: BarData, prev_bar: BarData) -> bool:
    if any(np.isnan(v) for v in [bar.h4_wt_delta, prev_bar.h4_wt_delta]):
        return False
    prev_delta = bar.h4_prev_wt_delta if not np.isnan(bar.h4_prev_wt_delta) else prev_bar.h4_wt_delta
    if np.isnan(prev_delta):
        return False
    return _h4_value_changed(bar, prev_bar) and prev_delta <= 0.0 and bar.h4_wt_delta > 0.0


def _h4_gap_converging(bar: BarData) -> bool:
    if any(np.isnan(v) for v in [bar.h4_wt_delta, bar.h4_prev_wt_delta]):
        return False
    return abs(bar.h4_wt_delta) < abs(bar.h4_prev_wt_delta)


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
    shorts_enabled = bool(params.get("short_trading_enabled", SHORT_TRADING_ENABLED))
    allow_shorts = bool(params.get("allow_shorts", True)) and shorts_enabled
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


def _h4_long_momentum_improving(bar: BarData) -> bool:
    if any(np.isnan(v) for v in [bar.h4_wt_delta, bar.h4_prev_wt_delta]):
        return False
    return bar.h4_wt_delta > bar.h4_prev_wt_delta


def _h4_filter_ok(bar: BarData, side: Side, params: dict) -> bool:
    long_filter_max = float(params.get("wt_h4_long_filter_max", -20.0))
    short_filter_min = float(params.get("wt_h4_short_filter_min", 20.0))

    if side == "long":
        return (
            not np.isnan(bar.h4_wt1)
            and not np.isnan(bar.h4_wt2)
            and bar.h4_wt1 <= long_filter_max
            and bar.h4_wt2 <= long_filter_max
            and _h4_long_momentum_improving(bar)
        )

    return (
        not np.isnan(bar.h4_wt1)
        and not np.isnan(bar.h4_wt2)
        and bar.h4_wt1 >= short_filter_min
        and bar.h4_wt2 >= short_filter_min
        and _h4_converging(bar, "short")
    )


def _long_close_level_ok(bar: BarData, params: dict) -> bool:
    h1_min = float(
        params.get("wt_long_close_min_level", params.get("wt_long_exit_min_level", 0.0))
    )
    h4_min = float(params.get("wt_h4_long_close_min", 0.0))
    return (
        not np.isnan(bar.wt1)
        and not np.isnan(bar.wt2)
        and not np.isnan(bar.h4_wt1)
        and not np.isnan(bar.h4_wt2)
        and bar.wt1 >= h1_min
        and bar.wt2 >= h1_min
        and bar.h4_wt1 >= h4_min
        and bar.h4_wt2 >= h4_min
    )


def _long_entry_window_ok(bar: BarData, prev_bar: BarData, params: dict) -> bool:
    window_bars = int(params.get("wt_long_entry_window_bars", 0) or 0)
    if _cross_up(bar, prev_bar):
        return True
    return window_bars > 0 and _has_recent_signal(bar.bars_since_wt_green_dot, window_bars)


def _long_exit_momentum_weakening(bar: BarData, prev_bar: BarData) -> bool:
    if any(np.isnan(v) for v in [bar.wt1, bar.wt_delta, prev_bar.wt1, prev_bar.wt_delta]):
        return False
    return _cross_down(bar, prev_bar) or bar.wt1 < prev_bar.wt1 or bar.wt_delta < prev_bar.wt_delta


def _entry_meta(bar: BarData, prev_bar: BarData, params: dict) -> dict:
    zero_line = float(params.get("wt_zero_line", WT_ZERO_LINE))
    level_now = _signal_level(bar)
    return {
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


def generate_entry_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: Optional[PositionState],
) -> Signal:
    """
    Entry logic for BEE7:
      - long on a fresh or recent H1 bullish cross in a low H1 zone
      - long uses a softer H4 filter: low/neutral H4 zone + improving H4 delta
      - shorts are opened only by generate_short_reversal_entry_signal()
    """
    if position is not None:
        return Signal(action="none")

    if any(np.isnan(v) for v in [bar.wt1, bar.wt2, prev_bar.wt1, prev_bar.wt2]):
        return Signal(action="none")

    min_level = float(params.get("wt_min_signal_level", 0.0))
    level_now = _signal_level(bar)

    allow_longs, _allow_shorts = _direction_flags(params)
    long_entry_max_above_zero = float(params.get("wt_long_entry_max_above_zero", 0.0))
    long_h4_ok = _h4_filter_ok(bar, "long", params)

    fresh_long_cross = (
        allow_longs
        and _long_entry_window_ok(bar, prev_bar, params)
        and bar.wt1 <= long_entry_max_above_zero
        and bar.wt2 <= long_entry_max_above_zero
        and level_now >= min_level
        and long_h4_ok
    )
    long_cond = fresh_long_cross

    meta = _entry_meta(bar, prev_bar, params)

    if long_cond:
        return Signal(
            action="open_long",
            reason="WT_H1_GREEN_WINDOW_H4_SOFT_FILTER",
            meta={
                **meta,
                "cross_type": "bullish",
                "bars_since_green_dot": round(bar.bars_since_wt_green_dot, 4)
                if not np.isnan(bar.bars_since_wt_green_dot)
                else np.nan,
            },
        )
    return Signal(action="none")


def generate_short_reversal_entry_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    long_exit_signal: Signal,
) -> Signal:
    """Open short only after the normal long close signal fires."""
    _allow_longs, allow_shorts = _direction_flags(params)
    if (
        not allow_shorts
        or long_exit_signal.action != "close_force"
        or long_exit_signal.reason != "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG"
    ):
        return Signal(action="none")

    if any(np.isnan(v) for v in [bar.wt1, bar.wt2, prev_bar.wt1, prev_bar.wt2]):
        return Signal(action="none")

    return Signal(
        action="open_short",
        reason="LONG_EXIT_REVERSE_SHORT",
        meta={
            **_entry_meta(bar, prev_bar, params),
            "cross_type": "bearish_reversal",
            "source_exit_reason": long_exit_signal.reason,
        },
    )


def generate_emergency_exit_signal(
    bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    """Emergency risk exit checked before partial profit taking."""
    if position.side != "long":
        return Signal(action="none")

    emergency_sl_enabled = bool(params.get("wt_long_emergency_sl_enabled", True))
    emergency_capital_pct = float(
        params.get("wt_long_emergency_sl_capital_pct", params.get("wt_long_emergency_sl_pct", 0.02)) or 0.0
    )
    remaining_fraction = max(float(position.remaining_fraction), 1e-9)
    emergency_price_pct = emergency_capital_pct / remaining_fraction
    emergency_stop = position.entry_price * (1.0 - emergency_price_pct)
    if (
        not emergency_sl_enabled
        or emergency_capital_pct <= 0.0
        or emergency_stop <= 0.0
        or np.isnan(bar.low)
        or bar.low > emergency_stop
    ):
        return Signal(action="none")

    return Signal(
        action="close_force",
        reason="LONG_EMERGENCY_SL_CAPITAL",
        exit_price=emergency_stop,
        meta={
            "exit_wt1": round(bar.wt1, 4),
            "exit_wt2": round(bar.wt2, 4),
            "exit_delta": round(bar.wt_delta, 4),
            "exit_signal_level": round(_signal_level(bar), 4),
            "bars_in_position": position.bars_in_position,
            "exit_trigger": "LONG_EMERGENCY_SL_CAPITAL",
            "stop_price": round(emergency_stop, 4),
            "emergency_sl_capital_pct": emergency_capital_pct,
            "emergency_sl_price_pct": emergency_price_pct,
            "remaining_fraction_before": position.remaining_fraction,
            "exit_h4_wt1": round(bar.h4_wt1, 4) if not np.isnan(bar.h4_wt1) else np.nan,
            "exit_h4_wt2": round(bar.h4_wt2, 4) if not np.isnan(bar.h4_wt2) else np.nan,
            "exit_h4_delta": round(bar.h4_wt_delta, 4) if not np.isnan(bar.h4_wt_delta) else np.nan,
        },
    )


def generate_partial_exit_signal(
    bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    """Partial TP management for long and short positions."""
    if position.remaining_fraction <= 0.0:
        return Signal(action="none")

    if position.side == "long":
        if not position.tp1_taken:
            if not bool(params.get("wt_long_tp1_enabled", True)):
                return Signal(action="none")
            tp_name = "TP1"
            tp_pct = float(params.get("wt_long_tp1_pct", 0.01) or 0.0)
            close_fraction = float(params.get("wt_long_tp1_fraction", 1.0 / 3.0) or 0.0)
        elif not position.tp2_taken:
            if not bool(params.get("wt_long_tp2_enabled", True)):
                return Signal(action="none")
            tp_name = "TP2"
            tp_pct = float(params.get("wt_long_tp2_pct", 0.02) or 0.0)
            close_fraction = float(params.get("wt_long_tp2_fraction", 1.0 / 3.0) or 0.0)
        else:
            return Signal(action="none")
        target_price = position.entry_price * (1.0 + tp_pct)
        if np.isnan(bar.high) or bar.high < target_price:
            return Signal(action="none")
        reason = f"LONG_{tp_name}_PARTIAL"
    else:
        if not position.tp1_taken:
            if not bool(params.get("wt_short_tp1_enabled", True)):
                return Signal(action="none")
            tp_name = "TP1"
            tp_pct = float(params.get("wt_short_tp1_pct", params.get("wt_long_tp1_pct", 0.01)) or 0.0)
            close_fraction = float(
                params.get("wt_short_tp1_fraction", params.get("wt_long_tp1_fraction", 1.0 / 3.0)) or 0.0
            )
        elif not position.tp2_taken:
            if not bool(params.get("wt_short_tp2_enabled", params.get("wt_long_tp2_enabled", True))):
                return Signal(action="none")
            tp_name = "TP2"
            tp_pct = float(params.get("wt_short_tp2_pct", params.get("wt_long_tp2_pct", 0.02)) or 0.0)
            close_fraction = float(
                params.get("wt_short_tp2_fraction", params.get("wt_long_tp2_fraction", 1.0 / 3.0)) or 0.0
            )
        else:
            return Signal(action="none")
        target_price = position.entry_price * (1.0 - tp_pct)
        if np.isnan(bar.low) or bar.low > target_price:
            return Signal(action="none")
        reason = f"SHORT_{tp_name}_PARTIAL"

    if tp_pct <= 0.0 or close_fraction <= 0.0:
        return Signal(action="none")

    close_fraction = min(close_fraction, position.remaining_fraction)
    return Signal(
        action="close_partial",
        reason=reason,
        exit_price=target_price,
        meta={
            "exit_wt1": round(bar.wt1, 4),
            "exit_wt2": round(bar.wt2, 4),
            "exit_delta": round(bar.wt_delta, 4),
            "exit_signal_level": round(_signal_level(bar), 4),
            "bars_in_position": position.bars_in_position,
            "exit_trigger": reason,
            "exit_h4_wt1": round(bar.h4_wt1, 4) if not np.isnan(bar.h4_wt1) else np.nan,
            "exit_h4_wt2": round(bar.h4_wt2, 4) if not np.isnan(bar.h4_wt2) else np.nan,
            "exit_h4_delta": round(bar.h4_wt_delta, 4) if not np.isnan(bar.h4_wt_delta) else np.nan,
            "close_fraction": close_fraction,
            "remaining_fraction_before": position.remaining_fraction,
            "tp_level": tp_name,
            "tp_pct": tp_pct,
        },
    )


def generate_exit_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    """
    Exit logic for BEE7:
      - emergency long stop is disabled by default
      - remaining long closes when H1/H4 close levels are met and H1 momentum weakens
      - remaining short closes when the normal long-entry signal appears
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

    if position.side == "long":
        emergency_sig = generate_emergency_exit_signal(bar, params, position)
        if emergency_sig.action != "none":
            emergency_sig.meta["bars_in_position"] = position.bars_in_position
            return emergency_sig
        if (
            bool(params.get("wt_long_tp1_breakeven_enabled", True))
            and position.tp1_taken
            and not position.tp2_taken
            and position.bars_in_position > int(position.tp1_protection_after_bars or 0)
            and not np.isnan(bar.low)
            and bar.low <= position.entry_price
        ):
            meta = _meta("LONG_TP1_BREAKEVEN_EXIT")
            meta["breakeven_price"] = round(position.entry_price, 4)
            meta["remaining_fraction_before"] = position.remaining_fraction
            return Signal(
                action="close_force",
                reason="LONG_TP1_BREAKEVEN_EXIT",
                exit_price=position.entry_price,
                meta=meta,
            )
        if _cross_down(bar, prev_bar):
            position.h1_red_close_count += 1
        if _long_close_level_ok(bar, params) and _long_exit_momentum_weakening(bar, prev_bar):
            meta = _meta("WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG")
            meta["h1_red_close_count"] = position.h1_red_close_count
            meta["long_close_level_h1"] = float(
                params.get("wt_long_close_min_level", params.get("wt_long_exit_min_level", 0.0))
            )
            meta["long_close_level_h4"] = float(params.get("wt_h4_long_close_min", 0.0))
            meta["exit_momentum_weakening"] = True
            meta["exit_wt1_slope"] = round(bar.wt1 - prev_bar.wt1, 4)
            meta["exit_delta_slope"] = round(bar.wt_delta - prev_bar.wt_delta, 4)
            return Signal(
                action="close_force",
                reason="WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG",
                meta=meta,
            )
    if position.side == "short":
        if (
            bool(params.get("wt_short_tp1_breakeven_enabled", params.get("wt_long_tp1_breakeven_enabled", True)))
            and position.tp1_taken
            and not position.tp2_taken
            and position.bars_in_position > int(position.tp1_protection_after_bars or 0)
            and not np.isnan(bar.high)
            and bar.high >= position.entry_price
        ):
            meta = _meta("SHORT_TP1_BREAKEVEN_EXIT")
            meta["breakeven_price"] = round(position.entry_price, 4)
            meta["remaining_fraction_before"] = position.remaining_fraction
            return Signal(
                action="close_force",
                reason="SHORT_TP1_BREAKEVEN_EXIT",
                exit_price=position.entry_price,
                meta=meta,
            )

        long_entry = generate_entry_signal(bar, prev_bar, params, None)
        if long_entry.action == "open_long":
            if _cross_up(bar, prev_bar):
                position.h1_green_close_count += 1
            meta = _meta("WT_LONG_SIGNAL_EXIT_SHORT")
            meta["h1_green_close_count"] = position.h1_green_close_count
            meta["long_entry_reason"] = long_entry.reason
            meta["long_entry_cross_type"] = (long_entry.meta or {}).get("cross_type", "")
            return Signal(
                action="close_force",
                reason="WT_LONG_SIGNAL_EXIT_SHORT",
                meta=meta,
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
    h4_prev_wt1_col = htf_prev_wt1_column(
        int(params["wt_channel_len"]),
        int(params["wt_avg_len"]),
        h4_interval,
    )
    h4_prev_wt2_col = htf_prev_wt2_column(
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
    h4_prev_wt1 = row[h4_prev_wt1_col] if h4_prev_wt1_col in row.index else row.get("h4_prev_wt1", np.nan)
    h4_prev_wt2 = row[h4_prev_wt2_col] if h4_prev_wt2_col in row.index else row.get("h4_prev_wt2", np.nan)
    h4_prev_wt1 = _float_or_nan(h4_prev_wt1)
    h4_prev_wt2 = _float_or_nan(h4_prev_wt2)
    h4_prev_wt_delta = (
        h4_prev_wt1 - h4_prev_wt2
        if not np.isnan(h4_prev_wt1) and not np.isnan(h4_prev_wt2)
        else _float_or_nan(row.get("h4_prev_wt_delta", np.nan))
    )
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
        h4_prev_wt1=h4_prev_wt1,
        h4_prev_wt2=h4_prev_wt2,
        h4_prev_wt_delta=h4_prev_wt_delta,
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

