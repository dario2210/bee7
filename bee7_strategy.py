"""
bee7_strategy.py
=================
Backtest layer for the first bee7 WaveTrend strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from bee7_engine import (
    PositionState,
    Signal,
    apply_slippage,
    bar_from_row,
    build_position_state,
    compute_trade_close,
    generate_emergency_exit_signal,
    generate_entry_signal,
    generate_exit_signal,
    generate_partial_exit_signal,
    generate_short_reversal_entry_signal,
)
from bee7_params import FEE_RATE


@dataclass
class TradeRecord:
    side: str
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    gross_ret: float
    fee_ret: float
    net_ret: float
    pnl: float
    reason: str
    capital_before: float
    capital_after: float
    position_notional: float
    fee_usd: float
    slippage_usd: float
    entry_wt1: float = 0.0
    entry_wt2: float = 0.0
    entry_delta: float = 0.0
    entry_zone: str = ""
    entry_signal_level: float = 0.0
    entry_cross_type: str = ""
    entry_ema_filter: float = 0.0
    entry_ema_filter_len: int = 20
    entry_htf_ema200: float = 0.0
    entry_atr: float = 0.0
    entry_stop_price: float = 0.0
    entry_h4_wt1: float = 0.0
    entry_h4_wt2: float = 0.0
    entry_h4_delta: float = 0.0
    exit_wt1: float = 0.0
    exit_wt2: float = 0.0
    exit_delta: float = 0.0
    exit_zone: str = ""
    exit_signal_level: float = 0.0
    exit_bars: int = 0
    holding_hours: float = np.nan
    time_to_tp1_hours: float = np.nan
    exit_trigger: str = ""
    exit_h4_wt1: float = 0.0
    exit_h4_wt2: float = 0.0
    exit_h4_delta: float = 0.0
    close_fraction: float = 1.0
    remaining_fraction_after: float = 0.0
    logical_trade_no: int = 0
    trade_event: str = ""
    trade_label: str = ""


class Bee7Strategy:
    """WaveTrend crossover strategy used by backtest, WFO and live runner."""

    def __init__(self, params: dict, fee_rate: float = FEE_RATE):
        self.params = params
        self.fee_rate = params.get("fee_rate", fee_rate)
        self.slippage_bps = params.get("slippage_bps", 0.0)
        self.spread_bps = params.get("spread_bps", 0.0)
        self.position: Optional[PositionState] = None
        self.next_trade_id = 1

    @staticmethod
    def _trade_event(signal: Signal) -> str:
        if signal.action == "close_partial":
            if "TP2" in signal.reason:
                return "TP2"
            if "TP1" in signal.reason:
                return "TP1"
            return "TP"
        return "EXIT"

    @staticmethod
    def _holding_hours(entry_time, exit_time) -> float:
        try:
            start = pd.to_datetime(entry_time, utc=True)
            end = pd.to_datetime(exit_time, utc=True)
            if pd.isna(start) or pd.isna(end):
                return np.nan
            return max(0.0, (end - start).total_seconds() / 3600.0)
        except Exception:
            return np.nan

    def _close_position(self, capital, bar, signal, capital_at_open, entry_meta=None):
        pos = self.position
        raw_exit = signal.exit_price if signal.exit_price is not None else bar.close
        exit_price = apply_slippage(
            raw_exit,
            pos.side,
            "close",
            self.slippage_bps,
            self.spread_bps,
        )
        requested_fraction = float((signal.meta or {}).get("close_fraction", pos.remaining_fraction))
        close_fraction = min(max(requested_fraction, 0.0), pos.remaining_fraction)
        close_notional = capital_at_open * close_fraction

        result = compute_trade_close(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            side=pos.side,
            fee_rate=self.fee_rate,
            capital_at_open=close_notional,
        )
        gross_ret = result["gross_ret"]
        net_ret = result["net_ret"]
        fee_ret = result["fee_ret"]
        fee_usd = result["fee_usd"]
        pnl = result["pnl"]
        slip_delta = abs(exit_price - raw_exit)
        slippage_usd = (slip_delta / pos.entry_price) * close_notional if pos.entry_price else 0.0
        new_capital = capital + pnl

        em = entry_meta or pos.entry_meta or {}
        xm = signal.meta or {}
        remaining_after = max(0.0, pos.remaining_fraction - close_fraction)
        trade_id = int(pos.trade_id or 0)
        trade_event = self._trade_event(signal)
        trade_label = f"{trade_id} {trade_event}".strip()
        holding_hours = self._holding_hours(pos.entry_time, bar.time)
        time_to_tp1_hours = holding_hours if trade_event == "TP1" else np.nan

        rec = TradeRecord(
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=bar.time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_ret=gross_ret,
            fee_ret=fee_ret,
            net_ret=net_ret,
            pnl=pnl,
            reason=signal.reason,
            capital_before=capital,
            capital_after=new_capital,
            position_notional=close_notional,
            fee_usd=fee_usd,
            slippage_usd=slippage_usd,
            entry_wt1=em.get("entry_wt1", 0.0),
            entry_wt2=em.get("entry_wt2", 0.0),
            entry_delta=em.get("entry_delta", 0.0),
            entry_zone=em.get("entry_zone", ""),
            entry_signal_level=em.get("entry_signal_level", 0.0),
            entry_cross_type=em.get("cross_type", ""),
            entry_ema_filter=em.get("entry_ema_filter", 0.0),
            entry_ema_filter_len=em.get("entry_ema_filter_len", 20),
            entry_htf_ema200=em.get("entry_htf_ema200", 0.0),
            entry_atr=em.get("entry_atr", 0.0),
            entry_stop_price=em.get("entry_stop_price", pos.stop_price),
            entry_h4_wt1=em.get("entry_h4_wt1", 0.0),
            entry_h4_wt2=em.get("entry_h4_wt2", 0.0),
            entry_h4_delta=em.get("entry_h4_delta", 0.0),
            exit_wt1=xm.get("exit_wt1", 0.0),
            exit_wt2=xm.get("exit_wt2", 0.0),
            exit_delta=xm.get("exit_delta", 0.0),
            exit_zone=xm.get("exit_zone", ""),
            exit_signal_level=xm.get("exit_signal_level", 0.0),
            exit_bars=xm.get("bars_in_position", 0),
            holding_hours=holding_hours,
            time_to_tp1_hours=time_to_tp1_hours,
            exit_trigger=xm.get("exit_trigger", signal.reason),
            exit_h4_wt1=xm.get("exit_h4_wt1", 0.0),
            exit_h4_wt2=xm.get("exit_h4_wt2", 0.0),
            exit_h4_delta=xm.get("exit_h4_delta", 0.0),
            close_fraction=close_fraction,
            remaining_fraction_after=remaining_after,
            logical_trade_no=trade_id,
            trade_event=trade_event,
            trade_label=trade_label,
        )
        if signal.action == "close_partial" and remaining_after > 1e-9:
            pos.remaining_fraction = remaining_after
            if "TP2" in signal.reason:
                pos.tp2_taken = True
            elif "TP1" in signal.reason:
                pos.tp1_taken = True
                pos.tp1_protection_after_bars = pos.bars_in_position + 1
            self.position = pos
        else:
            self.position = None
        return rec, new_capital

    def _open_position_from_signal(self, sig: Signal, bar) -> None:
        side = "long" if sig.action == "open_long" else "short"
        entry_price = apply_slippage(
            bar.close,
            side,
            "open",
            self.slippage_bps,
            self.spread_bps,
        )
        entry_meta = dict(sig.meta or {})
        self.position = build_position_state(
            side=side,
            entry_price=entry_price,
            entry_time=bar.time,
            bar=bar,
            params=self.params,
            entry_meta=entry_meta,
        )
        self.position.trade_id = self.next_trade_id
        self.next_trade_id += 1
        self.position.entry_meta["entry_stop_price"] = (
            round(self.position.stop_price, 4)
            if not np.isnan(self.position.stop_price)
            else np.nan
        )

    def run(self, df, initial_capital):
        capital = initial_capital
        capital_at_open = initial_capital
        trades = []
        equity_curve = []
        self.position = None
        self.next_trade_id = 1

        if len(df) == 0:
            return pd.DataFrame(), pd.DataFrame(columns=["time", "equity"]), capital

        equity_curve.append((df["time"].iloc[0], capital))

        for i in range(1, len(df)):
            bar = bar_from_row(df.iloc[i], self.params)
            prev = bar_from_row(df.iloc[i - 1], self.params)
            pending_entry_signal = None

            if any(np.isnan(v) for v in [bar.wt1, bar.wt2, prev.wt1, prev.wt2]):
                continue

            if self.position is not None:
                sig = generate_emergency_exit_signal(bar, self.params, self.position)
                if sig.action != "none":
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))

            if self.position is not None:
                partial_guard = 0
                sig = generate_partial_exit_signal(bar, self.params, self.position)
                while sig.action != "none" and self.position is not None and partial_guard < 3:
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))
                    partial_guard += 1
                    if self.position is None:
                        break
                    sig = generate_partial_exit_signal(bar, self.params, self.position)

            if self.position is not None:
                sig = generate_exit_signal(bar, prev, self.params, self.position)
                if sig.action != "none":
                    closing_side = self.position.side
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))
                    if closing_side == "long":
                        reverse_sig = generate_short_reversal_entry_signal(bar, prev, self.params, sig)
                        if reverse_sig.action == "open_short":
                            pending_entry_signal = reverse_sig

            if self.position is None:
                sig = pending_entry_signal or generate_entry_signal(bar, prev, self.params, self.position)
                if sig.action in ("open_long", "open_short"):
                    self._open_position_from_signal(sig, bar)
                    capital_at_open = capital

        if self.position is not None:
            # Do not turn an unfinished trade into a synthetic closed trade.
            # This keeps FORCE_EXIT_END out of trade statistics such as Sharpe,
            # expectancy and winrate while preserving the tested time range.
            last_time = df["time"].iloc[-1]
            if not equity_curve or equity_curve[-1][0] != last_time:
                equity_curve.append((last_time, capital))
            self.position = None

        cols = [
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "gross_ret",
            "fee_ret",
            "net_ret",
            "pnl",
            "reason",
            "capital_before",
            "capital_after",
            "position_notional",
            "fee_usd",
            "slippage_usd",
            "entry_wt1",
            "entry_wt2",
            "entry_delta",
            "entry_zone",
            "entry_signal_level",
            "entry_cross_type",
            "entry_ema_filter",
            "entry_ema_filter_len",
            "entry_htf_ema200",
            "entry_atr",
            "entry_stop_price",
            "entry_h4_wt1",
            "entry_h4_wt2",
            "entry_h4_delta",
            "exit_wt1",
            "exit_wt2",
            "exit_delta",
            "exit_zone",
            "exit_signal_level",
            "exit_bars",
            "holding_hours",
            "time_to_tp1_hours",
            "exit_trigger",
            "exit_h4_wt1",
            "exit_h4_wt2",
            "exit_h4_delta",
            "close_fraction",
            "remaining_fraction_after",
            "logical_trade_no",
            "trade_event",
            "trade_label",
        ]

        if trades:
            trades_df = pd.DataFrame([{col: getattr(t, col) for col in cols} for t in trades])
        else:
            trades_df = pd.DataFrame(columns=cols)

        equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
        equity_df = equity_df.dropna(subset=["time"]).reset_index(drop=True)
        return trades_df, equity_df, capital

