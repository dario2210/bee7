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
    generate_entry_signal,
    generate_exit_signal,
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
    exit_trigger: str = ""
    exit_h4_wt1: float = 0.0
    exit_h4_wt2: float = 0.0
    exit_h4_delta: float = 0.0


class Bee7Strategy:
    """WaveTrend crossover strategy used by backtest, WFO and live runner."""

    def __init__(self, params: dict, fee_rate: float = FEE_RATE):
        self.params = params
        self.fee_rate = params.get("fee_rate", fee_rate)
        self.slippage_bps = params.get("slippage_bps", 0.0)
        self.spread_bps = params.get("spread_bps", 0.0)
        self.position: Optional[PositionState] = None

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

        result = compute_trade_close(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            side=pos.side,
            fee_rate=self.fee_rate,
            capital_at_open=capital_at_open,
        )
        gross_ret = result["gross_ret"]
        net_ret = result["net_ret"]
        fee_ret = result["fee_ret"]
        fee_usd = result["fee_usd"]
        pnl = result["pnl"]
        slip_delta = abs(exit_price - raw_exit)
        slippage_usd = (slip_delta / pos.entry_price) * capital_at_open if pos.entry_price else 0.0
        new_capital = capital + pnl

        em = entry_meta or pos.entry_meta or {}
        xm = signal.meta or {}

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
            capital_before=capital_at_open,
            capital_after=new_capital,
            position_notional=capital_at_open,
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
            exit_trigger=xm.get("exit_trigger", signal.reason),
            exit_h4_wt1=xm.get("exit_h4_wt1", 0.0),
            exit_h4_wt2=xm.get("exit_h4_wt2", 0.0),
            exit_h4_delta=xm.get("exit_h4_delta", 0.0),
        )
        self.position = None
        return rec, new_capital

    def run(self, df, initial_capital):
        capital = initial_capital
        capital_at_open = initial_capital
        trades = []
        equity_curve = []
        self.position = None

        if len(df) == 0:
            return pd.DataFrame(), pd.DataFrame(columns=["time", "equity"]), capital

        equity_curve.append((df["time"].iloc[0], capital))

        for i in range(1, len(df)):
            bar = bar_from_row(df.iloc[i], self.params)
            prev = bar_from_row(df.iloc[i - 1], self.params)

            if any(np.isnan(v) for v in [bar.wt1, bar.wt2, prev.wt1, prev.wt2]):
                continue

            if self.position is not None:
                sig = generate_exit_signal(bar, prev, self.params, self.position)
                if sig.action != "none":
                    rec, capital = self._close_position(capital, bar, sig, capital_at_open)
                    trades.append(rec)
                    equity_curve.append((bar.time, capital))

            if self.position is None:
                sig = generate_entry_signal(bar, prev, self.params, self.position)
                if sig.action in ("open_long", "open_short"):
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
                    self.position.entry_meta["entry_stop_price"] = (
                        round(self.position.stop_price, 4)
                        if not np.isnan(self.position.stop_price)
                        else np.nan
                    )
                    capital_at_open = capital

        if self.position is not None:
            last_bar = bar_from_row(df.iloc[-1], self.params)
            force_sig = Signal(action="close_force", reason="FORCE_EXIT_END", exit_price=last_bar.close)
            rec, capital = self._close_position(capital, last_bar, force_sig, capital_at_open)
            trades.append(rec)
            equity_curve.append((last_bar.time, capital))

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
            "exit_trigger",
            "exit_h4_wt1",
            "exit_h4_wt2",
            "exit_h4_delta",
        ]

        if trades:
            trades_df = pd.DataFrame([{col: getattr(t, col) for col in cols} for t in trades])
        else:
            trades_df = pd.DataFrame(columns=cols)

        equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
        equity_df = equity_df.dropna(subset=["time"]).reset_index(drop=True)
        return trades_df, equity_df, capital

