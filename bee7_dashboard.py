"""
bee7_dashboard.py  –  Bee7 WaveTrend Dashboard  http://IP:8067
"""
from __future__ import annotations
import argparse, json, os, threading
from pathlib import Path
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bee7_params import (
    INITIAL_CAPITAL, FEE_RATE,
    TMA_LOW_MIN, TMA_LOW_MAX, TMA_HIGH_MIN, TMA_HIGH_MAX,
    OPT_DAYS, LIVE_DAYS,
    BINANCE_SYMBOL, BINANCE_INTERVAL, BINANCE_MARKET, BINANCE_START_DATE,
    DEFAULT_PARAMS, save_params,
    WT_CHANNEL_LEN_GRID, WT_AVG_LEN_GRID, WT_SIGNAL_LEN_GRID, WT_MIN_SIGNAL_LEVEL_GRID,
    WT_MIN_SIGNAL_LEVEL_OPTIONS,
    WT_REENTRY_WINDOW_GRID, WT_USE_EMA_FILTER_GRID, WT_EMA_FILTER_LEN_GRID,
    WT_USE_HTF_TREND_FILTER_GRID,
    WT_EMA_FILTER_LEN_OPTIONS,
    WT_LONG_ENTRY_MAX_ABOVE_ZERO_GRID, WT_LONG_ENTRY_MAX_ABOVE_ZERO_OPTIONS,
    WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID, WT_SHORT_ENTRY_MIN_BELOW_ZERO_OPTIONS,
    WT_H4_LONG_FILTER_MAX_GRID, WT_H4_LONG_FILTER_MAX_OPTIONS,
    WT_H4_SHORT_FILTER_MIN_GRID, WT_H4_SHORT_FILTER_MIN_OPTIONS,
)
from bee7_data     import load_klines, prepare_indicators
from bee7_strategy import Bee7Strategy
from bee7_wfo      import get_latest_best_params, walk_forward_optimization
from bee7_stats    import (
    compute_stats, fee_summary_by_period,
    breakdown_by_side, breakdown_by_period, breakdown_by_cross_type,
)
from bee7_binance  import update_csv_cache, wfo_bars

# ─── Kolory ──────────────────────────────────────────────────────────────────
C = {
    "bg"      : "#09111b",
    "surface" : "rgba(11, 21, 36, 0.86)",
    "surf2"   : "rgba(18, 31, 52, 0.9)",
    "border"  : "rgba(131, 153, 179, 0.18)",
    "text"    : "#eef6ff",
    "muted"   : "#95a6bc",
    "green"   : "#22d3aa",
    "red"     : "#ff5d73",
    "blue"    : "#69b7ff",
    "amber"   : "#ffb454",
    "purple"  : "#8b5cf6",
    "coral"   : "#ff7a59",
}

# ─── Global run state ─────────────────────────────────────────────────────────
import time as _time
_SERVER_TOKEN = str(int(_time.time()))   # unikalny token per restart kontenera
_lock   = threading.Lock()
_state  = {"running": False, "stop": False, "status": "", "progress": "",
           "result": None, "result_version": 0}
_chart_df_cache: dict = {}   # (symbol, tf) → df z wskaźnikami
_APP_DIR = Path(__file__).resolve().parent

def gs():
    with _lock: return dict(_state)

def ss(**kw):
    with _lock:
        if "result" in kw:
            _state["result_version"] += 1
        _state.update(kw)

# ─── Helpers ─────────────────────────────────────────────────────────────────
lbl = lambda t: html.Div(t, style={
    "fontSize":"11px","color":C["muted"],"marginBottom":"4px",
    "fontWeight":"500","letterSpacing":"0.05em","textTransform":"uppercase"})

def inp(id_, val, **kw):
    return dcc.Input(id=id_, value=val, debounce=True, style={
        "width":"100%","background":C["surf2"],"border":f"1px solid {C['border']}",
        "borderRadius":"14px","color":C["text"],"padding":"11px 14px","fontSize":"13px",
        "boxSizing":"border-box"}, **kw)

def drp(id_, opts, val):
    return dcc.Dropdown(id=id_, options=opts, value=val, clearable=False,
                        className="wt-drp",
                        style={"color":"#0b1220"})

card_s = {
    "background": C["surface"],
    "border": f"1px solid {C['border']}",
    "borderRadius": "24px",
    "padding": "18px 20px",
    "marginBottom": "14px",
    "boxShadow": "0 20px 60px rgba(0, 0, 0, 0.28)",
    "backdropFilter": "blur(20px)",
}

sec = lambda t: html.Div(t, style={
    "fontSize":"11px","fontWeight":"600","color":C["muted"],
    "textTransform":"uppercase","letterSpacing":"0.06em",
    "marginBottom":"8px","marginTop":"4px"})

def field(label, ctrl):
    return html.Div([lbl(label), ctrl], style={"marginBottom":"10px"})

def btn(id_, label, color=C["blue"], text="#fff"):
    return html.Button(label, id=id_, n_clicks=0, style={
        "width":"100%","background":color,"border":"none","borderRadius":"16px",
        "color":text,"padding":"12px 14px","fontSize":"13px","fontWeight":"700",
        "cursor":"pointer","marginBottom":"8px"})


def _parse_bool_value(value, default=False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _clean_selected_values(values, fallback, caster):
    source = fallback if values is None or len(values) == 0 else values
    cleaned = []
    for value in source:
        casted = _parse_bool_value(value, False) if caster is bool else caster(value)
        if casted not in cleaned:
            cleaned.append(casted)
    return cleaned


def _direction_flags(direction) -> tuple[str, bool, bool]:
    choice = str(direction or DEFAULT_PARAMS.get("trade_direction", "both")).strip().lower()
    if choice == "long":
        return "long", True, False
    if choice == "short":
        return "short", False, True
    return "both", True, True


def _strategy_params_from_controls(
    direction,
    channel_len,
    avg_len,
    signal_len,
    min_level,
    reentry_window,
    ema_filter,
    htf_filter,
    ema_filter_len,
    long_zone,
    short_zone,
    h4_long_filter,
    h4_short_filter,
    fee_rate: float,
    slippage_bps: float,
) -> dict:
    params = dict(DEFAULT_PARAMS)
    trade_direction, allow_longs, allow_shorts = _direction_flags(direction)
    params.update(
        {
            "trade_direction": trade_direction,
            "allow_longs": allow_longs,
            "allow_shorts": allow_shorts,
            "wt_channel_len": int(channel_len if channel_len not in (None, "") else DEFAULT_PARAMS["wt_channel_len"]),
            "wt_avg_len": int(avg_len if avg_len not in (None, "") else DEFAULT_PARAMS["wt_avg_len"]),
            "wt_signal_len": int(signal_len if signal_len not in (None, "") else DEFAULT_PARAMS["wt_signal_len"]),
            "wt_min_signal_level": 0.0,
            "wt_long_entry_window_bars": 0,
            "wt_short_entry_window_bars": 0,
            "wt_long_require_ema20_reclaim": False,
            "wt_short_require_ema20_reject": False,
            "wt_long_require_htf_trend": False,
            "wt_short_require_htf_trend": False,
            "wt_ema_filter_len": int(DEFAULT_PARAMS["wt_ema_filter_len"]),
            "wt_long_entry_max_above_zero": float(
                long_zone if long_zone not in (None, "") else DEFAULT_PARAMS["wt_long_entry_max_above_zero"]
            ),
            "wt_short_entry_min_below_zero": float(
                short_zone if short_zone not in (None, "") else DEFAULT_PARAMS["wt_short_entry_min_below_zero"]
            ),
            "wt_h4_long_filter_max": float(
                h4_long_filter if h4_long_filter not in (None, "") else DEFAULT_PARAMS["wt_h4_long_filter_max"]
            ),
            "wt_h4_short_filter_min": float(
                h4_short_filter if h4_short_filter not in (None, "") else DEFAULT_PARAMS["wt_h4_short_filter_min"]
            ),
            "wt_h4_invalidation_exit_enabled": True,
            "atr_stop_enabled": False,
            "breakeven_trigger_atr": 0.0,
            "trailing_trigger_atr": 0.0,
            "wt_exhaustion_exit_enabled": False,
            "max_bars_in_trade": 0,
            "fee_rate": float(fee_rate),
            "slippage_bps": float(slippage_bps),
        }
    )
    return params


def _grid_overrides_from_controls(
    channel_grid,
    avg_grid,
    signal_grid,
    min_level_grid,
    reentry_grid,
    ema_filter_grid,
    htf_filter_grid,
    ema_len_grid,
    long_zone_grid,
    short_zone_grid,
    h4_long_filter_grid,
    h4_short_filter_grid,
) -> dict:
    return {
        "wt_channel_len": _clean_selected_values(channel_grid, WT_CHANNEL_LEN_GRID, int),
        "wt_avg_len": _clean_selected_values(avg_grid, WT_AVG_LEN_GRID, int),
        "wt_signal_len": _clean_selected_values(signal_grid, WT_SIGNAL_LEN_GRID, int),
        "wt_min_signal_level": _clean_selected_values(min_level_grid, WT_MIN_SIGNAL_LEVEL_GRID, float),
        "wt_reentry_window_bars": _clean_selected_values(reentry_grid, WT_REENTRY_WINDOW_GRID, int),
        "wt_use_ema_filter": _clean_selected_values(ema_filter_grid, WT_USE_EMA_FILTER_GRID, bool),
        "wt_use_htf_filter": _clean_selected_values(htf_filter_grid, WT_USE_HTF_TREND_FILTER_GRID, bool),
        "wt_ema_filter_len": _clean_selected_values(ema_len_grid, WT_EMA_FILTER_LEN_GRID, int),
        "wt_long_entry_max_above_zero": _clean_selected_values(
            long_zone_grid,
            WT_LONG_ENTRY_MAX_ABOVE_ZERO_GRID,
            float,
        ),
        "wt_short_entry_min_below_zero": _clean_selected_values(
            short_zone_grid,
            WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID,
            float,
        ),
        "wt_h4_long_filter_max": _clean_selected_values(
            h4_long_filter_grid,
            WT_H4_LONG_FILTER_MAX_GRID,
            float,
        ),
        "wt_h4_short_filter_min": _clean_selected_values(
            h4_short_filter_grid,
            WT_H4_SHORT_FILTER_MIN_GRID,
            float,
        ),
    }


def _grid_combo_count(grid_overrides: dict) -> int:
    total = 1
    for values in grid_overrides.values():
        total *= max(1, len(values))
    return total


PARAM_SUMMARY_ORDER = [
    "trade_direction",
    "wt_channel_len",
    "wt_avg_len",
    "wt_signal_len",
    "wt_long_entry_max_above_zero",
    "wt_short_entry_min_below_zero",
    "wt_h4_long_filter_max",
    "wt_h4_short_filter_min",
    "wt_h4_invalidation_exit_enabled",
    "fee_rate",
    "slippage_bps",
]

PARAM_SUMMARY_LABELS = {
    "trade_direction": "Direction",
    "wt_channel_len": "Channel",
    "wt_avg_len": "Average",
    "wt_signal_len": "Signal",
    "wt_long_entry_max_above_zero": "Long zone H1",
    "wt_short_entry_min_below_zero": "Short zone H1",
    "wt_h4_long_filter_max": "Long filter H4",
    "wt_h4_short_filter_min": "Short filter H4",
    "wt_h4_invalidation_exit_enabled": "H4 emergency exit",
    "fee_rate": "Fee rate",
    "slippage_bps": "Slippage bps",
}


def _format_param_value(value):
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return round(value, 4)
    return value


def _params_table_frame(params: dict | None) -> pd.DataFrame:
    if not params:
        return pd.DataFrame()

    rows = []
    for key in PARAM_SUMMARY_ORDER:
        if key not in params:
            continue
        rows.append(
            {
                "parametr": PARAM_SUMMARY_LABELS.get(key, key),
                "wartość": _format_param_value(params[key]),
            }
        )

    return pd.DataFrame(rows)


def _params_summary_card(title: str, subtitle: str, params: dict | None) -> html.Div:
    params_df = _params_table_frame(params)
    if params_df.empty:
        return html.Div("Brak parametrów do pokazania.", style={"color": C["muted"]})

    return html.Div([
        html.Div(title, style={"fontSize": "12px", "color": C["text"], "fontWeight": "600", "marginBottom": "4px"}),
        html.Div(subtitle, style={"fontSize": "11px", "color": C["muted"], "marginBottom": "10px"}),
        dash_table.DataTable(
            data=params_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in params_df.columns],
            style_cell={
                "background": C["surface"],
                "color": C["text"],
                "border": f"1px solid {C['border']}",
                "fontSize": "12px",
                "padding": "6px 10px",
                "textAlign": "left",
            },
            style_header={
                "background": C["surf2"],
                "color": C["muted"],
                "fontWeight": "600",
                "fontSize": "10px",
                "textTransform": "uppercase",
                "border": f"1px solid {C['border']}",
            },
        ),
    ], style=card_s)

def mcrd(label, val, sub=None, col=None):
    return html.Div([
        html.Div(label,style={"fontSize":"11px","color":C["muted"],"marginBottom":"4px",
                               "fontWeight":"500","letterSpacing":"0.04em","textTransform":"uppercase"}),
        html.Div(val,  style={"fontSize":"22px","fontWeight":"600","color":col or C["text"]}),
        html.Div(sub,  style={"fontSize":"11px","color":C["muted"],"marginTop":"3px"}) if sub else None,
    ], className="metric-card", style={"background":"linear-gradient(180deg, rgba(255,255,255,0.05), rgba(10,18,31,0.85))",
              "border":f"1px solid {C['border']}","borderRadius":"22px","padding":"18px 18px",
              "minWidth":"150px"})


def hero_banner() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("TradingView open source", className="hero-eyebrow"),
            html.H2("Bee7 WaveTrend console"),
            html.P(
                "Bee7 zachowuje dashboard bee1, ale w galezi BEE7_2 pracuje jako "
                "WaveTrend H1 z natychmiastowym wejsciem na kropce oraz filtrem WT liczonym z H4."
            ),
        ], className="hero-copy"),
        html.Div([
            html.Div("Note", className="hero-note-title"),
            html.P(
                "Long pojawia sie na zielonej kropce H1 przy glebokim WT oraz tylko wtedy, gdy linie WT z H4 sa nisko i zblizaja sie do siebie. "
                "Short dziala lustrzanie, a wyjscie nastepuje dopiero na przeciwnym setupie."
            ),
        ], className="hero-note"),
    ], className="hero-panel")

# ─── Plotly theme ─────────────────────────────────────────────────────────────
PT = {"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)",
      "font":{"color":C["text"],"size":12},
      "xaxis":{"gridcolor":C["border"],"zeroline":False},
      "yaxis":{"gridcolor":C["border"],"zeroline":False},
      "margin":{"l":50,"r":20,"t":30,"b":40}}

def fig_eq(eq_df, capital):
    if eq_df is None or eq_df.empty:
        return go.Figure(layout=PT)
    eq = eq_df["equity"].values
    t  = pd.to_datetime(eq_df["time"])
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.7,0.3],vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=t,y=eq,mode="lines",
        line={"color":C["blue"],"width":2},fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",name="Equity"),row=1,col=1)
    rm  = np.maximum.accumulate(eq)
    dd  = (eq-rm)/rm*100
    fig.add_trace(go.Scatter(x=t,y=dd,mode="lines",
        line={"color":C["red"],"width":1.5},fill="tozeroy",
        fillcolor="rgba(239,68,68,0.12)",name="DD%"),row=2,col=1)
    fig.add_hline(y=capital,line_dash="dot",line_color=C["muted"],row=1,col=1)
    fig.update_layout(**PT,showlegend=False,hovermode="x unified")
    fig.update_yaxes(title_text="Kapitał (USD)",row=1,col=1,gridcolor=C["border"])
    fig.update_yaxes(title_text="DD %",row=2,col=1,gridcolor=C["border"])
    return fig

def fig_wfo(wd):
    if wd is None or wd.empty: return go.Figure(layout=PT)
    clr=[C["green"] if r>0 else C["red"] for r in wd["live_return_pct"]]
    fig=go.Figure(go.Bar(x=wd["window_id"],y=wd["live_return_pct"],marker_color=clr))
    fig.add_hline(y=0,line_color=C["muted"],line_width=1)
    fig.update_layout(**PT,showlegend=False)
    return fig

def fig_pdist(wd):
    if wd is None or wd.empty: return go.Figure(layout=PT)
    specs = [
        ("best_wt_channel_len", "Channel", C["blue"]),
        ("best_wt_avg_len", "Average", C["green"]),
        ("best_wt_signal_len", "Signal", C["amber"]),
        ("best_wt_long_entry_max_above_zero", "Long zone H1", C["green"]),
        ("best_wt_short_entry_min_below_zero", "Short zone H1", C["red"]),
        ("best_wt_h4_long_filter_max", "Long filter H4", C["purple"]),
        ("best_wt_h4_short_filter_min", "Short filter H4", C["coral"]),
    ]
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=[title for _, title, _ in specs],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for idx, (col, _title, clr) in enumerate(specs):
        if col not in wd.columns:
            continue
        row = idx // 2 + 1
        col_idx = idx % 2 + 1
        vc = wd[col].value_counts().sort_index()
        labels = ["on" if v is True else "off" if v is False else str(v) for v in vc.index]
        fig.add_trace(
            go.Bar(x=labels, y=vc.values, marker_color=clr, showlegend=False),
            row=row,
            col=col_idx,
        )
    fig.update_layout(**PT, height=920)
    return fig

def fig_fee(fd):
    if fd is None or fd.empty: return go.Figure(layout=PT)
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=fd["period"].astype(str),y=fd["net_pnl_usd"],
        name="Net PnL",
        marker_color=[C["green"] if v>=0 else C["red"] for v in fd["net_pnl_usd"]]),
        secondary_y=False)
    fig.add_trace(go.Scatter(x=fd["period"].astype(str),y=fd["fee_usd"],
        name="Fee USD",mode="lines+markers",
        line={"color":C["amber"],"width":2},marker={"size":5}),secondary_y=True)
    fig.update_layout(**PT,hovermode="x unified",
        legend={"x":0.01,"y":0.99,"bgcolor":"rgba(0,0,0,0)",
                "font":{"color":C["text"],"size":11}})
    fig.update_yaxes(title_text="Net PnL",secondary_y=False,gridcolor=C["border"])
    fig.update_yaxes(title_text="Fee USD",secondary_y=True,gridcolor="rgba(0,0,0,0)")
    return fig

def _trade_detail_panel(trade: dict | None) -> html.Div:
    """Panel ze szczegolami wybranego trade'u."""
    if not trade:
        return html.Div(
            "Kliknij wiersz w tabeli Transakcje, aby zobaczyc szczegoly trade'u.",
            style={"color": C["muted"], "fontSize": "12px", "marginTop": "12px",
                   "textAlign": "center", "padding": "16px"},
        )

    def fmt_float(value, pattern: str, default: str = "n/d") -> str:
        number = _as_float(value, float("nan"))
        return format(number, pattern) if pd.notna(number) else default

    def fmt_pct(value, digits: int = 3, default: str = "n/d") -> str:
        number = _as_float(value, float("nan"))
        if pd.isna(number):
            return default
        return f"{number * 100:.{digits}f}%"

    def fmt_time(value) -> str:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d %H:%M")
        return str(value or "n/d")

    def fmt_text(value, default: str = "n/d") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text and text.lower() != "nan" else default

    def row2(l1, v1, l2, v2):
        cell_style = {"flex": "1", "minWidth": "160px"}
        value_style = {"fontSize": "13px", "color": C["text"], "fontWeight": "500"}
        return html.Div([
            html.Div([lbl(l1), html.Div(v1, style=value_style)], style=cell_style),
            html.Div([lbl(l2), html.Div(v2, style=value_style)], style=cell_style),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "6px"})

    trade_no = fmt_text(trade.get("trade_no"), "")
    title = f"Trade #{trade_no}" if trade_no else "Szczegoly trade'u"
    side = fmt_text(trade.get("side"), "").lower()
    pnl = _as_float(trade.get("pnl"), 0.0)
    pnl_clr = C["green"] if pnl >= 0 else C["red"]
    side_clr = C["green"] if side == "long" else C["red"]

    return html.Div([
        html.Div(title, style={"fontSize": "11px", "fontWeight": "600",
            "color": C["muted"], "textTransform": "uppercase", "letterSpacing": "0.06em",
            "marginBottom": "10px"}),
        html.Div([
            row2(
                "Side/Powod",
                html.Span(
                    f"{fmt_text(trade.get('side'), '').upper()} -> {fmt_text(trade.get('reason'))}",
                    style={"color": side_clr, "fontWeight": "600"},
                ),
                "Wynik",
                html.Span(
                    f"{pnl:+.2f} USD  ({fmt_pct(trade.get('net_ret'), digits=3, default='+0.000%')})",
                    style={"color": pnl_clr, "fontWeight": "600"},
                ),
            ),
            row2(
                "Wejscie",
                f"{fmt_time(trade.get('entry_time'))}  @  {fmt_float(trade.get('entry_price'), '.4f')}",
                "Wyjscie",
                f"{fmt_time(trade.get('exit_time'))}  @  {fmt_float(trade.get('exit_price'), '.4f')}",
            ),
            row2(
                "Bary w pozycji",
                fmt_text(trade.get("exit_bars", trade.get("bars_in_position", "n/d"))),
                "Fee",
                f"{fmt_float(trade.get('fee_usd'), '.2f', '0.00')} USD",
            ),
            html.Div(style={"borderTop": f"1px solid {C['border']}", "margin": "8px 0"}),
            html.Div("Snapshot wejscia", style={"fontSize": "10px", "color": C["muted"],
                "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em",
                "marginBottom": "6px"}),
            row2("WT1", fmt_float(trade.get("entry_wt1"), ".2f"),
                 "WT2", fmt_float(trade.get("entry_wt2"), ".2f")),
            row2("Delta", fmt_float(trade.get("entry_delta"), ".2f"),
                 "Min level", fmt_float(trade.get("entry_signal_level"), ".2f")),
            row2("EMA filter", fmt_float(trade.get("entry_ema_filter"), ".2f"),
                 "EMA length", fmt_text(trade.get("entry_ema_filter_len", "n/d"))),
            row2("HTF EMA200", fmt_float(trade.get("entry_htf_ema200"), ".2f"),
                 "ATR / stop", f"{fmt_float(trade.get('entry_atr'), '.2f')} / {fmt_float(trade.get('entry_stop_price'), '.2f')}"),
            row2("H4 WT1", fmt_float(trade.get("entry_h4_wt1"), ".2f"),
                 "H4 WT2", fmt_float(trade.get("entry_h4_wt2"), ".2f")),
            row2("H4 delta", fmt_float(trade.get("entry_h4_delta"), ".2f"),
                 "Cross", fmt_text(trade.get("entry_cross_type"))),
            row2("Strefa", fmt_text(trade.get("entry_zone")),
                 "Window WFO", fmt_text(trade.get("window_id", "n/d"))),
            html.Div(style={"borderTop": f"1px solid {C['border']}", "margin": "8px 0"}),
            html.Div("Snapshot wyjscia", style={"fontSize": "10px", "color": C["muted"],
                "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em",
                "marginBottom": "6px"}),
            row2("WT1 wyjscie", fmt_float(trade.get("exit_wt1"), ".2f"),
                 "WT2 wyjscie", fmt_float(trade.get("exit_wt2"), ".2f")),
            row2("Delta wyjscie", fmt_float(trade.get("exit_delta"), ".2f"),
                 "Strefa", fmt_text(trade.get("exit_zone"))),
            row2("H4 WT1 wyjscie", fmt_float(trade.get("exit_h4_wt1"), ".2f"),
                 "H4 WT2 wyjscie", fmt_float(trade.get("exit_h4_wt2"), ".2f")),
            row2("Trigger", fmt_text(trade.get("exit_trigger", trade.get("reason"))),
                 "H4 delta wyjscie", fmt_float(trade.get("exit_h4_delta"), ".2f")),
        ]),
    ], style={"background": C["surface"], "border": f"1px solid {C['border']}",
              "borderRadius": "10px", "padding": "16px 20px", "marginTop": "12px"})


def _chart_source_df(symbol: str, tf: str) -> pd.DataFrame:
    key = (symbol.lower(), tf)
    if key not in _chart_df_cache:
        try:
            csv_path = _APP_DIR / f"{symbol.lower()}_{tf}.csv"
            _chart_df_cache[key] = prepare_indicators(load_klines(str(csv_path)))
        except Exception:
            _chart_df_cache[key] = pd.DataFrame()
    return _chart_df_cache[key]


def _invalidate_chart_cache(symbol: str, tf: str) -> None:
    _chart_df_cache.pop((symbol.lower(), tf), None)


def _utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _unix_seconds(value) -> int:
    return int(_utc_timestamp(value).timestamp())


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _annotate_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    tdf = trades_df.copy().reset_index(drop=True)
    if "trade_no" not in tdf.columns:
        tdf["trade_no"] = np.arange(1, len(tdf) + 1)

    numeric_cols = [
        "entry_price", "exit_price", "gross_ret", "fee_ret", "net_ret",
        "pnl", "fee_usd", "entry_wt1", "entry_wt2", "entry_delta",
        "entry_signal_level", "entry_ema_filter", "entry_ema_filter_len",
        "entry_htf_ema200", "entry_atr", "entry_stop_price",
        "entry_h4_wt1", "entry_h4_wt2", "entry_h4_delta",
        "exit_wt1", "exit_wt2", "exit_delta",
        "exit_signal_level", "exit_h4_wt1", "exit_h4_wt2", "exit_h4_delta",
    ]
    for col in numeric_cols:
        if col in tdf.columns:
            tdf[col] = pd.to_numeric(tdf[col], errors="coerce")

    return tdf


def _normalize_chart_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    tdf = _annotate_trades(trades_df)
    if tdf.empty:
        return tdf

    for col in ("entry_time", "exit_time"):
        if col in tdf.columns:
            tdf[col] = pd.to_datetime(tdf[col], utc=True, errors="coerce")
    return tdf


def _trade_table_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    tdf = _normalize_chart_trades(trades_df)
    if tdf.empty:
        return pd.DataFrame()

    cols = [
        c for c in [
            "trade_no", "side", "entry_time", "exit_time", "entry_price",
            "exit_price", "gross_ret", "fee_ret", "net_ret", "pnl", "fee_usd", "reason",
        ] if c in tdf.columns
    ]
    disp = tdf[cols].copy()
    for col in ["gross_ret", "fee_ret", "net_ret"]:
        if col in disp.columns:
            disp[col] = (disp[col] * 100).round(3).astype(str) + "%"
    for col in ["pnl", "fee_usd", "entry_price", "exit_price"]:
        if col in disp.columns:
            disp[col] = disp[col].round(2)
    for col in ["entry_time", "exit_time"]:
        if col in disp.columns:
            disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d %H:%M")
    if "trade_no" in disp.columns:
        disp["trade_no"] = disp["trade_no"].astype(int)
    return disp


def _serialize_trade_record(trade_row: pd.Series | dict) -> dict:
    data = trade_row.to_dict() if isinstance(trade_row, pd.Series) else dict(trade_row)
    return json.loads(pd.DataFrame([data]).to_json(orient="records", date_format="iso"))[0]


def _filter_chart_trades(trades_df: pd.DataFrame, filter_mode: str = "all") -> pd.DataFrame:
    tdf = _normalize_chart_trades(trades_df)
    if tdf.empty:
        return tdf

    if filter_mode == "long":
        tdf = tdf[tdf["side"] == "long"]
    elif filter_mode == "short":
        tdf = tdf[tdf["side"] == "short"]
    elif filter_mode in ("REVERSE", "FORCE", "TIME") and "reason" in tdf.columns:
        tdf = tdf[tdf["reason"].astype(str).str.startswith(filter_mode)]

    return tdf.reset_index(drop=True)


def _crop_chart_df(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    windows_df: pd.DataFrame | None = None,
    selected_trade: dict | None = None,
    pad_bars: int = 96,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    bounds: list[pd.Timestamp] = []

    if trades_df is not None and not trades_df.empty:
        for col in ("entry_time", "exit_time"):
            if col not in trades_df.columns:
                continue
            series = pd.to_datetime(trades_df[col], utc=True, errors="coerce").dropna()
            if not series.empty:
                bounds.extend([series.min(), series.max()])

    if windows_df is not None and not windows_df.empty:
        for col in ("live_start", "live_end"):
            if col not in windows_df.columns:
                continue
            series = pd.to_datetime(windows_df[col], utc=True, errors="coerce").dropna()
            if not series.empty:
                bounds.extend([series.min(), series.max()])

    if selected_trade:
        for value in (selected_trade.get("entry_time"), selected_trade.get("exit_time")):
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(ts):
                bounds.append(ts)

    if not bounds:
        if len(df) <= 3000:
            return df
        return df.iloc[-3000:].reset_index(drop=True)

    start_bound = min(bounds)
    end_bound = max(bounds)
    times = df["time"]
    start_idx = max(0, int(times.searchsorted(start_bound, side="left")) - pad_bars)
    end_idx = min(len(df) - 1, int(times.searchsorted(end_bound, side="right")) + pad_bars)
    return df.iloc[start_idx:end_idx + 1].reset_index(drop=True)


def lightweight_chart_payload(
    symbol: str,
    tf: str,
    trades_df: pd.DataFrame,
    windows_df: pd.DataFrame | None = None,
    selected_trade: dict | None = None,
    filter_mode: str = "all",
    params: dict | None = None,
) -> dict[str, object]:
    source_df = _chart_source_df(symbol, tf)
    if source_df is None or source_df.empty:
        return {
            "candles": [],
            "lines": [],
            "signalLines": [],
            "markers": [],
            "tradePins": [],
            "focusRange": None,
            "selectedTradeNo": None,
        }

    normalized_trades = _normalize_chart_trades(trades_df)
    filtered_trades = _filter_chart_trades(normalized_trades, filter_mode)
    df = _crop_chart_df(
        source_df,
        normalized_trades,
        windows_df=windows_df,
        selected_trade=selected_trade,
    )
    if df.empty:
        return {
            "candles": [],
            "lines": [],
            "signalLines": [],
            "markers": [],
            "tradePins": [],
            "focusRange": None,
            "selectedTradeNo": None,
        }

    candles = [
        {
            "time": _unix_seconds(row.time),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in df.itertuples(index=False)
    ]

    params = params or {}
    ema_filter_len = int(params.get("wt_ema_filter_len", DEFAULT_PARAMS["wt_ema_filter_len"]) or 20)
    ema_col = f"ema_{ema_filter_len}"
    line_specs = [
        (ema_col, f"EMA filter {ema_filter_len}", C["purple"], 2.0, True),
    ]
    lines = []
    for column, label, color, width, visible in line_specs:
        if column not in df.columns:
            continue
        series = []
        for row in df[["time", column]].itertuples(index=False):
            value = getattr(row, column)
            if pd.isna(value):
                continue
            series.append({"time": _unix_seconds(row.time), "value": round(float(value), 6)})
        lines.append(
            {
                "id": column,
                "label": label,
                "color": color,
                "lineWidth": width,
                "visible": visible,
                "data": series,
            }
        )

    signal_specs = [
        ("wt1", "WT1", C["blue"], 1.9, True),
        ("wt2", "WT2", C["amber"], 2.2, True),
    ]
    signal_lines = []
    for column, label, color, width, visible in signal_specs:
        if column not in df.columns:
            continue
        series = []
        for row in df[["time", column]].itertuples(index=False):
            value = getattr(row, column)
            if pd.isna(value):
                continue
            series.append({"time": _unix_seconds(row.time), "value": round(float(value), 4)})
        signal_lines.append(
            {
                "id": column,
                "label": label,
                "color": color,
                "lineWidth": width,
                "visible": visible,
                "data": series,
            }
        )

    time_values = [_unix_seconds(ts) for ts in df["time"]]
    level_specs = [
        ("wt-low-min", "WT lower band", TMA_LOW_MIN, "rgba(34, 211, 170, 0.22)", 1),
        ("wt-low-max", "WT lower zone", TMA_LOW_MAX, "rgba(34, 211, 170, 0.22)", 1),
        ("wt-zero", "WT zero", 0.0, "rgba(149, 166, 188, 0.20)", 1),
        ("wt-high-min", "WT upper zone", TMA_HIGH_MIN, "rgba(255, 180, 84, 0.22)", 1),
        ("wt-high-max", "WT upper band", TMA_HIGH_MAX, "rgba(255, 180, 84, 0.22)", 1),
    ]
    for line_id, label, level, color, width in level_specs:
        signal_lines.append(
            {
                "id": line_id,
                "label": label,
                "color": color,
                "lineWidth": width,
                "visible": True,
                "data": [{"time": time_value, "value": float(level)} for time_value in time_values],
            }
        )

    markers: list[dict[str, object]] = []
    trade_pins: list[dict[str, object]] = []
    if not filtered_trades.empty:
        for trade in filtered_trades.itertuples(index=False):
            trade_no = int(_as_float(getattr(trade, "trade_no", 0), 0.0))
            side = str(getattr(trade, "side", "")).lower()
            pnl = _as_float(getattr(trade, "pnl", 0.0), 0.0)
            reason = str(getattr(trade, "reason", "") or "")
            entry_price = _as_float(getattr(trade, "entry_price", 0.0), 0.0)
            exit_price = _as_float(getattr(trade, "exit_price", 0.0), 0.0)
            markers.append(
                {
                    "time": _unix_seconds(trade.entry_time),
                    "position": "belowBar" if side == "long" else "aboveBar",
                    "shape": "arrowUp" if side == "long" else "arrowDown",
                    "color": C["green"] if side == "long" else C["coral"],
                    "text": f"#{trade_no}",
                }
            )
            markers.append(
                {
                    "time": _unix_seconds(trade.exit_time),
                    "position": "aboveBar" if side == "long" else "belowBar",
                    "shape": "circle",
                    "color": "#22c55e" if pnl >= 0 else "#ef4444",
                    "text": f"#{trade_no}",
                }
            )
            trade_pins.append(
                {
                    "tradeNo": trade_no,
                    "time": _unix_seconds(trade.entry_time),
                    "price": round(entry_price, 6),
                    "label": f"#{trade_no}",
                    "anchor": "below" if side == "long" else "above",
                    "kind": "entry",
                    "side": side,
                    "color": C["green"] if side == "long" else C["coral"],
                    "tooltip": f"Trade #{trade_no} entry | {side.upper()} | {reason or 'OPEN'}",
                }
            )
            trade_pins.append(
                {
                    "tradeNo": trade_no,
                    "time": _unix_seconds(trade.exit_time),
                    "price": round(exit_price, 6),
                    "label": f"#{trade_no}",
                    "anchor": "above" if side == "long" else "below",
                    "kind": "exit",
                    "side": side,
                    "color": "#22c55e" if pnl >= 0 else "#ef4444",
                    "tooltip": f"Trade #{trade_no} exit | {side.upper()} | {reason or f'{pnl:+.2f} USD'}",
                }
            )

    if windows_df is not None and not windows_df.empty and "live_start" in windows_df.columns and len(windows_df) <= 20:
        for row in windows_df.itertuples(index=False):
            live_start = pd.to_datetime(getattr(row, "live_start", None), utc=True, errors="coerce")
            if pd.isna(live_start):
                continue
            markers.append(
                {
                    "time": _unix_seconds(live_start),
                    "position": "aboveBar",
                    "shape": "square",
                    "color": C["blue"],
                    "text": f"W{getattr(row, 'window_id', '')}",
                }
            )

    focus_range = None
    selected_trade_no = None
    if selected_trade:
        selected_trade_no = selected_trade.get("trade_no")
        entry_time = pd.to_datetime(selected_trade.get("entry_time"), utc=True, errors="coerce")
        exit_time = pd.to_datetime(selected_trade.get("exit_time"), utc=True, errors="coerce")
        if not pd.isna(entry_time):
            times = df["time"]
            entry_idx = int(times.searchsorted(entry_time, side="left"))
            exit_idx = int(times.searchsorted(exit_time, side="left")) if not pd.isna(exit_time) else entry_idx
            start_idx = max(0, min(entry_idx, len(df) - 1) - 48)
            end_idx = min(len(df) - 1, max(entry_idx, exit_idx, 0) + 48)
            focus_range = {
                "from": _unix_seconds(times.iloc[start_idx]),
                "to": _unix_seconds(times.iloc[end_idx]),
            }

    return {
        "symbol": symbol,
        "tf": tf,
        "candles": candles,
        "lines": lines,
        "signalLines": signal_lines,
        "markers": markers,
        "tradePins": trade_pins,
        "focusRange": focus_range,
        "selectedTradeNo": selected_trade_no,
    }


def fig_chart(
    symbol        : str,
    tf            : str,
    trades_df     : pd.DataFrame,
    windows_df    : pd.DataFrame | None = None,
    selected_trade: dict | None          = None,
    filter_mode   : str                  = "all",
) -> go.Figure:
    """Wykres cenowy z EMA, markerami transakcji, RSI/TMA i nakładką WFO."""
    key = (symbol.lower(), tf)
    if key not in _chart_df_cache:
        try:
            _chart_df_cache[key] = prepare_indicators(
                load_klines(str(_APP_DIR / f"{symbol.lower()}_{tf}.csv")))
        except Exception:
            return go.Figure(layout=PT)
    df = _chart_df_cache[key]
    if df is None or df.empty:
        return go.Figure(layout=PT)

    # ── filtrowanie transakcji ────────────────────────────────────────────────
    tdf = pd.DataFrame()
    if not trades_df.empty:
        tdf = trades_df.copy()
        for c in ("entry_time", "exit_time"):
            if c in tdf.columns:
                tdf[c] = pd.to_datetime(tdf[c], utc=True, errors="coerce")
        if filter_mode == "long":
            tdf = tdf[tdf["side"] == "long"]
        elif filter_mode == "short":
            tdf = tdf[tdf["side"] == "short"]
        elif filter_mode in ("TP","SL","TRAIL","TIME") and "reason" in tdf.columns:
            tdf = tdf[tdf["reason"].str.startswith(filter_mode)]
        tdf = tdf.reset_index(drop=True)

    # ── wykryj wybrany trade ──────────────────────────────────────────────────
    sel_entry_t = sel_exit_t = None
    if selected_trade:
        try:
            sel_entry_t = pd.to_datetime(selected_trade.get("entry_time"), utc=True, errors="coerce")
            sel_exit_t  = pd.to_datetime(selected_trade.get("exit_time"),  utc=True, errors="coerce")
        except Exception:
            pass

    # ── subploty ──────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.03,
    )

    # ── świece ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing={"line":{"color":C["green"],"width":1},"fillcolor":C["green"]},
        decreasing={"line":{"color":C["red"],  "width":1},"fillcolor":C["red"]},
        hoverinfo="x+y",
    ), row=1, col=1)

    # ── EMA ────────────────────────────────────────────────────────────────────
    for col_name, clr, wid, vis in [
        ("ema_200", C["amber"],  2.0, True),
        ("ema_150", C["purple"], 1.2, "legendonly"),
        ("ema_100", C["blue"],   1.2, "legendonly"),
        ("ema_50",  C["muted"],  1.0, "legendonly"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[col_name], mode="lines",
                name=col_name.replace("_"," ").upper(),
                line={"color":clr,"width":wid}, visible=vis,
            ), row=1, col=1)

    # ── nakładka okien WFO ────────────────────────────────────────────────────
    if windows_df is not None and not windows_df.empty and "live_start" in windows_df.columns:
        wdf_times = windows_df.copy()
        wdf_times["live_start"] = pd.to_datetime(wdf_times["live_start"], utc=True, errors="coerce")
        wdf_times["live_end"]   = pd.to_datetime(wdf_times["live_end"],   utc=True, errors="coerce")
        colors_wfo = ["rgba(59,130,246,0.04)", "rgba(168,85,247,0.04)"]
        for i, row_w in wdf_times.iterrows():
            if pd.isna(row_w["live_start"]) or pd.isna(row_w["live_end"]):
                continue
            fig.add_vrect(
                x0=row_w["live_start"], x1=row_w["live_end"],
                fillcolor=colors_wfo[i % 2], line_width=0,
                layer="below", row=1, col=1,
            )
        # pionowe linie granic okien
        for _, row_w in wdf_times.iterrows():
            if pd.isna(row_w["live_start"]):
                continue
            fig.add_vline(
                x=row_w["live_start"].timestamp() * 1000,
                line_dash="dot", line_color=C["border"], line_width=1,
                row=1, col=1,
            )

    # ── helper: hover text dla wejścia ───────────────────────────────────────
    def _entry_hover(t):
        lines = [
            f"{'LONG' if t.get('side')=='long' else 'SHORT'} IN",
            f"Cena: {t.get('entry_price',0):.4f}",
            f"RSI: {t.get('entry_rsi',0):.2f}  TMA: {t.get('entry_tma_rsi',0):.2f}",
            f"EMA: {t.get('entry_ema',0):.2f}  slope: {t.get('entry_ema_slope',0)*100:.4f}%",
            f"Dist: {t.get('entry_ema_dist',0)*100:.3f}%",
            f"Strefa TMA: {t.get('entry_tma_zone','')}  cross: {t.get('entry_cross_type','')}",
            f"TP: {t.get('entry_tp_price',0):.4f}  SL: {t.get('entry_sl_price',0):.4f}",
        ]
        return "<br>".join(lines)

    def _exit_hover(t):
        pnl = t.get('pnl', 0)
        pnl_s = f"+{pnl:.2f}" if pnl > 0 else f"{pnl:.2f}"
        lines = [
            f"OUT {t.get('reason','')}",
            f"Cena: {t.get('exit_price',0):.4f}",
            f"PnL: {pnl_s} USD  net_ret: {t.get('net_ret',0)*100:.3f}%",
            f"Fee: {t.get('fee_usd',0):.2f} USD",
            f"RSI wyj: {t.get('exit_rsi',0):.2f}  TMA: {t.get('exit_tma_rsi',0):.2f}",
            f"RSI peak: {t.get('exit_rsi_peak','n/d')}  over_zone: {t.get('exit_over_zone',False)}",
            f"Bary: {t.get('exit_bars',0)}",
        ]
        if t.get("window_id") is not None:
            lines.append(f"Okno WFO: {t.get('window_id')}")
        return "<br>".join(lines)

    # ── linie zysk/strata ─────────────────────────────────────────────────────
    if not tdf.empty:
        for mask, clr, lbl in [
            (tdf["pnl"] > 0,  C["green"], "Zysk"),
            (tdf["pnl"] <= 0, C["red"],   "Strata"),
        ]:
            sub = tdf[mask]
            if not sub.empty:
                xs, ys = [], []
                for _, t in sub.iterrows():
                    xs += [t["entry_time"], t["exit_time"], None]
                    ys += [t["entry_price"], t["exit_price"], None]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", name=lbl, opacity=0.40,
                    line={"color":clr,"width":1,"dash":"dot"},
                ), row=1, col=1)

        # ── markery wejść Long ────────────────────────────────────────────────
        lng = tdf[tdf["side"] == "long"]
        if not lng.empty:
            fig.add_trace(go.Scatter(
                x=lng["entry_time"], y=lng["entry_price"],
                mode="markers", name="Long IN",
                marker={"symbol":"triangle-up","size":11,"color":C["green"],
                        "line":{"width":1,"color":"#fff"}},
                customdata=lng.to_dict("records"),
                hovertemplate=[_entry_hover(r) + "<extra></extra>"
                               for r in lng.to_dict("records")],
            ), row=1, col=1)

        # ── markery wejść Short ───────────────────────────────────────────────
        sht = tdf[tdf["side"] == "short"]
        if not sht.empty:
            fig.add_trace(go.Scatter(
                x=sht["entry_time"], y=sht["entry_price"],
                mode="markers", name="Short IN",
                marker={"symbol":"triangle-down","size":11,"color":C["red"],
                        "line":{"width":1,"color":"#fff"}},
                customdata=sht.to_dict("records"),
                hovertemplate=[_entry_hover(r) + "<extra></extra>"
                               for r in sht.to_dict("records")],
            ), row=1, col=1)

        # ── markery wyjść ─────────────────────────────────────────────────────
        _exit_style = {
            "TP":    (C["green"], "star",     12),
            "SL":    (C["red"],   "x",        10),
            "TRAIL": (C["amber"], "diamond",   9),
            "TIME":  (C["muted"], "circle",    8),
        }
        if "reason" in tdf.columns:
            for reason, (clr, sym, sz) in _exit_style.items():
                sub = tdf[tdf["reason"].str.startswith(reason) if "reason" in tdf.columns else tdf.index < 0]
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=sub["exit_time"], y=sub["exit_price"],
                        mode="markers", name=f"OUT {reason}",
                        marker={"symbol":sym,"size":sz,"color":clr,
                                "line":{"width":1.5,"color":clr}},
                        customdata=sub.to_dict("records"),
                        hovertemplate=[_exit_hover(r) + "<extra></extra>"
                                       for r in sub.to_dict("records")],
                    ), row=1, col=1)

    # ── podświetlenie wybranego trade'u ───────────────────────────────────────
    if sel_entry_t is not None and sel_exit_t is not None:
        sel_price_in  = selected_trade.get("entry_price", 0)
        sel_price_out = selected_trade.get("exit_price",  0)
        sel_pnl       = selected_trade.get("pnl", 0)
        sel_clr       = C["green"] if sel_pnl >= 0 else C["red"]
        fig.add_trace(go.Scatter(
            x=[sel_entry_t, sel_exit_t], y=[sel_price_in, sel_price_out],
            mode="lines+markers", name="► Wybrany trade",
            line={"color":sel_clr,"width":3},
            marker={"size":14,"color":sel_clr,"symbol":["triangle-up","x"],
                    "line":{"width":2,"color":"#fff"}},
        ), row=1, col=1)

    # ── RSI ────────────────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["rsi"], mode="lines", name="RSI",
            line={"color":C["blue"],"width":1.5},
        ), row=2, col=1)
    if "tma_rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["tma_rsi"], mode="lines", name="TMA(RSI)",
            line={"color":C["purple"],"width":1.5},
        ), row=2, col=1)

    # strefy TMA
    fig.add_hrect(y0=TMA_LOW_MIN,  y1=TMA_LOW_MAX,
                  fillcolor=C["green"], opacity=0.10, line_width=0, row=2, col=1)
    fig.add_hrect(y0=TMA_HIGH_MIN, y1=TMA_HIGH_MAX,
                  fillcolor=C["red"],   opacity=0.10, line_width=0, row=2, col=1)
    for yv in (30, 50, 70):
        fig.add_hline(y=yv, line_dash="dot", line_color=C["border"],
                      line_width=1, row=2, col=1)

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        **PT, height=720, showlegend=True, hovermode="x unified",
        legend={"orientation":"v","x":1.01,"y":1,"bgcolor":"rgba(0,0,0,0)",
                "font":{"size":10,"color":C["text"]}},
        xaxis_rangeslider_visible=False,
    )
    fig.update_layout(margin={"l":60,"r":150,"t":20,"b":40})
    fig.update_yaxes(title_text="Cena (USDT)", row=1, col=1, gridcolor=C["border"])
    fig.update_yaxes(title_text="RSI", row=2, col=1,
                     gridcolor=C["border"], range=[0, 100])
    fig.update_xaxes(rangeslider_visible=False, type="date")

    # ── zakres X ─────────────────────────────────────────────────────────────
    if sel_entry_t is not None and sel_exit_t is not None and not df.empty:
        # zoom do wybranego trade'u ± 30 barów
        bar_times = df["time"].values
        i_entry = np.searchsorted(bar_times, np.datetime64(sel_entry_t.to_datetime64()))
        i_exit  = np.searchsorted(bar_times, np.datetime64(sel_exit_t.to_datetime64()))
        i0 = max(0, i_entry - 30)
        i1 = min(len(bar_times)-1, i_exit  + 30)
        fig.update_xaxes(range=[bar_times[i0], bar_times[i1]])
    elif not df.empty:
        t_end   = df["time"].max()
        t_start = t_end - pd.Timedelta(days=90)
        fig.update_xaxes(range=[t_start, t_end])

    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar():
    SW = "285px"
    return html.Div([
        # nagłówek
        html.Div([
            html.Div("Bee7 Bot",style={"fontSize":"16px","fontWeight":"700","color":C["text"]}),
            html.Div("WaveTrend mirrored profile",style={"fontSize":"11px","color":C["muted"],"marginTop":"2px"}),
        ],style={"marginBottom":"20px","paddingBottom":"14px","borderBottom":f"1px solid {C['border']}"}),

        # dane
        html.Div([
            sec("Dane"),
            field("Symbol", inp("inp-sym", BINANCE_SYMBOL, type="text")),
            field("Timeframe", drp("inp-tf",[
                {"label":tf,"value":tf}
                for tf in ["1m","5m","15m","30m","1h","2h","4h","12h","1d"]
            ], BINANCE_INTERVAL)),
            field("Rynek", drp("inp-mkt",[
                {"label":"Spot","value":"spot"},
                {"label":"Futures","value":"futures"},
            ],"spot")),
            html.Div([
                html.Div([lbl("Data od"),
                    dcc.Input(id="inp-from",value="2021-01-01",type="text",debounce=True,
                        style={"width":"100%","background":"#22263a","border":f"1px solid {C['border']}",
                               "borderRadius":"6px","color":C["text"],"padding":"6px 8px","fontSize":"13px"})
                ],style={"flex":"1"}),
                html.Div([lbl("Data do"),
                    dcc.Input(id="inp-to",value="",type="text",debounce=True,
                        placeholder="dziś",
                        style={"width":"100%","background":"#22263a","border":f"1px solid {C['border']}",
                               "borderRadius":"6px","color":C["text"],"padding":"6px 8px","fontSize":"13px"})
                ],style={"flex":"1"}),
            ],style={"display":"flex","gap":"8px","marginBottom":"10px"}),
            field("Kapitał (USD)", inp("inp-cap", INITIAL_CAPITAL, type="number", min=100, step=100)),
        ],style=card_s),

        html.Div([
            sec("Uruchomienie"),
            field("Tryb", drp("inp-run-mode", [
                {"label": "Backtest manualny", "value": "backtest"},
                {"label": "WFO", "value": "wfo"},
            ], "wfo")),
            field("Kierunek", drp("inp-direction", [
                {"label": "Oba kierunki", "value": "both"},
                {"label": "Tylko long", "value": "long"},
                {"label": "Tylko short", "value": "short"},
            ], DEFAULT_PARAMS.get("trade_direction", "both"))),
            html.Div([
                html.Div([field("Fee rate %", inp("inp-fee", round(FEE_RATE*100,4),
                                                   type="number",min=0,max=1,step=0.001))],style={"flex":"1"}),
                html.Div([field("Slippage bps", inp("inp-slip", DEFAULT_PARAMS.get("slippage_bps", 0.0),
                                                     type="number",min=0,max=20,step=0.5))],style={"flex":"1"}),
            ],style={"display":"flex","gap":"8px"}),
            html.Div("⚠ fee_ret = −(2×fee)  wejście + wyjście",
                style={"fontSize":"10px","color":C["amber"],"marginTop":"-6px","marginBottom":"8px"}),
        ],style=card_s),

        html.Div([
            sec("Backtest ręczny"),
            html.Div([
                html.Div([field("Channel", inp("inp-bt-channel", DEFAULT_PARAMS["wt_channel_len"], type="number", min=2, step=1))], style={"flex":"1"}),
                html.Div([field("Average", inp("inp-bt-avg", DEFAULT_PARAMS["wt_avg_len"], type="number", min=2, step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Signal", inp("inp-bt-signal", DEFAULT_PARAMS["wt_signal_len"], type="number", min=2, step=1))], style={"flex":"1"}),
                html.Div(
                    [field("Min level", inp("inp-bt-min-level", DEFAULT_PARAMS["wt_min_signal_level"], type="number", step=1))],
                    style={"display":"none"},
                ),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Long zone H1", inp("inp-bt-long-zone", DEFAULT_PARAMS["wt_long_entry_max_above_zero"], type="number", step=1))], style={"flex":"1"}),
                html.Div([field("Short zone H1", inp("inp-bt-short-zone", DEFAULT_PARAMS["wt_short_entry_min_below_zero"], type="number", step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Long filter H4", inp("inp-bt-h4-long", DEFAULT_PARAMS["wt_h4_long_filter_max"], type="number", step=1))], style={"flex":"1"}),
                html.Div([field("Short filter H4", inp("inp-bt-h4-short", DEFAULT_PARAMS["wt_h4_short_filter_min"], type="number", step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Re-entry", inp("inp-bt-reentry", DEFAULT_PARAMS["wt_long_entry_window_bars"], type="number", min=0, max=12, step=1))], style={"display":"none"}),
                html.Div([field("EMA filter", drp("inp-bt-ema-filter", [
                    {"label": "Off", "value": False},
                    {"label": "On", "value": True},
                ], DEFAULT_PARAMS["wt_long_require_ema20_reclaim"]))], style={"display":"none"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("HTF trend", drp("inp-bt-htf-filter", [
                    {"label": "Off", "value": False},
                    {"label": "On", "value": True},
                ], DEFAULT_PARAMS["wt_long_require_htf_trend"]))], style={"display":"none"}),
                html.Div([field("EMA length", inp("inp-bt-ema-len", DEFAULT_PARAMS["wt_ema_filter_len"], type="number", min=2, max=200, step=1))], style={"display":"none"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div(
                "BEE7_2: wejście jest od razu na świeżej kropce H1 w głębokiej strefie WT, a filtr robią dwie linie WaveTrend z H4. Stop loss, re-entry i filtry EMA są wyłączone, a awaryjny exit zamyka pozycję gdy H4 zaczyna przeczyć setupowi.",
                style={"fontSize":"11px","color":C["muted"],"marginTop":"4px"},
            ),
        ],style=card_s),

        html.Div([
            sec("Konfiguracja WFO"),
            html.Div([
                html.Div([field("Opt (dni)", inp("inp-opt", OPT_DAYS,  type="number",min=14,max=365,step=7))],style={"flex":"1"}),
                html.Div([field("Live (dni)",inp("inp-live",LIVE_DAYS, type="number",min=7, max=90, step=7))],style={"flex":"1"}),
            ],style={"display":"flex","gap":"8px"}),
            field("Scoring", drp("inp-score",[
                {"label":"Balanced",    "value":"balanced"},
                {"label":"Return only", "value":"return_only"},
                {"label":"Defensive",   "value":"defensive"},
            ],"balanced")),
            sec("Siatka Channel"),
            dcc.Checklist(id="chk-grid-channel",
                options=[{"label":f" {v}","value":v}
                         for v in WT_CHANNEL_LEN_GRID],
                value=WT_CHANNEL_LEN_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Siatka Average"),
            dcc.Checklist(id="chk-grid-avg",
                options=[{"label":f" {v}","value":v}
                         for v in WT_AVG_LEN_GRID],
                value=WT_AVG_LEN_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Siatka Signal"),
            dcc.Checklist(id="chk-grid-signal",
                options=[{"label":f" {v}","value":v} for v in WT_SIGNAL_LEN_GRID],
                value=WT_SIGNAL_LEN_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            html.Div([
                sec("Siatka Min level"),
                dcc.Checklist(id="chk-grid-min-level",
                    options=[{"label":f" {v:.1f}","value":v} for v in WT_MIN_SIGNAL_LEVEL_OPTIONS],
                    value=WT_MIN_SIGNAL_LEVEL_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
                sec("Siatka Re-entry"),
                dcc.Checklist(id="chk-grid-reentry",
                    options=[{"label":f" {v}","value":v} for v in WT_REENTRY_WINDOW_GRID],
                    value=WT_REENTRY_WINDOW_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
                sec("Siatka EMA on/off"),
                dcc.Checklist(id="chk-grid-ema-filter",
                    options=[
                        {"label":" Off","value":False},
                        {"label":" On","value":True},
                    ],
                    value=WT_USE_EMA_FILTER_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
                sec("Siatka HTF trend"),
                dcc.Checklist(id="chk-grid-htf-filter",
                    options=[
                        {"label":" Off","value":False},
                        {"label":" On","value":True},
                    ],
                    value=WT_USE_HTF_TREND_FILTER_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
                sec("Siatka EMA length"),
                dcc.Checklist(id="chk-grid-ema-len",
                    options=[{"label":f" {v}","value":v} for v in WT_EMA_FILTER_LEN_OPTIONS],
                    value=WT_EMA_FILTER_LEN_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            ], style={"display":"none"}),
            sec("Long zone max"),
            dcc.Checklist(id="chk-grid-long-zone",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_LONG_ENTRY_MAX_ABOVE_ZERO_OPTIONS],
                value=WT_LONG_ENTRY_MAX_ABOVE_ZERO_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Short zone min"),
            dcc.Checklist(id="chk-grid-short-zone",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_SHORT_ENTRY_MIN_BELOW_ZERO_OPTIONS],
                value=WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Long filter H4"),
            dcc.Checklist(id="chk-grid-h4-long",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_H4_LONG_FILTER_MAX_OPTIONS],
                value=WT_H4_LONG_FILTER_MAX_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Short filter H4"),
            dcc.Checklist(id="chk-grid-h4-short",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_H4_SHORT_FILTER_MIN_OPTIONS],
                value=WT_H4_SHORT_FILTER_MIN_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            html.Div(
                "WFO w BEE7_2 testuje głębokość wejścia H1, osobne progi H4 dla long/short oraz klasyczne długości WaveTrend. Awaryjny exit H4 jest stały i włączony.",
                style={"fontSize":"11px","color":C["muted"],"marginTop":"8px"},
            ),
        ],id="panel-wfo",style=card_s),

        # przyciski
        html.Div([
            html.Button("▶  Uruchom", id="btn-run",  n_clicks=0, style={
                "flex":"1","background":C["blue"],"border":"none","borderRadius":"8px",
                "color":"#fff","padding":"10px","fontSize":"13px","fontWeight":"600","cursor":"pointer"}),
            html.Button("■  Stop",    id="btn-stop", n_clicks=0, style={
                "flex":"1","background":C["red"],"border":"none","borderRadius":"8px",
                "color":"#fff","padding":"10px","fontSize":"13px","fontWeight":"600","cursor":"pointer",
                "opacity":"0.4"},
                disabled=True),
        ],id="btn-row",style={"display":"flex","gap":"8px","marginBottom":"8px"}),

        html.Div(id="run-status",style={
            "fontSize":"12px","color":C["muted"],"textAlign":"center","minHeight":"20px"}),
        html.Div(id="run-progress",style={
            "fontSize":"12px","color":C["amber"],"textAlign":"center","minHeight":"16px","marginTop":"4px"}),

        dcc.Store(id="store-result"),
        dcc.Store(id="store-run-meta", data={"running": False}),

    ],style={
        "width":"320px","minWidth":"320px","height":"100vh","overflowY":"auto",
        "background":"rgba(8, 16, 26, 0.92)","borderRight":f"1px solid {C['border']}",
        "padding":"24px 18px","boxSizing":"border-box",
        "position":"fixed","top":"0","left":"0","scrollbarWidth":"thin",
        "backdropFilter":"blur(24px)",
    })

# ─── Main panel ───────────────────────────────────────────────────────────────
def main_panel():
    ts = {"fontSize":"13px","padding":"8px 16px","color":C["muted"],
          "background":"transparent","border":"none",
          "borderBottom":"2px solid transparent","cursor":"pointer"}
    ta = {**ts,"color":C["text"],"borderBottom":f"2px solid {C['blue']}"}
    return html.Div([
        hero_banner(),
        html.Div(id="metrics-row",style={"display":"grid","gap":"14px",
                                          "gridTemplateColumns":"repeat(auto-fit, minmax(160px, 1fr))",
                                          "marginBottom":"18px"}),
        dcc.Tabs(id="tabs",value="equity",children=[
            dcc.Tab(label="Equity / DD",  value="equity",  style=ts,selected_style=ta),
            dcc.Tab(label="Wykres",       value="chart",   style=ts,selected_style=ta),
            dcc.Tab(label="Transakcje",   value="trades",  style=ts,selected_style=ta),
            dcc.Tab(label="Opłaty",       value="fees",    style=ts,selected_style=ta),
            dcc.Tab(label="Okna WFO",     value="wfo-win", style=ts,selected_style=ta),
            dcc.Tab(label="Parametry",    value="wfo-par", style=ts,selected_style=ta),
            dcc.Tab(label="Breakdown",    value="brkdwn",  style=ts,selected_style=ta),
        ],style={"borderBottom":f"1px solid {C['border']}","marginBottom":"14px"}),
        html.Div(id="tab-content"),
        dcc.Store(id="store-selected-trade", data=None),
        dcc.Store(id="store-chart-filter",   data="all"),
        dcc.Store(id="store-chart-payload",  data=None),
        dcc.Store(id="store-server-token",   data=None, storage_type="session"),
        dcc.Download(id="download-trades"),
        html.Div(id="chart-render-signal", style={"display":"none"}),
        dcc.Interval(id="poll",interval=15000,n_intervals=0),
        dcc.Location(id="_url", refresh=True),
    ],style={"marginLeft":"320px","padding":"24px","minHeight":"100vh",
             "background":"transparent","boxSizing":"border-box"})

# ─── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Bee7 WaveTrend Dashboard",
    suppress_callback_exceptions=True,
    external_scripts=[
        "https://unpkg.com/lightweight-charts@5.0.8/dist/lightweight-charts.standalone.production.js",
    ],
)

# CSS: dropdown czarne litery na białym tle
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
{%css%}
<style>
* { box-sizing: border-box; }
body {
    margin:0;
    min-height:100vh;
    color:""" + C["text"] + """;
    background:
        radial-gradient(circle at top left, rgba(105, 183, 255, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 180, 84, 0.12), transparent 28%),
        linear-gradient(160deg, #08101a 0%, #0b1524 45%, #132238 100%);
    font-family:"Space Grotesk","Segoe UI",sans-serif;
}
pre, code, .dash-spreadsheet-container {
    font-family:"IBM Plex Mono",Consolas,monospace;
}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:""" + C["border"] + """;border-radius:3px}

/* ── Dropdown ── */
.wt-drp,
.wt-drp > div,
.wt-drp .Select-control,
.wt-drp .Select-control *,
.wt-drp [class*="control"],
.wt-drp [class*="control"] *,
.wt-drp [class*="singleValue"],
.wt-drp [class*="singleValue"] *,
.wt-drp [class*="valueContainer"],
.wt-drp [class*="valueContainer"] *,
.wt-drp input {
    color:#0b1220 !important;
    -webkit-text-fill-color:#0b1220 !important;
    opacity:1 !important;
}
.wt-drp .Select-control {
    background: #f5f7fb !important;
    border: 1px solid """ + C["border"] + """ !important;
    border-radius: 14px !important;
    min-height: 42px !important;
    cursor: pointer !important;
}
.wt-drp .Select-placeholder { color:#0b1220 !important; opacity:1 !important; font-size:13px !important; }
.wt-drp .Select-value        { color:#0b1220 !important; background:#f5f7fb !important; border-radius:10px !important; margin:5px 8px !important; padding:0 8px !important; border:none !important; }
.wt-drp .Select-value-label  { color:#0b1220 !important; opacity:1 !important; font-size:13px !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp .Select-input input  { color:#0b1220 !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp .Select.has-value.Select--single > .Select-control .Select-value,
.wt-drp .Select.has-value.Select--single > .Select-control .Select-value .Select-value-label { color:#0b1220 !important; opacity:1 !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp .Select-arrow        { border-top-color: #334155 !important; }
.wt-drp .Select-menu-outer {
    background: #142336 !important;
    border: 1px solid """ + C["border"] + """ !important;
    z-index: 9999 !important;
}
.wt-drp .Select-menu-outer,
.wt-drp .Select-menu-outer *,
.wt-drp .Select-menu,
.wt-drp .Select-menu *,
.wt-drp [class*="menu"],
.wt-drp [class*="menu"] *,
.wt-drp .Select-option,
.wt-drp .VirtualizedSelectOption,
.wt-drp [class*="option"] {
    color: """ + C["text"] + """ !important;
    -webkit-text-fill-color: """ + C["text"] + """ !important;
}
.wt-drp .Select-option {
    background: #142336 !important;
    color: """ + C["text"] + """ !important;
    font-size: 13px !important;
}
.wt-drp .Select-option:hover,
.wt-drp .Select-option.is-focused { background: rgba(105, 183, 255, 0.18) !important; color:""" + C["text"] + """ !important; }
.wt-drp .Select-option.is-selected { background: rgba(34, 211, 170, 0.18) !important; color:#fff !important; }

/* Dash v2 react-select */
.wt-drp [class*="control"]       { background:#f5f7fb !important; border-color:""" + C["border"] + """ !important; min-height:42px !important; box-shadow:none !important; }
.wt-drp [class*="singleValue"]   { color:#0b1220 !important; background:#f5f7fb !important; border-radius:10px !important; padding:2px 8px !important; font-size:13px !important; opacity:1 !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp [class*="singleValue"] * { color:#0b1220 !important; opacity:1 !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp [class*="placeholder"]   { color:#0b1220 !important; opacity:1 !important; font-size:13px !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp [class*="placeholder"] * { color:#0b1220 !important; opacity:1 !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp [class*="valueContainer"],
.wt-drp [class*="valueContainer"] * { color:#0b1220 !important; opacity:1 !important; -webkit-text-fill-color:#0b1220 !important; }
.wt-drp [class*="menu"]          { background:#142336 !important; border:1px solid """ + C["border"] + """ !important; z-index:9999 !important; }
.wt-drp [class*="option"]        { background:#142336 !important; color:""" + C["text"] + """ !important; font-size:13px !important; }
.wt-drp [class*="option"]:hover  { background:rgba(105, 183, 255, 0.18) !important; }
.wt-drp [class*="dropdownIndicator"] svg { fill:#334155 !important; }
.wt-drp [class*="indicatorSeparator"] { background:""" + C["border"] + """ !important; }
.wt-drp [class*="Input"] input   { color:#0b1220 !important; opacity:1 !important; -webkit-text-fill-color:#0b1220 !important; }

/* ── Checkboxy ── */
input[type=checkbox] { accent-color:""" + C["blue"] + """; width:14px; height:14px; cursor:pointer; }

/* ── Tabela ── */
.dash-spreadsheet-container .dash-spreadsheet-inner td,
.dash-spreadsheet-container .dash-spreadsheet-inner th {
    font-size:12px !important; color:""" + C["text"] + """ !important;
    background:""" + C["surface"] + """ !important; border:1px solid """ + C["border"] + """ !important;
}
.dash-spreadsheet-container .dash-spreadsheet-inner th {
    background:""" + C["surf2"] + """ !important; font-weight:600 !important;
    text-transform:uppercase !important; font-size:10px !important; letter-spacing:0.05em !important;
}
input[type=number],input[type=text]{outline:none;}
input[type=number]:focus,input[type=text]:focus{border-color:""" + C["blue"] + """ !important;}
button:hover{opacity:0.88;} button:active{transform:scale(0.98);}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""

app.layout = html.Div([sidebar(), main_panel()])

app.clientside_callback(
    """
    function(chartPayload, activeTab) {
        if (activeTab !== "chart") {
            return window.dash_clientside.no_update;
        }
        if (!chartPayload) {
            return "";
        }
        if (window.dash_clientside && window.dash_clientside.bee7_dashboard) {
            return window.dash_clientside.bee7_dashboard.render(chartPayload);
        }
        return "";
    }
    """,
    Output("chart-render-signal", "children"),
    Input("store-chart-payload", "data"),
    Input("tabs", "value"),
)

# ─── Worker thread ─────────────────────────────────────────────────────────────
def _worker(
    symbol,
    tf,
    market,
    start,
    end,
    capital,
    run_mode,
    direction,
    fee,
    slip,
    opt_days,
    live_days,
    score_mode,
    bt_channel,
    bt_avg,
    bt_signal,
    bt_min_level,
    bt_reentry,
    bt_ema_filter,
    bt_htf_filter,
    bt_ema_len,
    bt_long_zone,
    bt_short_zone,
    bt_h4_long,
    bt_h4_short,
    grid_channel,
    grid_avg,
    grid_signal,
    grid_min_level,
    grid_reentry,
    grid_ema_filter,
    grid_htf_filter,
    grid_ema_len,
    grid_long_zone,
    grid_short_zone,
    grid_h4_long,
    grid_h4_short,
):

    csv_path = str(_APP_DIR / f"{symbol.lower()}_{tf}.csv")
    try:
        # 1) Pobierz dane
        ss(status="Pobieram dane z Binance...", progress="")
        update_csv_cache(csv_path=csv_path, symbol=symbol, interval=tf,
                         start_date=start or BINANCE_START_DATE,
                         market=market or "spot", verbose=False)
        _invalidate_chart_cache(symbol, tf)

        if gs()["stop"]:
            ss(running=False, status="Zatrzymano.", progress=""); return

        ss(status="Wczytuję i przygotowuję dane...")
        df = prepare_indicators(load_klines(csv_path))
        _chart_df_cache[(symbol.lower(), tf)] = df.copy()

        # ── walidacja i filtrowanie dat ────────────────────────────────────────
        def _parse_date(s, label):
            """Parsuje datę ze stringa — zwraca Timestamp lub None przy błędzie."""
            if not s or str(s).strip() == "":
                return None
            try:
                return pd.Timestamp(str(s).strip(), tz="UTC")
            except Exception:
                ss(running=False,
                   status=f"✗ Nieprawidłowa data '{s}' w polu '{label}'. "
                          f"Użyj formatu YYYY-MM-DD.", progress="")
                return "ERROR"

        t_start = _parse_date(start, "Data od")
        if t_start == "ERROR": return
        t_end   = _parse_date(end,   "Data do")
        if t_end   == "ERROR": return

        if t_start: df = df[df["time"] >= t_start]
        if t_end:   df = df[df["time"] <= t_end]
        df = df.reset_index(drop=True)

        if len(df) < 100:
            ss(running=False, status="✗ Za mało danych po filtrowaniu.", progress=""); return

        fee_rate_val = float(fee if fee is not None and fee != "" else FEE_RATE * 100) / 100.0
        slip_bps_val = float(slip if slip is not None and slip != "" else DEFAULT_PARAMS.get("slippage_bps", 0.0))
        run_mode = str(run_mode or "wfo").lower()
        strategy_params = _strategy_params_from_controls(
            direction,
            bt_channel,
            bt_avg,
            bt_signal,
            bt_min_level,
            bt_reentry,
            bt_ema_filter,
            bt_htf_filter,
            bt_ema_len,
            bt_long_zone,
            bt_short_zone,
            bt_h4_long,
            bt_h4_short,
            fee_rate_val,
            slip_bps_val,
        )

        if run_mode == "backtest":
            ss(status="Backtest w toku...", progress="")
            strat = Bee7Strategy(strategy_params, fee_rate=fee_rate_val)
            trades_bt, equity_bt, _ = strat.run(df, capital)
            stats = compute_stats(trades_bt, equity_bt, capital, print_output=False)
            fee_df = fee_summary_by_period(trades_bt, capital, "ME") if not trades_bt.empty else pd.DataFrame()
            side_bk = breakdown_by_side(trades_bt) if not trades_bt.empty else pd.DataFrame()
            cross_bk = breakdown_by_cross_type(trades_bt) if not trades_bt.empty else pd.DataFrame()
            yr_bk = breakdown_by_period(trades_bt, "YE") if not trades_bt.empty else pd.DataFrame()
            q_bk = breakdown_by_period(trades_bt, "QE") if not trades_bt.empty else pd.DataFrame()
            result = {
                "mode": "backtest",
                "stats": stats,
                "capital": capital,
                "tf": tf,
                "symbol": symbol,
                "params_used": strategy_params,
                "trades": trades_bt.to_dict("records") if not trades_bt.empty else [],
                "equity": equity_bt.to_dict("records") if equity_bt is not None and not equity_bt.empty else [],
                "fee_df": fee_df.to_dict("records") if not fee_df.empty else [],
                "side_bk": side_bk.to_dict("records") if not side_bk.empty else [],
                "cross_bk": cross_bk.to_dict("records") if not cross_bk.empty else [],
                "yr_bk": yr_bk.to_dict("records") if not yr_bk.empty else [],
                "q_bk": q_bk.to_dict("records") if not q_bk.empty else [],
                "windows_df": [],
            }
            n = len(trades_bt)
            ret = stats.get("net_return_pct", 0)
            ss(
                running=False,
                stop=False,
                result=result,
                status=f"✓ Backtest gotowy  |  {n} tradów  |  {ret:+.2f}%",
                progress="",
            )
            return

        opt_days_val = int(opt_days or OPT_DAYS)
        live_days_val = int(live_days or LIVE_DAYS)
        grid_overrides = _grid_overrides_from_controls(
            grid_channel,
            grid_avg,
            grid_signal,
            grid_min_level,
            grid_reentry,
            grid_ema_filter,
            grid_htf_filter,
            grid_ema_len,
            grid_long_zone,
            grid_short_zone,
            grid_h4_long,
            grid_h4_short,
        )

        ob, lb = wfo_bars(tf, opt_days_val, live_days_val)
        total  = max(0, (len(df)-ob)//lb)
        combo_total = _grid_combo_count(grid_overrides)
        ss(
            status=f"WFO w toku  |  ~{total} okien  |  {combo_total} kombinacji/okno",
            progress=f"Okno 1 / {max(total, 1)}  |  kombinacja 0 / {combo_total}",
        )

        def _on_combo_progress(wid, total_windows, combo_idx, combo_total):
            total_windows = max(int(total_windows or 0), 1)
            window_no = int(wid) + 1
            pct = (combo_idx / combo_total * 100.0) if combo_total else 0.0
            ss(
                progress=(
                    f"Okno {window_no} / {total_windows}  |  "
                    f"kombinacja {combo_idx} / {combo_total}  |  {pct:.0f}%"
                )
            )

        def _on_window_done(wid, total, wstats, trades_list, equity_sofar, cap_sofar):
            ss(progress=f"Okno {wid + 1} / {total}  |  kombinacja {combo_total} / {combo_total}  |  100%")
            # Zbuduj wynik cząstkowy i od razu go wyświetl
            try:
                all_tr = pd.concat(trades_list, ignore_index=True) \
                         if trades_list else pd.DataFrame()
                wdf    = pd.DataFrame(wstats)
                st     = compute_stats(all_tr, equity_sofar, capital, print_output=False)
                fd     = fee_summary_by_period(all_tr, capital, "ME") \
                         if not all_tr.empty else pd.DataFrame()
                sb     = breakdown_by_side(all_tr)    if not all_tr.empty else pd.DataFrame()
                cb     = breakdown_by_cross_type(all_tr) if not all_tr.empty else pd.DataFrame()
                yb     = breakdown_by_period(all_tr, "YE") if not all_tr.empty else pd.DataFrame()
                qb     = breakdown_by_period(all_tr, "QE") if not all_tr.empty else pd.DataFrame()
                r = {
                    "mode":"wfo","stats":st,"capital":capital,"tf":tf,"symbol":symbol,
                    "trades"    : all_tr.to_dict("records") if not all_tr.empty else [],
                    "equity"    : equity_sofar.to_dict("records")
                                  if equity_sofar is not None and not equity_sofar.empty else [],
                    "fee_df"    : fd.to_dict("records") if not fd.empty else [],
                    "side_bk"   : sb.to_dict("records") if not sb.empty else [],
                    "cross_bk"  : cb.to_dict("records") if not cb.empty else [],
                    "yr_bk"     : yb.to_dict("records") if not yb.empty else [],
                    "q_bk"      : qb.to_dict("records") if not qb.empty else [],
                    "windows_df": wdf.to_dict("records") if not wdf.empty else [],
                }
                nw  = len(wdf)
                n   = len(all_tr)
                ret = st.get("net_return_pct", 0)
                ss(result=r,
                   status=f"WFO w toku  |  okno {wid+1}/{total}  |  "
                          f"{n} tradów  |  {ret:+.2f}%")
            except Exception:
                pass  # błąd w preview nie powinien przerywać WFO

        all_trades, equity_wfo, windows_df, _final_cap, stopped = walk_forward_optimization(
            df,
            interval=tf,
            score_mode=score_mode or "balanced",
            verbose=False,
            on_window_done=_on_window_done,
            on_combo_progress=_on_combo_progress,
            should_stop=lambda: gs()["stop"],
            fee_rate=fee_rate_val,
            opt_days=opt_days_val,
            live_days=live_days_val,
            initial_capital=capital,
            base_params=strategy_params,
            grid_overrides=grid_overrides,
        )

        stats   = compute_stats(all_trades, equity_wfo, capital, print_output=False)
        fee_df  = fee_summary_by_period(all_trades, capital, "ME") if all_trades is not None and not all_trades.empty else pd.DataFrame()
        side_bk = breakdown_by_side(all_trades) if all_trades is not None and not all_trades.empty else pd.DataFrame()
        cross_bk = breakdown_by_cross_type(all_trades) if all_trades is not None and not all_trades.empty else pd.DataFrame()
        yr_bk   = breakdown_by_period(all_trades, "YE") if all_trades is not None and not all_trades.empty else pd.DataFrame()
        q_bk    = breakdown_by_period(all_trades, "QE") if all_trades is not None and not all_trades.empty else pd.DataFrame()
        best_params = get_latest_best_params(windows_df) if windows_df is not None and not windows_df.empty else {}
        if best_params:
            best_params = {**strategy_params, **best_params}
        if best_params:
            save_params(best_params, str(_APP_DIR / "bee7_wfo_best_params.json"))

        result = {
            "mode":"wfo","stats":stats,"capital":capital,"tf":tf,"symbol":symbol,
            "params_used": strategy_params,
            "trades"    : all_trades.to_dict("records")  if all_trades is not None and not all_trades.empty else [],
            "equity"    : equity_wfo.to_dict("records")  if equity_wfo is not None and not equity_wfo.empty else [],
            "fee_df"    : fee_df.to_dict("records")      if not fee_df.empty  else [],
            "side_bk"   : side_bk.to_dict("records")    if not side_bk.empty else [],
            "cross_bk"  : cross_bk.to_dict("records")   if not cross_bk.empty else [],
            "yr_bk"     : yr_bk.to_dict("records")      if not yr_bk.empty   else [],
            "q_bk"      : q_bk.to_dict("records")       if not q_bk.empty    else [],
            "windows_df": windows_df.to_dict("records")  if windows_df is not None and not windows_df.empty else [],
            "best_params": best_params,
        }
        nw  = len(windows_df) if windows_df is not None and not windows_df.empty else 0
        n   = len(all_trades) if all_trades is not None and not all_trades.empty else 0
        ret = stats.get("net_return_pct", 0)
        if stopped:
            ss(
                running=False,
                stop=False,
                result=result,
                status=f"Zatrzymano  |  ukończone okna {nw}/{total}  |  {n} tradów  |  {ret:+.2f}%",
                progress="",
            )
            return

        ss(running=False, stop=False, result=result,
           status=f"✓ WFO gotowe  |  {nw} okien  |  {n} tradów  |  {ret:+.2f}%"
                  + ("  |  zapisano best params" if best_params else ""),
           progress="")

    except Exception as e:
        import traceback
        ss(running=False, stop=False,
           status=f"✗ Błąd: {str(e)[:100]}", progress=traceback.format_exc()[-200:])

# ─── Callback: auto-reload gdy serwer był restartowany ───────────────────────
@app.callback(
    Output("_url","href"),
    Output("store-server-token","data"),
    Input("poll","n_intervals"),
    State("store-server-token","data"),
    prevent_initial_call=True,
)
def check_server_token(_, stored_token):
    if stored_token is None:
        # pierwsze załadowanie — zapamiętaj token
        return dash.no_update, _SERVER_TOKEN
    if stored_token != _SERVER_TOKEN:
        # serwer był restartowany — wymuś przeładowanie strony
        return "/", _SERVER_TOKEN
    return dash.no_update, dash.no_update

# ─── Callback: filter wykresu → osobny store ──────────────────────────────────
@app.callback(
    Output("store-chart-filter","data"),
    Input("chart-filter-radio","value"),
    prevent_initial_call=True,
)
def on_chart_filter(fval):
    return fval

# ─── Callback: klik w tabeli → zoom na wykresie ───────────────────────────────
@app.callback(
    Output("store-selected-trade","data"),
    Output("tabs","value"),
    Input("trades-table","active_cell"),
    State("trades-table","derived_viewport_data"),
    State("store-result","data"),
    prevent_initial_call=True,
)
def on_trade_select(active_cell, viewport_data, result_data):
    if not active_cell or not viewport_data or not result_data:
        return dash.no_update, dash.no_update
    row_idx = active_cell.get("row", -1)
    if row_idx < 0 or row_idx >= len(viewport_data):
        return dash.no_update, dash.no_update

    trade_no = viewport_data[row_idx].get("trade_no")
    trades_df = _normalize_chart_trades(pd.DataFrame(result_data.get("trades", [])))
    if trades_df.empty or "trade_no" not in trades_df.columns:
        return dash.no_update, dash.no_update

    try:
        trade_no = int(float(trade_no))
    except Exception:
        return dash.no_update, dash.no_update

    selected = trades_df[trades_df["trade_no"] == trade_no]
    if selected.empty:
        return dash.no_update, dash.no_update
    return _serialize_trade_record(selected.iloc[0]), "chart"


@app.callback(
    Output("download-trades", "data"),
    Input("btn-export-trades", "n_clicks"),
    State("store-result", "data"),
    prevent_initial_call=True,
)
def export_trades(n_clicks, result_data):
    if not n_clicks or not result_data:
        return dash.no_update

    trades_df = _normalize_chart_trades(pd.DataFrame(result_data.get("trades", [])))
    if trades_df.empty:
        return dash.no_update

    export_df = trades_df.copy()
    for col in ("entry_time", "exit_time"):
        if col in export_df.columns:
            export_df[col] = pd.to_datetime(export_df[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")

    mode = str(result_data.get("mode", "run")).lower()
    symbol = str(result_data.get("symbol", "asset")).upper()
    tf = str(result_data.get("tf", "tf")).lower()
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"bee7_{mode}_{symbol}_{tf}_trades_{stamp}.csv"
    return dcc.send_data_frame(export_df.to_csv, filename, index=False, encoding="utf-8-sig")

# ─── Callback: uruchom / stop ─────────────────────────────────────────────────
@app.callback(
    Output("btn-run","disabled"),
    Output("btn-stop","disabled"),
    Output("btn-stop","style"),
    Input("btn-run","n_clicks"),
    Input("btn-stop","n_clicks"),
    State("inp-sym","value"),  State("inp-tf","value"),
    State("inp-mkt","value"),  State("inp-from","value"),
    State("inp-to","value"),   State("inp-cap","value"),
    State("inp-run-mode","value"),
    State("inp-direction","value"),
    State("inp-fee","value"),  State("inp-slip","value"),
    State("inp-opt","value"),  State("inp-live","value"),
    State("inp-score","value"),
    State("inp-bt-channel","value"), State("inp-bt-avg","value"),
    State("inp-bt-signal","value"), State("inp-bt-min-level","value"),
    State("inp-bt-reentry","value"), State("inp-bt-ema-filter","value"),
    State("inp-bt-htf-filter","value"),
    State("inp-bt-ema-len","value"),
    State("inp-bt-long-zone","value"), State("inp-bt-short-zone","value"),
    State("inp-bt-h4-long","value"), State("inp-bt-h4-short","value"),
    State("chk-grid-channel","value"), State("chk-grid-avg","value"),
    State("chk-grid-signal","value"), State("chk-grid-min-level","value"),
    State("chk-grid-reentry","value"), State("chk-grid-ema-filter","value"),
    State("chk-grid-htf-filter","value"),
    State("chk-grid-ema-len","value"),
    State("chk-grid-long-zone","value"), State("chk-grid-short-zone","value"),
    State("chk-grid-h4-long","value"), State("chk-grid-h4-short","value"),
    prevent_initial_call=True,
)
def on_run_stop(nr, ns,
    sym, tf, mkt, frm, to, cap,
    run_mode, direction, fee, slip, opt, live, score,
    bt_channel, bt_avg, bt_signal, bt_min_level,
    bt_reentry, bt_ema_filter, bt_htf_filter, bt_ema_len, bt_long_zone, bt_short_zone, bt_h4_long, bt_h4_short,
    grid_channel, grid_avg, grid_signal, grid_min_level,
    grid_reentry, grid_ema_filter, grid_htf_filter, grid_ema_len, grid_long_zone, grid_short_zone,
    grid_h4_long, grid_h4_short):

    _sty_active = {"flex":"1","background":C["red"],"border":"none","borderRadius":"8px",
                   "color":"#fff","padding":"10px","fontSize":"13px","fontWeight":"600",
                   "cursor":"pointer","opacity":"1"}
    _sty_idle   = {**_sty_active, "opacity":"0.4"}

    if ctx.triggered_id == "btn-stop":
        ss(stop=True)
        return False, True, _sty_idle         # Run odblokuj, Stop zablokuj

    if ctx.triggered_id == "btn-run" and not gs()["running"]:
        ss(running=True, stop=False, status="Uruchamiam...", progress="", result=None)
        t = threading.Thread(target=_worker, daemon=True, args=(
            (sym or BINANCE_SYMBOL).strip().upper(),
            tf or BINANCE_INTERVAL,
            mkt or "spot",
            frm, to,
            float(cap or INITIAL_CAPITAL),
            run_mode or "wfo",
            direction or DEFAULT_PARAMS.get("trade_direction", "both"),
            fee, slip, opt, live, score,
            bt_channel, bt_avg, bt_signal, bt_min_level,
            bt_reentry, bt_ema_filter, bt_htf_filter, bt_ema_len, bt_long_zone, bt_short_zone, bt_h4_long, bt_h4_short,
            grid_channel, grid_avg, grid_signal, grid_min_level,
            grid_reentry, grid_ema_filter, grid_htf_filter, grid_ema_len, grid_long_zone, grid_short_zone,
            grid_h4_long, grid_h4_short,
        ))
        t.start()
        return True, False, _sty_active        # Run zablokuj, Stop aktywuj

    return dash.no_update, dash.no_update, dash.no_update

# ─── Callback: polling przycisków (odblokuj Run gdy WFO skończone) ────────────
@app.callback(
    Output("run-status","children"),
    Output("run-progress","children"),
    Output("store-result","data"),
    Output("store-run-meta","data"),
    Input("poll","n_intervals"),
    State("store-result","data"),
    State("store-run-meta","data"),
    prevent_initial_call=True,
)
def poll_status(_, prev_result, prev_meta):
    g = gs()
    result = g["result"]
    result_version = int(g.get("result_version", 0))
    last_version = int((prev_meta or {}).get("result_version", -1))
    result_out = result if result is not None and result_version != last_version else dash.no_update
    meta = {
        "running": bool(g["running"]),
        "has_result": bool(result is not None or prev_result),
        "result_version": result_version,
    }
    return g["status"], g["progress"], result_out, meta


@app.callback(
    Output("poll","interval"),
    Input("store-run-meta","data"),
)
def tune_poll_interval(meta):
    if meta and meta.get("running"):
        return 1000
    return 15000


@app.callback(
    Output("btn-run","disabled", allow_duplicate=True),
    Output("btn-stop","disabled", allow_duplicate=True),
    Input("store-run-meta","data"),
    prevent_initial_call=True,
)
def poll_buttons(meta):
    running = bool(meta and meta.get("running"))
    return running, not running


def _empty_results_view() -> html.Div:
    return html.Div(
        "Skonfiguruj parametry i kliknij ▶ Uruchom",
        style={"color": C["muted"], "fontSize": "14px", "textAlign": "center", "marginTop": "80px"},
    )


def _result_metrics(result_data: dict | None) -> list:
    if not result_data:
        return []

    stats = result_data.get("stats", {})
    capital = result_data.get("capital", INITIAL_CAPITAL)
    symbol = result_data.get("symbol", "")
    tf = result_data.get("tf", "")

    def pc(value):
        return C["green"] if value > 0 else (C["red"] if value < 0 else C["text"])

    net_ret = stats.get("net_return_pct", 0)
    net_pnl = stats.get("net_profit_usd", 0)
    fee_tot = stats.get("fee_total_usd", 0)
    wr = stats.get("winrate_pct", 0)
    pf = stats.get("profit_factor", 0)
    dd = stats.get("max_drawdown_pct", 0)
    cagr = stats.get("cagr_pct", 0)
    exp_v = stats.get("expectancy_usd", 0)
    n_tr = stats.get("n_trades", 0)
    fin_cap = stats.get("final_capital", capital)
    rdd = stats.get("return_drawdown_ratio", float("nan"))
    sharpe = stats.get("sharpe_ratio", float("nan"))
    sortino = stats.get("sortino_ratio", float("nan"))
    rr = stats.get("risk_reward_ratio", float("nan"))
    consistency = stats.get("consistency_pct", float("nan"))
    stability = stats.get("stability_score", float("nan"))

    def metric_or_na(value, pattern="{:.2f}"):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/d"
        return pattern.format(value)

    def metric_color(value):
        if value is None:
            return C["text"]
        try:
            if np.isnan(value):
                return C["text"]
        except TypeError:
            return C["text"]
        return pc(value)

    return [
        mcrd("Net zwrot", f"{net_ret:+.2f}%", f"${net_pnl:+,.0f}", pc(net_ret)),
        mcrd("Kapitał końc.", f"${fin_cap:,.0f}", f"start: ${capital:,.0f}"),
        mcrd("Opłaty", f"${fee_tot:,.0f}", f"{stats.get('fee_total_pct', 0):.1f}%", C["amber"]),
        mcrd(
            "Winrate",
            f"{wr:.1f}%",
            f"PF: {pf:.2f}" if pf and not (isinstance(pf, float) and np.isnan(pf)) else "PF: n/d",
        ),
        mcrd("Max DD", f"{dd:.1f}%", None, C["red"]),
        mcrd("CAGR", f"{cagr:.1f}%", f"{symbol} {tf}", pc(cagr)),
        mcrd("Transakcji", str(n_tr), f"exp: ${exp_v:.1f}/tr"),
        mcrd("Return/DD", metric_or_na(rdd), "net return / max DD", metric_color(rdd)),
        mcrd("Sharpe", metric_or_na(sharpe), f"Sortino: {metric_or_na(sortino)}"),
        mcrd("R:R avg", metric_or_na(rr), "avg win / avg loss"),
        mcrd("Consistency", metric_or_na(consistency, "{:.0f}%"), f"stability: {metric_or_na(stability)}"),
    ]


def _build_chart_tab(selected_trade: dict | None, chart_filter_val: str | None) -> html.Div:
    chart_filt = chart_filter_val or "all"
    filter_opts = [
        {"label": "Wszystkie", "value": "all"},
        {"label": "Long", "value": "long"},
        {"label": "Short", "value": "short"},
        {"label": "Reverse", "value": "REVERSE"},
        {"label": "Force", "value": "FORCE"},
    ]
    return html.Div([
        html.Div([
            html.Span(
                "Filtr: ",
                style={"fontSize": "11px", "color": C["muted"], "marginRight": "8px", "lineHeight": "28px"},
            ),
            dcc.RadioItems(
                id="chart-filter-radio",
                options=filter_opts,
                value=chart_filt,
                inline=True,
                inputStyle={"marginRight": "3px", "accentColor": C["blue"]},
                labelStyle={"color": C["text"], "fontSize": "12px", "marginRight": "12px", "cursor": "pointer"},
            ),
        ], style={
            "display": "flex",
            "alignItems": "center",
            "marginBottom": "14px",
            "background": C["surf2"],
            "borderRadius": "16px",
            "padding": "10px 14px",
            "border": f"1px solid {C['border']}",
        }),
        html.Div([
            html.Div([
                html.Div([
                    html.Div("price map", className="panel-eyebrow"),
                    html.H3("Chart + trade markers"),
                ]),
                html.Div([
                    "Charting by ",
                    html.A("TradingView", href="https://www.tradingview.com/", target="_blank", rel="noreferrer"),
                    " using Lightweight Charts",
                ], className="tv-attribution"),
            ], className="panel-head"),
            html.Div(id="tv-chart", className="tv-chart"),
            html.Div(
                "Górny panel pokazuje cenę, linię EMA filtra i markery trade'ów, a dolny przebiegi WT1 i WT2 "
                "z poziomem zera oraz strefami sygnałowymi.",
                className="chart-caption",
            ),
        ], className="result-panel"),
        html.Div(_trade_detail_panel(selected_trade), className="detail-panel-wrap"),
    ])

# ─── Callback: polling (status + wyniki) ──────────────────────────────────────
@app.callback(
    Output("store-chart-payload","data"),
    Output("metrics-row","children"),
    Output("tab-content","children"),
    Input("tabs","value"),
    Input("store-result","data"),
    Input("store-chart-filter","data"),
    Input("store-selected-trade","data"),
    prevent_initial_call=True,
)
def render_results(tab, result_data, chart_filter_val, selected_trade_val):
    r = result_data
    if not r:
        return None, [], _empty_results_view()

    stats = r.get("stats", {})
    capital = r.get("capital", INITIAL_CAPITAL)
    symbol = r.get("symbol", "")
    tf = r.get("tf", "")
    metrics = _result_metrics(r)

    try:
        if tab == "equity":
            equity_df = pd.DataFrame(r.get("equity", []))
            return dash.no_update, metrics, dcc.Graph(
                figure=fig_eq(equity_df, capital),
                style={"height": "480px"},
                config={"displayModeBar": False},
            )

        if tab == "chart":
            trades_df = pd.DataFrame(r.get("trades", []))
            windows_df = pd.DataFrame(r.get("windows_df", []))
            chart_payload = lightweight_chart_payload(
                symbol,
                tf,
                trades_df,
                windows_df=windows_df,
                selected_trade=selected_trade_val,
                filter_mode=chart_filter_val or "all",
                params=r.get("best_params") or r.get("params_used") or DEFAULT_PARAMS,
            )
            return chart_payload, metrics, _build_chart_tab(selected_trade_val, chart_filter_val)

        if tab == "trades":
            trades_df = pd.DataFrame(r.get("trades", []))
            if trades_df.empty:
                return dash.no_update, metrics, html.Div("Brak transakcji", style={"color": C["muted"]})

            disp = _trade_table_frame(trades_df)
            mode_label = "WFO live windows" if str(r.get("mode", "")).lower() == "wfo" else "Backtest"

            return dash.no_update, metrics, html.Div([
                html.Div([
                    html.Div([
                        html.Div(f"{len(disp)} transakcji", style={"fontSize": "12px", "color": C["muted"]}),
                        html.Div(
                            f"Źródło: {mode_label}. Numer # jest lokalny dla bieżącego wyniku i resetuje się od 1 przy nowym uruchomieniu.",
                            style={"fontSize": "11px", "color": C["muted"], "marginTop": "4px"},
                        ),
                    ]),
                    html.Button(
                        "Eksport CSV",
                        id="btn-export-trades",
                        n_clicks=0,
                        style={
                            "background": C["surf2"],
                            "color": C["text"],
                            "border": f"1px solid {C['border']}",
                            "borderRadius": "12px",
                            "padding": "10px 14px",
                            "fontSize": "12px",
                            "fontWeight": "600",
                            "cursor": "pointer",
                            "whiteSpace": "nowrap",
                        },
                    ),
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "16px", "marginBottom": "10px"}),
                dash_table.DataTable(
                    id="trades-table",
                    data=disp.to_dict("records"),
                    columns=[
                        {"name": "#" if c == "trade_no" else c, "id": c}
                        for c in disp.columns
                    ],
                    page_size=25,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "background": C["surface"],
                        "color": C["text"],
                        "border": f"1px solid {C['border']}",
                        "fontSize": "12px",
                        "padding": "6px 10px",
                        "textAlign": "left",
                    },
                    style_header={
                        "background": C["surf2"],
                        "color": C["muted"],
                        "fontWeight": "600",
                        "fontSize": "10px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.05em",
                        "border": f"1px solid {C['border']}",
                    },
                    style_data_conditional=[
                        {"if": {"column_id": "trade_no"}, "fontWeight": "700", "color": C["amber"]},
                        {"if": {"filter_query": '{side} = "long"', "column_id": "side"}, "color": C["green"]},
                        {"if": {"filter_query": '{side} = "short"', "column_id": "side"}, "color": C["red"]},
                        {"if": {"filter_query": '{reason} contains "REVERSE"', "column_id": "reason"}, "color": C["amber"]},
                        {"if": {"filter_query": '{reason} contains "FORCE"', "column_id": "reason"}, "color": C["muted"]},
                    ],
                    sort_action="native",
                    filter_action="native",
                ),
            ])

        if tab == "fees":
            fee_df = pd.DataFrame(r.get("fee_df", []))
            fee_tot = stats.get("fee_total_usd", 0)
            return dash.no_update, metrics, html.Div([
                html.Div([
                    mcrd("Łączne opłaty", f"${fee_tot:,.2f}", None, C["amber"]),
                    mcrd("% kapitału", f"{stats.get('fee_total_pct', 0):.2f}%", None, C["amber"]),
                    mcrd("Slippage", f"${stats.get('slippage_total_usd', 0):,.2f}"),
                ], style={"display": "flex", "gap": "10px", "marginBottom": "14px"}),
                dcc.Graph(figure=fig_fee(fee_df), style={"height": "300px"}, config={"displayModeBar": False}),
                dash_table.DataTable(
                    data=fee_df.to_dict("records") if not fee_df.empty else [],
                    columns=[{"name": c, "id": c} for c in fee_df.columns] if not fee_df.empty else [],
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "background": C["surface"],
                        "color": C["text"],
                        "border": f"1px solid {C['border']}",
                        "fontSize": "12px",
                        "padding": "6px 10px",
                    },
                    style_header={
                        "background": C["surf2"],
                        "color": C["muted"],
                        "fontWeight": "600",
                        "fontSize": "10px",
                        "textTransform": "uppercase",
                        "border": f"1px solid {C['border']}",
                    },
                    sort_action="native",
                ) if not fee_df.empty else html.Div(),
            ])

        if tab == "wfo-win":
            windows_df = pd.DataFrame(r.get("windows_df", []))
            if windows_df.empty:
                return dash.no_update, metrics, html.Div("Uruchom tryb WFO.", style={"color": C["muted"]})

            prof = (windows_df["live_return_pct"] > 0).sum()
            nw = len(windows_df)
            num_cols = windows_df.select_dtypes(include="number").columns
            wd_disp = windows_df.copy()
            wd_disp[num_cols] = wd_disp[num_cols].round(4)
            return dash.no_update, metrics, html.Div([
                html.Div([
                    mcrd("Okien live", str(nw)),
                    mcrd("Zyskownych", f"{prof}/{nw}", f"{prof / nw * 100:.0f}%", C["green"]),
                    mcrd("Śr. zwrot", f"{windows_df['live_return_pct'].mean():.3f}%"),
                    mcrd("Mediana", f"{windows_df['live_return_pct'].median():.3f}%"),
                ], style={"display": "flex", "gap": "10px", "marginBottom": "14px"}),
                dcc.Graph(figure=fig_wfo(windows_df), style={"height": "260px"}, config={"displayModeBar": False}),
                html.Div(style={"height": "12px"}),
                dash_table.DataTable(
                    data=wd_disp.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in wd_disp.columns],
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "background": C["surface"],
                        "color": C["text"],
                        "border": f"1px solid {C['border']}",
                        "fontSize": "12px",
                        "padding": "6px 10px",
                    },
                    style_header={
                        "background": C["surf2"],
                        "color": C["muted"],
                        "fontWeight": "600",
                        "fontSize": "10px",
                        "textTransform": "uppercase",
                        "border": f"1px solid {C['border']}",
                    },
                    style_data_conditional=[
                        {"if": {"filter_query": "{live_return_pct} > 0", "column_id": "live_return_pct"}, "color": C["green"]},
                        {"if": {"filter_query": "{live_return_pct} < 0", "column_id": "live_return_pct"}, "color": C["red"]},
                    ],
                    sort_action="native",
                ),
            ])

        if tab == "wfo-par":
            windows_df = pd.DataFrame(r.get("windows_df", []))
            if str(r.get("mode", "")).lower() == "backtest":
                return dash.no_update, metrics, _params_summary_card(
                    "Parametry backtestu manualnego",
                    "To jest dokładny zestaw ustawień użyty do ostatniego zwykłego backtestu.",
                    r.get("params_used", {}),
                )

            if windows_df.empty:
                return dash.no_update, metrics, html.Div("Uruchom tryb WFO.", style={"color": C["muted"]})

            best_params = r.get("best_params", {})
            return dash.no_update, metrics, html.Div([
                dcc.Graph(
                    figure=fig_pdist(windows_df),
                    style={"height": "920px"},
                    config={"displayModeBar": False},
                ),
                _params_summary_card(
                    "Stabilne parametry po WFO",
                    "To są parametry wyliczone z ostatnich aktywnych okien i zapisywane jako best params.",
                    best_params,
                ),
            ])

        if tab == "brkdwn":
            side_df = pd.DataFrame(r.get("side_bk", []))
            cross_df = pd.DataFrame(r.get("cross_bk", []))
            yr_df = pd.DataFrame(r.get("yr_bk", []))
            q_df = pd.DataFrame(r.get("q_bk", []))

            def mini(df, title):
                if df is None or df.empty:
                    return html.Div(f"{title}: brak", style={"color": C["muted"], "fontSize": "12px"})
                num_cols = df.select_dtypes(include="number").columns
                df2 = df.copy()
                df2[num_cols] = df2[num_cols].round(3)
                return html.Div([
                    html.Div(title, style={"fontSize": "12px", "color": C["text"], "marginBottom": "6px", "fontWeight": "500"}),
                    dash_table.DataTable(
                        data=df2.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in df2.columns],
                        style_cell={
                            "background": C["surface"],
                            "color": C["text"],
                            "border": f"1px solid {C['border']}",
                            "fontSize": "12px",
                            "padding": "5px 10px",
                        },
                        style_header={
                            "background": C["surf2"],
                            "color": C["muted"],
                            "fontWeight": "600",
                            "fontSize": "10px",
                            "textTransform": "uppercase",
                            "border": f"1px solid {C['border']}",
                        },
                    ),
                ], style={**card_s, "marginBottom": "14px"})

            ext = []
            for key, label in [
                ("expectancy_usd", "Expectancy/trade USD"),
                ("return_drawdown_ratio", "Return/DD"),
                ("sharpe_ratio", "Sharpe"),
                ("sortino_ratio", "Sortino"),
                ("risk_reward_ratio", "R:R avg"),
                ("consistency_pct", "Consistency %"),
                ("stability_score", "Stability"),
                ("avg_winner_usd", "Śr. winner USD"),
                ("avg_loser_usd", "Śr. loser USD"),
                ("median_ret_pct", "Median ret %"),
                ("longest_losing_streak", "Seria strat"),
                ("exposure_pct", "Czas w pozycji %"),
            ]:
                value = stats.get(key)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    ext.append(mcrd(label, f"{value:.2f}" if isinstance(value, float) else str(value)))

            return dash.no_update, metrics, html.Div([
                html.Div(ext, style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "14px"}),
                mini(side_df, "Long vs Short"),
                mini(cross_df, "Bullish vs Bearish H1 cross"),
                mini(yr_df, "Breakdown roczny"),
                mini(q_df, "Breakdown kwartalny"),
            ])
    except Exception as tab_err:
        import traceback as _tb
        print(f"[POLL TAB ERROR] tab={tab}: {_tb.format_exc()}", flush=True)
        return dash.no_update, metrics, html.Div([
            html.Div(f"⚠ Błąd w zakładce [{tab}]:", style={"color": C["red"], "fontWeight": "600", "marginBottom": "6px"}),
            html.Pre(
                str(tab_err),
                style={
                    "color": C["amber"],
                    "fontSize": "11px",
                    "whiteSpace": "pre-wrap",
                    "background": C["surf2"],
                    "padding": "10px",
                    "borderRadius": "6px",
                },
            ),
        ], style={"padding": "20px"})
    # ── nie czyść store gdy brak nowego wyniku ────────────────────────────────
    # Jeśli _state["result"] jest None, nie nadpisuj store (zostawiamy poprzednie dane)

    # Pobierz dane do renderowania: nowy wynik lub poprzedni ze store

    if not r:
        empty = html.Div("Skonfiguruj parametry i kliknij ▶ Uruchom",
                         style={"color":C["muted"],"fontSize":"14px",
                                "textAlign":"center","marginTop":"80px"})
        return None, [], empty

    stats       = r.get("stats", {})
    capital     = r.get("capital", INITIAL_CAPITAL)
    symbol      = r.get("symbol","")
    tf          = r.get("tf","")
    trades_df   = pd.DataFrame(r.get("trades",   []))
    equity_df   = pd.DataFrame(r.get("equity",   []))
    fee_df      = pd.DataFrame(r.get("fee_df",   []))
    windows_df  = pd.DataFrame(r.get("windows_df",[]))
    side_df     = pd.DataFrame(r.get("side_bk",  []))
    yr_df       = pd.DataFrame(r.get("yr_bk",    []))
    q_df        = pd.DataFrame(r.get("q_bk",     []))

    def pc(v): return C["green"] if v>0 else (C["red"] if v<0 else C["text"])

    net_ret  = stats.get("net_return_pct",0)
    net_pnl  = stats.get("net_profit_usd",0)
    fee_tot  = stats.get("fee_total_usd",0)
    wr       = stats.get("winrate_pct",0)
    pf       = stats.get("profit_factor",0)
    dd       = stats.get("max_drawdown_pct",0)
    cagr     = stats.get("cagr_pct",0)
    exp_v    = stats.get("expectancy_usd",0)
    n_tr     = stats.get("n_trades",0)
    fin_cap  = stats.get("final_capital",capital)

    metrics = [
        mcrd("Net zwrot",    f"{net_ret:+.2f}%",  f"${net_pnl:+,.0f}", pc(net_ret)),
        mcrd("Kapitał końc.",f"${fin_cap:,.0f}",   f"start: ${capital:,.0f}"),
        mcrd("Opłaty",       f"${fee_tot:,.0f}",   f"{stats.get('fee_total_pct',0):.1f}%", C["amber"]),
        mcrd("Winrate",      f"{wr:.1f}%",          f"PF: {pf:.2f}" if pf and not (isinstance(pf,float) and np.isnan(pf)) else "PF: n/d"),
        mcrd("Max DD",       f"{dd:.1f}%",          None, C["red"]),
        mcrd("CAGR",         f"{cagr:.1f}%",        f"{symbol} {tf}", pc(cagr)),
        mcrd("Transakcji",   str(n_tr),             f"exp: ${exp_v:.1f}/tr"),
    ]

    # ── zakładki ──────────────────────────────────────────────────────────────
    chart_payload = dash.no_update
    content = html.Div()
    try:
        if tab == "equity":
            content = dcc.Graph(figure=fig_eq(equity_df, capital),
                                style={"height":"480px"},config={"displayModeBar":False})

        elif tab == "chart":
            sel_trade  = selected_trade_val
            chart_filt = chart_filter_val or "all"
            chart_payload = lightweight_chart_payload(
                symbol,
                tf,
                trades_df,
                windows_df=windows_df,
                selected_trade=sel_trade,
                filter_mode=chart_filt,
                params=r.get("best_params") or r.get("params_used") or DEFAULT_PARAMS,
            )
            _filter_opts = [
                {"label":"Wszystkie","value":"all"},
                {"label":"Long",     "value":"long"},
                {"label":"Short",    "value":"short"},
                {"label":"Reverse",  "value":"REVERSE"},
                {"label":"Force",    "value":"FORCE"},
            ]
            content = html.Div([
                html.Div([
                    html.Span("Filtr: ", style={"fontSize":"11px","color":C["muted"],
                              "marginRight":"8px","lineHeight":"28px"}),
                    dcc.RadioItems(
                        id="chart-filter-radio",
                        options=_filter_opts,
                        value=chart_filt,
                        inline=True,
                        inputStyle={"marginRight":"3px","accentColor":C["blue"]},
                        labelStyle={"color":C["text"],"fontSize":"12px","marginRight":"12px",
                                    "cursor":"pointer"},
                    ),
                ], style={"display":"flex","alignItems":"center","marginBottom":"14px",
                          "background":C["surf2"],"borderRadius":"16px","padding":"10px 14px",
                          "border":f"1px solid {C['border']}"}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("price map", className="panel-eyebrow"),
                            html.H3("Chart + trade markers"),
                        ]),
                        html.Div([
                            "Charting by ",
                            html.A("TradingView", href="https://www.tradingview.com/", target="_blank", rel="noreferrer"),
                            " using Lightweight Charts",
                        ], className="tv-attribution"),
                    ], className="panel-head"),
                    html.Div(id="tv-chart", className="tv-chart"),
                    html.Div(
                        "Cena, linia EMA filtra, markery transakcji oraz panel WaveTrend sa renderowane przez "
                        "ten sam otwarty silnik Lightweight Charts od TradingView.",
                        className="chart-caption",
                    ),
                ], className="result-panel"),
                html.Div(
                    _trade_detail_panel(sel_trade),
                    className="detail-panel-wrap",
                ),
            ])

        elif tab == "trades":
            if trades_df.empty:
                content = html.Div("Brak transakcji",style={"color":C["muted"]})
            else:
                cols = [c for c in ["side","entry_time","exit_time","entry_price",
                                     "exit_price","gross_ret","fee_ret","net_ret",
                                     "pnl","fee_usd","reason"] if c in trades_df.columns]
                disp = trades_df[cols].copy()
                for c in ["gross_ret","fee_ret","net_ret"]:
                    if c in disp.columns:
                        disp[c] = (disp[c]*100).round(3).astype(str)+"%"
                for c in ["pnl","fee_usd","entry_price","exit_price"]:
                    if c in disp.columns: disp[c] = disp[c].round(2)
                for c in ["entry_time","exit_time"]:
                    if c in disp.columns:
                        disp[c] = pd.to_datetime(disp[c]).dt.strftime("%Y-%m-%d %H:%M")
                content = html.Div([
                    html.Div(f"{len(disp)} transakcji",
                             style={"fontSize":"12px","color":C["muted"],"marginBottom":"8px"}),
                    dash_table.DataTable(
                        id="trades-table",
                        data=disp.to_dict("records"),
                        columns=[{"name":c,"id":c} for c in disp.columns],
                        page_size=25,
                        style_table={"overflowX":"auto"},
                        style_cell={"background":C["surface"],"color":C["text"],
                                    "border":f"1px solid {C['border']}",
                                    "fontSize":"12px","padding":"6px 10px","textAlign":"left"},
                        style_header={"background":C["surf2"],"color":C["muted"],
                                      "fontWeight":"600","fontSize":"10px",
                                      "textTransform":"uppercase","letterSpacing":"0.05em",
                                      "border":f"1px solid {C['border']}"},
                        style_data_conditional=[
                            {"if":{"filter_query":'{side} = "long"', "column_id":"side"},"color":C["green"]},
                            {"if":{"filter_query":'{side} = "short"',"column_id":"side"},"color":C["red"]},
                            {"if":{"filter_query":'{reason} contains "REVERSE"', "column_id":"reason"},"color":C["amber"]},
                            {"if":{"filter_query":'{reason} contains "FORCE"', "column_id":"reason"},"color":C["muted"]},
                        ],
                        sort_action="native", filter_action="native",
                    ),
                ])

        elif tab == "fees":
            content = html.Div([
                html.Div([
                    mcrd("Łączne opłaty",f"${fee_tot:,.2f}", None, C["amber"]),
                    mcrd("% kapitału",   f"{stats.get('fee_total_pct',0):.2f}%", None, C["amber"]),
                    mcrd("Slippage",     f"${stats.get('slippage_total_usd',0):,.2f}"),
                ],style={"display":"flex","gap":"10px","marginBottom":"14px"}),
                dcc.Graph(figure=fig_fee(fee_df),style={"height":"300px"},
                          config={"displayModeBar":False}),
                dash_table.DataTable(
                    data=fee_df.to_dict("records") if not fee_df.empty else [],
                    columns=[{"name":c,"id":c} for c in fee_df.columns] if not fee_df.empty else [],
                    page_size=20,style_table={"overflowX":"auto"},
                    style_cell={"background":C["surface"],"color":C["text"],
                                "border":f"1px solid {C['border']}","fontSize":"12px","padding":"6px 10px"},
                    style_header={"background":C["surf2"],"color":C["muted"],
                                  "fontWeight":"600","fontSize":"10px","textTransform":"uppercase",
                                  "border":f"1px solid {C['border']}"},
                    sort_action="native",
                ) if not fee_df.empty else html.Div(),
            ])

        elif tab == "wfo-win":
            if windows_df.empty:
                content = html.Div("Uruchom tryb WFO.",style={"color":C["muted"]})
            else:
                prof = (windows_df["live_return_pct"]>0).sum()
                nw   = len(windows_df)
                num_cols = windows_df.select_dtypes(include="number").columns
                wd_disp  = windows_df.copy()
                wd_disp[num_cols] = wd_disp[num_cols].round(4)
                content = html.Div([
                    html.Div([
                        mcrd("Okien live",str(nw)),
                        mcrd("Zyskownych",f"{prof}/{nw}",f"{prof/nw*100:.0f}%",C["green"]),
                        mcrd("Śr. zwrot",f"{windows_df['live_return_pct'].mean():.3f}%"),
                        mcrd("Mediana",  f"{windows_df['live_return_pct'].median():.3f}%"),
                    ],style={"display":"flex","gap":"10px","marginBottom":"14px"}),
                    dcc.Graph(figure=fig_wfo(windows_df),style={"height":"260px"},
                              config={"displayModeBar":False}),
                    html.Div(style={"height":"12px"}),
                    dash_table.DataTable(
                        data=wd_disp.to_dict("records"),
                        columns=[{"name":c,"id":c} for c in wd_disp.columns],
                        page_size=20,style_table={"overflowX":"auto"},
                        style_cell={"background":C["surface"],"color":C["text"],
                                    "border":f"1px solid {C['border']}","fontSize":"12px","padding":"6px 10px"},
                        style_header={"background":C["surf2"],"color":C["muted"],
                                      "fontWeight":"600","fontSize":"10px","textTransform":"uppercase",
                                      "border":f"1px solid {C['border']}"},
                        style_data_conditional=[
                            {"if":{"filter_query":"{live_return_pct} > 0","column_id":"live_return_pct"},"color":C["green"]},
                            {"if":{"filter_query":"{live_return_pct} < 0","column_id":"live_return_pct"},"color":C["red"]},
                        ],
                        sort_action="native",
                    ),
                ])

        elif tab == "wfo-par":
            content = html.Div(
                dcc.Graph(figure=fig_pdist(windows_df),style={"height":"640px"},
                          config={"displayModeBar":False})
                if not windows_df.empty
                else html.Div("Uruchom tryb WFO.",style={"color":C["muted"]})
            )

        elif tab == "brkdwn":
            def mini(df, title):
                if df is None or df.empty:
                    return html.Div(f"{title}: brak",style={"color":C["muted"],"fontSize":"12px"})
                num_c = df.select_dtypes(include="number").columns
                df2   = df.copy(); df2[num_c] = df2[num_c].round(3)
                return html.Div([
                    html.Div(title,style={"fontSize":"12px","color":C["text"],
                                         "marginBottom":"6px","fontWeight":"500"}),
                    dash_table.DataTable(
                        data=df2.to_dict("records"),
                        columns=[{"name":c,"id":c} for c in df2.columns],
                        style_cell={"background":C["surface"],"color":C["text"],
                                    "border":f"1px solid {C['border']}","fontSize":"12px","padding":"5px 10px"},
                        style_header={"background":C["surf2"],"color":C["muted"],
                                      "fontWeight":"600","fontSize":"10px","textTransform":"uppercase",
                                      "border":f"1px solid {C['border']}"},
                    ),
                ],style={**card_s,"marginBottom":"14px"})

            ext = []
            for k,lbl_t in [("expectancy_usd","Expectancy/trade USD"),
                            ("avg_winner_usd","Śr. winner USD"),
                            ("avg_loser_usd","Śr. loser USD"),
                            ("median_ret_pct","Median ret %"),
                            ("longest_losing_streak","Seria strat"),
                            ("exposure_pct","Czas w pozycji %")]:
                v = stats.get(k)
                if v is not None and not (isinstance(v,float) and np.isnan(v)):
                    ext.append(mcrd(lbl_t,f"{v:.2f}" if isinstance(v,float) else str(v)))

            content = html.Div([
                html.Div(ext,style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"14px"}),
                mini(side_df,"Long vs Short"),
                mini(yr_df,  "Breakdown roczny"),
                mini(q_df,   "Breakdown kwartalny"),
            ])

    except Exception as _tab_err:
        import traceback as _tb
        print(f"[POLL TAB ERROR] tab={tab}: {_tb.format_exc()}", flush=True)
        content = html.Div([
            html.Div(f"⚠ Błąd w zakładce [{tab}]:",
                     style={"color":C["red"],"fontWeight":"600","marginBottom":"6px"}),
            html.Pre(str(_tab_err),
                     style={"color":C["amber"],"fontSize":"11px","whiteSpace":"pre-wrap",
                            "background":C["surf2"],"padding":"10px","borderRadius":"6px"}),
        ], style={"padding":"20px"})

    return chart_payload, metrics, content

# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8067)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    print(f"\n{'='*50}\n  Bee7 WaveTrend Dashboard  →  http://{a.host}:{a.port}\n{'='*50}\n")
    app.run(host=a.host, port=a.port, debug=a.debug)


