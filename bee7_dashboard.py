"""
bee7_dashboard.py  -  Bee7 WaveTrend Dashboard  http://IP:8067
"""
from __future__ import annotations
import argparse, datetime as _dt, io, json, os, threading
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
    WT_LONG_CLOSE_MIN_LEVEL_GRID, WT_LONG_CLOSE_MIN_LEVEL_OPTIONS,
    WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID, WT_SHORT_ENTRY_MIN_BELOW_ZERO_OPTIONS,
    WT_SHORT_CLOSE_MAX_LEVEL_GRID, WT_SHORT_CLOSE_MAX_LEVEL_OPTIONS,
    WT_H4_LONG_FILTER_MAX_GRID, WT_H4_LONG_FILTER_MAX_OPTIONS,
    WT_H4_LONG_CLOSE_MIN_GRID, WT_H4_LONG_CLOSE_MIN_OPTIONS,
    WT_H4_SHORT_FILTER_MIN_GRID, WT_H4_SHORT_FILTER_MIN_OPTIONS,
    WT_H4_SHORT_CLOSE_MAX_GRID, WT_H4_SHORT_CLOSE_MAX_OPTIONS,
    WT_LONG_EMERGENCY_SL_CAPITAL_PCT_GRID, WT_LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS,
)
from bee7_data     import (
    htf_wt1_column,
    htf_wt2_column,
    load_klines,
    prepare_indicators,
    wt_columns,
)
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
_SAVED_RUNS_DIR = _APP_DIR / "results" / "saved_runs"
CHART_VIEW_OPTIONS = [
    {"label": "Standard", "value": "standard"},
    {"label": "Diagnostyka sygnałów", "value": "diagnostic"},
    {"label": "Tylko transakcje", "value": "trades"},
]
CHART_VIEW_VALUES = {item["value"] for item in CHART_VIEW_OPTIONS}

def gs():
    with _lock: return dict(_state)

def ss(**kw):
    with _lock:
        if "result" in kw:
            _state["result_version"] += 1
        _state.update(kw)

def _fmt_duration(seconds: float | int | None) -> str:
    try:
        seconds = int(max(0, float(seconds)))
    except Exception:
        return "n/d"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"

def _elapsed_eta(start_ts: float, done_units: int, total_units: int) -> tuple[str, str]:
    elapsed = max(0.0, _time.time() - float(start_ts or _time.time()))
    if done_units <= 0 or total_units <= 0:
        return _fmt_duration(elapsed), "liczę..."
    per_unit = elapsed / max(done_units, 1)
    remaining = max(0, total_units - done_units) * per_unit
    return _fmt_duration(elapsed), _fmt_duration(remaining)

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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _safe_slug(value, fallback="run") -> str:
    text = str(value or fallback).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def _saved_result_filename(result_data: dict) -> str:
    stats = result_data.get("stats", {}) if isinstance(result_data, dict) else {}
    mode = _safe_slug(str(result_data.get("mode", "run")).lower() if isinstance(result_data, dict) else "run")
    symbol = _safe_slug(str(result_data.get("symbol", "asset")).upper() if isinstance(result_data, dict) else "asset")
    tf = _safe_slug(str(result_data.get("tf", "tf")).lower() if isinstance(result_data, dict) else "tf")
    ret = float(stats.get("net_return_pct", 0.0) or 0.0)
    ret_token = ("p" if ret >= 0 else "m") + f"{abs(ret):.2f}".replace(".", "p")
    n_trades = int(float(stats.get("n_trades", len(result_data.get("trades", []))) or 0))
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"bee7_{stamp}_{mode}_{symbol}_{tf}_{ret_token}pct_tr{n_trades}.json"


def _saved_result_path(filename: str) -> Path:
    safe_name = Path(str(filename or "")).name
    if not safe_name.endswith(".json"):
        raise ValueError("Nieprawidłowy plik wyniku.")
    path = (_SAVED_RUNS_DIR / safe_name).resolve()
    root = _SAVED_RUNS_DIR.resolve()
    if root not in path.parents:
        raise ValueError("Plik musi znajdować się w katalogu zapisanych wyników.")
    return path


def _save_result_file(result_data: dict) -> str:
    if not result_data:
        raise ValueError("Brak wyniku do zapisu.")
    _SAVED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _saved_result_filename(result_data)
    payload = {
        "schema": "bee7_dashboard_result_v1",
        "saved_at": pd.Timestamp.utcnow().isoformat(),
        "meta": {
            "mode": result_data.get("mode"),
            "symbol": result_data.get("symbol"),
            "tf": result_data.get("tf"),
            "capital": result_data.get("capital"),
            "stats": _json_safe(result_data.get("stats", {})),
        },
        "result": _json_safe(result_data),
    }
    path = _saved_result_path(filename)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)
    return filename


def _load_result_file(filename: str) -> dict:
    path = _saved_result_path(filename)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    if not isinstance(result, dict):
        raise ValueError("Plik nie zawiera wyniku dashboardu.")
    result["_loaded_from_file"] = path.name
    return result


def _saved_result_options(limit: int = 100) -> list[dict[str, str]]:
    if not _SAVED_RUNS_DIR.exists():
        return []
    files = sorted(
        _SAVED_RUNS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    options = []
    for path in files:
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Warsaw")
        label = f"{mtime.strftime('%Y-%m-%d %H:%M')} | {path.stem}"
        options.append({"label": label, "value": path.name})
    return options


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
    # BEE7 keeps both directions enabled; long and short use independent levels.
    return "both", True, True


def _pct_value(value, fallback: float = 0.0) -> float:
    if value in (None, ""):
        return float(fallback)
    pct = float(value)
    return pct / 100.0 if pct > 1.0 else pct


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
    short_close_level,
    h4_long_filter,
    h4_short_filter,
    h4_short_close_level,
    long_close_level,
    h4_long_close_level,
    long_tp1_pct,
    long_sl_pct,
    fee_rate: float,
    slippage_bps: float,
) -> dict:
    params = dict(DEFAULT_PARAMS)
    trade_direction, allow_longs, allow_shorts = _direction_flags(direction)
    emergency_sl_pct = _pct_value(
        long_sl_pct,
        float(DEFAULT_PARAMS.get("wt_long_emergency_sl_capital_pct", 0.0)),
    )
    long_entry_level = float(
        long_zone if long_zone not in (None, "") else DEFAULT_PARAMS["wt_long_entry_max_above_zero"]
    )
    long_close_h1 = float(
        long_close_level
        if long_close_level not in (None, "")
        else DEFAULT_PARAMS["wt_long_close_min_level"]
    )
    h4_long_open = float(
        h4_long_filter if h4_long_filter not in (None, "") else DEFAULT_PARAMS["wt_h4_long_filter_max"]
    )
    h4_long_close = float(
        h4_long_close_level
        if h4_long_close_level not in (None, "")
        else DEFAULT_PARAMS["wt_h4_long_close_min"]
    )
    short_entry_level = float(
        short_zone if short_zone not in (None, "") else DEFAULT_PARAMS["wt_short_entry_min_below_zero"]
    )
    short_close_h1 = float(
        short_close_level
        if short_close_level not in (None, "")
        else DEFAULT_PARAMS["wt_short_close_max_level"]
    )
    h4_short_open = float(
        h4_short_filter if h4_short_filter not in (None, "") else DEFAULT_PARAMS["wt_h4_short_filter_min"]
    )
    h4_short_close = float(
        h4_short_close_level
        if h4_short_close_level not in (None, "")
        else DEFAULT_PARAMS["wt_h4_short_close_max"]
    )
    params.update(
        {
            "trade_direction": trade_direction,
            "allow_longs": allow_longs,
            "allow_shorts": allow_shorts,
            "short_trading_enabled": allow_shorts,
            "wt_channel_len": int(channel_len if channel_len not in (None, "") else DEFAULT_PARAMS["wt_channel_len"]),
            "wt_avg_len": int(avg_len if avg_len not in (None, "") else DEFAULT_PARAMS["wt_avg_len"]),
            "wt_signal_len": int(signal_len if signal_len not in (None, "") else DEFAULT_PARAMS["wt_signal_len"]),
            "wt_min_signal_level": 0.0,
            "wt_long_entry_window_bars": int(
                reentry_window
                if reentry_window not in (None, "")
                else DEFAULT_PARAMS["wt_long_entry_window_bars"]
            ),
            "wt_short_entry_window_bars": int(
                reentry_window
                if reentry_window not in (None, "")
                else DEFAULT_PARAMS["wt_long_entry_window_bars"]
            ),
            "wt_long_require_ema20_reclaim": False,
            "wt_short_require_ema20_reject": False,
            "wt_long_require_htf_trend": False,
            "wt_short_require_htf_trend": False,
            "wt_ema_filter_len": int(DEFAULT_PARAMS["wt_ema_filter_len"]),
            "wt_long_entry_max_above_zero": long_entry_level,
            "wt_long_close_min_level": long_close_h1,
            "wt_long_exit_min_level": long_close_h1,
            "wt_short_entry_min_below_zero": short_entry_level,
            "wt_short_close_max_level": short_close_h1,
            "wt_short_exit_max_level": short_close_h1,
            "wt_h4_long_filter_max": h4_long_open,
            "wt_h4_long_close_min": h4_long_close,
            "wt_h4_short_filter_min": h4_short_open,
            "wt_h4_short_close_max": h4_short_close,
            "wt_long_tp1_enabled": True,
            "wt_long_tp1_pct": float(
                (long_tp1_pct if long_tp1_pct not in (None, "") else DEFAULT_PARAMS.get("wt_long_tp1_pct", 0.01) * 100.0)
            ) / 100.0,
            "wt_long_tp1_fraction": float(DEFAULT_PARAMS.get("wt_long_tp1_fraction", 1.0 / 3.0)),
            "wt_long_tp2_enabled": True,
            "wt_long_tp2_pct": float(DEFAULT_PARAMS.get("wt_long_tp2_pct", 0.02)),
            "wt_long_tp2_fraction": float(DEFAULT_PARAMS.get("wt_long_tp2_fraction", 1.0 / 3.0)),
            "wt_long_emergency_sl_enabled": emergency_sl_pct > 0.0,
            "wt_long_emergency_sl_capital_pct": emergency_sl_pct,
            "wt_short_tp1_enabled": True,
            "wt_short_tp1_pct": float(
                (long_tp1_pct if long_tp1_pct not in (None, "") else DEFAULT_PARAMS.get("wt_short_tp1_pct", 0.01) * 100.0)
            ) / 100.0,
            "wt_short_tp1_fraction": float(DEFAULT_PARAMS.get("wt_short_tp1_fraction", 1.0 / 3.0)),
            "wt_short_tp1_breakeven_enabled": True,
            "wt_short_tp2_enabled": True,
            "wt_short_tp2_pct": float(DEFAULT_PARAMS.get("wt_short_tp2_pct", DEFAULT_PARAMS.get("wt_long_tp2_pct", 0.02))),
            "wt_short_tp2_fraction": float(DEFAULT_PARAMS.get("wt_short_tp2_fraction", 1.0 / 3.0)),
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
    short_close_level_grid,
    h4_long_filter_grid,
    h4_short_filter_grid,
    h4_short_close_grid,
    long_close_level_grid,
    h4_long_close_level_grid,
    long_sl_pct_grid,
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
        "wt_long_close_min_level": _clean_selected_values(
            long_close_level_grid,
            WT_LONG_CLOSE_MIN_LEVEL_GRID,
            float,
        ),
        "wt_short_entry_min_below_zero": _clean_selected_values(
            short_zone_grid,
            WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID,
            float,
        ),
        "wt_short_close_max_level": _clean_selected_values(
            short_close_level_grid,
            WT_SHORT_CLOSE_MAX_LEVEL_GRID,
            float,
        ),
        "wt_h4_long_filter_max": _clean_selected_values(
            h4_long_filter_grid,
            WT_H4_LONG_FILTER_MAX_GRID,
            float,
        ),
        "wt_h4_long_close_min": _clean_selected_values(
            h4_long_close_level_grid,
            WT_H4_LONG_CLOSE_MIN_GRID,
            float,
        ),
        "wt_h4_short_filter_min": _clean_selected_values(
            h4_short_filter_grid,
            WT_H4_SHORT_FILTER_MIN_GRID,
            float,
        ),
        "wt_h4_short_close_max": _clean_selected_values(
            h4_short_close_grid,
            WT_H4_SHORT_CLOSE_MAX_GRID,
            float,
        ),
        "wt_long_emergency_sl_capital_pct": _clean_selected_values(
            long_sl_pct_grid,
            WT_LONG_EMERGENCY_SL_CAPITAL_PCT_GRID,
            float,
        ),
    }


def _grid_combo_count(grid_overrides: dict) -> int:
    short_keys = {
        "wt_short_entry_min_below_zero",
        "wt_short_close_max_level",
        "wt_h4_short_filter_min",
        "wt_h4_short_close_max",
    }
    total = 1
    short_profile_count = 1
    for key, values in grid_overrides.items():
        if key in short_keys:
            short_profile_count = max(short_profile_count, len(values or []))
            continue
        total *= max(1, len(values))
    total *= short_profile_count
    return total


PARAM_SUMMARY_ORDER = [
    "trade_direction",
    "wt_long_entry_window_bars",
    "wt_long_entry_max_above_zero",
    "wt_long_close_min_level",
    "wt_short_entry_min_below_zero",
    "wt_short_close_max_level",
    "wt_h4_long_filter_max",
    "wt_h4_long_close_min",
    "wt_h4_short_filter_min",
    "wt_h4_short_close_max",
    "wt_long_emergency_sl_capital_pct",
    "wt_long_tp1_pct",
    "wt_long_tp1_fraction",
    "wt_long_tp2_pct",
    "wt_long_tp2_fraction",
    "fee_rate",
    "slippage_bps",
]

PARAM_SUMMARY_LABELS = {
    "trade_direction": "Direction",
    "wt_long_entry_window_bars": "Entry window H1",
    "wt_long_entry_max_above_zero": "Long open level H1",
    "wt_long_close_min_level": "Long close level H1",
    "wt_short_entry_min_below_zero": "Short open level H1",
    "wt_short_close_max_level": "Short close level H1",
    "wt_h4_long_filter_max": "Long open level H4",
    "wt_h4_long_close_min": "Long close level H4",
    "wt_h4_short_filter_min": "Short open level H4",
    "wt_h4_short_close_max": "Short close level H4",
    "wt_long_emergency_sl_capital_pct": "Stop loss",
    "wt_long_tp1_pct": "Long TP1",
    "wt_long_tp1_fraction": "Long TP1 fraction",
    "wt_long_tp2_pct": "Long TP2",
    "wt_long_tp2_fraction": "Long TP2 fraction",
    "fee_rate": "Fee rate",
    "slippage_bps": "Slippage bps",
}


def _format_param_value(value, key: str | None = None):
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        if key and key.endswith("_pct"):
            return f"{value * 100.0:.2f}%"
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
                "wartość": _format_param_value(params[key], key),
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
                "Bee7 zachowuje dashboard bee1, ale testuje bardziej elastyczne wejscia H1 "
                "oraz miekki filtr H4 oparty na ostatniej zamknietej swiecy H4."
            ),
        ], className="hero-copy"),
        html.Div([
            html.Div("Note", className="hero-note-title"),
            html.P(
                "Long moze pojawic sie na zielonej kropce H1 albo kilka swiec po niej, gdy H1 jest w strefie open. "
                "H4 filtruje kontekst przez poziom open i poprawiajaca sie delte, a wyjscie longa wymaga close level H1/H4 oraz slabniecia H1."
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
        ("best_wt_long_entry_max_above_zero", "Long open level H1", C["green"]),
        ("best_wt_long_close_min_level", "Long close level H1", C["amber"]),
        ("best_wt_h4_long_filter_max", "Long open level H4", C["purple"]),
        ("best_wt_h4_long_close_min", "Long close level H4", C["blue"]),
    ]
    rows = max(1, (len(specs) + 1) // 2)
    fig = make_subplots(
        rows=rows,
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
    fig.update_layout(**PT, height=max(520, rows * 260))
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


def _direction_stats_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or "side" not in trades_df.columns or "pnl" not in trades_df.columns:
        return pd.DataFrame(
            [
                {"side": side, "trades": 0, "wins": 0, "losses": 0, "win_pnl_usd": 0.0, "loss_pnl_usd": 0.0, "net_pnl_usd": 0.0}
                for side in ("long", "short")
            ]
        )

    df = trades_df.copy()
    df["side"] = df["side"].astype(str).str.lower()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    records = []
    for side in ("long", "short"):
        sub = df[df["side"] == side]
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] <= 0]
        records.append(
            {
                "side": side,
                "trades": int(len(sub)),
                "wins": int(len(wins)),
                "losses": int(len(losses)),
                "win_pnl_usd": float(wins["pnl"].sum()),
                "loss_pnl_usd": float(losses["pnl"].sum()),
                "net_pnl_usd": float(sub["pnl"].sum()),
            }
        )
    return pd.DataFrame(records)


def _direction_stats_from_result(result_data: dict | None) -> pd.DataFrame:
    if not result_data:
        return _direction_stats_df(pd.DataFrame())
    return _direction_stats_df(pd.DataFrame(result_data.get("trades", [])))


def _direction_row(direction_df: pd.DataFrame, side: str) -> dict:
    if direction_df is None or direction_df.empty:
        return {"side": side, "trades": 0, "wins": 0, "losses": 0, "win_pnl_usd": 0.0, "loss_pnl_usd": 0.0, "net_pnl_usd": 0.0}
    rows = direction_df[direction_df["side"] == side]
    if rows.empty:
        return {"side": side, "trades": 0, "wins": 0, "losses": 0, "win_pnl_usd": 0.0, "loss_pnl_usd": 0.0, "net_pnl_usd": 0.0}
    return rows.iloc[0].to_dict()


def _money(value) -> str:
    value = float(value or 0.0)
    sign = "+" if value > 0 else ""
    return f"${sign}{value:,.0f}"


def fig_report_price_wt(result_data: dict) -> go.Figure:
    symbol = str(result_data.get("symbol", ""))
    tf = str(result_data.get("tf", ""))
    params = result_data.get("best_params") or result_data.get("params_used") or DEFAULT_PARAMS
    trades_df = pd.DataFrame(result_data.get("trades", []))
    windows_df = pd.DataFrame(result_data.get("windows_df", []))
    payload = lightweight_chart_payload(symbol, tf, trades_df, windows_df=windows_df, params=params)

    candles = pd.DataFrame(payload.get("candles", []))
    if candles.empty:
        return go.Figure(layout=PT)
    candles["dt"] = pd.to_datetime(candles["time"], unit="s", utc=True, errors="coerce")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.04)
    fig.add_trace(
        go.Candlestick(
            x=candles["dt"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="OHLC",
            increasing={"line": {"color": C["green"], "width": 1}, "fillcolor": C["green"]},
            decreasing={"line": {"color": C["coral"], "width": 1}, "fillcolor": C["coral"]},
        ),
        row=1,
        col=1,
    )

    for line in payload.get("lines", []):
        data = pd.DataFrame(line.get("data", []))
        if data.empty:
            continue
        data["dt"] = pd.to_datetime(data["time"], unit="s", utc=True, errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=data["dt"],
                y=data["value"],
                mode="lines",
                name=line.get("label", line.get("id")),
                line={"color": line.get("color"), "width": line.get("lineWidth", 2)},
            ),
            row=1,
            col=1,
        )

    if not trades_df.empty:
        tdf = _normalize_chart_trades(trades_df)
        for side, color, symbol_marker, name in [
            ("long", C["green"], "triangle-up", "Long IN"),
            ("short", C["red"], "triangle-down", "Short IN"),
        ]:
            sub = tdf[tdf["side"] == side] if "side" in tdf.columns else pd.DataFrame()
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["entry_time"],
                    y=sub["entry_price"],
                    mode="markers",
                    name=name,
                    marker={"symbol": symbol_marker, "size": 8, "color": color, "line": {"width": 0.5, "color": "#ffffff"}},
                ),
                row=1,
                col=1,
            )

    signal_lookup = {}
    for line in payload.get("signalLines", []):
        data = pd.DataFrame(line.get("data", []))
        if data.empty:
            continue
        data["dt"] = pd.to_datetime(data["time"], unit="s", utc=True, errors="coerce")
        dash_style = "dot" if int(line.get("lineStyle", 0) or 0) == 2 else "solid"
        fig.add_trace(
            go.Scatter(
                x=data["dt"],
                y=data["value"],
                mode="lines",
                name=line.get("label", line.get("id")),
                line={"color": line.get("color"), "width": line.get("lineWidth", 2), "dash": dash_style},
            ),
            row=2,
            col=1,
        )
        signal_lookup[line.get("id")] = dict(zip(data["time"], data["value"]))

    marker_groups: dict[str, list[dict[str, object]]] = {}
    for marker in payload.get("signalMarkers", []):
        sid = marker.get("seriesId")
        value = signal_lookup.get(sid, {}).get(marker.get("time"))
        if value is None:
            continue
        label = str(marker.get("text", "cross"))
        marker_groups.setdefault(label, []).append(
            {
                "time": pd.to_datetime(marker.get("time"), unit="s", utc=True, errors="coerce"),
                "value": value,
                "color": marker.get("color", C["blue"]),
            }
        )
    for label, records in marker_groups.items():
        marker_df = pd.DataFrame(records)
        fig.add_trace(
            go.Scatter(
                x=marker_df["time"],
                y=marker_df["value"],
                mode="markers",
                name=label,
                marker={"size": 6, "color": marker_df["color"], "symbol": "circle"},
            ),
            row=2,
            col=1,
        )

    for yv, color in [
        (TMA_LOW_MIN, "rgba(34, 211, 170, 0.35)"),
        (TMA_LOW_MAX, "rgba(34, 211, 170, 0.35)"),
        (0.0, "rgba(149, 166, 188, 0.40)"),
        (TMA_HIGH_MIN, "rgba(255, 93, 115, 0.35)"),
        (TMA_HIGH_MAX, "rgba(255, 93, 115, 0.35)"),
    ]:
        fig.add_hline(y=yv, line_dash="dot", line_color=color, line_width=1, row=2, col=1)

    fig.update_layout(
        **PT,
        height=720,
        title=f"{symbol.upper()} {tf} | Cena + WaveTrend H1/H4",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": -0.12, "x": 0, "font": {"size": 9, "color": C["text"]}},
    )
    fig.update_yaxes(title_text="Cena", row=1, col=1, gridcolor=C["border"])
    fig.update_yaxes(title_text="WT", row=2, col=1, gridcolor=C["border"])
    return fig


def _report_filename(result_data: dict) -> str:
    mode = _safe_slug(str(result_data.get("mode", "run")).lower())
    symbol = _safe_slug(str(result_data.get("symbol", "asset")).upper())
    tf = _safe_slug(str(result_data.get("tf", "tf")).lower())
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"bee7_{mode}_{symbol}_{tf}_report_{stamp}.pdf"


def _report_value(value, decimals: int = 2) -> str:
    if value is None:
        return "n/d"
    try:
        if pd.isna(value):
            return "n/d"
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{decimals}f}"
    return str(value)


def _report_summary_rows(result_data: dict) -> list[list[str]]:
    stats = result_data.get("stats", {})
    capital = result_data.get("capital", INITIAL_CAPITAL)
    consistency = _report_value(stats.get("consistency_pct"), 0)
    consistency = f"{consistency}%" if consistency != "n/d" else consistency
    return [
        ["Tryb", str(result_data.get("mode", "n/d")).upper()],
        ["Symbol / TF", f"{str(result_data.get('symbol', '')).upper()} {result_data.get('tf', '')}"],
        ["Kapital startowy", _money(capital)],
        ["Kapital koncowy", _money(stats.get("final_capital", capital))],
        ["Net zwrot", f"{float(stats.get('net_return_pct', 0.0) or 0.0):+.2f}%"],
        ["Net PnL", _money(stats.get("net_profit_usd", 0.0))],
        ["Max DD", f"{float(stats.get('max_drawdown_pct', 0.0) or 0.0):.2f}%"],
        ["CAGR", f"{float(stats.get('cagr_pct', 0.0) or 0.0):.2f}%"],
        ["Transakcje", _report_value(stats.get("n_trades", 0), 0)],
        ["Winrate", f"{float(stats.get('winrate_pct', 0.0) or 0.0):.2f}%"],
        ["Profit Factor", _report_value(stats.get("profit_factor"))],
        ["Expectancy / trade", _money(stats.get("expectancy_usd", 0.0))],
        ["Return / Drawdown", _report_value(stats.get("return_drawdown_ratio"))],
        ["Sharpe / Sortino", f"{_report_value(stats.get('sharpe_ratio'))} / {_report_value(stats.get('sortino_ratio'))}"],
        ["R:R avg", _report_value(stats.get("risk_reward_ratio"))],
        ["Consistency / Stability", f"{consistency} / {_report_value(stats.get('stability_score'))}"],
        ["Oplaty", f"{_money(stats.get('fee_total_usd', 0.0))} ({_report_value(stats.get('fee_total_pct', 0.0))}%)"],
    ]


def _df_for_report_table(df: pd.DataFrame, max_rows: int = 80) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().head(max_rows)
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else round(float(v), 4))
    return out.fillna("")


def _table_flowables(title: str, df: pd.DataFrame, styles, table_style, max_cols: int = 7, max_rows: int = 80) -> list:
    from reportlab.platypus import Paragraph, Spacer, Table

    if df is None or df.empty:
        return [Paragraph(title, styles["Heading2"]), Paragraph("Brak danych.", styles["Muted"]), Spacer(1, 8)]

    df2 = _df_for_report_table(df, max_rows=max_rows)
    flowables = [Paragraph(title, styles["Heading2"])]
    columns = list(df2.columns)
    for start in range(0, len(columns), max_cols):
        chunk_cols = columns[start:start + max_cols]
        data = [chunk_cols] + df2[chunk_cols].astype(str).values.tolist()
        flowables.append(Table(data, repeatRows=1, style=table_style, hAlign="LEFT"))
        flowables.append(Spacer(1, 8))
    if len(df) > max_rows:
        flowables.append(Paragraph(f"Pokazano pierwsze {max_rows} z {len(df)} wierszy.", styles["Muted"]))
        flowables.append(Spacer(1, 8))
    return flowables


def _params_df(params: dict | None) -> pd.DataFrame:
    if not params:
        return pd.DataFrame()
    return pd.DataFrame([{"parametr": key, "wartosc": _report_value(value)} for key, value in sorted(params.items())])


def _figure_flowable(title: str, fig: go.Figure, styles, width: int = 1120, height: int = 620):
    import plotly.io as pio
    from reportlab.platypus import Image, Paragraph, Spacer

    fig = go.Figure(fig)
    fig.update_layout(width=width, height=height)
    image = pio.to_image(fig, format="png", width=width, height=height, scale=1.4)
    stream = io.BytesIO(image)
    return [
        Paragraph(title, styles["Heading2"]),
        Image(stream, width=740, height=740 * height / width),
        Spacer(1, 12),
    ]


def _build_report_pdf(result_data: dict) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        raise RuntimeError("Brakuje biblioteki reportlab. Po deployu/rebuildzie zaktualizowane requirements ją doinstalują.") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBee", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name="Muted", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#64748b")))
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    styles["Heading2"].textColor = colors.HexColor("#0f172a")

    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )

    stats = result_data.get("stats", {})
    capital = result_data.get("capital", INITIAL_CAPITAL)
    equity_df = pd.DataFrame(result_data.get("equity", []))
    fee_df = pd.DataFrame(result_data.get("fee_df", []))
    windows_df = pd.DataFrame(result_data.get("windows_df", []))
    direction_df = _direction_stats_from_result(result_data)
    side_df = pd.DataFrame(result_data.get("side_bk", []))
    cross_df = pd.DataFrame(result_data.get("cross_bk", []))
    yr_df = pd.DataFrame(result_data.get("yr_bk", []))
    q_df = pd.DataFrame(result_data.get("q_bk", []))
    params = result_data.get("best_params") or result_data.get("params_used") or {}

    story = [
        Paragraph("Bee7 - raport wynikow", styles["TitleBee"]),
        Paragraph(
            f"Wygenerowano: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
            f"{str(result_data.get('mode', '')).upper()} {str(result_data.get('symbol', '')).upper()} {result_data.get('tf', '')}",
            styles["Muted"],
        ),
        Spacer(1, 10),
        Paragraph("Podsumowanie", styles["Heading2"]),
        Table(_report_summary_rows(result_data), colWidths=[130, 250], style=table_style, hAlign="LEFT"),
        Spacer(1, 12),
        Paragraph("Long / Short", styles["Heading2"]),
        Table(
            [["side", "trades", "wins", "losses", "win_pnl_usd", "loss_pnl_usd", "net_pnl_usd"]]
            + _df_for_report_table(direction_df).astype(str).values.tolist(),
            repeatRows=1,
            style=table_style,
            hAlign="LEFT",
        ),
        Spacer(1, 12),
        Paragraph("Parametry", styles["Heading2"]),
        Table(
            [["parametr", "wartosc"]] + _params_df(params).astype(str).values.tolist(),
            repeatRows=1,
            style=table_style,
            hAlign="LEFT",
        ),
        PageBreak(),
    ]

    try:
        story.extend(_figure_flowable("Equity / Drawdown", fig_eq(equity_df, capital), styles, width=1120, height=620))
        story.extend(_figure_flowable("Cena + WT H1/H4", fig_report_price_wt(result_data), styles, width=1120, height=700))
        if not fee_df.empty:
            story.extend(_figure_flowable("Oplaty", fig_fee(fee_df), styles, width=1120, height=420))
        if not windows_df.empty:
            story.extend(_figure_flowable("Okna WFO", fig_wfo(windows_df), styles, width=1120, height=420))
            story.extend(_figure_flowable("Rozkład parametrów WFO", fig_pdist(windows_df), styles, width=1120, height=820))
    except Exception as exc:
        raise RuntimeError("Nie udało się wyrenderować wykresów do PDF. Sprawdź, czy zainstalował się pakiet kaleido.") from exc

    story.append(PageBreak())
    story.extend(_table_flowables("Long vs Short", side_df, styles, table_style))
    story.extend(_table_flowables("Statystyki long/short - wygrane i przegrane", direction_df, styles, table_style))
    story.extend(_table_flowables("Bullish vs Bearish H1 cross", cross_df, styles, table_style))
    story.extend(_table_flowables("Breakdown roczny", yr_df, styles, table_style))
    story.extend(_table_flowables("Breakdown kwartalny", q_df, styles, table_style))
    if not fee_df.empty:
        story.extend(_table_flowables("Oplaty wg okresu", fee_df, styles, table_style))
    if not windows_df.empty:
        story.extend(_table_flowables("Okna WFO", windows_df, styles, table_style, max_cols=7, max_rows=60))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

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
    trade_label = fmt_text(trade.get("trade_label"), "")
    title = f"Trade #{trade_no} / {trade_label}" if trade_label else (f"Trade #{trade_no}" if trade_no else "Szczegoly trade'u")
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
                "Czas do TP1 h",
                fmt_float(trade.get("time_to_tp1_hours"), ".2f", "n/d"),
            ),
            row2("Czas w pozycji h", fmt_float(trade.get("holding_hours"), ".2f", "n/d"),
                 "Fee", f"{fmt_float(trade.get('fee_usd'), '.2f', '0.00')} USD"),
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
    if "logical_trade_no" not in tdf.columns:
        tdf["logical_trade_no"] = tdf["trade_no"]
    if "trade_event" not in tdf.columns:
        reasons = tdf.get("reason", pd.Series([""] * len(tdf), index=tdf.index)).astype(str)
        tdf["trade_event"] = np.select(
            [
                reasons.str.contains("TP2_PARTIAL", case=False, na=False),
                reasons.str.contains("TP1_PARTIAL", case=False, na=False),
            ],
            ["TP2", "TP1"],
            default="EXIT",
        )
    if "trade_label" not in tdf.columns:
        logical_ids = pd.to_numeric(tdf["logical_trade_no"], errors="coerce").fillna(tdf["trade_no"])
        tdf["trade_label"] = [
            f"{int(float(trade_id))} {str(event or 'EXIT').upper()}"
            for trade_id, event in zip(logical_ids, tdf["trade_event"])
        ]

    numeric_cols = [
        "entry_price", "exit_price", "gross_ret", "fee_ret", "net_ret",
        "pnl", "fee_usd", "entry_wt1", "entry_wt2", "entry_delta",
        "entry_signal_level", "entry_ema_filter", "entry_ema_filter_len",
        "entry_htf_ema200", "entry_atr", "entry_stop_price",
        "entry_h4_wt1", "entry_h4_wt2", "entry_h4_delta",
        "exit_wt1", "exit_wt2", "exit_delta",
        "exit_signal_level", "exit_h4_wt1", "exit_h4_wt2", "exit_h4_delta",
        "holding_hours", "time_to_tp1_hours",
        "close_fraction", "remaining_fraction_after", "position_notional", "logical_trade_no",
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
            "exit_price", "logical_trade_no", "trade_event", "trade_label",
            "holding_hours", "time_to_tp1_hours",
            "close_fraction", "remaining_fraction_after",
            "gross_ret", "fee_ret", "net_ret", "pnl", "fee_usd", "reason",
        ] if c in tdf.columns
    ]
    disp = tdf[cols].copy()
    for col in ["gross_ret", "fee_ret", "net_ret"]:
        if col in disp.columns:
            disp[col] = (disp[col] * 100).round(3).astype(str) + "%"
    for col in ["pnl", "fee_usd", "entry_price", "exit_price"]:
        if col in disp.columns:
            disp[col] = disp[col].round(2)
    for col in ["holding_hours", "time_to_tp1_hours"]:
        if col in disp.columns:
            disp[col] = disp[col].round(2)
    for col in ["close_fraction", "remaining_fraction_after"]:
        if col in disp.columns:
            disp[col] = (disp[col] * 100).round(1).astype(str) + "%"
    for col in ["entry_time", "exit_time"]:
        if col in disp.columns:
            disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d %H:%M")
    if "trade_no" in disp.columns:
        disp["trade_no"] = disp["trade_no"].astype(int)
    if "logical_trade_no" in disp.columns:
        disp["logical_trade_no"] = disp["logical_trade_no"].astype(int)
    return disp


def _pine_number(value, digits: int = 8) -> str:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        return "na"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _pine_string(value) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\r", " ").replace("\n", " ")
    return f'"{text}"'


def _pine_timestamp(value) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return f'timestamp("UTC",{ts.year},{ts.month},{ts.day},{ts.hour},{ts.minute})'


def _pine_trade_add_lines(result_data: dict) -> list[str]:
    trades_df = _normalize_chart_trades(pd.DataFrame(result_data.get("trades", [])))
    if trades_df.empty:
        return []

    lines: list[str] = []
    added_entries: set[int] = set()

    for idx, trade in trades_df.iterrows():
        side = str(trade.get("side", "")).strip().lower()
        if side not in ("long", "short"):
            continue

        entry_price = pd.to_numeric(trade.get("entry_price"), errors="coerce")
        exit_price = pd.to_numeric(trade.get("exit_price"), errors="coerce")
        entry_ts = _pine_timestamp(trade.get("entry_time"))
        exit_ts = _pine_timestamp(trade.get("exit_time"))
        if pd.isna(entry_price) or entry_ts is None:
            continue

        try:
            logical_trade_no = int(float(trade.get("logical_trade_no", trade.get("trade_no", idx + 1))))
        except Exception:
            logical_trade_no = int(idx) + 1
        pnl = _pine_number(trade.get("pnl", 0.0), 8)
        direction = side.upper()
        entry_action, exit_action = ("BUY", "SELL") if side == "long" else ("SELL", "BUY")
        event = str(trade.get("trade_event", "") or "").strip().upper()
        if not event:
            reason = str(trade.get("reason", "")).upper()
            event = "TP2" if "TP2_PARTIAL" in reason else "TP1" if "TP1_PARTIAL" in reason else "EXIT"
        entry_comment = f"T{logical_trade_no} OPEN {direction} {entry_action}"
        exit_comment = f"T{logical_trade_no} {event} {direction} {exit_action}"

        if logical_trade_no not in added_entries:
            lines.append(
                f"    f_add({entry_ts},{_pine_string(entry_action)},{_pine_number(entry_price, 8)},"
                f"{logical_trade_no},{_pine_string(entry_comment)},{pnl})"
            )
            added_entries.add(logical_trade_no)
        if pd.notna(exit_price) and exit_ts is not None:
            lines.append(
                f"    f_add({exit_ts},{_pine_string(exit_action)},{_pine_number(exit_price, 8)},"
                f"{logical_trade_no},{_pine_string(exit_comment)},{pnl})"
            )

    return lines


def _build_pine_trades_overlay(result_data: dict) -> str:
    symbol = str(result_data.get("symbol", "asset") or "asset").upper()
    tf = str(result_data.get("tf", "tf") or "tf")
    mode = str(result_data.get("mode", "run") or "run").upper()
    add_lines = _pine_trade_add_lines(result_data)
    generated = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    n_events = len(add_lines)
    n_trades = len(result_data.get("trades", []) or [])
    events_block = "\n".join(add_lines) if add_lines else "    // No trades available"

    return f"""//@version=5
indicator("BEE7 Trades Overlay with Trade Numbers", overlay=true, max_labels_count=500)

// Generated by BEE7 dashboard: {generated}
// Source: {symbol} {tf} {mode}, trades: {n_trades}, events: {n_events}

showLabels = input.bool(true, "Pokaz etykiety z numerem transakcji")
showPrice = input.bool(false, "Pokaz cene w etykiecie")
showPnl = input.bool(false, "Pokaz PnL w etykiecie")
showMarkers = input.bool(true, "Pokaz trojkaty BUY/SELL")
maxLabelsToDraw = input.int(500, "Maks. etykiet", minval=0, maxval=500)

var int[] txTime = array.new_int()
var string[] txAction = array.new_string()
var float[] txPrice = array.new_float()
var int[] txTradeNo = array.new_int()
var string[] txComment = array.new_string()
var float[] txPnl = array.new_float()
var bool[] drawn = array.new_bool()
var int labelsDrawn = 0

f_add(_t, _action, _price, _tradeNo, _comment, _pnl) =>
    array.push(txTime, _t)
    array.push(txAction, _action)
    array.push(txPrice, _price)
    array.push(txTradeNo, _tradeNo)
    array.push(txComment, _comment)
    array.push(txPnl, _pnl)
    array.push(drawn, false)

if barstate.isfirst
{events_block}

buyOnBar = false
sellOnBar = false

if array.size(txTime) > 0
    for i = 0 to array.size(txTime) - 1
        t = array.get(txTime, i)
        inBar = time <= t and t < time_close
        if inBar
            action = array.get(txAction, i)
            price = array.get(txPrice, i)
            comment = array.get(txComment, i)
            pnl = array.get(txPnl, i)
            isBuy = action == "BUY"
            buyOnBar := buyOnBar or isBuy
            sellOnBar := sellOnBar or not isBuy
            if showLabels and not array.get(drawn, i) and labelsDrawn < maxLabelsToDraw
                txt = comment + (showPrice ? "\\n" + str.tostring(price, format.mintick) : "") + (showPnl ? "\\nPnL: " + str.tostring(pnl, "#.##") : "")
                label.new(x=bar_index, y=isBuy ? low : high, text=txt, xloc=xloc.bar_index, yloc=isBuy ? yloc.belowbar : yloc.abovebar, style=isBuy ? label.style_label_up : label.style_label_down, color=isBuy ? color.lime : color.red, textcolor=color.white, size=size.small, tooltip=comment)
                array.set(drawn, i, true)
                labelsDrawn := labelsDrawn + 1

plotshape(showMarkers and buyOnBar, title="BUY", style=shape.triangleup, location=location.belowbar, size=size.tiny, color=color.lime, text="BUY")
plotshape(showMarkers and sellOnBar, title="SELL", style=shape.triangledown, location=location.abovebar, size=size.tiny, color=color.red, text="SELL")
"""


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


def _chart_param(params: dict | None, key: str, default):
    params = params or {}
    value = params.get(key, params.get(f"best_{key}", default))
    return default if value is None else value


def _chart_line_data(df: pd.DataFrame, column: str, precision: int) -> list[dict[str, float | int]]:
    if column not in df.columns:
        return []
    series: list[dict[str, float | int]] = []
    for row in df[["time", column]].itertuples(index=False):
        value = getattr(row, column)
        if pd.isna(value):
            continue
        series.append({"time": _unix_seconds(row.time), "value": round(float(value), precision)})
    return series


def _cross_markers(
    df: pd.DataFrame,
    wt1_col: str,
    wt2_col: str,
    label: str,
    marker_prefix: str,
) -> list[dict[str, object]]:
    if wt1_col not in df.columns or wt2_col not in df.columns:
        return []

    prev_wt1 = df[wt1_col].shift(1)
    prev_wt2 = df[wt2_col].shift(1)
    green = (prev_wt1 <= prev_wt2) & (df[wt1_col] > df[wt2_col])
    red = (prev_wt1 >= prev_wt2) & (df[wt1_col] < df[wt2_col])
    markers: list[dict[str, object]] = []

    for row in df.loc[green.fillna(False), ["time"]].itertuples(index=False):
        markers.append(
            {
                "seriesId": wt1_col,
                "time": _unix_seconds(row.time),
                "position": "belowBar",
                "shape": "circle",
                "color": C["green"],
                "text": f"{marker_prefix} L",
                "label": f"{label} long cross",
            }
        )

    for row in df.loc[red.fillna(False), ["time"]].itertuples(index=False):
        markers.append(
            {
                "seriesId": wt1_col,
                "time": _unix_seconds(row.time),
                "position": "aboveBar",
                "shape": "circle",
                "color": C["red"],
                "text": f"{marker_prefix} S",
                "label": f"{label} short cross",
            }
        )

    return markers


def _chart_view_mode(value: str | None) -> str:
    mode = str(value or "standard").strip().lower()
    return mode if mode in CHART_VIEW_VALUES else "standard"


def _fmt_diag(value, digits: int = 2) -> str:
    try:
        number = float(value)
        if np.isnan(number):
            return "n/d"
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return "n/d"


def _safe_diag_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value).lower()).strip("-") or "other"


def _diagnostic_trade_lookup(trades_df: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    if trades_df is None or trades_df.empty:
        return lookup
    for trade in trades_df.itertuples(index=False):
        entry_time = pd.to_datetime(getattr(trade, "entry_time", None), utc=True, errors="coerce")
        if pd.isna(entry_time):
            continue
        side = str(getattr(trade, "side", "")).lower()
        if side not in ("long", "short"):
            continue
        lookup[(_unix_seconds(entry_time), side)] = {
            "trade_no": int(_as_float(getattr(trade, "trade_no", 0), 0.0)),
            "side": side,
            "reason": str(getattr(trade, "reason", "") or ""),
            "pnl": _as_float(getattr(trade, "pnl", 0.0), 0.0),
        }
    return lookup


def _active_trade_at(trades_df: pd.DataFrame, ts: pd.Timestamp) -> dict[str, object] | None:
    if trades_df is None or trades_df.empty:
        return None
    if "entry_time" not in trades_df.columns or "exit_time" not in trades_df.columns:
        return None
    active = trades_df[
        (trades_df["entry_time"] <= ts)
        & (trades_df["exit_time"] > ts)
    ]
    if active.empty:
        return None
    row = active.iloc[0]
    return {
        "trade_no": int(_as_float(row.get("trade_no", 0), 0.0)),
        "side": str(row.get("side", "")).lower(),
    }


def _diag_h4_state(row: pd.Series, side: str, params: dict, h4_wt1_col: str, h4_wt2_col: str) -> tuple[bool, str]:
    h4_wt1 = _as_float(row.get(h4_wt1_col, np.nan), np.nan)
    h4_wt2 = _as_float(row.get(h4_wt2_col, np.nan), np.nan)
    h4_delta = h4_wt1 - h4_wt2 if not np.isnan(h4_wt1) and not np.isnan(h4_wt2) else np.nan
    h4_prev_delta = _as_float(row.get("h4_prev_wt_delta", np.nan), np.nan)
    if any(np.isnan(v) for v in [h4_wt1, h4_wt2, h4_delta, h4_prev_delta]):
        return False, "brak danych H4"

    if side == "long":
        threshold = float(params.get("wt_h4_long_filter_max", DEFAULT_PARAMS["wt_h4_long_filter_max"]))
        in_zone = h4_wt1 <= threshold and h4_wt2 <= threshold
        improving = h4_delta > h4_prev_delta
        if not in_zone:
            return False, f"H4 za wysoko (WT1/WT2 muszą być <= {threshold:g})"
        if not improving:
            return False, "delta H4 nie poprawia się dla longa"
        return True, "H4 OK"

    threshold = float(params.get("wt_h4_short_filter_min", DEFAULT_PARAMS["wt_h4_short_filter_min"]))
    in_zone = h4_wt1 >= threshold and h4_wt2 >= threshold
    current_gap = abs(h4_delta)
    prev_gap = abs(h4_prev_delta)
    converging = current_gap < prev_gap
    correct_direction = h4_delta < h4_prev_delta
    if not in_zone:
        return False, f"H4 za nisko (WT1/WT2 muszą być >= {threshold:g})"
    if not converging or not correct_direction:
        return False, "H4 nie zbliża linii dla shorta"
    return True, "H4 OK"


def _diagnostic_rejection(
    row: pd.Series,
    side: str,
    params: dict,
    h1_wt1_col: str,
    h1_wt2_col: str,
    h4_wt1_col: str,
    h4_wt2_col: str,
    active_trade: dict[str, object] | None,
) -> tuple[str, str]:
    wt1 = _as_float(row.get(h1_wt1_col, np.nan), np.nan)
    wt2 = _as_float(row.get(h1_wt2_col, np.nan), np.nan)
    level_now = min(abs(wt1), abs(wt2)) if not np.isnan(wt1) and not np.isnan(wt2) else np.nan
    min_level = float(params.get("wt_min_signal_level", DEFAULT_PARAMS["wt_min_signal_level"]))
    allow_longs = bool(params.get("allow_longs", True))
    allow_shorts = bool(params.get("allow_shorts", True)) and bool(params.get("short_trading_enabled", False))

    if side == "long" and not allow_longs:
        return "direction", "long wyłączony w ustawieniach"
    if side == "short" and not allow_shorts:
        return "direction", "short wyłączony w ustawieniach"
    if np.isnan(wt1) or np.isnan(wt2):
        return "data", "brak danych WT H1"

    if side == "long":
        threshold = float(params.get("wt_long_entry_max_above_zero", DEFAULT_PARAMS["wt_long_entry_max_above_zero"]))
        if wt1 > threshold or wt2 > threshold:
            return "zone", f"H1 poza strefą long (WT1/WT2 muszą być <= {threshold:g})"
    else:
        threshold = float(params.get("wt_short_entry_min_below_zero", DEFAULT_PARAMS["wt_short_entry_min_below_zero"]))
        if wt1 < threshold or wt2 < threshold:
            return "zone", f"H1 poza strefą short (WT1/WT2 muszą być >= {threshold:g})"

    if np.isnan(level_now) or level_now < min_level:
        return "level", f"za słaby poziom WT ({_fmt_diag(level_now)} < {min_level:g})"

    h4_ok, h4_reason = _diag_h4_state(row, side, params, h4_wt1_col, h4_wt2_col)
    if not h4_ok:
        return "h4", h4_reason

    if active_trade:
        return "position", f"pozycja już otwarta: #{active_trade['trade_no']} {str(active_trade['side']).upper()}"

    return "other", "warunki wyglądają OK, ale brak trade'u w wyniku; sprawdź okno WFO/parametry"


def _diagnostic_signal_overlays(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    params: dict,
    h1_wt1_col: str,
    h1_wt2_col: str,
    h4_wt1_col: str,
    h4_wt2_col: str,
    filter_mode: str = "all",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if df.empty or h1_wt1_col not in df.columns or h1_wt2_col not in df.columns:
        return [], []

    trade_lookup = _diagnostic_trade_lookup(trades_df)
    filter_mode = str(filter_mode or "all").lower()
    colors = {
        "accepted_long": C["green"],
        "accepted_short": C["coral"],
        "h4": C["amber"],
        "zone": C["blue"],
        "position": C["muted"],
        "direction": "#fb923c",
        "level": C["purple"],
        "data": C["muted"],
        "other": "#c084fc",
    }
    signal_markers: list[dict[str, object]] = []
    price_pins: list[dict[str, object]] = []

    prev_wt1 = df[h1_wt1_col].shift(1)
    prev_wt2 = df[h1_wt2_col].shift(1)
    long_cross = (prev_wt1 <= prev_wt2) & (df[h1_wt1_col] > df[h1_wt2_col])
    short_cross = (prev_wt1 >= prev_wt2) & (df[h1_wt1_col] < df[h1_wt2_col])

    for pos, (_, row) in enumerate(df.iterrows()):
        if bool(long_cross.iloc[pos]):
            side = "long"
        elif bool(short_cross.iloc[pos]):
            side = "short"
        else:
            continue
        if filter_mode in ("long", "short") and side != filter_mode:
            continue

        ts = pd.to_datetime(row["time"], utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        time_sec = _unix_seconds(ts)
        accepted_trade = trade_lookup.get((time_sec, side))
        active_trade = None if accepted_trade else _active_trade_at(trades_df, ts)
        if accepted_trade:
            code = "accepted"
            reason = f"wejście wykonane jako trade #{accepted_trade['trade_no']}"
            color = colors["accepted_long"] if side == "long" else colors["accepted_short"]
            text = f"{'L' if side == 'long' else 'S'}✓"
            label = f"{'LONG' if side == 'long' else 'SHORT'} #{accepted_trade['trade_no']}"
        else:
            code, reason = _diagnostic_rejection(
                row,
                side,
                params,
                h1_wt1_col,
                h1_wt2_col,
                h4_wt1_col,
                h4_wt2_col,
                active_trade,
            )
            color = colors.get(code, colors["other"])
            text = f"{'L' if side == 'long' else 'S'}?"
            label = f"{'LONG' if side == 'long' else 'SHORT'} odrzucony"

        wt1 = _as_float(row.get(h1_wt1_col, np.nan), np.nan)
        wt2 = _as_float(row.get(h1_wt2_col, np.nan), np.nan)
        h4_wt1 = _as_float(row.get(h4_wt1_col, np.nan), np.nan)
        h4_wt2 = _as_float(row.get(h4_wt2_col, np.nan), np.nan)
        tooltip = (
            f"{label}\n"
            f"Decyzja: {'WEJŚCIE' if accepted_trade else 'BRAK WEJŚCIA'}\n"
            f"Powód: {reason}\n"
            f"WT1 H1: {_fmt_diag(wt1)} | WT2 H1: {_fmt_diag(wt2)}\n"
            f"WT1 H4: {_fmt_diag(h4_wt1)} | WT2 H4: {_fmt_diag(h4_wt2)}\n"
            f"Cena: {_fmt_diag(row.get('close', np.nan), 4)}"
        )

        signal_markers.append(
            {
                "seriesId": h1_wt1_col,
                "time": time_sec,
                "position": "belowBar" if side == "long" else "aboveBar",
                "shape": "circle",
                "color": color,
                "text": text,
                "label": reason,
            }
        )
        price_pins.append(
            {
                "tradeNo": accepted_trade.get("trade_no") if accepted_trade else None,
                "time": time_sec,
                "price": round(float(row["close"]), 6),
                "label": text,
                "anchor": "below" if side == "long" else "above",
                "kind": "signal",
                "decision": "accepted" if accepted_trade else "rejected",
                "rejectCode": _safe_diag_token(code),
                "side": side,
                "color": color,
                "tooltip": tooltip,
            }
        )

    return signal_markers, price_pins


def lightweight_chart_payload(
    symbol: str,
    tf: str,
    trades_df: pd.DataFrame,
    windows_df: pd.DataFrame | None = None,
    selected_trade: dict | None = None,
    filter_mode: str = "all",
    params: dict | None = None,
    view_mode: str = "standard",
) -> dict[str, object]:
    view_mode = _chart_view_mode(view_mode)
    source_df = _chart_source_df(symbol, tf)
    if source_df is None or source_df.empty:
        return {
            "chartView": view_mode,
            "candles": [],
            "lines": [],
            "signalLines": [],
            "signalMarkers": [],
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
            "chartView": view_mode,
            "candles": [],
            "lines": [],
            "signalLines": [],
            "signalMarkers": [],
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

    channel_len = int(_chart_param(params, "wt_channel_len", DEFAULT_PARAMS["wt_channel_len"]))
    avg_len = int(_chart_param(params, "wt_avg_len", DEFAULT_PARAMS["wt_avg_len"]))
    signal_len = int(_chart_param(params, "wt_signal_len", DEFAULT_PARAMS["wt_signal_len"]))
    h4_interval = str(_chart_param(params, "wt_h4_filter_interval", DEFAULT_PARAMS["wt_h4_filter_interval"]))

    h1_wt1_col, h1_wt2_col = wt_columns(channel_len, avg_len, signal_len)
    h4_wt1_col = htf_wt1_column(channel_len, avg_len, h4_interval)
    h4_wt2_col = htf_wt2_column(channel_len, avg_len, signal_len, h4_interval)
    if h1_wt1_col not in df.columns or h1_wt2_col not in df.columns:
        h1_wt1_col, h1_wt2_col = "wt1", "wt2"
    if h4_wt1_col not in df.columns or h4_wt2_col not in df.columns:
        h4_wt1_col, h4_wt2_col = "h4_wt1", "h4_wt2"

    signal_specs = [
        (h1_wt1_col, f"WT1 H1 ({channel_len}/{avg_len})", C["blue"], 2.0, True, 0),
        (h1_wt2_col, f"WT2 H1 ({signal_len})", C["amber"], 2.1, True, 0),
        (h4_wt1_col, f"WT1 H4 ({channel_len}/{avg_len})", C["purple"], 3.0, True, 0),
        (h4_wt2_col, f"WT2 H4 ({signal_len})", C["coral"], 3.2, True, 0),
    ]
    signal_lines = []
    for column, label, color, width, visible, line_style in signal_specs:
        if column not in df.columns:
            continue
        signal_lines.append(
            {
                "id": column,
                "label": label,
                "color": color,
                "lineWidth": width,
                "lineStyle": line_style,
                "visible": visible,
                "data": _chart_line_data(df, column, 4),
            }
        )

    h1_cross_markers = _cross_markers(df, h1_wt1_col, h1_wt2_col, "H1", "H1")
    h4_cross_markers = _cross_markers(df, h4_wt1_col, h4_wt2_col, "H4", "H4")
    diagnostic_signal_markers: list[dict[str, object]] = []
    diagnostic_pins: list[dict[str, object]] = []
    if view_mode == "diagnostic":
        diagnostic_signal_markers, diagnostic_pins = _diagnostic_signal_overlays(
            df,
            normalized_trades,
            params,
            h1_wt1_col,
            h1_wt2_col,
            h4_wt1_col,
            h4_wt2_col,
            filter_mode=filter_mode,
        )
        signal_markers = diagnostic_signal_markers + h4_cross_markers
    elif view_mode == "trades":
        signal_markers = []
    else:
        signal_markers = h1_cross_markers + h4_cross_markers

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
    added_entry_markers: set[int] = set()
    if not filtered_trades.empty:
        for trade in filtered_trades.itertuples(index=False):
            trade_no = int(_as_float(getattr(trade, "trade_no", 0), 0.0))
            logical_trade_no = int(_as_float(getattr(trade, "logical_trade_no", trade_no), float(trade_no)))
            trade_label = str(getattr(trade, "trade_label", "") or f"{logical_trade_no} EXIT")
            side = str(getattr(trade, "side", "")).lower()
            pnl = _as_float(getattr(trade, "pnl", 0.0), 0.0)
            reason = str(getattr(trade, "reason", "") or "")
            entry_price = _as_float(getattr(trade, "entry_price", 0.0), 0.0)
            exit_price = _as_float(getattr(trade, "exit_price", 0.0), 0.0)
            if logical_trade_no not in added_entry_markers:
                markers.append(
                    {
                        "time": _unix_seconds(trade.entry_time),
                        "position": "belowBar" if side == "long" else "aboveBar",
                        "shape": "arrowUp" if side == "long" else "arrowDown",
                        "color": C["green"] if side == "long" else C["coral"],
                        "text": f"#{logical_trade_no}",
                    }
                )
                trade_pins.append(
                    {
                        "tradeNo": trade_no,
                        "time": _unix_seconds(trade.entry_time),
                        "price": round(entry_price, 6),
                        "label": f"{logical_trade_no} OPEN",
                        "anchor": "below" if side == "long" else "above",
                        "kind": "entry",
                        "side": side,
                        "color": C["green"] if side == "long" else C["coral"],
                        "tooltip": f"Trade {logical_trade_no} open | {side.upper()} | {reason or 'OPEN'}",
                    }
                )
                added_entry_markers.add(logical_trade_no)
            markers.append(
                {
                    "time": _unix_seconds(trade.exit_time),
                    "position": "aboveBar" if side == "long" else "belowBar",
                    "shape": "circle",
                    "color": "#22c55e" if pnl >= 0 else "#ef4444",
                    "text": trade_label,
                }
            )
            trade_pins.append(
                {
                    "tradeNo": trade_no,
                    "time": _unix_seconds(trade.exit_time),
                    "price": round(exit_price, 6),
                    "label": trade_label,
                    "anchor": "above" if side == "long" else "below",
                    "kind": "exit",
                    "side": side,
                    "color": "#22c55e" if pnl >= 0 else "#ef4444",
                    "tooltip": f"Trade {trade_label} | {side.upper()} | {reason or f'{pnl:+.2f} USD'}",
                }
            )

    if view_mode == "diagnostic":
        trade_pins.extend(diagnostic_pins)

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
        "chartView": view_mode,
        "candles": candles,
        "lines": lines,
        "signalLines": signal_lines,
        "signalMarkers": signal_markers,
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
    saved_opts = _saved_result_options()
    return html.Div([
        # nagłówek
        html.Div([
            html.Div("Bee7 Bot",style={"fontSize":"16px","fontWeight":"700","color":C["text"]}),
            html.Div("WaveTrend independent short profile",style={"fontSize":"11px","color":C["muted"],"marginTop":"2px"}),
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
                {"label": "Long + independent short", "value": "both"},
            ], "both")),
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
            ], style={"display":"none"}),
            html.Div([
                html.Div([field("Signal", inp("inp-bt-signal", DEFAULT_PARAMS["wt_signal_len"], type="number", min=2, step=1))], style={"flex":"1"}),
                html.Div(
                    [field("Min level", inp("inp-bt-min-level", DEFAULT_PARAMS["wt_min_signal_level"], type="number", step=1))],
                    style={"display":"none"},
                ),
            ], style={"display":"none"}),
            html.Div([
                html.Div([field("Long open level H1", inp("inp-bt-long-zone", DEFAULT_PARAMS["wt_long_entry_max_above_zero"], type="number", step=1))], style={"flex":"1"}),
                html.Div([field("Long close level H1", inp("inp-bt-long-close-level", DEFAULT_PARAMS["wt_long_close_min_level"], type="number", step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Short open level H1", inp("inp-bt-short-zone", DEFAULT_PARAMS["wt_short_entry_min_below_zero"], type="number", step=1))], style={"flex":"1"}),
                html.Div([field("Short close level H1", inp("inp-bt-short-close-level", DEFAULT_PARAMS["wt_short_close_max_level"], type="number", step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Long open level H4", inp("inp-bt-h4-long", DEFAULT_PARAMS["wt_h4_long_filter_max"], type="number", step=1))], style={"flex":"1"}),
                html.Div([field("Long close level H4", inp("inp-bt-h4-long-close", DEFAULT_PARAMS["wt_h4_long_close_min"], type="number", step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Short open level H4", inp("inp-bt-h4-short", DEFAULT_PARAMS["wt_h4_short_filter_min"], type="number", step=1))], style={"flex":"1"}),
                html.Div([field("Short close level H4", inp("inp-bt-h4-short-close", DEFAULT_PARAMS["wt_h4_short_close_max"], type="number", step=1))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("TP1 % long", inp("inp-bt-long-tp1-pct", round(DEFAULT_PARAMS["wt_long_tp1_pct"] * 100.0, 2), type="number", min=0, step=0.1))], style={"flex":"1"}),
                html.Div([field("Stop loss", drp(
                    "inp-bt-long-sl-pct",
                    [{"label": "Wyłączony", "value": 0.0}] + [
                        {"label": f"{v * 100:.0f}%", "value": v}
                        for v in WT_LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS
                        if v > 0.0
                    ],
                    DEFAULT_PARAMS["wt_long_emergency_sl_capital_pct"],
                ))], style={"flex":"1"}),
            ], style={"display":"flex","gap":"8px"}),
            html.Div([
                html.Div([field("Entry window H1", inp("inp-bt-reentry", DEFAULT_PARAMS["wt_long_entry_window_bars"], type="number", min=0, max=12, step=1))], style={"flex":"1"}),
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
                "BEE7: long i short maja osobne poziomy wejscia oraz wyjscia H1/H4. Przeciwny sygnal zamyka aktywna pozycje i otwiera druga strone na tym samym barze. TP1/TP2 i break-even po TP1 dzialaja dla long i short.",
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
            html.Div(
                "WaveTrend WFO fixed: Channel 10 / Average 21 / Signal 3",
                style={"fontSize":"11px","color":C["muted"],"marginTop":"4px","marginBottom":"8px"},
            ),
            html.Div([
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
            ], style={"display":"none"}),
            html.Div([
                sec("Entry window H1"),
                dcc.Checklist(id="chk-grid-reentry",
                    options=[{"label":f" {v}","value":v} for v in WT_REENTRY_WINDOW_GRID],
                    value=WT_REENTRY_WINDOW_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            ]),
            html.Div([
                sec("Stop loss"),
                dcc.Checklist(id="chk-grid-long-sl-pct",
                    options=[
                        {
                            "label": " Wyłączony" if v == 0.0 else f" {v * 100:.0f}%",
                            "value": v,
                        }
                        for v in WT_LONG_EMERGENCY_SL_CAPITAL_PCT_OPTIONS
                    ],
                    value=WT_LONG_EMERGENCY_SL_CAPITAL_PCT_GRID, inline=True,
                    inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                    labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            ]),
            html.Div([
                sec("Siatka Min level"),
                dcc.Checklist(id="chk-grid-min-level",
                    options=[{"label":f" {v:.1f}","value":v} for v in WT_MIN_SIGNAL_LEVEL_OPTIONS],
                    value=WT_MIN_SIGNAL_LEVEL_GRID, inline=True,
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
            sec("Long open level H1"),
            dcc.Checklist(id="chk-grid-long-zone",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_LONG_ENTRY_MAX_ABOVE_ZERO_OPTIONS],
                value=WT_LONG_ENTRY_MAX_ABOVE_ZERO_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Long close level H1"),
            dcc.Checklist(id="chk-grid-long-close-level",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_LONG_CLOSE_MIN_LEVEL_OPTIONS],
                value=WT_LONG_CLOSE_MIN_LEVEL_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Short open level H1"),
            dcc.Checklist(id="chk-grid-short-zone",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_SHORT_ENTRY_MIN_BELOW_ZERO_OPTIONS],
                value=WT_SHORT_ENTRY_MIN_BELOW_ZERO_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Short close level H1"),
            dcc.Checklist(id="chk-grid-short-close-level",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_SHORT_CLOSE_MAX_LEVEL_OPTIONS],
                value=WT_SHORT_CLOSE_MAX_LEVEL_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Long open level H4"),
            dcc.Checklist(id="chk-grid-h4-long",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_H4_LONG_FILTER_MAX_OPTIONS],
                value=WT_H4_LONG_FILTER_MAX_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Long close level H4"),
            dcc.Checklist(id="chk-grid-h4-long-close",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_H4_LONG_CLOSE_MIN_OPTIONS],
                value=WT_H4_LONG_CLOSE_MIN_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Short open level H4"),
            dcc.Checklist(id="chk-grid-h4-short",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_H4_SHORT_FILTER_MIN_OPTIONS],
                value=WT_H4_SHORT_FILTER_MIN_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            sec("Short close level H4"),
            dcc.Checklist(id="chk-grid-h4-short-close",
                options=[{"label":f" {v:.1f}","value":v} for v in WT_H4_SHORT_CLOSE_MAX_OPTIONS],
                value=WT_H4_SHORT_CLOSE_MAX_GRID, inline=True,
                inputStyle={"marginRight":"4px","accentColor":C["blue"]},
                labelStyle={"color":"#e8eaf6","fontSize":"12px","marginRight":"10px"}),
            html.Div(
                "WFO w BEE7 testuje long i short osobno. Zeby nie mnozyc siatki do milionow kombinacji, short jest liczony jako kilka niezaleznych profili open/close H1/H4.",
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

        html.Div([
            sec("Zapis / odczyt wyników"),
            html.Div([
                html.Button("Zapisz wynik", id="btn-save-result", n_clicks=0, style={
                    "flex":"1","background":C["surf2"],"border":f"1px solid {C['border']}",
                    "borderRadius":"10px","color":C["text"],"padding":"9px 10px",
                    "fontSize":"12px","fontWeight":"600","cursor":"pointer"}),
                html.Button("Odśwież", id="btn-refresh-results", n_clicks=0, style={
                    "width":"88px","background":"transparent","border":f"1px solid {C['border']}",
                    "borderRadius":"10px","color":C["muted"],"padding":"9px 10px",
                    "fontSize":"12px","fontWeight":"600","cursor":"pointer"}),
            ], style={"display":"flex","gap":"8px","marginBottom":"10px"}),
            dcc.Dropdown(
                id="saved-result-select",
                options=saved_opts,
                value=(saved_opts[0]["value"] if saved_opts else None),
                clearable=False,
                placeholder="Brak zapisanych wyników",
                className="wt-drp",
                style={"color":"#0b1220"},
            ),
            html.Button("Wczytaj wybrany plik", id="btn-load-result", n_clicks=0, style={
                "width":"100%","background":C["blue"],"border":"none","borderRadius":"10px",
                "color":"#fff","padding":"10px","fontSize":"12px","fontWeight":"700",
                "cursor":"pointer","marginTop":"10px"}),
            html.Button("Raport PDF", id="btn-report-pdf", n_clicks=0, style={
                "width":"100%","background":C["green"],"border":"none","borderRadius":"10px",
                "color":"#07111b","padding":"10px","fontSize":"12px","fontWeight":"800",
                "cursor":"pointer","marginTop":"8px"}),
            html.Div(id="saved-result-status", style={
                "fontSize":"11px","color":C["muted"],"textAlign":"center",
                "minHeight":"16px","marginTop":"8px","lineHeight":"1.4"}),
        ], style={**card_s, "padding":"14px 16px"}),

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
        dcc.Store(id="store-chart-view",     data="standard"),
        dcc.Store(id="store-chart-payload",  data=None),
        dcc.Store(id="store-server-token",   data=None, storage_type="session"),
        dcc.Download(id="download-trades"),
        dcc.Download(id="download-pine-trades"),
        dcc.Download(id="download-report"),
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
    bt_short_close_level,
    bt_h4_long,
    bt_h4_short,
    bt_h4_short_close,
    bt_long_close_level,
    bt_h4_long_close,
    bt_long_tp1_pct,
    bt_long_sl_pct,
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
    grid_short_close_level,
    grid_h4_long,
    grid_h4_short,
    grid_h4_short_close,
    grid_long_close_level,
    grid_h4_long_close,
    grid_long_sl_pct,
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
            ss(
                running=False,
                status=f"✗ Za mało danych po filtrowaniu: zostało {len(df)} świec, minimum 100.",
                progress="Sprawdź zakres dat, timeframe albo czy CSV po pobraniu zawiera wybrany okres.",
            )
            return

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
            bt_short_close_level,
            bt_h4_long,
            bt_h4_short,
            bt_h4_short_close,
            bt_long_close_level,
            bt_h4_long_close,
            bt_long_tp1_pct,
            bt_long_sl_pct,
            fee_rate_val,
            slip_bps_val,
        )

        if run_mode == "backtest":
            bt_started_at = _time.time()
            ss(
                status="Backtest w toku...",
                progress=f"Przetwarzam {len(df)} świec  |  czas 0s  |  ETA liczę...",
            )
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
            bt_elapsed = _fmt_duration(_time.time() - bt_started_at)
            ss(
                running=False,
                stop=False,
                result=result,
                status=f"✓ Backtest gotowy  |  {n} tradów  |  {ret:+.2f}%",
                progress=f"Czas wykonania: {bt_elapsed}",
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
            grid_short_close_level,
            grid_h4_long,
            grid_h4_short,
            grid_h4_short_close,
            grid_long_close_level,
            grid_h4_long_close,
            grid_long_sl_pct,
        )

        ob, lb = wfo_bars(tf, opt_days_val, live_days_val)
        total  = max(0, (len(df)-ob)//lb)
        combo_total = _grid_combo_count(grid_overrides)
        wfo_started_at = _time.time()
        progress_total_windows = max(total, 1)
        progress_combo_total = max(combo_total, 1)
        total_progress_units = max(1, progress_total_windows * progress_combo_total)

        def _wfo_progress_text(wid, total_windows, combo_idx, combo_count):
            total_windows = max(int(total_windows or 0), 1)
            combo_count = max(int(combo_count or 0), 1)
            window_idx = max(0, min(int(wid or 0), total_windows - 1))
            combo_idx = max(0, min(int(combo_idx or 0), combo_count))
            done_units = min(total_progress_units, window_idx * combo_count + combo_idx)
            elapsed, eta = _elapsed_eta(wfo_started_at, done_units, total_progress_units)
            window_pct = combo_idx / combo_count * 100.0
            total_pct = done_units / total_progress_units * 100.0
            return (
                f"Okno {window_idx + 1} / {total_windows}  |  "
                f"kombinacja {combo_idx} / {combo_count}  |  "
                f"{window_pct:.0f}% okna  |  całość {total_pct:.1f}%  |  "
                f"czas {elapsed}  |  ETA {eta}"
            )

        ss(
            status=f"WFO w toku  |  ~{total} okien  |  {combo_total} kombinacji/okno",
            progress=_wfo_progress_text(0, progress_total_windows, 0, progress_combo_total),
        )

        def _on_combo_progress(wid, total_windows, combo_idx, combo_total):
            ss(progress=_wfo_progress_text(wid, total_windows, combo_idx, combo_total))

        def _on_window_done(wid, total, wstats, trades_list, equity_sofar, cap_sofar):
            ss(progress=_wfo_progress_text(wid, total, combo_total, combo_total))
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
        wfo_elapsed = _fmt_duration(_time.time() - wfo_started_at)
        if stopped:
            ss(
                running=False,
                stop=False,
                result=result,
                status=f"Zatrzymano  |  ukończone okna {nw}/{total}  |  {n} tradów  |  {ret:+.2f}%",
                progress=f"Czas wykonania: {wfo_elapsed}",
            )
            return

        ss(running=False, stop=False, result=result,
           status=f"✓ WFO gotowe  |  {nw} okien  |  {n} tradów  |  {ret:+.2f}%"
                  + ("  |  zapisano best params" if best_params else ""),
           progress=f"Czas wykonania: {wfo_elapsed}")

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


@app.callback(
    Output("store-chart-view","data"),
    Input("chart-view-radio","value"),
    prevent_initial_call=True,
)
def on_chart_view(value):
    return _chart_view_mode(value)

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


@app.callback(
    Output("download-pine-trades", "data"),
    Input("btn-export-pine-trades", "n_clicks"),
    State("store-result", "data"),
    prevent_initial_call=True,
)
def export_trades_pine(n_clicks, result_data):
    if not n_clicks or not result_data:
        return dash.no_update

    script = _build_pine_trades_overlay(result_data)
    if not script.strip():
        return dash.no_update

    mode = str(result_data.get("mode", "run")).lower()
    symbol = str(result_data.get("symbol", "asset")).upper()
    tf = str(result_data.get("tf", "tf")).lower()
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"bee7_{mode}_{symbol}_{tf}_trades_overlay_{stamp}.pine"
    return dcc.send_string(script, filename)


@app.callback(
    Output("download-report", "data"),
    Output("saved-result-status", "children", allow_duplicate=True),
    Input("btn-report-pdf", "n_clicks"),
    State("store-result", "data"),
    prevent_initial_call=True,
)
def export_report_pdf(n_clicks, result_data):
    if not n_clicks:
        return dash.no_update, dash.no_update
    if not result_data:
        return dash.no_update, "Brak wyniku do raportu."
    try:
        pdf_bytes = _build_report_pdf(result_data)
        filename = _report_filename(result_data)
        return dcc.send_bytes(lambda buffer: buffer.write(pdf_bytes), filename), f"Wygenerowano raport: {filename}"
    except Exception as exc:
        return dash.no_update, f"Błąd raportu PDF: {exc}"


@app.callback(
    Output("saved-result-select", "options"),
    Output("saved-result-select", "value"),
    Output("saved-result-status", "children"),
    Input("btn-save-result", "n_clicks"),
    Input("btn-refresh-results", "n_clicks"),
    State("store-result", "data"),
    State("saved-result-select", "value"),
    prevent_initial_call=True,
)
def save_or_refresh_result(save_clicks, refresh_clicks, result_data, current_value):
    triggered = ctx.triggered_id
    if triggered == "btn-save-result":
        if not result_data:
            return dash.no_update, dash.no_update, "Brak wyniku do zapisu."
        try:
            filename = _save_result_file(result_data)
            options = _saved_result_options()
            return options, filename, f"Zapisano: {filename}"
        except Exception as exc:
            return dash.no_update, dash.no_update, f"Błąd zapisu: {exc}"

    options = _saved_result_options()
    values = {opt["value"] for opt in options}
    value = current_value if current_value in values else (options[0]["value"] if options else None)
    msg = "Lista zapisów odświeżona." if options else "Brak zapisanych wyników."
    return options, value, msg


@app.callback(
    Output("store-result", "data", allow_duplicate=True),
    Output("store-selected-trade", "data", allow_duplicate=True),
    Output("saved-result-status", "children", allow_duplicate=True),
    Input("btn-load-result", "n_clicks"),
    State("saved-result-select", "value"),
    prevent_initial_call=True,
)
def load_saved_result(n_clicks, filename):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    if not filename:
        return dash.no_update, dash.no_update, "Wybierz zapisany plik."
    try:
        result = _load_result_file(filename)
        mode = str(result.get("mode", "wynik")).upper()
        symbol = str(result.get("symbol", ""))
        tf = str(result.get("tf", ""))
        ret = float(result.get("stats", {}).get("net_return_pct", 0.0) or 0.0)
        ss(
            running=False,
            stop=False,
            result=result,
            status=f"✓ Wczytano zapisany wynik  |  {mode} {symbol} {tf}  |  {ret:+.2f}%",
            progress="",
        )
        return result, None, f"Wczytano: {filename}"
    except Exception as exc:
        return dash.no_update, dash.no_update, f"Błąd wczytywania: {exc}"

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
    State("inp-bt-short-close-level","value"),
    State("inp-bt-h4-long","value"), State("inp-bt-h4-short","value"),
    State("inp-bt-h4-short-close","value"),
    State("inp-bt-long-close-level","value"),
    State("inp-bt-h4-long-close","value"),
    State("inp-bt-long-tp1-pct","value"),
    State("inp-bt-long-sl-pct","value"),
    State("chk-grid-channel","value"), State("chk-grid-avg","value"),
    State("chk-grid-signal","value"), State("chk-grid-min-level","value"),
    State("chk-grid-reentry","value"), State("chk-grid-ema-filter","value"),
    State("chk-grid-htf-filter","value"),
    State("chk-grid-ema-len","value"),
    State("chk-grid-long-zone","value"), State("chk-grid-short-zone","value"),
    State("chk-grid-short-close-level","value"),
    State("chk-grid-h4-long","value"), State("chk-grid-h4-short","value"),
    State("chk-grid-h4-short-close","value"),
    State("chk-grid-long-close-level","value"),
    State("chk-grid-h4-long-close","value"),
    State("chk-grid-long-sl-pct","value"),
    prevent_initial_call=True,
)
def on_run_stop(nr, ns,
    sym, tf, mkt, frm, to, cap,
    run_mode, direction, fee, slip, opt, live, score,
    bt_channel, bt_avg, bt_signal, bt_min_level,
    bt_reentry, bt_ema_filter, bt_htf_filter, bt_ema_len,
    bt_long_zone, bt_short_zone, bt_short_close_level, bt_h4_long, bt_h4_short, bt_h4_short_close,
    bt_long_close_level, bt_h4_long_close, bt_long_tp1_pct, bt_long_sl_pct,
    grid_channel, grid_avg, grid_signal, grid_min_level,
    grid_reentry, grid_ema_filter, grid_htf_filter, grid_ema_len,
    grid_long_zone, grid_short_zone, grid_short_close_level, grid_h4_long, grid_h4_short, grid_h4_short_close,
    grid_long_close_level, grid_h4_long_close, grid_long_sl_pct):

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
            bt_reentry, bt_ema_filter, bt_htf_filter, bt_ema_len,
            bt_long_zone, bt_short_zone, bt_short_close_level, bt_h4_long, bt_h4_short, bt_h4_short_close,
            bt_long_close_level, bt_h4_long_close, bt_long_tp1_pct, bt_long_sl_pct,
            grid_channel, grid_avg, grid_signal, grid_min_level,
            grid_reentry, grid_ema_filter, grid_htf_filter, grid_ema_len,
            grid_long_zone, grid_short_zone, grid_short_close_level, grid_h4_long, grid_h4_short, grid_h4_short_close,
            grid_long_close_level, grid_h4_long_close, grid_long_sl_pct,
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
    direction_df = _direction_stats_from_result(result_data)
    long_stats = _direction_row(direction_df, "long")
    short_stats = _direction_row(direction_df, "short")

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
        mcrd("Long / Short", f"{int(long_stats['trades'])} / {int(short_stats['trades'])}", "liczba trade'ów"),
        mcrd(
            "Long W/L",
            f"{int(long_stats['wins'])}/{int(long_stats['losses'])}",
            f"{_money(long_stats['win_pnl_usd'])} / {_money(long_stats['loss_pnl_usd'])}",
            C["green"] if float(long_stats["net_pnl_usd"]) >= 0 else C["red"],
        ),
        mcrd(
            "Short W/L",
            f"{int(short_stats['wins'])}/{int(short_stats['losses'])}",
            f"{_money(short_stats['win_pnl_usd'])} / {_money(short_stats['loss_pnl_usd'])}",
            C["green"] if float(short_stats["net_pnl_usd"]) >= 0 else C["red"],
        ),
        mcrd("Return/DD", metric_or_na(rdd), "net return / max DD", metric_color(rdd)),
        mcrd("Sharpe", metric_or_na(sharpe), f"Sortino: {metric_or_na(sortino)}"),
        mcrd("R:R avg", metric_or_na(rr), "avg win / avg loss"),
        mcrd("Consistency", metric_or_na(consistency, "{:.0f}%"), f"stability: {metric_or_na(stability)}"),
    ]


def _legend_chip(label: str, color: str, note: str) -> html.Span:
    return html.Span([
        html.Span(style={
            "display": "inline-block",
            "width": "9px",
            "height": "9px",
            "borderRadius": "999px",
            "background": color,
            "marginRight": "6px",
            "boxShadow": f"0 0 0 2px {color}33",
        }),
        html.Span(label, style={"fontWeight": "700", "color": C["text"]}),
        html.Span(f" - {note}", style={"color": C["muted"]}),
    ], className="chart-legend-chip")


def _build_chart_tab(
    selected_trade: dict | None,
    chart_filter_val: str | None,
    chart_view_val: str | None,
) -> html.Div:
    chart_filt = chart_filter_val or "all"
    chart_view = _chart_view_mode(chart_view_val)
    filter_opts = [
        {"label": "Wszystkie", "value": "all"},
        {"label": "Long", "value": "long"},
        {"label": "Short", "value": "short"},
        {"label": "Reverse", "value": "REVERSE"},
        {"label": "Force", "value": "FORCE"},
    ]
    return html.Div([
        html.Div([
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
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
            html.Div([
                html.Span(
                    "Widok: ",
                    style={"fontSize": "11px", "color": C["muted"], "marginRight": "8px", "lineHeight": "28px"},
                ),
                dcc.RadioItems(
                    id="chart-view-radio",
                    options=CHART_VIEW_OPTIONS,
                    value=chart_view,
                    inline=True,
                    inputStyle={"marginRight": "3px", "accentColor": C["amber"]},
                    labelStyle={"color": C["text"], "fontSize": "12px", "marginRight": "12px", "cursor": "pointer"},
                ),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "gap": "14px",
            "flexWrap": "wrap",
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
            html.Div([
                _legend_chip("pełny punkt", C["green"], "realne wejście long / short na wykresie ceny"),
                _legend_chip("żółty", C["amber"], "odrzucone przez filtr H4"),
                _legend_chip("niebieski", C["blue"], "odrzucone przez strefę H1"),
                _legend_chip("szary", C["muted"], "sygnał pojawił się przy otwartej pozycji"),
                _legend_chip("pomarańczowy", "#fb923c", "wyłączony kierunek long/short"),
            ], className="chart-legend"),
            html.Div(
                "Górny panel pokazuje cenę, linię EMA filtra i markery trade'ów. Dolny panel ma tę samą "
                "wysokość i pokazuje WT1/WT2 dla H1 oraz H4. W trybie Diagnostyka kropki H1 pokazują również "
                "sygnały odrzucone; najedź na marker L?/S?, aby zobaczyć powód.",
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
    Input("store-chart-view","data"),
    Input("store-selected-trade","data"),
    prevent_initial_call=True,
)
def render_results(tab, result_data, chart_filter_val, chart_view_val, selected_trade_val):
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
                view_mode=chart_view_val or "standard",
            )
            return chart_payload, metrics, _build_chart_tab(selected_trade_val, chart_filter_val, chart_view_val)

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
                    html.Div([
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
                        html.Button(
                            "Eksport Pine",
                            id="btn-export-pine-trades",
                            n_clicks=0,
                            title="Eksport Pine Script z markerami transakcji",
                            style={
                                "background": C["blue"],
                                "color": "#07111b",
                                "border": "none",
                                "borderRadius": "12px",
                                "padding": "10px 14px",
                                "fontSize": "12px",
                                "fontWeight": "800",
                                "cursor": "pointer",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
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
            direction_df = _direction_stats_from_result(r)
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
                mini(direction_df, "Long / Short - wygrane i przegrane"),
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
                view_mode=chart_view_val or "standard",
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
                        "Górny panel pokazuje cenę, linię EMA filtra i markery trade'ów. Dolny panel ma tę samą "
                        "wysokość i pokazuje WT1/WT2 dla H1 oraz H4 z zielonymi i czerwonymi kropkami przecięć.",
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
    print(f"\n{'='*50}\n  Bee7 WaveTrend Dashboard  ->  http://{a.host}:{a.port}\n{'='*50}\n")
    app.run(host=a.host, port=a.port, debug=a.debug)


