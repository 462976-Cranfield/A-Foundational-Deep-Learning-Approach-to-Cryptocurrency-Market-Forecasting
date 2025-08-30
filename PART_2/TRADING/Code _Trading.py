"""
CODE PART2
EVALUATION OF THE TRADING METHOD
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from importlib import import_module
from typing import List, Tuple, Dict, Iterator, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# =========================
# ==== CONFIG SETUP =======
# =========================

FEATURE_COLS: List[str] = [
    "open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_signal",
    "macd_histogram", "macd_divergence", "macd_slope", "tenkan_sen", "kijun_sen",
    "senkou_span_a", "senkou_span_b", "bollinger_mavg", "bollinger_hband",
    "bollinger_lband", "bollinger_width",
]
TARGET_COLS: List[str] = ["Y_1h", "Y_4h", "Y_12h", "Y_24h", "Y_48h"]

SEQ_H1_DEF = 100
SEQ_H4_DEF = 25
SEQ_D_DEF  = 30

DEF_CONTEXT_DAYS_H1 = 300
DEF_CONTEXT_DAYS_H4 = 300
DEF_CONTEXT_DAYS_D1 = 300

ANCHOR_STEP_HOURS = 24
MIN_ROWS_FOR_SCALER = 10

HORIZON_NAMES_DEFAULT = ["1h", "4h", "12h", "24h", "48h"]

# ==============================
# ==== SCALER & IO HELPERS =====
# ==============================

@dataclass
class RobustScaler:
    median_: np.ndarray
    scale_: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "RobustScaler":
        Xv = X.values.astype(np.float64)
        med = np.nanmedian(Xv, axis=0)
        q75 = np.nanpercentile(Xv, 75, axis=0)
        q25 = np.nanpercentile(Xv, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        return cls(median_=med, scale_=iqr)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X.values.astype(np.float64)
        Z = (Xv - self.median_) / (self.scale_ + self.eps)
        return Z.astype(np.float32)

def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["timestamp", "datetime", "time", "DateTime", "Datetime", "date", "Date"]:
        if c in df.columns:
            return c
    raise ValueError("Aucune colonne temporelle trouvée (ex: 'timestamp'/'datetime'/'date').")

def _to_utc_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)

def _load_csv(path: str, expect_interval: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    tcol = _detect_time_col(df)
    df[tcol] = _to_utc_naive(df[tcol])
    df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    df = df.rename(columns={tcol: "timestamp"})
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes features manquantes pour {expect_interval}: {missing_cols}")
    return df

def load_datasets(h1_csv: str, h4_csv: str, d1_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h1 = _load_csv(h1_csv, "1h")
    h4 = _load_csv(h4_csv, "4h")
    d1 = _load_csv(d1_csv, "1d")
    return h1, h4, d1

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ==========================================
# ==== FENÊTRES, SCALERS, ANCHORS (SETUP) ===
# ==========================================

def _seq_time_requirement(bars: int, freq: str) -> pd.Timedelta:
    if freq == "1h":
        return pd.Timedelta(hours=bars)
    if freq == "4h":
        return pd.Timedelta(hours=4 * bars)
    if freq == "1d":
        return pd.Timedelta(days=bars)
    raise ValueError("freq must be '1h'|'4h'|'1d'")

def _context_time_requirement(days: int) -> pd.Timedelta:
    return pd.Timedelta(days=days)

def _ensure_window(df: pd.DataFrame, t_end: pd.Timestamp, bars: int, freq: str) -> pd.DataFrame:
    if freq == "1h":
        t_start = t_end - pd.Timedelta(hours=bars)
    elif freq == "4h":
        t_start = t_end - pd.Timedelta(hours=4 * bars)
    elif freq == "1d":
        t_start = t_end - pd.Timedelta(days=bars)
    else:
        raise ValueError("freq must be '1h'|'4h'|'1d'")
    win = df[(df["timestamp"] >= t_start) & (df["timestamp"] < t_end)]
    return win.tail(bars)

def _fit_scaler(df: pd.DataFrame, t_end: pd.Timestamp, mode: str, context_days: Optional[int]) -> RobustScaler:
    if mode not in {"rolling", "expanding"}:
        raise ValueError("mode must be 'rolling' or 'expanding'")
    if mode == "expanding":
        ctx = df[df["timestamp"] < t_end]
    else:
        days = int(context_days or DEF_CONTEXT_DAYS_D1)
        t_start = t_end - pd.Timedelta(days=days)
        ctx = df[(df["timestamp"] >= t_start) & (df["timestamp"] < t_end)]
    ctx = ctx.dropna(subset=FEATURE_COLS)
    if len(ctx) < MIN_ROWS_FOR_SCALER:
        raise RuntimeError(f"Pas assez de données pour fitter le scaler (len={len(ctx)}) jusqu'à {t_end} (mode={mode}).")
    return RobustScaler.fit(ctx[FEATURE_COLS])

def _earliest_anchor_allowed(
    h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
    seq_h1: int, seq_h4: int, seq_d: int,
    mode: str,
    ctx_days_h1: int, ctx_days_h4: int, ctx_days_d1: int,
) -> pd.Timestamp:
    tmin_h1 = h1["timestamp"].min()
    tmin_h4 = h4["timestamp"].min()
    tmin_d1 = d1["timestamp"].min()
    req_h1 = _seq_time_requirement(seq_h1, "1h") + (pd.Timedelta(0) if mode == "expanding" else _context_time_requirement(ctx_days_h1))
    req_h4 = _seq_time_requirement(seq_h4, "4h") + (pd.Timedelta(0) if mode == "expanding" else _context_time_requirement(ctx_days_h4))
    req_d1 = _seq_time_requirement(seq_d,  "1d") + (pd.Timedelta(0) if mode == "expanding" else _context_time_requirement(ctx_days_d1))
    earliest = max(tmin_h1 + req_h1, tmin_h4 + req_h4, tmin_d1 + req_d1)
    return pd.to_datetime(earliest).normalize()

def build_anchor_sequences(
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    t_anchor: pd.Timestamp,
    norm_mode: str = "rolling",
    context_days_h1: int = DEF_CONTEXT_DAYS_H1,
    context_days_h4: int = DEF_CONTEXT_DAYS_H4,
    context_days_d1: int = DEF_CONTEXT_DAYS_D1,
    seq_h1: int = SEQ_H1_DEF,
    seq_h4: int = SEQ_H4_DEF,
    seq_d: int  = SEQ_D_DEF,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc_h1 = _fit_scaler(h1, t_anchor, mode=norm_mode, context_days=context_days_h1)
    sc_h4 = _fit_scaler(h4, t_anchor, mode=norm_mode, context_days=context_days_h4)
    sc_d1 = _fit_scaler(d1, t_anchor, mode=norm_mode, context_days=context_days_d1)

    seq1 = _ensure_window(h1, t_anchor, seq_h1, "1h")
    seq4 = _ensure_window(h4, t_anchor, seq_h4, "4h")
    seqd = _ensure_window(d1, t_anchor, seq_d,  "1d")

    if len(seq1) != seq_h1 or len(seq4) != seq_h4 or len(seqd) != seq_d:
        raise RuntimeError(
            f"Pas assez de lignes à l'anchor {t_anchor}: H1={len(seq1)}/{seq_h1}, H4={len(seq4)}/{seq_h4}, D1={len(seqd)}/{seq_d}"
        )

    X1 = sc_h1.transform(seq1[FEATURE_COLS])
    X4 = sc_h4.transform(seq4[FEATURE_COLS])
    XD = sc_d1.transform(seqd[FEATURE_COLS])

    return X1, X4, XD

def make_daily_anchors(
    h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
    start: Optional[str],
    days: int,
    norm_mode: str,
    ctx_days_h1: int, ctx_days_h4: int, ctx_days_d1: int,
    seq_h1: int, seq_h4: int, seq_d: int,
) -> List[pd.Timestamp]:
    earliest = _earliest_anchor_allowed(
        h1, h4, d1, seq_h1, seq_h4, seq_d, mode=norm_mode,
        ctx_days_h1=ctx_days_h1, ctx_days_h4=ctx_days_h4, ctx_days_d1=ctx_days_d1
    )
    t0 = max(pd.to_datetime(start).normalize(), earliest) if start else earliest
    tmax = min(h1["timestamp"].max(), h4["timestamp"].max(), d1["timestamp"].max())

    anchors: List[pd.Timestamp] = []
    t = t0
    for _ in range(days * 2):
        if t > tmax:
            break
        anchors.append(t)
        t = t + pd.Timedelta(hours=ANCHOR_STEP_HOURS)
        if len(anchors) >= days:
            break
    return anchors

def get_eval_iter(
    datasets_dir: str,
    n_days: int,
    start_date: Optional[str] = None,
    norm_mode: str = "rolling",
    context_days_h1: int = DEF_CONTEXT_DAYS_H1,
    context_days_h4: int = DEF_CONTEXT_DAYS_H4,
    context_days_d1: int = DEF_CONTEXT_DAYS_D1,
    seq_h1: int = SEQ_H1_DEF,
    seq_h4: int = SEQ_H4_DEF,
    seq_d: int  = SEQ_D_DEF,
) -> Iterator[Tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]]:
    h1_csv = os.path.join(datasets_dir, "btc_h1_test.csv")
    h4_csv = os.path.join(datasets_dir, "btc_h4_test.csv")
    d1_csv = os.path.join(datasets_dir, "btc_1d_test.csv")
    h1, h4, d1 = load_datasets(h1_csv, h4_csv, d1_csv)

    anchors = make_daily_anchors(
        h1, h4, d1,
        start=start_date,
        days=n_days,
        norm_mode=norm_mode,
        ctx_days_h1=context_days_h1,
        ctx_days_h4=context_days_h4,
        ctx_days_d1=context_days_d1,
        seq_h1=seq_h1, seq_h4=seq_h4, seq_d=seq_d,
    )

    produced = 0
    for t_anchor in anchors:
        try:
            X1, X4, XD = build_anchor_sequences(
                h1, h4, d1, t_anchor,
                norm_mode=norm_mode,
                context_days_h1=context_days_h1,
                context_days_h4=context_days_h4,
                context_days_d1=context_days_d1,
                seq_h1=seq_h1, seq_h4=seq_h4, seq_d=seq_d
            )
        except Exception:
            continue
        yield (t_anchor, X1, X4, XD)
        produced += 1
        if produced >= n_days:
            break

# =====================================
# ==== UTILS BACKTEST & PREDICTION ====
# =====================================

def softmax_numpy(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / np.sum(e)
    return s

def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device=device, dtype=torch.float32, non_blocking=True)

def load_models(
    ckpt_paths: List[str],
    model_module: str,
    model_class: str,
    device: torch.device,
) -> List[pl.LightningModule]:
    print(f"[load_models] Import '{model_module}.{model_class}' ...")
    if model_module in (None, "", "__local__", "local"):
        cls = globals().get(model_class, None)
        if cls is None:
            raise RuntimeError(
                f"Classe '{model_class}' introuvable dans live_eval.py. "
                f"Assure-toi d'avoir collé ta classe (CryptoTFTModel) plus haut dans ce fichier."
            )
        mod = None
    else:
        mod = import_module(model_module)
        cls = getattr(mod, model_class)

    models: List[pl.LightningModule] = []
    for p in ckpt_paths:
        print(f"[load_models] Loading checkpoint: {p}")
        m = cls.load_from_checkpoint(p, map_location=device)
        m.eval()
        m.to(device)
        models.append(m)
    print(f"[load_models] {len(models)} modèles chargés.")
    return models

@torch.inference_mode()
def _predict_one_model(
    model: pl.LightningModule,
    x_h1: np.ndarray,
    x_h4: np.ndarray,
    x_d: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    th1 = to_device(torch.from_numpy(np.asarray(x_h1)), device).unsqueeze(0)
    th4 = to_device(torch.from_numpy(np.asarray(x_h4)), device).unsqueeze(0)
    td  = to_device(torch.from_numpy(np.asarray(x_d)),  device).unsqueeze(0)

    out = None
    batch_dict = {"x_h1": th1, "x_h4": th4, "x_d": td}

    if hasattr(model, "predict_step"):
        try: out = model.predict_step(batch_dict, 0)
        except Exception: out = None
    if out is None:
        try: out = model(th1, th4, td)
        except Exception: out = None
    if out is None:
        try: out = model(batch_dict)
        except Exception: out = None
    if out is None and hasattr(model, "predict"):
        try: out = model.predict(batch_dict)
        except Exception: out = None
    if out is None:
        raise RuntimeError("Impossible d'exécuter l'inférence : adapte _predict_one_model à l'API de ton modèle.")

    if isinstance(out, (list, tuple)): out = out[0]
    if isinstance(out, torch.Tensor):
        out = out.detach().float().cpu().numpy()
    out = np.asarray(out).reshape(-1)
    return np.clip(out, -1.0, 1.0)

def predict_models(
    models: List[pl.LightningModule],
    x_h1: np.ndarray,
    x_h4: np.ndarray,
    x_d: np.ndarray,
) -> np.ndarray:
    preds = [ _predict_one_model(m, x_h1, x_h4, x_d) for m in models ]
    return np.stack(preds, axis=0)

def aggregate_preds(preds: np.ndarray, method: str = "mean") -> np.ndarray:
    if method == "mean":   return preds.mean(axis=0)
    if method == "median": return np.median(preds, axis=0)
    raise ValueError("aggregate_preds.method must be 'mean' or 'median'")

def consensus_gating(y_hat: np.ndarray, tau: float = 0.15) -> Tuple[float, np.ndarray]:
    if tau <= 0: raise ValueError("tau must be > 0")
    logits = np.abs(y_hat) / float(tau)
    w = softmax_numpy(logits)
    s = float(np.sum(w * y_hat))
    return s, w

# ==============================
# ==== PRICES & SIGNALS DF =====
# ==============================

def _parse_h1_prices(
    path_csv: str,
    datetime_col: str = "datetime",
    price_col: str = "close",
) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    if datetime_col not in df.columns:
        for c in ["timestamp", "time", "date", "Date", "Datetime", "DateTime"]:
            if c in df.columns:
                datetime_col = c
                break
    if datetime_col not in df.columns:
        raise ValueError(f"Colonne datetime introuvable dans {path_csv}")
    if price_col not in df.columns:
        raise ValueError(f"Colonne prix '{price_col}' introuvable dans {path_csv}")

    df[datetime_col] = _to_utc_naive(df[datetime_col])
    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df = df.set_index(datetime_col)
    return df[[price_col]].rename(columns={price_col: "close"})

def _align_to_next_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).floor("h") + pd.Timedelta(hours=1)

def _build_signals_df(
    anchors: List[Dict],
    thr: float,
    horizon_names: List[str],
    tp_horizon: str,
    confirmation_threshold: float,
) -> pd.DataFrame:
    rows = []
    for a in anchors:
        anchor_ts: pd.Timestamp = a["anchor_ts"]
        y_hat: np.ndarray = a["y_hat"]
        w: np.ndarray = a["w"]
        s: float = a["s"]

        # Get primary horizon prediction (e.g., 24h)
        tp_horizon_idx = horizon_names.index(tp_horizon) if tp_horizon in horizon_names else 0
        yhat_primary = y_hat[tp_horizon_idx]

        # Confirmation: Check if >= confirmation_threshold of other horizons agree
        other_horizons = [i for i, h in enumerate(horizon_names) if h != tp_horizon]
        if other_horizons:
            other_preds = y_hat[other_horizons]
            agree_count = np.sum((other_preds > 0) == (yhat_primary > 0))
            agree_ratio = agree_count / len(other_horizons)
            if agree_ratio < confirmation_threshold:
                signal = "flat"
                sig_val = 0
            else:
                if s > thr:
                    signal = "long"
                    sig_val = 1
                elif s < -thr:
                    signal = "short"
                    sig_val = -1
                else:
                    signal = "flat"
                    sig_val = 0
        else:
            if s > thr:
                signal = "long"
                sig_val = 1
            elif s < -thr:
                signal = "short"
                sig_val = -1
            else:
                signal = "flat"
                sig_val = 0

        row = {
            "datetime_anchor": anchor_ts,
            "score_s": s,
            "signal": signal,
            "signal_val": sig_val,
            "datetime_exec": _align_to_next_hour(anchor_ts),
            "predicted_tp_level": float(yhat_primary),  # Will scale with entry_price later
        }
        for i, h in enumerate(horizon_names):
            row[f"yhat_{h}"] = float(y_hat[i])
            row[f"w_{h}"] = float(w[i])

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("datetime_anchor").reset_index(drop=True)
    return df

# =====================================
# ==== TRADES, SIMU, METRICS, PLOTS ===
# =====================================

def _build_trades_and_position(
    signals_df: pd.DataFrame,
    h1_prices: pd.DataFrame,
    fee_rate: float,
    horizon_names: List[str],
    tp_horizon: str,
    tp_scaling: float,
    sl_percent: float,
    trailing_percent: float,
    use_trailing: bool,
    max_hold_hours: int,
    opposing_signal_multiplier: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, float]]:
    if signals_df.empty:
        raise ValueError("signals_df est vide.")

    exec_times = pd.to_datetime(signals_df["datetime_exec"])
    t0, tN = exec_times.min(), exec_times.max()

    h1 = h1_prices.loc[(h1_prices.index >= t0) & (h1_prices.index <= tN)].copy()
    if h1.empty:
        raise ValueError("Fenêtre H1 vide pour la période des anchors. Vérifie les timestamps.")

    pos  = pd.Series(0, index=h1.index, dtype=float)
    fees = pd.Series(0.0, index=h1.index, dtype=float)

    trades = []
    current_pos = 0
    open_anchor_ts = None
    open_exec_ts = None
    open_price = None
    open_weights = None
    open_signal_row = None
    open_tp_level = None
    open_sl_level = None
    trailing_sl = None
    open_time = None

    sig_by_exec = {pd.Timestamp(r["datetime_exec"]): r for _, r in signals_df.iterrows()}
    horizon_pnl = {h: 0.0 for h in horizon_names}
    exit_reasons = []

    for ts in h1.index:
        current_price = float(h1.loc[ts, "close"])

        # Check if current position should be closed (TP/SL/Time/Opposing)
        if current_pos != 0 and open_price is not None:
            hours_held = int((ts - open_exec_ts).total_seconds() // 3600) if open_exec_ts else 0
            price_return = current_pos * (current_price / open_price - 1.0)  # Positive for profitable LONG/SHORT
            should_close = False
            exit_reason = None

            # TP Check
            if current_pos > 0:  # LONG
                if current_price >= open_tp_level:
                    should_close = True
                    exit_reason = "tp_hit"
                elif current_price <= open_sl_level:
                    should_close = True
                    exit_reason = "sl_hit"
            elif current_pos < 0:  # SHORT
                if current_price <= open_tp_level:
                    should_close = True
                    exit_reason = "tp_hit"
                elif current_price >= open_sl_level:
                    should_close = True
                    exit_reason = "sl_hit"

            # Trailing Stop
            if use_trailing and not should_close:
                if current_pos > 0:  # LONG
                    trailing_sl = max(trailing_sl, current_price * (1 - trailing_percent))
                    if current_price <= trailing_sl:
                        should_close = True
                        exit_reason = "trailing_sl_hit"
                elif current_pos < 0:  # SHORT
                    trailing_sl = min(trailing_sl, current_price * (1 + trailing_percent))
                    if current_price >= trailing_sl:
                        should_close = True
                        exit_reason = "trailing_sl_hit"

            # Time Limit
            if hours_held >= max_hold_hours and not should_close:
                should_close = True
                exit_reason = "max_hold"

            # Close if necessary
            if should_close:
                exit_price = current_price
                pnl_gross = current_pos * (exit_price / open_price - 1.0)
                close_fee = fee_rate * abs(current_pos)
                open_fee  = fee_rate * abs(current_pos)
                pnl_net = pnl_gross - (open_fee + close_fee)
                weights_dict = {f"w_{h}": float(open_weights[i]) for i, h in enumerate(horizon_names)} if open_weights is not None else {}
                for i, h in enumerate(horizon_names):
                    if open_weights is not None:
                        horizon_pnl[h] += float(open_weights[i]) * pnl_net

                duration_h = hours_held
                trades.append({
                    "entry_anchor": open_signal_row["datetime_anchor"] if open_signal_row is not None else pd.NaT,
                    "entry_exec": open_exec_ts,
                    "exit_anchor": pd.NaT if exit_reason != "opposing_signal" else sig_by_exec.get(ts, {}).get("datetime_anchor", pd.NaT),
                    "exit_exec": ts,
                    "direction": "long" if current_pos > 0 else "short",
                    "entry_price": open_price,
                    "exit_price": exit_price,
                    "pnl_gross": pnl_gross,
                    "fees": open_fee + close_fee,
                    "pnl": pnl_net,
                    "duration_h": duration_h,
                    "exit_reason": exit_reason,
                    **weights_dict
                })
                fees.loc[ts] += close_fee
                exit_reasons.append(exit_reason)
                current_pos = 0
                open_price = None
                open_tp_level = None
                open_sl_level = None
                trailing_sl = None
                open_time = None

        # Check for new signal or opposing signal
        if ts in sig_by_exec:
            row = sig_by_exec[ts]
            target_pos = int(row["signal_val"])
            s = row["score_s"]
            thr = float(args.thr) if 'args' in globals() else 0.1  # Fallback for testing

            # Opposing signal check for existing position
            if current_pos != 0 and target_pos != current_pos and abs(s) > thr * opposing_signal_multiplier:
                exit_price = current_price
                pnl_gross = current_pos * (exit_price / open_price - 1.0)
                close_fee = fee_rate * abs(current_pos)
                open_fee  = fee_rate * abs(current_pos)
                pnl_net = pnl_gross - (open_fee + close_fee)
                weights_dict = {f"w_{h}": float(open_weights[i]) for i, h in enumerate(horizon_names)} if open_weights is not None else {}
                for i, h in enumerate(horizon_names):
                    if open_weights is not None:
                        horizon_pnl[h] += float(open_weights[i]) * pnl_net

                duration_h = int((ts - open_exec_ts).total_seconds() // 3600) if open_exec_ts else 0
                trades.append({
                    "entry_anchor": open_signal_row["datetime_anchor"] if open_signal_row is not None else pd.NaT,
                    "entry_exec": open_exec_ts,
                    "exit_anchor": row["datetime_anchor"],
                    "exit_exec": ts,
                    "direction": "long" if current_pos > 0 else "short",
                    "entry_price": open_price,
                    "exit_price": exit_price,
                    "pnl_gross": pnl_gross,
                    "fees": open_fee + close_fee,
                    "pnl": pnl_net,
                    "duration_h": duration_h,
                    "exit_reason": "opposing_signal",
                    **weights_dict
                })
                fees.loc[ts] += close_fee
                exit_reasons.append("opposing_signal")
                current_pos = 0
                open_price = None
                open_tp_level = None
                open_sl_level = None
                trailing_sl = None
                open_time = None

            # Open new position if signal is valid
            if target_pos != 0 and current_pos == 0:
                open_anchor_ts = row["datetime_anchor"]
                open_exec_ts = ts
                open_price = current_price
                open_weights = np.array([row[f"w_{h}"] for h in horizon_names], dtype=float)
                open_signal_row = row
                yhat_primary = row["predicted_tp_level"]
                open_tp_level = open_price * (1 + target_pos * yhat_primary * tp_scaling)
                open_sl_level = open_price * (1 - target_pos * sl_percent)
                trailing_sl = open_sl_level if use_trailing else None
                open_time = ts
                fees.loc[ts] += fee_rate * abs(target_pos)
                current_pos = target_pos

        pos.loc[ts] = current_pos

    # Close any open position at end
    if current_pos != 0 and open_price is not None:
        ts = h1.index[-1]
        exit_price = float(h1.loc[ts, "close"])
        pnl_gross = current_pos * (exit_price / open_price - 1.0)
        open_fee  = fee_rate * abs(current_pos)
        close_fee = fee_rate * abs(current_pos)
        pnl_net = pnl_gross - (open_fee + close_fee)
        weights_dict = {f"w_{h}": float(open_weights[i]) for i, h in enumerate(horizon_names)} if open_weights is not None else {}
        for i, h in enumerate(horizon_names):
            if open_weights is not None:
                horizon_pnl[h] += float(open_weights[i]) * pnl_net
        duration_h = int((ts - open_exec_ts).total_seconds() // 3600) if open_exec_ts else 0
        trades.append({
            "entry_anchor": open_signal_row["datetime_anchor"] if open_signal_row is not None else pd.NaT,
            "entry_exec": open_exec_ts,
            "exit_anchor": pd.NaT,
            "exit_exec": ts,
            "direction": "long" if current_pos > 0 else "short",
            "entry_price": open_price,
            "exit_price": exit_price,
            "pnl_gross": pnl_gross,
            "fees": open_fee + close_fee,
            "pnl": pnl_net,
            "duration_h": duration_h,
            "exit_reason": "end_of_data",
            **weights_dict
        })
        fees.loc[ts] += (open_fee + close_fee)
        exit_reasons.append("end_of_data")

    trades_df = pd.DataFrame(trades)
    metrics = {
        "tp_hit_rate": sum(1 for r in exit_reasons if r == "tp_hit") / len(exit_reasons) if exit_reasons else 0.0,
        "sl_hit_rate": sum(1 for r in exit_reasons if r == "sl_hit") / len(exit_reasons) if exit_reasons else 0.0,
        "trailing_sl_hit_rate": sum(1 for r in exit_reasons if r == "trailing_sl_hit") / len(exit_reasons) if exit_reasons else 0.0,
        "opposing_signal_rate": sum(1 for r in exit_reasons if r == "opposing_signal") / len(exit_reasons) if exit_reasons else 0.0,
        "max_hold_rate": sum(1 for r in exit_reasons if r == "max_hold") / len(exit_reasons) if exit_reasons else 0.0,
    }
    return trades_df, pos, fees, horizon_pnl, metrics

def simulate_trading(
    signals_df: pd.DataFrame,
    h1_prices: pd.DataFrame,
    fee_bps: int = 5,
    tp_horizon: str = "24h",
    tp_scaling: float = 0.7,
    sl_percent: float = 0.02,
    trailing_percent: float = 0.01,
    use_trailing: bool = True,
    max_hold_hours: int = 48,
    opposing_signal_multiplier: float = 1.5,
) -> pd.DataFrame:
    fee_rate = float(fee_bps) / 1e4
    horizon_names = [c.replace("w_", "") for c in signals_df.columns if c.startswith("w_")]
    trades_df, pos, fees_series, _, _ = _build_trades_and_position(
        signals_df, h1_prices, fee_rate, horizon_names, tp_horizon, tp_scaling, sl_percent,
        trailing_percent, use_trailing, max_hold_hours, opposing_signal_multiplier
    )

    px = h1_prices.loc[pos.index].copy()
    px["ret"] = px["close"].pct_change().fillna(0.0)
    pos_shift = pos.shift(1).fillna(0.0)
    px["ret_strat_gross"] = pos_shift.values * px["ret"].values
    fee_series_aligned = fees_series.reindex(px.index).fillna(0.0)
    px["ret_fee"] = -fee_series_aligned
    px["ret_strat"] = px["ret_strat_gross"] + px["ret_fee"]
    px["equity"] = (1.0 + px["ret_strat"]).cumprod()
    roll_max = px["equity"].cummax()
    px["drawdown"] = px["equity"] / roll_max - 1.0

    equity_df = px[["close","ret","ret_strat_gross","ret_fee","ret_strat","equity","drawdown"]].copy()
    equity_df["position"] = pos
    return equity_df

def compute_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    horizon_pnl: Dict[str, float],
    extra_metrics: Dict[str, float],
) -> Dict:
    eps = 1e-12
    rets = equity_df["ret_strat"].dropna()
    if len(rets) < 2:
        return {"error": "Série de retours trop courte."}

    ann_factor = math.sqrt(8760.0)
    mu = rets.mean()
    sigma = rets.std(ddof=1)
    sharpe = float((mu / (sigma + eps)) * ann_factor)

    downside = rets[rets < 0.0]
    ds_sigma = downside.std(ddof=1)
    sortino = float((mu / (ds_sigma + eps)) * ann_factor)

    equity = equity_df["equity"]
    if equity.iloc[0] <= 0:
        cagr = float("nan")
    else:
        years = len(equity) / 8760.0
        cagr = float(equity.iloc[-1] ** (1.0 / max(years, 1e-9)) - 1.0)

    mdd = float(equity_df["drawdown"].min())
    calmar = float(cagr / abs(mdd) if mdd < 0 else float("inf"))

    if trades_df is not None and not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        total_trades = len(trades_df)
        hit_ratio = float(wins / total_trades) if total_trades > 0 else float("nan")
        gains = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        losses = trades_df.loc[trades_df["pnl"] < 0, "pnl"].abs().sum()
        profit_factor = float(gains / (losses + eps)) if losses > 0 else float("inf")
        avg_duration_h = float(trades_df["duration_h"].mean()) if "duration_h" in trades_df.columns else float("nan")
    else:
        hit_ratio = float("nan")
        profit_factor = float("nan")
        avg_duration_h = float("nan")
        total_trades = 0

    pos = equity_df["position"].fillna(0.0)
    turnover = float(pos.diff().abs().sum())
    avg_expo = float(pos.abs().mean())

    metrics = {
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "max_drawdown": float(mdd),
        "hit_ratio": hit_ratio,
        "profit_factor": profit_factor,
        "avg_trade_duration_h": avg_duration_h,
        "num_trades": int(total_trades),
        "turnover": turnover,
        "avg_exposure": avg_expo,
        "pnl_by_horizon": {k: float(v) for k, v in horizon_pnl.items()},
        **extra_metrics
    }
    return metrics

def plot_equity_and_drawdown(equity_df: pd.DataFrame, trades_df: pd.DataFrame, anchor_report: pd.DataFrame, horizon_names: List[str], outdir: str) -> None:
    ensure_dir(outdir)

    # Existing: Equity Curve
    plt.figure(figsize=(10, 4))
    equity_df["equity"].plot()
    plt.title("Equity Curve")
    plt.xlabel("Time (H1)")
    plt.ylabel("Equity (x)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "equity_curve.png"), dpi=150)
    plt.close()

    # Existing: Drawdown Curve
    plt.figure(figsize=(10, 4))
    equity_df["drawdown"].plot()
    plt.title("Drawdown")
    plt.xlabel("Time (H1)")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drawdown_curve.png"), dpi=150)
    plt.close()

    # New: Per-Horizon Equity Curves
    plt.figure(figsize=(12, 6))
    for horizon in horizon_names:
        horizon_equity = pd.Series(1.0, index=trades_df["exit_exec"])
        cum_pnl = 0.0
        for _, trade in trades_df.iterrows():
            weight = trade.get(f"w_{horizon}", 0.0)
            cum_pnl += trade["pnl"] * weight
            horizon_equity.loc[trade["exit_exec"]] = 1.0 + cum_pnl
        horizon_equity = horizon_equity.ffill().reindex(equity_df.index, method='ffill').fillna(1.0)
        plt.plot(horizon_equity.index, horizon_equity, label=f"Horizon {horizon}")
    plt.title("Per-Horizon Equity Curves")
    plt.xlabel("Time (H1)")
    plt.ylabel("Equity (x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_horizon_equity.png"), dpi=150)
    plt.close()

    # New: Signal Strength vs. PnL Scatter
    valid_trades = anchor_report.dropna(subset=["trade_pnl"])
    if not valid_trades.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_trades["score_s"].abs(), valid_trades["trade_pnl"], alpha=0.5)
        plt.title("Signal Strength (|s|) vs. Trade PnL")
        plt.xlabel("Signal Strength (|s|)")
        plt.ylabel("Trade PnL")
        plt.axhline(0, color='red', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "signal_strength_vs_pnl.png"), dpi=150)
        plt.close()

    # New: Position Heatmap (Calendar View)
    pos_df = equity_df[["position"]].copy()
    pos_df["date"] = pos_df.index.date
    pos_df["hour"] = pos_df.index.hour
    pos_pivot = pos_df.pivot_table(values="position", index="date", columns="hour", fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pos_pivot, cmap="RdYlGn", center=0, cbar_kws={'label': 'Position (Long=1, Short=-1, Flat=0)'})
    plt.title("Position Heatmap Over Time")
    plt.xlabel("Hour of Day")
    plt.ylabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "position_heatmap.png"), dpi=150)
    plt.close()

    # New: Drawdown Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(equity_df["drawdown"].dropna(), bins=50, edgecolor='black')
    plt.title("Drawdown Distribution")
    plt.xlabel("Drawdown")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drawdown_histogram.png"), dpi=150)
    plt.close()

    # New: Rolling Sharpe (30-day)
    rolling_window = 30 * 24  # 30 days in hours
    rolling_rets = equity_df["ret_strat"].rolling(window=rolling_window, min_periods=24).mean()
    rolling_std = equity_df["ret_strat"].rolling(window=rolling_window, min_periods=24).std()
    rolling_sharpe = (rolling_rets / (rolling_std + 1e-12)) * np.sqrt(8760)  # Annualized
    plt.figure(figsize=(10, 4))
    rolling_sharpe.plot()
    plt.title("30-Day Rolling Sharpe Ratio")
    plt.xlabel("Time (H1)")
    plt.ylabel("Sharpe Ratio (Annualized)")
    plt.axhline(0, color='red', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rolling_sharpe.png"), dpi=150)
    plt.close()



def run_2025_and_plot(
    datasets_dir: str = ".",
    h1_csv: str = "btc_h1_test.csv",
    datetime_col: str = "datetime",
    price_col: str = "close",
    ckpt_paths: str = "split_1.ckpt,split_2.ckpt,split_3.ckpt,split_4.ckpt",
    model_module: str = "__local__",
    model_class: str = "CryptoTFTModel",
    # Fenêtre d'éval
    start_date: str = "2025-01-01",
    end_date: str   = "2025-08-31",
    # Paramètres “live eval”
    norm_mode: str = "rolling",
    ctx_days_h1: int = 180,
    ctx_days_h4: int = 180,
    ctx_days_d1: int = 180,
    seq_h1: int = 100,
    seq_h4: int = 25,
    seq_d:  int = 30,
    agg_method: str = "mean",
    tau: float = 0.10,
    thr: float = 0.035,          # ≈ P80 selon tes percentiles
    fee_bps: int = 6,            # 0.06% par côté
    horizons: str = "1h,4h,12h,24h,48h",
    tp_horizon: str = "24h",
    tp_scaling: float = 0.7,     # à ajuster si tu veux coller pile à RR~1.5:1 vis-à-vis de ton SL
    sl_percent: float = 0.02,
    trailing_percent: float = 0.01,
    use_trailing: bool = True,
    max_hold_hours: int = 48,
    confirmation_threshold: float = 0.55,
    opposing_signal_multiplier: float = 1.5,
    outdir: str = "live_eval_outputs2"
):
    # ------- préparations -------
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    ckpt_list = [p.strip() for p in ckpt_paths.split(",") if p.strip()]
    models = load_models(ckpt_list, model_module, model_class, device)

    # H1 prix pour plotting et PnL
    h1_prices = _parse_h1_prices(h1_csv, datetime_col=datetime_col, price_col=price_col)

    # n_days dynamiques
    n_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    horizon_names = [h.strip() for h in horizons.split(",")]

    # ------- anchors + prédictions -------
    eval_iter = get_eval_iter(
        datasets_dir=datasets_dir,
        n_days=int(n_days),
        start_date=start_date,
        norm_mode=norm_mode,
        context_days_h1=int(ctx_days_h1),
        context_days_h4=int(ctx_days_h4),
        context_days_d1=int(ctx_days_d1),
        seq_h1=int(seq_h1), seq_h4=int(seq_h4), seq_d=int(seq_d),
    )

    anchors = []
    for i, (anchor_ts, x_h1, x_h4, x_d) in enumerate(eval_iter, start=1):
        preds = predict_models(models, x_h1, x_h4, x_d)
        y_hat = aggregate_preds(preds, method=agg_method)
        s, w = consensus_gating(y_hat, tau=float(tau))
        anchors.append({"anchor_ts": pd.Timestamp(anchor_ts), "y_hat": y_hat, "w": w, "s": s})

    if not anchors:
        raise RuntimeError("Aucun anchor produit pour la fenêtre demandée (vérifie les CSV et dates).")

    signals_df = _build_signals_df(
        anchors, thr=float(thr), horizon_names=horizon_names,
        tp_horizon=tp_horizon, confirmation_threshold=float(confirmation_threshold)
    )

    # ------- backtest -------
    fee_rate = float(fee_bps) / 1e4
    trades_df, pos, fees_series, horizon_pnl, extra_metrics = _build_trades_and_position(
        signals_df, h1_prices, fee_rate, horizon_names,
        tp_horizon, tp_scaling, sl_percent, trailing_percent,
        use_trailing, max_hold_hours, opposing_signal_multiplier
    )

    equity_df = simulate_trading(
        signals_df, h1_prices, fee_bps=int(fee_bps),
        tp_horizon=tp_horizon, tp_scaling=tp_scaling, sl_percent=sl_percent,
        trailing_percent=trailing_percent, use_trailing=use_trailing,
        max_hold_hours=max_hold_hours, opposing_signal_multiplier=opposing_signal_multiplier
    )

    metrics = compute_metrics(equity_df, trades_df, horizon_pnl, extra_metrics)

    # ------- Plot "prix + entrées/sorties" -------
    # Sous-série prix sur la fenêtre (avec petite marge)
    t0 = pd.to_datetime(start_date)
    t1 = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    px = h1_prices.loc[(h1_prices.index >= t0) & (h1_prices.index <= t1)].copy()
    if px.empty:
        raise RuntimeError("Série H1 vide sur la période 2025 (vérifie la préparation des CSV).")

    # Palette (validée)
    COL_PRICE = "#333333"
    COL_LONG_IN = "#2ECC71"
    COL_SHORT_IN = "#E74C3C"
    COL_TP = "#2ECC71"
    COL_SL = "#E74C3C"
    COL_MAXHOLD = "#F39C12"
    COL_OPPOSING = "#8E44AD"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(px.index, px["close"], lw=1.0, color=COL_PRICE, label="BTCUSDT Close (H1)")

    # Filtrer trades dans la fenêtre
    if trades_df is not None and not trades_df.empty:
        td = trades_df.copy()
        # on garde les trades dont l'entrée OU la sortie est dans la fenêtre (pour voir ceux qui chevauchent)
        m = ((td["entry_exec"] >= t0) & (td["entry_exec"] <= t1)) | ((td["exit_exec"] >= t0) & (td["exit_exec"] <= t1))
        td = td[m].reset_index(drop=True)

        # Markers d’entrée
        long_entries = td[td["direction"] == "long"]
        short_entries = td[td["direction"] == "short"]

        ax.scatter(long_entries["entry_exec"], long_entries["entry_price"],
                   marker="^", s=60, color=COL_LONG_IN, label="Entry LONG")
        ax.scatter(short_entries["entry_exec"], short_entries["entry_price"],
                   marker="v", s=60, color=COL_SHORT_IN, label="Entry SHORT")

        # Markers de sortie par raison
        # TP
        tp_mask = td["exit_reason"] == "tp_hit"
        ax.scatter(td.loc[tp_mask, "exit_exec"], td.loc[tp_mask, "exit_price"],
                   marker="o", s=50, facecolors="white", edgecolors=COL_TP, linewidths=1.5,
                   label="TP")
        # SL
        sl_mask = td["exit_reason"] == "sl_hit"
        ax.scatter(td.loc[sl_mask, "exit_exec"], td.loc[sl_mask, "exit_price"],
                   marker="x", s=60, color=COL_SL, label="SL")
        # Max hold
        mh_mask = td["exit_reason"] == "max_hold"
        ax.scatter(td.loc[mh_mask, "exit_exec"], td.loc[mh_mask, "exit_price"],
                   marker="D", s=50, color=COL_MAXHOLD, label="Max hold")
        # Opposing signal
        op_mask = td["exit_reason"] == "opposing_signal"
        ax.scatter(td.loc[op_mask, "exit_exec"], td.loc[op_mask, "exit_price"],
                   marker="s", s=50, color=COL_OPPOSING, label="Opposing")

        # Trailing stop (si jamais présent)
        tr_mask = td["exit_reason"] == "trailing_sl_hit"
        if tr_mask.any():
            ax.scatter(td.loc[tr_mask, "exit_exec"], td.loc[tr_mask, "exit_price"],
                       marker="P", s=60, color="#16A085", label="Trailing SL")

        # Liaisons entrée → sortie (vert si PnL>0, rouge si PnL<0)
        for _, r in td.iterrows():
            if pd.isna(r["exit_exec"]):
                continue
            line_col = COL_LONG_IN if r["pnl"] > 0 else COL_SHORT_IN
            ax.plot([r["entry_exec"], r["exit_exec"]],
                    [r["entry_price"], r["exit_price"]],
                    lw=1.2, alpha=0.6, color=line_col)

    # Habillage
    ax.set_title("BTCUSDT — BUY/SELL From (2025-01-01 → 2025-08-31)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.25)

    # Légende (uniquement pour handles présents)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", framealpha=0.85)

    # Encadré métriques
    mm = metrics
    text = (
        f"Sharpe ann.: {mm.get('sharpe_annualized', float('nan')):.2f}\n"
        f"Sortino ann.: {mm.get('sortino_annualized', float('nan')):.2f}\n"
        f"CAGR (window): {mm.get('cagr', float('nan')):.2%}\n"
        f"Max DD: {mm.get('max_drawdown', 0.0):.2%}\n"
        f"Hit ratio: {mm.get('hit_ratio', float('nan')):.2f}\n"
        f"Profit factor: {mm.get('profit_factor', float('nan')):.2f}\n"
        f"Trades: {mm.get('num_trades', 0)} | Durée moy.: {mm.get('avg_trade_duration_h', 0.0):.1f}h\n"
        f"Exposure moy.: {mm.get('avg_exposure', 0.0):.2f} | Turnover: {mm.get('turnover', 0.0):.0f}"
    )
    ax.text(0.995, 0.02, text, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    ensure_dir(outdir)
    out_path = os.path.join(outdir, "btc_trades_2025.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    # Sauvegardes utiles
    signals_df.to_csv(os.path.join(outdir, "signals_2025.csv"), index=False)
    equity_df.to_csv(os.path.join(outdir, "equity_2025.csv"))
    trades_df.to_csv(os.path.join(outdir, "trades_2025.csv"), index=False)
    with open(os.path.join(outdir, "metrics_2025.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Chart sauvegardé → {out_path}")
    print("[Résumé 2025]")
    for k, v in metrics.items():
        if k != "pnl_by_horizon":
            print(f"{k:22s}: {v}")
        else:
            print("pnl_by_horizon:", v)

# =====================
# ======== MAIN =======
# =====================

def main(args: argparse.Namespace) -> None:
    print("=== Live Eval (Setup + Backtest unifiés) ===")
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu")
    print(f"[device] Using: {device}")

    # Modèles
    ckpt_paths = [p.strip() for p in args.ckpt_paths.split(",") if p.strip()]
    models = load_models(ckpt_paths, args.model_module, args.model_class, device)

    # Prix H1 bruts
    h1_prices = _parse_h1_prices(args.h1_csv, args.datetime_col, args.price_col)

    # Anchors & séquences normalisées (sans fuite)
    print(f"[anchors] get_eval_iter(...), n_days={args.n_days}, start={args.start_date}, norm={args.norm_mode}")
    eval_iter = get_eval_iter(
        datasets_dir=args.datasets_dir,
        n_days=int(args.n_days),
        start_date=args.start_date,
        norm_mode=args.norm_mode,
        context_days_h1=int(args.ctx_days_h1),
        context_days_h4=int(args.ctx_days_h4),
        context_days_d1=int(args.ctx_days_d1),
        seq_h1=int(args.seq_h1), seq_h4=int(args.seq_h4), seq_d=int(args.seq_d),
    )

    horizon_names = [h.strip() for h in args.horizons.split(",")] if args.horizons else HORIZON_NAMES_DEFAULT

    anchors = []
    for i, (anchor_ts, x_h1, x_h4, x_d) in enumerate(eval_iter, start=1):
        preds = predict_models(models, x_h1, x_h4, x_d)
        y_hat = aggregate_preds(preds, method=args.agg_method)
        s, w = consensus_gating(y_hat, tau=float(args.tau))
        anchors.append({"anchor_ts": pd.Timestamp(anchor_ts), "y_hat": y_hat, "w": w, "s": s})
        if (i % 5 == 0) or (i == args.n_days):
            print(f"[progress] anchors traités: {i}/{args.n_days}")

    if not anchors:
        raise RuntimeError("Aucun anchor produit. Vérifie datasets_dir, dates et n_days.")

    # Signaux
    signals_df = _build_signals_df(
        anchors, thr=float(args.thr), horizon_names=horizon_names,
        tp_horizon=args.tp_horizon, confirmation_threshold=args.confirmation_threshold
    )

    # Percentile calculation for |s|
    abs_s = signals_df["score_s"].abs()
    percentiles = abs_s.quantile([0.7, 0.8, 0.9]).to_dict()
    print("\n=== Percentiles of |s| ===")
    for p, v in percentiles.items():
        print(f"{int(p*100)}th percentile: {v:.4f}")
    ensure_dir(args.outdir)
    with open(os.path.join(args.outdir, "s_percentiles.json"), "w") as f:
        json.dump({f"p{int(p*100)}": float(v) for p, v in percentiles.items()}, f, indent=2)

    # Debug dump
    y_cols = [f"yhat_{h}" for h in horizon_names]
    w_cols = [f"w_{h}" for h in horizon_names]
    dbg_cols = ["datetime_anchor", "datetime_exec", "score_s", "signal", "predicted_tp_level"] + y_cols + w_cols
    signals_df[dbg_cols].to_csv(os.path.join(args.outdir, "signals_debug.csv"), index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Aperçu des prédictions/poids par anchor ===")
    print(signals_df[dbg_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Trades, position, fees, attribution PnL
    fee_rate = float(args.fee_bps) / 1e4
    trades_df, position_series, fees_series, horizon_pnl, extra_metrics = _build_trades_and_position(
        signals_df, h1_prices, fee_rate, horizon_names,
        args.tp_horizon, args.tp_scaling, args.sl_percent, args.trailing_percent,
        args.use_trailing, args.max_hold_hours, args.opposing_signal_multiplier
    )

    # Equity pas horaire
    equity_df = simulate_trading(
        signals_df, h1_prices, fee_bps=int(args.fee_bps),
        tp_horizon=args.tp_horizon, tp_scaling=args.tp_scaling, sl_percent=args.sl_percent,
        trailing_percent=args.trailing_percent, use_trailing=args.use_trailing,
        max_hold_hours=args.max_hold_hours, opposing_signal_multiplier=args.opposing_signal_multiplier
    )

    # Rapport par anchor
    anchor_report = signals_df.copy()
    anchor_report["entry_price"] = np.nan
    anchor_report["exit_price"] = np.nan
    anchor_report["trade_pnl"]  = np.nan
    anchor_report["pnl_cum"]    = np.nan
    anchor_report["exit_reason"] = np.nan

    trades_df = trades_df.sort_values("exit_exec").reset_index(drop=True)
    tr_by_exit = {pd.Timestamp(r["exit_exec"]): i for i, r in trades_df.iterrows() if pd.notna(r["exit_exec"])}

    pnl_cum = 0.0
    for i, row in anchor_report.iterrows():
        exec_ts = pd.Timestamp(row["datetime_exec"])
        tr_i = tr_by_exit.get(exec_ts)
        if tr_i is not None:
            tr = trades_df.loc[tr_i]
            anchor_report.at[i, "entry_price"] = tr["entry_price"]
            anchor_report.at[i, "exit_price"]  = tr["exit_price"]
            anchor_report.at[i, "trade_pnl"]   = tr["pnl"]
            anchor_report.at[i, "exit_reason"] = tr["exit_reason"]
            pnl_cum += float(tr["pnl"])
        anchor_report.at[i, "pnl_cum"] = pnl_cum

    # Métriques
    metrics = compute_metrics(equity_df, trades_df, horizon_pnl, extra_metrics)

    # Sauvegardes
    ensure_dir(args.outdir)
    y_cols = [f"yhat_{h}" for h in horizon_names]
    w_cols = [f"w_{h}" for h in horizon_names]
    base_cols = ["datetime_anchor", "datetime_exec", "score_s", "signal", "predicted_tp_level"] + y_cols + w_cols + \
                ["entry_price", "exit_price", "trade_pnl", "pnl_cum", "exit_reason"]

    anchor_report[base_cols].to_csv(os.path.join(args.outdir, "anchors_report.csv"), index=False)
    equity_df.to_csv(os.path.join(args.outdir, "equity.csv"))
    trades_df.to_csv(os.path.join(args.outdir, "trades.csv"), index=False)
    pd.DataFrame([{"horizon": k, "pnl": float(v)} for k, v in horizon_pnl.items()]) \
        .to_csv(os.path.join(args.outdir, "pnl_by_horizon.csv"), index=False)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_equity_and_drawdown(equity_df, trades_df, anchor_report, horizon_names, args.outdir)

    print(f"[save] anchors_report / equity / trades / pnl_by_horizon / metrics -> {args.outdir}")
    print("[plots] equity_curve.png / drawdown_curve.png / per_horizon_equity.png / signal_strength_vs_pnl.png / position_heatmap.png / drawdown_histogram.png / rolling_sharpe.png")
    print("\n=== Summary Metrics ===")
    for k, v in metrics.items():
        if k != "pnl_by_horizon":
            print(f"{k:22s}: {v}")
        else:
            print("pnl_by_horizon:", v)
  
    if args.plot_entries_exits:
        run_2025_and_plot(
              datasets_dir=args.datasets_dir,
              h1_csv=args.h1_csv,
              datetime_col=args.datetime_col,
              price_col=args.price_col,
              ckpt_paths=args.ckpt_paths,
              model_module=args.model_module,
              model_class=args.model_class,
              start_date=args.plot_start,
              end_date=args.plot_end,
              thr=(args.plot_thr if args.plot_thr is not None else args.thr),
              fee_bps=args.fee_bps,
              horizons=args.horizons,
              tp_horizon=args.tp_horizon,
              tp_scaling=args.tp_scaling,
              sl_percent=args.sl_percent,
              trailing_percent=args.trailing_percent,
              use_trailing=args.use_trailing,
              max_hold_hours=args.max_hold_hours,
              confirmation_threshold=args.confirmation_threshold,
              opposing_signal_multiplier=args.opposing_signal_multiplier,
              outdir=args.outdir
          )
        print(f"[plot] Entrées/sorties sauvegardées -> {args.plot_out}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Configuration for Crypto TFT model evaluation and backtesting")

    # Dataset and input paths
    parser.add_argument("--datasets-dir", type=str, default=".", 
                        help="Directory containing H1, H4, and D1 CSV files")
    parser.add_argument("--h1-csv", type=str, default="btc_h1_test.csv", 
                        help="Raw H1 CSV file for execution (containing at least the close price)")
    parser.add_argument("--datetime-col", type=str, default="datetime", 
                        help="Name of the datetime column in the H1 price CSV")
    parser.add_argument("--price-col", type=str, default="close", 
                        help="Name of the price column in the H1 CSV")
    parser.add_argument("--ckpt-paths", type=str, 
                        default="split_1.ckpt,split_2.ckpt,split_3.ckpt,split_4.ckpt", 
                        help="Comma-separated paths to .ckpt model checkpoint files")
    parser.add_argument("--model-module", type=str, default="__local__", 
                        help="Python module containing the model (e.g., '__local__' to use the class defined in this file)")
    parser.add_argument("--model-class", type=str, default="CryptoTFTModel", 
                        help="Name of the LightningModule class (e.g., CryptoTFTModel)")

    # Setup and normalization parameters
    parser.add_argument("--n-days", type=int, default=500, 
                        help="Number of daily anchor points for data processing")
    parser.add_argument("--start-date", type=str, default="2023-06-01", 
                        help="Start date for data processing (format: YYYY-MM-DD)")
    parser.add_argument("--norm-mode", type=str, choices=["rolling", "expanding"], default="rolling", 
                        help="Normalization mode: 'rolling' for a fixed window or 'expanding' for all past data")
    parser.add_argument("--ctx-days-h1", type=int, default=180, 
                        help="Rolling window size for H1 timeframe in days")
    parser.add_argument("--ctx-days-h4", type=int, default=180, 
                        help="Rolling window size for H4 timeframe in days")
    parser.add_argument("--ctx-days-d1", type=int, default=180, 
                        help="Rolling window size for D1 timeframe in days")
    parser.add_argument("--seq-h1", type=int, default=100, 
                        help="Sequence length for H1 timeframe (number of bars)")
    parser.add_argument("--seq-h4", type=int, default=25, 
                        help="Sequence length for H4 timeframe (number of bars)")
    parser.add_argument("--seq-d", type=int, default=30, 
                        help="Sequence length for D1 timeframe (number of bars)")

    # Evaluation and backtest parameters
    parser.add_argument("--agg-method", type=str, choices=["mean", "median"], default="mean", 
                        help="Aggregation method for combining model predictions")
    parser.add_argument("--tau", type=float, default=0.10, 
                        help="Temperature parameter for consensus gating")
    parser.add_argument("--thr", type=float, default=0.035, 
                        help="Symmetric threshold for long/short vs flat positions")
    parser.add_argument("--fee-bps", type=int, default=6, 
                        help="Trading cost in basis points (entry + exit)")
    parser.add_argument("--horizons", type=str, default="1h,4h,12h,24h,48h", 
                        help="Comma-separated names of prediction horizons")
    parser.add_argument("--tp_horizon", type=str, default="24h", 
                        help="Horizon for take-profit calculation (e.g., '24h')")
    parser.add_argument("--tp_scaling", type=float, default=0.7, 
                        help="Scaling factor for take-profit prediction (range: 0.5-1.0)")
    parser.add_argument("--sl_percent", type=float, default=0.02, 
                        help="Stop-loss percentage (e.g., 0.02 for 2%)")
    parser.add_argument("--trailing_percent", type=float, default=0.01, 
                        help="Trailing stop percentage (e.g., 0.01 for 1%)")
    parser.add_argument("--use_trailing", action="store_true", 
                        help="Enable trailing stop mechanism")
    parser.add_argument("--max_hold_hours", type=int, default=48, 
                        help="Maximum hours to hold an open position")
    parser.add_argument("--confirmation_threshold", type=float, default=0.55, 
                        help="Fraction of other horizons agreeing with the primary horizon")
    parser.add_argument("--opposing_signal_multiplier", type=float, default=1.5, 
                        help="Multiplier for handling strong opposing signals")

    # I/O and device options
    parser.add_argument("--outdir", type=str, default="live_eval_outputs_2", 
                        help="Output directory for results and logs")
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force inference to run on CPU instead of GPU")

    # Plotting options (2025 window)
    parser.add_argument("--plot-entries-exits", action="store_true", 
                        default=("ipykernel" in sys.modules), 
                        help="Generate a plot of entry/exit points for a specified window (e.g., 2025)")
    parser.add_argument("--plot-start", type=str, default="2025-01-01", 
                        help="Start of the plotting window (UTC, format: YYYY-MM-DD)")
    parser.add_argument("--plot-end", type=str, default="2025-08-31 23:59:59", 
                        help="End of the plotting window (UTC, format: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--plot-out", type=str, default="live_eval_outputs/entries_exits_2025.png", 
                        help="Output path for the entry/exit plot")
    parser.add_argument("--plot-thr", type=float, default=0.035, 
                        help="Threshold for absolute signal strength in the plot (defaults to --thr)")

    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    main(args)