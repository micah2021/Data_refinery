"""
baseline_comparison.py — ARIMA & LSTM Baseline Comparison
==========================================================
Compares your RLRF model against:
  - ARIMA      (classical time-series, gold standard baseline)
  - Naive      (last observed value — sanity check)
  - Ridge      (linear ML baseline)
  - LSTM       (deep learning baseline — optional, needs tensorflow)

This is your Table 4 in the paper:
  "Comparison of RLRF against baseline methods"

Run:
    pip install statsmodels
    python baseline_comparison.py --disease malaria
    python baseline_comparison.py --all-diseases
    python baseline_comparison.py --disease cholera --include-lstm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("baseline_comparison")

DB_PATH   = os.getenv("DB_PATH",  "./nigeria.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(exist_ok=True)


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def load_time_series(disease: str, lga_id: int | None = None) -> pd.DataFrame:
    """Load weekly incidence time series for one disease."""
    lga_clause = f"AND lga_id = {lga_id}" if lga_id else ""
    sql = f"""
        SELECT epi_year, epi_week,
               AVG(incidence_rate)    AS incidence_rate,
               AVG(reporting_weight)  AS weight
        FROM feature_store
        WHERE disease_category = '{disease}'
          AND incidence_rate IS NOT NULL
          {lga_clause}
        GROUP BY epi_year, epi_week
        ORDER BY epi_year, epi_week
    """
    with connect() as conn:
        df = pd.read_sql_query(sql, conn)
    df["t"] = range(len(df))
    log.info("Time series: %d weekly observations for %s", len(df), disease)
    return df


def metrics(y_true, y_pred) -> dict:
    """Compute RMSE, MAE, R², MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(mape),
        "n":    len(y_true),
    }


# ── Baseline 1: Naive (last value) ───────────────────────────────────────────

def run_naive(series: pd.Series, horizon: int = 4) -> dict:
    """Predict t+horizon = value at t. Simplest possible baseline."""
    y_true = series.iloc[horizon:].values
    y_pred = series.iloc[:-horizon].values
    m = metrics(y_true, y_pred)
    log.info("Naive:  RMSE=%.4f  R²=%.3f", m["rmse"], m["r2"])
    return {"model": "Naive (last value)", **m}


# ── Baseline 2: Ridge regression ─────────────────────────────────────────────

def run_ridge(df: pd.DataFrame, horizon: int = 4) -> dict:
    """
    Ridge regression with lag features — linear ML baseline.
    Uses same lag structure as RLRF to isolate the benefit of the
    tree ensemble vs linear approximation.
    """
    df = df.copy()
    series = df["incidence_rate"]
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = series.shift(lag)
    df["week_sin"] = np.sin(2 * np.pi * df["epi_week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["epi_week"] / 52)
    df["target"]   = series.shift(-horizon)
    df = df.dropna()

    feat_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("week_")]
    X = df[feat_cols].values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tscv   = TimeSeriesSplit(n_splits=5)
    preds_all, true_all = [], []
    for tr_idx, te_idx in tscv.split(X):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[tr_idx], y[tr_idx])
        preds_all.extend(ridge.predict(X[te_idx]))
        true_all.extend(y[te_idx])

    m = metrics(true_all, preds_all)
    log.info("Ridge:  RMSE=%.4f  R²=%.3f", m["rmse"], m["r2"])
    return {"model": "Ridge Regression", **m}


# ── Baseline 3: ARIMA ─────────────────────────────────────────────────────────

def run_arima(series: pd.Series, horizon: int = 4) -> dict:
    """
    ARIMA(2,1,2) — classical time-series baseline.
    Used in walk-forward validation: train on all past, predict next horizon.
    This is computationally intensive for large series — we cap at 200 obs.
    """
    if not HAS_STATSMODELS:
        log.warning("statsmodels not installed — skipping ARIMA. pip install statsmodels")
        return {"model": "ARIMA(2,1,2)", "rmse": None, "mae": None, "r2": None,
                "mape": None, "n": 0, "note": "statsmodels not installed"}

    series = series.dropna()
    # Cap for speed — ARIMA walk-forward is O(n²)
    if len(series) > 200:
        series = series.iloc[-200:]
    log.info("Running ARIMA walk-forward on %d observations...", len(series))

    preds, trues = [], []
    # Walk-forward: train on first 70%, predict one step at a time
    train_size = int(len(series) * 0.7)
    for i in range(train_size, len(series) - horizon):
        train = series.iloc[:i]
        try:
            model = ARIMA(train, order=(2, 1, 2))
            result = model.fit()
            fc = result.forecast(steps=horizon)
            preds.append(float(fc.iloc[-1]))
            trues.append(float(series.iloc[i + horizon - 1]))
        except Exception:
            continue

    if not preds:
        return {"model": "ARIMA(2,1,2)", "rmse": None, "mae": None,
                "r2": None, "mape": None, "n": 0}

    m = metrics(trues, preds)
    log.info("ARIMA:  RMSE=%.4f  R²=%.3f  (n=%d predictions)", m["rmse"], m["r2"], m["n"])
    return {"model": "ARIMA(2,1,2)", **m}


# ── Baseline 4: LSTM ──────────────────────────────────────────────────────────

def run_lstm(series: pd.Series, horizon: int = 4, lookback: int = 8) -> dict:
    """
    LSTM deep learning baseline.
    Requires tensorflow: pip install tensorflow
    Uses same lookback window as RLRF lag features for fair comparison.
    """
    if not HAS_TF:
        log.warning("tensorflow not installed — skipping LSTM. pip install tensorflow")
        return {"model": "LSTM", "rmse": None, "mae": None, "r2": None,
                "mape": None, "n": 0, "note": "tensorflow not installed"}

    series = series.dropna().values.astype(float)
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(lookback, len(series_scaled) - horizon):
        X_seq.append(series_scaled[i-lookback:i])
        y_seq.append(series_scaled[i + horizon - 1])
    X_seq = np.array(X_seq).reshape(-1, lookback, 1)
    y_seq = np.array(y_seq)

    split = int(len(X_seq) * 0.8)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_tr, y_tr, epochs=30, batch_size=32,
        validation_split=0.1, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    preds_scaled = model.predict(X_te, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    trues = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    m = metrics(trues, preds)
    log.info("LSTM:  RMSE=%.4f  R²=%.3f", m["rmse"], m["r2"])
    return {"model": "LSTM (32-16 units)", **m}


# ── Load RLRF results from saved metadata ─────────────────────────────────────

def load_rlrf_scores(disease: str) -> dict:
    """Load RLRF cross-validation scores from saved model metadata."""
    meta_path = MODEL_DIR / f"outbreak_{disease}_ALL_meta.json"
    if not meta_path.exists():
        log.warning("No saved RLRF model for %s — run outbreak_model.py first", disease)
        return {"model": "RLRF (N-Step SARSA + RF)", "rmse": None,
                "mae": None, "r2": None, "mape": None, "n": 0}
    meta = json.loads(meta_path.read_text())
    cv   = meta["cv_scores"]
    return {
        "model": "RLRF (N-Step SARSA + RF)",
        "rmse":  cv["mean_rmse"],
        "mae":   cv["mean_mae"],
        "r2":    cv["mean_r2"],
        "mape":  None,
        "n":     None,
    }


# ── Full comparison ───────────────────────────────────────────────────────────

def run_comparison(
    disease: str,
    include_lstm: bool = False,
    lga_id: int | None = None,
) -> pd.DataFrame:
    """
    Run all baselines and RLRF, return comparison DataFrame.
    This is Table 4 in your paper.
    """
    log.info("="*55)
    log.info("Baseline comparison — disease=%s", disease)

    df_ts = load_time_series(disease, lga_id)
    series = df_ts["incidence_rate"]

    results = []
    results.append(run_naive(series))
    results.append(run_ridge(df_ts))
    results.append(run_arima(series))
    if include_lstm:
        results.append(run_lstm(series))
    results.append(load_rlrf_scores(disease))

    comparison_df = pd.DataFrame(results)
    return comparison_df


def print_comparison_table(df: pd.DataFrame, disease: str) -> None:
    """Print a LaTeX-ready comparison table for your paper."""
    print(f"\n{'='*70}")
    print(f"  TABLE 4 — Baseline Comparison ({disease.replace('_',' ').title()})")
    print(f"  Metric: 4-week-ahead incidence prediction (Nigerian data only)")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE%':>8}")
    print(f"  {'-'*65}")

    best_r2   = df["r2"].dropna().max()
    best_rmse = df["rmse"].dropna().min()

    for _, row in df.iterrows():
        r2_str   = f"{row['r2']:.3f}"   if pd.notna(row.get("r2"))   else "  N/A"
        rmse_str = f"{row['rmse']:.4f}" if pd.notna(row.get("rmse")) else "  N/A"
        mae_str  = f"{row['mae']:.4f}"  if pd.notna(row.get("mae"))  else "  N/A"
        mape_str = f"{row['mape']:.1f}" if pd.notna(row.get("mape")) else "  N/A"

        marker = " ← BEST" if (
            pd.notna(row.get("r2")) and abs(row["r2"] - best_r2) < 1e-6
        ) else ""
        print(f"  {row['model']:<35} {rmse_str:>8} {mae_str:>8} {r2_str:>8} {mape_str:>8}{marker}")

    # Improvement over best baseline
    rlrf_row = df[df["model"].str.contains("RLRF")]
    if not rlrf_row.empty and pd.notna(rlrf_row["r2"].values[0]):
        baselines = df[~df["model"].str.contains("RLRF")]
        best_baseline_r2   = baselines["r2"].dropna().max()
        best_baseline_rmse = baselines["rmse"].dropna().min()
        rlrf_r2   = rlrf_row["r2"].values[0]
        rlrf_rmse = rlrf_row["rmse"].values[0]
        print(f"\n  RLRF improvement over best baseline:")
        print(f"    R²   : +{rlrf_r2 - best_baseline_r2:.3f} ({((rlrf_r2/best_baseline_r2)-1)*100:.1f}% relative)")
        if pd.notna(rlrf_rmse) and pd.notna(best_baseline_rmse):
            print(f"    RMSE : {rlrf_rmse - best_baseline_rmse:.4f} ({'lower is better' if rlrf_rmse < best_baseline_rmse else 'higher'})")

    print(f"{'='*70}\n")

    # LaTeX snippet
    print("  LaTeX table snippet (paste into your paper):")
    print(f"  \\begin{{tabular}}{{lrrrr}}")
    print(f"  \\hline")
    print(f"  Model & RMSE & MAE & R$^2$ & MAPE \\\\ \\hline")
    for _, row in df.iterrows():
        r2   = f"{row['r2']:.3f}"   if pd.notna(row.get("r2"))   else "—"
        rmse = f"{row['rmse']:.4f}" if pd.notna(row.get("rmse")) else "—"
        mae  = f"{row['mae']:.4f}"  if pd.notna(row.get("mae"))  else "—"
        mape = f"{row['mape']:.1f}" if pd.notna(row.get("mape")) else "—"
        bold = "\\textbf{" if "RLRF" in str(row["model"]) else ""
        endb = "}"         if "RLRF" in str(row["model"]) else ""
        print(f"  {bold}{row['model']}{endb} & {rmse} & {mae} & {r2} & {mape} \\\\")
    print(f"  \\hline")
    print(f"  \\end{{tabular}}\n")


def save_comparison(df: pd.DataFrame, disease: str) -> Path:
    out = MODEL_DIR / f"baseline_comparison_{disease}.json"
    out.write_text(df.to_json(orient="records", indent=2))
    log.info("Comparison saved: %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison for outbreak model")
    parser.add_argument("--disease",       default="malaria")
    parser.add_argument("--all-diseases",  action="store_true")
    parser.add_argument("--include-lstm",  action="store_true")
    parser.add_argument("--lga-id",        type=int, default=None)
    parser.add_argument("--db",            default=DB_PATH)
    args = parser.parse_args()

    global DB_PATH
    DB_PATH = args.db

    if args.all_diseases:
        with sqlite3.connect(DB_PATH) as conn:
            diseases = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
                ).fetchall()
            ]
        all_results = {}
        for d in diseases:
            try:
                df = run_comparison(d, args.include_lstm, args.lga_id)
                print_comparison_table(df, d)
                save_comparison(df, d)
                all_results[d] = df.to_dict(orient="records")
            except Exception as exc:
                log.error("Failed for %s: %s", d, exc)

        # Master summary
        summary_path = MODEL_DIR / "baseline_comparison_all.json"
        summary_path.write_text(json.dumps(all_results, indent=2))
        log.info("All comparisons saved: %s", summary_path)
    else:
        df = run_comparison(args.disease, args.include_lstm, args.lga_id)
        print_comparison_table(df, args.disease)
        save_comparison(df, args.disease)


if __name__ == "__main__":
    main()
