"""
export_predictions.py — Export RLRF predictions to nigeria.db
=============================================================
Uses train_model.py's own imports to resolve the pickle correctly.

Run:
    python export_predictions.py
"""

# ── CRITICAL: import train_model FIRST before any pickle loading ──────────────
import train_model  # resolves RandomForestRegressor, GradientBoostingRegressor etc.

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH   = "./nigeria.db"
MODEL_DIR = Path("./models")


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            pred_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            lga_id              INTEGER NOT NULL,
            disease_category    TEXT    NOT NULL,
            epi_year            INTEGER,
            epi_week            INTEGER,
            predicted_week      INTEGER,
            predicted_year      INTEGER,
            predicted_incidence REAL,
            lower_bound         REAL,
            upper_bound         REAL,
            model_type          TEXT DEFAULT 'rlrf',
            model_tag           TEXT,
            predicted_at        TEXT DEFAULT (datetime('now')),
            UNIQUE(lga_id, disease_category, epi_year, epi_week, model_tag)
        )
    """)
    conn.commit()


def load_model(pkl_path):
    """Load pkl — train_model already imported so classes resolve."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_features(conn, disease):
    """Pull feature data using train_model's own engineer_features function."""
    df = pd.read_sql_query(f"""
        SELECT fs.lga_id, fs.epi_year, fs.epi_week,
               fs.incidence_lag_1w, fs.incidence_lag_2w,
               fs.incidence_lag_4w, fs.incidence_lag_8w,
               fs.rainfall_mm, fs.temp_max_c, fs.humidity_pct,
               fs.ndvi, fs.flood_risk_flag,
               fs.poverty_headcount_pct, fs.food_insecurity_pct,
               fs.nhia_coverage_pct, fs.piped_water_pct, fs.sanitation_pct,
               fs.pop_density, fs.epi_week AS epi_wk,
               fs.reporting_weight, fs.active_alert_flag,
               fs.zone, fs.lga_type
        FROM feature_store fs
        WHERE fs.disease_category = '{disease}'
          AND fs.incidence_rate IS NOT NULL
        ORDER BY fs.lga_id, fs.epi_year, fs.epi_week
        LIMIT 10000
    """, conn)
    return df


def make_predictions(model, df):
    """Build X matching model's expected features and predict."""
    # Use train_model's get_feature_columns if available
    try:
        feat_cols = train_model.get_feature_columns(df)
    except Exception:
        candidates = [
            "incidence_lag_1w", "incidence_lag_2w",
            "incidence_lag_4w", "incidence_lag_8w",
            "rainfall_mm", "temp_max_c", "humidity_pct",
            "ndvi", "flood_risk_flag",
            "poverty_headcount_pct", "food_insecurity_pct",
            "nhia_coverage_pct", "piped_water_pct", "sanitation_pct",
            "pop_density", "epi_week", "reporting_weight", "active_alert_flag",
        ]
        feat_cols = [c for c in candidates if c in df.columns]

    X = df[feat_cols].fillna(df[feat_cols].median())

    # Match model's expected feature count
    if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
        if hasattr(model, "feature_names_in_"):
            avail = [c for c in model.feature_names_in_ if c in X.columns]
            X = X[avail]
        else:
            X = X.iloc[:, :model.n_features_in_]

    return model.predict(X)


def write_predictions(conn, df, preds, disease):
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        pw  = int(row["epi_week"]) + 4
        py  = int(row["epi_year"])
        if pw > 52:
            pw -= 52
            py += 1
        val = max(0.0, float(preds[i]))
        rows.append((
            int(row["lga_id"]), disease,
            int(row["epi_year"]), int(row["epi_week"]),
            pw, py,
            round(val, 4),
            round(val * 0.85, 4),
            round(val * 1.15, 4),
            "rlrf", f"{disease}_ALL",
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO model_predictions
        (lga_id, disease_category, epi_year, epi_week,
         predicted_week, predicted_year,
         predicted_incidence, lower_bound, upper_bound,
         model_type, model_tag)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    return len(rows)


def main():
    print("=" * 52)
    print("Export RLRF Predictions → nigeria.db")
    print("=" * 52)

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    pkl_files = sorted(MODEL_DIR.glob("*_rlrf.pkl"))
    if not pkl_files:
        print("No *_rlrf.pkl files found in models/")
        conn.close()
        return

    print(f"Found {len(pkl_files)} models\n")
    total = 0

    for pkl_path in pkl_files:
        disease = pkl_path.stem.replace("_rlrf", "")
        print(f"Processing: {disease}")
        try:
            model = load_model(pkl_path)
            df    = get_features(conn, disease)
            if df.empty:
                print(f"  No feature data — skipping")
                continue
            preds = make_predictions(model, df)
            n     = write_predictions(conn, df, preds, disease)
            total += n
            print(f"  ✅ {n:,} predictions written")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    count = conn.execute(
        "SELECT COUNT(*) FROM model_predictions"
    ).fetchone()[0]

    print(f"\n{'='*52}")
    print(f"Total predictions in DB: {count:,}")

    by_disease = conn.execute("""
        SELECT disease_category, COUNT(*) AS n
        FROM model_predictions
        GROUP BY disease_category ORDER BY 1
    """).fetchall()
    for row in by_disease:
        print(f"  {row[0]:<22} {row[1]:>8,}")
    print("=" * 52)
    conn.close()

    print("\nDone. Now run:")
    print("  git add -f nigeria.db")
    print('  git commit -m "Add model predictions"')
    print("  git push")


if __name__ == "__main__":
    main()
