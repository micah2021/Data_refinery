"""
shap_bridge.py — Standalone SHAP analysis for Nigeria RLRF models
==================================================================
Works directly with your existing .pkl model files and feature_store.
No dependency on outbreak_model.py internals.

Run:
    python shap_bridge.py --disease malaria
    python shap_bridge.py --all-diseases
"""

from __future__ import annotations
import argparse, json, logging, os, pickle, sqlite3, sys, warnings
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import sklearn BEFORE loading any pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

try:
    import shap
except ImportError:
    print("Run: pip install shap")
    sys.exit(1)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("shap_bridge")

DB_PATH   = os.getenv("DB_PATH", "./nigeria.db")
MODEL_DIR = Path("./models")
PLOT_DIR  = MODEL_DIR / "shap_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ZONE_COLOURS = {
    "NC": "#4CAF50", "NE": "#F44336", "NW": "#FF9800",
    "SE": "#2196F3", "SS": "#9C27B0", "SW": "#00BCD4",
}


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def load_model(disease: str):
    """Load model from .pkl with sklearn pre-imported."""
    pkl_path = MODEL_DIR / f"{disease}_rlrf.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No model at {pkl_path}")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded model: %s  type=%s", pkl_path, type(model).__name__)
    return model


def load_feature_data(disease: str) -> pd.DataFrame:
    """Load and engineer features from feature_store."""
    sql = f"""
        SELECT fs.*, l.lga_name, l.state
        FROM feature_store fs
        JOIN lga l ON l.lga_id = fs.lga_id
        WHERE fs.disease_category = '{disease}'
          AND fs.incidence_rate IS NOT NULL
        ORDER BY fs.lga_id, fs.epi_year, fs.epi_week
    """
    with connect() as conn:
        df = pd.read_sql_query(sql, conn)
    log.info("Loaded %d rows for %s", len(df), disease)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y matching what your train_model.py used."""
    df = df.copy().sort_values(["lga_id", "epi_year", "epi_week"])

    # Target
    df["target"] = df.groupby("lga_id")["incidence_rate"].shift(-4)
    df = df.dropna(subset=["target"])

    # Cyclical week
    df["epi_week_sin"] = np.sin(2 * np.pi * df["epi_week"] / 52)
    df["epi_week_cos"] = np.cos(2 * np.pi * df["epi_week"] / 52)

    # Encode categoricals
    df["zone_enc"]     = LabelEncoder().fit_transform(df["zone"].fillna("NC"))
    df["lga_type_enc"] = LabelEncoder().fit_transform(df["lga_type"].fillna("rural"))

    # Feature columns — matches feature_store schema
    candidate_cols = [
        "incidence_lag_1w", "incidence_lag_2w",
        "incidence_lag_4w", "incidence_lag_8w",
        "rainfall_mm", "temp_max_c", "humidity_pct",
        "ndvi", "flood_risk_flag",
        "poverty_headcount_pct", "food_insecurity_pct",
        "nhia_coverage_pct", "piped_water_pct", "sanitation_pct",
        "pop_density", "zone_enc", "lga_type_enc",
        "epi_week", "epi_week_sin", "epi_week_cos",
        "reporting_weight", "active_alert_flag",
    ]
    feat_cols = [c for c in candidate_cols if c in df.columns]

    # Try to match model's expected features if available
    model_cols = None
    registry_path = MODEL_DIR / "model_registry.json"
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text())
            disease_name = df["disease_category"].iloc[0] if "disease_category" in df.columns else ""
            entry = next(
                (d for d in reg.get("diseases", [])
                 if d.get("disease") == disease_name), None
            )
            if entry and "feature_columns" in entry:
                model_cols = [c for c in entry["feature_columns"] if c in df.columns]
        except Exception:
            pass

    if model_cols:
        feat_cols = model_cols
        log.info("Using feature columns from registry: %d cols", len(feat_cols))
    else:
        log.info("Using default feature columns: %d cols", len(feat_cols))

    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    return df[feat_cols], df["target"], df


def get_feature_labels(cols: list[str]) -> list[str]:
    labels = {
        "incidence_lag_1w":      "Incidence — 1 week ago",
        "incidence_lag_2w":      "Incidence — 2 weeks ago",
        "incidence_lag_4w":      "Incidence — 4 weeks ago",
        "incidence_lag_8w":      "Incidence — 8 weeks ago",
        "rainfall_mm":           "Rainfall (mm)",
        "temp_max_c":            "Max Temperature (°C)",
        "humidity_pct":          "Humidity (%)",
        "ndvi":                  "Vegetation Index (NDVI)",
        "flood_risk_flag":       "Flood Risk",
        "poverty_headcount_pct": "Poverty Rate (%)",
        "food_insecurity_pct":   "Food Insecurity (%)",
        "nhia_coverage_pct":     "Health Insurance (%)",
        "piped_water_pct":       "Piped Water Access (%)",
        "sanitation_pct":        "Sanitation Access (%)",
        "pop_density":           "Population Density",
        "zone_enc":              "Geographic Zone",
        "lga_type_enc":          "LGA Type",
        "epi_week":              "Epidemiological Week",
        "epi_week_sin":          "Week (sin)",
        "epi_week_cos":          "Week (cos)",
        "reporting_weight":      "Data Quality Weight",
        "active_alert_flag":     "Active Alert",
    }
    return [labels.get(c, c) for c in cols]


def compute_shap(model, X: pd.DataFrame, sample_size: int = 2000):
    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X_s = X.iloc[idx].reset_index(drop=True)
    else:
        X_s = X.reset_index(drop=True)

    log.info("Computing SHAP values for %d samples...", len(X_s))
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)
    log.info("SHAP done. Shape: %s", np.array(shap_values).shape)
    return shap_values, explainer.expected_value, X_s


def plot_summary(sv, X_s, disease, feat_labels):
    mean_abs = np.abs(sv).mean(axis=0)
    top_n = min(15, len(feat_labels))
    order = np.argsort(mean_abs)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#E53935" if i >= len(order)-5 else "#90A4AE"
               for i in range(len(order))]
    ax.barh(
        [feat_labels[i] for i in order],
        mean_abs[order],
        color=colours, edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(
        f"Feature Importance — {disease.replace('_',' ').title()}\n"
        f"RLRF Model (N-Step SARSA + Random Forest) — Nigerian Data",
        fontsize=11, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOT_DIR / f"shap_summary_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Summary plot → %s", out)
    return out


def plot_beeswarm(sv, X_s, disease, feat_labels, top_n=12):
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:]
    sv_top   = sv[:, top_idx]
    X_top    = X_s.iloc[:, top_idx].copy()
    X_top.columns = [feat_labels[i] for i in top_idx]

    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv_top, X_top, plot_type="dot", show=False)
    plt.title(
        f"SHAP Beeswarm — {disease.replace('_',' ').title()}\n"
        "Red=increases incidence, Blue=decreases incidence",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOT_DIR / f"shap_beeswarm_{disease}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Beeswarm plot → %s", out)
    return out


def plot_waterfall(sv, expected, X_s, disease, feat_labels, idx=0):
    sv_row  = sv[idx]
    X_row   = X_s.iloc[idx]
    order   = np.argsort(np.abs(sv_row))[-12:]
    feats   = [feat_labels[i] for i in order]
    sv_ord  = sv_row[order]
    vals    = X_row.iloc[order].values

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#E53935" if v > 0 else "#1E88E5" for v in sv_ord]
    ax.barh(range(len(feats)), sv_ord, color=colours, alpha=0.85)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(
        [f"{f}\n(={v:.2f})" for f, v in zip(feats, vals)], fontsize=8
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution", fontsize=10)
    ax.set_title(
        f"Single Prediction Explanation — {disease.replace('_',' ').title()}\n"
        f"Baseline={float(expected):.4f}  Final={float(expected)+sv_row.sum():.4f}",
        fontsize=10, fontweight="bold",
    )
    red  = mpatches.Patch(color="#E53935", label="Increases prediction")
    blue = mpatches.Patch(color="#1E88E5", label="Decreases prediction")
    ax.legend(handles=[red, blue], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOT_DIR / f"shap_waterfall_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Waterfall plot → %s", out)
    return out


def plot_zone_breakdown(sv, X_s, df_full, disease, feat_labels, top_n=5):
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:]
    top_names = [feat_labels[i] for i in top_idx]

    zones_col = df_full["zone"].iloc[:len(X_s)].reset_index(drop=True)
    rows = []
    for zone in sorted(ZONE_COLOURS.keys()):
        mask = (zones_col == zone).values
        if mask.sum() == 0:
            continue
        sv_z = sv[mask][:, top_idx]
        for fi, fn in enumerate(top_names):
            rows.append({
                "zone": zone, "feature": fn,
                "mean_shap": float(np.abs(sv_z[:, fi]).mean()),
            })

    zone_df  = pd.DataFrame(rows)
    fig, ax  = plt.subplots(figsize=(11, 6))
    x        = np.arange(len(top_names))
    width    = 0.13
    for i, zone in enumerate(sorted(ZONE_COLOURS.keys())):
        zd   = zone_df[zone_df["zone"] == zone]
        vals = [
            zd[zd["feature"] == f]["mean_shap"].values[0]
            if f in zd["feature"].values else 0
            for f in top_names
        ]
        ax.bar(x + i*width, vals, width,
               label=zone, color=ZONE_COLOURS[zone], alpha=0.85)

    ax.set_xticks(x + width*2.5)
    ax.set_xticklabels(top_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP|", fontsize=10)
    ax.set_title(
        f"Zone-Stratified Feature Importance — {disease.replace('_',' ').title()}\n"
        "Which factors drive predictions differ by Nigerian zone",
        fontsize=11, fontweight="bold",
    )
    ax.legend(title="Zone", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOT_DIR / f"shap_zone_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Zone breakdown → %s", out)
    return out


def print_shap_report(sv, X_s, disease, expected, feat_labels):
    mean_abs  = np.abs(sv).mean(axis=0)
    total     = mean_abs.sum()
    top10_idx = np.argsort(mean_abs)[-10:][::-1]

    lag_pct = sum(
        mean_abs[i] for i, c in enumerate(X_s.columns) if "lag" in c
    ) / total * 100
    climate_cols = {"rainfall_mm","temp_max_c","humidity_pct","ndvi","flood_risk_flag"}
    climate_pct  = sum(
        mean_abs[i] for i, c in enumerate(X_s.columns) if c in climate_cols
    ) / total * 100
    ses_cols = {"poverty_headcount_pct","food_insecurity_pct",
                "nhia_coverage_pct","piped_water_pct","sanitation_pct"}
    ses_pct = sum(
        mean_abs[i] for i, c in enumerate(X_s.columns) if c in ses_cols
    ) / total * 100

    print(f"\n{'='*58}")
    print(f"  SHAP REPORT — {disease.upper()}")
    print(f"{'='*58}")
    print(f"  Baseline (avg prediction) : {float(expected):.4f} per 10k pop")
    print(f"  Lag features              : {lag_pct:.1f}% of total SHAP")
    print(f"  Climate features          : {climate_pct:.1f}%")
    print(f"  SES features              : {ses_pct:.1f}%")
    print(f"\n  Top 10 features:")
    print(f"  {'Rank':<5} {'Feature':<35} {'SHAP':>8} {'%':>6}")
    print(f"  {'-'*56}")
    for rank, idx in enumerate(top10_idx, 1):
        pct = mean_abs[idx] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {rank:<5} {feat_labels[idx]:<35} {mean_abs[idx]:>8.4f} {pct:>5.1f}%  {bar}")
    print(f"{'='*58}\n")

    # Save JSON
    report = {
        "disease": disease,
        "n_samples": len(X_s),
        "baseline": float(expected),
        "lag_pct": round(lag_pct, 1),
        "climate_pct": round(climate_pct, 1),
        "ses_pct": round(ses_pct, 1),
        "top_features": [
            {"rank": r+1, "feature": feat_labels[i],
             "mean_abs_shap": round(float(mean_abs[i]), 4),
             "pct": round(float(mean_abs[i]/total*100), 1)}
            for r, i in enumerate(top10_idx)
        ],
    }
    out = MODEL_DIR / f"shap_report_{disease}.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("Report → %s", out)


def run(disease: str):
    log.info("="*50)
    log.info("SHAP — %s", disease)

    model          = load_model(disease)
    df             = load_feature_data(disease)
    X, y, df_full  = engineer_features(df)

    # If model expects different n_features, trim/pad X
    if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
        log.warning(
            "Model expects %d features, X has %d — using model's feature_names_in_ if available",
            model.n_features_in_, X.shape[1],
        )
        if hasattr(model, "feature_names_in_"):
            avail = [c for c in model.feature_names_in_ if c in X.columns]
            X = X[avail]
            log.info("Trimmed X to %d columns", len(avail))

    feat_labels    = get_feature_labels(list(X.columns))
    sv, expected, X_s = compute_shap(model, X)

    plot_summary(sv, X_s, disease, feat_labels)
    plot_beeswarm(sv, X_s, disease, feat_labels)
    plot_waterfall(sv, expected, X_s, disease, feat_labels)
    plot_zone_breakdown(sv, X_s, df_full, disease, feat_labels)
    print_shap_report(sv, X_s, disease, expected, feat_labels)

    log.info("All plots saved to %s", PLOT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease",      default="malaria")
    parser.add_argument("--all-diseases", action="store_true")
    args = parser.parse_args()

    if args.all_diseases:
        with sqlite3.connect(DB_PATH) as conn:
            diseases = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
                ).fetchall()
            ]
        for d in diseases:
            try:
                run(d)
            except Exception as e:
                log.error("Failed %s: %s", d, e)
    else:
        run(args.disease)


if __name__ == "__main__":
    main()
