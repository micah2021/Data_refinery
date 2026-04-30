"""
shap_explainer.py — SHAP Explainability for Nigeria Outbreak Predictor
=======================================================================
Generates SHAP (SHapley Additive exPlanations) values for the RLRF model.

SHAP answers the question reviewers will ask:
  "WHY does your model predict high incidence for this LGA?"

Outputs
-------
  models/shap_values_<disease>.pkl     — raw SHAP values
  models/shap_summary_<disease>.png    — global feature importance plot
  models/shap_beeswarm_<disease>.png   — beeswarm (direction of effect)
  models/shap_waterfall_<disease>.png  — single prediction explanation
  models/shap_dependence_<disease>.png — top feature dependence plots
  models/shap_report_<disease>.json    — publishable summary statistics

Run:
    pip install shap matplotlib
    python shap_explainer.py --disease malaria
    python shap_explainer.py --all-diseases
    python shap_explainer.py --disease cholera --lga-id 118
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

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
log = logging.getLogger("shap_explainer")

DB_PATH   = os.getenv("DB_PATH",  "./nigeria.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
PLOT_DIR  = MODEL_DIR / "shap_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Nigeria zone colour map — consistent with dashboard
ZONE_COLOURS = {
    "NC": "#4CAF50", "NE": "#F44336", "NW": "#FF9800",
    "SE": "#2196F3", "SS": "#9C27B0", "SW": "#00BCD4",
}

FEATURE_LABELS = {
    "incidence_lag_1w":      "Incidence (1 week ago)",
    "incidence_lag_2w":      "Incidence (2 weeks ago)",
    "incidence_lag_4w":      "Incidence (4 weeks ago)",
    "incidence_lag_8w":      "Incidence (8 weeks ago)",
    "rainfall_mm":           "Monthly Rainfall (mm)",
    "temp_max_c":            "Max Temperature (°C)",
    "humidity_pct":          "Humidity (%)",
    "ndvi":                  "Vegetation Index (NDVI)",
    "flood_risk_flag":       "Flood Risk",
    "poverty_headcount_pct": "Poverty Rate (%)",
    "food_insecurity_pct":   "Food Insecurity (%)",
    "nhia_coverage_pct":     "Health Insurance Coverage (%)",
    "piped_water_pct":       "Piped Water Access (%)",
    "sanitation_pct":        "Sanitation Access (%)",
    "pop_density":           "Population Density",
    "zone_enc":              "Geographic Zone",
    "lga_type_enc":          "LGA Type (urban/rural)",
    "epi_week":              "Epidemiological Week",
    "epi_week_sin":          "Week (sine encoding)",
    "epi_week_cos":          "Week (cosine encoding)",
    "reporting_weight":      "Data Quality Weight",
    "active_alert_flag":     "Active Outbreak Alert",
}


def load_model_and_data(disease: str, zone: str | None = None):
    """Load saved RLRF model and rebuild feature matrix."""
    # Import here to avoid circular dependency
    import sys
    sys.path.insert(0, ".")
    from outbreak_model import OutbreakPredictor, load_features, build_dataset

    predictor = OutbreakPredictor.load(disease, zone)
    df = load_features(disease, zone)
    X, y, w = build_dataset(df)

    # Align df index with X
    df_aligned = df.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    log.info(
        "Loaded model + data: disease=%s zone=%s  X.shape=%s",
        disease, zone or "ALL", X.shape,
    )
    return predictor, X, y, w, df_aligned


def compute_shap_values(
    predictor,
    X: pd.DataFrame,
    sample_size: int = 2000,
) -> tuple:
    """
    Compute SHAP values using TreeExplainer (fast, exact for RF).

    We sample up to `sample_size` rows for speed — SHAP is O(n*d)
    so 2000 rows gives stable estimates in seconds rather than minutes.
    """
    if not HAS_SHAP:
        raise ImportError(
            "shap not installed. Run: pip install shap"
        )

    # Sample for speed
    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    log.info("Computing SHAP values for %d samples...", len(X_sample))
    explainer   = shap.TreeExplainer(predictor.model)
    shap_values = explainer.shap_values(X_sample)
    expected    = explainer.expected_value

    log.info("SHAP values computed. Shape: %s", np.array(shap_values).shape)
    return shap_values, expected, X_sample, explainer


def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    disease: str,
    top_n: int = 15,
) -> Path:
    """
    Global feature importance plot — publishable Figure 1.
    Shows mean |SHAP| per feature — what matters most overall.
    """
    # Rename columns to readable labels
    X_labeled = X_sample.rename(columns=FEATURE_LABELS)
    cols_labeled = [FEATURE_LABELS.get(c, c) for c in X_sample.columns]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    cols_labeled,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#E53935" if i >= len(importance_df) - 5 else "#90A4AE"
               for i in range(len(importance_df))]
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=colours, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value| — average impact on predicted incidence",
                  fontsize=10)
    ax.set_title(
        f"Global Feature Importance — {disease.replace('_',' ').title()} "
        f"Outbreak Predictor\n(trained on Nigerian data only, N-Step SARSA + Random Forest)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Annotate top bar
    top_bar = bars[-1]
    ax.annotate(
        f"Most important: {importance_df['feature'].iloc[-1]}",
        xy=(top_bar.get_width(), top_bar.get_y() + top_bar.get_height()/2),
        xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=8, color="#E53935",
    )

    plt.tight_layout()
    out = PLOT_DIR / f"shap_summary_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Summary plot saved: %s", out)
    return out


def plot_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    disease: str,
    top_n: int = 12,
) -> Path:
    """
    Beeswarm plot — shows direction AND magnitude of each feature.
    Red = high feature value pushes prediction UP.
    Blue = high feature value pushes prediction DOWN.
    This is your publishable Figure 2.
    """
    X_labeled = X_sample.rename(columns=FEATURE_LABELS)

    # Select top_n features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:]
    sv_top   = shap_values[:, top_idx]
    X_top    = X_sample.iloc[:, top_idx]
    X_top_labeled = X_top.rename(columns=FEATURE_LABELS)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv_top, X_top_labeled,
        plot_type="dot",
        show=False,
        color_bar=True,
        plot_size=None,
    )
    plt.title(
        f"SHAP Beeswarm — {disease.replace('_',' ').title()} Predictor\n"
        f"Red = high value increases predicted incidence, "
        f"Blue = high value decreases it",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOT_DIR / f"shap_beeswarm_{disease}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Beeswarm plot saved: %s", out)
    return out


def plot_waterfall(
    explainer,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    disease: str,
    sample_idx: int = 0,
    lga_name: str = "Sample LGA",
) -> Path:
    """
    Waterfall plot — explains ONE specific prediction.
    Shows exactly which features pushed the prediction up or down
    from the baseline (average prediction across training set).
    This is your publishable Figure 3 — single case explanation.
    """
    sv = shap_values[sample_idx]
    base = float(explainer.expected_value)
    X_row = X_sample.iloc[sample_idx]

    # Sort by absolute SHAP
    order = np.argsort(np.abs(sv))[-12:]
    features  = [FEATURE_LABELS.get(X_sample.columns[i], X_sample.columns[i])
                 for i in order]
    sv_sorted = sv[order]
    vals      = X_row.iloc[order].values

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#E53935" if v > 0 else "#1E88E5" for v in sv_sorted]

    y_pos = range(len(features))
    ax.barh(y_pos, sv_sorted, color=colours, alpha=0.85, edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{f}\n(value={v:.2f})" for f, v in zip(features, vals)],
        fontsize=8,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to prediction)", fontsize=10)
    ax.set_title(
        f"Prediction Explanation — {disease.replace('_',' ').title()}\n"
        f"LGA: {lga_name}  |  Baseline prediction: {base:.4f}  |  "
        f"Final: {base + sv.sum():.4f}",
        fontsize=10, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    red_patch  = mpatches.Patch(color="#E53935", label="Increases predicted incidence")
    blue_patch = mpatches.Patch(color="#1E88E5", label="Decreases predicted incidence")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    out = PLOT_DIR / f"shap_waterfall_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Waterfall plot saved: %s", out)
    return out


def plot_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    disease: str,
    df_aligned: pd.DataFrame,
) -> list[Path]:
    """
    Dependence plots for top 4 features — show how one feature's value
    affects its SHAP contribution, coloured by zone.
    Your publishable Figure 4.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top4_idx = np.argsort(mean_abs)[-4:]
    paths = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for plot_i, feat_idx in enumerate(top4_idx):
        feat_name   = X_sample.columns[feat_idx]
        feat_label  = FEATURE_LABELS.get(feat_name, feat_name)
        feat_vals   = X_sample.iloc[:, feat_idx].values
        shap_vals   = shap_values[:, feat_idx]

        # Zone colour for each point
        zones  = df_aligned["zone"].iloc[:len(X_sample)].values
        colours = [ZONE_COLOURS.get(z, "#90A4AE") for z in zones]

        ax = axes_flat[plot_i]
        sc = ax.scatter(
            feat_vals, shap_vals,
            c=colours, alpha=0.4, s=8, edgecolors="none",
        )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel(feat_label, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(feat_label, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Zone legend on first plot only
        if plot_i == 0:
            handles = [
                mpatches.Patch(color=c, label=z)
                for z, c in ZONE_COLOURS.items()
            ]
            ax.legend(
                handles=handles, fontsize=7,
                title="Zone", title_fontsize=7,
                loc="upper left",
            )

    fig.suptitle(
        f"Feature Dependence Plots — {disease.replace('_',' ').title()} Predictor\n"
        f"How each top feature's value drives the prediction (coloured by zone)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOT_DIR / f"shap_dependence_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Dependence plots saved: %s", out)
    return [out]


def zone_shap_breakdown(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    df_aligned: pd.DataFrame,
    disease: str,
) -> Path:
    """
    Zone-level SHAP breakdown — unique to your study.
    Shows which features matter MOST in each Nigerian zone.
    This is a novel contribution: zone-stratified explainability.
    """
    zones = df_aligned["zone"].iloc[:len(X_sample)].reset_index(drop=True)
    top5_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-5:]
    top5_names = [
        FEATURE_LABELS.get(X_sample.columns[i], X_sample.columns[i])
        for i in top5_idx
    ]

    zone_data = []
    for zone in sorted(ZONE_COLOURS.keys()):
        mask = (zones == zone).values
        if mask.sum() == 0:
            continue
        sv_zone = shap_values[mask][:, top5_idx]
        for feat_i, feat_name in enumerate(top5_names):
            zone_data.append({
                "zone":       zone,
                "feature":    feat_name,
                "mean_shap":  float(np.abs(sv_zone[:, feat_i]).mean()),
            })

    zone_df = pd.DataFrame(zone_data)

    fig, ax = plt.subplots(figsize=(11, 6))
    zones_list = sorted(ZONE_COLOURS.keys())
    x = np.arange(len(top5_names))
    width = 0.13
    for i, zone in enumerate(zones_list):
        zd = zone_df[zone_df["zone"] == zone]
        if zd.empty:
            continue
        vals = [
            zd[zd["feature"] == f]["mean_shap"].values[0]
            if f in zd["feature"].values else 0
            for f in top5_names
        ]
        ax.bar(
            x + i * width, vals, width,
            label=zone, color=ZONE_COLOURS[zone], alpha=0.85,
        )

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(top5_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(
        f"Zone-Stratified Feature Importance — {disease.replace('_',' ').title()}\n"
        f"Which factors drive predictions differ by Nigerian zone",
        fontsize=11, fontweight="bold",
    )
    ax.legend(title="Zone", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOT_DIR / f"shap_zone_{disease}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Zone SHAP breakdown saved: %s", out)
    return out


def save_shap_report(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    disease: str,
    expected_value: float,
) -> Path:
    """
    Save publishable SHAP statistics to JSON.
    Use these numbers directly in your paper's Results section.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top10 = np.argsort(mean_abs)[-10:][::-1]

    report = {
        "disease":          disease,
        "n_samples":        int(len(X_sample)),
        "baseline_prediction": float(expected_value),
        "top_features": [
            {
                "rank":        i + 1,
                "feature":     X_sample.columns[idx],
                "label":       FEATURE_LABELS.get(X_sample.columns[idx], X_sample.columns[idx]),
                "mean_abs_shap": float(mean_abs[idx]),
                "pct_of_total": float(mean_abs[idx] / mean_abs.sum() * 100),
            }
            for i, idx in enumerate(top10)
        ],
        "total_shap_magnitude": float(mean_abs.sum()),
        "lag_features_pct": float(
            sum(mean_abs[i] for i, c in enumerate(X_sample.columns)
                if "lag" in c) / mean_abs.sum() * 100
        ),
        "climate_features_pct": float(
            sum(mean_abs[i] for i, c in enumerate(X_sample.columns)
                if c in ["rainfall_mm","temp_max_c","humidity_pct","ndvi","flood_risk_flag"]
                ) / mean_abs.sum() * 100
        ),
        "ses_features_pct": float(
            sum(mean_abs[i] for i, c in enumerate(X_sample.columns)
                if c in ["poverty_headcount_pct","food_insecurity_pct",
                         "nhia_coverage_pct","piped_water_pct","sanitation_pct"]
                ) / mean_abs.sum() * 100
        ),
    }

    out = MODEL_DIR / f"shap_report_{disease}.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("SHAP report saved: %s", out)

    # Print paper-ready summary
    print(f"\n{'='*55}")
    print(f"  SHAP REPORT — {disease.upper()}")
    print(f"{'='*55}")
    print(f"  Baseline prediction : {expected_value:.4f} (avg incidence / 10k)")
    print(f"  Lag features        : {report['lag_features_pct']:.1f}% of total SHAP")
    print(f"  Climate features    : {report['climate_features_pct']:.1f}%")
    print(f"  SES features        : {report['ses_features_pct']:.1f}%")
    print(f"\n  Top 10 features:")
    for f in report["top_features"]:
        bar = "█" * int(f["pct_of_total"] / 2)
        print(f"  {f['rank']:>2}. {f['label']:<35} {f['mean_abs_shap']:.4f}  ({f['pct_of_total']:.1f}%)  {bar}")
    print(f"{'='*55}\n")
    return out


def run_full_shap(disease: str, zone: str | None = None, lga_id: int | None = None):
    """Run the complete SHAP pipeline for one disease."""
    log.info("="*55)
    log.info("SHAP Explainability — disease=%s zone=%s", disease, zone or "ALL")
    log.info("="*55)

    predictor, X, y, w, df_aligned = load_model_and_data(disease, zone)
    shap_values, expected, X_sample, explainer = compute_shap_values(predictor, X)

    # Save raw SHAP values
    shap_pkl = MODEL_DIR / f"shap_values_{disease}.pkl"
    with open(shap_pkl, "wb") as f:
        pickle.dump({
            "shap_values": shap_values,
            "expected":    expected,
            "columns":     list(X_sample.columns),
        }, f)

    # Generate all plots
    plot_summary(shap_values, X_sample, disease)
    plot_beeswarm(shap_values, X_sample, disease)

    # Waterfall for specific LGA or first sample
    sample_idx = 0
    lga_name   = "Sample LGA"
    if lga_id and "lga_id" in df_aligned.columns:
        mask = df_aligned["lga_id"] == lga_id
        if mask.any():
            sample_idx = mask.idxmax()
            lga_name   = df_aligned.loc[sample_idx, "lga_name"]
    plot_waterfall(explainer, shap_values, X_sample, disease, sample_idx, lga_name)
    plot_dependence(shap_values, X_sample, disease, df_aligned)
    zone_shap_breakdown(shap_values, X_sample, df_aligned, disease)
    save_shap_report(shap_values, X_sample, disease, float(expected))

    log.info("All SHAP outputs saved to: %s", PLOT_DIR)
    return PLOT_DIR


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for outbreak model")
    parser.add_argument("--disease",      default="malaria")
    parser.add_argument("--zone",         default=None)
    parser.add_argument("--lga-id",       type=int, default=None)
    parser.add_argument("--all-diseases", action="store_true")
    args = parser.parse_args()

    if not HAS_SHAP:
        print("ERROR: shap not installed. Run:  pip install shap matplotlib")
        return

    if args.all_diseases:
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            diseases = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
                ).fetchall()
            ]
        for d in diseases:
            try:
                run_full_shap(d, args.zone, args.lga_id)
            except Exception as exc:
                log.error("Failed for %s: %s", d, exc)
    else:
        run_full_shap(args.disease, args.zone, args.lga_id)


if __name__ == "__main__":
    main()
