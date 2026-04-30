"""
ndhs_maternal_collector.py — Nigeria DHS Maternal & Child Health Collector
===========================================================================
Ingests maternal and child health indicators from:
  - Nigeria DHS 2018 / 2021 (dhsprogram.com) — gold standard
  - DHIS2 HMIS maternal indicators (if configured)
  - CSV fallback (generated synthetic baseline if no DHS file yet)

Place your DHS dataset CSV at: data/ndhs_maternal.csv
Expected columns (flexible — script maps what it finds):
  state, lga_name, year, quarter,
  anc_coverage_pct, skilled_birth_pct, institutional_delivery_pct,
  maternal_mortality_ratio, neonatal_mortality_rate, under5_mortality_rate,
  stunting_rate_pct, wasting_rate_pct, exclusive_bf_pct, vitamin_a_coverage_pct

Run:
    python ndhs_maternal_collector.py
    python ndhs_maternal_collector.py --source synthetic   # generate baseline
    python ndhs_maternal_collector.py --source csv         # from data/ndhs_maternal.csv
    python ndhs_maternal_collector.py --dry-run
"""

from __future__ import annotations
import argparse
import logging
import os
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ndhs_maternal")

DB_PATH = os.getenv("DB_PATH", "./nigeria.db")
CSV_PATH = os.getenv("NDHS_CSV_PATH", "./data/ndhs_maternal.csv")

# ---------------------------------------------------------------------------
# Zone-level maternal health baselines (Nigeria DHS 2018 report)
# Used to generate realistic synthetic data when no CSV is available
# ---------------------------------------------------------------------------
ZONE_MATERNAL_BASELINES: dict[str, dict] = {
    "NE": {
        "anc_coverage_pct":           43.0,
        "skilled_birth_pct":          17.0,
        "institutional_delivery_pct": 15.0,
        "maternal_mortality_ratio":   1549.0,
        "neonatal_mortality_rate":    49.0,
        "under5_mortality_rate":      198.0,
        "stunting_rate_pct":          58.0,
        "wasting_rate_pct":           18.0,
        "exclusive_bf_pct":           27.0,
        "vitamin_a_coverage_pct":     55.0,
    },
    "NW": {
        "anc_coverage_pct":           47.0,
        "skilled_birth_pct":          13.0,
        "institutional_delivery_pct": 12.0,
        "maternal_mortality_ratio":   1787.0,
        "neonatal_mortality_rate":    52.0,
        "under5_mortality_rate":      212.0,
        "stunting_rate_pct":          60.0,
        "wasting_rate_pct":           20.0,
        "exclusive_bf_pct":           25.0,
        "vitamin_a_coverage_pct":     52.0,
    },
    "NC": {
        "anc_coverage_pct":           67.0,
        "skilled_birth_pct":          48.0,
        "institutional_delivery_pct": 45.0,
        "maternal_mortality_ratio":   879.0,
        "neonatal_mortality_rate":    37.0,
        "under5_mortality_rate":      131.0,
        "stunting_rate_pct":          42.0,
        "wasting_rate_pct":           12.0,
        "exclusive_bf_pct":           33.0,
        "vitamin_a_coverage_pct":     68.0,
    },
    "SW": {
        "anc_coverage_pct":           87.0,
        "skilled_birth_pct":          79.0,
        "institutional_delivery_pct": 76.0,
        "maternal_mortality_ratio":   321.0,
        "neonatal_mortality_rate":    22.0,
        "under5_mortality_rate":      79.0,
        "stunting_rate_pct":          21.0,
        "wasting_rate_pct":           5.0,
        "exclusive_bf_pct":           41.0,
        "vitamin_a_coverage_pct":     78.0,
    },
    "SE": {
        "anc_coverage_pct":           84.0,
        "skilled_birth_pct":          72.0,
        "institutional_delivery_pct": 70.0,
        "maternal_mortality_ratio":   398.0,
        "neonatal_mortality_rate":    26.0,
        "under5_mortality_rate":      88.0,
        "stunting_rate_pct":          24.0,
        "wasting_rate_pct":           6.0,
        "exclusive_bf_pct":           39.0,
        "vitamin_a_coverage_pct":     74.0,
    },
    "SS": {
        "anc_coverage_pct":           82.0,
        "skilled_birth_pct":          63.0,
        "institutional_delivery_pct": 60.0,
        "maternal_mortality_ratio":   487.0,
        "neonatal_mortality_rate":    29.0,
        "under5_mortality_rate":      97.0,
        "stunting_rate_pct":          28.0,
        "wasting_rate_pct":           7.0,
        "exclusive_bf_pct":           38.0,
        "vitamin_a_coverage_pct":     71.0,
    },
}

# Year-over-year improvement rates (% per year, approximate)
ANNUAL_IMPROVEMENT: dict[str, float] = {
    "anc_coverage_pct":           1.2,
    "skilled_birth_pct":          1.5,
    "institutional_delivery_pct": 1.4,
    "maternal_mortality_ratio":   -18.0,   # negative = improving
    "neonatal_mortality_rate":    -0.8,
    "under5_mortality_rate":      -2.1,
    "stunting_rate_pct":          -0.6,
    "wasting_rate_pct":           -0.3,
    "exclusive_bf_pct":           0.8,
    "vitamin_a_coverage_pct":     1.0,
}

INDICATOR_COLS = list(ANNUAL_IMPROVEMENT.keys())


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def generate_synthetic(
    lgas: list[sqlite3.Row],
    years: list[int],
    quarters: list[int],
    seed: int = 42,
) -> list[dict]:
    """
    Generate realistic synthetic maternal health data from DHS zone baselines.
    Used when no real DHS CSV is available — gives the maternal_health table
    enough rows to start model training.

    Each LGA gets small random variation (±5%) around its zone baseline
    so the data is heterogeneous rather than flat.
    """
    random.seed(seed)
    BASE_YEAR = 2018  # DHS 2018 is the baseline reference
    rows = []

    for lga in lgas:
        zone = lga["zone"]
        baseline = ZONE_MATERNAL_BASELINES.get(zone, ZONE_MATERNAL_BASELINES["NC"])
        # LGA-level noise factor (±5%) — fixed per LGA for consistency
        noise = {col: random.uniform(-0.05, 0.05) for col in INDICATOR_COLS}

        for year in years:
            for quarter in quarters:
                row: dict = {
                    "lga_id":  lga["lga_id"],
                    "year":    year,
                    "quarter": quarter,
                    "source":  "manual",
                    "data_quality_score": 0.45,  # low — synthetic data
                }
                year_delta = year - BASE_YEAR
                for col in INDICATOR_COLS:
                    base_val = baseline[col]
                    improvement = ANNUAL_IMPROVEMENT[col] * year_delta
                    lga_noise = base_val * noise[col]
                    # Quarterly variation (smaller than annual)
                    q_noise = base_val * random.uniform(-0.02, 0.02)
                    raw = base_val + improvement + lga_noise + q_noise

                    # Apply sensible bounds per indicator
                    if col.endswith("_pct"):
                        raw = _clamp(raw, 0.0, 100.0)
                    elif col == "maternal_mortality_ratio":
                        raw = _clamp(raw, 50.0, 2500.0)
                    elif col.endswith("_rate"):
                        raw = _clamp(raw, 0.0, 300.0)

                    row[col] = round(raw, 1)
                rows.append(row)

    return rows


def ingest_csv(csv_path: str, lgas: list[sqlite3.Row]) -> list[dict]:
    """Ingest real DHS/DHIS2 CSV data."""
    if not HAS_PANDAS:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)
    log.info("CSV loaded: %d rows, columns: %s", len(df), list(df.columns))

    # Build LGA name → id lookup (case-insensitive)
    lga_map = {row["lga_name"].lower(): row["lga_id"] for row in lgas}

    rows = []
    skipped = 0
    for _, r in df.iterrows():
        lga_name = str(r.get("lga_name", "")).lower().strip()
        lga_id = lga_map.get(lga_name)
        if lga_id is None:
            skipped += 1
            continue

        row = {
            "lga_id":  lga_id,
            "year":    int(r.get("year", 2021)),
            "quarter": int(r.get("quarter", 1)),
            "source":  "ndhs",
            "data_quality_score": 0.85,
        }
        for col in INDICATOR_COLS:
            val = r.get(col)
            row[col] = float(val) if pd.notna(val) else None
        rows.append(row)

    log.info("CSV parse: %d valid rows, %d skipped (LGA not found)", len(rows), skipped)
    return rows


def upsert_maternal(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    conflict = "lga_id, year, quarter"
    updates = ", ".join(
        f"{c}=excluded.{c}" for c in cols if c not in ("lga_id", "year", "quarter")
    )
    sql = (
        f"INSERT INTO maternal_health ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
    )
    values = [tuple(r[c] for c in cols) for r in rows]
    conn.executemany(sql, values)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Populate maternal_health table")
    parser.add_argument(
        "--source", choices=["csv", "synthetic", "auto"], default="auto",
        help="auto=use CSV if present, else synthetic (default: auto)"
    )
    parser.add_argument("--years", nargs="*", type=int,
                        default=list(range(2015, 2024)))
    parser.add_argument("--quarters", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    with connect(args.db) as conn:
        lgas = conn.execute(
            "SELECT lga_id, lga_name, state, zone FROM lga"
        ).fetchall()
    log.info("LGAs loaded: %d", len(lgas))

    # Decide source
    csv_path = Path(CSV_PATH)
    source = args.source
    if source == "auto":
        source = "csv" if csv_path.exists() else "synthetic"
    log.info("Source selected: %s", source)

    if source == "csv":
        rows = ingest_csv(str(csv_path), lgas)
    else:
        log.info(
            "Generating synthetic data for %d LGAs × %d years × %d quarters",
            len(lgas), len(args.years), len(args.quarters),
        )
        rows = generate_synthetic(lgas, args.years, args.quarters)

    log.info("Total rows to write: %d", len(rows))

    if args.dry_run:
        log.info("DRY RUN — not writing")
        for r in rows[:2]:
            print(r)
        return

    BATCH = 500
    total = 0
    with connect(args.db) as conn:
        for i in range(0, len(rows), BATCH):
            written = upsert_maternal(conn, rows[i:i + BATCH])
            total += written
            log.info("  Written %d / %d", total, len(rows))

    log.info("Done. maternal_health rows written: %d", total)

    with connect(args.db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM maternal_health").fetchone()[0]
    log.info("Verified: maternal_health row count = %d", count)


if __name__ == "__main__":
    main()
