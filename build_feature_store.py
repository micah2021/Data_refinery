"""
build_feature_store.py — Populate feature_store from raw tables
===============================================================
Runs after fix_socioeconomic.py and ndhs_maternal_collector.py.
Rebuilds feature_store with all lag features, climate signals,
SES scores, and reporting weights.

Run:
    python build_feature_store.py
    python build_feature_store.py --dry-run
    python build_feature_store.py --disease malaria cholera
"""

from __future__ import annotations
import argparse
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

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
log = logging.getLogger("feature_builder")

DB_PATH = os.getenv("DB_PATH", "./nigeria.db")


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = -64000")   # 64 MB cache for large joins
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_prerequisites(conn: sqlite3.Connection) -> list[str]:
    """Warn if prerequisite tables are thin."""
    warnings = []
    checks = [
        ("disease_record",  1000,  "Run data_collector.py first"),
        ("socioeconomic",   500,   "Run fix_socioeconomic.py first"),
        ("climate_health",  1000,  "Check FAO/NiMET collector"),
        ("maternal_health", 100,   "Run ndhs_maternal_collector.py first"),
        ("lga",             36,    "Run seed_nigeria.py first"),
    ]
    for table, min_rows, advice in checks:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count < min_rows:
            warnings.append(f"{table} has only {count} rows — {advice}")
        else:
            log.info("  %-20s %6d rows OK", table, count)
    return warnings


def build_core_features(conn: sqlite3.Connection, diseases: list[str] | None) -> int:
    """Main feature engineering query — joins all raw tables into feature_store."""
    log.info("Clearing existing feature_store...")
    conn.execute("DELETE FROM feature_store")

    disease_filter = ""
    if diseases:
        placeholders = ",".join(f"'{d}'" for d in diseases)
        disease_filter = f"AND dr.disease_category IN ({placeholders})"

    sql = f"""
    INSERT INTO feature_store (
        lga_id, epi_year, epi_week, disease_category,
        incidence_rate,
        rainfall_mm, temp_max_c, humidity_pct, ndvi,
        season, flood_risk_flag,
        poverty_headcount_pct, food_insecurity_pct,
        nhia_coverage_pct, piped_water_pct, sanitation_pct,
        zone, lga_type, pop_density,
        reporting_weight, completeness_pct,
        active_alert_flag, alert_level
    )
    SELECT
        dr.lga_id,
        dr.epi_year,
        dr.epi_week,
        dr.disease_category,

        -- incidence rate per 10,000 population
        CASE
            WHEN l.pop_density IS NOT NULL
                 AND l.area_km2 IS NOT NULL
                 AND l.pop_density * l.area_km2 > 0
            THEN CAST(SUM(dr.case_count) AS REAL) /
                 (l.pop_density * l.area_km2 / 10000.0)
            ELSE CAST(SUM(dr.case_count) AS REAL)
        END AS incidence_rate,

        -- climate (match on year + month derived from epi_week)
        AVG(ch.rainfall_mm)      AS rainfall_mm,
        AVG(ch.temp_max_c)       AS temp_max_c,
        AVG(ch.humidity_pct)     AS humidity_pct,
        AVG(ch.ndvi)             AS ndvi,
        MAX(ch.season)           AS season,
        MAX(ch.flood_risk_flag)  AS flood_risk_flag,

        -- socioeconomic (match on year)
        AVG(se.poverty_headcount_pct)    AS poverty_headcount_pct,
        AVG(se.food_insecurity_pct)      AS food_insecurity_pct,
        AVG(se.nhia_coverage_pct)        AS nhia_coverage_pct,
        AVG(se.piped_water_pct)          AS piped_water_pct,
        AVG(se.sanitation_pct)           AS sanitation_pct,

        -- geography
        l.zone,
        l.lga_type,
        l.pop_density,

        -- data quality weight (down-weights imputed/low-quality records)
        COALESCE(AVG(dr.data_quality_score), 0.5) AS reporting_weight,
        COALESCE(dql.completeness_pct, 50.0)       AS completeness_pct,

        -- outbreak alert signal
        CASE WHEN sa.alert_id IS NOT NULL THEN 1 ELSE 0 END AS active_alert_flag,
        sa.alert_level

    FROM disease_record dr
    JOIN lga l
        ON l.lga_id = dr.lga_id
    LEFT JOIN climate_health ch
        ON ch.lga_id = dr.lga_id
        AND ch.year  = dr.epi_year
        AND ch.month = CAST(
            strftime('%m',
                CASE WHEN dr.report_date IS NOT NULL
                     THEN dr.report_date
                     ELSE date(dr.epi_year || '-01-01',
                               '+' || ((dr.epi_week - 1) * 7) || ' days')
                END
            ) AS INTEGER
        )
    LEFT JOIN socioeconomic se
        ON se.lga_id = dr.lga_id
        AND se.year  = dr.epi_year
    LEFT JOIN (
        SELECT lga_id,
               AVG(completeness_pct) AS completeness_pct
        FROM data_quality_log
        WHERE table_name = 'disease_record'
        GROUP BY lga_id
    ) dql ON dql.lga_id = dr.lga_id
    LEFT JOIN surveillance_alert sa
        ON sa.lga_id   = dr.lga_id
        AND sa.disease  = dr.disease_category
        AND sa.alert_date = dr.report_date
    WHERE dr.epi_year IS NOT NULL
      AND dr.epi_week IS NOT NULL
      {disease_filter}
    GROUP BY
        dr.lga_id, dr.epi_year, dr.epi_week, dr.disease_category,
        l.zone, l.lga_type, l.pop_density, ch.season,
        dql.completeness_pct, sa.alert_id, sa.alert_level
    """

    log.info("Running core feature query (this may take 30–60s on large datasets)...")
    conn.execute(sql)
    count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
    log.info("Core features inserted: %d rows", count)
    return count


def build_lag_features(conn: sqlite3.Connection) -> None:
    """
    Add autoregressive lag features.
    These are the strongest predictors in outbreak models —
    last week's incidence predicts this week's far better than
    any demographic variable.
    """
    log.info("Computing lag features...")

    # 1-week lag
    conn.execute("""
    UPDATE feature_store AS fs
    SET incidence_lag_1w = (
        SELECT f2.incidence_rate
        FROM feature_store f2
        WHERE f2.lga_id          = fs.lga_id
          AND f2.disease_category = fs.disease_category
          AND f2.epi_year         = CASE WHEN fs.epi_week > 1
                                         THEN fs.epi_year
                                         ELSE fs.epi_year - 1 END
          AND f2.epi_week         = CASE WHEN fs.epi_week > 1
                                         THEN fs.epi_week - 1
                                         ELSE 52 END
        LIMIT 1
    )
    """)

    # 2-week lag
    conn.execute("""
    UPDATE feature_store AS fs
    SET incidence_lag_2w = (
        SELECT f2.incidence_rate
        FROM feature_store f2
        WHERE f2.lga_id          = fs.lga_id
          AND f2.disease_category = fs.disease_category
          AND f2.epi_year         = CASE WHEN fs.epi_week > 2
                                         THEN fs.epi_year
                                         ELSE fs.epi_year - 1 END
          AND f2.epi_week         = CASE WHEN fs.epi_week > 2
                                         THEN fs.epi_week - 2
                                         ELSE 52 - (2 - fs.epi_week) END
        LIMIT 1
    )
    """)

    # 4-week lag
    conn.execute("""
    UPDATE feature_store AS fs
    SET incidence_lag_4w = (
        SELECT f2.incidence_rate
        FROM feature_store f2
        WHERE f2.lga_id          = fs.lga_id
          AND f2.disease_category = fs.disease_category
          AND f2.epi_year         = CASE WHEN fs.epi_week > 4
                                         THEN fs.epi_year
                                         ELSE fs.epi_year - 1 END
          AND f2.epi_week         = CASE WHEN fs.epi_week > 4
                                         THEN fs.epi_week - 4
                                         ELSE 52 - (4 - fs.epi_week) END
        LIMIT 1
    )
    """)

    # 8-week lag
    conn.execute("""
    UPDATE feature_store AS fs
    SET incidence_lag_8w = (
        SELECT f2.incidence_rate
        FROM feature_store f2
        WHERE f2.lga_id          = fs.lga_id
          AND f2.disease_category = fs.disease_category
          AND f2.epi_year         = CASE WHEN fs.epi_week > 8
                                         THEN fs.epi_year
                                         ELSE fs.epi_year - 1 END
          AND f2.epi_week         = CASE WHEN fs.epi_week > 8
                                         THEN fs.epi_week - 8
                                         ELSE 52 - (8 - fs.epi_week) END
        LIMIT 1
    )
    """)

    lag_count = conn.execute(
        "SELECT COUNT(*) FROM feature_store WHERE incidence_lag_1w IS NOT NULL"
    ).fetchone()[0]
    log.info("Lag features computed: %d rows have lag_1w", lag_count)


def print_summary(conn: sqlite3.Connection) -> None:
    """Print a summary of what the feature_store now contains."""
    total = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
    print("\n" + "="*52)
    print("  FEATURE STORE SUMMARY")
    print("="*52)
    print(f"  Total rows        : {total:,}")

    diseases = conn.execute(
        "SELECT disease_category, COUNT(*) as n FROM feature_store GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    print(f"\n  {'Disease':<28} {'Rows':>8}")
    print(f"  {'-'*28} {'-'*8}")
    for d in diseases:
        print(f"  {d[0]:<28} {d[1]:>8,}")

    zones = conn.execute(
        "SELECT zone, COUNT(*) as n FROM feature_store GROUP BY 1 ORDER BY 1"
    ).fetchall()
    print(f"\n  {'Zone':<10} {'Rows':>8}")
    print(f"  {'-'*10} {'-'*8}")
    for z in zones:
        print(f"  {z[0]:<10} {z[1]:>8,}")

    lag_pct = conn.execute(
        "SELECT ROUND(100.0 * COUNT(incidence_lag_1w) / COUNT(*), 1) FROM feature_store"
    ).fetchone()[0]
    print(f"\n  Lag feature coverage: {lag_pct}%")
    print("="*52 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build feature_store from raw tables")
    parser.add_argument("--disease", nargs="*", help="Filter to specific diseases")
    parser.add_argument("--skip-lags", action="store_true", help="Skip lag feature computation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    log.info("="*52)
    log.info("Feature Store Builder starting")
    log.info("DB: %s", args.db)
    log.info("="*52)

    with connect(args.db) as conn:
        log.info("Checking prerequisites...")
        warnings = check_prerequisites(conn)
        for w in warnings:
            log.warning("PREREQ WARNING: %s", w)

        if args.dry_run:
            log.info("DRY RUN — prerequisites checked, no data written")
            return

        started = datetime.utcnow()
        count = build_core_features(conn, args.disease)

        if not args.skip_lags and count > 0:
            build_lag_features(conn)

        elapsed = (datetime.utcnow() - started).seconds
        log.info("Feature store built in %ds", elapsed)

        print_summary(conn)


if __name__ == "__main__":
    main()
