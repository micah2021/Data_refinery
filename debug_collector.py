"""
debug_collector.py
Run this to find the exact error preventing WorldBank and NCDC from writing.
Usage: python debug_collector.py
"""
import sqlite3
import csv
import os
import requests
import time

DB_PATH = "./nigeria.db"

# ── Test 1: WorldBank socioeconomic write ─────────────────────
print("="*55)
print("TEST 1: WorldBank → socioeconomic")
print("="*55)
try:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    # Get one LGA
    lga = conn.execute("SELECT lga_id FROM lga LIMIT 1").fetchone()
    lga_id = lga["lga_id"]
    print(f"  Using lga_id={lga_id}")

    # Try inserting one row
    test_row = {
        "lga_id": lga_id,
        "year": 2023,
        "poverty_headcount_pct": 40.5,
        "gdp_per_capita_usd": 2100.0,
        "gini_coefficient": 0.43,
        "literacy_rate_pct": 62.0,
        "piped_water_pct": 29.0,
        "sanitation_pct": 38.0,
        "electricity_access_pct": 55.0,
        "source": "world_bank",
    }

    conn.execute("""
        INSERT OR REPLACE INTO socioeconomic
            (lga_id, year, poverty_headcount_pct, gdp_per_capita_usd,
             gini_coefficient, literacy_rate_pct, piped_water_pct,
             sanitation_pct, electricity_access_pct, source)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        test_row["lga_id"], test_row["year"],
        test_row["poverty_headcount_pct"], test_row["gdp_per_capita_usd"],
        test_row["gini_coefficient"], test_row["literacy_rate_pct"],
        test_row["piped_water_pct"], test_row["sanitation_pct"],
        test_row["electricity_access_pct"], test_row["source"],
    ))
    conn.commit()
    print("  [OK] Direct insert into socioeconomic works!")

    # Now test ON CONFLICT upsert (same as collector uses)
    cols = list(test_row.keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    conflict = "lga_id, year"
    update_cols = [c for c in cols if c not in ["lga_id", "year"]]
    sql = (
        f"INSERT INTO socioeconomic ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    )
    print(f"\n  SQL: {sql[:100]}...")
    conn.execute(sql, tuple(test_row[c] for c in cols))
    conn.commit()
    print("  [OK] ON CONFLICT upsert also works!")
    conn.close()

except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ── Test 2: NCDC surveillance_alert write ─────────────────────
print("\n" + "="*55)
print("TEST 2: NCDC → surveillance_alert")
print("="*55)
try:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    lga = conn.execute("SELECT lga_id FROM lga LIMIT 1").fetchone()
    lga_id = lga["lga_id"]

    # Try direct insert
    conn.execute("""
        INSERT OR IGNORE INTO surveillance_alert
            (lga_id, alert_date, disease, alert_level,
             suspected_cases, confirmed_cases, deaths)
        VALUES (?,?,?,?,?,?,?)
    """, (lga_id, "2024-01-15", "malaria", "suspected", 10, 5, 0))
    conn.commit()
    print("  [OK] Direct insert into surveillance_alert works!")

    # Check CSV file
    csv_path = "./data/ncdc_alerts.csv"
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader)
            print(f"\n  CSV columns: {list(first.keys())}")
            print(f"  First row: {dict(first)}")

            # Check if lga_name matches any LGA in db
            lga_names = {r["lga_name"].lower() for r in
                        conn.execute("SELECT lga_name FROM lga").fetchall()}
            csv_lga = first.get("lga_name", "").lower()
            print(f"\n  CSV lga_name '{csv_lga}' found in DB: {csv_lga in lga_names}")
    conn.close()

except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ── Test 3: Check table row counts ────────────────────────────
print("\n" + "="*55)
print("TEST 3: Current database row counts")
print("="*55)
conn = sqlite3.connect(DB_PATH)
for table in ["lga", "disease_record", "surveillance_alert",
              "climate_health", "socioeconomic", "data_quality_log"]:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:30s}: {count:>10,} rows")
    except Exception as e:
        print(f"  {table:30s}: ERROR — {e}")
conn.close()
