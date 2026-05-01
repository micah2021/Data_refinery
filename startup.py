"""
startup.py — Auto-build nigeria.db on Streamlit Cloud
======================================================
Called from app.py on first run. Creates and seeds the database
from scratch using synthetic data (reproducible with random.seed(42)).

Steps:
1. Create schema from nigeria_db_schema.sql
2. Seed all tables via seed_nigeria.py functions
3. Run data_collector.py to populate feature_store
"""

import os
import sqlite3
import random
import csv
import sys
from pathlib import Path
from datetime import date, timedelta

DB_PATH = os.getenv("DB_PATH", "./nigeria.db")
SCHEMA_PATH = "./nigeria_db_schema.sql"

random.seed(42)


def db_is_ready() -> bool:
    """Check if database exists and has meaningful data."""
    db = Path(DB_PATH)
    if not db.exists() or db.stat().st_size < 1_000_000:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM lga").fetchone()[0]
        conn.close()
        return count > 100
    except Exception:
        return False


def create_schema():
    """Create database schema from SQL file."""
    if Path(DB_PATH).exists():
        os.remove(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    print("✓ Schema created")


def run_setup():
    """Full database setup — called once on Streamlit Cloud."""
    print("=" * 50)
    print("Nigeria Health AI — Auto Database Setup")
    print("=" * 50)

    # Step 1: Schema
    print("\n[1/4] Creating schema...")
    create_schema()

    # Step 2: Seed LGAs + disease records
    print("\n[2/4] Seeding LGA and disease data...")
    try:
        # Import and run seed functions directly
        sys.path.insert(0, ".")
        import seed_nigeria
        import importlib
        importlib.reload(seed_nigeria)
        seed_nigeria.main()
        print("✓ Seed data inserted")
    except Exception as e:
        print(f"  seed_nigeria failed: {e} — running minimal seed")
        _minimal_seed()

    # Step 3: Run data collector
    print("\n[3/4] Running data collector...")
    try:
        import data_collector
        importlib.reload(data_collector)
        data_collector.main()
        print("✓ Data collector complete")
    except Exception as e:
        print(f"  data_collector failed: {e} — skipping")

    # Step 4: Build feature store
    print("\n[4/4] Building feature store...")
    try:
        import build_feature_store
        importlib.reload(build_feature_store)
        build_feature_store.main()
        print("✓ Feature store built")
    except Exception as e:
        print(f"  build_feature_store failed: {e} — skipping")

    # Verify
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute(
        "SELECT SUM(cnt) FROM ("
        "SELECT COUNT(*) as cnt FROM lga UNION ALL "
        "SELECT COUNT(*) FROM disease_record UNION ALL "
        "SELECT COUNT(*) FROM feature_store)"
    ).fetchone()[0]
    conn.close()
    print(f"\n✅ Database ready — {total:,} total rows across key tables")


def _minimal_seed():
    """Fallback minimal seed if seed_nigeria.py fails."""
    from seed_nigeria import (
        seed_lga_table, generate_ncdc_csv,
        generate_nimet_csv, seed_disease_records, STATES
    )
    seed_lga_table(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    lgas = [dict(r) for r in conn.execute(
        "SELECT lga_id, lga_name, state, zone, lga_type FROM lga"
    ).fetchall()]
    conn.close()
    Path("./data").mkdir(exist_ok=True)
    generate_ncdc_csv("./data/ncdc_alerts.csv", lgas)
    generate_nimet_csv("./data/nimet_climate.csv", lgas)
    sample = random.sample(lgas, min(100, len(lgas)))
    seed_disease_records(DB_PATH, sample)


if __name__ == "__main__":
    run_setup()
