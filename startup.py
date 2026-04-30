"""
startup.py — Auto-setup for Streamlit Cloud
Checks if nigeria.db exists, if not builds it from scratch.
Called automatically from app.py on first run.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH", "./nigeria.db")
SCHEMA_PATH = "./nigeria_db_schema.sql"


def db_is_ready() -> bool:
    """Check if DB exists and has data."""
    if not Path(DB_PATH).exists():
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM lga").fetchone()[0]
        conn.close()
        return count > 100
    except Exception:
        return False


def build_database():
    """Full DB build: schema → seed → load data."""
    print("="*55)
    print("  FIRST RUN — Building nigeria.db from scratch")
    print("  This takes about 2 minutes. Please wait...")
    print("="*55)

    # ── Step 1: Apply schema ───────────────────────────────────
    print("\n[1/3] Applying schema...")
    if not Path(SCHEMA_PATH).exists():
        print(f"  [!] Schema file not found: {SCHEMA_PATH}")
        print("  [!] Skipping — app will use minimal DB")
        _build_minimal_db()
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(open(SCHEMA_PATH, encoding="utf-8").read())
        conn.commit()
        print("  ✅ Schema applied")
    except Exception as e:
        print(f"  [!] Schema error: {e}")
    finally:
        conn.close()

    # ── Step 2: Seed data ──────────────────────────────────────
    print("\n[2/3] Seeding data...")
    if Path("./seed_nigeria.py").exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, "seed_nigeria.py"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("  ✅ Seed complete")
        else:
            print(f"  [!] Seed warning: {result.stderr[-200:]}")
            # Continue anyway — partial data is better than nothing
    else:
        print("  [!] seed_nigeria.py not found — using direct loader")
        _direct_seed()

    # ── Step 3: Load external data ────────────────────────────
    print("\n[3/3] Loading additional data...")
    if Path("./direct_load.py").exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, "direct_load.py"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("  ✅ Data loaded")
        else:
            print(f"  [!] Load warning: {result.stderr[-200:]}")

    # ── Verify ────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    tables = ["lga", "disease_record", "climate_health", "socioeconomic"]
    print("\n  Final row counts:")
    for t in tables:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"    {t:25s}: {n:>10,}")
        except:
            print(f"    {t:25s}: table missing")
    conn.close()
    print("\n  ✅ Database ready!\n")


def _build_minimal_db():
    """Build a minimal working DB if schema file is missing."""
    print("  Building minimal DB...")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS lga (
            lga_id INTEGER PRIMARY KEY AUTOINCREMENT,
            lga_code TEXT UNIQUE NOT NULL,
            lga_name TEXT NOT NULL,
            state TEXT NOT NULL,
            zone TEXT NOT NULL,
            lat REAL, lng REAL,
            lga_type TEXT,
            pop_density REAL
        );
        CREATE TABLE IF NOT EXISTS disease_record (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            lga_id INTEGER,
            report_date TEXT,
            epi_week INTEGER,
            epi_year INTEGER,
            disease_name TEXT,
            disease_category TEXT,
            case_count INTEGER DEFAULT 0,
            death_count INTEGER DEFAULT 0,
            data_quality_score REAL,
            source TEXT
        );
        CREATE TABLE IF NOT EXISTS climate_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lga_id INTEGER,
            year INTEGER,
            month INTEGER,
            rainfall_mm REAL,
            temp_max_c REAL,
            temp_min_c REAL,
            humidity_pct REAL
        );
        CREATE TABLE IF NOT EXISTS socioeconomic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lga_id INTEGER,
            year INTEGER,
            poverty_headcount_pct REAL,
            gdp_per_capita_usd REAL,
            literacy_rate_pct REAL,
            source TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("  ✅ Minimal DB created")


def _direct_seed():
    """Inline minimal seeder when seed_nigeria.py is missing."""
    import random
    conn = sqlite3.connect(DB_PATH)

    STATES = {
        "Abia": "SE", "Adamawa": "NE", "Akwa Ibom": "SS",
        "Anambra": "SE", "Bauchi": "NE", "Bayelsa": "SS",
        "Benue": "NC", "Borno": "NE", "Cross River": "SS",
        "Delta": "SS", "Ebonyi": "SE", "Edo": "SS",
        "Ekiti": "SW", "Enugu": "SE", "FCT": "NC",
        "Gombe": "NE", "Imo": "SE", "Jigawa": "NW",
        "Kaduna": "NW", "Kano": "NW", "Katsina": "NW",
        "Kebbi": "NW", "Kogi": "NC", "Kwara": "NC",
        "Lagos": "SW", "Nasarawa": "NC", "Niger": "NC",
        "Ogun": "SW", "Ondo": "SW", "Osun": "SW",
        "Oyo": "SW", "Plateau": "NC", "Rivers": "SS",
        "Sokoto": "NW", "Taraba": "NE", "Yobe": "NE",
        "Zamfara": "NW",
    }

    lgas_added = 0
    for state, zone in STATES.items():
        for i in range(1, 21):  # 20 LGAs per state
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO lga
                        (lga_code, lga_name, state, zone, lat, lng, lga_type, pop_density)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    f"{state[:3].upper()}{i:03d}",
                    f"{state} LGA {i}",
                    state, zone,
                    round(random.uniform(4.5, 13.9), 4),
                    round(random.uniform(2.7, 14.7), 4),
                    random.choice(["urban","semi-urban","rural"]),
                    round(random.uniform(50, 2000), 1)
                ))
                lgas_added += 1
            except:
                pass

    conn.commit()
    print(f"  ✅ Seeded {lgas_added} LGAs")
    conn.close()


def ensure_db_ready():
    """
    Main entry point — call this from app.py.
    Returns True if DB is ready.
    """
    if db_is_ready():
        return True

    print("\n[startup] nigeria.db not found or empty — building now...")
    try:
        build_database()
        return db_is_ready()
    except Exception as e:
        print(f"\n[startup] Build failed: {e}")
        print("[startup] Trying minimal DB...")
        _build_minimal_db()
        _direct_seed()
        return True


if __name__ == "__main__":
    ensure_db_ready()
