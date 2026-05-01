"""
startup.py — Auto-build nigeria.db on Streamlit Cloud
Seeds all tables directly without needing external APIs.
"""

import os
import sqlite3
import random
import sys
from pathlib import Path
from datetime import date, timedelta

DB_PATH = os.getenv("DB_PATH", "./nigeria.db")
random.seed(42)


def db_is_ready() -> bool:
    db = Path(DB_PATH)
    if not db.exists() or db.stat().st_size < 500_000:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        lga_count = conn.execute("SELECT COUNT(*) FROM lga").fetchone()[0]
        disease_count = conn.execute("SELECT COUNT(*) FROM disease_record").fetchone()[0]
        conn.close()
        return lga_count > 100 and disease_count > 1000
    except Exception:
        return False


def run_setup():
    import importlib
    sys.path.insert(0, ".")

    print("Nigeria Health AI — Auto Database Setup")

    # Step 1: Schema
    print("[1/6] Creating schema...")
    if Path(DB_PATH).exists():
        os.remove(DB_PATH)
    with open("nigeria_db_schema.sql", "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

    # Step 2: Seed LGAs
    print("[2/6] Seeding 770 LGAs...")
    import seed_nigeria
    importlib.reload(seed_nigeria)
    seed_nigeria.seed_lga_table(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    lgas = [dict(r) for r in conn.execute(
        "SELECT lga_id, lga_name, state, zone, lga_type FROM lga"
    ).fetchall()]
    conn.close()
    print(f"  {len(lgas)} LGAs loaded")

    # Step 3: Disease records
    print("[3/6] Seeding disease records...")
    sample = random.sample(lgas, min(200, len(lgas)))
    seed_nigeria.seed_disease_records(DB_PATH, sample)

    # Step 4: Surveillance alerts
    print("[4/6] Seeding surveillance alerts...")
    _seed_alerts_direct(lgas)

    # Step 5: Socioeconomic
    print("[5/6] Seeding socioeconomic data...")
    _seed_socioeconomic_direct(lgas)

    # Step 6: Climate + feature store via data_collector
    print("[6/6] Running data collector...")
    try:
        Path("./data").mkdir(exist_ok=True)
        seed_nigeria.generate_ncdc_csv("./data/ncdc_alerts.csv", lgas)
        seed_nigeria.generate_nimet_csv("./data/nimet_climate.csv", lgas)
        import data_collector
        importlib.reload(data_collector)
        data_collector.main()
    except Exception as e:
        print(f"  data_collector partial: {e}")
        try:
            import build_feature_store
            importlib.reload(build_feature_store)
            build_feature_store.main()
        except Exception as e2:
            print(f"  feature store: {e2}")

    # Summary
    conn = sqlite3.connect(DB_PATH)
    for table in ["lga","disease_record","surveillance_alert",
                  "climate_health","socioeconomic","feature_store"]:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n:,}")
        except Exception:
            pass
    conn.close()
    print("Setup complete!")


def _seed_alerts_direct(lgas):
    DISEASES = ["malaria","cholera","typhoid","tuberculosis",
                "meningitis","lassa_fever","yellow_fever","diarrhoeal"]
    LEVELS = ["suspected","confirmed","outbreak_declared","rumour"]
    start = date(2019, 1, 1)
    delta_days = (date(2025, 6, 30) - start).days
    rows = []
    for _ in range(3000):
        lga = random.choice(lgas)
        d = start + timedelta(days=random.randint(0, delta_days))
        disease = random.choice(DISEASES)
        level = random.choices(LEVELS, weights=[0.40,0.35,0.10,0.15])[0]
        suspected = random.randint(1, 200)
        confirmed = random.randint(0, suspected) if level in ("confirmed","outbreak_declared") else 0
        deaths = random.randint(0, max(1, confirmed // 10))
        rows.append((
            lga["lga_id"], d.isoformat(),
            d.strftime("W%V"), d.year,
            disease, level, suspected, confirmed, deaths,
            f"NCDC-{d.year}-{random.randint(10000,99999)}",
            round(random.uniform(0.5, 0.95), 3), "ncdc_idsr"
        ))
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("""
        INSERT OR IGNORE INTO surveillance_alert
            (lga_id, alert_date, epi_week, epi_year, disease_category,
             alert_level, suspected_cases, confirmed_cases, deaths,
             ncdc_ref, data_quality_score, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM surveillance_alert").fetchone()[0]
    conn.close()
    print(f"  {n:,} alerts inserted")


def _seed_socioeconomic_direct(lgas):
    ZONE_SES = {
        "NE": {"poverty":(0.55,0.80),"water":(0.25,0.55),"sanitation":(0.15,0.45)},
        "NW": {"poverty":(0.50,0.75),"water":(0.30,0.60),"sanitation":(0.20,0.50)},
        "NC": {"poverty":(0.35,0.65),"water":(0.40,0.70),"sanitation":(0.30,0.60)},
        "SE": {"poverty":(0.25,0.55),"water":(0.50,0.80),"sanitation":(0.40,0.70)},
        "SS": {"poverty":(0.30,0.60),"water":(0.45,0.75),"sanitation":(0.35,0.65)},
        "SW": {"poverty":(0.20,0.50),"water":(0.55,0.85),"sanitation":(0.50,0.80)},
    }
    rows = []
    for lga in lgas:
        ses = ZONE_SES.get(lga["zone"], ZONE_SES["NC"])
        for year in range(2015, 2024):
            rows.append((
                lga["lga_id"], year,
                round(random.uniform(*ses["poverty"]), 3),
                round(random.uniform(*ses["water"]), 3),
                round(random.uniform(*ses["sanitation"]), 3),
                round(random.uniform(0.1, 0.6), 3),
                round(random.uniform(0.3, 0.9), 3),
                round(random.uniform(0.2, 0.8), 3),
                "world_bank_synthetic"
            ))
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("""
        INSERT OR IGNORE INTO socioeconomic
            (lga_id, year, poverty_headcount_pct, water_access_pct,
             sanitation_pct, nhia_coverage_pct, reporting_weight,
             health_facility_density, source)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM socioeconomic").fetchone()[0]
    conn.close()
    print(f"  {n:,} socioeconomic rows inserted")


def train_models_if_needed():
    """Train RLRF models if pkl files are missing."""
    import importlib, sys
    sys.path.insert(0, ".")
    from pathlib import Path as _P
    models_dir = _P("./models")
    models_dir.mkdir(exist_ok=True)

    DISEASES = ["malaria","cholera","typhoid","tuberculosis",
                "meningitis","lassa_fever","yellow_fever","diarrhoeal"]

    missing = [d for d in DISEASES
               if not (models_dir / f"{d}_rlrf.pkl").exists()]

    if not missing:
        print("All models already trained.")
        return

    print(f"Training {len(missing)} models: {missing}")
    try:
        import train_model
        importlib.reload(train_model)
        for disease in missing:
            print(f"  Training {disease}...")
            try:
                train_model.train(disease)
                print(f"  ✓ {disease} trained")
            except Exception as e:
                print(f"  ✗ {disease} failed: {e}")
    except Exception as e:
        print(f"train_model import failed: {e}")


if __name__ == "__main__":
    run_setup()
