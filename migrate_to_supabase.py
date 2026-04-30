"""
migrate_to_supabase.py — Migrate nigeria.db (SQLite) → Supabase (PostgreSQL)
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

SQLITE_PATH = "./nigeria.db"

DB_USER     = "postgres.dnjqokmellloqmjcczgq"
DB_PASSWORD = quote_plus("GOdwinmicah@2023")
DB_HOST     = "aws-0-eu-west-1.pooler.supabase.com"
DB_PORT     = "6543"   # transaction pooler port
DB_NAME     = "postgres"

SUPABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

TABLES = [
    "lga", "disease_record", "surveillance_alert",
    "climate_health", "socioeconomic", "maternal_health",
    "data_quality_log", "feature_store",
]

print("Connecting to SQLite...")
sqlite_conn = sqlite3.connect(SQLITE_PATH)

print(f"Connecting to Supabase on port {DB_PORT}...")
engine = create_engine(
    SUPABASE_URL,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True,
)

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✓ Supabase connection successful!\n")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    # Try port 5432 as fallback
    print("\nTrying port 5432...")
    SUPABASE_URL2 = SUPABASE_URL.replace(":6543/", ":5432/")
    engine = create_engine(SUPABASE_URL2, connect_args={"sslmode": "require"})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Connected on port 5432!\n")
    except Exception as e2:
        print(f"✗ Both ports failed: {e2}")
        print("\nTry going to Supabase dashboard → Settings → Database")
        print("and check which port is shown under 'Connection string'")
        exit(1)

for table in TABLES:
    print(f"  Migrating {table}...")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", sqlite_conn)
        print(f"    Loaded {len(df):,} rows from SQLite")
        if df.empty:
            print(f"    Skipping — no data")
            continue
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=500, method="multi")
        print(f"    ✓ {len(df):,} rows → Supabase")
    except Exception as e:
        print(f"    ERROR on {table}: {e}")

sqlite_conn.close()
print("\n✅ Migration complete!")
