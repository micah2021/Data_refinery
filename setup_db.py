"""
Run this ONCE to create nigeria.db from the schema.
Usage: python setup_db.py
"""
import sqlite3
import os

DB_PATH = "./nigeria.db"
SCHEMA_PATH = "./nigeria_db_schema.sql"

# Delete old broken db if exists
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"[✓] Removed old {DB_PATH}")

# Read schema
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema_sql = f.read()

# Create fresh db
conn = sqlite3.connect(DB_PATH)
try:
    conn.executescript(schema_sql)
    conn.commit()
    print(f"[✓] Schema applied from {SCHEMA_PATH}")
except Exception as e:
    print(f"[!] Schema error: {e}")

# Verify all tables created
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()
views = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
).fetchall()

print(f"\n[✓] Tables created ({len(tables)}):")
for t in tables:
    print(f"    {t[0]}")

print(f"\n[✓] Views created ({len(views)}):")
for v in views:
    print(f"    {v[0]}")

conn.close()
print(f"\n[✓] {DB_PATH} is ready! Now run: python seed_nigeria.py")
