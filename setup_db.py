"""
Run this ONCE to create nigeria.db from the schema.
Usage: python setup_db.py
"""

from __future__ import annotations

import os
import sqlite3

DB_PATH = "./nigeria.db"
SCHEMA_PATH = "./nigeria_db_schema.sql"


def create_schema(
    db_path: str = DB_PATH,
    schema_path: str = SCHEMA_PATH,
) -> tuple[list[str], list[str]]:
    """Create a SQLite database from the provided schema and return created tables/views."""
    # Delete old broken db if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[✓] Removed old {db_path}")

    # Read schema
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        schema_sql = schema_file.read()

    # Create fresh db
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
        print(f"[✓] Schema applied from {schema_path}")

        # Verify all tables and views created
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        views = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
        ).fetchall()
    except Exception as exc:
        print(f"[!] Schema error: {exc}")
        raise
    finally:
        conn.close()

    table_names = [row[0] for row in tables]
    view_names = [row[0] for row in views]
    return table_names, view_names


def main() -> None:
    """Create nigeria.db and print a summary."""
    tables, views = create_schema()
    print(f"\n[✓] Tables created ({len(tables)}):")
    for table_name in tables:
        print(f"    {table_name}")

    print(f"\n[✓] Views created ({len(views)}):")
    for view_name in views:
        print(f"    {view_name}")

    print(f"\n[✓] {DB_PATH} is ready! Now run: python seed_nigeria.py")


if __name__ == "__main__":
    main()
