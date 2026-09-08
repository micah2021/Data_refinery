"""Validate expected SQLite schema for nigeria.db using a small YAML contract."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - explicit runtime guidance
    raise SystemExit("PyYAML is required. Install with: pip install PyYAML") from exc


def load_schema_config(schema_path: Path) -> dict:
    """Load and return validation rules from YAML."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as schema_file:
        return yaml.safe_load(schema_file) or {}


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return a set of column names for a table."""
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def validate(db_path: Path, schema_path: Path) -> list[str]:
    """Validate expected tables/columns and optional non-empty table checks."""
    config = load_schema_config(schema_path)
    expected_tables = config.get("expected_tables", {})
    non_empty_tables = config.get("non_empty_tables", [])

    errors: list[str] = []
    with sqlite3.connect(db_path) as conn:
        for table_name, expected_columns in expected_tables.items():
            columns = get_table_columns(conn, table_name)
            if not columns:
                errors.append(f"Missing table: {table_name}")
                continue

            missing_columns = [col for col in expected_columns if col not in columns]
            if missing_columns:
                errors.append(
                    f"Table '{table_name}' missing columns: {', '.join(sorted(missing_columns))}"
                )

        for table_name in non_empty_tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if count <= 0:
                errors.append(f"Table '{table_name}' is empty; expected at least one row")

    return errors


def main() -> int:
    schema_path = Path(os.getenv("VALIDATION_SCHEMA", "validation/schema.yaml"))
    config = load_schema_config(schema_path)
    db_env_var = config.get("database_env_var", "DB_PATH")
    db_path = Path(os.getenv(db_env_var, "nigeria.db"))

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    errors = validate(db_path=db_path, schema_path=schema_path)
    if errors:
        print("Schema validation failed:")
        for error in errors:
            print(f" - {error}")
        return 1

    print(f"Schema validation passed for {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
