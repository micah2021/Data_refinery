"""Tests for database schema creation behavior in setup_db.py."""

from __future__ import annotations

from pathlib import Path

from setup_db import create_schema


EXPECTED_TABLES = {
    "lga",
    "facility",
    "disease_record",
    "maternal_health",
    "surveillance_alert",
    "climate_health",
    "socioeconomic",
    "data_quality_log",
    "feature_store",
}
EXPECTED_VIEWS = {
    "v_low_quality_lgas",
    "v_maternal_risk_summary",
    "v_weekly_zone_incidence",
}


def test_create_schema_creates_expected_tables(tmp_path: Path, repo_root: Path) -> None:
    """Schema helper should create all core tables defined in nigeria_db_schema.sql."""
    db_path = tmp_path / "test_nigeria.db"
    schema_path = repo_root / "nigeria_db_schema.sql"

    tables, views = create_schema(str(db_path), str(schema_path))

    assert EXPECTED_TABLES.issubset(set(tables))
    assert EXPECTED_VIEWS.issubset(set(views))
