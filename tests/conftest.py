"""Shared fixtures for deterministic unit tests."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def schema_sql(repo_root: Path) -> str:
    """Load the repository schema once for schema-related tests."""
    schema_path = repo_root / "nigeria_db_schema.sql"
    return schema_path.read_text(encoding="utf-8")


@pytest.fixture
def in_memory_db(schema_sql: str) -> sqlite3.Connection:
    """Create a fast in-memory SQLite database for deterministic tests."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    conn.execute(
        "INSERT INTO lga (lga_code, lga_name, state, zone) VALUES (?, ?, ?, ?)",
        ("LGA-001", "Aba North", "Abia", "SE"),
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root for tests that need fixture files."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_ncdc_csv_text() -> str:
    """Small in-memory CSV fixture used to test pandas CSV ingestion."""
    return (
        "alert_date,state,lga_name,disease,alert_level,suspected_cases,"
        "confirmed_cases,deaths,ncdc_ref\n"
        "2025-01-01,Abia,Aba North,cholera,suspected,4,1,0,NCDC-1\n"
    )
