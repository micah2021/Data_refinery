"""Focused tests for data_collector CSV ingestion behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_collector import Config, Database, NCDCSurveillanceCollector


@pytest.mark.usefixtures("in_memory_db")
def test_ncdc_collector_fetch_returns_dataframe_with_expected_columns(
    tmp_path: Path,
    sample_ncdc_csv_text: str,
) -> None:
    """Collector should parse sample CSV into a DataFrame with expected columns."""
    pd = pytest.importorskip("pandas")

    csv_path = tmp_path / "ncdc_alerts.csv"
    csv_path.write_text(sample_ncdc_csv_text, encoding="utf-8")

    db_path = tmp_path / "collector.db"
    config = Config(db_path=str(db_path), schema_path="./nigeria_db_schema.sql")
    database = Database(config)

    collector = NCDCSurveillanceCollector(config=config, db=database)
    collector.CSV_PATH = str(csv_path)

    frame = collector.fetch()

    assert isinstance(frame, pd.DataFrame)
    assert set(
        [
            "alert_date",
            "state",
            "lga_name",
            "disease",
            "alert_level",
            "suspected_cases",
            "confirmed_cases",
            "deaths",
            "ncdc_ref",
        ]
    ).issubset(set(frame.columns))
