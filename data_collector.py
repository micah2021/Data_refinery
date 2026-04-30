"""
data_collector.py — Nigeria Health AI Data Refinery
====================================================
Modular ingestion pipeline for nigeria.db.

Architecture
------------
- BaseCollector      : abstract base all collectors inherit from
- SourceRegistry     : register / discover collectors at runtime
- Pipeline           : orchestrate runs, logging, scheduling
- RefinerRegistry    : pluggable data-quality / imputation steps
- FeatureBuilder     : populate feature_store from raw tables

Adding a new data source
------------------------
1. Create a subclass of BaseCollector in collectors/
2. Decorate with @SourceRegistry.register("my_source")
3. Implement fetch(), parse(), validate(), load()
   — that's it. Pipeline picks it up automatically.

Adding a new refinery step
--------------------------
1. Create a subclass of BaseRefiner in refiners/
2. Decorate with @RefinerRegistry.register("my_step")
3. Implement run(conn) -> RefineResult

Environment variables (.env)
-----------------------------
DB_PATH          path to nigeria.db           (default: ./nigeria.db)
LOG_LEVEL        DEBUG / INFO / WARNING        (default: INFO)
DHIS2_URL        DHIS2 base URL
DHIS2_USER       DHIS2 username
DHIS2_PASS       DHIS2 password
NCDC_API_KEY     NCDC IDSR API key (when available)
WORLD_BANK_KEY   World Bank API key (optional — public API works without)
FAO_KEY          FAO FAOSTAT key
NIMET_KEY        NiMET climate data key
"""

from __future__ import annotations

import abc
import importlib
import inspect
import json
import logging
import os
import pkgutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Generator, Optional

# ---------------------------------------------------------------------------
# Optional dependencies — import lazily so the file loads without them
# ---------------------------------------------------------------------------
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nigeria_collector")


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    """Central config. Override via environment variables or subclass."""
    db_path: str         = os.getenv("DB_PATH", "./nigeria.db")
    schema_path: str     = os.getenv("SCHEMA_PATH", "./nigeria_db_schema.sql")
    log_level: str       = LOG_LEVEL
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_retries: int     = int(os.getenv("MAX_RETRIES", "3"))
    batch_size: int      = int(os.getenv("BATCH_SIZE", "500"))

    # Source credentials
    dhis2_url:    str = os.getenv("DHIS2_URL", "")
    dhis2_user:   str = os.getenv("DHIS2_USER", "")
    dhis2_pass:   str = os.getenv("DHIS2_PASS", "")
    ncdc_api_key: str = os.getenv("NCDC_API_KEY", "")
    world_bank_key: str = os.getenv("WORLD_BANK_KEY", "")
    fao_key:      str = os.getenv("FAO_KEY", "")
    nimet_key:    str = os.getenv("NIMET_KEY", "")

    def validate(self) -> list[str]:
        """Return list of warnings about missing credentials."""
        warnings = []
        if not self.dhis2_url:
            warnings.append("DHIS2_URL not set — DHIS2 collector will be skipped")
        if not self.ncdc_api_key:
            warnings.append("NCDC_API_KEY not set — NCDC collector will use CSV fallback")
        return warnings


# ===========================================================================
# Database helpers
# ===========================================================================

class Database:
    """Thin wrapper around SQLite with WAL mode and schema bootstrap."""

    def __init__(self, config: Config):
        self.path = config.db_path
        self._ensure_schema(config.schema_path)

    def _ensure_schema(self, schema_path: str) -> None:
        schema_file = Path(schema_path)
        if not schema_file.exists():
            log.warning("Schema file not found at %s — skipping bootstrap", schema_path)
            return
        with self.connect() as conn:
            conn.executescript(schema_file.read_text())
        log.info("Schema applied from %s", schema_path)

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert(
        self,
        conn: sqlite3.Connection,
        table: str,
        rows: list[dict],
        conflict_cols: list[str],
    ) -> int:
        """
        INSERT OR REPLACE using conflict_cols as the uniqueness key.
        Returns number of rows written.
        """
        if not rows:
            return 0
        cols = list(rows[0].keys())
        placeholders = ", ".join("?" * len(cols))
        col_names = ", ".join(cols)
        conflict = ", ".join(conflict_cols)
        sql = (
            f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols if c not in conflict_cols)
        )
        values = [tuple(r[c] for c in cols) for r in rows]
        conn.executemany(sql, values)
        return len(rows)


# ===========================================================================
# HTTP session factory
# ===========================================================================

def make_session(
    retries: int = 3,
    backoff: float = 1.0,
    timeout: int = 30,
) -> "requests.Session":
    """Shared requests.Session with retry logic and timeouts."""
    if not HAS_REQUESTS:
        raise ImportError("requests is required: pip install requests")
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request = lambda method, url, **kw: \
        requests.Session.request(session, method, url, timeout=timeout, **kw)
    return session


# ===========================================================================
# Result types
# ===========================================================================

@dataclass
class CollectResult:
    source: str
    table: str
    rows_fetched: int = 0
    rows_written: int = 0
    rows_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: str = ""
    success: bool = True

    def finish(self) -> "CollectResult":
        self.finished_at = datetime.utcnow().isoformat()
        if self.errors:
            self.success = False
        return self

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"[{status}] {self.source} → {self.table} | "
            f"fetched={self.rows_fetched} written={self.rows_written} "
            f"skipped={self.rows_skipped} errors={len(self.errors)}"
        )


@dataclass
class RefineResult:
    step: str
    records_processed: int = 0
    records_updated: int = 0
    flags_raised: int = 0
    errors: list[str] = field(default_factory=list)
    success: bool = True

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"[{status}] refiner:{self.step} | "
            f"processed={self.records_processed} updated={self.records_updated} "
            f"flags={self.flags_raised}"
        )


# ===========================================================================
# Base Collector
# ===========================================================================

class BaseCollector(abc.ABC):
    """
    Abstract base for all data source collectors.

    Subclass this and implement fetch(), parse(), validate(), load().
    Use @SourceRegistry.register("name") on your subclass.

    The Pipeline calls run() which chains the four methods and
    logs a DataQualityLog entry automatically.
    """

    #: Human-readable name shown in logs and reports
    source_name: ClassVar[str] = "unnamed_source"

    #: Target table in nigeria.db
    target_table: ClassVar[str] = ""

    #: Columns that define uniqueness for upsert
    conflict_cols: ClassVar[list[str]] = []

    #: Set False to exclude from default pipeline runs
    enabled: ClassVar[bool] = True

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.log = logging.getLogger(f"collector.{self.source_name}")
        self.session = make_session(
            retries=config.max_retries,
            timeout=config.request_timeout,
        ) if HAS_REQUESTS else None

    # ------------------------------------------------------------------ #
    # Abstract interface — implement all four in subclasses
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def fetch(self) -> Any:
        """
        Retrieve raw data from the external source.
        Return anything — list, DataFrame, dict, file path.
        Raise on unrecoverable errors.
        """

    @abc.abstractmethod
    def parse(self, raw: Any) -> list[dict]:
        """
        Transform raw data into a list of dicts matching target_table columns.
        Do NOT touch the database here.
        """

    @abc.abstractmethod
    def validate(self, rows: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Filter / fix rows. Return (valid_rows, error_messages).
        Drop rows that can't be fixed; log them as errors.
        """

    @abc.abstractmethod
    def load(self, conn: sqlite3.Connection, rows: list[dict]) -> int:
        """
        Write valid rows to the database. Return number of rows written.
        Default implementation calls db.upsert(); override for custom logic.
        """

    # ------------------------------------------------------------------ #
    # Hooks — override in subclasses for custom behaviour
    # ------------------------------------------------------------------ #

    def pre_run(self) -> None:
        """Called before fetch(). Use for auth, setup, or state checks."""

    def post_run(self, result: CollectResult) -> None:
        """Called after load(). Use for notifications or cleanup."""

    # ------------------------------------------------------------------ #
    # Orchestration — do not override
    # ------------------------------------------------------------------ #

    def run(self) -> CollectResult:
        result = CollectResult(source=self.source_name, table=self.target_table)
        self.log.info("Starting collection: %s → %s", self.source_name, self.target_table)
        try:
            self.pre_run()
            raw = self.fetch()
            rows = self.parse(raw)
            result.rows_fetched = len(rows)
            valid_rows, errors = self.validate(rows)
            result.rows_skipped = len(rows) - len(valid_rows)
            result.errors.extend(errors)

            # Data write in its own transaction
            with self.db.connect() as conn:
                result.rows_written = self.load(conn, valid_rows)

            # Quality log in a separate transaction so it never
            # rolls back the data we just wrote
            try:
                with self.db.connect() as conn:
                    self._log_quality(conn, result)
            except Exception as log_exc:
                self.log.warning("Quality log failed (data still saved): %s", log_exc)

            self.post_run(result)
        except Exception as exc:
            result.errors.append(str(exc))
            result.success = False
            self.log.exception("Collection failed: %s", exc)
        finally:
            result.finish()
            self.log.info(str(result))
        return result

    def _log_quality(self, conn: sqlite3.Connection, result: CollectResult) -> None:
        """Write a data_quality_log entry for this run."""
        completeness = (
            100.0 * result.rows_written / result.rows_fetched
            if result.rows_fetched > 0 else 0.0
        )
        try:
            conn.execute(
                """
                INSERT INTO data_quality_log
                    (table_name, records_checked,
                     completeness_pct, bias_flags, notes, action_taken)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.target_table,
                    result.rows_fetched,
                    round(completeness, 2),
                    json.dumps(result.errors[:10]),
                    f"source={self.source_name} skipped={result.rows_skipped}",
                    "written" if result.rows_written > 0 else "skipped",
                ),
            )
        except Exception as e:
            self.log.warning("Could not write quality log: %s", e)


# ===========================================================================
# Base Refiner
# ===========================================================================

class BaseRefiner(abc.ABC):
    """
    Abstract base for data-quality / imputation / harmonisation steps.
    Refiners run after all collectors finish, before FeatureBuilder.
    """

    step_name: ClassVar[str] = "unnamed_refiner"
    enabled: ClassVar[bool] = True

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.log = logging.getLogger(f"refiner.{self.step_name}")

    @abc.abstractmethod
    def run(self, conn: sqlite3.Connection) -> RefineResult:
        """Execute the refinery step. conn is an open, uncommitted connection."""


# ===========================================================================
# Registries — self-populating via decorator
# ===========================================================================

class SourceRegistry:
    """
    Registry of all BaseCollector subclasses.

    Usage:
        @SourceRegistry.register("dhis2_disease")
        class DHIS2DiseaseCollector(BaseCollector): ...
    """
    _registry: ClassVar[dict[str, type[BaseCollector]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(klass: type[BaseCollector]) -> type[BaseCollector]:
            klass.source_name = name
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def all_enabled(cls) -> list[type[BaseCollector]]:
        return [k for k in cls._registry.values() if k.enabled]

    @classmethod
    def get(cls, name: str) -> Optional[type[BaseCollector]]:
        return cls._registry.get(name)

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._registry.keys())


class RefinerRegistry:
    """Registry of all BaseRefiner subclasses."""
    _registry: ClassVar[dict[str, type[BaseRefiner]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(klass: type[BaseRefiner]) -> type[BaseRefiner]:
            klass.step_name = name
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def all_enabled(cls) -> list[type[BaseRefiner]]:
        return [k for k in cls._registry.values() if k.enabled]


# ===========================================================================
# Built-in collectors
# ===========================================================================

@SourceRegistry.register("dhis2_disease")
class DHIS2DiseaseCollector(BaseCollector):
    """
    Pulls weekly disease counts from a DHIS2 instance via the analytics API.
    Requires DHIS2_URL, DHIS2_USER, DHIS2_PASS in environment.

    Extend by adding more dataElement UIDs to ELEMENT_MAP.
    """
    target_table = "disease_record"
    conflict_cols = ["lga_id", "report_date", "disease_category", "source", "facility_id"]

    ELEMENT_MAP: ClassVar[dict[str, str]] = {
        # dataElement UID → disease_category in nigeria.db
        # Replace UIDs with real ones from your DHIS2 instance
        "fbfJHSPpUQD": "malaria",
        "cYeuwXTCPkU": "cholera",
        "hfdmMSPBgLG": "typhoid",
        "rbkr8PL0rwM": "tuberculosis",
    }

    def pre_run(self) -> None:
        if not self.config.dhis2_url:
            raise RuntimeError("DHIS2_URL not configured — skipping DHIS2 collector")

    def fetch(self) -> dict:
        url = f"{self.config.dhis2_url}/api/analytics"
        element_list = ";".join(self.ELEMENT_MAP.keys())
        params = {
            "dimension": [
                f"dx:{element_list}",
                "ou:LEVEL-4",          # LGA level
                "pe:LAST_52_WEEKS",
            ],
            "displayProperty": "NAME",
            "outputIdScheme": "CODE",
        }
        resp = self.session.get(
            url, params=params,
            auth=(self.config.dhis2_user, self.config.dhis2_pass),
        )
        resp.raise_for_status()
        return resp.json()

    def parse(self, raw: dict) -> list[dict]:
        rows = []
        headers = [h["name"] for h in raw.get("headers", [])]
        for row in raw.get("rows", []):
            r = dict(zip(headers, row))
            element_uid = r.get("dx", "")
            rows.append({
                "lga_id": self._resolve_lga(r.get("ou", "")),
                "facility_id": None,
                "report_date": self._parse_period(r.get("pe", "")),
                "epi_week": self._epi_week(r.get("pe", "")),
                "epi_year": self._epi_year(r.get("pe", "")),
                "icd10_code": None,    # harmonised by ICD10Refiner later
                "disease_name": r.get("dx", ""),
                "disease_category": self.ELEMENT_MAP.get(element_uid, "other_infectious"),
                "case_count": int(float(r.get("value", 0))),
                "death_count": 0,
                "age_group": "unknown",
                "sex": "unknown",
                "is_confirmed": 0,
                "data_quality_score": None,  # set by QualityScoreRefiner
                "source": "dhis2",
                "raw_record_ref": element_uid,
            })
        return rows

    def validate(self, rows: list[dict]) -> tuple[list[dict], list[str]]:
        valid, errors = [], []
        for r in rows:
            if r["lga_id"] is None:
                errors.append(f"Unknown LGA code: {r.get('raw_record_ref')}")
                continue
            if r["case_count"] < 0:
                errors.append(f"Negative case count row skipped: {r}")
                continue
            valid.append(r)
        return valid, errors

    def load(self, conn: sqlite3.Connection, rows: list[dict]) -> int:
        return self.db.upsert(conn, self.target_table, rows, self.conflict_cols)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_lga(self, ou_code: str) -> Optional[int]:
        """Look up lga_id from the LGA code in nigeria.db."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT lga_id FROM lga WHERE lga_code = ?", (ou_code,)
            ).fetchone()
        return row["lga_id"] if row else None

    @staticmethod
    def _parse_period(pe: str) -> str:
        """Convert DHIS2 period (e.g. 202401W) to ISO date."""
        try:
            year = int(pe[:4])
            week = int(pe[5:]) if "W" in pe else 1
            from datetime import date
            import datetime as dt
            return dt.date.fromisocalendar(year, week, 1).isoformat()
        except Exception:
            return date.today().isoformat()

    @staticmethod
    def _epi_week(pe: str) -> Optional[int]:
        try:
            return int(pe.split("W")[1]) if "W" in pe else None
        except Exception:
            return None

    @staticmethod
    def _epi_year(pe: str) -> Optional[int]:
        try:
            return int(pe[:4])
        except Exception:
            return None


@SourceRegistry.register("world_bank_socioeconomic")
class WorldBankCollector(BaseCollector):
    """
    Pulls World Bank indicators for Nigeria via the public API.
    No API key required for basic access.

    Add indicators to INDICATOR_MAP to ingest more variables.
    """
    target_table = "socioeconomic"
    conflict_cols = ["lga_id", "year"]

    INDICATOR_MAP: ClassVar[dict[str, str]] = {
        "SI.POV.NAHC":  "poverty_headcount_pct",
        "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
        "SI.POV.GINI":  "gini_coefficient",
        "SE.ADT.LITR.ZS": "literacy_rate_pct",
        "SH.H2O.SAFE.ZS": "piped_water_pct",
        "SH.STA.ACSN":  "sanitation_pct",
        "EG.ELC.ACCS.ZS": "electricity_access_pct",
    }

    BASE_URL = "https://api.worldbank.org/v2"

    def fetch(self) -> list[dict]:
        all_data = []
        for indicator, col in self.INDICATOR_MAP.items():
            url = f"{self.BASE_URL}/country/NG/indicator/{indicator}"
            params = {"format": "json", "per_page": 100, "mrv": 10}
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if len(data) >= 2:
                for entry in data[1] or []:
                    if entry.get("value") is not None:
                        all_data.append({
                            "indicator": indicator,
                            "column": col,
                            "year": int(entry["date"]),
                            "value": float(entry["value"]),
                        })
            time.sleep(0.3)  # be polite to the API
        return all_data

    def parse(self, raw: list[dict]) -> list[dict]:
        """
        World Bank data is national-level. We distribute to all LGAs as a
        state-level proxy — future versions should use state-disaggregated data.
        """
        by_year: dict[int, dict] = {}
        for entry in raw:
            yr = entry["year"]
            if yr not in by_year:
                by_year[yr] = {"year": yr}
            by_year[yr][entry["column"]] = entry["value"]

        rows = []
        with self.db.connect() as conn:
            lgas = conn.execute("SELECT lga_id FROM lga").fetchall()
        for lga_row in lgas:
            for yr, indicators in by_year.items():
                rows.append({"lga_id": lga_row["lga_id"], **indicators})
        return rows

    def validate(self, rows: list[dict]) -> tuple[list[dict], list[str]]:
        valid, errors = [], []
        for r in rows:
            if not r.get("lga_id") or not r.get("year"):
                errors.append(f"Missing lga_id or year: {r}")
                continue
            if r.get("poverty_headcount_pct", 0) and not (0 <= r["poverty_headcount_pct"] <= 100):
                errors.append(f"Out-of-range poverty value: {r['poverty_headcount_pct']}")
                continue
            valid.append(r)
        return valid, errors

    def load(self, conn: sqlite3.Connection, rows: list[dict]) -> int:
        return self.db.upsert(conn, self.target_table, rows, self.conflict_cols)


@SourceRegistry.register("ncdc_surveillance")
class NCDCSurveillanceCollector(BaseCollector):
    """
    Ingests NCDC IDSR outbreak alerts.

    Currently implemented as CSV file ingestion (NCDC publishes weekly
    situation reports as PDFs/CSVs). When NCDC exposes a proper API,
    swap fetch() to use self.session with self.config.ncdc_api_key.

    CSV format expected: alert_date,state,lga_name,disease,alert_level,
                         suspected_cases,confirmed_cases,deaths,ncdc_ref
    """
    target_table = "surveillance_alert"
    conflict_cols = ["lga_id", "alert_date", "disease", "alert_level"]

    CSV_PATH: ClassVar[str] = os.getenv("NCDC_CSV_PATH", "./data/ncdc_alerts.csv")

    def fetch(self) -> Any:
        if not HAS_PANDAS:
            raise ImportError("pandas is required for CSV ingestion: pip install pandas")
        path = Path(self.CSV_PATH)
        if not path.exists():
            self.log.warning("NCDC CSV not found at %s — returning empty", self.CSV_PATH)
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=["alert_date"])

    def parse(self, raw: Any) -> list[dict]:
        if not HAS_PANDAS or raw.empty:
            return []
        rows = []
        with self.db.connect() as conn:
            lga_map = {
                row["lga_name"].lower(): row["lga_id"]
                for row in conn.execute("SELECT lga_id, lga_name FROM lga").fetchall()
            }
        for _, r in raw.iterrows():
            lga_id = lga_map.get(str(r.get("lga_name", "")).lower())
            if lga_id is None:
                continue  # skip rows with no matching LGA
            rows.append({
                "lga_id": lga_id,
                "alert_date": str(r.get("alert_date", ""))[:10],
                "disease": r.get("disease", ""),
                "alert_level": r.get("alert_level", "suspected"),
                "suspected_cases": int(r.get("suspected_cases", 0) or 0),
                "confirmed_cases": int(r.get("confirmed_cases", 0) or 0),
                "deaths": int(r.get("deaths", 0) or 0),
                "ncdc_ref": r.get("ncdc_ref", ""),
            })
        return rows

    def validate(self, rows: list[dict]) -> tuple[list[dict], list[str]]:
        valid, errors = [], []
        valid_levels = {"rumour", "suspected", "confirmed", "outbreak_declared"}
        for r in rows:
            if r.get("lga_id") is None:
                errors.append(f"LGA not found for: {r.get('alert_date')} {r.get('disease')}")
                continue
            if r.get("alert_level") not in valid_levels:
                r["alert_level"] = "suspected"
            valid.append(r)
        return valid, errors

    def load(self, conn: sqlite3.Connection, rows: list[dict]) -> int:
        return self.db.upsert(conn, self.target_table, rows, self.conflict_cols)


@SourceRegistry.register("fao_climate")
class FAOClimateCollector(BaseCollector):
    """
    Ingests FAO FAOSTAT climate/food data and NiMET climate records.

    Currently fetches from FAOSTAT's public API for Nigeria.
    NiMET data (rainfall, temperature) should be ingested via CSV
    until NiMET exposes an API — set NIMET_CSV_PATH in .env.

    Extend FAOSTAT_ITEMS to pull more food/climate indicators.
    """
    target_table = "climate_health"
    conflict_cols = ["lga_id", "year", "month"]

    FAOSTAT_BASE = "https://fenixservices.fao.org/faostat/api/v1/en/data"
    FAOSTAT_ITEMS: ClassVar[dict[str, str]] = {
        "432": "food_insecurity_pct",   # Prevalence of undernourishment
    }
    NIMET_CSV_PATH: ClassVar[str] = os.getenv("NIMET_CSV_PATH", "./data/nimet_climate.csv")

    def fetch(self) -> list[dict]:
        records = []

        # NiMET CSV ingestion
        nimet_path = Path(self.NIMET_CSV_PATH)
        if nimet_path.exists() and HAS_PANDAS:
            df = pd.read_csv(nimet_path)
            records.extend(df.to_dict(orient="records"))
        else:
            self.log.warning("NiMET CSV not found — climate_health will be incomplete")

        return records

    def parse(self, raw: list[dict]) -> list[dict]:
        rows = []
        with self.db.connect() as conn:
            lga_map = {
                row["lga_name"].lower(): row["lga_id"]
                for row in conn.execute("SELECT lga_id, lga_name FROM lga").fetchall()
            }
        for r in raw:
            lga_id = lga_map.get(str(r.get("lga_name", "")).lower())
            month = int(r.get("month", 0))
            rows.append({
                "lga_id": lga_id,
                "year": int(r.get("year", 0)),
                "month": month,
                "season": self._season(month),
                "rainfall_mm": r.get("rainfall_mm"),
                "temp_max_c": r.get("temp_max_c"),
                "temp_min_c": r.get("temp_min_c"),
                "humidity_pct": r.get("humidity_pct"),
                "ndvi": r.get("ndvi"),
                "flood_risk_flag": int(bool(r.get("flood_risk_flag", 0))),
                "drought_flag": int(bool(r.get("drought_flag", 0))),
                "source": "nimet",
            })
        return rows

    def validate(self, rows: list[dict]) -> tuple[list[dict], list[str]]:
        valid, errors = [], []
        for r in rows:
            if r.get("lga_id") is None:
                errors.append(f"LGA not found in climate row: {r}")
                continue
            if not (1 <= r.get("month", 0) <= 12):
                errors.append(f"Invalid month {r.get('month')} skipped")
                continue
            valid.append(r)
        return valid, errors

    def load(self, conn: sqlite3.Connection, rows: list[dict]) -> int:
        return self.db.upsert(conn, self.target_table, rows, self.conflict_cols)

    @staticmethod
    def _season(month: int) -> str:
        """Nigeria seasons: harmattan Nov-Feb, dry Mar-Apr, wet May-Oct."""
        if month in (11, 12, 1, 2):
            return "harmattan"
        if month in (3, 4):
            return "dry"
        return "wet"


# ===========================================================================
# Built-in refiners
# ===========================================================================

@RefinerRegistry.register("icd10_harmoniser")
class ICD10HarmoniserRefiner(BaseRefiner):
    """
    Maps raw disease_name strings to standardised ICD-10 codes.
    Extend MAPPING with additional disease names as your data grows.
    """
    MAPPING: ClassVar[dict[str, str]] = {
        "malaria":              "B54",
        "cholera":              "A00",
        "typhoid":              "A01.0",
        "typhoid fever":        "A01.0",
        "tuberculosis":         "A15",
        "tb":                   "A15",
        "hiv":                  "B20",
        "meningitis":           "G03",
        "yellow fever":         "A95",
        "lassa fever":          "A96.2",
        "hypertension":         "I10",
        "diabetes":             "E11",
        "acute watery diarrhoea": "A09",
        "pneumonia":            "J18",
    }

    def run(self, conn: sqlite3.Connection) -> RefineResult:
        result = RefineResult(step=self.step_name)
        rows = conn.execute(
            "SELECT record_id, disease_name FROM disease_record WHERE icd10_code IS NULL"
        ).fetchall()
        result.records_processed = len(rows)
        updated = 0
        for row in rows:
            code = self.MAPPING.get(row["disease_name"].lower().strip())
            if code:
                conn.execute(
                    "UPDATE disease_record SET icd10_code = ? WHERE record_id = ?",
                    (code, row["record_id"]),
                )
                updated += 1
        result.records_updated = updated
        self.log.info("ICD-10 harmonised %d/%d records", updated, len(rows))
        return result


@RefinerRegistry.register("quality_scorer")
class QualityScorerRefiner(BaseRefiner):
    """
    Assigns data_quality_score (0–1) to each disease_record based on:
    - LGA reporting completeness from data_quality_log
    - Whether the case is confirmed vs suspected
    - Facility type (teaching hospital > PHC > unknown)

    Extend the scoring logic here as you learn more about your data.
    """

    FACILITY_WEIGHTS: ClassVar[dict[str, float]] = {
        "federal_teaching_hospital": 1.0,
        "state_specialist_hospital": 0.9,
        "general_hospital":          0.8,
        "primary_health_centre":     0.65,
        "private_hospital":          0.7,
        "private_clinic":            0.6,
        "maternity_home":            0.55,
    }

    def run(self, conn: sqlite3.Connection) -> RefineResult:
        result = RefineResult(step=self.step_name)

        # Build completeness lookup from quality log
        completeness = {
            row["lga_id"]: row["completeness_pct"] / 100.0
            for row in conn.execute(
                """
                SELECT lga_id, AVG(completeness_pct) AS completeness_pct
                FROM data_quality_log
                WHERE table_name = 'disease_record'
                GROUP BY lga_id
                """
            ).fetchall()
        }

        records = conn.execute(
            """
            SELECT dr.record_id, dr.lga_id, dr.is_confirmed, dr.facility_id,
                   f.facility_type
            FROM disease_record dr
            LEFT JOIN facility f ON f.facility_id = dr.facility_id
            WHERE dr.data_quality_score IS NULL
            """
        ).fetchall()

        result.records_processed = len(records)
        updated = 0
        for row in records:
            base = completeness.get(row["lga_id"], 0.5)
            confirmed_bonus = 0.15 if row["is_confirmed"] else 0.0
            facility_w = self.FACILITY_WEIGHTS.get(row["facility_type"] or "", 0.5)
            score = round(min(1.0, (base * 0.5) + (facility_w * 0.35) + confirmed_bonus), 3)
            conn.execute(
                "UPDATE disease_record SET data_quality_score = ? WHERE record_id = ?",
                (score, row["record_id"]),
            )
            updated += 1
        result.records_updated = updated
        return result


@RefinerRegistry.register("rural_gap_imputer")
class RuralGapImputerRefiner(BaseRefiner):
    """
    Fills missing disease_record counts for rural LGAs using zone-level medians.

    This directly addresses the urban-reporting-bias validity threat.
    The imputed rows are flagged with data_quality_score = 0.3 so models
    can down-weight them appropriately.

    Strategy can be extended to use spatial interpolation (e.g. kriging)
    by swapping the median imputation logic below.
    """

    def run(self, conn: sqlite3.Connection) -> RefineResult:
        result = RefineResult(step=self.step_name)

        # Find rural LGAs with no records in the last 52 weeks
        rural_gaps = conn.execute(
            """
            SELECT l.lga_id, l.zone
            FROM lga l
            WHERE l.lga_type = 'rural'
              AND l.lga_id NOT IN (
                  SELECT DISTINCT lga_id FROM disease_record
                  WHERE report_date >= date('now', '-52 weeks')
              )
            """
        ).fetchall()

        result.records_processed = len(rural_gaps)
        if not rural_gaps:
            return result

        for gap in rural_gaps:
            # Use zone median as imputed value
            zone_median = conn.execute(
                """
                SELECT disease_category, CAST(AVG(case_count) AS INTEGER) AS median_count
                FROM disease_record dr
                JOIN lga l ON l.lga_id = dr.lga_id
                WHERE l.zone = ?
                  AND dr.report_date >= date('now', '-52 weeks')
                GROUP BY disease_category
                """,
                (gap["zone"],),
            ).fetchall()

            for row in zone_median:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO disease_record
                        (lga_id, report_date, disease_name, disease_category,
                         case_count, death_count, data_quality_score, source)
                    VALUES (?, date('now'), ?, ?, ?, 0, 0.3, 'imputed')
                    """,
                    (
                        gap["lga_id"],
                        row["disease_category"],
                        row["disease_category"],
                        row["median_count"],
                    ),
                )
            result.records_updated += 1
            result.flags_raised += 1

        self.log.info(
            "Imputed %d rural LGA gaps using zone medians", result.records_updated
        )
        return result


# ===========================================================================
# Feature Builder
# ===========================================================================

class FeatureBuilder:
    """
    Populates feature_store from the raw tables.
    Runs after all collectors and refiners complete.

    The feature_store is what AI models train on — never the raw tables.
    Extend build_features() to add new engineered features as your
    research questions evolve.
    """

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.log = logging.getLogger("feature_builder")

    def build(self) -> int:
        self.log.info("Building feature store...")
        with self.db.connect() as conn:
            rows_written = self._build_features(conn)
        self.log.info("Feature store: %d rows written", rows_written)
        return rows_written

    def _build_features(self, conn: sqlite3.Connection) -> int:
        """
        Core feature engineering query.
        Joins disease_record + climate_health + socioeconomic + lga
        into weekly LGA-disease snapshots.
        """
        conn.execute("DELETE FROM feature_store")  # full rebuild each run

        conn.execute(
            """
            INSERT INTO feature_store (
                lga_id, epi_year, epi_week, disease_category,
                incidence_rate,
                rainfall_mm, temp_max_c, humidity_pct, ndvi, season,
                flood_risk_flag,
                poverty_headcount_pct, food_insecurity_pct,
                nhia_coverage_pct, piped_water_pct, sanitation_pct,
                zone, lga_type, pop_density,
                reporting_weight, completeness_pct,
                active_alert_flag
            )
            SELECT
                dr.lga_id,
                dr.epi_year,
                dr.epi_week,
                dr.disease_category,

                -- incidence rate per 10,000 population
                CASE WHEN l.pop_density > 0
                     THEN CAST(SUM(dr.case_count) AS REAL) /
                          (l.pop_density * l.area_km2 / 10000.0)
                     ELSE NULL END AS incidence_rate,

                AVG(ch.rainfall_mm)     AS rainfall_mm,
                AVG(ch.temp_max_c)      AS temp_max_c,
                AVG(ch.humidity_pct)    AS humidity_pct,
                AVG(ch.ndvi)            AS ndvi,
                ch.season,
                MAX(ch.flood_risk_flag) AS flood_risk_flag,

                se.poverty_headcount_pct,
                se.food_insecurity_pct,
                se.nhia_coverage_pct,
                se.piped_water_pct,
                se.sanitation_pct,

                l.zone,
                l.lga_type,
                l.pop_density,

                AVG(dr.data_quality_score)  AS reporting_weight,
                dql.completeness_pct,

                CASE WHEN sa.alert_id IS NOT NULL THEN 1 ELSE 0 END AS active_alert_flag

            FROM disease_record dr
            JOIN lga l ON l.lga_id = dr.lga_id
            LEFT JOIN climate_health ch
                ON ch.lga_id = dr.lga_id
                AND ch.year   = dr.epi_year
                AND ch.month  = CAST(strftime('%m', dr.report_date) AS INTEGER)
            LEFT JOIN socioeconomic se
                ON se.lga_id = dr.lga_id
                AND se.year  = dr.epi_year
            LEFT JOIN data_quality_log dql
                ON dql.lga_id     = dr.lga_id
                AND dql.table_name = 'disease_record'
            LEFT JOIN surveillance_alert sa
                ON sa.lga_id      = dr.lga_id
                AND sa.disease     = dr.disease_category
                AND sa.alert_date  = dr.report_date
            WHERE dr.epi_year IS NOT NULL
              AND dr.epi_week IS NOT NULL
            GROUP BY
                dr.lga_id, dr.epi_year, dr.epi_week,
                dr.disease_category, ch.season,
                se.poverty_headcount_pct, se.food_insecurity_pct,
                se.nhia_coverage_pct, se.piped_water_pct, se.sanitation_pct,
                l.zone, l.lga_type, l.pop_density,
                dql.completeness_pct, sa.alert_id
            """
        )

        # Add lag features (autoregressive — key for outbreak prediction)
        self._add_lag_features(conn)

        count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
        return count

    def _add_lag_features(self, conn: sqlite3.Connection) -> None:
        """Add 1w/2w/4w/8w incidence lags via window functions."""
        conn.execute(
            """
            UPDATE feature_store AS fs
            SET
                incidence_lag_1w = (
                    SELECT f2.incidence_rate
                    FROM feature_store f2
                    WHERE f2.lga_id = fs.lga_id
                      AND f2.disease_category = fs.disease_category
                      AND f2.epi_year * 100 + f2.epi_week =
                          CASE WHEN fs.epi_week > 1
                               THEN fs.epi_year * 100 + fs.epi_week - 1
                               ELSE (fs.epi_year - 1) * 100 + 52 END
                ),
                incidence_lag_4w = (
                    SELECT f2.incidence_rate
                    FROM feature_store f2
                    WHERE f2.lga_id = fs.lga_id
                      AND f2.disease_category = fs.disease_category
                      AND f2.epi_year * 100 + f2.epi_week =
                          CASE WHEN fs.epi_week > 4
                               THEN fs.epi_year * 100 + fs.epi_week - 4
                               ELSE (fs.epi_year - 1) * 100 + (52 - (4 - fs.epi_week)) END
                )
            """
        )


# ===========================================================================
# Plugin auto-loader
# ===========================================================================

def load_plugins(plugin_dirs: list[str]) -> None:
    """
    Auto-discover and import collector/refiner subclasses from plugin directories.

    Any .py file in these directories that subclasses BaseCollector or BaseRefiner
    and uses the registry decorators will be registered automatically.

    Usage:
        load_plugins(["./collectors", "./refiners", "./plugins"])
    """
    for plugin_dir in plugin_dirs:
        path = Path(plugin_dir)
        if not path.exists():
            continue
        sys.path.insert(0, str(path.parent))
        for finder, module_name, _ in pkgutil.iter_modules([str(path)]):
            try:
                importlib.import_module(f"{path.name}.{module_name}")
                log.debug("Loaded plugin: %s.%s", path.name, module_name)
            except Exception as exc:
                log.warning("Failed to load plugin %s: %s", module_name, exc)


# ===========================================================================
# Pipeline
# ===========================================================================

class Pipeline:
    """
    Orchestrates the full collect → refine → feature-build cycle.

    Usage:
        pipeline = Pipeline(config)
        pipeline.run()                           # all sources
        pipeline.run(sources=["dhis2_disease"])  # specific source
        pipeline.run(skip_refiners=True)         # collect only
        pipeline.run(skip_features=True)         # no feature rebuild
    """

    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config)
        self.log = logging.getLogger("pipeline")

        # Auto-load plugins from standard directories
        load_plugins(["./collectors", "./refiners", "./plugins"])

    def run(
        self,
        sources: Optional[list[str]] = None,
        skip_refiners: bool = False,
        skip_features: bool = False,
    ) -> dict[str, Any]:
        started = datetime.utcnow().isoformat()
        self.log.info("=== Pipeline started at %s ===", started)

        # Warn about missing credentials
        for warning in self.config.validate():
            self.log.warning(warning)

        collect_results = self._run_collectors(sources)
        refine_results = [] if skip_refiners else self._run_refiners()
        feature_rows = 0 if skip_features else FeatureBuilder(self.config, self.db).build()

        summary = {
            "started_at": started,
            "finished_at": datetime.utcnow().isoformat(),
            "collectors": [str(r) for r in collect_results],
            "refiners": [str(r) for r in refine_results],
            "feature_rows": feature_rows,
            "success": all(r.success for r in collect_results),
        }
        self.log.info("=== Pipeline complete | feature_rows=%d ===", feature_rows)
        return summary

    def _run_collectors(self, sources: Optional[list[str]]) -> list[CollectResult]:
        results = []
        collector_classes = SourceRegistry.all_enabled()
        if sources:
            collector_classes = [
                c for c in collector_classes if c.source_name in sources
            ]
        for klass in collector_classes:
            collector = klass(self.config, self.db)
            results.append(collector.run())
        return results

    def _run_refiners(self) -> list[RefineResult]:
        results = []
        with self.db.connect() as conn:
            for klass in RefinerRegistry.all_enabled():
                refiner = klass(self.config, self.db)
                try:
                    result = refiner.run(conn)
                    results.append(result)
                    self.log.info(str(result))
                except Exception as exc:
                    self.log.exception("Refiner %s failed: %s", klass.step_name, exc)
                    results.append(RefineResult(step=klass.step_name, success=False,
                                                errors=[str(exc)]))
        return results


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Nigeria Health AI — data collection pipeline"
    )
    parser.add_argument(
        "--sources", nargs="*", metavar="SOURCE",
        help=f"Run only these sources. Available: {SourceRegistry.names()}",
    )
    parser.add_argument(
        "--skip-refiners", action="store_true",
        help="Skip data-quality refiners",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature store rebuild",
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="List all registered collectors and exit",
    )
    parser.add_argument(
        "--db", default=None,
        help="Override DB_PATH from environment",
    )
    args = parser.parse_args()

    if args.list_sources:
        print("Registered collectors:")
        for name in SourceRegistry.names():
            klass = SourceRegistry.get(name)
            print(f"  {name:30s} → {klass.target_table}  (enabled={klass.enabled})")
        print("\nRegistered refiners:")
        for klass in RefinerRegistry.all_enabled():
            print(f"  {klass.step_name}")
        return

    config = Config()
    if args.db:
        config.db_path = args.db

    pipeline = Pipeline(config)
    summary = pipeline.run(
        sources=args.sources or None,
        skip_refiners=args.skip_refiners,
        skip_features=args.skip_features,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
